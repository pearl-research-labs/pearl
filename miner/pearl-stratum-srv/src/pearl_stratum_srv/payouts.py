"""PPLNS payout calculator.

When the pool finds a block, this module computes who gets what.

PPLNS (Pay-Per-Last-N-Shares):
  - We look back at the last N "difficulty units" of shares submitted across
    ALL workers (sum of share.difficulty values).
  - Each worker's payout share = their contribution / N.
  - Pool operator takes `fee_percent` off the top.

This pays smoothly across short-window luck (PROP can pay 0 if you're unlucky
during a round); recent contributors get a bigger slice (active miners earn
better than long-absent ones). Industry standard for pools serving heterogeneous
hashrates.

We do NOT execute on-chain payouts here. We return a list of (recipient, sats)
tuples; the caller persists them into the `payouts` table for the operator to
review and send via `payouts_send.py` (separate manual CLI for safety).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PayoutPolicy:
    fee_percent: float = 1.0          # Operator cut. 1% is competitive; 0% is "we're being nice."
    pplns_n: int = 100_000_000        # Total difficulty units in the lookback window. Tunable; see below.
    operator_address: str = ""        # Where the fee goes. Required (validated by caller).
    min_payout_sats: int = 100_000    # Skip recipients owed less than this; carries forward in future revs.


@dataclass
class PayoutEntry:
    """One row of the payout calculation."""
    recipient: str
    amount_sats: int
    share_difficulty: int    # how much this recipient contributed in the window


@dataclass
class PayoutResult:
    """Full breakdown of how a block reward is divided."""
    block_reward_sats: int
    fee_sats: int
    miners_pool_sats: int            # block_reward - fee, distributed across miners
    entries: list[PayoutEntry]
    pplns_window_difficulty: int     # actual difficulty summed in the window (may be < pplns_n if young pool)
    rejected_for_min_payout_sats: int  # sum of dust that didn't make the min_payout cut


def compute_pplns_payouts(
    block_reward_sats: int,
    shares_by_addr: list[tuple[str, int]],
    policy: PayoutPolicy,
) -> PayoutResult:
    """Compute payouts for one found block.

    `shares_by_addr` is `[(worker_addr, total_difficulty_in_window)]` —
    e.g. the result of `ShareDb.shares_in_window()`. Should already be
    filtered to the PPLNS window the caller wants (typically: last `pplns_n`
    difficulty-units sliding window).

    Returns full breakdown including the operator fee + per-recipient amounts.
    """
    if block_reward_sats < 0:
        raise ValueError("block_reward_sats must be non-negative")
    if not (0 <= policy.fee_percent <= 100):
        raise ValueError("fee_percent must be 0..100")
    if not policy.operator_address:
        raise ValueError("operator_address is required (fee recipient)")

    fee_sats = int(block_reward_sats * policy.fee_percent / 100.0)
    miners_pool_sats = block_reward_sats - fee_sats

    total_diff = sum(d for _, d in shares_by_addr)
    if total_diff <= 0 or miners_pool_sats <= 0:
        # No miners to pay. Operator gets the whole thing.
        return PayoutResult(
            block_reward_sats=block_reward_sats,
            fee_sats=block_reward_sats,
            miners_pool_sats=0,
            entries=[PayoutEntry(policy.operator_address, block_reward_sats, 0)],
            pplns_window_difficulty=total_diff,
            rejected_for_min_payout_sats=0,
        )

    # Proportional split. Use integer math so totals don't lose sats to rounding.
    entries: list[PayoutEntry] = []
    awarded_running = 0
    sorted_shares = sorted(shares_by_addr, key=lambda x: (-x[1], x[0]))
    for addr, diff in sorted_shares[:-1]:
        sats = (miners_pool_sats * diff) // total_diff
        entries.append(PayoutEntry(addr, sats, diff))
        awarded_running += sats
    # Last entry absorbs any rounding remainder (keeps the sum exact).
    last_addr, last_diff = sorted_shares[-1]
    last_sats = miners_pool_sats - awarded_running
    entries.append(PayoutEntry(last_addr, last_sats, last_diff))

    # Min-payout filter: roll dust amounts into the operator's pool (in this
    # revision; a future rev should carry forward to next-block PPLNS).
    rejected_sats = 0
    kept: list[PayoutEntry] = []
    for e in entries:
        if e.amount_sats < policy.min_payout_sats:
            rejected_sats += e.amount_sats
        else:
            kept.append(e)

    # Fee + dust both go to operator. If operator is also a miner this combines.
    operator_total_sats = fee_sats + rejected_sats
    # Merge operator entry if they're also in `kept`.
    merged = False
    for e in kept:
        if e.recipient == policy.operator_address:
            e.amount_sats += operator_total_sats
            merged = True
            break
    if not merged:
        kept.append(PayoutEntry(policy.operator_address, operator_total_sats, 0))

    return PayoutResult(
        block_reward_sats=block_reward_sats,
        fee_sats=fee_sats,
        miners_pool_sats=miners_pool_sats,
        entries=kept,
        pplns_window_difficulty=total_diff,
        rejected_for_min_payout_sats=rejected_sats,
    )
