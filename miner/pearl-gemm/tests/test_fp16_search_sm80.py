"""On-GPU bit-exactness of the A100 (``sm_80``) full-matrix FP16 lottery search.

Runs the hand-written search + first-winner latch kernel on the local GA100 and
asserts, for the SAME noised operands ``A'``/``B'``:

  * every committed tile's 32-byte jackpot ticket matches, bit-for-bit, an
    independent Python scan that computes each tile via the verifier reference
    (``tests/helpers/fp16_policy_reference.replay_and_evaluate`` for the
    bit-exact A100 accumulation, then ``fp16_pipeline_reference``'s XOR-fold +
    keyed-BLAKE3 ticket), and
  * the latched first-winner ``(tile_row, tile_col, ticket)`` equals the Python
    scan's first winner (lowest flat tile index ``tr*ntc + tc``), both for an
    EASY ``nbits`` (winners exist) and a HARD ``nbits`` (no winner -> empty
    latch).

Policy is deliberately NOT evaluated in the search kernel (it latches purely on
the jackpot difficulty threshold, mirroring fp8); the host driver scores
``fp16_policy`` on the latched tile later. Gated to ``sm_80`` hardware.
"""

import blake3
import numpy as np
import pytest
import torch

from pearl_gemm._utils._arch import Arch, arch_of

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or arch_of() is not Arch.SM80,
    reason="the A100 FP16 lottery-search kernel targets sm_80 (GA100) hardware",
)

# DimType discriminants (crate::api::layout::DimType).
_BLAKE, _FOLD = 2, 1
_LABEL_JACKPOT = b"pearl/v4/FP8/jackpot"


class _Gen:
    """Deterministic xorshift FP16 stream with spread magnitudes (dense
    breakpoints, the honest regime), reused from the pipeline test."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def _n(self):
        self.s ^= (self.s << 13) & 0xFFFFFFFFFFFFFFFF
        self.s ^= self.s >> 7
        self.s ^= (self.s << 17) & 0xFFFFFFFFFFFFFFFF
        return self.s

    def operand(self, n):
        out = np.empty(n, np.float16)
        for i in range(n):
            r = self._n()
            sign = 1.0 if (r & 1) == 0 else -1.0
            exp = int((r >> 1) % 9) - 3
            mant = 1.0 + ((r >> 8) % 1024) / 1024.0
            out[i] = np.float16(sign * mant * (2.0 ** exp))
        return out


def _pow_key(seed_a: bytes) -> bytes:
    """The jackpot subkey the host precomputes and passes to the kernel."""
    return blake3.blake3(_LABEL_JACKPOT, key=seed_a).digest(length=32)


def _oracle_scan(a16, b16, seed_a, nbits, rp_dims, cp_dims, h, w, k):
    """Independent Python scan: per-tile tickets and the first winner."""
    from tests.helpers.fp16_pipeline_reference import (
        AxisPattern as RefAxis,
        check_jackpot_difficulty,
        compute_jackpot_ticket,
        lane_assignment,
        xor_fold_extract,
    )
    from tests.helpers.fp16_policy_reference import replay_and_evaluate as ref_replay

    lanes = lane_assignment(RefAxis.new(rp_dims), RefAxis.new(cp_dims))
    ntr, ntc = a16.shape[0] // h, b16.shape[0] // w
    tickets = []
    first_winner = None  # (flat_idx, tr, tc, ticket)
    for tr in range(ntr):
        for tc in range(ntc):
            a_tile = a16[tr * h : (tr + 1) * h]
            b_tile = b16[tc * w : (tc + 1) * w]
            tile_bits, _pc, _rep = ref_replay(a_tile, b_tile, h, w, k)
            message = xor_fold_extract(tile_bits.reshape(-1), lanes)
            ticket = compute_jackpot_ticket(seed_a, message)
            tickets.append(ticket)
            if first_winner is None and check_jackpot_difficulty(ticket, nbits, h, w, k):
                first_winner = (tr * ntc + tc, tr, tc, ticket)
    return tickets, first_winner


# (h, w, k, rp_dims, cp_dims, m, n): h/w == pattern tile_size, 16 blake lanes,
# whole lottery tiles in m and n.
_SHAPES = [
    (4, 64, 128, [(4, _BLAKE)], [(4, _BLAKE), (16, _FOLD)], 16, 256),
    (4, 64, 256, [(4, _BLAKE)], [(4, _BLAKE), (16, _FOLD)], 12, 128),
    (16, 64, 128, [(4, _BLAKE), (4, _FOLD)], [(4, _BLAKE), (16, _FOLD)], 32, 128),
]


@pytest.mark.parametrize(("h", "w", "k", "rp", "cp", "m", "n"), _SHAPES)
def test_search_is_bit_exact_vs_oracle(h, w, k, rp, cp, m, n):
    from pearl_gemm.fp16_search import search
    from pearl_gemm.fp16_pipeline import AxisPattern

    g = _Gen(0x5EED_1234_ABCD_0001 ^ (h * 131 + w * 17 + k + m * 7 + n * 3))
    a = g.operand(m * k).reshape(m, k)
    b = g.operand(n * k).reshape(n, k)
    a16 = a.view(np.uint16)
    b16 = b.view(np.uint16)
    seed_a = bytes([(h + w + k) & 0xFF]) * 32
    pow_key = _pow_key(seed_a)
    rpa, cpa = AxisPattern.new(rp), AxisPattern.new(cp)
    a_dev = torch.from_numpy(np.ascontiguousarray(a)).cuda()
    b_dev = torch.from_numpy(np.ascontiguousarray(b)).cuda()

    # EASY target: every ticket is a winner, so the first tile (0, 0) must latch.
    easy_nbits = 0x207F_FFFF  # exponent 0x20 -> target ~ U256::MAX, so bound saturates
    hit, tickets = search(a_dev, b_dev, pow_key, easy_nbits, rpa, cpa, collect_tickets=True)

    ref_tickets, ref_first = _oracle_scan(a16, b16, seed_a, easy_nbits, rp, cp, h, w, k)
    got_tickets = [
        b"".join(int(wd).to_bytes(4, "little") for wd in tickets[i].numpy())
        for i in range(tickets.shape[0])
    ]
    mismatches = sum(1 for i in range(len(ref_tickets)) if got_tickets[i] != ref_tickets[i])
    assert mismatches == 0, f"{mismatches}/{len(ref_tickets)} per-tile ticket mismatches"

    assert ref_first is not None, "easy target should have a winner"
    assert hit.found
    assert (hit.tile_row, hit.tile_col) == (ref_first[1], ref_first[2]), (
        f"latched tile {(hit.tile_row, hit.tile_col)} != oracle {(ref_first[1], ref_first[2])}"
    )
    assert hit.ticket == ref_first[3], "latched ticket differs from oracle first winner"

    # HARD target: impossible (zero) difficulty -> no winner, empty latch.
    hard_hit = search(a_dev, b_dev, pow_key, 0, rpa, cpa)
    _, ref_hard = _oracle_scan(a16, b16, seed_a, 0, rp, cp, h, w, k)
    assert ref_hard is None, "zero difficulty should admit no winner"
    assert not hard_hit.found
    assert hard_hit.tile_row == -1 and hard_hit.tile_col == -1


def test_search_first_winner_is_lowest_tile_index():
    """With a medium bound chosen so only SOME tiles win, the latched tile is the
    lowest flat index ``tr*ntc + tc`` among winners, matching the oracle."""
    from pearl_gemm.fp16_search import difficulty_bound, search
    from pearl_gemm.fp16_pipeline import AxisPattern, check_jackpot_difficulty

    h, w, k = 4, 64, 128
    rp, cp = [(4, _BLAKE)], [(4, _BLAKE), (16, _FOLD)]
    m, n = 24, 256
    g = _Gen(0xC0FF_EE00_1357_9BDF)
    a = g.operand(m * k).reshape(m, k)
    b = g.operand(n * k).reshape(n, k)
    a16, b16 = a.view(np.uint16), b.view(np.uint16)
    seed_a = b"\x3c" * 32
    pow_key = _pow_key(seed_a)
    rpa, cpa = AxisPattern.new(rp), AxisPattern.new(cp)
    a_dev = torch.from_numpy(np.ascontiguousarray(a)).cuda()
    b_dev = torch.from_numpy(np.ascontiguousarray(b)).cuda()

    # Sweep nbits to find one that produces a partial (not all / not none) winner
    # set, then confirm the kernel's first winner == the oracle's lowest index.
    ref_all, _ = _oracle_scan(a16, b16, seed_a, 0x207F_FFFF, rp, cp, h, w, k)
    vals = [int.from_bytes(t, "little") for t in ref_all]
    # Sweep exponent x mantissa so the bound lands between ticket values.
    candidates = [
        (exp << 24) | mant
        for exp in (0x1D, 0x1E, 0x1F, 0x20)
        for mant in (0x7FFFFF, 0x3FFFFF, 0x1FFFFF, 0x0FFFFF, 0x03FFFF, 0x00FFFF)
    ]
    found_partial = False
    for nbits in candidates:
        bound = difficulty_bound(nbits, h, w, k)
        winners = [i for i, v in enumerate(vals) if v <= bound]
        if 0 < len(winners) < len(vals):
            hit = search(a_dev, b_dev, pow_key, nbits, rpa, cpa)
            ntc = n // w
            expected = min(winners)
            assert hit.found
            assert hit.tile_row * ntc + hit.tile_col == expected, (
                f"nbits={nbits:#x}: kernel first winner "
                f"{hit.tile_row * ntc + hit.tile_col} != oracle {expected}"
            )
            # Cross-check the latched ticket clears the threshold.
            assert check_jackpot_difficulty(hit.ticket, nbits, h, w, k)
            found_partial = True
            break
    assert found_partial, "no nbits produced a partial winner set; widen the sweep"
