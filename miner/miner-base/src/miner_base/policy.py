"""Work normalization for the lottery threshold.

Tile admission policy lives solely in the Rust plain verifier
(``zk-pow/src/api/fp8/jackpot_policy.rs`` via
``verify_plain_proof_for_cert_version``); this package must not re-implement it.
"""

from __future__ import annotations


def effective_work(tile_rows: int, tile_cols: int, k: int, rank: int) -> int:
    """``tile_elems * (k - k % r)``: the v4 difficulty adjustment
    (``zk-pow/src/api/verify.rs``) — ``|IA| * |IB| * k`` with ``k`` floored
    to the rank multiple the accumulation consumes."""
    return tile_rows * tile_cols * (k - k % rank)
