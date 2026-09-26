"""Lottery extractor, work normalization, and winning condition.

The lottery runs on the NON-PEELED first-stage product ``acc = A' @ B'.T``
(``matmul_dtype`` stored as FP32). ``XorFoldExtractor`` compresses the tile,
``is_winning`` gates the strong hash against ``target * work``.

Tile admission policy lives solely in the Rust plain verifier
(``zk-pow/src/api/fp8/jackpot_policy.rs`` via
``verify_plain_proof_for_cert_version``); this package must not re-implement it.
"""

from __future__ import annotations

import numpy as np
import torch

from .hardware import DType
from .layout import LANES as LAYOUT_LANES
from .transcript import jackpot_digest

_MAX_256 = (1 << 256) - 1


def _rotl32(x: int, n: int) -> int:
    x &= 0xFFFFFFFF
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


class XorFoldExtractor:
    """Reference epilogue: fold each of the tile's 16 committed subtiles into one lane.

    The lane layout (which tile element feeds which lane, in what order) is
    committed via the operand patterns (``layout.py``);
    ``lane_indices[j]`` = lane ``j``'s flat tile indices in fold order. The
    16 x u32 output is 64 bytes -- one BLAKE3 block for the lottery hash.
    (Consensus-whitelisted in the real protocol; a placeholder here.)
    """

    LANES = LAYOUT_LANES

    def __init__(self, lane_indices: list[list[int]]):
        assert len(lane_indices) == self.LANES, f"expected {self.LANES} lanes"
        self.lane_indices = lane_indices
        self._tile_elems = sum(len(idxs) for idxs in lane_indices)

    def extract(self, c_tile: torch.Tensor) -> bytes:
        assert c_tile.dtype == DType.MATMUL.value, "extractor runs on matmul_dtype"
        # Reinterpret each matmul_dtype word as its same-width unsigned integer.
        words = c_tile.reshape(-1).contiguous().view(torch.int32).numpy().astype(np.uint32)
        assert words.size == self._tile_elems, "tile shape does not match the committed layout"
        lanes = [0] * self.LANES
        for j, idxs in enumerate(self.lane_indices):
            for i in idxs:
                lanes[j] = _rotl32((lanes[j] * 0x9E3779B1 + int(words[i])) & 0xFFFFFFFF, 13)
        return b"".join(int(x).to_bytes(4, "little") for x in lanes)  # 64 bytes


def effective_work(tile_rows: int, tile_cols: int, k: int, rank: int) -> int:
    """``tile_elems * (k - k % r)``: the v4 difficulty adjustment
    (``zk-pow/src/api/verify.rs``) — ``|IA| * |IB| * k`` with ``k`` floored
    to the rank multiple the accumulation consumes."""
    return tile_rows * tile_cols * (k - k % rank)


def is_winning(extracted: bytes, noise_seed_a: bytes, target: int, work: int) -> bytes | None:
    """The labelled jackpot digest when it meets ``target * work`` (capped)."""

    threshold = min(int(target * work), _MAX_256)
    digest = jackpot_digest(extracted, noise_seed_a)
    return digest if int.from_bytes(digest, "little") <= threshold else None
