"""Scheme-neutral lottery extractor: the committed mixed-radix tile layout, the
XOR-fold message extractor, the keyed-BLAKE3 jackpot ticket, and the difficulty
check -- torch-free, bit-for-bit against the Rust verifier.

These four helpers are reused verbatim by the FP16 verifier from the FP8 modules
(``crate::api::layout::{AxisPattern, lane_assignment}``,
``crate::api::fp8::utils::xor_fold_extract``,
``crate::api::fp8::transcript::compute_jackpot_ticket``, and
``crate::api::proof_utils::check_jackpot_difficulty``); the FP16 tile-pipeline
driver folds its replayed tile into the ticket through them. Nothing here touches
CUDA, so the standalone sm_80 harness and the CI test share this exact code.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import blake3

# DimType discriminants (``crate::api::layout::DimType``).
NULL = 0
FOLD = 1
BLAKE = 2
NONE = 3

# 16 Blake lanes = 16 u32 entries = 64-byte jackpot message (one BLAKE3 block).
JACKPOT_ENTRIES = 16

# The FP8/v4 jackpot label; the FP16 scheme folds through the same transcript.
LABEL_JACKPOT = b"pearl/v4/FP8/jackpot"

_U32 = 0xFFFFFFFF


@dataclass(frozen=True)
class AxisPattern:
    """A typed mixed-radix pattern for one tile axis (``crate::api::layout``).

    ``dims`` is an ordered low-stride-first list of ``(length, DimType)`` with
    implicit stride = product of the preceding lengths. Constructed canonical:
    length-1 dims dropped, adjacent same-type dims merged -- matching
    ``AxisPattern::new`` so the offset enumerations are identical.
    """

    dims: tuple[tuple[int, int], ...]

    @staticmethod
    def new(dims) -> "AxisPattern":
        canonical: list[list[int]] = []
        for length, dim_type in dims:
            if length < 1:
                raise ValueError("dim length must be >= 1")
            if length == 1:
                continue
            if canonical and canonical[-1][1] == dim_type:
                canonical[-1][0] *= length
            else:
                canonical.append([length, dim_type])
        if canonical and canonical[-1][1] == NULL:
            raise ValueError("trailing Null dim is redundant; omit it")
        return AxisPattern(tuple((l, t) for l, t in canonical))

    def _offsets_where(self, include) -> list[int]:
        offsets = [0]
        stride = 1
        for length, t in self.dims:
            if include(t):
                base = len(offsets)
                for d in range(1, length):
                    for i in range(base):
                        offsets.append(offsets[i] + d * stride)
            stride *= length
        return offsets

    def fold_offsets(self) -> list[int]:
        return self._offsets_where(lambda t: t == FOLD)

    def blake_offsets(self) -> list[int]:
        return self._offsets_where(lambda t: t == BLAKE)

    def tile_offsets(self) -> list[int]:
        return self._offsets_where(lambda t: t != NULL)

    def _product(self, include) -> int:
        p = 1
        for length, t in self.dims:
            if include(t):
                p *= length
        return p

    def fold_size(self) -> int:
        return self._product(lambda t: t == FOLD)

    def blake_size(self) -> int:
        return self._product(lambda t: t == BLAKE)

    def tile_size(self) -> int:
        return self.fold_size() * self.blake_size()


def lane_assignment(rows: AxisPattern, cols: AxisPattern) -> list[list[int]]:
    """Map each of the 16 lanes to its subtile's flat row-major tile indices,
    bit-for-bit with ``crate::api::layout::lane_assignment``.

    Lane ``rank(b_r) * |B_c| + rank(b_c)`` folds subtile ``(A_r + b_r) x (A_c +
    b_c)`` row-major over the sorted fold offsets; flat indices address the
    ``rows.tile_size() x cols.tile_size()`` output tile.
    """

    def subtile_positions(axis: AxisPattern) -> list[list[int]]:
        tile = axis.tile_offsets()
        fold = axis.fold_offsets()
        pos = {off: i for i, off in enumerate(tile)}
        return [[pos[a + b] for a in fold] for b in axis.blake_offsets()]

    row_subtiles = subtile_positions(rows)
    col_subtiles = subtile_positions(cols)
    n_cols = cols.tile_size()
    lanes: list[list[int]] = []
    for rs in row_subtiles:
        for cs in col_subtiles:
            lanes.append([r * n_cols + c for r in rs for c in cs])
    return lanes


def xor_fold_extract(tile_bits, lane_indices: list[list[int]]) -> bytes:
    """The 64-byte lottery message: per lane, a rolling
    ``(acc*0x9E3779B1 + f32_bits).rotate_left(13)`` over the lane's cells,
    little-endian; bit-for-bit with ``crate::api::fp8::utils::xor_fold_extract``.

    ``tile_bits`` is the flat row-major tile as f32 *bit patterns* (``u32``), so
    ``-0.0`` (``0x80000000``) and ``+0.0`` fold differently, exactly as the
    reference hashes ``f32::to_bits()``.
    """
    if len(lane_indices) != JACKPOT_ENTRIES:
        raise ValueError(f"expected {JACKPOT_ENTRIES} lanes, got {len(lane_indices)}")
    out = bytearray(4 * JACKPOT_ENTRIES)
    for lane, idxs in enumerate(lane_indices):
        acc = 0
        for i in idxs:
            acc = (acc * 0x9E3779B1) & _U32
            acc = (acc + (int(tile_bits[i]) & _U32)) & _U32
            acc = ((acc << 13) | (acc >> 19)) & _U32  # rotate_left(13)
        out[lane * 4 : lane * 4 + 4] = struct.pack("<I", acc)
    return bytes(out)


def compute_jackpot_ticket(seed_a: bytes, message: bytes) -> bytes:
    """``H_"jackpot"(message; seed_a)`` -- the 32-byte jackpot digest, bit-for-bit
    with ``crate::api::fp8::transcript::compute_jackpot_ticket``.

    ``subkey = BLAKE3(LABEL_JACKPOT, key=seed_a)``; then
    ``jackpot = BLAKE3(message, key=subkey)``.
    """
    if len(seed_a) != 32:
        raise ValueError(f"seed_a must be 32 bytes, got {len(seed_a)}")
    subkey = blake3.blake3(LABEL_JACKPOT, key=seed_a).digest(length=32)
    return blake3.blake3(message, key=subkey).digest(length=32)


def nbits_to_difficulty(nbits: int) -> int:
    """Bitcoin compact ``nbits`` -> absolute U256 target (``proof_utils``)."""
    exponent = nbits >> 24
    mantissa = nbits & 0x00FFFFFF
    if mantissa == 0 or exponent == 0 or (mantissa & 0x00800000):
        return 0
    if exponent <= 3:
        return mantissa >> (8 * (3 - exponent))
    return (mantissa << (8 * (exponent - 3))) & ((1 << 256) - 1)


def check_jackpot_difficulty(jackpot: bytes, nbits: int, h: int, w: int, k: int) -> bool:
    """Whether the jackpot clears the difficulty target, bit-for-bit with
    ``crate::api::proof_utils::check_jackpot_difficulty``:
    ``le(jackpot) <= min(U256::MAX, target * saturating(h*w*k))``.
    """
    umax = (1 << 256) - 1
    target = nbits_to_difficulty(nbits)
    adjustment = h * w * k
    if adjustment > 0xFFFFFFFF:  # u32 checked_mul -> u32::MAX on overflow
        adjustment = 0xFFFFFFFF
    bound = umax if adjustment != 0 and target > umax // adjustment else target * adjustment
    return int.from_bytes(jackpot, "little") <= bound
