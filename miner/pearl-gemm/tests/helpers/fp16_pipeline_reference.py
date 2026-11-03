"""Independent Python port of the FP16 scheme's end-to-end tile verification
(``zk-pow/src/api/fp16/verify.rs::verify_tile`` / ``verify_tile_proof``).

The trusted oracle the on-GPU sm_80 :mod:`pearl_gemm.fp16_pipeline` driver is
validated against. The numeric datapath reuses the sibling reference oracles
(:mod:`fp16_noisy_quant_reference`, :mod:`fp16_policy_reference`, whose matmuls
are the committed A100 datapath); the scheme-neutral lottery extractor (mixed-
radix lane assignment, XOR-fold message, keyed-BLAKE3 jackpot ticket, difficulty
check) is reimplemented here from scratch -- deliberately not importing the
driver's ``_layout``, so the two agree only if both match the Rust verifier.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from fractions import Fraction

import blake3
import numpy as np

from .a100_fp16_reference import a100_matmul_bits
from .fp16_noisy_quant_reference import (
    MAX_FP16,
    NORM_FLOOR,
    bf16_max,
    bf16_to_f32,
    derive_row_scales,
    f32_to_bf16,
    f32_to_fp16,
    fp16_to_f32,
)
from .fp16_noisy_quant_reference import row_norms as ref_row_norms
from .fp16_policy_reference import PolicyReport
from .fp16_policy_reference import replay_and_evaluate as ref_replay_and_evaluate

# DimType discriminants and the FP8/v4 jackpot label (shared transcript).
NULL, FOLD, BLAKE = 0, 1, 2
LABEL_JACKPOT = b"pearl/v4/FP8/jackpot"
JACKPOT_ENTRIES = 16
_U32 = 0xFFFFFFFF


@dataclass(frozen=True)
class AxisPattern:
    """Canonical mixed-radix axis pattern (``crate::api::layout::AxisPattern``)."""

    dims: tuple[tuple[int, int], ...]

    @staticmethod
    def new(dims) -> "AxisPattern":
        canon: list[list[int]] = []
        for length, t in dims:
            if length == 1:
                continue
            if canon and canon[-1][1] == t:
                canon[-1][0] *= length
            else:
                canon.append([length, t])
        return AxisPattern(tuple((l, t) for l, t in canon))

    def _offsets(self, keep) -> list[int]:
        offs, stride = [0], 1
        for length, t in self.dims:
            if keep(t):
                base = len(offs)
                for d in range(1, length):
                    offs.extend(offs[i] + d * stride for i in range(base))
            stride *= length
        return offs

    def fold_offsets(self):
        return self._offsets(lambda t: t == FOLD)

    def blake_offsets(self):
        return self._offsets(lambda t: t == BLAKE)

    def tile_offsets(self):
        return self._offsets(lambda t: t != NULL)

    def tile_size(self) -> int:
        n = 1
        for length, t in self.dims:
            if t != NULL:
                n *= length
        return n


def lane_assignment(rows: AxisPattern, cols: AxisPattern) -> list[list[int]]:
    def subtile_positions(axis: AxisPattern):
        tile = axis.tile_offsets()
        fold = axis.fold_offsets()
        pos = {o: i for i, o in enumerate(tile)}
        return [[pos[a + b] for a in fold] for b in axis.blake_offsets()]

    rsub = subtile_positions(rows)
    csub = subtile_positions(cols)
    n_cols = cols.tile_size()
    return [[r * n_cols + c for r in rs for c in cs] for rs in rsub for cs in csub]


def xor_fold_extract(tile_bits, lanes: list[list[int]]) -> bytes:
    out = bytearray(4 * JACKPOT_ENTRIES)
    for li, idxs in enumerate(lanes):
        acc = 0
        for i in idxs:
            acc = (acc * 0x9E3779B1) & _U32
            acc = (acc + (int(tile_bits[i]) & _U32)) & _U32
            acc = ((acc << 13) | (acc >> 19)) & _U32
        out[li * 4 : li * 4 + 4] = struct.pack("<I", acc)
    return bytes(out)


def compute_jackpot_ticket(seed_a: bytes, message: bytes) -> bytes:
    subkey = blake3.blake3(LABEL_JACKPOT, key=seed_a).digest(length=32)
    return blake3.blake3(message, key=subkey).digest(length=32)


def nbits_to_difficulty(nbits: int) -> int:
    exp, mant = nbits >> 24, nbits & 0x00FFFFFF
    if mant == 0 or exp == 0 or (mant & 0x00800000):
        return 0
    return mant >> (8 * (3 - exp)) if exp <= 3 else (mant << (8 * (exp - 3))) & ((1 << 256) - 1)


def check_jackpot_difficulty(jackpot: bytes, nbits: int, h: int, w: int, k: int) -> bool:
    umax = (1 << 256) - 1
    target = nbits_to_difficulty(nbits)
    adj = min(h * w * k, 0xFFFFFFFF)
    bound = umax if adj != 0 and target > umax // adj else target * adj
    return int.from_bytes(jackpot, "little") <= bound


def _rne_f32(fr: Fraction) -> np.float32:
    """Correctly-rounded (RNE, ties-to-even) f32 of an exact rational, checking
    the +-1 ULP neighbours of the f64 approximation (double-rounding error is at
    most one f32 ULP, so the true nearest is always among them)."""
    c0 = np.float32(float(fr))
    up = np.nextafter(c0, np.float32(np.inf))
    dn = np.nextafter(c0, np.float32(-np.inf))
    best, best_key = None, None
    for c in (dn, c0, up):
        dist = abs(Fraction(float(c)) - fr)
        key = (dist, int(np.float32(c).view(np.uint32)) & 1)  # tie -> even last bit
        if best_key is None or key < best_key:
            best, best_key = c, key
    return best


def _noised_operand(rows16: np.ndarray, e16: np.ndarray, f16: np.ndarray, r: int) -> np.ndarray:
    """Rebuild one noised FP16 operand with Rust's exact two-step rounding:
    ``code = RNE_f16(clamp(RNE_f32(alpha*x + RNE_f32(beta*N))))`` -- a single
    f32 FMA (RNE) followed by an independent RNE to FP16 (so double-rounding
    ties resolve exactly as ``f32::mul_add`` then ``f32 -> f16`` do)."""
    num_rows, k = rows16.shape
    floor = f32_to_bf16(np.float32(NORM_FLOOR))
    noise = (
        a100_matmul_bits(e16.reshape(-1), f16.reshape(-1), num_rows, k, r)
        .view(np.float32)
        .reshape(num_rows, k)
    )
    out = np.zeros((num_rows, k), dtype=np.uint16)
    for i in range(num_rows):
        l2 = bf16_max(int(ref_row_norms(rows16[i])[0]), floor)
        linf = bf16_max(int(ref_row_norms(rows16[i])[1]), floor)
        alpha, beta = derive_row_scales(l2, linf, r)
        af = np.float32(bf16_to_f32(alpha))
        bf = np.float32(bf16_to_f32(beta))
        for j in range(k):
            x = np.float16(np.uint16(rows16[i, j]).view(np.float16))
            bn = np.float32(bf * np.float32(noise[i, j]))  # f32 product, RNE
            exact = Fraction(float(af)) * Fraction(float(x)) + Fraction(float(bn))
            val = float(_rne_f32(exact))  # single RNE to f32 == mul_add
            clamped = min(max(val, -MAX_FP16), MAX_FP16)
            out[i, j] = f32_to_fp16(np.float32(clamped))  # independent RNE to f16
    return out


@dataclass
class TileVerify:
    tile_bits: np.ndarray  # (h, w) uint32 f32 bit patterns
    report: PolicyReport
    message: bytes
    ticket: bytes
    built_a: np.ndarray  # (h, k) uint16
    built_b: np.ndarray  # (w, k) uint16


def verify_tile(
    a_rows: np.ndarray,
    b_rows: np.ndarray,
    e_a: np.ndarray,
    f_a: np.ndarray,
    e_b: np.ndarray,
    f_b: np.ndarray,
    rows_pattern: AxisPattern,
    cols_pattern: AxisPattern,
    seed_a: bytes,
    r: int,
) -> TileVerify:
    """Reproduces ``verify_tile``: rebuild both noised operands, replay+score the
    tile, then fold the lottery ticket. ``a_rows`` ``(h, k)`` and ``b_rows``
    ``(w, k)`` are FP16 arrays (float16 or uint16 bit patterns); ``e_*``/``f_*``
    are the FP16 noise factors. Does *not* raise on a rejecting tile."""
    a16 = np.asarray(a_rows).view(np.uint16) if a_rows.dtype == np.float16 else np.asarray(a_rows, np.uint16)
    b16 = np.asarray(b_rows).view(np.uint16) if b_rows.dtype == np.float16 else np.asarray(b_rows, np.uint16)
    h, k = a16.shape
    w, _ = b16.shape

    e_a16 = np.asarray(e_a).view(np.uint16) if np.asarray(e_a).dtype == np.float16 else np.asarray(e_a, np.uint16)
    f_a16 = np.asarray(f_a).view(np.uint16) if np.asarray(f_a).dtype == np.float16 else np.asarray(f_a, np.uint16)
    e_b16 = np.asarray(e_b).view(np.uint16) if np.asarray(e_b).dtype == np.float16 else np.asarray(e_b, np.uint16)
    f_b16 = np.asarray(f_b).view(np.uint16) if np.asarray(f_b).dtype == np.float16 else np.asarray(f_b, np.uint16)
    na = _noised_operand(a16, e_a16.reshape(h, r), f_a16.reshape(k, r), r)
    nb = _noised_operand(b16, e_b16.reshape(w, r), f_b16.reshape(k, r), r)

    tile_bits, _percell, report = ref_replay_and_evaluate(na, nb, h, w, k)

    lanes = lane_assignment(rows_pattern, cols_pattern)
    message = xor_fold_extract(tile_bits.reshape(-1), lanes)
    ticket = compute_jackpot_ticket(seed_a, message)
    return TileVerify(tile_bits=tile_bits, report=report, message=message, ticket=ticket,
                      built_a=na, built_b=nb)
