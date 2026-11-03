"""Integer/float-exact Python port of the FP16 scheme's deterministic noise-line
generation (``zk-pow/src/api/fp16/noise.rs``: ``sample_line`` / ``sample_noise``).

The trusted oracle the on-GPU ``sm_80`` ``fp16_noise_lines`` kernel is validated
against. The keyed-BLAKE3 XOF uses the same ``blake3`` crate the reference
hashes with; every BF16/FP16 rounding matches the Rust helpers
(``crate::api::fp8::{compute,dtype}``, ``fp16::dtype``) bit-for-bit, and the L2
normalization uses the exact integer ``isqrt`` + one-BF16-division recipe.
"""

from __future__ import annotations

import math

import blake3
import numpy as np

from .fp16_noisy_quant_reference import (
    bf16_div,
    bf16_mul,
    bf16_to_f32,
    f32_to_bf16,
    f32_to_fp16,
)

LABEL_NOISE_LINE = b"pearl/v4/FP16/noise-line"
INT_SQRT_PREC = 32
NOISE_TARGET_NORM = 256.0

SIDE_A, SIDE_B = 0, 1
FACTOR_E, FACTOR_F = 0, 1


def line_key(seed: bytes) -> bytes:
    """``subkey(LABEL_NOISE_LINE, seed)`` = keyed BLAKE3 of the label under seed."""
    assert len(seed) == 32
    return blake3.blake3(LABEL_NOISE_LINE, key=seed).digest(length=32)


def xof_bytes(seed: bytes, side: int, factor: int, line: int, r: int) -> bytes:
    """The raw keyed-BLAKE3-XOF bytes of one line (``sample_line_xof_bytes``)."""
    material = bytes([side & 0xFF, factor & 0xFF]) + int(line).to_bytes(4, "little")
    assert len(material) <= 64
    material = material.ljust(64, b"\x00")
    return blake3.blake3(material, key=line_key(seed)).digest(length=r)


def normalize_line(byts: bytes) -> np.ndarray:
    """Decode + L2-normalize XOF bytes to FP16 ``u16`` (``normalize_line``)."""
    xs = []
    sumsq = 0
    for b in byts:
        sign = 1 - 2 * (b >> 7)
        mag = (b & 0x7F) + 1
        xs.append(sign * mag)
        sumsq += mag * mag
    norm_scaled = math.isqrt(sumsq * (INT_SQRT_PREC * INT_SQRT_PREC))  # exact floor isqrt
    numer = f32_to_bf16(np.float32(NOISE_TARGET_NORM * INT_SQRT_PREC))  # 8192, exact bf16
    denom = f32_to_bf16(np.float32(float(norm_scaled)))
    scale = bf16_div(numer, denom)
    out = np.empty(len(byts), dtype=np.uint16)
    for i, xi in enumerate(xs):
        xb = f32_to_bf16(np.float32(float(xi)))
        entry = bf16_mul(xb, scale)
        out[i] = f32_to_fp16(bf16_to_f32(entry))
    return out


def sample_line(seed: bytes, side: int, factor: int, line: int, r: int) -> np.ndarray:
    return normalize_line(xof_bytes(seed, side, factor, line, r))


def noise_lines(seed: bytes, side: int, factor: int, indices, r: int) -> np.ndarray:
    """``(len(indices), r)`` ``u16`` FP16 lines, row ``i`` = ``sample_line(... indices[i])``."""
    rows = [sample_line(seed, side, factor, int(idx), r) for idx in indices]
    if not rows:
        return np.empty((0, r), dtype=np.uint16)
    return np.stack(rows)


def sample_noise(seed_a: bytes, seed_b: bytes, k: int, r: int, a_rows, b_cols):
    """``(e_a, f_a, e_b, f_b)`` matching the reference ``sample_noise`` layout."""
    e_a = noise_lines(seed_a, SIDE_A, FACTOR_E, a_rows, r)
    e_b = noise_lines(seed_b, SIDE_B, FACTOR_E, b_cols, r)
    f_a = noise_lines(seed_b, SIDE_A, FACTOR_F, range(k), r)
    f_b = noise_lines(seed_b, SIDE_B, FACTOR_F, range(k), r)
    return e_a, f_a, e_b, f_b
