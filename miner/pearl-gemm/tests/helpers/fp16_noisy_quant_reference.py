"""Integer/float-exact Python port of the FP16 scheme's fused noisy quantize
(``zk-pow/src/api/fp16/quantization.rs``).

The trusted oracle the on-GPU sm_80 ``fp16_noisy_quant`` kernels are validated
against. Every BF16/FP16 rounding matches the Rust helpers
(``crate::api::fp8::{compute,dtype}``, ``fp16::dtype``, ``prequant``) bit-for-bit;
the ``E@F^T`` noise reuses :func:`a100_matmul_bits` (the committed A100 datapath).
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .a100_fp16_reference import a100_matmul_bits

MAX_FP16 = 65504.0
DELTA = 0.5
NORM_FLOOR = 1.0 / 4294967296.0  # 2^-32
NOISE_TARGET_NORM = 256.0


# ---- dtype conversions ----

def fp16_to_f32(bits: int) -> np.float32:
    return np.float32(np.uint16(bits).view(np.float16))


def f32_to_fp16(x: np.float32) -> int:
    """RNE f32 -> FP16 (input assumed finite and in range)."""
    return int(np.float32(x).astype(np.float16).view(np.uint16))


def bf16_to_f32(bits: int) -> np.float32:
    return np.uint32(np.uint32(bits) << np.uint32(16)).view(np.float32)


def f32_to_bf16(x: np.float32) -> int:
    """RNE f32 -> BF16, matching ``dtype::f32_to_bf16``."""
    bits = int(np.float32(x).view(np.uint32))
    round_bit = (bits >> 16) & 1
    return ((bits + 0x7FFF + round_bit) >> 16) & 0xFFFF


# ---- BF16 element-wise ops (compute.rs) ----

def bf16_mul(a: int, b: int) -> int:
    return f32_to_bf16(np.float32(bf16_to_f32(a) * bf16_to_f32(b)))


def bf16_div(a: int, b: int) -> int:
    return f32_to_bf16(np.float32(bf16_to_f32(a) / bf16_to_f32(b)))


def bf16_max(a: int, b: int) -> int:
    return a if bf16_to_f32(a) >= bf16_to_f32(b) else b


def _round_to_odd_f32(s: float, residual: float) -> np.float32:
    """Correctly-rounded f32 of the exact real ``s + residual`` via round-to-odd
    (``s`` already the f64 RNE approximation; ``residual`` the exact f64 tail)."""
    s32 = np.float32(s)
    err = (s - float(s32)) + residual  # sign(x - s32); nonzero iff x != s32
    if err != 0.0 and (int(s32.view(np.uint32)) & 1) == 0 and np.isfinite(s32):
        direction = np.float32(np.inf) if err > 0.0 else np.float32(-np.inf)
        return np.nextafter(s32, direction)
    return s32


def bf16_fma(a: int, b: int, c: int) -> int:
    """Single-rounding ``a*b + c`` in BF16 (``compute::bf16_fma``)."""
    a64 = float(bf16_to_f32(a))
    b64 = float(bf16_to_f32(b))
    c64 = float(bf16_to_f32(c))
    p = a64 * b64  # exact (two 8-bit significands)
    s = p + c64
    t = s - c64
    r = (p - t) + (c64 - (s - t))  # TwoSum residual, exact
    return f32_to_bf16(_round_to_odd_f32(s, r))


def round_l2_to_grid(l2: int) -> int:
    """``prequant::round_l2_to_grid``: round to nearest multiple of 4 ulps, ties up."""
    return (l2 + 2) & ~3 & 0xFFFF


# ---- per-row norms + scales ----

def row_norms(row: np.ndarray) -> tuple[int, int]:
    """``(l2, linf)`` BF16 bits for an FP16 row (uint16), matching ``row_norms``."""
    k = len(row)
    sumsq = np.float32(0.0)
    absmax = np.float32(0.0)
    for bits in row:
        v = fp16_to_f32(int(bits))
        sumsq = np.float32(sumsq + np.float32(v * v))  # sequential f32 fold
        absmax = np.float32(max(absmax, abs(v)))
    l2 = round_l2_to_grid(f32_to_bf16(np.float32(np.sqrt(np.float32(sumsq / np.float32(k))))))
    linf = f32_to_bf16(absmax)
    return l2, linf


def derive_row_scales(l2: int, linf: int, r: int) -> tuple[int, int]:
    max_fp16 = f32_to_bf16(np.float32(MAX_FP16))
    delta_r = f32_to_bf16(np.float32(DELTA * math.sqrt(r)))
    delta_over_std = f32_to_bf16(
        np.float32(DELTA * math.sqrt(r) / (NOISE_TARGET_NORM * NOISE_TARGET_NORM))
    )
    noised_bound = bf16_fma(delta_r, l2, linf)
    alpha = bf16_div(max_fp16, noised_bound)
    beta = bf16_mul(bf16_mul(alpha, l2), delta_over_std)
    return alpha, beta


def _rne_f32(fr: Fraction) -> np.float32:
    """Correctly-rounded (RNE, ties-to-even) f32 of an exact rational, checking
    the +-1 ULP neighbours of the f64 approximation (the true nearest is always
    among them)."""
    c0 = np.float32(float(fr))
    best, best_key = None, None
    for c in (np.nextafter(c0, np.float32(-np.inf)), c0, np.nextafter(c0, np.float32(np.inf))):
        dist = abs(Fraction(float(c)) - fr)
        key = (dist, int(np.float32(c).view(np.uint32)) & 1)  # tie -> even last bit
        if best_key is None or key < best_key:
            best, best_key = c, key
    return best


def _fma_f32(af: np.float32, x: np.float32, bn: np.float32) -> np.float32:
    """Correctly-rounded (RNE) f32 ``af*x + bn`` -- Rust ``f32::mul_add``.

    A true f32 FMA, so the caller's subsequent RNE to FP16 double-rounds exactly
    as Rust (``mul_add`` then ``f32 -> f16``) and the GPU (``fmaf`` then
    ``__float2half_rn``) do; round-to-odd here would diverge in ties.
    """
    exact = Fraction(float(af)) * Fraction(float(x)) + Fraction(float(bn))
    return _rne_f32(exact)


def noisy_quantize(
    rows: np.ndarray, e: np.ndarray, f: np.ndarray, norms: list[tuple[int, int]], r: int
) -> dict:
    """Returns ``{noised, alpha, beta, l2}`` (all uint16 arrays)."""
    num_rows = len(norms)
    k = rows.shape[1]
    rows_bits = np.ascontiguousarray(rows).view(np.uint16).reshape(num_rows, k)
    floor = f32_to_bf16(np.float32(NORM_FLOOR))
    # N = E @ F^T on the committed A100 FP16 datapath (f32 bits -> f32).
    noise_bits = a100_matmul_bits(
        e.reshape(-1).view(np.uint16), f.reshape(-1).view(np.uint16), num_rows, k, r
    )
    noise = noise_bits.view(np.float32).reshape(num_rows, k)

    noised = np.zeros((num_rows, k), dtype=np.uint16)
    alphas = np.zeros(num_rows, dtype=np.uint16)
    betas = np.zeros(num_rows, dtype=np.uint16)
    l2s = np.zeros(num_rows, dtype=np.uint16)
    for i in range(num_rows):
        l2 = bf16_max(norms[i][0], floor)
        linf = bf16_max(norms[i][1], floor)
        alpha, beta = derive_row_scales(l2, linf, r)
        af = bf16_to_f32(alpha)
        bf = bf16_to_f32(beta)
        for j in range(k):
            x = fp16_to_f32(int(rows_bits[i, j]))
            bn = np.float32(bf * np.float32(noise[i, j]))
            val = float(_fma_f32(af, x, bn))
            clamped = min(max(val, -MAX_FP16), MAX_FP16)
            noised[i, j] = f32_to_fp16(np.float32(clamped))
        alphas[i] = alpha
        betas[i] = beta
        l2s[i] = l2
    return {"noised": noised, "alpha": alphas, "beta": betas, "l2": l2s}
