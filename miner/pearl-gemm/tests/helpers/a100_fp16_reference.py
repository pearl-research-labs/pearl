"""Integer-exact Python port of the A100 (``sm_80``) ``HMMA.16816.F32`` FP16
accumulation model.

This mirrors ``zk-pow/src/api/fp16/accumulate.rs`` (``a100_dot`` /
``a100_matmul``) bit-for-bit and is the trusted oracle the on-GPU sm_80 GEMM is
validated against. The k axis is split into groups of ``GROUP = 8`` products;
per group ``eta = max(product stored-exp sums, accumulator stored exp clamped
>= -126)``; every product and the accumulator are truncated toward zero onto the
``2^(eta - W)`` grid (``W = 24``); the integers are summed exactly; the sum is
rounded toward zero to FP32 after every group. A zero result is ``+0``;
``|result| >= 2^128`` overflows.

All arithmetic is done with Python ``int`` (unbounded), so the model is exact;
the final per-group result is encoded as an IEEE binary32 bit pattern directly,
so no host float rounding ever intervenes.
"""

from __future__ import annotations

import struct

import numpy as np

W = 24  # internal accumulator significand bits
GROUP = 8  # products per hardware accumulation group
_NEG = (-(2**31)) // 2  # sentinel "no exponent" (i32::MIN / 2 in the Rust model)
_FP32_MIN_EXP = -149


def decompose_fp16(bits: int) -> tuple[int, int, int]:
    """``(sign, significand, stored_exp)`` of an FP16 bit pattern.

    Matches ``dtype::decompose_fp16``: zero -> ``(1, 0, 0)``; subnormal ->
    ``(sign, man, -14)``; normal -> ``(sign, 1024 | man, exp - 15)``. NaN/inf
    (exponent field ``0x1F``) is rejected (never occurs in the scheme).
    """
    bits &= 0xFFFF
    exp = (bits >> 10) & 0x1F
    man = bits & 0x03FF
    if exp == 0x1F:
        raise ValueError(f"FP16 NaN/inf encoding {bits:#06x} is not allowed")
    sign = -1 if (bits & 0x8000) else 1
    if exp == 0 and man == 0:
        return (1, 0, 0)
    if exp == 0:
        return (sign, man, -14)
    return (sign, 0x400 | man, exp - 15)


def _acc_parts(c_bits: int) -> tuple[int, int, int, int]:
    """``(sign, significand, stored_exp, ulp_exp)`` of an FP32 accumulator given
    its u32 bit pattern. ``value = sign * significand * 2^ulp``; ``stored_exp``
    is ``_NEG`` for zero (does not participate in the alignment max)."""
    c_bits &= 0xFFFFFFFF
    sign = -1 if (c_bits >> 31) else 1
    exp_field = (c_bits >> 23) & 0xFF
    man = c_bits & 0x7FFFFF
    if exp_field == 0 and man == 0:
        return (1, 0, _NEG, 0)
    if exp_field == 0xFF:
        raise ValueError("non-finite FP32 accumulator")
    if exp_field > 0:
        el = exp_field - 127
        return (sign, 0x800000 | man, el, el - 23)
    # subnormal: value = man * 2^-149, stored exponent clamped to -126
    return (sign, man, -126, _FP32_MIN_EXP)


def _shift(x: int, s: int) -> int:
    """``x`` shifted by ``s`` (left if positive, truncating-toward-zero right if
    negative). Python ``>>`` floors, so operate on the magnitude."""
    if s >= 0:
        return x << s
    if -s >= 127:
        return 0
    if x >= 0:
        return x >> (-s)
    return -((-x) >> (-s))


def _rz_to_f32_bits(s: int, unit: int) -> int:
    """Round the integer ``s * 2^unit`` toward zero to an FP32 value, returned as
    its u32 bit pattern (24 significant bits; subnormals floored onto the
    ``2^-149`` grid). Mirrors the Rust ``rz_to_f32`` exactly: after truncating
    the magnitude to 24 significant bits (floored to the ``2^-149`` subnormal
    grid), ``sign * truncated * 2^unit`` is an exact IEEE value (<=24 sig bits),
    so the f64->f32 cast introduces no rounding."""
    if s == 0:
        return 0
    sign = -1.0 if s < 0 else 1.0
    a = abs(s)
    nb = a.bit_length() - 1  # floor(log2|s|)
    keep = max(nb + unit - 23, _FP32_MIN_EXP)
    drop = min(max(keep - unit, 0), 127)
    truncated = (a >> drop) << drop
    # Exact in f64 (truncated has <=24 significant bits), then exact down-cast.
    val = np.float64(sign) * np.float64(truncated) * np.float64(2.0) ** np.float64(unit)
    return int(np.float32(val).view(np.uint32))


def a100_dot_bits(a: list[int], b: list[int], c_bits: int) -> int:
    """A100 FP16 dot product of operand bit-pattern rows ``a`` and ``b`` with
    FP32 carry-in ``c_bits`` (u32); returns the FP32 result as a u32 bit
    pattern. Raises on overflow/non-finite (as the verifier aborts)."""
    assert len(a) == len(b), "operand length mismatch"
    cur = c_bits & 0xFFFFFFFF
    k = len(a)
    g0 = 0
    while g0 < k:
        g1 = min(g0 + GROUP, k)
        cs, cm, cel, culp = _acc_parts(cur)
        eta = cel
        for u in range(g0, g1):
            _, ma, ea = decompose_fp16(a[u])
            _, mb, eb = decompose_fp16(b[u])
            if ma != 0 and mb != 0:
                eta = max(eta, ea + eb)
        if eta == _NEG:
            g0 = g1
            continue
        unit = eta - W
        total = 0
        for u in range(g0, g1):
            sa, ma, ea = decompose_fp16(a[u])
            sb, mb, eb = decompose_fp16(b[u])
            if ma == 0 or mb == 0:
                continue
            prod = ma * mb  # < 2^22, exact
            sh = (ea + eb) - 20 - unit  # product LSB is 2^(ea+eb-20)
            aligned = _shift(prod, sh)
            total += (sa * sb) * aligned
        csh = culp - unit
        acc_aligned = _shift(cm, csh)
        total += cs * acc_aligned
        new_bits = _rz_to_f32_bits(total, unit)
        if (new_bits >> 23) & 0xFF == 0xFF:
            raise OverflowError("A100 accumulation overflowed to non-finite")
        cur = new_bits
        g0 = g1
    return cur & 0xFFFFFFFF


def a100_dot(a: list[int], b: list[int], c: float) -> float:
    """Float wrapper around :func:`a100_dot_bits`."""
    c_bits = struct.unpack("<I", struct.pack("<f", c))[0]
    d_bits = a100_dot_bits(a, b, c_bits)
    return struct.unpack("<f", struct.pack("<I", d_bits))[0]


def a100_matmul_bits(
    a: np.ndarray, b: np.ndarray, m: int, n: int, k: int, acc: np.ndarray | None = None
) -> np.ndarray:
    """Device matmul of ``m x k`` operand ``a`` against ``n x k`` operand ``b``
    (row-major FP16 bit patterns, ``uint16``), with optional ``m x n`` FP32
    carry-in (``uint32`` bits). Returns the ``m x n`` FP32 tile as ``uint32``
    bit patterns. ``b`` is the transposed logical operand (row j is column j)."""
    a = np.asarray(a, dtype=np.uint16).reshape(m, k)
    b = np.asarray(b, dtype=np.uint16).reshape(n, k)
    out = np.zeros((m, n), dtype=np.uint32)
    for i in range(m):
        ai = a[i].tolist()
        for j in range(n):
            c_bits = int(acc[i, j]) if acc is not None else 0
            out[i, j] = a100_dot_bits(ai, b[j].tolist(), c_bits)
    return out
