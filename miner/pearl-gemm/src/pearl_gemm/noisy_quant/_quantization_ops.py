"""Packed rounding, scale-chain, and E1 device helpers."""

import struct

import cutlass
import cutlass.cute as cute

from .._utils._stats_ops import _asm_f32, _asm_u32
from ..protocol_constants import (
    _INT_SQRT_PREC,
    NOISE_TARGET_NORM,
    QUANT_MAX,
    STRIP_COIN_SHIFT,
    STRIP_COIN_THRESHOLD,
    STRIP_EXP_OFFSET,
    STRIP_G,
    STRIP_G_SMALL_RMS,
    STRIP_SMALL_RMS,
    R,
)
from ..tensor_hash_plus_stats._blake3_ops import (
    SINGLE_BLOCK_KEYED_FLAGS,
    compress_rolled,
)

MAX_E4M3 = QUANT_MAX  # 448.0, the e4m3 quant-grid ceiling
_E4M3_BIAS = 7
_NORM_FLOOR = 2.0**-32  # row_norms' (near-)zero-row floor on l2 and linf
# Noise-line normalization constants: the target norm times the
# fixed-point isqrt factor, and the squared factor applied under the isqrt.
_LINE_NORM_NUM = float(NOISE_TARGET_NORM * _INT_SQRT_PREC)  # 8192, exact bf16
_LINE_SQRT_SQ = _INT_SQRT_PREC * _INT_SQRT_PREC  # 1024

# The v4 noise-line address prefix ``side(1) | factor(1)`` (side A = 0, B = 1;
# factor E = 0, F = 1). The
# full 64-byte block message is ``side | factor | u32 LE line`` zero-padded;
# ``k`` and ``r`` are bound by the side's noise seed, not the address. E1/F1
# are A's row lines and shared basis, E2/F2 are B's.
_L_E1 = bytes((0, 0))
_L_F1 = bytes((0, 1))
_L_E2 = bytes((1, 0))
_L_F2 = bytes((1, 1))
_LABEL_LEN = len(_L_E1)
_ROW_OFF = _LABEL_LEN  # byte offset of the line index in the message
_ROW_W, _ROW_SH = _ROW_OFF // 4, (_ROW_OFF % 4) * 8


def _noise_base_words(label: bytes) -> tuple[int, ...]:
    """The 16 message words of a noise-line draw with the line index zeroed."""
    if len(label) != _LABEL_LEN:
        raise ValueError(f"noise line address must be {_LABEL_LEN} bytes, got {len(label)}")
    material = (label + struct.pack("<I", 0)).ljust(64, b"\x00")
    return struct.unpack("<16I", material)


def _rndb(x):
    """Round fp32 through BF16 (RTNE) and widen back to fp32."""
    return cutlass.Float32(x).to(cutlass.BFloat16).to(cutlass.Float32)


def _e4m3x2_to_f16x2(v):
    """Packed exact widening of two e4m3 codes to f16x2 (register-direct peel operands)."""
    return _asm_u32("cvt.rn.f16x2.e4m3x2 $0, $1;", "=r,h", cutlass.Uint16(v))


def _f32x2_to_bf16x2(hi, lo):
    """Pack {hi, lo} fp32 into a bf16x2 register (two RTNE roundings)."""
    return _asm_u32(
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        "=r,f,f",
        cutlass.Float32(hi),
        cutlass.Float32(lo),
    )


def _f32_hex(x: float) -> str:
    return f"0f{struct.unpack('<I', struct.pack('<f', x))[0]:08X}"


def _mul_f32(a, b):
    """One RNE fp32 multiply (never contracted, subnormals kept)."""
    return _asm_f32("mul.rn.f32 $0, $1, $2;", "=f,f,f", cutlass.Float32(a), cutlass.Float32(b))


def _bf16_bits_to_f32(bits):
    """Exact widening of the BF16 in the low half of ``bits``."""
    return _asm_f32(
        "{\n\t.reg .b32 t;\n\tshl.b32 t, $1, 16;\n\tmov.b32 $0, t;\n\t}",
        "=f,r",
        cutlass.Uint32(bits),
    )


def _fold(coefficient, beta, codes, byte: int, noise):
    """``fma(beta, noise, coefficient * q)`` for int8 code ``byte`` of ``codes``,
    each op one RNE fp32 rounding (the verifier's ``fp32_fma`` is the correctly
    rounded FMA)."""
    return _asm_f32(
        "{\n\t"
        ".reg .b32 t;\n\t"
        ".reg .f32 q, s;\n\t"
        f"bfe.s32 t, $3, {8 * byte}, 8;\n\t"
        "cvt.rn.f32.s32 q, t;\n\t"
        "mul.rn.f32 s, $1, q;\n\t"
        "fma.rn.f32 $0, $2, $4, s;\n\t"
        "}",
        "=f,f,f,r,f",
        cutlass.Float32(coefficient),
        cutlass.Float32(beta),
        cutlass.Uint32(codes),
        cutlass.Float32(noise),
    )


def _f32x2_to_e4m3x2(hi, lo) -> cutlass.Uint32:
    """e4m3 codes of two fp32 values (RNE, saturating to +-448), ``lo`` in the low byte."""
    return _asm_u32(
        "{\n\t.reg .b16 h;\n\tcvt.rn.satfinite.e4m3x2.f32 h, $1, $2;\n\tcvt.u32.u16 $0, h;\n\t}",
        "=r,f,f",
        cutlass.Float32(hi),
        cutlass.Float32(lo),
    )


def _strip_state(alpha, l2):
    """A row's strip state from its scaled RMS ``s = alpha * l2``, per
    byte lane: ``0x7F + |e4m3(g * s)|`` in the low half (a code magnitude
    below it borrows its byte's top bit into place) and ``safe exponent | 1``
    in code position in the high half."""
    return _asm_u32(
        "{\n\t"
        ".reg .f32 s, g, w;\n\t"
        ".reg .b32 b, e, m;\n\t"
        ".reg .b16 h;\n\t"
        ".reg .pred p;\n\t"
        "mul.rn.f32 s, $1, $2;\n\t"
        f"setp.lt.f32 p, s, {_f32_hex(STRIP_SMALL_RMS)};\n\t"
        f"selp.f32 g, {_f32_hex(STRIP_G_SMALL_RMS)}, {_f32_hex(STRIP_G)}, p;\n\t"
        "mul.rn.f32 w, g, s;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h, w, w;\n\t"
        "cvt.u32.u16 m, h;\n\t"
        "and.b32 m, m, 0x7F;\n\t"
        "add.u32 m, m, 0x7F;\n\t"
        "mov.b32 b, s;\n\t"
        "bfe.u32 e, b, 23, 8;\n\t"
        f"sub.u32 e, e, {127 + STRIP_EXP_OFFSET - _E4M3_BIAS};\n\t"
        "and.b32 e, e, 0xF;\n\t"
        "shl.b32 e, e, 3;\n\t"
        "or.b32 e, e, 1;\n\t"
        "shl.b32 e, e, 16;\n\t"
        "or.b32 m, m, e;\n\t"
        "mul.lo.u32 $0, m, 0x0101;\n\t"
        "}",
        "=r,f,f",
        cutlass.Float32(alpha),
        cutlass.Float32(l2),
    )


def _coin_mask(mixed):
    """``0x8080`` if the group-start mix's coin field passes, else 0."""
    return _asm_u32(
        "{\n\t"
        ".reg .b32 t;\n\t"
        ".reg .pred p;\n\t"
        "mov.b32 t, $1;\n\t"
        f"bfe.u32 t, t, {STRIP_COIN_SHIFT}, 7;\n\t"
        f"setp.lt.u32 p, t, {STRIP_COIN_THRESHOLD};\n\t"
        "selp.b32 $0, 0x8080, 0, p;\n\t"
        "}",
        "=r,f",
        cutlass.Float32(mixed),
    )


def _rewrite_strip_pair(codes, strip_bound, strip_code, coin_mask):
    """Rewrite each of the two e4m3 ``codes`` whose magnitude is below the
    row's strip bound (when ``coin_mask`` is set) to ``sign | safe exponent |
    mantissa | 1``; ``strip_bound`` / ``strip_code`` are the halves of
    ``_strip_state``."""
    return _asm_u32(
        "{\n\t"
        ".reg .b32 m, t, k, d;\n\t"
        "and.b32 m, $1, 0x7F7F;\n\t"
        "sub.u32 t, $2, m;\n\t"
        "and.b32 t, t, $4;\n\t"
        "prmt.b32 k, t, t, 0x0098;\n\t"
        "and.b32 d, $1, 0x8787;\n\t"
        "or.b32 d, d, $3;\n\t"
        "lop3.b32 $0, d, $1, k, 0xE4;\n\t"
        "}",
        "=r,r,r,r,r",
        cutlass.Uint32(codes),
        cutlass.Uint32(strip_bound),
        cutlass.Uint32(strip_code),
        cutlass.Uint32(coin_mask),
    )


def _l2_grid_round_f32(x):
    """Round non-negative fp32 through BF16 to the nearest 4-bit grid step."""
    return _asm_f32(
        "{\n\t"
        ".reg .b16 h;\n\t"
        ".reg .b32 t;\n\t"
        "cvt.rn.bf16.f32 h, $1;\n\t"
        "cvt.u32.u16 t, h;\n\t"
        "add.u32 t, t, 2;\n\t"
        "and.b32 t, t, 0xFFFFFFFC;\n\t"
        "shl.b32 t, t, 16;\n\t"
        "mov.b32 $0, t;\n\t"
        "}",
        "=f,f",
        cutlass.Float32(x),
    )


# ---- shared reference-chain helpers (traced inline by both kernels) ----


def _scale_chain(ssq, amax, kf, delta_r, delta_over_std):
    """Derive BF16-valued ``(alpha, beta, l2)`` from row sumsq and absmax.

    Mirrors the reference chain: ``row_norms`` floors both norms at
    ``2^-32`` (the zero-row guard), then ``derive_row_scales`` computes
    ``alpha = QUANT_MAX / fma(delta_r, l2, linf)`` and
    ``beta = (alpha * l2) * delta_over_std`` (no floor on the scaled
    signal).
    """
    mean = cutlass.Float32(ssq) / kf
    l2 = cute.arch.fmax(
        _l2_grid_round_f32(cutlass.Float32(cute.math.sqrt(mean))),
        cutlass.Float32(_NORM_FLOOR),
    )
    # The committed FP10 norm rounds the maximum product to BF16 before the
    # scale-chain FMA. Round here as well when stats came from another producer.
    linf = cute.arch.fmax(_rndb(cutlass.Float32(amax)), cutlass.Float32(_NORM_FLOOR))
    nb = _rndb(cutlass.Float32(delta_r) * l2 + linf)
    alpha = _rndb(cutlass.Float32(MAX_E4M3) / nb)
    beta = _rndb(_rndb(alpha * l2) * cutlass.Float32(delta_over_std))
    return alpha, beta, l2


def _ge0_i32(x):
    """1 if the Int32 ``x`` >= 0 else 0 (branchless: helpers are traced)."""
    return cutlass.Int32(1) - cutlass.Int32(cutlass.Uint32(x) >> 31)


def _generate_noise_line(key, index_u32, msg_base, output):
    """Generate one normalized BF16-valued noise line from keyed BLAKE3 bytes.

    ``msg_base`` (from ``_noise_base_words``) carries the label and instance
    index; ``index_u32`` is the line's global index (an E row or an F column).
    """
    cv = [key[i] for i in range(8)]
    msg = [cutlass.Uint32(w) for w in msg_base]
    msg[_ROW_W] = msg[_ROW_W] + (index_u32 << _ROW_SH)
    if _ROW_SH:
        msg[_ROW_W + 1] = msg[_ROW_W + 1] + (index_u32 >> (32 - _ROW_SH))
    w = compress_rolled(cv, msg, 64, SINGLE_BLOCK_KEYED_FLAGS)
    xs = []
    ssq = cutlass.Int32(0)
    for i in range(R):  # plain range: helpers are traced, not preprocessed
        b = cutlass.Int32((w[i // 4] >> (8 * (i % 4))) & 0xFF)
        mag = (b & 0x7F) + 1
        xs.append((1 - 2 * (b >> 7)) * mag)
        ssq = ssq + mag * mag
    v = ssq * _LINE_SQRT_SQ  # <= 16 * 128^2 * 1024 = 2^28
    c = cutlass.Float32(cute.math.sqrt(cutlass.Float32(v))).to(cutlass.Int32)
    c = c + _ge0_i32(v - (c + 1) * (c + 1))
    c = c + _ge0_i32(v - (c + 1) * (c + 1))
    c = c - _ge0_i32(c * c - v - 1)
    c = c - _ge0_i32(c * c - v - 1)
    scale = _rndb(cutlass.Float32(_LINE_NORM_NUM) / _rndb(c.to(cutlass.Float32)))
    for i in range(R):
        output[i] = _rndb(xs[i].to(cutlass.Float32) * scale)
