"""Packed rounding, scale-chain, and E1 device helpers."""

import struct

import cutlass
import cutlass.cute as cute

from .._utils._stats_ops import _asm_f32, _asm_u32
from ..protocol_constants import _INT_SQRT_PREC, NOISE_TARGET_NORM, QUANT_MAX, R
from ..tensor_hash_plus_stats._blake3_ops import (
    SINGLE_BLOCK_KEYED_FLAGS,
    compress_rolled,
)

MAX_E4M3 = QUANT_MAX  # 448.0, the e4m3 quant-grid ceiling
_NORM_FLOOR = 2.0**-32  # row_norms' (near-)zero-row floor on l2 and linf
# Noise-line normalization constants (Noiser._lines): the target norm times the
# fixed-point isqrt factor, and the squared factor applied under the isqrt.
_LINE_NORM_NUM = float(NOISE_TARGET_NORM * _INT_SQRT_PREC)  # 8192, exact bf16
_LINE_SQRT_SQ = _INT_SQRT_PREC * _INT_SQRT_PREC  # 1024

# The v4 noise-line address prefix ``side(1) | factor(1)`` (``miner_base.noise``:
# ``Side.A = 0, Side.B = 1``; ``Factor.E = 0, Factor.F = 1``). The
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
    """The 16 message words of ``OperandNoiser._line_digest`` with the line index zeroed."""
    if len(label) != _LABEL_LEN:
        raise ValueError(f"noise line address must be {_LABEL_LEN} bytes, got {len(label)}")
    material = (label + struct.pack("<I", 0)).ljust(64, b"\x00")
    return struct.unpack("<16I", material)


def _rndb(x):
    """Round fp32 through BF16 (RTNE) and widen back to fp32."""
    return cutlass.Float32(x).to(cutlass.BFloat16).to(cutlass.Float32)


# Packed instructions preserve the reference's fp32-op-then-BF16-RTNE contract.


def _mul_bf16x2(a, b):
    """Two reference BF16 muls: bf16x2(a) * bf16x2(b), RTNE."""
    return _asm_u32("mul.rn.bf16x2 $0, $1, $2;", "=r,r,r", cutlass.Uint32(a), cutlass.Uint32(b))


def _fma_bf16x2(a, b, c):
    """Two reference BF16 fmas: a*b + c per lane with a SINGLE RTNE rounding.

    Exactly ``ComputeOps.fma``'s hardware-FMA contract (the product is exact,
    only the sum rounds once) -- the reference's fp64 TwoSum + round-to-odd
    emulation is defined to match this native instruction.
    """
    return _asm_u32(
        "fma.rn.bf16x2 $0, $1, $2, $3;",
        "=r,r,r,r",
        cutlass.Uint32(a),
        cutlass.Uint32(b),
        cutlass.Uint32(c),
    )


def _f32x2_to_bf16x2(hi, lo):
    """Pack {hi, lo} fp32 into a bf16x2 register (two RTNE roundings)."""
    return _asm_u32(
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        "=r,f,f",
        cutlass.Float32(hi),
        cutlass.Float32(lo),
    )


def _splat_bf16x2(s):
    """Broadcast a bf16 in the low half of ``s`` into both bf16x2 lanes."""
    return _asm_u32("prmt.b32 $0, $1, $1, 0x1010;", "=r,r", cutlass.Uint32(s))


def _open_codes_bf16x2(word, scale2, byte_selector):
    """Open one int8 code pair of ``word`` into bf16x2: ``bf16(q * s)`` per lane.

    ``byte_selector`` is the trace-time prmt selector choosing the pair's
    bytes (0x4140: low u16 pair, 0x4342: high pair of a raw ldmatrix word);
    ``scale2`` is the pair's group scale splatted into both lanes. Bit-exact
    against ``PrequantMatrix.open()``'s ``bf16(f32(q)*f32(s))``, verified
    exhaustively.
    """
    # The exponent byte 0x43 rides in a register and the selector is the
    # immediate: with the roles reversed, ptxas rematerializes the selector
    # before every call.
    return _asm_u32(
        "{\n\t"
        ".reg .b32 t, mg, sb, v;\n\t"
        f"prmt.b32 t, $1, $3, 0x{byte_selector:04X};\n\t"
        "and.b32 mg, t, 0xFF7FFF7F;\n\t"
        "and.b32 sb, t, 0xFF80FF80;\n\t"
        "sub.rn.bf16x2 v, mg, sb;\n\t"
        "mul.rn.bf16x2 $0, v, $2;\n\t"
        "}",
        "=r,r,r,r",
        cutlass.Uint32(word),
        cutlass.Uint32(scale2),
        cutlass.Uint32(0x43434343),
    )


def _open_int8x2_bf16x2(codes, scale2):
    """Open the LOW u16 code pair of ``codes`` (see ``_open_codes_bf16x2``)."""
    return _open_codes_bf16x2(codes, scale2, 0x4140)


def _open_int8x2_bf16x2_hi(codes_word, scale2):
    """Open the HIGH u16 pair of a raw ldmatrix word, skipping the u16
    extraction (see ``_open_codes_bf16x2``)."""
    return _open_codes_bf16x2(codes_word, scale2, 0x4342)


def _bf16x2_to_e4m3x2(v) -> cutlass.Uint16:
    """Two reference quant steps: e4m3(clamp(x, +-448)) per bf16 lane."""
    from cutlass._mlir import ir as _ir
    from cutlass._mlir.dialects import llvm as _llvm

    res = _llvm.inline_asm(
        _ir.IntegerType.get_signless(16),
        [cutlass.Uint32(v).ir_value()],
        "{\n\t"
        ".reg .b32 t;\n\t"
        ".reg .f32 vlo, vhi;\n\t"
        "shl.b32 t, $1, 16;\n\t"
        "mov.b32 vlo, t;\n\t"
        "and.b32 t, $1, 0xffff0000;\n\t"
        "mov.b32 vhi, t;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 $0, vhi, vlo;\n\t"
        "}",
        "=h,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Uint16(res)


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
    """Derive BF16-valued ``(alpha, beta)`` from row sumsq and absmax.

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
    # amax is an exact bf16 value (no eps bump); 2^-32 is bf16-exact too.
    linf = cute.arch.fmax(cutlass.Float32(amax), cutlass.Float32(_NORM_FLOOR))
    nb = _rndb(cutlass.Float32(delta_r) * l2 + linf)
    alpha = _rndb(cutlass.Float32(MAX_E4M3) / nb)
    beta = _rndb(_rndb(alpha * l2) * cutlass.Float32(delta_over_std))
    return alpha, beta


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
