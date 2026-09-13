"""Packed per-element row-stats primitives shared by the noising and
pre-quantization kernels.

The row stats are fp32 ``sumsq`` and ``absmax`` over BF16 elements consumed
as packed ``bf16x2`` u32 words. Inline PTX keeps those packed operations
explicit and is bit-exact against the reference chain:

* squaring a bf16 value is exact in fp32 (8+8 mantissa bits), so the fma
  chain adds exact squares to the fp32 accumulator;
* abs/max never round.

The fp32 accumulation ORDER is whatever the calling kernel fixes; the
protocol's ``_round_l2_to_grid`` absorbs the ulp-level difference vs torch's
pairwise sum (bit-exact on alpha/beta across several different fixed orders
-- re-verify against the reference whenever a new order is introduced).
"""

import cutlass


def _asm_u32(asm: str, cons: str, *args) -> cutlass.Uint32:
    from cutlass._mlir import ir as _ir
    from cutlass._mlir.dialects import llvm as _llvm

    res = _llvm.inline_asm(
        _ir.IntegerType.get_signless(32),
        [a.ir_value() for a in args],
        asm,
        cons,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Uint32(res)


def _asm_f32(asm: str, cons: str, *args) -> cutlass.Float32:
    from cutlass._mlir import ir as _ir
    from cutlass._mlir.dialects import llvm as _llvm

    res = _llvm.inline_asm(
        _ir.F32Type.get(),
        [a.ir_value() for a in args],
        asm,
        cons,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Float32(res)


def abs_max_bf16x2(acc, v):
    """acc = max(acc, |v|) per bf16 lane. abs/max never round -> exact."""
    return _asm_u32(
        "{\n\t.reg .b32 t;\n\tand.b32 t, $2, 0x7FFF7FFF;\n\tmax.bf16x2 $0, $1, t;\n\t}",
        "=r,r,r",
        cutlass.Uint32(acc),
        cutlass.Uint32(v),
    )


def hmax_bf16x2_f32(v):
    """max of the two bf16 lanes, widened (exactly) to fp32."""
    return _asm_f32(
        "{\n\t"
        ".reg .b32 t;\n\t"
        ".reg .f32 a, b;\n\t"
        "shl.b32 t, $1, 16;\n\t"
        "mov.b32 a, t;\n\t"
        "and.b32 t, $1, 0xFFFF0000;\n\t"
        "mov.b32 b, t;\n\t"
        "max.f32 $0, a, b;\n\t"
        "}",
        "=f,r",
        cutlass.Uint32(v),
    )


def bf16x2_lo_f32(v):
    """Exactly widen the low bf16 lane to fp32."""
    return _asm_f32(
        "{\n\t.reg .b32 t;\n\tshl.b32 t, $1, 16;\n\tmov.b32 $0, t;\n\t}",
        "=f,r",
        cutlass.Uint32(v),
    )


def bf16x2_hi_f32(v):
    """Exactly widen the high bf16 lane to fp32."""
    return _asm_f32(
        "{\n\t.reg .b32 t;\n\tand.b32 t, $1, 0xFFFF0000;\n\tmov.b32 $0, t;\n\t}",
        "=f,r",
        cutlass.Uint32(v),
    )
