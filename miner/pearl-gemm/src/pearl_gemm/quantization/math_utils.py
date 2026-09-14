import cutlass
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

TORCH_TO_CUTLASS = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}


@dsl_user_op
def rint_f32(v: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    """Round *v* to the nearest integer (``cvt.rni.f32.f32``)."""
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.cutlass_dsl.T.f32(),
            [v.ir_value()],
            "cvt.rni.f32.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fmin_f32(a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    """Return the minimum of *a* and *b* (``min.f32``).

    Complements :func:`cutlass.cute.arch.fmax`.
    """
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.cutlass_dsl.T.f32(),
            [a.ir_value(), b.ir_value()],
            "min.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
