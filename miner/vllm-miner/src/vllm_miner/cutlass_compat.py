"""Restore a CuTe alias still used by locked wheel-bundled kernels.

vLLM 0.28.0 and the pearl_gemm CuTe kernels both resolve
``nvidia-cutlass-dsl==4.6.2``, which removed the deprecated
``cute.make_fragment`` alias. The vLLM wheel's bundled flash-attn CuTe
helpers and FlashInfer's block-scaled GEMM sources still called it as of
FlashInfer 0.6.14. Restore that shared process-level alias before either
kernel path is used. The marker makes repeated runtime
initialization idempotent, and CUTLASS versions that still provide the alias
remain unchanged. Remove this shim only after both locked wheels drop their
references; the 0.28 bump moved both, so that is worth re-checking.
"""

from miner_utils import get_logger

_LOGGER = get_logger("vllm_miner.cutlass_compat")

_COMPAT_APPLIED_ATTR = "_vllm_miner_cutlass46_compat_applied"


def apply_cutlass_46_compat() -> None:
    """Restore ``cute.make_fragment`` when CUTLASS no longer provides it."""
    try:
        import cutlass.cute as cute
    except ImportError:
        # No cutlass-dsl in this process (e.g. non-CUDA); nothing to patch.
        return

    if getattr(cute, _COMPAT_APPLIED_ATTR, False):
        return

    # ``cute.make_fragment`` -> ``cute.make_rmem_tensor`` (NFC rename in 4.6).
    if not hasattr(cute, "make_fragment") and hasattr(cute, "make_rmem_tensor"):
        cute.make_fragment = cute.make_rmem_tensor
        _LOGGER.info("Applied cutlass-dsl 4.6 alias: cute.make_fragment")

    setattr(cute, _COMPAT_APPLIED_ATTR, True)
