"""Bit-exact A100 (``sm_80``) FP16 fused noisy quantization for the FP16
proof-of-useful-work scheme.

Reproduces ``zk-pow/src/api/fp16/quantization.rs``'s ``noisy_quantize``
bit-for-bit on real GA100 tensor cores: per-row norms + BF16 scale derivation and
the fused ``alpha*X + beta*(E@F^T)`` noised quantize, with the noise matmul on the
committed :func:`pearl_gemm.fp16_gemm.fp16_gemm_a100` datapath.
"""

from ._host import (
    BuiltRows16,
    noised_elementwise,
    noisy_quantize,
    row_scales,
)

__all__ = ["BuiltRows16", "noised_elementwise", "noisy_quantize", "row_scales"]
