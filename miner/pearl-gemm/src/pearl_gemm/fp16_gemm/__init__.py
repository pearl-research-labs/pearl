"""Bit-exact A100 (``sm_80``) FP16 -> FP32 GEMM for the FP16 proof-of-useful-work
scheme.

The single public entry, :func:`fp16_gemm_a100`, computes the FP16 GEMM tile the
scheme's verifier replays, bit-for-bit, on real GA100 tensor cores. See
``docs/sm80_feasibility.md`` for the feasibility verdict and backend choice.
"""

from ._host import fp16_gemm_a100

__all__ = ["fp16_gemm_a100"]
