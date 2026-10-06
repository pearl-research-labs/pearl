"""Host launch for the bit-exact A100 (``sm_80``) FP16 fused noisy quantization.

Reproduces ``zk-pow/src/api/fp16/quantization.rs``'s ``noisy_quantize`` on real
GA100 silicon, bit-for-bit against the verifier. The pipeline is:

1. ``N = E @ F^T`` on the committed A100 FP16 datapath -- the same
   :func:`pearl_gemm.fp16_gemm.fp16_gemm_a100` kernel the verifier replays. With
   ``e`` ``(num_rows, r)`` and ``f`` ``(k, r)`` (both fp16), ``fp16_gemm_a100(e,
   f)`` returns the ``(num_rows, k)`` f32 tile ``N[i, j] = sum_t e[i,t] f[j,t]``,
   matching ``a100_matmul(e, f, None, num_rows, k, r)`` exactly.
2. Per-row ``row_norms`` (sequential-f32 ``sumsq``, grid-rounded BF16 ``l2``,
   BF16 ``linf``), floored to ``2^-32``, then ``derive_row_scales`` in BF16.
3. The fused elementwise ``noised = fma_f32(alpha*x, beta*N)`` -> clamp +-65504
   -> RNE-to-FP16.

The two bespoke passes are a small hand-written CUDA extension compiled with
``nvcc -arch=sm_80`` (loaded through ``torch.utils.cpp_extension``), exactly like
``fp16_gemm``. All BF16/FP16 rounding is replicated bit-for-bit from the Rust
helpers; see ``_kernel_sm80.cu``.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import torch

from .._utils._arch import Arch, require_arch
from ..fp16_gemm import fp16_gemm_a100

_SUPPORTED_ARCHS = (Arch.SM80,)


@functools.lru_cache(maxsize=1)
def _extension():
    """Compile (once) and return the loaded ``sm_80`` CUDA extension."""
    from torch.utils.cpp_extension import load

    src = os.path.join(os.path.dirname(__file__), "_kernel_sm80.cu")
    return load(
        name="pearl_fp16_noisy_quant_sm80",
        sources=[src],
        extra_cuda_cflags=["-arch=sm_80", "-O3"],
        verbose=False,
    )


@dataclass
class BuiltRows16:
    """The rebuilt FP16 operand plus per-row scales (mirrors the Rust struct).

    ``noised_part`` is ``(num_rows, k)`` float16; ``alpha``, ``beta``, ``l2`` are
    ``(num_rows,)`` int16 tensors carrying BF16 bit patterns (view as uint16).
    """

    noised_part: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    l2: torch.Tensor


def row_scales(rows: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row ``(alpha, beta, l2)`` BF16 bit patterns for an ``(num_rows, k)``
    FP16 operand, bit-exact to ``row_norms`` + flooring + ``derive_row_scales``.

    Returns three ``(num_rows,)`` int16 tensors (BF16 bits; ``.view(torch.uint16)``
    or ``numpy.view(uint16)`` to read the patterns).
    """
    if rows.ndim != 2 or rows.dtype != torch.float16:
        raise ValueError("rows must be a 2D float16 tensor")
    require_arch("fp16_row_scales", rows.device, *_SUPPORTED_ARCHS)
    alpha, beta, l2 = _extension().fp16_row_scales(rows.contiguous(), int(r))
    return alpha, beta, l2


def noised_elementwise(
    rows: torch.Tensor, noise: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Fused ``RNE_fp16(clamp(fma_f32(alpha*rows, beta*noise)))``.

    ``noise`` is the ``(num_rows, k)`` f32 ``E@F^T`` tile; ``alpha``/``beta`` are
    the per-row int16 BF16 scales from :func:`row_scales`. Returns ``(num_rows,
    k)`` float16.
    """
    require_arch("fp16_noised_elementwise", rows.device, *_SUPPORTED_ARCHS)
    return _extension().fp16_noised_elementwise(
        rows.contiguous(), noise.contiguous(), alpha, beta
    )


def noisy_quantize(
    rows: torch.Tensor, e: torch.Tensor, f: torch.Tensor, r: int
) -> BuiltRows16:
    """Fused per-row noisy quantization of an FP16 operand (sm_80).

    ``rows`` is ``(num_rows, k)`` float16; ``e`` is ``(num_rows, r)`` and ``f`` is
    ``(k, r)`` float16 noise lines, so ``N = E @ F^T`` is ``(num_rows, k)`` -- the
    noise is built on the committed ``fp16_gemm_a100`` datapath here, exactly as
    the reference's ``a100_matmul(e, f, None, num_rows, k, r)``. Returns the
    noised FP16 operand and the per-row BF16 scales as a :class:`BuiltRows16`.

    Note the ``fp16_gemm_a100`` tile granularity applies to the noise matmul:
    ``num_rows % 16 == 0``, ``k % 8 == 0``, ``r % 16 == 0``.
    """
    if rows.ndim != 2 or rows.dtype != torch.float16:
        raise ValueError("rows must be a 2D float16 tensor")
    if e.ndim != 2 or f.ndim != 2 or e.dtype != torch.float16 or f.dtype != torch.float16:
        raise ValueError("e and f must be 2D float16 tensors")
    num_rows, k = rows.shape
    if e.shape[0] != num_rows or e.shape[1] != r:
        raise ValueError(f"e must be (num_rows={num_rows}, r={r}), got {tuple(e.shape)}")
    if f.shape[0] != k or f.shape[1] != r:
        raise ValueError(f"f must be (k={k}, r={r}), got {tuple(f.shape)}")
    require_arch("fp16_noisy_quantize", rows.device, *_SUPPORTED_ARCHS)

    # N = E @ F^T on the committed A100 FP16 datapath (bit-exact to a100_matmul).
    noise = fp16_gemm_a100(e.contiguous(), f.contiguous())  # (num_rows, k) f32

    alpha, beta, l2 = row_scales(rows, r)
    noised_part = noised_elementwise(rows, noise, alpha, beta)
    return BuiltRows16(noised_part=noised_part, alpha=alpha, beta=beta, l2=l2)
