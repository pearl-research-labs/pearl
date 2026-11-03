"""Host launch for the bit-exact A100 (``sm_80``) FP16 -> FP32 GEMM tile.

This is the datapath the FP16 proof-of-useful-work scheme's verifier replays
(``zk-pow/src/api/fp16/accumulate.rs``). A plain ``sm_80`` FP16 tensor-core GEMM
with FP32 accumulation and a pinned ascending-k reduction (no split-k, no
atomics) reproduces that model bit-for-bit on real GA100 silicon, because the
model was measured from this exact ``HMMA.16816.F32`` datapath.

The kernel is a small hand-written CUDA extension compiled with
``nvcc -arch=sm_80`` (loaded through ``torch.utils.cpp_extension``). The CuTe
path was not used: the repo's ``nvidia-cutlass-dsl`` is pinned ``>=4.6.0`` but
4.5.0 is installed, and the existing warp-mma kernel (``_kernel_sm120.py``) is
specialized for the FP8 lottery datapath. See ``docs/sm80_feasibility.md``.
"""

from __future__ import annotations

import functools
import os

import torch

from .._utils._arch import Arch, require_arch

_SUPPORTED_ARCHS = (Arch.SM80,)


@functools.lru_cache(maxsize=1)
def _extension():
    """Compile (once) and return the loaded ``sm_80`` CUDA extension."""
    from torch.utils.cpp_extension import load

    src = os.path.join(os.path.dirname(__file__), "_kernel_sm80.cu")
    return load(
        name="pearl_fp16_gemm_sm80",
        sources=[src],
        extra_cuda_cflags=["-arch=sm_80", "-O3"],
        verbose=False,
    )


def fp16_gemm_a100(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    acc: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """The A100 ``sm_80`` bit-exact FP16 GEMM tile ``D = a @ b.T (+ acc)``.

    ``a`` is ``(m, k)`` float16 and ``b`` is ``(n, k)`` float16 -- the transposed
    logical right operand, so ``D[i, j]`` accumulates ``sum_u a[i, u] * b[j, u]``,
    exactly as ``a100_matmul``'s ``b`` is laid out. ``acc`` is an optional
    ``(m, n)`` float32 carry-in (``+0`` when omitted). Returns the ``(m, n)``
    float32 output tile; every element matches the verifier's replay of the A100
    ``HMMA.16816.F32`` accumulation bit-for-bit.

    Requires ``k % 16 == 0``, ``m % 16 == 0``, ``n % 8 == 0`` (the native
    ``mma.sync.m16n8k16`` tile granularity). The reduction order over ``k`` is
    pinned (ascending, FP32 accumulator chained across the whole axis; no
    split-k, no atomics), which is what makes the result deterministic and
    bit-exact against the verifier.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("a and b must be float16")
    m, k = a.shape
    n, bk = b.shape
    if bk != k:
        raise ValueError(f"b must have k={k}, got {bk}")
    require_arch("fp16_gemm_a100", a.device, *_SUPPORTED_ARCHS)
    a = a.contiguous()
    b = b.contiguous()
    if acc is not None:
        if acc.shape != (m, n) or acc.dtype != torch.float32:
            raise ValueError("acc must be (m, n) float32")
        acc = acc.contiguous()
    d = _extension().fp16_gemm_a100(a, b, acc)
    if out is not None:
        if out.shape != (m, n) or out.dtype != torch.float32:
            raise ValueError("out must be (m, n) float32")
        out.copy_(d)
        return out
    return d
