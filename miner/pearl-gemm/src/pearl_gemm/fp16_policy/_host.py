"""Host launch for the bit-exact A100 (``sm_80``) FP16 accumulation policy.

Reproduces the FP16 scheme's "unpredictable accumulation steps" jackpot policy
(``zk-pow/src/api/fp16/policy.rs`` :: ``replay_and_evaluate`` / ``evaluate``) on
real GA100 silicon, bit-for-bit against the verifier. The kernel replays the
``m x n`` tile with the *same* bit-exact A100 accumulation as
:func:`pearl_gemm.fp16_gemm.fp16_gemm_a100` (``zk-pow/src/api/fp16/accumulate.rs``
:: ``a100_dot``) but in software, which is what lets it emit the per-group
census the native ``HMMA`` instruction hides: per group, whether the step is a
*breakpoint* (the accumulator alignment or the final FP32 round-toward-zero
discarded a nonzero bit) and how many products truncated.

The kernel fuses the per-cell reduction of ``evaluate`` in-device, emitting per
cell the three integers it consumes -- ``n_bp`` (breakpoint steps), ``n_runs``
(maximal runs of non-breakpoint steps; empty no-op groups extend the surrounding
run, exactly as ``PolicyStep::default()`` does) and ``n_pt`` (products truncated
outside breakpoints) -- plus the recomputed FP32 tile. The host folds those
per-cell integers into the tile totals ``breakpoints`` and ``numerator`` and the
f64 ``f_bp`` / ``rho`` / ``accept``, matching ``evaluate`` bit-for-bit.

The kernel is a small hand-written CUDA extension compiled with
``nvcc -arch=sm_80`` (loaded through ``torch.utils.cpp_extension``), exactly like
``fp16_gemm`` / ``fp16_noisy_quant`` / ``fp16_noise_lines``. ``fp16_gemm`` is left
untouched; this is a standalone, census-capable module.
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass

import torch

from .._utils._arch import Arch, require_arch

_SUPPORTED_ARCHS = (Arch.SM80,)

# Mirrors zk-pow/src/api/fp16/{accumulate,policy}.rs.
GROUP = 8
NOISE_RANK = 32
MIN_FBP = 0.30
MIN_RHO = 1.2


@dataclass
class PolicyReport:
    """The outcome of the policy over one opened tile (mirrors the Rust struct).

    ``f_bp`` and ``rho`` are Python floats (IEEE f64, bit-identical to the Rust
    ``f64`` divisions); ``breakpoints`` and ``numerator`` are the exact integer
    tile totals; ``accept`` is ``f_bp >= 0.30 and rho >= 1.2``.
    """

    f_bp: float
    rho: float
    accept: bool
    breakpoints: int
    numerator: int


@functools.lru_cache(maxsize=1)
def _extension():
    """Compile (once) and return the loaded ``sm_80`` CUDA extension."""
    from torch.utils.cpp_extension import load

    src = os.path.join(os.path.dirname(__file__), "_kernel_sm80.cu")
    return load(
        name="pearl_fp16_policy_sm80",
        sources=[src],
        extra_cuda_cflags=["-arch=sm_80", "-O3"],
        verbose=False,
    )


def policy_census(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw per-cell census for the ``m x n`` tile ``a @ b.T``.

    ``a`` is ``(m, k)`` float16 and ``b`` is ``(n, k)`` float16 (the transposed
    logical right operand, as in :func:`pearl_gemm.fp16_gemm.fp16_gemm_a100`).
    Returns ``(tile, n_bp, n_runs, n_pt)``: the ``(m, n)`` float32 recomputed
    tile and three ``(m, n)`` int64 per-cell census tensors.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("a and b must be float16")
    m, k = a.shape
    n, bk = b.shape
    if bk != k:
        raise ValueError(f"b must have k={k}, got {bk}")
    require_arch("fp16_policy_census", a.device, *_SUPPORTED_ARCHS)
    tile, n_bp, n_runs, n_pt = _extension().fp16_policy_census(a.contiguous(), b.contiguous())
    return tile, n_bp, n_runs, n_pt


def evaluate(
    n_bp: torch.Tensor, n_runs: torch.Tensor, n_pt: torch.Tensor, k: int
) -> PolicyReport:
    """Fold the per-cell census into the tile :class:`PolicyReport`.

    Bit-for-bit mirror of ``policy.rs`` :: ``evaluate``:
    ``numerator = sum_cells[ GROUP*n_bp + NOISE_RANK*n_runs + n_pt ]``,
    ``breakpoints = sum_cells n_bp``,
    ``f_bp = breakpoints / (cells * ceil(k/GROUP))``,
    ``rho = numerator / (cells * k)`` (both f64),
    ``accept = f_bp >= MIN_FBP and rho >= MIN_RHO``.
    """
    cells = n_bp.numel()
    if cells == 0 or k <= 0:
        raise ValueError("empty tile")
    # Exact integer totals (Python ints are unbounded).
    breakpoints = int(n_bp.sum().item())
    numerator = int(
        (GROUP * n_bp.to(torch.int64) + NOISE_RANK * n_runs.to(torch.int64) + n_pt.to(torch.int64))
        .sum()
        .item()
    )
    steps_per_cell = math.ceil(k / GROUP)
    f_bp = breakpoints / float(cells * steps_per_cell)
    rho = numerator / float(cells * k)
    return PolicyReport(
        f_bp=f_bp,
        rho=rho,
        accept=f_bp >= MIN_FBP and rho >= MIN_RHO,
        breakpoints=breakpoints,
        numerator=numerator,
    )


def replay_and_evaluate(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, PolicyReport]:
    """A100 ``sm_80`` replay + policy evaluation of the ``m x n`` tile ``a @ b.T``.

    Convenience wrapper over :func:`policy_census` + :func:`evaluate`, the sm_80
    analogue of ``policy.rs`` :: ``replay_and_evaluate``. ``a`` is ``(m, k)`` and
    ``b`` is ``(n, k)`` float16. Returns the ``(m, n)`` float32 tile (bit-exact
    to the verifier's replay) and the :class:`PolicyReport`.
    """
    k = int(a.shape[1])
    tile, n_bp, n_runs, n_pt = policy_census(a, b)
    report = evaluate(n_bp, n_runs, n_pt, k)
    return tile, report
