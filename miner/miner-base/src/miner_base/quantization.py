"""The FP8 quant scheme (per-row scaled noising + FP8 quantization).

A quantized operand is a plain ``torch.float8_e4m3fn`` tensor (no wrapper).
The scheme is *fused*: per-row scales ``alpha``, ``beta`` are derived from the
row's norms and ``alpha (.) X + beta (.) (E @ F)`` (``(.)`` = per-row
broadcast) is quantized to FP8 in one step. The same beta-scaled ``E``, ``F``
are reused for the peel columns, so the peel cancels exactly and the useful
product comes out scaled by ``outer(alpha_A, alpha_B)`` (undone in
reconstruction).
"""

from __future__ import annotations

import math

import torch

from .compute_ops import ComputeOps, DType
from .hardware import Hardware
from .noise import NOISE_TARGET_NORM
from .prequant import RowNorms

# The quant grid ceiling: largest finite magnitude of the QUANT dtype (448.0 for e4m3).
QUANT_MAX = float(torch.finfo(DType.QUANT.value).max)
DELTA = 0.5  # noise-to-signal ratio (in L2) of the injected E@F noise


class Fp8QuantScheme:
    """The v0 whitelisted scheme: per-row scaled noising + FP8 quantization."""

    def row_norms(
        self, compute: ComputeOps, row_norms: RowNorms
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Floor ``row_norms``'s ``(l2, linf)`` at ``2**-32`` (a zero row would
        otherwise divide by zero in `noisy_quantize`)."""
        floor = compute.const(2.0**-32)
        return compute.max(row_norms.l2, floor), compute.max(row_norms.linf, floor)

    def noisy_quantize(
        self,
        rows: torch.Tensor,
        e: torch.Tensor,
        f: torch.Tensor,
        hw: Hardware,
        row_norms: RowNorms,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce ``A'`` (or ``B'``) from clean ``rows`` + factors ``e``, ``f``.

        ``rows``: (n x k) BF16; ``e``: (n x r) FP8; ``f``: (r x k) FP8.
        ``row_norms`` is passed straight through to `row_norms` (the method;
        see there). Returns ``(a_prime, alpha, beta, l2)``: the quantized
        noised operand (FP8), the per-row scales (n x 1, BF16) used to build
        it, and the floored ``l2`` those scales were derived from --
        ``sigma_i = DELTA * alpha_i * l2_i`` is the jackpot policy's per-row
        noise std (``policy.py``).
        """
        assert e.dtype == DType.FACTOR.value and f.dtype == DType.FACTOR.value
        X = rows.to(torch.bfloat16)
        l2, linf = self.row_norms(hw.compute, row_norms)

        # Per-row scale derivation: pick alpha, beta so the quantized
        # alpha (.) X + beta (.) E@F satisfies, per row:
        #   (1) |alpha*X + beta*E@F| <= QUANT_MAX  -- no loss to saturation;
        #   (2) rms(beta*E@F) = DELTA * rms(alpha*X)  -- noise-to-signal (L2)
        #       is DELTA.
        # We have X's norms (l2 := rms(X), linf) but never measure E@F; its
        # construction bounds it instead: each e_row / f_col is a uniform draw
        # renormalized to L2 norm NOISE_TARGET_NORM (noise.py), so with
        # c := NOISE_TARGET_NORM^2 an E@F entry (= <e_row, f_col>) has
        #   |entry| <= c           -- Cauchy-Schwarz, DETERMINISTIC up to FP8
        #                             rounding of the lines (norm overshoot
        #                             <= ~2^-4 = e4m3 half-ulp per line, ~13%
        #                             on the bound; the clamp below covers it)
        #   rms(entry) ~ c/sqrt(r) -- IN EXPECTATION (random ~unit directions:
        #                             E[<u,v>^2] = 1/r), so (2) is approximate.
        # (2) => beta = alpha * DELTA*l2 / (c/sqrt(r)) = alpha*l2*delta_over_std;
        # into (1) at the peaks (alpha*linf + beta*c = QUANT_MAX):
        #   alpha = QUANT_MAX / (linf + DELTA*sqrt(r) * l2).
        #
        # Universal constants -- fixed by (QUANT_MAX, DELTA, r), not the data;
        # a real implementation precomputes them once.
        r = e.shape[1]
        quant_max = hw.compute.const(QUANT_MAX)
        delta_r = hw.compute.const(DELTA * math.sqrt(r))
        delta_over_std = hw.compute.const(
            DELTA * math.sqrt(r) / (NOISE_TARGET_NORM * NOISE_TARGET_NORM)
        )

        noised_bound = hw.compute.fma(delta_r, l2, linf)
        # (Near-)zero-row guard: an all-zero row gives noised_bound == 0
        # (alpha = inf => alpha*X = NaN), a denormal-tiny row overflows alpha
        # past BF16 max. The 2^-32 floor caps alpha at QUANT_MAX * 2^32
        # (finite in BF16); a huge alpha on a ~zero row is harmless since
        # reconstruction divides it back out. Any floor keeping QUANT_MAX/floor
        # finite works; 2^-32 leaves rows with peak > 2^-32 (real data) untouched.
        alpha = hw.compute.div(quant_max, noised_bound)
        beta = hw.compute.mul(hw.compute.mul(alpha, l2), delta_over_std)

        # hw.matmul(a, b) contracts as a @ b.T, so f transposed yields e @ f = E@F (n x k).
        noise = hw.cast(hw.matmul(e, f.transpose(0, 1)), DType.MATMUL, DType.COMPUTE)
        # fma is noticeably more accurate than separated ops
        noised = hw.compute.fma(alpha, X, hw.compute.mul(beta, noise))
        # Clamp needed very rarely.
        noised = hw.compute.clamp(noised, -quant_max, quant_max)
        return hw.cast(noised, DType.COMPUTE, DType.QUANT), alpha, beta, l2
