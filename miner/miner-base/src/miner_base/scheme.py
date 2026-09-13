"""The with-peel scheme: stacked rows + the single fused tile kernel.

Recall the algebra (each side X draws its own factors
``E_X``, ``F_X`` from its noise seed; the injected rank-r noise is
``N_X = E_X @ F_X``):

    A' = Q(A + EA @ FA)                                  # m x k, FP8
    B' = Q(B + EB @ FB)                                  # n x k, FP8
    A'' = [ A'  ||  EA                   ||  A' @ FB.T ]  # m x (k + 2r)
    B'' = [ B'  ||  (EB@FB - B') @ FA.T  ||  -EB       ]  # n x (k + 2r)
    C'' = A'' @ B''.T  ==  (A' - EA@FA) @ (B' - EB@FB).T  ~=  A @ B.T

``compute_tile`` returns BOTH the non-peeled first-stage product
``acc = A' @ B'.T`` (the lottery runs on this, so the scheme does not depend on
BF16 bit-exact matmuls) and the fully peeled ``C''`` tile (the approximated
useful product).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .hardware import DType, Hardware
from .noise import OperandNoiser
from .prequant import RowNorms
from .quantization import Fp8QuantScheme


@dataclass
class StackedRows:
    """One side of the matmul: the FP8 first-k columns + the 2r peel columns."""

    quant_part: torch.Tensor  # (rows x k), FP8 (quant_dtype)
    peel_part: torch.Tensor  # (rows x 2r), peel_dtype (BF16)
    alpha: torch.Tensor  # (rows x 1), compute_dtype: per-row scale on the clean operand
    l2: torch.Tensor  # (rows x 1), compute_dtype: floored row rms the scales were derived from


class PearlScheme:
    """Builds A''/B'' rows and the shared tile kernel; white-box with the matmul op."""

    def __init__(self, hardware: Hardware, quant: Fp8QuantScheme, k: int, r: int):
        self.hw = hardware
        self.quant = quant
        self.k = k
        self.r = r

    def build_a_rows(
        self,
        rows: torch.Tensor,
        noise_a: OperandNoiser,
        noise_b: OperandNoiser,
        row_indices: list[int],
        row_norms: RowNorms,
    ) -> StackedRows:
        """``row_indices`` are the GLOBAL noise-line addresses of ``rows`` (for a
        gathered MoE activation: the pre-routing token indices, giving
        noise-once across experts)."""
        assert rows.shape[1] == self.k, "A rows must have width k"
        assert len(row_indices) == rows.shape[0], "one row index per A row"
        e_a = noise_a.E(row_indices)  # (m x r) FP8
        f_a = noise_a.F()  # (r x k) FP8
        f_b = noise_b.F()  # (r x k) FP8

        a_prime, alpha_a, beta_a, l2_a = self.quant.noisy_quantize(
            rows, e_a, f_a, self.hw, row_norms
        )  # (m x k) FP8

        # peel = [ beta_a (.) EA | A' @ FB.T ]  (reference-only path: builds C'', not the lottery)
        # The EA peel column carries the SAME beta_a scale used on the noise, so
        # A' - (beta_a (.) EA) @ FA = alpha_a (.) A telescopes exactly.
        peel_e = self.hw.compute.mul(beta_a, self.hw.cast_to_peel(e_a, DType.FACTOR))  # (m x r)
        a_fb = self.hw.matmul_peel(
            self.hw.cast(a_prime, DType.QUANT, DType.COMPUTE), f_b
        )  # A' @ FB.T -> (m x r)
        peel_afb = self.hw.cast_to_peel(a_fb, DType.MATMUL)
        peel = torch.cat([peel_e, peel_afb], dim=1)
        assert peel.shape[1] == 2 * self.r
        return StackedRows(a_prime, peel, alpha_a, l2_a)

    def build_b_rows(
        self,
        rows: torch.Tensor,
        noise_a: OperandNoiser,
        noise_b: OperandNoiser,
        row_indices: list[int],
        row_norms: RowNorms,
    ) -> StackedRows:
        """``row_indices`` are the GLOBAL noise-line addresses of ``rows`` (for
        the stacked MoE weight: ``w * n_e + j``, giving per-expert lines)."""
        assert rows.shape[1] == self.k, "B rows must have width k"
        assert len(row_indices) == rows.shape[0], "one row index per B row"
        e_b = noise_b.E(row_indices)  # (n x r) FP8
        f_a = noise_a.F()  # (r x k) FP8
        f_b = noise_b.F()  # (r x k) FP8

        b_prime, alpha_b, beta_b, l2_b = self.quant.noisy_quantize(
            rows, e_b, f_b, self.hw, row_norms
        )  # (n x k) FP8

        # peel = [ (beta_b (.) EB@FB - B') @ FA.T | -(beta_b (.) EB) ]  (reference-only path).
        # Both EB-bearing terms carry the SAME beta_b scale used on the noise, so
        # B' - (beta_b (.) EB) @ FB = alpha_b (.) B telescopes exactly.
        ebfb = self.hw.cast(
            self.hw.matmul_peel(e_b, f_b.transpose(0, 1)), DType.MATMUL, DType.COMPUTE
        )
        b_prime_c = self.hw.cast(b_prime, DType.QUANT, DType.COMPUTE)
        diff = self.hw.compute.fma(beta_b, ebfb, -b_prime_c)  # (beta_b (.) EB@FB - B')
        mid = self.hw.cast_to_peel(self.hw.matmul_peel(diff, f_a), DType.MATMUL)  # (n x r)
        neg_eb = -self.hw.compute.mul(
            beta_b, self.hw.cast_to_peel(e_b, DType.FACTOR)
        )  # -(beta_b (.) EB)
        peel = torch.cat([mid, neg_eb], dim=1)
        assert peel.shape[1] == 2 * self.r
        return StackedRows(b_prime, peel, alpha_b, l2_b)

    def compute_tile(self, a: StackedRows, b: StackedRows) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(acc, C_tile)``, both ``matmul_dtype`` (FP32): ``acc`` = the
        non-peeled ``A' @ B'.T`` from the bit-exact FP8 MMA (the lottery input);
        ``C_tile`` = the peeled ``C'' ~= A @ B.T``, finished by the
        reference-only peel matmul (not bit-exact-required across devices)."""
        assert a.quant_part.shape[1] == self.k
        assert b.quant_part.shape[1] == self.k
        assert a.peel_part.shape[1] == 2 * self.r == b.peel_part.shape[1]

        acc = self.hw.matmul(a.quant_part, b.quant_part)  # FP8 @ FP8, non-peeled
        c_tile = self.hw.matmul_peel(a.peel_part, b.peel_part, acc=acc)  # reference-only peel
        assert c_tile.shape == (a.quant_part.shape[0], b.quant_part.shape[0])
        return acc, c_tile

    def unscale(
        self, c_full: torch.Tensor, alpha_a: torch.Tensor, alpha_b: torch.Tensor
    ) -> torch.Tensor:
        """Undo the per-row scales to recover ``C_approx ~= A @ B.T`` (BF16):
        the peeled product is ``C'' = outer(alpha_a, alpha_b) (.) (A @ B.T)``,
        so divide the outer product out. Reference-only reconstruction (the
        verifier never runs it), so plain FP32 arithmetic is fine."""
        scale = alpha_a.to(DType.MATMUL.value) @ alpha_b.to(DType.MATMUL.value).transpose(0, 1)
        unscaled = (c_full / scale).to(DType.MATMUL.value)
        return self.hw.cast(unscaled, DType.MATMUL, DType.COMPUTE)
