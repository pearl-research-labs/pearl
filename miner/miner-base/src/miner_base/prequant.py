"""Block-scaled int8 + BF16 prequantized operands (1 scale per 8 int8 codes)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

DEFAULT_BLOCK_SIZE = 8
L2_ROUNDED_BITS = 2
_QMAX = 127


@dataclass(frozen=True)
class RowNorms:
    """Per-row ``(l2, linf)`` in BF16 ``(n, 1)``, with ``l2`` rounded to the 4-ulp grid."""

    l2: torch.Tensor
    linf: torch.Tensor


@dataclass
class PrequantMatrix:
    """Block-scaled 2D operand: ``int_values`` ``(n, k)`` int8 and ``scales`` ``(n, k/8)`` BF16."""

    int_values: torch.Tensor
    scales: torch.Tensor

    def __post_init__(self) -> None:
        assert self.int_values.dim() == 2, "int_values must be 2D"
        assert self.int_values.dtype == torch.int8, (
            f"int_values must be int8, got {self.int_values.dtype}"
        )
        n, k = self.int_values.shape
        assert k % DEFAULT_BLOCK_SIZE == 0, f"k={k} must be a multiple of {DEFAULT_BLOCK_SIZE}"
        assert self.scales.dtype == torch.bfloat16, f"scales must be BF16, got {self.scales.dtype}"
        assert self.scales.shape == (n, k // DEFAULT_BLOCK_SIZE), (
            f"scales shape {tuple(self.scales.shape)} != ({n}, {k // DEFAULT_BLOCK_SIZE})"
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.int_values.shape[0], self.int_values.shape[1])

    def planes(self) -> list[torch.Tensor]:
        return [self.int_values, self.scales]

    def open(self) -> torch.Tensor:
        n, k = self.shape
        blocks = self.int_values.to(torch.float32).reshape(n, -1, DEFAULT_BLOCK_SIZE)
        opened = blocks * self.scales.to(torch.float32).unsqueeze(-1)
        return opened.reshape(n, k).to(torch.bfloat16)

    def exact_norms(self) -> RowNorms:
        """Compute ``RowNorms`` directly from int8 blocks and BF16 scales (pre-BF16 rounding)."""
        n, k = self.shape
        blocks = self.int_values.to(torch.float32).reshape(n, -1, DEFAULT_BLOCK_SIZE)
        scales_f32 = self.scales.to(torch.float32)
        sumsq = (blocks.pow(2).sum(dim=-1) * scales_f32.pow(2)).sum(dim=-1, keepdim=True)
        linf = (blocks.abs().amax(dim=-1) * scales_f32.abs()).amax(dim=-1, keepdim=True)
        l2 = round_l2_to_grid((sumsq / k).sqrt().to(torch.bfloat16))
        return RowNorms(l2=l2, linf=linf.to(torch.bfloat16))

    @classmethod
    def encode(cls, rows: torch.Tensor) -> PrequantMatrix:
        """Encode ``(n, k)`` matrix to int8 codes + per-8-element BF16 scales."""
        assert rows.dim() == 2
        n, k = rows.shape
        assert k % DEFAULT_BLOCK_SIZE == 0, f"k={k} must be a multiple of {DEFAULT_BLOCK_SIZE}"

        blocks = rows.to(torch.float32).reshape(n, -1, DEFAULT_BLOCK_SIZE)
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(2.0**-100)
        inv_qmax = torch.reciprocal(torch.tensor(float(_QMAX), dtype=torch.float32))
        scales = (amax * inv_qmax).to(torch.bfloat16)
        # Correctly-rounded FP32 reciprocal, matching the device's rcp.rn.f32.
        inv_scales = torch.reciprocal(scales.to(torch.float32))
        int_values = (blocks * inv_scales).round().clamp(-_QMAX, _QMAX)
        return cls(
            int_values=int_values.reshape(n, k).to(torch.int8),
            scales=scales.squeeze(-1),
        )


def round_l2_to_grid(l2: torch.Tensor) -> torch.Tensor:
    """Round non-negative BF16 ``l2`` to the nearest multiple of ``2**L2_ROUNDED_BITS`` ulps (ties up)."""
    step = 1 << L2_ROUNDED_BITS
    bits = l2.view(torch.int16)
    return ((bits + (step >> 1)) & -step).view(torch.bfloat16)
