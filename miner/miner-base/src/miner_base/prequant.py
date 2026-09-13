"""The miner's prequantized input dtype: int8 values + per-block BF16 scales.

``int8 blk8 bf16s``: each row is split into contiguous blocks of 8 elements;
a block stores 8 quantized int8 values plus one shared BF16 scale, and opens
(dequantizes) to BF16 as ``int_value * scale``. The commitment hashes the two
tensors separately and combines the digests --
``blake3(commit(int_values) || commit(scales))`` -- see
``commitment.commit_planes``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

DEFAULT_BITS = 8  # int8 values
DEFAULT_BLOCK_SIZE = 8  # one scale per 1x8 block
L2_ROUNDED_BITS = 2  # low explicit BF16 mantissa bits cleared from l2 (see round_l2_to_grid)


def _qmax(bits: int) -> int:
    """Largest stored magnitude for ``bits``-bit signed values (symmetric grid)."""
    return 2 ** (bits - 1) - 1


@dataclass(frozen=True)
class RowNorms:
    """Per-row ``(l2, linf)``, BF16 ``(n x 1)`` each, feeding
    `~.quantization.Fp8QuantScheme.row_norms`. ``l2`` is
    ``rms(X) = sqrt(sumsq / k)``, grid-rounded (:func:`round_l2_to_grid`);
    both are computed by the caller straight from an operand's
    pre-BF16-rounding values, right where its ``sumsq`` is produced (see
    :meth:`PrequantMatrix.exact_norms`)."""

    l2: torch.Tensor
    linf: torch.Tensor


@dataclass
class PrequantMatrix:
    """A prequantized 2D operand.

    ``int_values``: (n x k) int8, each in ``[-qmax, qmax]``.
    ``scales``: (n x k/block_size) BF16, one positive scale per block.
    """

    int_values: torch.Tensor
    scales: torch.Tensor
    block_size: int = DEFAULT_BLOCK_SIZE
    bits: int = DEFAULT_BITS

    def __post_init__(self) -> None:
        assert self.int_values.dim() == 2, "int_values must be 2D"
        assert self.int_values.dtype == torch.int8, (
            f"int_values must be int8, got {self.int_values.dtype}"
        )
        n, k = self.int_values.shape
        assert k % self.block_size == 0, f"k={k} must be a multiple of block_size={self.block_size}"
        assert self.scales.dtype == torch.bfloat16, f"scales must be BF16, got {self.scales.dtype}"
        assert self.scales.shape == (n, k // self.block_size), (
            f"scales shape {tuple(self.scales.shape)} != ({n}, {k // self.block_size})"
        )

    @property
    def shape(self) -> tuple[int, int]:
        """The shape (n, k) of the operand this opens to."""
        return (self.int_values.shape[0], self.int_values.shape[1])

    @property
    def qmax(self) -> int:
        """Largest stored magnitude: 127 for int8 (the grid is symmetric)."""
        return _qmax(self.bits)

    def planes(self) -> list[torch.Tensor]:
        """The tensors to commit to, in order: int_values, scales."""
        return [self.int_values, self.scales]

    def open(self) -> torch.Tensor:
        """Dequantize to the (n x k) BF16 operand: ``int_value * scale`` per block.

        The product is exact in FP32 (int8 x BF16 fits easily), so the only
        rounding is the final cast to BF16.
        """
        n, k = self.shape
        blocks = self.int_values.to(torch.float32).reshape(n, -1, self.block_size)
        opened = blocks * self.scales.to(torch.float32).unsqueeze(-1)
        return opened.reshape(n, k).to(torch.bfloat16)

    def exact_norms(self) -> RowNorms:
        """`RowNorms` computed directly from the int8 blocks + BF16 scales,
        bypassing :meth:`open`'s per-element BF16 rounding.

        Per block: ``scale^2 * sum(int_i^2)`` and ``scale * max(|int_i|)``.
        ``sum(int_i^2)`` over a ``block_size``-wide int8 block is exact in
        FP32 (e.g. ``<= 127^2 * 8`` for the default block size of 8), so each
        block's contribution to ``sumsq`` only picks up the single FP32
        rounding of the ``scale^2`` multiply, rather than accumulating
        ``block_size`` independently BF16-rounded terms as :meth:`open`
        followed by a plain sum-of-squares would.
        """
        n, k = self.shape
        blocks = self.int_values.to(torch.float32).reshape(n, -1, self.block_size)
        scales_f32 = self.scales.to(torch.float32)  # (n x n_blocks)
        sumsq = (blocks.pow(2).sum(dim=-1) * scales_f32.pow(2)).sum(dim=-1, keepdim=True)
        linf = (blocks.abs().amax(dim=-1) * scales_f32.abs()).amax(dim=-1, keepdim=True)
        l2 = round_l2_to_grid((sumsq / k).sqrt().to(torch.bfloat16))
        return RowNorms(l2=l2, linf=linf.to(torch.bfloat16))

    @classmethod
    def encode(cls, rows: torch.Tensor) -> PrequantMatrix:
        """Quantize a (n x k) matrix into this format.

        Per block: ``scale = bf16(amax * reciprocal(qmax))``, ``int_value =
        round(x * reciprocal(scale))`` clamped to ``[-qmax, qmax]``, where
        ``reciprocal`` is the correctly-rounded FP32 reciprocal. A scale
        that rounds below ``amax / qmax`` trims the block max by at most
        half a step via the clamp. All-zero blocks get a tiny scale floor
        so the per-block reciprocal stays well-defined (and FP32-normal).

        The scale multiply is the fast form of ``bf16(amax / qmax)``: for
        every reachable amax (a BF16 magnitude at or above the floor) the
        FP32 quotient sits far enough from any BF16 rounding midpoint that
        the BF16 cast absorbs the multiply's error, so the two produce
        identical bits (``tests/test_prequant_encode.py`` checks the whole
        domain exhaustively). Device implementations may take further
        bit-preserving shortcuts on the same grounds: ``rcp.approx.f32``
        plus one Newton step reproduces the correctly-rounded reciprocal
        for every BF16-precision normal scale, and ``|x *
        reciprocal(scale)| <= qmax * (1 + 2^-9 + O(2^-24)) < qmax + 0.5``
        means a round-to-nearest-even saturating int8 convert reproduces
        ``round().clamp(-qmax, qmax)`` (no -128 is ever produced).

        Reference *suggestion*, not a protocol requirement: a miner may
        derive ``int_values``/``scales`` by any procedure, so long as the
        result is a valid `PrequantMatrix`.
        """
        block_size = cls.block_size
        bits = cls.bits
        assert rows.dim() == 2
        n, k = rows.shape
        assert k % block_size == 0, f"k={k} must be a multiple of block_size={block_size}"
        qmax = _qmax(bits)

        blocks = rows.to(torch.float32).reshape(n, -1, block_size)
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(2.0**-100)
        inv_qmax = torch.reciprocal(torch.tensor(float(qmax), dtype=torch.float32))
        scales = (amax * inv_qmax).to(torch.bfloat16)  # (n x n_blocks x 1)
        # Correctly-rounded FP32 reciprocal, matching the device's rcp.rn.f32;
        # the scale floor keeps it finite and normal at either extreme.
        inv_scales = torch.reciprocal(scales.to(torch.float32))
        int_values = (blocks * inv_scales).round().clamp(-qmax, qmax)
        return cls(
            int_values=int_values.reshape(n, k).to(torch.int8),
            scales=scales.squeeze(-1),
            block_size=block_size,
            bits=bits,
        )


def round_l2_to_grid(l2: torch.Tensor) -> torch.Tensor:
    """Round non-negative BF16 ``l2`` to the nearest multiple of ``2**L2_ROUNDED_BITS`` ulps (ties up).

    Non-negative IEEE-754 floats order like their bit patterns, so this is
    "add half a grid step to the int16 view, clear the low bits" (any carry
    into the mantissa/exponent falls out for free). Vs. ceiling to the same
    grid: equally robust to a 1-ulp host discrepancy, but half the mean bias
    (1.5 -> 0.5 ulp), removing ceiling's ~0.7% high bias on ``beta`` (hence
    on the injected noise-to-signal). Called wherever a row's exact ``sumsq``
    is turned into ``l2`` (see `RowNorms`), so a <=1-ulp FP32 discrepancy in
    ``sumsq`` between miner and verifier can't change the result.
    """
    step = 1 << L2_ROUNDED_BITS
    bits = l2.view(torch.int16)
    return ((bits + (step >> 1)) & -step).view(torch.bfloat16)
