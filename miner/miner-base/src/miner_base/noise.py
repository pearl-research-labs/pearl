"""Deterministic low-rank noise: the v4 random-access ``SampleLine`` map.

Twin of ``zk-pow/src/api/fp8/noise.rs``. All factors
are stacks of "lines" from one keyed-BLAKE3 rule (:meth:`OperandNoiser._lines`).
A line is ``r`` XOF bytes -> sign x UNIFORM magnitude in ``[1, 128]`` (never
zero, no modulo bias), L2-NORMALIZED to the constant norm
``c := NOISE_TARGET_NORM`` (independent of ``r``), then rounded to FP8 e4m3.
Normalization is exact integer math plus one BF16 division (`ComputeOps.div`),
hence byte-identical on any host: ``norm_scaled = isqrt(sumsq *
_INT_SQRT_PREC^2)`` (``math.isqrt`` = exact FLOOR integer sqrt, so this is
``floor(||x||_2 * _INT_SQRT_PREC)``, an exact int), then ``scale =
(NOISE_TARGET_NORM * _INT_SQRT_PREC) / norm_scaled`` (the one BF16 rounding),
``entry_i = x_i * scale``.

Why a CONSTANT norm: an ``E@F`` entry is ``<e_row, f_col>`` of two such lines
(``F`` is ``k`` lines, transposed, so its columns are lines too), so its peak
(Cauchy-Schwarz, ``~c^2``) and rms (``~c^2/sqrt(r)``) are KNOWN constants and
``quantization.py``'s ``noisy_quantize`` can derive its per-row scales from
``X``'s norms alone, never measuring ``E@F``. The exact procedure is chosen
purely for cross-host bit-exactness; the draw need not be Gaussian -- soundness
only needs it deterministic and public.

Keying follows one random-access rule: the per-side noise seed is
subkeyed under ``pearl/v4/FP8/noise-line``, and each line is addressed by a
``(side, factor, line)`` tuple zero-padded to one 64-byte BLAKE3 block:

  * ``E_X`` (rows x r) -- one line per SELECTED GLOBAL row/column index. For a
    gathered MoE activation the address is the pre-routing token index, so a
    token routed to several experts is noised ONCE; for the stacked MoE weight
    it is the global stacked row ``w * n_e + j``, so distinct experts draw
    distinct lines.
  * ``F_X`` (r x k) -- the side's shared basis: lines ``0..k``, transposed.
    Never varies across experts.

Each side draws from its OWN seed (``F_A`` from ``noise seedA``, ``F_B`` from
``noise seedB``). The address omits ``k`` and ``r``: the seed already binds
them (through ``pA`` / ``pB``).
"""

from __future__ import annotations

import enum
import math

import torch
from blake3 import blake3

from .compute_ops import ComputeOps, DType
from .transcript import LABEL_NOISE_LINE, encode_u32_le, subkey

NOISE_TARGET_NORM = 256  # constant approximate L2 norm of every noise line
_INT_SQRT_PREC = 32  # fixed-point factor carrying log2(32)=5 fractional norm bits through isqrt


class Side(enum.IntEnum):
    """Which operand a noise line belongs to. Values are the committed wire bytes."""

    A = 0
    B = 1


class _Factor(enum.IntEnum):
    """Which factor a line contributes to: the row/col-keyed ``E`` lines or the
    shared ``F`` basis. Values are the committed wire bytes."""

    E = 0
    F = 1


class OperandNoiser:
    """One side's noise factors (``E_X``, ``F_X``) from its noise seed."""

    def __init__(
        self,
        noise_seed: bytes,
        side: Side,
        rank: int,
        common_dim: int,
        compute: ComputeOps,
    ):
        self._key = subkey(LABEL_NOISE_LINE, noise_seed)
        self.side = side
        self.r = rank
        self.k = common_dim
        self.compute = compute
        self._basis: torch.Tensor | None = None

    def E(self, line_indices: list[int]) -> torch.Tensor:
        """``(len(line_indices) x r)`` FP8: one line per selected GLOBAL index."""
        return self._lines(_Factor.E, line_indices)

    def F(self) -> torch.Tensor:
        """``(r x k)`` FP8: the side's shared basis (lines ``0..k``, transposed).
        Memoized; never varies across experts."""
        if self._basis is None:
            self._basis = self._lines(_Factor.F, range(self.k)).transpose(0, 1)
        return self._basis

    def _line_digest(self, factor: _Factor, line: int) -> bytes:
        """ONE line's raw draw: ``r`` keyed-BLAKE3-XOF bytes of the address
        ``side(1) | factor(1) | line(u32 LE)``, zero-padded to 64 bytes (one
        BLAKE3 block). No ``k``/``r`` in the address -- the seed binds them."""
        material = (bytes([self.side, factor]) + encode_u32_le(line)).ljust(64, b"\x00")
        assert len(self._key) == 32 and len(material) == 64
        return blake3(material, key=self._key).digest(length=self.r)

    def _lines(self, factor: _Factor, indices) -> torch.Tensor:
        """Stack of keyed, L2-normalized lines, one per index; the draw is per line."""
        idx = list(indices)
        raw = b"".join(self._line_digest(factor, i) for i in idx)
        b = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(len(idx), self.r)
        x = b.to(torch.int64)
        x = (1 - 2 * (x >> 7)) * ((x & 0x7F) + 1)
        # norm_scaled = floor(||x||_2 * _INT_SQRT_PREC), exact int (math.isqrt
        # floors exactly => bit-exact on any host). _INT_SQRT_PREC cancels in
        # the division; it only carries fractional norm bits through the isqrt.
        norm_scaled = [
            math.isqrt(int(sumsq) * (_INT_SQRT_PREC * _INT_SQRT_PREC))
            for sumsq in (x * x).sum(dim=1)
        ]
        scale = self.compute.div(
            self.compute.const(NOISE_TARGET_NORM * _INT_SQRT_PREC),
            self.compute.const(norm_scaled),
        )
        return (x.to(torch.bfloat16) * scale.unsqueeze(1)).to(DType.FACTOR.value)


Factor = _Factor
