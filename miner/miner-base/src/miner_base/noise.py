"""Deterministic low-rank noise: the v4 random-access ``SampleLine`` map.

Twin of ``zk-pow/src/api/fp8/noise.rs``. All factors
are stacks of "lines" from one keyed-BLAKE3 rule (:meth:`_OperandNoiser._lines`).
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

``E_A`` is keyed by ``noise seedA``. Both F bases are keyed by ``noise seedB``
(``F_A`` uses ``Side.A`` addresses so it is a distinct draw from ``F_B``);
``E_B`` is keyed by ``noise seedB``. The address omits ``k`` and ``r``: the
seed already binds them (through ``pA`` / ``pB``).

:class:`Noiser` is the one public way to draw factors: it applies that seed
rule itself, so a caller never chooses which seed keys which factor.
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


class _OperandNoiser:
    """One side's noise factors (``E_X``, ``F_X``), each under its own key.

    Private: :class:`Noiser` is the only constructor, and it is what binds the
    protocol's seeds to the two keys. ``e_seed`` may be ``None`` while the
    side's E seed is unknown (the A side before seedA exists); ``F`` is still
    drawable then, ``E`` is not.
    """

    def __init__(
        self,
        side: Side,
        rank: int,
        common_dim: int,
        compute: ComputeOps,
        *,
        e_seed: bytes | None,
        f_seed: bytes,
    ):
        self._key = subkey(LABEL_NOISE_LINE, e_seed) if e_seed is not None else None
        self._f_key = subkey(LABEL_NOISE_LINE, f_seed)
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
        key = self._f_key if factor is _Factor.F else self._key
        if key is None:
            raise ValueError(f"E_{self.side.name} needs the side's E seed (seedA for Side.A)")
        assert len(key) == 32 and len(material) == 64
        return blake3(material, key=key).digest(length=self.r)

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


class Noiser:
    """One job's four noise factors under the protocol's seed rule.

    ``F_A``, ``E_B`` and ``F_B`` are keyed by ``seed_b`` and exist from it
    alone: the B side and its complete peel are job constants, fixed before
    any A is known. ``E_A`` is the only seedA-keyed factor; construct with
    ``seed_a`` (or take :meth:`with_seed_a`) before reading it.

    This is the only public way to draw factors, so no caller picks which seed
    keys which factor.
    """

    def __init__(
        self,
        seed_b: bytes,
        rank: int,
        common_dim: int,
        compute: ComputeOps,
        *,
        seed_a: bytes | None = None,
    ):
        self.seed_a = seed_a
        self.seed_b = seed_b
        self.r = rank
        self.k = common_dim
        self.compute = compute
        self._a = _OperandNoiser(Side.A, rank, common_dim, compute, e_seed=seed_a, f_seed=seed_b)
        self._b = _OperandNoiser(Side.B, rank, common_dim, compute, e_seed=seed_b, f_seed=seed_b)

    def with_seed_a(self, seed_a: bytes) -> Noiser:
        """The same job once ``seed_a`` is known. The seedB-keyed bases are
        shared, so a memoized ``F_A`` / ``F_B`` is not redrawn."""
        other = Noiser(self.seed_b, self.r, self.k, self.compute, seed_a=seed_a)
        other._a._basis = self._a._basis
        other._b = self._b
        return other

    def E_A(self, row_indices: list[int]) -> torch.Tensor:
        """``(len(row_indices) x r)`` FP8, keyed by seedA: one line per GLOBAL A row."""
        return self._a.E(row_indices)

    def F_A(self) -> torch.Tensor:
        """``(r x k)`` FP8, keyed by seedB at ``Side.A`` addresses; memoized."""
        return self._a.F()

    def E_B(self, row_indices: list[int]) -> torch.Tensor:
        """``(len(row_indices) x r)`` FP8, keyed by seedB: one line per GLOBAL B row."""
        return self._b.E(row_indices)

    def F_B(self) -> torch.Tensor:
        """``(r x k)`` FP8, keyed by seedB at ``Side.B`` addresses; memoized."""
        return self._b.F()


Factor = _Factor
