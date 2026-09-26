"""Per-hardware kernel surface, in three tiers:

* **Device-specific (bit-exact).** Feeds the *lottery ticket*; miner and
  verifier must reproduce it byte-for-byte with a pinned accumulation order:
  the ``quant_dtype @ quant_dtype`` MMA (`Hardware.matmul` -- ONE pinned
  opcode for both the tile GEMM and the rank-r ``E @ F`` noise dot) and the
  operand-path casts (`Hardware.cast`). Abstract here; one kernel per device.

* **Shared compute-dtype ops.** The element-wise ``compute_dtype`` (BF16)
  primitives of :class:`Hardware`, identical on most hardwares.

* **Reference-only (deviable).** Builds ``C'' = A'' @ B''.T`` -- the
  *approximated useful product*. Not lottery-critical, never re-run by the
  verifier, so real implementations may deviate freely
  (`Hardware.matmul_peel`, `Hardware.cast_to_peel`).

:class:`ReferenceHardware` implements both kernel tiers in plain ``torch`` so
the miner runs end-to-end on CPU; :class:`Hopper` and :class:`Blackwell` provide
bit-exact CPU emulations of their devices' FP8 arithmetic.

Design note (propagate-or-fail): the device-specific tier's correct-or-NaN
contract (`Hardware.matmul` / `Hardware.cast` below) propagates invalid
values (NaN) to the output rather than failing fast -- a
reference-implementation choice, not a protocol requirement. A verifier is
free to instead fail immediately at the first operation that would produce
one. More generally, any implementation may reject as soon as its realized
arithmetic diverges materially from the mathematical ground truth, of
which NaN is only the sharpest instance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .compute_ops import ComputeOps, DType

__all__ = [
    "DType",
    "Fp8Replay",
    "Hardware",
    "ReferenceHardware",
    "Hopper",
    "Blackwell",
    "hardware_for",
]


@dataclass
class Fp8Replay:
    """Accumulators after each rank-32 atom, stored as FP32 tensors.

    On Hopper, ``totals`` records ``RNE_FP32(C + c)``, where ``C`` is the
    total before the current window and ``c`` is its local accumulator.
    """

    totals: torch.Tensor  # (rows_a, rows_b, ceil(k/32))
    window_partials: torch.Tensor | None  # local c before promotion; same shape, or None
    window_groups: int | None  # atoms per promotion window, or None without promotion


def _assert_quant_operands(op: str, a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.dtype == DType.QUANT.value and b.dtype == DType.QUANT.value, (
        f"{op} is quant_dtype @ quant_dtype only, got {a.dtype} @ {b.dtype}"
    )


def _assert_matmul_operands(a: torch.Tensor, b: torch.Tensor) -> None:
    _assert_quant_operands("matmul", a, b)
    assert a.dim() == 2 and b.dim() == 2, "matmul operands must be 2D"
    assert a.shape[1] == b.shape[1], (
        f"matmul contraction dim mismatch: {a.shape[1]} vs {b.shape[1]}"
    )


class Hardware:
    """Arithmetic of a specific accelerator (see module docstring for the three tiers)."""

    name: str = "generic"
    compute: ComputeOps = ComputeOps()

    # ================= Device-specific (bit-exact) surface ==============

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Bit-exact ``quant_dtype @ quant_dtype`` MMA computing ``a @ b.T`` -> ``matmul_dtype``.

        Pinned to the device's native FP8 tensor-core opcode -- on Blackwell the
        CTA-scope tcgen05.mma (UMMA) with its truncated group accumulation.
        This single opcode defines BOTH the main tile GEMM and the rank-r
        ``E @ F`` noise dot (K < 32 contractions are zero-padded to the
        instruction's K = 32; zero products are no-ops in the group sum).
        There is deliberately no smaller-scope alternative: SM100 has no
        warp-scope FP8 tensor core that matches this arithmetic, so kernels
        computing the noise dot must issue a real tcgen05.mma.

        Security contract (correct-or-NaN): feeds the lottery, so no input --
        including undecodable bytes like e4m3 NaN (``0x7F``/``0xFF``) -- may
        silently produce a wrong finite result. Decode correctly, or propagate NaN
        into every output element that contracts over it (matching real
        hardware's IEEE NaN propagation); never clamp/zero/fix up an undecodable
        operand. Non-finite tiles are rejected by the plain verifier's
        finiteness check at strip-open (``zk-pow/src/api/verify.rs``).
        """
        raise NotImplementedError

    def cast(self, x: torch.Tensor, from_dtype: DType, to_dtype: DType) -> torch.Tensor:
        """Bit-exact operand-path dtype conversion: COMPUTE<->QUANT or MATMUL->COMPUTE.

        Same correct-or-NaN contract as :meth:`matmul`: NaN/inf or out-of-range
        inputs must map to the target dtype's NaN, never silently saturate to a
        finite max/min value.
        """
        raise NotImplementedError

    def matmul_fp8_replay(self, a: torch.Tensor, b: torch.Tensor) -> Fp8Replay:
        """Replay matrix multiplication while recording intermediate accumulators."""
        raise NotImplementedError

    # Fractional bits below the atom's alignment exponent (`W` in ``jackpot_policy.rs``):
    # 13 on Hopper (WGMMA), 25 on Blackwell (tcgen05.mma).
    fp8_window_bits: int
    # Row noise standard deviation / scaled row norm: 1.0 on Hopper, 0.5 on Blackwell.
    noise_fraction: float

    # ============ Reference-only (deviable) surface =====================
    # Used ONLY to build C'' = A'' @ B''.T (the approximated product). The
    # verifier never re-runs the peel, and the lottery ignores it, so a real
    # implementation is free to deviate (fuse, reorder, use any BF16 kernel).

    def matmul_peel(
        self, a: torch.Tensor, b: torch.Tensor, acc: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Reference-only matmul ``a @ b.T`` (optionally ``+ acc``) for the peel; NOT bit-exact-required."""
        raise NotImplementedError

    def cast_to_peel(self, x: torch.Tensor, from_dtype: DType) -> torch.Tensor:
        """Reference-only cast into ``peel_dtype`` (from a peel matmul output or a noise factor)."""
        raise NotImplementedError


class ReferenceHardware(Hardware):
    """Plain-``torch`` reference so the miner runs end-to-end on CPU.

    ``matmul`` is a CPU STUB. The protocol uses Hopper's FP8 WGMMA or
    Blackwell's ``tcgen05.mma`` kind ``f8f6f4``, with device-specific accumulation.
    The BF16-upcast torch matmul below does NOT reproduce that bit-for-bit;
    verification against a real device needs the device-accurate ``matmul``
    swapped in (:class:`Hopper` or :class:`Blackwell` below).
    All lottery-critical computation (``Fp8QuantScheme`` / ``PearlScheme``) composes
    only these primitives, so swapping in a device's bit-exact ``matmul`` / ``cast``
    yields that device's exact results.
    """

    name = "reference"
    noise_fraction = 0.5

    # -- device-specific surface --

    def matmul(self, a, b):
        _assert_matmul_operands(a, b)
        # STUB (see class docstring): BF16-upcast matmul, fp32 accumulation.
        # NOT bit-exact to the device's FP8 MMA opcode.
        return (a.to(torch.bfloat16) @ b.to(torch.bfloat16).transpose(-1, -2)).to(
            DType.MATMUL.value
        )

    # Largest finite e4m3 magnitude; values beyond it (or non-finite) must cast
    # to NaN, never saturate (the correct-or-NaN contract in `cast`).
    _E4M3_MAX = 448.0

    def cast(self, x, from_dtype, to_dtype):
        # Only the bit-exact operand-path conversions are permitted here.
        assert (from_dtype, to_dtype) in (
            (DType.COMPUTE, DType.QUANT),
            (DType.MATMUL, DType.COMPUTE),
            (DType.QUANT, DType.COMPUTE),
        ), f"cast {from_dtype} -> {to_dtype} is not a protocol conversion (see cast_to_peel)"
        assert x.dtype == from_dtype.value, f"cast expected {from_dtype.value}, got {x.dtype}"
        out = x.to(to_dtype.value)
        if to_dtype is DType.QUANT:
            # torch's FP8 cast saturates out-of-range inputs in recent versions;
            # enforce correct-or-NaN explicitly (torch-version independent). The
            # quantization path clamps to +-448 first, so this never fires there.
            xf = x.to(torch.float32)
            bad = ~torch.isfinite(xf) | (xf.abs() > self._E4M3_MAX)
            if bool(bad.any()):
                out = out.clone()
                out.view(torch.uint8)[bad] = 0x7F  # e4m3 NaN encoding
        return out

    # -- reference-only surface (usual torch matmul / cast; may deviate) --

    def matmul_peel(self, a, b, acc=None):
        operand_dtypes = (DType.PEEL.value, DType.QUANT.value)  # BF16 or FP8; never matmul_dtype
        assert a.dtype in operand_dtypes and b.dtype in operand_dtypes, (
            f"matmul_peel operands must be BF16/FP8, got {a.dtype} @ {b.dtype}"
        )
        assert a.dim() == 2 and b.dim() == 2, "matmul_peel operands must be 2D"
        assert a.shape[1] == b.shape[1], (
            f"matmul_peel contraction dim mismatch: {a.shape[1]} vs {b.shape[1]}"
        )
        out = a.to(DType.MATMUL.value) @ b.to(DType.MATMUL.value).transpose(-1, -2)
        if acc is not None:
            assert acc.dtype == DType.MATMUL.value, "matmul_dtype accumulator mismatch"
            assert acc.shape == out.shape, f"acc shape {acc.shape} != product shape {out.shape}"
            out = out + acc
        return out  # matmul_dtype

    def cast_to_peel(self, x, from_dtype):
        assert from_dtype in (DType.MATMUL, DType.FACTOR), (
            f"cast_to_peel source must be MATMUL or FACTOR, got {from_dtype}"
        )
        assert x.dtype == from_dtype.value, (
            f"cast_to_peel expected {from_dtype.value}, got {x.dtype}"
        )
        return x.to(DType.PEEL.value)


class Hopper(ReferenceHardware):
    """NVIDIA Hopper FP8 WGMMA arithmetic, emulated by :mod:`.fp8_sim_hopper`.

    Each atom accumulates 32 products at 13 fractional bits below its alignment
    exponent. Each window starts at zero and chains up to four atoms; its result
    is added to the FP32 total with round-to-nearest-even. The last window may
    contain fewer than 128 products.
    """

    name = "Hopper"
    fp8_window_bits = 13
    noise_fraction = 1.0

    def matmul(self, a, b):
        _assert_matmul_operands(a, b)
        from .fp8_sim_hopper import matmul_fp8_sim_hopper

        return matmul_fp8_sim_hopper(a, b).to(DType.MATMUL.value)

    def matmul_fp8_replay(self, a, b):
        _assert_matmul_operands(a, b)
        from .fp8_sim_hopper import PROMOTE_GROUPS, matmul_fp8_sim_hopper_replay

        _, totals, windows = matmul_fp8_sim_hopper_replay(a, b)
        return Fp8Replay(
            totals=totals,
            window_partials=windows,
            window_groups=PROMOTE_GROUPS,
        )


class Blackwell(ReferenceHardware):
    """NVIDIA Blackwell: ``matmul`` is the tcgen05.mma (kind::f8f6f4) FP8 atom arithmetic.

    Emulated bit-exactly by :mod:`.fp8_sim_blackwell` (a torch port of the
    Blackwell-characterized simulator, itself verified against B200 silicon):
    e4m3 products are exact and un-normalized, each K = 32 atom is a SINGLE
    accumulation group of (acc + 32 products) truncated towards zero at 25
    fractional bits below the group's max stored exponent (accumulator
    included), the group sum rounds to FP32 towards zero, and K is consumed
    in ascending chained atoms with the zero-padded remainder atom last --
    the placement measured for cuBLAS/CUTLASS SM100 on Blackwell. The noise dot
    (K = 16, zero-padded to one K = 32 atom) is a single group of this same
    arithmetic. ``cast`` and the deviable peel surface stay inherited.
    """

    name = "Blackwell"
    fp8_window_bits = 25
    noise_fraction = 0.5

    def matmul(self, a, b):
        _assert_matmul_operands(a, b)
        from .fp8_sim_blackwell import matmul_fp8_sim_blackwell

        return matmul_fp8_sim_blackwell(a, b).to(DType.MATMUL.value)


def hardware_for(device: object) -> Hardware:
    """Resolve Hopper or Blackwell to its bit-exact CPU emulation; reject other devices."""
    from .params import Device

    match device:
        case Device.HOPPER:
            return Hopper()
        case Device.BLACKWELL:
            return Blackwell()
        case _:
            raise ValueError(f"no hardware emulation for {device!r}")
