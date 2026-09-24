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
the miner runs end-to-end on CPU; :class:`Blackwell` pins SM100 FP8 MMA
arithmetic via a bit-exact CPU emulation.

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

import torch

from .compute_ops import ComputeOps, DType

__all__ = ["DType", "Hardware", "ReferenceHardware", "Blackwell", "hardware_for"]


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

    def matmul_fp8_partials(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Like ``matmul`` (FP8 @ FP8, no carry-in) but returns, for each output
        cell, the running FP32 accumulator after every accumulation group --
        shape ``(a.shape[0], b.shape[0], ceil(k / 32))``, the last slice equal
        to the final matmul. These are the partial sums ``c_v`` the jackpot
        policy's prefix-inclusive anchor (``policy.py`` check 3) consumes.
        """
        raise NotImplementedError

    # `W` in the jackpot policy's convention (``jackpot_policy.rs``): a summand
    # more than `W` bits below the accumulation window's anchor is dropped.
    # 25 on Blackwell (tcgen05.mma's 25 fractional bits below the anchor).
    fp8_window_bits: int

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

    ``matmul`` is a CPU STUB. The protocol's matmul is the device's real
    FP8 @ FP8 MMA opcode (SM100 ``tcgen05.mma`` kind ``f8f6f4``); for its
    exact arithmetic see :class:`Blackwell` below.
    The BF16-upcast torch matmul below does NOT reproduce that bit-for-bit;
    verification against a real device needs the device-accurate ``matmul``
    swapped in (:class:`Blackwell` below).
    All lottery-critical computation (``Fp8QuantScheme`` / ``PearlScheme``) composes
    only these primitives, so swapping in a device's bit-exact ``matmul`` / ``cast``
    yields that device's exact results.
    """

    name = "reference"

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

    def matmul(self, a, b):
        _assert_matmul_operands(a, b)
        from .fp8_sim_blackwell import matmul_fp8_sim_blackwell

        return matmul_fp8_sim_blackwell(a, b).to(DType.MATMUL.value)

    def matmul_fp8_partials(self, a, b):
        _assert_quant_operands("matmul_fp8_partials", a, b)
        from .fp8_sim_blackwell import matmul_fp8_sim_blackwell_partials

        return matmul_fp8_sim_blackwell_partials(a, b)


def hardware_for(device: object) -> Hardware:
    """Resolve a committed device to its bit-exact CPU emulation.

    Blackwell (SM100) is the only device this miner mines on or verifies
    against; every other committed device is rejected.
    """
    from .params import Device

    match device:
        case Device.BLACKWELL:
            return Blackwell()
        case _:
            raise ValueError(f"no hardware emulation for {device!r}")
