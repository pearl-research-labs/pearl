"""Dtype roles and the shared compute-dtype (BF16) element-wise primitives.

The :class:`ComputeOps` primitives are on the bit-exact operand-building path
(they feed the per-row scale derivation, hence the lottery) but have no
accumulation order to pin -- plain element-wise / reductions -- so they are
expected to be identical across devices and are implemented once, concretely,
here rather than per :class:`~.hardware.Hardware` subclass. Each
:class:`Hardware` holds an instance as its :attr:`~.hardware.Hardware.compute`
field.
"""

from __future__ import annotations

import enum

import torch


class DType(enum.Enum):
    """The dtype roles, mapped to concrete ``torch`` dtypes for this reference."""

    QUANT = torch.float8_e4m3fn  # operand precision of the first k columns (MMA input)
    PEEL = torch.bfloat16  # storage of the 2r peel columns / peel MMA operand precision
    COMPUTE = torch.bfloat16  # native element-wise arithmetic dtype
    MATMUL = torch.float32  # MMA accumulator output; output-only, never a matmul operand
    FACTOR = torch.float8_e4m3fn  # dtype of the noise factors E1, E2, F1, F2 (== QUANT)


class ComputeOps:
    """Shared compute-dtype ops, held as a field on :class:`~.hardware.Hardware` (see module docstring)."""

    @staticmethod
    def _expect_compute(*tensors: torch.Tensor) -> None:
        for t in tensors:
            assert t.dtype == DType.COMPUTE.value, (
                f"expected compute_dtype (BF16) operand, got {t.dtype}"
            )

    def const(self, value: float | int | list[float] | list[int]) -> torch.Tensor:
        """Wrap Python scalar(s) as a ``compute_dtype`` (BF16) tensor (0-dim for
        a scalar, 1-D for a list); round to nearest-even BF16."""
        return torch.tensor(value, dtype=DType.COMPUTE.value)

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Element-wise addition in ``compute_dtype`` (BF16)."""
        self._expect_compute(x, y)
        return x + y

    def mul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Element-wise multiplication in ``compute_dtype`` (BF16); broadcasts like ``torch.mul``."""
        self._expect_compute(x, y)
        return x * y

    def div(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Element-wise division in ``compute_dtype`` (BF16); broadcasts like ``torch.div``."""
        self._expect_compute(x, y)
        return x / y

    def fma(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Fused multiply-add ``a*b + c`` with a SINGLE rounding to ``compute_dtype`` (BF16).

        Hardware-FMA contract: the product is exact, only the sum rounds (one
        rounding vs. mul+add's two). A native full-rate GPU instruction: CUDA
        ``__hfma`` on ``__nv_bfloat16`` (``cuda_bf16.h``, compute capability
        >= 8.0), PTX ``fma.rn.bf16`` -- "rounding the result once in
        round-to-nearest-even mode".
        """
        # TODO: consider forbidding some edge cases and simplify the logic.
        self._expect_compute(a, b, c)
        a64, b64, c64 = a.to(torch.float64), b.to(torch.float64), c.to(torch.float64)
        p = a64 * b64  # exact: two 8-bit significands need 16 <= 53 bits
        s = p + c64  # fp64 RNE approximation of the exact sum x
        t = s - c64
        r = (p - t) + (c64 - (s - t))  # TwoSum: x - s, exact
        s32 = s.to(torch.float32)  # brackets x: |s32 - x| < 0.51 ulp
        err = (s - s32.to(torch.float64)) + r  # sign(x - s32); nonzero iff x != s32
        # Round-to-odd fixup: if x is not representable and s32's significand
        # is even, the bracketing odd neighbour is the next fp32 toward x.
        fix = (err != 0) & ((s32.view(torch.int32) & 1) == 0) & s32.isfinite()
        toward = torch.where(err > 0, torch.inf, -torch.inf).to(torch.float32)
        return torch.where(fix, torch.nextafter(s32, toward), s32).to(DType.COMPUTE.value)

    def sqrt(self, x: torch.Tensor) -> torch.Tensor:
        """Element-wise square root in ``compute_dtype`` (BF16)."""
        self._expect_compute(x)
        return x.sqrt()

    def abs_max(self, x: torch.Tensor) -> torch.Tensor:
        """Per-row max of absolute values: ``(n, k) -> (n, 1)`` in ``compute_dtype``."""
        self._expect_compute(x)
        return x.abs().amax(dim=1, keepdim=True)

    def min(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Element-wise minimum in ``compute_dtype`` (BF16); broadcasts like ``torch.min``.

        NaN-propagating (IEEE ``minimum``, matches ``torch.minimum``): required for
        `clamp`'s correct-or-NaN contract below. Kernels must not substitute a
        non-propagating primitive here (e.g. CUDA ``fminf``/PTX ``min.f32``).
        """
        self._expect_compute(x, y)
        return torch.minimum(x, y)

    def max(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Element-wise maximum in ``compute_dtype`` (BF16); broadcasts like ``torch.max``.

        NaN-propagating; same contract and caveat as `min` (``fmaxf``/``max.f32`` forbidden).
        """
        self._expect_compute(x, y)
        return torch.maximum(x, y)

    def clamp(self, x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """Element-wise clamp into ``[lo, hi]`` in ``compute_dtype`` (BF16); composed of `min`/`max`.

        Correct-or-NaN: ``clamp(NaN, lo, hi)`` is NaN, never ``lo``/``hi``. Guards
        `.quantization.Fp8QuantScheme.noisy_quantize`'s final clamp so an
        adversarial NaN ``noised`` value reaches the FP8 cast instead of pinning to
        ``+-QUANT_MAX``.
        """
        return self.max(self.min(x, hi), lo)
