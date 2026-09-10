"""Functional host launch for fused B-side stats, noising, quantization, and peel.

The B side runs the identical fused kernel as ``noisy_quant`` on mirrored
operands: the noise dot ``E2 @ F2`` consumes
``pack_noise_factor(F2)`` and the peel contraction consumes raw ``F1``, with
E2 lines drawn under B's noise-line key at the ``B | E`` address. In cert-v4
``F1`` (= ``F_A``) is per launch, so production launches pass a zero ``F1``
and complete the peel's mid half per launch with :func:`b_peel_for_a`.
"""

import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .._utils._compile import get_or_compile
from .._utils._stream import get_stream
from .._utils._validation import require_tensor
from ..noisy_quant._host import (
    _DEFAULT_CONFIG,
    NoisyQuantConfig,
    _launch_prep,
)
from ..noisy_quant._quantization_ops import _L_E2, _noise_base_words
from ..protocol_constants import R
from ._kernel import _GramFactor, _PeelFixup

# Same kernel, same tuning space: the config type is shared with the A side.
NoisyQuantBConfig = NoisyQuantConfig

# Caller-facing buffer names, in ``_launch_prep``'s shared role order. The
# factor sides swap relative to the A kernel: the packed noise-UMMA operand is
# F2 and the raw peel operand is F1.
_B_SIDE_NAMES = (
    "codes",
    "scales",
    "c_b",
    "commit_stats",
    "f2_hl",
    "f1",
    "alpha_b",
    "beta_b",
    "e2",
    "b_prime",
    "b_peel",
)

_gram_cache: dict[tuple, object] = {}
_fixup_cache: dict[tuple, object] = {}


def noisy_quant_b(
    codes: torch.Tensor,
    scales: torch.Tensor,
    c_b: torch.Tensor,
    commit_stats: torch.Tensor,
    f2_hl: torch.Tensor,
    f1: torch.Tensor,
    alpha_b: torch.Tensor,
    beta_b: torch.Tensor,
    e2: torch.Tensor,
    b_prime: torch.Tensor,
    b_peel: torch.Tensor,
    gram: torch.Tensor,
    *,
    config: NoisyQuantBConfig = _DEFAULT_CONFIG,
) -> None:
    """Launch B-side noising into caller-owned outputs.

    ``gram`` is the one caller-owned workspace: the peel epilogue's ``(R, R)``
    f32 ``F2 @ F1^T`` factor (see ``_kernel.py``).

    ``codes`` and ``scales`` are the two committed block-scaled weight planes
    (``PrequantMatrix``'s int8 codes and BF16 group scales) and ``commit_stats``
    their per-512-element-block ``(sumsq, absmax)`` pairs from
    ``tensor_hash_plus_stats_b``. ``f2_hl`` is ``pack_noise_factor(F2)`` and
    ``f1`` the raw ``(R, k)`` peel factor. Outputs mirror the A side:
    ``b_peel`` columns ``[0, R)`` hold the ``(beta (.) E2@F2 - B') @ F1^T``
    mid half and ``[R, 2R)`` hold ``-(beta (.) E2)``. ``beta_b`` is emitted
    for parity/debugging only; nothing downstream reads it (it is folded into
    ``b_peel``). ``inv_alpha_b`` stays caller-side: build it with
    ``torch.reciprocal(alpha_b.float()).contiguous()``, since ``alpha_b`` is
    BF16 and ``mixed_gemm`` requires a contiguous FP32 operand.

    ``c_b`` is B's 32-byte noise-line key (``Subkey("noise-line", noise
    seedB)``). The launch covers the whole matrix -- E2 rows are addressed by
    global row index -- so cancellation granularity is the launch.

    The producer chain is ``pre_quant`` -> ``tensor_hash_plus_stats_b`` ->
    ``noise_lines``.
    """
    n, k = codes.shape
    device = codes.device
    require_tensor(
        "gram",
        gram,
        dtype=torch.float32,
        shape=(R, R),
        device=device,
        alignment=16,
    )
    _launch_prep(
        codes,
        scales,
        c_b,
        commit_stats,
        f2_hl,
        f1,
        alpha_b,
        beta_b,
        e2,
        b_prime,
        b_peel,
        entry="noisy_quant_b",
        names=_B_SIDE_NAMES,
        msg_base=_noise_base_words(_L_E2),
        config=config,
    )

    # The fused kernel emits the peel in its A-side layout; the epilogue
    # reduces G = F2 @ F1^T into the gram workspace and rewrites the peel in
    # place into build_b_rows' order (see _kernel.py).
    capability = torch.cuda.get_device_capability(device)
    stream = get_stream(device.index or 0)
    gram_args = (
        from_dlpack(f2_hl, assumed_align=16),
        from_dlpack(f1, assumed_align=16),
        from_dlpack(gram, assumed_align=16),
    )
    gram_compiled = get_or_compile(
        _gram_cache,
        (capability, k),
        lambda: cute.compile(_GramFactor(k), *gram_args, stream),
    )
    gram_compiled(*gram_args, stream)

    fixup_args = (
        from_dlpack(b_peel, assumed_align=16),
        from_dlpack(gram, assumed_align=16),
    )
    fixup_compiled = get_or_compile(
        _fixup_cache,
        (capability, n),
        lambda: cute.compile(_PeelFixup(n), *fixup_args, stream),
    )
    fixup_compiled(*fixup_args, stream)


def b_peel_for_a(
    b_prime: torch.Tensor,
    e2: torch.Tensor,
    f2: torch.Tensor,
    beta_b: torch.Tensor,
    b_peel: torch.Tensor,
    f1_lines: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """B's ``(n, 2R)`` peel for one A: ``[(beta_b (.) E_B@F_B - B') @ F_A^T | -(beta_b (.) E_B)]``.

    v4 draws ``F_A`` from the launch's own ``noise seedA``, so the mid half is a
    per-launch product over all of ``B'`` (an ``(n, k) x (k, R)`` FP8 GEMM: one
    extra streaming read of the weight per mined forward). It is computed as
    ``beta_b (.) (E_B @ (F_B @ F_A^T)) - B' @ F_A^T``; the reference's ``(n, k)``
    BF16 difference never materializes. The mid half is the reference-only
    (tolerance) path; the second half is the per-job block ``noisy_quant_b``
    left in ``b_peel`` (launched with a zero ``f1``), copied through.

    ``f1_lines`` is the ``(k, R)`` ``noise_lines(..., LABEL_F1, ...)`` draw
    under A's noise-line key; ``e2``/``f2``/``beta_b`` are ``noisy_quant_b``'s
    outputs/inputs for the same job.

    Contract (every operand contiguous, on one CUDA device; ``k % 16 == 0``
    and ``n % 16 == 0`` as ``_scaled_mm`` requires, which every mineable
    shape satisfies):

    - ``b_prime`` ``(n, k)`` FP8 e4m3 (``noisy_quant_b``'s quantized B)
    - ``e2`` ``(n, R)`` FP8 e4m3, ``f2`` ``(R, k)`` FP8 e4m3
    - ``beta_b`` ``n`` BF16 values, ``(n,)`` or the reference's ``(n, 1)``
    - ``b_peel`` ``(n, 2R)`` BF16 (only its right half is read)
    - ``f1_lines`` ``(k, R)`` FP8 e4m3
    - ``out`` ``(n, 2R)`` BF16, allocated on ``b_prime.device`` when omitted

    Violations raise ``TypeError``/``ValueError`` before any device work.
    Beyond ``out`` the helper allocates its FP32 intermediates per call
    (``(n, R)`` product, ``(R, R)`` gram, ``(R, k)`` transposed ``F_A``); see
    ``vllm_miner.memory._launch_bytes`` for the budget.
    """
    require_tensor("b_prime", b_prime, dtype=torch.float8_e4m3fn)
    if b_prime.ndim != 2:
        raise ValueError(f"b_prime must be (n, k), got shape {tuple(b_prime.shape)}")
    n, k = b_prime.shape
    device = b_prime.device
    if n % 16 or k % 16:
        raise ValueError(f"b_prime must have n and k multiples of 16, got (n, k)=({n}, {k})")
    require_tensor("e2", e2, dtype=torch.float8_e4m3fn, shape=(n, R), device=device)
    require_tensor("f2", f2, dtype=torch.float8_e4m3fn, shape=(R, k), device=device)
    require_tensor("beta_b", beta_b, dtype=torch.bfloat16, device=device)
    if beta_b.numel() != n:
        raise ValueError(f"beta_b must hold n={n} values, got shape {tuple(beta_b.shape)}")
    require_tensor("b_peel", b_peel, dtype=torch.bfloat16, shape=(n, 2 * R), device=device)
    require_tensor("f1_lines", f1_lines, dtype=torch.float8_e4m3fn, shape=(k, R), device=device)
    if out is None:
        out = torch.empty(n, 2 * R, dtype=torch.bfloat16, device=device)
    else:
        require_tensor("out", out, dtype=torch.bfloat16, shape=(n, 2 * R), device=device)

    f1 = f1_lines.t().contiguous()  # (R, k) F_A
    one = _unit_scale(device)
    # (n, k) row-major @ (k, R) column-major in FP8; f32 accumulate/out.
    b_fa = torch._scaled_mm(b_prime, f1.t(), scale_a=one, scale_b=one, out_dtype=torch.float32)
    gram = f2.float() @ f1.float().t()  # (R, R) F_B @ F_A^T
    mid = beta_b.reshape(n, 1).float() * (e2.float() @ gram) - b_fa
    out[:, :R] = mid
    out[:, R:] = b_peel[:, R:]
    return out


_unit_scales: dict[torch.device, torch.Tensor] = {}


def _unit_scale(device: torch.device) -> torch.Tensor:
    """``_scaled_mm``'s FP32 scalar ``1.0`` scale, allocated once per device."""
    one = _unit_scales.get(device)
    if one is None:
        one = _unit_scales[device] = torch.ones((), dtype=torch.float32, device=device)
    return one
