"""Functional host launch for fused B-side stats, noising, quantization, and peel.

The B side runs the identical fused kernel as ``noisy_quant`` on mirrored
operands: the noise dot ``E2 @ F2`` consumes
``pack_noise_factor(F2)`` and the peel contraction consumes raw ``F1``, with
E2 lines drawn under B's noise-line key at the ``B | E`` address. Both F
bases are keyed by seedB (``F1`` = ``F_A`` at the ``A | F`` address), so one
launch per job emits the complete B peel.
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
    ``f1`` the raw ``(R, k)`` peel factor ``F_A`` (the ``LABEL_F1`` draw under
    the same key, transposed). Outputs mirror the A side:
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
