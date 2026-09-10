"""Functional host launch for keyed-BLAKE3 noise-line generation."""

import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from .._utils._compile import get_or_compile
from .._utils._stream import get_stream
from .._utils._validation import require_tensor
from ..noisy_quant._quantization_ops import (
    _L_E1,
    _L_E2,
    _L_F1,
    _L_F2,
    _noise_base_words,
)
from ..protocol_constants import R
from ._kernel import _NoiseLines

# The four v4 noise-line address prefixes ``side | factor`` (``OperandNoiser``'s
# draws): E1/F1 are A's row lines and shared basis, E2/F2 are B's.
LABEL_E1 = _L_E1
LABEL_E2 = _L_E2
LABEL_F1 = _L_F1
LABEL_F2 = _L_F2

_compile_cache: dict[tuple, object] = {}


def noise_lines(
    key: torch.Tensor,
    label: bytes,
    out: torch.Tensor,
) -> None:
    """Draw normalized noise lines for indices ``[0, count)`` into ``out``.

    One keyed-BLAKE3 line per row of ``out`` (a ``(count, R)`` e4m3 tensor),
    with the same line semantics as the fused prep kernels' in-kernel E
    generation: key = ``key`` (the side's noise-line key,
    ``Subkey("noise-line", noise seedX)``), single 64-byte block message
    ``label || u32(index)`` zero-padded where ``label`` is the ``side |
    factor`` address prefix, digest length R, sign/magnitude decode, exact
    isqrt L2 normalization to 256, e4m3 rounding
    (``miner_base.noise.OperandNoiser._lines``).

    For the F basis: draw ``k`` lines under ``LABEL_F1`` (A's key) /
    ``LABEL_F2`` (B's key) and materialize any factor passed raw to a prep
    kernel with ``factor = out.t().contiguous()``. ``pack_noise_factor``
    accepts the transpose directly and returns a contiguous packed blob; at
    ``R == PACKED_NOISE_K`` the ``(k, R)`` draw viewed as int8 already *is*
    the blob. On the B side, pass raw F1 to the peel side and
    ``pack_noise_factor(F2)`` to the noise-dot side, exactly the reference
    ``OperandNoiser.F`` assembly per side.
    """
    if not isinstance(label, bytes):
        raise TypeError("label must be bytes")
    msg_base = _noise_base_words(label)
    device = key.device
    require_tensor("key", key, dtype=torch.uint8, shape=(32,), device=device, alignment=4)
    if out.ndim != 2:
        raise ValueError("out must be 2D")
    count = out.shape[0]
    if count <= 0:
        raise ValueError("out must have at least one line (count > 0)")
    if count >= 1 << 32:
        raise ValueError("line count does not fit the message's u32 line index")
    # The kernel offsets ``out`` by ``index * R`` in 32-bit arithmetic
    # (_kernel.py); keep that product below 2**31 (a 2 GiB draw).
    if count * R >= 1 << 31:
        raise ValueError("line count exceeds the kernel's 32-bit index * R addressing")
    require_tensor(
        "out",
        out,
        dtype=torch.float8_e4m3fn,
        shape=(count, R),
        device=device,
        alignment=16,
    )

    args = (from_dlpack(key.view(torch.uint32)), from_dlpack(out.view(-1), assumed_align=16))
    stream = get_stream(device.index or 0)
    cache_key = (torch.cuda.get_device_capability(device), count, msg_base)
    compiled = get_or_compile(
        _compile_cache,
        cache_key,
        lambda: cute.compile(_NoiseLines(count, msg_base), *args, stream),
    )
    compiled(*args, stream)
