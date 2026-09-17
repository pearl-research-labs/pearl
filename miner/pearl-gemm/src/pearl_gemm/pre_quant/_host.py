"""Functional host launch for block-scale INT8 quantization (BF16 -> blobs)."""

from dataclasses import dataclass

import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .._utils._compile import get_or_compile
from .._utils._stream import get_stream
from .._utils._validation import require_tensor
from ..protocol_constants import BLOCK_SCALE_GROUP
from ._kernel import CHUNK_ELEMS, _pre_quant_launch


@dataclass(frozen=True)
class PreQuantConfig:
    """Compile-time quantization launch configuration."""

    threads_per_block: int = 256

    def __post_init__(self) -> None:
        if self.threads_per_block not in (128, 256, 512):
            raise ValueError("threads_per_block must be 128, 256, or 512")


def pre_quant_output_shapes(m: int, k: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """The ``(codes, scales)`` blob shapes: ``(m, k)`` int8 and ``(m, k/8)`` BF16."""
    if k % CHUNK_ELEMS:
        raise ValueError("k must be divisible by 512")
    return (m, k), (m, k // BLOCK_SCALE_GROUP)


# Shared default for the public entry point; the config is frozen, so one
# instance is safe to reuse as an argument default.
_DEFAULT_CONFIG = PreQuantConfig()

_compile_cache: dict[tuple, object] = {}


def pre_quant(
    a: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    config: PreQuantConfig = _DEFAULT_CONFIG,
) -> None:
    """Quantize BF16 rows to block-scaled INT8 into caller-owned buffers.

    ``codes`` receives the ``(m, k)`` int8 codes and ``scales`` the
    ``(m, k/8)`` BF16 block scales -- the two blobs
    ``tensor_hash_plus_stats`` commits to, one keyed Merkle tree each. The
    ``(sumsq, absmax)`` commit-stats partials over the dequantized rows are
    produced by ``tensor_hash_plus_stats`` from those blobs, not here.
    """
    if a.ndim != 2:
        raise ValueError("a must be 2D")
    m, k = a.shape
    codes_shape, scales_shape = pre_quant_output_shapes(m, k)
    device = a.device
    require_tensor("a", a, dtype=torch.bfloat16, shape=(m, k), alignment=16)
    require_tensor(
        "codes",
        codes,
        dtype=torch.int8,
        shape=codes_shape,
        device=device,
        alignment=16,
    )
    require_tensor(
        "scales",
        scales,
        dtype=torch.bfloat16,
        shape=scales_shape,
        device=device,
        alignment=16,
    )

    args = (
        from_dlpack(a, assumed_align=16),
        # Reinterpret both blobs as flat u32 streams. Valid because a row is
        # a whole number of 512-element blocks (512 code bytes / 128 scale
        # bytes each, both multiples of 4) and require_tensor pinned the
        # storage to 16 bytes. Any layout change must preserve both
        # invariants or these views break.
        from_dlpack(codes.view(torch.uint32).reshape(-1), assumed_align=16),
        from_dlpack(scales.view(torch.uint32).reshape(-1), assumed_align=16),
    )
    stream = get_stream(device.index or 0)
    cache_key = (torch.cuda.get_device_capability(device), m, k, config)

    def compile_variant():
        return cute.compile(
            _pre_quant_launch,
            *args,
            stream,
            threads_per_block=config.threads_per_block,
        )

    compiled = get_or_compile(_compile_cache, cache_key, compile_variant)
    compiled(*args, stream)
