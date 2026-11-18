"""Validation helpers for caller-owned kernel buffers."""

from collections.abc import Sequence

import torch


def require_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    shape: Sequence[int] | None = None,
    device: torch.device | None = None,
    contiguous: bool = True,
    alignment: int | None = None,
) -> None:
    """Validate a tensor before exposing its storage to a device kernel."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if alignment is not None and tensor.data_ptr() % alignment:
        raise ValueError(f"{name} must be {alignment}-byte aligned")


def _byte_span(tensor: torch.Tensor) -> tuple[int, int]:
    start = tensor.data_ptr()
    return start, start + tensor.numel() * tensor.element_size()


def require_disjoint_writes(
    writes: Sequence[tuple[str, torch.Tensor | None]],
    reads: Sequence[tuple[str, torch.Tensor | None]],
) -> None:
    """Reject a written buffer whose bytes overlap any other operand.

    Read/read aliasing is harmless and allowed; a write overlapping a read
    (or another write) lets one tile overwrite bytes another tile still
    reads. Spans are compared by ``data_ptr``, so no device read is needed.
    """
    write_spans = [(name, _byte_span(t)) for name, t in writes if t is not None]
    read_spans = [(name, _byte_span(t)) for name, t in reads if t is not None]
    for i, (name_w, (start_w, end_w)) in enumerate(write_spans):
        others = write_spans[i + 1 :] + read_spans
        for name_o, (start_o, end_o) in others:
            if start_w < end_o and start_o < end_w:
                raise ValueError(f"{name_w} is written and must not overlap {name_o} in memory")


def require_buffer(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    min_elements: int,
    device: torch.device,
    alignment: int | None = None,
) -> None:
    """Validate a flat workspace or output buffer."""
    require_tensor(
        name,
        tensor,
        dtype=dtype,
        device=device,
        alignment=alignment,
    )
    if tensor.numel() < min_elements:
        raise ValueError(
            f"{name} must contain at least {min_elements} elements, got {tensor.numel()}"
        )
