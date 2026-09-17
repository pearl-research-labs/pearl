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
