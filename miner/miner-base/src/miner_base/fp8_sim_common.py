"""
Util functions for FP matmul simulators.
"""

from __future__ import annotations

import torch

ATOM_K = 32  # products per MMA instruction along K (QGMMA group / tcgen05.mma atom)
FP32_MIN_EXP = -126  # fp32 minimum normal exponent
FP32_MANTISSA_BITS = 23  # fractional bits of the fp32 significand
E4M3_SIG_EXP_OFFSET = 10  # e4m3 value = sig * 2^(exp - 10): bias 7 + 3 mantissa bits


def split_e4m3(codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw e4m3 bytes -> ``(sign, exponent bits, significand)`` as integers (int32), with
    value ``(-1)^sign * sig * 2^(exp - 10)``.

    - NaN bytes (``0x7F``/``0xFF``) decode to garbage: callers mask them with `nan_mask`.
    - Subnormals keep their 3-bit significand un-normalized with the exponent
    bits treated as 1 (both devices consume them that way).
    """
    c = codes.to(torch.int32)
    sign = (c >> 7) & 1
    exp = (c >> 3) & 0xF
    man = c & 0x7
    sig = torch.where(exp != 0, man | 0x8, man)
    return sign, exp.clamp_(min=1), sig


def nan_mask(codes: torch.Tensor) -> torch.Tensor:
    """
    Returns a boolean tensor of the same shape as `codes` indicating whether each element is a NaN byte.
    """
    return ((codes & 0x7F) == 0x7F).any(dim=-1)


def fp8_to_raw_bytes(x: torch.Tensor) -> torch.Tensor:
    """View ``float8_e4m3fn`` tensor as its raw ``uint8`` codes."""
    assert x.dtype == torch.float8_e4m3fn and x.dim() == 2
    return x.contiguous().view(torch.uint8)
