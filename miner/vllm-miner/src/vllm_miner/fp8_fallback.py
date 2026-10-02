"""Job-independent W8A8-FP8 fallback GEMM for non-mined forwards (the only path
a captured CUDA graph ever replays)."""

from collections.abc import Callable

import torch

from .mining_config import SCALE_BLOCK

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
# Row chunk for the load-time dequant->FP8 pass (bounds the BF16 transient).
_BUILD_CHUNK_ROWS = 4096
_ActivationQuantizer = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
_activation_quantizer: _ActivationQuantizer | None = None


def register_activation_quantizer(quantizer: _ActivationQuantizer) -> None:
    """Install the serving framework's fused dynamic per-token FP8 op."""
    global _activation_quantizer
    _activation_quantizer = quantizer


def open_rows(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize FP10 rows to FP32: ``code * scale`` per 8-block."""
    rows, k = codes.shape
    blocks = codes.to(torch.float32).reshape(rows, -1, SCALE_BLOCK)
    return (blocks * scales.to(torch.float32).unsqueeze(-1)).reshape(rows, k)


def build_fp8_fallback(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row-scaled FP8 copy of the opened weight for ``torch._scaled_mm``."""
    n, k = weight.shape
    device = weight.device
    w_fp8 = torch.empty(n, k, dtype=torch.float8_e4m3fn, device=device)
    w_scale = torch.empty(1, n, dtype=torch.float32, device=device)
    for start in range(0, n, _BUILD_CHUNK_ROWS):
        end = min(start + _BUILD_CHUNK_ROWS, n)
        opened = open_rows(weight[start:end], weight_scale[start:end])
        scale = opened.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / _FP8_MAX
        w_fp8[start:end] = (opened / scale).to(torch.float8_e4m3fn)
        w_scale[0, start:end] = scale.squeeze(1)
    return w_fp8, w_scale


def fp8_fallback_gemm(
    x2d: torch.Tensor,
    w_fp8: torch.Tensor,
    w_fp8_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Dynamic per-token FP8 quant + rowwise ``torch._scaled_mm``."""
    if _activation_quantizer is None:
        # Framework-neutral fallback for standalone use and tests. Serving
        # adapters register their fused quantizer during worker initialization.
        x_f32 = x2d.to(torch.float32)
        x_scale = x_f32.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / _FP8_MAX
        x_fp8 = (x_f32 / x_scale).to(torch.float8_e4m3fn)
    else:
        x_fp8, x_scale = _activation_quantizer(x2d)
    return torch._scaled_mm(
        x_fp8,
        w_fp8.t(),
        scale_a=x_scale,
        scale_b=w_fp8_scale,
        bias=bias,
        out_dtype=out_dtype,
    )
