import torch
from pearl_gemm import quantize
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

from .mining_state import get_async_manager

MAX_VAL_7BIT = 63
MAX_VAL_8BIT = 127


def quantize_kernel(
    x: torch.Tensor,
    max_val: int = MAX_VAL_7BIT,
    smooth_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    """
    Symmetric per-token quantization with optional smooth scaling (CUDA kernel version)

    Args:
        x: Input tensor (any dtype)
        max_val: Maximum quantization value (63 for 7-bit, 127 for 8-bit)
        smooth_scale: Optional smooth_quant_scale to divide input by

    Returns:
        xq: Quantized int8 tensor
        xq_scales: Per-token scales (fp32)
        None: Zero point (not used for symmetric quant)
    """
    fast_math = get_async_manager()._conf.quantization_fast_math
    num_tokens, _ = x.shape
    x_q = torch.empty_like(x, dtype=torch.int8)
    x_s = torch.empty((num_tokens, 1), dtype=torch.float32, device=x.device)
    quantize(x, x_q, x_s, fast_math=fast_math, max_val=max_val, smooth_scale=smooth_scale)
    return x_q, x_s, None


# Convenience wrappers
def quant_7bit(
    x: torch.Tensor, smooth_scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, None]:
    """7-bit symmetric quantization (range: [-63, 63]), optional smooth scale."""
    return quantize_kernel(x, max_val=MAX_VAL_7BIT, smooth_scale=smooth_scale)


def quant_8bit(
    x: torch.Tensor, smooth_scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, None]:
    """8-bit symmetric quantization (range: [-127, 127]), optional smooth scale."""
    return quantize_kernel(x, max_val=MAX_VAL_8BIT, smooth_scale=smooth_scale)


def quant_fp8_block(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token-group fp8 quantization for the block-scaled GEMM2."""
    return moe_kernel_quantize_input(
        A=x,
        A_scale=None,
        quant_dtype=torch.float8_e4m3fn,
        per_act_token_quant=False,
        block_shape=[group_size, group_size],
    )
