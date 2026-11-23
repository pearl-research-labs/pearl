"""Decode a quantized linear layer's weight back to BF16 via its delegate's apply()."""

from typing import TYPE_CHECKING

import torch
from miner_utils import get_logger

from .settings import runtime_settings

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

_LOGGER = get_logger(__name__)
_RECON_FREE_MEMORY_FRACTION = 0.5


class UpcastBudgetError(MemoryError):
    """Reconstructing the weight would exceed the device's free-memory budget."""


def _warn_if_static_activation_scale(layer: torch.nn.Module) -> bool:
    if getattr(layer, "input_scale", None) is None:
        return False
    _LOGGER.warning(
        "pearl upcast: source scheme uses a static activation scale, adding a uniform "
        "quantization rounding factor to reconstructed weights. Prefer dynamic activations."
    )
    return True


def _estimate_reconstruct_peak_bytes(n: int, k: int, chunk_rows: int) -> int:
    chunk = min(chunk_rows, k)
    # destination BF16 + identity BF16 + delegate output (conservative FP32) + BF16 copy.
    return n * k * 2 + chunk * k * 2 + chunk * n * 4 + chunk * n * 2


def _check_reconstruct_budget(n: int, k: int, device: torch.device, chunk_rows: int) -> None:
    if device.type != "cuda":
        return
    peak = _estimate_reconstruct_peak_bytes(n, k, chunk_rows)
    free, _ = torch.cuda.mem_get_info(device)
    budget = int(free * _RECON_FREE_MEMORY_FRACTION)
    if peak > budget:
        raise UpcastBudgetError(
            f"reconstructing ({n}, {k}) BF16 needs ~{peak / 1e9:.2f} GB of transient "
            f"GPU memory, exceeding the upcast budget ({budget / 1e9:.2f} GB free on {device})."
        )


def reconstruct_bf16(
    layer: torch.nn.Module,
    source: "QuantizeMethodBase",
    n: int,
    k: int,
    device: torch.device,
    chunk_rows: int | None = None,
) -> torch.Tensor:
    """Decode ``layer``'s quantized weight to a dense ``(n, k)`` BF16 tensor in chunks."""
    if n <= 0 or k <= 0:
        raise ValueError(f"cannot reconstruct a weight with shape ({n}, {k})")
    if chunk_rows is None:
        chunk_rows = runtime_settings().recon_chunk_rows
    elif type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError(f"chunk_rows must be an integer >= 1, got {chunk_rows!r}")
    _check_reconstruct_budget(n, k, device, chunk_rows)
    _warn_if_static_activation_scale(layer)
    weight = torch.empty(n, k, dtype=torch.bfloat16, device=device)
    for start in range(0, k, chunk_rows):
        end = min(start + chunk_rows, k)
        rows = end - start
        ident = torch.zeros(rows, k, dtype=torch.bfloat16, device=device)
        diag = torch.arange(rows, device=device)
        ident[diag, start + diag] = 1.0
        out = source.apply(layer, ident, bias=None)
        if out.dim() != 2 or tuple(out.shape) != (rows, n):
            raise ValueError(
                f"upcast delegate returned shape {tuple(out.shape)} for an "
                f"({rows}, {k}) identity block; expected ({rows}, {n})"
            )
        weight[:, start:end] = out.to(torch.bfloat16).t()
    return weight.contiguous()
