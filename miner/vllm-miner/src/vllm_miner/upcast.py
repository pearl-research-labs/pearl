"""Decode a quantized linear layer's weight back to BF16 for the FP10 encoder.

Quantized checkpoints (FP8 / NVFP4 / MXFP4) are decoded through the delegate
quant method that loaded them, so the existing FP10 encode path
(:func:`create_layer_state`) can mine them. Rather than re-implementing each
nibble / scale layout, identity rows are pushed through the delegate's own
``apply`` and transposed, so vLLM's own kernels do the dequant.

Every supported scheme is *weight-quantized*, and with **dynamic** activation
quantization the round trip equals a reference dequant: amax scaling maps the
identity's ``1.0`` onto an exactly representable code (e.g. ``1.0 -> 448`` for
FP8 E4M3) and its ``0.0`` rows contribute nothing. A checkpoint-provided
*static* activation scale breaks that exactness -- see
:func:`_warn_if_static_activation_scale`.

The result is lossy: it does not recover precision the source checkpoint
already threw away, so callers must label it as mining over already-quantized
weights.
"""

from typing import TYPE_CHECKING

import torch
from miner_utils import get_logger

from .settings import runtime_settings

if TYPE_CHECKING:
    # Annotation only: importing vLLM at module load would pull it into CPU tests.
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

_LOGGER = get_logger(__name__)

# Reconstruction may not claim more than this fraction of the device's free memory
# for its transient (the destination weight plus one identity/output chunk). The
# packed source weights and the FP10 encode result are separate; this bounds only
# the upcast workspace, so a layer that would OOM the worker fails early instead
# of aborting startup halfway through ``process_weights_after_loading``.
_RECON_FREE_MEMORY_FRACTION = 0.5


class UpcastBudgetError(MemoryError):
    """Reconstructing the weight would exceed the device's free-memory budget.

    Raised by :func:`reconstruct_bf16` before allocating any workspace so the
    caller can fall back to serving the layer via its source scheme rather than
    aborting startup. The message carries the estimated peak bytes.
    """


def _warn_if_static_activation_scale(layer: torch.nn.Module) -> bool:
    """Return True and warn once if the delegate uses a static activation scale.

    Static scaling quantizes the identity's ``1.0`` to ``quant(1.0 / s) * s``
    instead of an exact code, adding a uniform rounding factor to every
    reconstructed weight. Surface it so callers don't reconstruct silently.
    """
    if getattr(layer, "input_scale", None) is None:
        return False
    _LOGGER.warning(
        "pearl upcast: the source scheme uses a static activation scale, so the "
        "identity round trip adds a uniform quantization rounding factor to every "
        "reconstructed weight. Prefer a dynamic-activation checkpoint for mining.",
    )
    return True


def _estimate_reconstruct_peak_bytes(n: int, k: int, chunk_rows: int) -> int:
    """Peak transient GPU bytes the upcast allocates at any one time.

    The destination ``(n, k)`` BF16 weight lives for the whole call. Within the
    hottest step -- ``weight[:, start:end] = out.to(bfloat16).t()`` -- four
    chunk-sized tensors are simultaneously live: the ``(rows, k)`` BF16 identity
    block, the ``(rows, n)`` delegate output, and its ``(rows, n)`` BF16 copy
    (the ``.t()`` is a view, not a copy). The delegate output is charged at
    **FP32** (4 bytes): fp8 / compressed-tensors kernels return the layer's
    compute dtype, which can be wider than BF16, so a BF16-only estimate would
    understate the peak and admit an OOM the guard is meant to prevent.
    """
    chunk = min(chunk_rows, k)
    destination = n * k * 2
    identity = chunk * k * 2
    delegate_output = chunk * n * 4  # conservative: delegate may return FP32
    conversion_copy = chunk * n * 2  # transient out.to(bfloat16)
    return destination + identity + delegate_output + conversion_copy


def _check_reconstruct_budget(n: int, k: int, device: torch.device, chunk_rows: int) -> None:
    """Refuse shapes whose upcast workspace would exceed the free-memory budget.

    ``can_mine_layer`` only vetoes shapes by tile/alignment and SM capability, so
    a quantized checkpoint with an unusually large mineable dense linear could
    pass the gate yet OOM the worker during reconstruction (the packed source
    weights are already resident). This runs *before* any workspace is allocated
    and raises :class:`UpcastBudgetError` so the caller can fall back to serving
    the layer via its source scheme instead of aborting startup.

    A CPU ``device`` is always allowed: the CPU tests build tiny layers and the
    host RAM budget is not the concern this guard protects against.
    """
    if device.type != "cuda":
        return
    peak = _estimate_reconstruct_peak_bytes(n, k, chunk_rows)
    free, _total = torch.cuda.mem_get_info(device)
    budget = int(free * _RECON_FREE_MEMORY_FRACTION)
    if peak > budget:
        raise UpcastBudgetError(
            f"reconstructing ({n}, {k}) BF16 needs ~{peak / 1e9:.2f} GB of transient "
            f"GPU memory, which exceeds the upcast budget ({budget / 1e9:.2f} GB free "
            f"on {device}). Serve this layer via its source scheme instead of mining."
        )


def reconstruct_bf16(
    layer: torch.nn.Module,
    source: "QuantizeMethodBase",
    n: int,
    k: int,
    device: torch.device,
    chunk_rows: int | None = None,
) -> torch.Tensor:
    """Decode ``layer``'s quantized weight to a dense ``(n, k)`` BF16 tensor.

    Called once per mineable linear at load time (from
    ``process_weights_after_loading``), never on the serving hot path. ``source``
    is the delegate linear method (already run through
    ``process_weights_after_loading``); its ``apply`` computes ``x @ W.T``, so
    feeding a ``k``-wide identity block yields the corresponding columns of
    ``W`` (as ``W.T`` rows).

    The loop walks ``k`` in ``chunk_rows`` slices (default
    ``PEARL_RECON_CHUNK_ROWS``); ``k`` need not be a multiple of the chunk size
    -- the final iteration is narrowed to the remainder (verified by the
    chunk-boundary test).

    Raises :class:`UpcastBudgetError` before allocating if the workspace would
    exceed the device's free-memory budget; callers should fall back to serving
    via the source scheme in that case.
    """
    if n <= 0 or k <= 0:
        raise ValueError(f"cannot reconstruct a weight with shape ({n}, {k})")
    if chunk_rows is None:
        # RuntimeSettings already enforces ge=1 for the configured default.
        chunk_rows = runtime_settings().recon_chunk_rows
    elif not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows < 1:
        # Validate the explicit override before any allocation: 0 would raise an
        # opaque error from range(), and a negative value would skip the loop and
        # return uninitialised torch.empty data as a "successful" reconstruction.
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
        # (rows, n): rows [start:end] of W.T -> columns [start:end] of W.
        out = source.apply(layer, ident, bias=None)
        # A delegate that folds a reshape/padding into ``apply`` would otherwise
        # broadcast or silently mis-slice into the destination columns.
        if out.dim() != 2 or tuple(out.shape) != (rows, n):
            raise ValueError(
                f"upcast delegate returned shape {tuple(out.shape)} for an "
                f"({rows}, {k}) identity block; expected ({rows}, {n})"
            )
        weight[:, start:end] = out.to(torch.bfloat16).t()
    return weight.contiguous()
