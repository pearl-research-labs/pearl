"""Opaque ``pearl::apply_linear`` torch custom op and eager MoE mining entry point."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from miner_utils import get_logger

from .capture import (
    gpu_mining_producer,
    in_graph_setup_no_mining,
    mining_attempt,
    mining_launches_suspended,
)
from .settings import miner_settings, runtime_settings
from .state import JobContext, LayerState, fp8_fallback_gemm, lookup_state

if TYPE_CHECKING:
    from .moe import MoeRouting

_LOGGER = get_logger(__name__)


def _eager_mining_blocked(state: LayerState, m_tokens: int) -> bool:
    return (
        not state.mineable
        or miner_settings().no_mining
        or m_tokens < runtime_settings().min_mining_tokens
        or in_graph_setup_no_mining()
        or mining_launches_suspended()
    )


def _eager_bucket(state: LayerState, m_tokens: int) -> int | None:
    from .pipeline import pick_bucket, variant_ready

    settings = runtime_settings()
    bucket = pick_bucket(m_tokens)
    if bucket is None:
        _LOGGER.warning(
            f"{m_tokens} tokens exceeds the largest PEARL_M_BUCKETS entry "
            f"({settings.m_buckets[-1]}); such forwards serve without mining.",
        )
        return None
    if settings.warmup_compile and not variant_ready(bucket, state):
        return None
    return bucket


def _try_mine(
    state: LayerState, x2d: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor | None:
    if torch.cuda.is_current_stream_capturing():
        return None
    from .pipeline import run_mining_forward

    return _mine_eager(
        state,
        x2d.shape[0],
        lambda ctx, bucket: run_mining_forward(state, ctx, x2d.contiguous(), bias, bucket),
    )


def try_mine_moe[T](
    state: LayerState,
    x2d: torch.Tensor,
    topk_ids: torch.Tensor,
    tail: "Callable[[torch.Tensor, MoeRouting], T]",
) -> T | None:
    """Mine one MoE layer's first grouped GEMM when admissible; else return None."""
    if torch.cuda.is_current_stream_capturing():
        return None
    from .moe import topk_ids_shape_error
    from .pipeline import run_mining_moe_forward

    problem = topk_ids_shape_error(topk_ids, x2d.shape[0], state.top_k)
    if problem is None and topk_ids.device != x2d.device:
        problem = f"topk_ids on {topk_ids.device} but activations on {x2d.device}"
    if problem is not None:
        _LOGGER.warning(f"{state.layer_name}: {problem}; serving this forward unmined")
        return None

    return _mine_eager(
        state,
        x2d.shape[0],
        lambda ctx, bucket: run_mining_moe_forward(
            state, ctx, x2d.contiguous(), topk_ids, bucket, tail=tail
        ),
    )


def _mine_eager[T](
    state: LayerState, m_tokens: int, launch: Callable[[JobContext, int], T | None]
) -> T | None:
    if _eager_mining_blocked(state, m_tokens):
        return None
    bucket = _eager_bucket(state, m_tokens)
    if bucket is None:
        return None

    from .job_prep import current_context, current_job

    try:
        with gpu_mining_producer() as admitted:
            if not admitted:
                return None
            with mining_attempt(state.weight.device) as attempt:
                if not attempt:
                    return None
                ctx = current_context(state, current_job())
                if ctx is None:
                    return None
                out = launch(ctx, bucket)
                attempt.mark_success(out is not None)
                return out
    except torch.cuda.OutOfMemoryError:
        _LOGGER.warning(
            f"mining OOM on {state.weight.device}; suspending device mining for "
            f"{runtime_settings().oom_cooldown_s}s while serving uses fallback",
        )
        return None
    except Exception as exc:
        state.disable_mining(f"deterministic forward failure ({type(exc).__name__})")
        return None


_LIB = torch.library.Library("pearl", "DEF")
_LIB.define("apply_linear(Tensor x, Tensor weight, Tensor? bias) -> Tensor")


def _validate_bf16_contract(x: torch.Tensor, bias: torch.Tensor | None) -> None:
    if x.dtype != torch.bfloat16:
        raise TypeError(f"pearl::apply_linear requires bfloat16 activations, got {x.dtype}")
    if bias is not None and bias.dtype != torch.bfloat16:
        raise TypeError(f"pearl::apply_linear requires bfloat16 bias, got {bias.dtype}")


@torch.library.impl(_LIB, "apply_linear", "CUDA")
def _apply_linear_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    _validate_bf16_contract(x, bias)
    state = lookup_state(weight)
    x2d = x.reshape(-1, x.shape[-1])
    if x2d.shape[1] != state.k or x2d.device != state.weight.device:
        raise ValueError(
            f"pearl::apply_linear expected (*, {state.k}) on {state.weight.device}, "
            f"got {tuple(x2d.shape)} on {x2d.device}"
        )

    out = _try_mine(state, x2d, bias)
    if out is None:
        if state.w_fp8 is None or state.w_fp8_scale is None:
            raise ValueError(f"pearl::apply_linear does not serve MoE layer {state.layer_name}")
        out = fp8_fallback_gemm(x2d, state.w_fp8, state.w_fp8_scale, bias, torch.bfloat16)
    return out.reshape(*x.shape[:-1], out.shape[-1])


@torch.library.register_fake("pearl::apply_linear")
def _apply_linear_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    _validate_bf16_contract(x, bias)
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=torch.bfloat16, device=x.device)
