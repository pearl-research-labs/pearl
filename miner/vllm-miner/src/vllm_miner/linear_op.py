"""The ``pearl`` torch.library namespace: one opaque mining-aware linear op.

``pearl::apply_linear`` hides the whole mining decision (the ``MINER_NO_MINING``
kill switch, context lookup, m bucketing, variant readiness, the four-op
pipeline, callback scheduling) from torch.compile / CUDA-graph tracing. When the
layer cannot or must not mine right now it runs the job-independent FP8 fallback
GEMM -- the only path a captured graph ever replays.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from miner_utils import get_logger

from .capture import (
    gpu_mining_producer,
    in_graph_setup_no_mining,
    mining_launches_suspended,
)
from .config import config as gpu_config
from .fp8_fallback import fp8_fallback_gemm
from .health import mining_attempt
from .settings import runtime_settings
from .state import JobContext, LayerState, lookup_state

if TYPE_CHECKING:
    from .moe import MoeRouting

_LOGGER = get_logger(__name__)


def _mining_disabled() -> bool:
    """Read the process-wide kill switch directly so malformed settings fail closed."""
    return gpu_config.settings.no_mining


def _eager_mining_blocked(state: LayerState, m_tokens: int) -> bool:
    return (
        not state.mineable
        or _mining_disabled()
        or m_tokens < runtime_settings().min_mining_tokens
        or in_graph_setup_no_mining()
        or mining_launches_suspended()
    )


def _eager_bucket(state: LayerState, m_tokens: int) -> int | None:
    """The bucket for one eager forward, or None when it cannot mine."""
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
    """Run the mined forward when this layer can mine right now, else None."""
    if torch.cuda.is_current_stream_capturing():
        # The per-launch host machinery (admission, events, completion
        # callbacks) is illegal inside a CUDA-graph capture; captured graphs
        # always record the job-independent FP8 fallback.
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
    """Mine one MoE layer's first grouped GEMM when it can mine right now.

    ``tail`` receives the permuted ``(cum_m, n_e)`` output and the routing
    that orders it and finishes the layer (activation, down GEMM, combine);
    it runs inside the admitted launch, before the launch's terminal event
    (``pipeline.run_mining_moe_forward``), so lifecycle drain and capture
    quiescence cover the whole mined forward. Returns the tail's result, or
    None -- the caller then serves its own unmined path (a tail OOM opens the
    device cooldown, any other tail failure disables the layer, exactly as
    for the launch itself). The routing is data dependent, so MoE layers
    never mine inside a CUDA graph capture.
    """
    if torch.cuda.is_current_stream_capturing():
        return None
    from .moe import topk_ids_shape_error
    from .pipeline import run_mining_moe_forward

    # A malformed router table declines this call only; it is not a layer
    # failure (``_mine_eager`` disables the layer on any other exception).
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
    """Admit and run one eager launch against the layer's live context."""
    if _eager_mining_blocked(state, m_tokens):
        return None
    bucket = _eager_bucket(state, m_tokens)
    if bucket is None:
        return None

    from .job_prep import current_context, current_job

    try:
        # The producer slot is what CUDA-graph capture waits on; taking it and
        # checking the gate is atomic, so a launch cannot start after capture.
        with gpu_mining_producer() as admitted:
            if not admitted:
                return None
            with mining_attempt(state.weight.device) as attempt:
                if not attempt:
                    return None
                # Prepares B in place on this stream when the job changed.
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


def _try_mine_fp16(state, x2d: torch.Tensor) -> None:
    """Opportunistically run the FP16 (A100/v5) search for one forward.

    FP16 does not fuse mining into serving: the layer's output is a plain linear
    (computed by the caller) and this runs the standalone search as a side
    effect, gated exactly like the FP8 eager path. A non-V5 live job, or no job,
    is a no-op (the manager's admission declines a non-submittable scheme)."""
    if _eager_mining_blocked_fp16(state, x2d.shape[0]):
        return
    from .fp16_mining import run_fp16_mining_forward
    from .job_prep import current_job

    job = current_job()
    if job is None:
        return
    run_fp16_mining_forward(state, job, x2d.contiguous())


def _eager_mining_blocked_fp16(state, m_tokens: int) -> bool:
    return (
        not state.mineable
        or _mining_disabled()
        or m_tokens < runtime_settings().min_mining_tokens
        or in_graph_setup_no_mining()
        or mining_launches_suspended()
    )


def _serve_fp16_linear(
    state, x2d: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """The FP16 layer's serving output: a plain linear against the committed
    FP16 weight, returned in the activation (BF16) dtype."""
    out = torch.nn.functional.linear(x2d.to(state.weight.dtype), state.weight)
    out = out.to(x2d.dtype)
    if bias is not None:
        out = out + bias
    return out


@torch.library.impl(_LIB, "apply_linear", "CUDA")
def _apply_linear_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    _validate_bf16_contract(x, bias)
    x2d = x.reshape(-1, x.shape[-1])

    # FP16 (A100/v5) is a parallel branch probed first: an FP16-registered
    # weight serves a plain linear and mines via the standalone search; an
    # FP8-registered or unregistered weight returns None here and falls through
    # to the FP8 path below, which stays byte-identical to its prior behavior.
    from .fp16_layer import lookup_fp16_state

    fp16_state = lookup_fp16_state(weight)
    if fp16_state is not None:
        if x2d.shape[1] != fp16_state.k or x2d.device != fp16_state.weight.device:
            raise ValueError(
                f"pearl::apply_linear expected (*, {fp16_state.k}) on "
                f"{fp16_state.weight.device}, got {tuple(x2d.shape)} on {x2d.device}"
            )
        _try_mine_fp16(fp16_state, x2d)
        out = _serve_fp16_linear(fp16_state, x2d, bias)
        return out.reshape(*x.shape[:-1], out.shape[-1])

    state = lookup_state(weight)
    if x2d.shape[1] != state.k or x2d.device != state.weight.device:
        raise ValueError(
            f"pearl::apply_linear expected (*, {state.k}) on {state.weight.device}, "
            f"got {tuple(x2d.shape)} on {x2d.device}"
        )

    out = _try_mine(state, x2d, bias)
    if out is None:
        if state.w_fp8 is None or state.w_fp8_scale is None:
            raise ValueError(f"pearl::apply_linear does not serve MoE layer {state.layer_name}")
        out = fp8_fallback_gemm(
            x2d,
            state.w_fp8,
            state.w_fp8_scale,
            bias,
            torch.bfloat16,
        )
    return out.reshape(*x.shape[:-1], out.shape[-1])


@torch.library.register_fake("pearl::apply_linear")
def _apply_linear_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    _validate_bf16_contract(x, bias)
    return torch.empty(
        (*x.shape[:-1], weight.shape[0]),
        dtype=torch.bfloat16,
        device=x.device,
    )
