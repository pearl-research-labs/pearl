"""The ``pearl`` torch.library namespace: one opaque mining-aware linear op.

``pearl::apply_linear`` hides the whole mining decision (the ``MINER_NO_MINING``
kill switch, context lookup, m bucketing, variant readiness, the four-op
pipeline, callback scheduling) from torch.compile / CUDA-graph tracing. When the
layer cannot or must not mine right now it runs the job-independent FP8 fallback
GEMM -- the only path a captured graph ever replays.
"""

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
from .state import LayerState, lookup_state

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
    """The padded bucket for one eager forward, or None when it cannot mine."""
    from .pipeline import pick_bucket, variant_ready

    settings = runtime_settings()
    bucket = pick_bucket(m_tokens)
    if bucket is None:
        _LOGGER.warning(
            f"{m_tokens} tokens exceeds the largest PEARL_M_BUCKETS entry "
            f"({settings.m_buckets[-1]}); such forwards serve without mining.",
        )
        return None
    if settings.warmup_compile and not variant_ready(bucket, state.n, state.k):
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
    m_tokens = x2d.shape[0]
    if _eager_mining_blocked(state, m_tokens):
        return None
    bucket = _eager_bucket(state, m_tokens)
    if bucket is None:
        return None

    from .job_prep import current_context, current_job
    from .pipeline import run_mining_forward

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
                out = run_mining_forward(state, ctx, x2d.contiguous(), bias, bucket)
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
