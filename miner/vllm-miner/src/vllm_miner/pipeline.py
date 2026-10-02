"""Per-forward A-side pipeline: four launch-only stages over the bucketed
activation (``pre_quant -> tensor_hash_plus_stats -> noisy_quant ->
mixed_gemm``), then an async status-check callback on the AsyncLoopManager.
An MoE layer routes the noised rows into expert order and runs
``grouped_mixed_gemm`` over its stacked expert weights instead of
``mixed_gemm``.

Launch shapes come from a small bucket set (bounds CuTe JIT variants).
A-side tensors are fresh per call: the winner callback needs the committed
planes after the forward.
"""

import math
import queue
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from miner_base.async_loop_manager import AsyncLoopManager, MiningLaunchDecision
from miner_base.commitment import Device
from miner_base.devices import local_device
from miner_utils import get_logger

from .capture import (
    mining_launches_suspended,
    mining_producer_admission_open,
    register_mining_completion,
)
from .health import report_device_oom
from .mining_config import (
    LotteryTileSpec,
    effective_work_per_matmul,
    max_mineable_k,
    mining_configuration,
    select_tile,
)
from .mining_state import get_async_manager
from .moe import MoeLaunch, MoeRouting, round_robin_routing, route_tokens
from .settings import runtime_settings
from .state import JobContext, LayerBuffers, LayerState
from .tuning import device_config_name, tuned, with_committed_leaf

if TYPE_CHECKING:
    from miner_base.commitment import MiningConfiguration
    from pearl_gemm import HitSignal


type WinnerCallbackFactory = Callable[..., Callable[[], None]]
# The B-side device operands a launch reads: a published JobContext (mined
# launches) or the layer's steady buffers directly (warmup, which needs no
# job -- the buffers are rewritten in place per job).
type BOperands = JobContext | LayerBuffers
# The serving adapter's work over an MoE launch's permuted ``(cum_m, n_e)``
# output and its routing (activation, down GEMM, combine), run inside the
# launch's terminal fence; its result is the mined forward's result.
type MoeTail = Callable[[torch.Tensor, MoeRouting], object]

_LOGGER = get_logger(__name__)


# Compiled-variant registry, keyed by ``_variant_key``: the bucket and
# the layer's ``(n, k, experts, top_k)``. The MoE fields matter -- a dense
# layer and an MoE layer of the same stacked ``(n, k)`` compile different
# kernels (mixed vs grouped), and the grouped tile follows ``m_bucket * top_k``.
_ready_variants: set[tuple[int, int, int, int, int]] = set()
_failed_variants: set[tuple[int, int, int, int, int]] = set()
_variant_lock = threading.Lock()


_HIT_SIGNALS: dict[int, "HitSignal"] = {}
_hit_signal_lock = threading.Lock()
_fallback_check_lock = threading.Condition()
_fallback_runner_lock = threading.Lock()
_fallback_checks = 0
_fallback_thread: threading.Thread | None = None
_MAX_COMPLETION_QUEUE = 256


class _CompletionLeasePool:
    """Process-wide bound for event-gated credited launch ownership."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._in_use = 0

    def try_acquire(self, limit: int) -> "_CompletionLease | None":
        with self._lock:
            if self._in_use >= limit:
                return None
            self._in_use += 1
        return _CompletionLease(self)

    def release(self) -> None:
        with self._lock:
            if self._in_use <= 0:
                raise RuntimeError("mining completion lease underflow")
            self._in_use -= 1

    def in_use(self) -> int:
        with self._lock:
            return self._in_use


class _CompletionLease:
    def __init__(self, pool: _CompletionLeasePool) -> None:
        self._pool = pool
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._pool.release()


_COMPLETION_LEASES = _CompletionLeasePool()


def fallback_winner_checks_idle() -> bool:
    """Drain predicate for event-gated accounting and winner continuations."""
    with _fallback_check_lock:
        return _fallback_checks == 0


def wait_for_mining_continuations_idle(timeout: float) -> bool:
    """Bound lifecycle mutation on every event-gated host continuation.

    The legacy ``_fallback_checks`` counter covers the single completion queue
    used by ordinary manager-backed launches and publication fallbacks alike,
    including credited accounting, winner inspection, and release callbacks.
    """
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError(f"timeout must be finite and non-negative, got {timeout!r}")
    with _fallback_check_lock:
        return _fallback_check_lock.wait_for(lambda: _fallback_checks == 0, timeout)


class _FallbackCompletion:
    def __init__(self) -> None:
        self.done = threading.Event()

    def query(self) -> bool:
        return self.done.is_set()


@dataclass(frozen=True)
class _FallbackWork:
    event: torch.cuda.Event
    callback: Callable[[], None]
    release: Callable[[], None]
    completion: _FallbackCompletion
    manager: AsyncLoopManager | None = None
    # An int, or a callable read after ``event`` completed (MoE launches copy
    # their per-expert counts to the host on the launch stream).
    credited_hashes: int | Callable[[], int] = 0
    state: LayerState | None = None
    device: torch.device | None = None
    completion_lease: _CompletionLease | None = None


_fallback_queue: queue.Queue[_FallbackWork] = queue.Queue(maxsize=_MAX_COMPLETION_QUEUE)


def _noop() -> None:
    pass


def _run_fallback_work(work: _FallbackWork) -> None:
    global _fallback_checks
    callback_entered = False
    try:
        work.event.synchronize()
        credited = work.credited_hashes
        if callable(credited):
            credited = credited()
        if work.manager is not None and credited:
            work.manager.increment_credited_hashes(credited)
        callback_entered = True
        work.callback()
    except BaseException as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError) and work.device is not None:
            report_device_oom(work.device)
        elif work.state is not None:
            work.state.disable_mining(
                f"asynchronous launch completion failed ({type(exc).__name__})"
            )
        _LOGGER.opt(exception=True).warning("mining completion callback crashed")
    finally:
        try:
            if not callback_entered:
                work.release()
        except BaseException:
            _LOGGER.opt(exception=True).warning("mining completion release callback crashed")
        finally:
            if work.completion_lease is not None:
                work.completion_lease.release()
            with _fallback_check_lock:
                _fallback_checks -= 1
                _fallback_check_lock.notify_all()
            work.completion.done.set()


def _fallback_worker() -> None:
    while True:
        _run_fallback_work(_fallback_queue.get())


def _ensure_fallback_runner_started() -> None:
    """Start the lifecycle-owned fallback worker before any retained launch."""
    global _fallback_thread
    with _fallback_runner_lock:
        if _fallback_thread is not None:
            if not _fallback_thread.is_alive():
                raise RuntimeError("mining completion worker stopped unexpectedly")
            return
        thread = threading.Thread(
            target=_fallback_worker,
            name="mining-completion",
            daemon=True,
        )
        thread.start()
        _fallback_thread = thread


def _start_launch_completion(
    event: torch.cuda.Event,
    callback: Callable[[], None],
    release: Callable[[], None],
    *,
    manager: AsyncLoopManager | None = None,
    credited_hashes: int | Callable[[], int] = 0,
    state: LayerState | None = None,
    device: torch.device | None = None,
    completion_lease: _CompletionLease | None = None,
) -> None:
    """Enqueue completion-gated accounting and optional winner inspection."""
    global _fallback_checks
    with _fallback_check_lock:
        _fallback_checks += 1
    completion = _FallbackCompletion()
    work = _FallbackWork(
        event=event,
        callback=callback,
        release=release,
        completion=completion,
        manager=manager,
        credited_hashes=credited_hashes,
        state=state,
        device=device,
        completion_lease=completion_lease,
    )
    try:
        # Capture quiescence waits this host-side continuation as well as the
        # CUDA event, so its D2H/reset operations cannot overlap graph capture.
        register_mining_completion(completion)
        _fallback_queue.put_nowait(work)
    except BaseException as exc:
        _LOGGER.warning(
            f"mining completion publication failed ({type(exc).__name__}); running synchronously"
        )
        _run_fallback_work(work)


@lru_cache(maxsize=32)
def _hit_signal_capacity(
    device_index: int,
    m_buckets: tuple[int, ...],
) -> tuple[int, int]:
    # Registration is framework-driven and can be incremental. Size for the
    # complete legal domain so first use cannot freeze a timing-dependent
    # subset of layers into the persistent buffer.
    del device_index  # device-keyed cache: future capabilities may differ
    return max(m_buckets), max_mineable_k()


def _hit_signal_for(device: torch.device) -> "HitSignal":
    """One persistent signal per device, with a lock-free primed lookup."""
    from pearl_gemm import HitSignal, HitSignalConfig

    index = device.index if device.index is not None else torch.cuda.current_device()
    max_m, max_k = _hit_signal_capacity(index, runtime_settings().m_buckets)
    signal = _HIT_SIGNALS.get(index)
    if signal is not None:
        if signal.cfg.max_m < max_m or signal.cfg.max_k < max_k:
            raise RuntimeError(
                f"mining hit signal capacity={signal.cfg} is below "
                f"configured capacity=({max_m}, {max_k})"
            )
        return signal

    with _hit_signal_lock:
        signal = _HIT_SIGNALS.get(index)
        if signal is None:
            # Model loading may run under inference mode; the consumer later
            # mutates these persistent buffers from its own thread.
            with torch.inference_mode(False):
                signal = HitSignal(HitSignalConfig(max_m=max_m, max_k=max_k), device=device)
            _HIT_SIGNALS[index] = signal
        return signal


class WinnerCheckLeases:
    """Bound admitted launches awaiting persistent-signal inspection.

    Each retained launch needs one completion callback so the process-wide
    signal is eventually inspected and re-armed. A callback then owns the
    snapshotted A planes through CPU validation, so the same bound caps both
    queued checks and their host-memory backlog. Mining that cannot be checked
    is not launched at all.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held = 0

    def try_acquire(self, limit: int) -> bool:
        with self._lock:
            if self._held >= limit:
                return False
            self._held += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._held <= 0:
                raise RuntimeError("winner-check lease released without an owner")
            self._held -= 1

    def held(self) -> int:
        with self._lock:
            return self._held


WINNER_LEASES = WinnerCheckLeases()


def pick_bucket(m_tokens: int) -> int | None:
    """Smallest configured bucket that fits ``m_tokens`` (None: don't mine)."""
    for bucket in runtime_settings().m_buckets:
        if m_tokens <= bucket:
            return bucket
    return None


def _lottery_tile(state: LayerState) -> LotteryTileSpec:
    """The committed tile of this layer's lottery lattice (fixed at creation)."""
    tile = select_tile(state.lottery_n, state.k, device=state.committed_device)
    if tile is None:
        raise RuntimeError(
            f"{state.layer_name}: no lottery tile for n_e={state.lottery_n}, k={state.k}"
        )
    return tile


def _launch_credit(m_bucket: int, state: LayerState) -> int:
    """Protocol credit for one dense launch against this layer.

    Difficulty-normalized effective work (m*n*k): the only crediting unit, so
    the gateway's total_hash/hashrate is comparable across shapes/k and to
    network difficulty. effective_work_per_matmul derives the committed tile
    from (n, k), so a tall-tile (16x32) layer is credited against that tile.
    """
    return effective_work_per_matmul(m_bucket, state.n, state.k, device=state.committed_device)


def moe_launch_credit(state: LayerState, expert_rows: Sequence[int]) -> int:
    """Protocol credit for one MoE launch from its per-expert row counts.

    The launch mines ``cum_m`` permuted rows against one ``n_e``-wide expert
    each. Its lattice restarts at every expert and an expert's trailing
    partial tile is never published, so the credited rows are the whole
    tiles of every expert.
    """
    tile = _lottery_tile(state)
    rows = sum(int(count) // tile.rows for count in expert_rows) * tile.rows
    if not rows:
        return 0
    return effective_work_per_matmul(rows, state.lottery_n, state.k, device=state.committed_device)


class _DeferredMoeCredit:
    """The exact credit of one MoE launch, readable once its event completed.

    The per-expert counts live on the device; a host read on the serving
    thread would stall the forward. This copies them into pinned memory on
    the launch stream and lets the event-gated completion worker reduce them.
    """

    def __init__(self, state: LayerState, m_valid: torch.Tensor) -> None:
        self.state = state
        self.counts = torch.empty_like(m_valid, device="cpu", pin_memory=True)
        self.counts.copy_(m_valid, non_blocking=True)

    def __call__(self) -> int:
        return moe_launch_credit(self.state, self.counts.tolist())


def _variant_key(m_bucket: int, state: LayerState) -> tuple[int, int, int, int, int]:
    return (m_bucket, state.n, state.k, state.experts, state.top_k)


def variant_ready(m_bucket: int, state: LayerState) -> bool:
    with _variant_lock:
        return _variant_key(m_bucket, state) in _ready_variants


def _mixed_gemm_record_is_legal(m: int, n: int, k: int, kwargs: dict, *, device: Device) -> bool:
    """Whether one saved record preserves protocol geometry and can launch."""
    from pearl_gemm import default_mixed_gemm_config, validate_mixed_gemm_config

    tile = select_tile(n, k, device=device)
    if tile is None:
        return False
    if kwargs.get("ltile_cols", tile.cols) != tile.cols:
        return False
    if kwargs.get("ltile_rows", tile.rows) != tile.rows:
        return False
    try:
        config = replace(
            default_mixed_gemm_config(),
            **{**kwargs, "ltile_cols": tile.cols, "ltile_rows": tile.rows},
        )
        validate_mixed_gemm_config(m, n, k, config)
    except (TypeError, ValueError):
        return False
    return True


@lru_cache(maxsize=256)
def _configs(config_name: str, m: int, n: int, k: int, *, device: Device | None = None):
    """Resolve and validate one device/shape pipeline configuration.

    ``device`` is the committed device the lottery tile is selected for (the
    current CUDA device's when omitted); the kernel configs are validated
    against the current device's family."""
    from pearl_gemm import (
        PreQuantConfig,
        TensorHashConfig,
        default_mixed_gemm_config,
        default_noisy_quant_config,
        tensor_hash_plus_stats_record_is_legal,
        validate_mixed_gemm_config,
        validate_noisy_quant_config,
    )
    from pearl_gemm.autotune import route_small_m_mixed_gemm

    if device is None:
        device = local_device()
    tile = select_tile(n, k, device=device)
    if tile is None:
        raise ValueError(f"({n}, {k}) is not mineable by any committed lottery tile")

    prequant = PreQuantConfig(**tuned(config_name, "pre_quant", m=m, k=k))
    # Overlay the cert-v4 1024-byte leaf on autotune knobs. Every m bucket
    # of the layer must build the same A tree; the verifier can only open 1024.
    committed_leaf = mining_configuration(k, n, device=device).a_chunk_size
    commit = TensorHashConfig(
        **with_committed_leaf(
            tuned(
                config_name,
                "tensor_hash_plus_stats",
                legal=tensor_hash_plus_stats_record_is_legal,
                m=m,
                k=k,
            ),
            committed_leaf,
        )
    )
    # Records overlay the device family's library defaults (see _b_configs).
    prepare = replace(default_noisy_quant_config(), **tuned(config_name, "noisy_quant", m=m, k=k))
    validate_noisy_quant_config(m, k, prepare)

    def gemm_legal(kwargs: dict) -> bool:
        return _mixed_gemm_record_is_legal(m, n, k, kwargs, device=device)

    # Exact autotune records win: they were measured on this (m, n, k). The
    # 64-row decode heuristic only fills the gap when no exact record exists
    # (nearest-shape 128/256-row records are the thing it is meant to beat).
    gemm_kwargs = tuned(
        config_name,
        "mixed_gemm",
        legal=gemm_legal,
        exact=True,
        m=m,
        n=n,
        k=k,
    )
    if not gemm_kwargs:
        gemm_kwargs = route_small_m_mixed_gemm(m, n, k, ltile_rows=tile.rows, ltile_cols=tile.cols)
    if gemm_kwargs is None:
        gemm_kwargs = tuned(
            config_name,
            "mixed_gemm",
            legal=gemm_legal,
            require_legal=True,
            m=m,
            n=n,
            k=k,
        )
    gemm = replace(
        default_mixed_gemm_config(),
        **{**gemm_kwargs, "ltile_cols": tile.cols, "ltile_rows": tile.rows},
    )
    # Saved records are filtered first, but this public validator remains the
    # final safety boundary for both tuned and default configurations.
    validate_mixed_gemm_config(m, n, k, gemm)
    return prequant, commit, prepare, gemm


def _launch_stages(
    ctx: BOperands,
    config: "MiningConfiguration",
    a_launch: torch.Tensor,
    hit_signal: "HitSignal",
    *,
    layer_id: int,
    record_hits: bool,
    routing: MoeRouting | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch every stage (no D2H sync); returns ``(codes, scales, a_keys, c)``.

    ``a_keys`` is the finalize's 96-byte ``seedA || noise-line keyA || jackpot
    key`` (``miner_base.commitment_hash.AKeys``). Every B-side operand, ``F_A``
    and the complete B peel included, is a job constant read from ``ctx``.

    ``routing`` (MoE) binds ``HR || HO`` into ``seedA``, then gathers the noised
    rows into expert order and runs ``grouped_mixed_gemm`` against the stacked
    ``B'`` viewed per expert; ``c`` is then the permuted ``(cum_m, n_e)``
    output. ``codes`` / ``scales`` stay the unpermuted committed planes.
    """
    from pearl_gemm import (
        R,
        mixed_gemm,
        noisy_quant,
        pre_quant,
        pre_quant_output_shapes,
        tensor_hash_plus_stats,
        tensor_hash_workspace_bytes,
    )

    device = a_launch.device
    m, k = a_launch.shape
    n = ctx.b_prime.shape[0]
    experts = config.experts
    if (routing is None) != (experts == 0):
        raise ValueError("MoE layers launch with routing, dense layers without")
    lottery_n = n // experts if experts else n
    prequant_cfg, commit_cfg, prepare_cfg, gemm_cfg = _configs(
        device_config_name(device),
        m,
        lottery_n,
        k,
        device=config.device,
    )

    codes_shape, scales_shape = pre_quant_output_shapes(m, k)
    codes = torch.empty(codes_shape, dtype=torch.int8, device=device)
    scales = torch.empty(scales_shape, dtype=torch.bfloat16, device=device)
    pre_quant(a_launch, codes, scales, config=prequant_cfg)

    root_codes = torch.zeros(32, dtype=torch.uint8, device=device)
    root_scales = torch.zeros(32, dtype=torch.uint8, device=device)
    roots = torch.zeros(
        tensor_hash_workspace_bytes(m, k, commit_cfg), dtype=torch.uint8, device=device
    )
    a_keys = torch.zeros(96, dtype=torch.uint8, device=device)
    stats = torch.zeros(2 * (m * k // 512), dtype=torch.float32, device=device)
    tensor_hash_plus_stats(
        codes,
        scales,
        ctx.key_a_dev,
        ctx.seed_b_dev,
        root_codes,
        root_scales,
        roots,
        a_keys,
        stats,
        p_a=config.p_a(m),
        config=commit_cfg,
        routing_commitments=None if routing is None else routing.commitments,
    )
    noise_key_a = a_keys[32:64]
    pow_key = a_keys[64:96]

    alpha_a = torch.empty(m, dtype=torch.bfloat16, device=device)
    beta_a = torch.empty_like(alpha_a)
    e1 = torch.empty(m, R, dtype=torch.float8_e4m3fn, device=device)
    a_prime = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=device)
    a_peel = torch.empty(m, 2 * R, dtype=torch.bfloat16, device=device)
    noisy_quant(
        codes,
        scales,
        noise_key_a,
        stats,
        ctx.f1_hl,
        ctx.f2,
        alpha_a,
        beta_a,
        e1,
        a_prime,
        a_peel,
        config=prepare_cfg,
    )

    rows = m if routing is None else routing.cum_m
    c = torch.empty(rows, lottery_n, dtype=torch.bfloat16, device=device)
    if routing is not None:
        _grouped_mixed_gemm(
            ctx,
            routing,
            a_prime,
            a_peel,
            alpha_a,
            pow_key,
            c,
            hit_signal,
            m,
            committed_device=config.device,
            layer_id=layer_id,
            record_hits=record_hits,
        )
        return codes, scales, a_keys, c
    mixed_gemm(
        a_prime,
        ctx.b_prime,
        a_peel,
        ctx.b_peel,
        alpha_a,
        ctx.inv_alpha_b,
        pow_key,
        ctx.threshold_dev,
        c,
        hit_signal,
        codes,
        scales,
        ctx.seed_b_dev,
        config=gemm_cfg,
        layer_id=layer_id,
        record_hits=record_hits,
    )
    return codes, scales, a_keys, c


def _grouped_mixed_gemm(
    ctx: BOperands,
    routing: MoeRouting,
    a_prime: torch.Tensor,
    a_peel: torch.Tensor,
    alpha_a: torch.Tensor,
    pow_key: torch.Tensor,
    c: torch.Tensor,
    hit_signal: "HitSignal",
    m_bucket: int,
    *,
    committed_device: Device,
    layer_id: int,
    record_hits: bool,
) -> None:
    """The MoE tail of ``_launch_stages``: gather the dense A-side rows into
    expert order (``routing.tokens``) and run ``grouped_mixed_gemm`` over the
    stacked ``B'``, ``b_peel`` and ``inv_alpha_b`` viewed as
    ``(experts, n_e, ...)``. The tile shape follows the bucket's
    ``m_bucket * top_k`` rows, not this launch's ``cum_m``, so the JIT variant
    is fixed per bucket (and pre-compiled by warmup).

    Hits publish payload-less: an MoE proof opens the retained *unpermuted*
    committed planes at the tile's global token indices (``MoeLaunch``), so
    the committed planes are neither gathered nor snapshotted here."""
    from pearl_gemm import GroupedMixedGemmConfig, grouped_mixed_gemm

    experts, n_e, k = routing.experts, ctx.b_prime.shape[0] // routing.experts, a_prime.shape[1]
    tile = select_tile(n_e, k, device=committed_device)
    if tile is None:
        raise RuntimeError(f"no lottery tile for n_e={n_e}, k={k}")
    rows = routing.tokens

    def gather(t: torch.Tensor) -> torch.Tensor:
        return t.index_select(0, rows)

    grouped_mixed_gemm(
        # index_select has no e4m3 kernel; the bytes are the same.
        gather(a_prime.view(torch.uint8)).view(torch.float8_e4m3fn),
        ctx.b_prime.view(experts, n_e, k),
        gather(a_peel),
        ctx.b_peel.view(experts, n_e, -1),
        gather(alpha_a),
        ctx.inv_alpha_b.view(experts, n_e),
        pow_key,
        ctx.threshold_dev,
        c,
        routing.m_indptr,
        hit_signal,
        None,
        None,
        ctx.seed_b_dev,
        routing.m_valid,
        config=GroupedMixedGemmConfig.auto(
            m_bucket * routing.top_k,
            experts,
            n=n_e,
            device=a_prime.device,
            ltile_rows=tile.rows,
            ltile_cols=tile.cols,
        ),
        layer_id=layer_id,
        record_hits=record_hits,
    )


def _record_stream_event() -> torch.cuda.Event:
    event = torch.cuda.Event()
    event.record()
    return event


def _register_launch_event(event: torch.cuda.Event) -> None:
    """Publish one launch fence, synchronizing it on registry failure."""
    try:
        register_mining_completion(event)
    except BaseException as exc:
        # The producer slot is still held. Waiting this exact event keeps
        # capture safe without a device-wide synchronization, even if Python
        # bookkeeping failed.
        event.synchronize()
        _LOGGER.warning(
            f"mining completion registration failed ({type(exc).__name__}); "
            "launch completed synchronously"
        )


def _synchronize_launch_stream(device: torch.device) -> None:
    """Last-resort fence when event construction/recording itself failed."""
    torch.cuda.current_stream(device).synchronize()


def mine_launch(
    state: LayerState,
    ctx: JobContext,
    m_bucket: int,
    m_tokens: int,
    fill: Callable[[torch.Tensor], object],
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Enqueue and credit one dense launch while its captured job stays current.

    Returns the launch's ``(m_bucket, n)`` output, or None when declined.
    """
    if state.experts:
        raise ValueError(f"MoE layer {state.layer_name} launches through mine_moe_launch")
    launched = _mine_launch(state, ctx, m_bucket, m_tokens, fill, bias=bias)
    return None if launched is None else launched.output


def mine_moe_launch(
    state: LayerState,
    ctx: JobContext,
    m_bucket: int,
    m_tokens: int,
    fill: Callable[[torch.Tensor], object],
    topk_ids: torch.Tensor,
    tail: MoeTail | None = None,
) -> "_MinedLaunch | None":
    """Enqueue and credit one MoE launch over the router's ``(m_tokens, top_k)``
    expert ids while its captured job stays current.

    Returns the permuted ``(cum_m, n_e)`` expert output with the routing that
    orders it (and ``tail``'s result), or None when declined. The routing
    (sort, commitments) is enqueued only once the launch is admitted and
    leased, inside the launch's completion fence -- as is ``tail``, the
    serving adapter's mining-specific work over the output (see
    :func:`run_mining_moe_forward`).
    """
    if not state.experts:
        raise ValueError(f"dense layer {state.layer_name} launches through mine_launch")
    return _mine_launch(
        state, ctx, m_bucket, m_tokens, fill, bias=None, topk_ids=topk_ids, tail=tail
    )


@dataclass(frozen=True)
class _MinedLaunch:
    """One admitted, enqueued launch: its output, for MoE layers the routing
    that orders it, and the ``tail``'s result when the caller supplied one."""

    output: torch.Tensor
    routing: MoeRouting | None
    result: object = None


def _mine_launch(
    state: LayerState,
    ctx: JobContext,
    m_bucket: int,
    m_tokens: int,
    fill: Callable[[torch.Tensor], object],
    *,
    bias: torch.Tensor | None,
    topk_ids: torch.Tensor | None = None,
    tail: MoeTail | None = None,
) -> _MinedLaunch | None:
    """Admission and launch shared by both public entry points (None: declined)."""
    if mining_launches_suspended():
        return None
    manager = get_async_manager()
    with manager.mining_launch_admission(ctx.job) as decision:
        if not decision.launch:
            return None
        return _mine_admitted_launch(
            manager,
            decision,
            state,
            ctx,
            m_bucket,
            m_tokens,
            fill,
            bias=bias,
            topk_ids=topk_ids,
            tail=tail,
        )


def _routing_for_launch(
    state: LayerState, ctx: JobContext, topk_ids: torch.Tensor | None
) -> MoeRouting | None:
    """Enqueue the routing of one admitted launch (None for dense layers)."""
    if not state.experts:
        if topk_ids is not None:
            raise ValueError(f"dense layer {state.layer_name} takes no topk_ids")
        return None
    if topk_ids is None:
        raise ValueError(f"MoE layer {state.layer_name} launches with the router's topk_ids")
    return route_tokens(topk_ids, state.experts, ctx.key_a_dev)


def _acquire_launch_leases(
    decision: MiningLaunchDecision,
) -> tuple[_CompletionLease, bool, WinnerCallbackFactory | None] | None:
    """Reserve all host continuation capacity before GPU allocation."""
    winner_callback_factory = None
    if decision.retain_winner:
        # Prepare every fallible callback dependency before enabling hit
        # publication; the later import is then a cache lookup only.
        from .winners import WinnerCheckCallback

        winner_callback_factory = WinnerCheckCallback
    completion_lease = _COMPLETION_LEASES.try_acquire(runtime_settings().completion_inflight_limit)
    if completion_lease is None:
        return None
    if decision.retain_winner and not WINNER_LEASES.try_acquire(
        runtime_settings().winner_check_inflight_limit
    ):
        completion_lease.release()
        return None
    return completion_lease, decision.retain_winner, winner_callback_factory


def _pow_key_host_buffer() -> torch.Tensor:
    """Pinned target for the launch's jackpot key (``a_keys[64:96]``).

    Allocated before the publisher is enqueued so the retained launch's only
    post-publication host allocation cannot fail; the copy itself rides the
    launch stream and is complete once the launch's event is."""
    return torch.empty(32, dtype=torch.uint8, pin_memory=True)


@dataclass
class _OwnedRecordDiscard:
    """Consume, and drop, the record a launch may have published when its
    winner continuation could not be installed.

    A retained launch enables publication the moment its kernel is enqueued;
    if the host work between that and ``_start_launch_completion`` fails (an
    MoE serving tail included), no consumer owns its record. Every winner
    check leaves a record under a foreign key in place by design, so an
    unowned record would keep the device-wide latch closed for every later
    winner. This event-gated continuation takes only the failed launch's own
    record (``take_owned_hit`` under its ``pow_key``) and releases the winner
    lease it inherits; another launch's live record is left for its owner.
    """

    signal: "HitSignal"
    # The launch's device-side keys, complete once the gating event is; the
    # pinned copy is preferred when it was enqueued before the failure.
    a_keys: torch.Tensor
    pow_key_host: torch.Tensor | None
    leases: WinnerCheckLeases

    def __call__(self) -> None:
        try:
            if self.pow_key_host is not None:
                owner = bytes(self.pow_key_host.tolist())
            else:
                owner = bytes(self.a_keys[64:96].cpu().tolist())
            if self.signal.take_owned_hit(owner) is not None:
                _LOGGER.warning("dropped the hit record of a launch whose continuation failed")
        except Exception as exc:
            from pearl_gemm import HitSignalPoisonedError

            if isinstance(exc, HitSignalPoisonedError):
                from .winners import disable_mining_after_poisoning

                disable_mining_after_poisoning(self.signal)
            else:
                _LOGGER.opt(exception=True).warning(
                    "failed launch's hit record could not be inspected"
                )
        finally:
            self.leases.release()


def _winner_continuation(
    *,
    retained: bool,
    callback_factory: WinnerCallbackFactory | None,
    hit_signal: "HitSignal",
    event: torch.cuda.Event,
    manager: AsyncLoopManager,
    pow_key_host: torch.Tensor | None,
    moe: MoeLaunch | None,
) -> tuple[Callable[[], None], Callable[[], None]]:
    if not retained:
        return _noop, _noop
    assert callback_factory is not None and pow_key_host is not None
    callback: Callable[[], None] = callback_factory(
        signal=hit_signal,
        event=event,
        leases=WINNER_LEASES,
        manager=manager,
        pow_key_host=pow_key_host,
        moe=moe,
    )
    return callback, WINNER_LEASES.release


def _check_bias(bias: torch.Tensor | None, state: LayerState) -> None:
    if bias is not None and state.experts:
        raise ValueError("an MoE launch returns the permuted expert output; add bias per expert")


def _mine_admitted_launch(
    manager: AsyncLoopManager,
    decision: MiningLaunchDecision,
    state: LayerState,
    ctx: JobContext,
    m_bucket: int,
    m_tokens: int,
    fill: Callable[[torch.Tensor], object],
    *,
    bias: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    tail: MoeTail | None = None,
) -> _MinedLaunch | None:
    """Own completion/accounting and winner retention for one enqueue.

    Returns the launch, or None when no lease was free. ``tail`` (MoE
    serving) runs over the permuted output and its routing inside the
    launch's terminal fence: its result is the launch's ``result``, and its
    failure is the launch's failure (no credit, cleanup fenced like every
    other escape).

    Ordering. The publisher is enabled the moment ``_launch_stages`` returns
    a retained launch, so every host operation that can fail is either
    performed before it (the pinned key and credit buffers) or, on failure
    after it, compensated by an event-gated owner-selective discard of this
    launch's own record (``_OwnedRecordDiscard``) so the shared latch never
    stays closed behind a record nobody owns.
    """
    # Every credited launch, retained or not, needs the lifecycle completion
    # worker. Start it before any GPU allocation so thread exhaustion fails
    # without leaving unowned kernels.
    _ensure_fallback_runner_started()
    _check_bias(bias, state)
    leases = _acquire_launch_leases(decision)
    if leases is None:
        return None
    completion_lease, leased, winner_callback_factory = leases
    device = state.weight.device

    signal_event: torch.cuda.Event | None = None
    work_may_be_enqueued = False
    completion_published = False
    # Set once a retained launch's publisher is enqueued: ``(hit_signal,
    # a_keys, pinned key copy or None until it is enqueued)``.
    published: tuple[HitSignal, torch.Tensor, torch.Tensor | None] | None = None
    try:
        pow_key_host = _pow_key_host_buffer() if leased else None
        a_launch = torch.empty(
            m_bucket,
            state.k,
            dtype=torch.bfloat16,
            device=device,
        )
        # The routing and ``fill`` may partially enqueue before raising. From
        # this point onward every escape must publish an exact completion or
        # fence this stream.
        work_may_be_enqueued = True
        routing = _routing_for_launch(state, ctx, topk_ids)
        # An MoE launch's credit depends on the per-expert counts; their
        # pinned host copy rides the launch stream ahead of the completion
        # event (allocated here, before the publisher is enabled).
        moe_credit = None if routing is None else _DeferredMoeCredit(state, routing.m_valid)
        # Live serving rows are written exactly once.
        fill(a_launch[:m_tokens])
        if m_tokens < m_bucket:
            a_launch[m_tokens:].zero_()
        hit_signal = _hit_signal_for(device)
        codes, scales, a_keys, c = _launch_stages(
            ctx,
            ctx.config,
            a_launch,
            hit_signal,
            layer_id=state.layer_id,
            record_hits=leased,
            routing=routing,
        )
        if pow_key_host is not None:
            published = (hit_signal, a_keys, None)
            # The winner check binds the record to this launch through its
            # jackpot key; the host copy rides the launch stream ahead of the
            # completion event.
            pow_key_host.copy_(a_keys[64:96], non_blocking=True)
            published = (hit_signal, a_keys, pow_key_host)
        if bias is not None:
            # Keep this serving operation inside the launch's exact terminal
            # fence. In-place addition avoids a second m*n allocation and makes
            # completion-lease memory accounting cover the whole mined path.
            c[:m_tokens].add_(bias)
        # The adapter's mining-specific tail (activation, down GEMM, combine)
        # belongs to this launch: it completes before the terminal event
        # below, so drain/capture quiescence covers it.
        result = _run_launch_tail(tail, c, routing)
        credit: int | Callable[[], int] = (
            _launch_credit(m_bucket, state) if moe_credit is None else moe_credit
        )

        signal_event = _record_stream_event()
        _register_launch_event(signal_event)
        callback, release = _winner_continuation(
            retained=leased,
            callback_factory=winner_callback_factory,
            hit_signal=hit_signal,
            event=signal_event,
            manager=manager,
            pow_key_host=pow_key_host,
            moe=None if routing is None else MoeLaunch(routing, codes, scales),
        )
        # The worker proves event success before crediting. Its host token also
        # keeps capture/drain closed through rare winner D2H/reset/proof work.
        _start_launch_completion(
            signal_event,
            callback,
            release,
            manager=manager,
            credited_hashes=credit,
            state=state,
            device=device,
            completion_lease=completion_lease,
        )
        completion_lease = None  # event-gated worker now owns this launch slot
        leased = False  # the event-gated callback now owns the winner lease, if any
        completion_published = True
        return _MinedLaunch(c, routing, result)
    finally:
        try:
            if (
                work_may_be_enqueued
                and not completion_published
                and _fence_abandoned_launch(state, published, completion_lease)
            ):
                leased = False  # both leases now belong to the discard
                completion_lease = None
        finally:
            if leased:
                WINNER_LEASES.release()
            if completion_lease is not None:
                completion_lease.release()


def _run_launch_tail(
    tail: MoeTail | None, output: torch.Tensor, routing: MoeRouting | None
) -> object:
    """Run the serving adapter's tail over an MoE launch's output, if given."""
    if tail is None:
        return None
    if routing is None:
        raise ValueError("a launch tail needs an MoE launch")
    return tail(output, routing)


def _fence_abandoned_launch(
    state: LayerState,
    published: tuple["HitSignal", torch.Tensor, torch.Tensor | None] | None,
    completion_lease: _CompletionLease | None,
) -> bool:
    """Fence a launch that raised after it may have enqueued work.

    Prefer another exact event; if event creation/record also fails,
    synchronize only this launch stream before returning. When a retained
    launch's publisher ran (``published``: signal, device keys, pinned key
    copy if enqueued) no consumer owns its record, so both leases are handed
    to an owner-selective discard gated on the same fence, or the discard
    runs in place on an already synchronized stream. Returns whether the
    leases were taken over. The failed launch is not credited.
    """
    device = state.weight.device
    cleanup_event: torch.cuda.Event | None = None
    try:
        cleanup_event = _record_stream_event()
        _register_launch_event(cleanup_event)
    except BaseException:
        cleanup_event = None
        _synchronize_launch_stream(device)
    if published is None:
        return False
    signal, a_keys, key_copy = published
    discard = _OwnedRecordDiscard(signal, a_keys, key_copy, WINNER_LEASES)
    if cleanup_event is None:
        # The discard never raises and releases the winner lease itself.
        discard()
        if completion_lease is not None:
            completion_lease.release()
        return True
    _start_launch_completion(
        cleanup_event,
        discard,
        WINNER_LEASES.release,
        state=state,
        device=device,
        completion_lease=completion_lease,
    )
    return True


def run_mining_forward(
    state: LayerState,
    ctx: JobContext,
    x2d: torch.Tensor,
    bias: torch.Tensor | None,
    m_bucket: int,
) -> torch.Tensor | None:
    """Mine one forward; returns ``C''[:m_tokens] (+ bias)`` in BF16, or None
    when the launch was declined (winner-check retention bound)."""
    if x2d.dtype != torch.bfloat16:
        raise RuntimeError(f"pearl::apply_linear requires BF16 input, got {x2d.dtype}")
    if bias is not None and bias.dtype != torch.bfloat16:
        raise RuntimeError(f"pearl::apply_linear requires BF16 bias, got {bias.dtype}")
    m_tokens = x2d.shape[0]
    c = mine_launch(
        state,
        ctx,
        m_bucket,
        m_tokens,
        lambda a: a[:m_tokens].copy_(x2d),
        bias=bias,
    )
    return None if c is None else c[:m_tokens]


def run_mining_moe_forward(
    state: LayerState,
    ctx: JobContext,
    x2d: torch.Tensor,
    topk_ids: torch.Tensor,
    m_bucket: int,
    tail: MoeTail | None = None,
) -> object | None:
    """Mine one MoE forward: the first grouped GEMM of every routed
    ``(token, expert)`` pair.

    The launch produces the ``(cum_m, n_e)`` output in the canonical
    expert-major, token-ascending row order together with the routing that
    defines it (``routing.slots`` maps rows back to the router's ``(token,
    slot)``). With ``tail`` -- the serving adapter's activation, down GEMM
    and combine over that pair -- the tail runs inside the admitted launch,
    before its terminal event, and its result is returned: drain/capture
    quiescence then covers the whole mined forward, and a tail failure is
    the launch's failure (fenced cleanup, no credit; an OOM opens the
    device cooldown like any launch OOM). Without ``tail`` the raw
    ``(output, routing)`` pair is returned (test and diagnostic use; a
    serving adapter must pass its tail). None when the launch was declined.
    """
    if x2d.dtype != torch.bfloat16:
        raise RuntimeError(f"pearl::apply_moe requires BF16 input, got {x2d.dtype}")
    m_tokens = x2d.shape[0]
    launched = mine_moe_launch(
        state, ctx, m_bucket, m_tokens, lambda a: a[:m_tokens].copy_(x2d), topk_ids, tail=tail
    )
    if launched is None:
        return None
    if launched.routing is None:
        raise RuntimeError("an admitted MoE serving launch emits its routing")
    if tail is None:
        return launched.output, launched.routing
    return launched.result


def warmup_layer_variants(
    state: LayerState,
    cancelled: Callable[[], bool] | None = None,
) -> bool:
    """Compile pipeline variants for every m bucket of this layer shape.

    Real launches with hit publication disabled run against the layer's
    steady buffers (whatever they hold; no job is needed) before serving
    starts. Successful variants unlock mining.
    """
    assert state.buffers is not None
    device = state.weight.device
    config = mining_configuration(state.k, state.n, state.experts, device=state.committed_device)
    for m_bucket in runtime_settings().m_buckets:
        if (cancelled is not None and cancelled()) or not mining_producer_admission_open():
            return False
        variant = _variant_key(m_bucket, state)
        with _variant_lock:
            if variant in _ready_variants or variant in _failed_variants:
                continue
        try:
            dummy = torch.zeros(m_bucket, state.k, dtype=torch.bfloat16, device=device)
            routing = None
            if state.experts:
                routing = round_robin_routing(
                    m_bucket, state.experts, state.top_k, state.buffers.key_a_dev
                )
            # The variant must be warm before the shape is advertised as
            # ready: the first mined serving forward would otherwise pay a
            # synchronous JIT that stalls serving (and a compile failure would
            # surface after a "successful" warmup).
            _launch_stages(
                state.buffers,
                config,
                dummy,
                _hit_signal_for(device),
                layer_id=state.layer_id,
                record_hits=False,
                routing=routing,
            )
            _record_stream_event().synchronize()
            if (cancelled is not None and cancelled()) or not mining_producer_admission_open():
                return False
        except torch.cuda.OutOfMemoryError:
            # A failed launch may still have queued work. Quiesce this exact
            # stream, then let the shared device breaker open its cooldown.
            _record_stream_event().synchronize()
            raise
        except Exception:
            # Publish failure only after any partial launch is complete.
            _record_stream_event().synchronize()
            with _variant_lock:
                _failed_variants.add(variant)
            _LOGGER.opt(exception=True).warning(
                f"pipeline variant m={m_bucket}, n={state.n}, k={state.k} failed to "
                f"compile; {state.layer_name} will not mine at this bucket"
            )
            continue
        with _variant_lock:
            _ready_variants.add(variant)
        _LOGGER.info(
            f"pipeline variant ready: m={m_bucket}, n={state.n}, k={state.k} ({state.layer_name})"
        )
    return all(variant_ready(m_bucket, state) for m_bucket in runtime_settings().m_buckets)
