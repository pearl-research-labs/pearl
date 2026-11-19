"""Per-forward A-side pipeline: pre_quant -> tensor_hash_plus_stats -> noisy_quant -> (grouped_)mixed_gemm."""

import math
import queue
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from miner_base.async_loop_manager import AsyncLoopManager, MiningLaunchDecision
from miner_base.protocol import Device, local_device
from miner_utils import get_logger

from .capture import (
    cuda_device_index,
    mining_launches_suspended,
    mining_producer_admission_open,
    register_mining_completion,
    report_device_oom,
)
from .mining_config import (
    LotteryTileSpec,
    effective_work_per_matmul,
    max_mineable_k,
    mining_configuration,
    select_tile,
)
from .moe import MoeLaunch, MoeRouting, round_robin_routing, route_tokens
from .runtime import get_async_manager
from .settings import runtime_settings
from .state import (
    JobContext,
    LayerBuffers,
    LayerState,
    device_config_name,
    tuned,
    with_committed_leaf,
)

if TYPE_CHECKING:
    from miner_base.protocol import MiningConfiguration
    from pearl_gemm import HitSignal

type WinnerCallbackFactory = Callable[..., Callable[[], None]]
type MoeTail = Callable[[torch.Tensor, MoeRouting], object]

_LOGGER = get_logger(__name__)

_ready_variants: set[tuple[int, int, int, int, int]] = set()
_failed_variants: set[tuple[int, int, int, int, int]] = set()
_variant_lock = threading.Lock()

_HIT_SIGNALS: dict[int, "HitSignal"] = {}
_hit_signal_lock = threading.Lock()
_completion_cond = threading.Condition()
_worker_lock = threading.Lock()
_pending_completions_count = 0
_completion_thread: threading.Thread | None = None
_MAX_COMPLETION_QUEUE = 256


class LeasePool:
    """Thread-safe bounded counter for in-flight completion and winner-check leases."""

    def __init__(self, name: str) -> None:
        self._name = name
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
                raise RuntimeError(f"{self._name} lease underflow")
            self._held -= 1

    def held(self) -> int:
        with self._lock:
            return self._held


_COMPLETION_LEASES = LeasePool("mining completion")
WINNER_LEASES = LeasePool("winner-check")


def completions_idle() -> bool:
    with _completion_cond:
        return _pending_completions_count == 0


def wait_for_mining_continuations_idle(timeout: float) -> bool:
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError(f"timeout must be finite and non-negative, got {timeout!r}")
    with _completion_cond:
        return _completion_cond.wait_for(lambda: _pending_completions_count == 0, timeout)


class _CompletionToken:
    def __init__(self) -> None:
        self.done = threading.Event()

    def query(self) -> bool:
        return self.done.is_set()


@dataclass(frozen=True)
class _CompletionWork:
    event: torch.cuda.Event
    callback: Callable[[], None]
    release: Callable[[], None]
    token: _CompletionToken
    manager: AsyncLoopManager | None = None
    credited_hashes: int | Callable[[], int] = 0
    state: LayerState | None = None
    device: torch.device | None = None
    holds_completion_lease: bool = False


_completion_queue: queue.Queue[_CompletionWork] = queue.Queue(maxsize=_MAX_COMPLETION_QUEUE)


def _noop() -> None:
    pass


def _run_completion_work(work: _CompletionWork) -> None:
    global _pending_completions_count
    callback_entered = False
    try:
        work.event.synchronize()
        credited = (
            work.credited_hashes() if callable(work.credited_hashes) else work.credited_hashes
        )
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
            if work.holds_completion_lease:
                _COMPLETION_LEASES.release()
            with _completion_cond:
                _pending_completions_count -= 1
                _completion_cond.notify_all()
            work.token.done.set()


def _completion_worker() -> None:
    while True:
        _run_completion_work(_completion_queue.get())


def _ensure_completion_worker_started() -> None:
    global _completion_thread
    with _worker_lock:
        if _completion_thread is not None:
            if not _completion_thread.is_alive():
                raise RuntimeError("mining completion worker stopped unexpectedly")
            return
        thread = threading.Thread(target=_completion_worker, name="mining-completion", daemon=True)
        thread.start()
        _completion_thread = thread


def _start_launch_completion(
    event: torch.cuda.Event,
    callback: Callable[[], None],
    release: Callable[[], None],
    *,
    manager: AsyncLoopManager | None = None,
    credited_hashes: int | Callable[[], int] = 0,
    state: LayerState | None = None,
    device: torch.device | None = None,
    holds_completion_lease: bool = False,
) -> None:
    global _pending_completions_count
    with _completion_cond:
        _pending_completions_count += 1
    token = _CompletionToken()
    work = _CompletionWork(
        event=event,
        callback=callback,
        release=release,
        token=token,
        manager=manager,
        credited_hashes=credited_hashes,
        state=state,
        device=device,
        holds_completion_lease=holds_completion_lease,
    )
    try:
        register_mining_completion(token)
        _completion_queue.put_nowait(work)
    except BaseException as exc:
        _LOGGER.warning(
            f"mining completion publication failed ({type(exc).__name__}); running synchronously"
        )
        _run_completion_work(work)


def _hit_signal_for(device: torch.device) -> "HitSignal":
    """One persistent hit signal per CUDA device, sized for the full protocol-legal domain."""
    from pearl_gemm import HitSignal, HitSignalConfig

    index = cuda_device_index(device)
    max_m, max_k = max(runtime_settings().m_buckets), max_mineable_k()
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
            with torch.inference_mode(False):
                signal = HitSignal(HitSignalConfig(max_m=max_m, max_k=max_k), device=device)
            _HIT_SIGNALS[index] = signal
        return signal


def pick_bucket(m_tokens: int) -> int | None:
    for bucket in runtime_settings().m_buckets:
        if m_tokens <= bucket:
            return bucket
    return None


def _lottery_tile(state: LayerState) -> LotteryTileSpec:
    tile = select_tile(state.lottery_n, state.k, device=state.committed_device)
    if tile is None:
        raise RuntimeError(
            f"{state.layer_name}: no lottery tile for n_e={state.lottery_n}, k={state.k}"
        )
    return tile


def moe_launch_credit(state: LayerState, expert_rows: Sequence[int]) -> int:
    """Credited effective work for an MoE launch's whole per-expert lottery tiles."""
    tile = _lottery_tile(state)
    rows = sum(int(count) // tile.rows for count in expert_rows) * tile.rows
    if not rows:
        return 0
    return effective_work_per_matmul(rows, state.lottery_n, state.k, device=state.committed_device)


class _DeferredMoeCredit:
    """Non-blocking pinned D2H copy of per-expert counts, reduced after event completion."""

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
    committed_leaf = mining_configuration(k, n, device=device).a_chunk_size
    assert committed_leaf is not None
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
    prepare = replace(default_noisy_quant_config(), **tuned(config_name, "noisy_quant", m=m, k=k))
    validate_noisy_quant_config(m, k, prepare)

    def gemm_legal(kwargs: dict) -> bool:
        return _mixed_gemm_record_is_legal(m, n, k, kwargs, device=device)

    gemm_kwargs = tuned(config_name, "mixed_gemm", legal=gemm_legal, exact=True, m=m, n=n, k=k)
    if not gemm_kwargs:
        gemm_kwargs = route_small_m_mixed_gemm(m, n, k, ltile_rows=tile.rows, ltile_cols=tile.cols)
    if gemm_kwargs is None:
        gemm_kwargs = tuned(
            config_name, "mixed_gemm", legal=gemm_legal, require_legal=True, m=m, n=n, k=k
        )
    gemm = replace(
        default_mixed_gemm_config(),
        **{**gemm_kwargs, "ltile_cols": tile.cols, "ltile_rows": tile.rows},
    )
    validate_mixed_gemm_config(m, n, k, gemm)
    return prequant, commit, prepare, gemm


def _launch_stages(
    buffers: LayerBuffers,
    config: "MiningConfiguration",
    a_launch: torch.Tensor,
    hit_signal: "HitSignal",
    *,
    layer_id: int,
    record_hits: bool,
    routing: MoeRouting | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch all four GPU stages without host synchronization; return (codes, scales, a_keys, c)."""
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
    n = buffers.b_prime.shape[0]
    experts = config.experts
    if (routing is None) != (experts == 0):
        raise ValueError("MoE layers launch with routing, dense layers without")
    lottery_n = n // experts if experts else n
    prequant_cfg, commit_cfg, prepare_cfg, gemm_cfg = _configs(
        device_config_name(device), m, lottery_n, k, device=config.device
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
        buffers.key_a_dev,
        buffers.seed_b_dev,
        root_codes,
        root_scales,
        roots,
        a_keys,
        stats,
        p_a=config.p_a(m),
        config=commit_cfg,
        routing_commitments=None if routing is None else routing.commitments,
    )

    alpha_a = torch.empty(m, dtype=torch.bfloat16, device=device)
    beta_a = torch.empty_like(alpha_a)
    e1 = torch.empty(m, R, dtype=torch.float8_e4m3fn, device=device)
    a_prime = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=device)
    a_peel = torch.empty(m, 2 * R, dtype=torch.bfloat16, device=device)
    noisy_quant(
        codes,
        scales,
        a_keys[32:64],
        stats,
        buffers.f1_hl,
        buffers.f2,
        alpha_a,
        beta_a,
        e1,
        a_prime,
        a_peel,
        config=prepare_cfg,
    )

    pow_key = a_keys[64:96]
    rows = m if routing is None else routing.cum_m
    c = torch.empty(rows, lottery_n, dtype=torch.bfloat16, device=device)
    if routing is not None:
        _grouped_mixed_gemm(
            buffers,
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
        buffers.b_prime,
        a_peel,
        buffers.b_peel,
        alpha_a,
        buffers.inv_alpha_b,
        pow_key,
        buffers.threshold_dev,
        c,
        hit_signal,
        codes,
        scales,
        buffers.seed_b_dev,
        config=gemm_cfg,
        layer_id=layer_id,
        record_hits=record_hits,
    )
    return codes, scales, a_keys, c


def _grouped_mixed_gemm(
    buffers: LayerBuffers,
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
    from pearl_gemm import GroupedMixedGemmConfig, grouped_mixed_gemm

    experts, n_e, k = routing.experts, buffers.b_prime.shape[0] // routing.experts, a_prime.shape[1]
    tile = select_tile(n_e, k, device=committed_device)
    if tile is None:
        raise RuntimeError(f"no lottery tile for n_e={n_e}, k={k}")
    rows = routing.tokens

    grouped_mixed_gemm(
        a_prime.view(torch.uint8).index_select(0, rows).view(torch.float8_e4m3fn),
        buffers.b_prime.view(experts, n_e, k),
        a_peel.index_select(0, rows),
        buffers.b_peel.view(experts, n_e, -1),
        alpha_a.index_select(0, rows),
        buffers.inv_alpha_b.view(experts, n_e),
        pow_key,
        buffers.threshold_dev,
        c,
        routing.m_indptr,
        hit_signal,
        None,
        None,
        buffers.seed_b_dev,
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


def _register_launch_event(event: torch.cuda.Event) -> None:
    try:
        register_mining_completion(event)
    except BaseException as exc:
        event.synchronize()
        _LOGGER.warning(
            f"mining completion registration failed ({type(exc).__name__}); "
            "launch completed synchronously"
        )


@dataclass
class _OwnedRecordDiscard:
    """Drain only a failed launch's own published hit record so the shared latch never wedges."""

    signal: "HitSignal"
    a_keys: torch.Tensor
    pow_key_host: torch.Tensor | None
    leases: LeasePool

    def __call__(self) -> None:
        try:
            owner = (
                bytes(self.pow_key_host.tolist())
                if self.pow_key_host is not None
                else bytes(self.a_keys[64:96].cpu().tolist())
            )
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


def _fence_abandoned_launch(
    state: LayerState,
    published: tuple["HitSignal", torch.Tensor, torch.Tensor | None] | None,
    holds_completion_lease: bool,
) -> bool:
    device = state.weight.device
    cleanup_event: torch.cuda.Event | None = None
    try:
        cleanup_event = torch.cuda.Event()
        cleanup_event.record()
        _register_launch_event(cleanup_event)
    except BaseException:
        cleanup_event = None
        torch.cuda.current_stream(device).synchronize()
    if published is None:
        return False
    signal, a_keys, key_copy = published
    discard = _OwnedRecordDiscard(signal, a_keys, key_copy, WINNER_LEASES)
    if cleanup_event is None:
        discard()
        if holds_completion_lease:
            _COMPLETION_LEASES.release()
        return True
    _start_launch_completion(
        cleanup_event,
        discard,
        WINNER_LEASES.release,
        state=state,
        device=device,
        holds_completion_lease=holds_completion_lease,
    )
    return True


@dataclass(frozen=True)
class _MinedLaunch:
    output: torch.Tensor
    routing: MoeRouting | None
    result: object = None


def _acquire_leases(
    decision: MiningLaunchDecision,
) -> tuple[bool, WinnerCallbackFactory | None] | None:
    winner_callback_factory: WinnerCallbackFactory | None = None
    if decision.retain_winner:
        from .winners import WinnerCheckCallback

        winner_callback_factory = WinnerCheckCallback
    if not _COMPLETION_LEASES.try_acquire(runtime_settings().completion_inflight_limit):
        return None
    if decision.retain_winner and not WINNER_LEASES.try_acquire(
        runtime_settings().winner_check_inflight_limit
    ):
        _COMPLETION_LEASES.release()
        return None
    return decision.retain_winner, winner_callback_factory


def _prepare_routing_and_moe_credit(
    state: LayerState, buffers: LayerBuffers, topk_ids: torch.Tensor | None
) -> tuple[MoeRouting | None, _DeferredMoeCredit | None]:
    if state.experts:
        if topk_ids is None:
            raise ValueError(f"MoE layer {state.layer_name} launches with the router's topk_ids")
        routing = route_tokens(topk_ids, state.experts, buffers.key_a_dev)
        return routing, _DeferredMoeCredit(state, routing.m_valid)
    if topk_ids is not None:
        raise ValueError(f"dense layer {state.layer_name} takes no topk_ids")
    return None, None


def _build_winner_callback(
    leased: bool,
    factory: WinnerCallbackFactory | None,
    hit_signal: "HitSignal",
    manager: AsyncLoopManager,
    pow_key_host: torch.Tensor | None,
    routing: MoeRouting | None,
    codes: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[Callable[[], None], Callable[[], None]]:
    if not leased:
        return _noop, _noop
    assert factory is not None and pow_key_host is not None
    cb = factory(
        signal=hit_signal,
        leases=WINNER_LEASES,
        manager=manager,
        pow_key_host=pow_key_host,
        moe=None if routing is None else MoeLaunch(routing, codes, scales),
    )
    return cb, WINNER_LEASES.release


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
    _ensure_completion_worker_started()
    if bias is not None and state.experts:
        raise ValueError("an MoE launch returns the permuted expert output; add bias per expert")
    acquired = _acquire_leases(decision)
    if acquired is None:
        return None
    leased, winner_callback_factory = acquired
    holds_completion = True

    device = state.weight.device
    buffers = state.buffers
    assert buffers is not None
    work_may_be_enqueued = False
    completion_published = False
    published: tuple[HitSignal, torch.Tensor, torch.Tensor | None] | None = None
    try:
        pow_key_host = torch.empty(32, dtype=torch.uint8, pin_memory=True) if leased else None
        a_launch = torch.empty(m_bucket, state.k, dtype=torch.bfloat16, device=device)
        work_may_be_enqueued = True

        routing, moe_credit = _prepare_routing_and_moe_credit(state, buffers, topk_ids)
        fill(a_launch[:m_tokens])
        if m_tokens < m_bucket:
            a_launch[m_tokens:].zero_()
        hit_signal = _hit_signal_for(device)
        codes, scales, a_keys, c = _launch_stages(
            buffers,
            ctx.config,
            a_launch,
            hit_signal,
            layer_id=state.layer_id,
            record_hits=leased,
            routing=routing,
        )
        if pow_key_host is not None:
            published = (hit_signal, a_keys, None)
            pow_key_host.copy_(a_keys[64:96], non_blocking=True)
            published = (hit_signal, a_keys, pow_key_host)
        if bias is not None:
            c[:m_tokens].add_(bias)
        if tail is not None and routing is None:
            raise ValueError("a launch tail needs an MoE launch")
        result = tail(c, routing) if tail is not None and routing is not None else None
        credit: int | Callable[[], int] = (
            effective_work_per_matmul(m_bucket, state.n, state.k, device=state.committed_device)
            if moe_credit is None
            else moe_credit
        )

        signal_event = torch.cuda.Event()
        signal_event.record()
        _register_launch_event(signal_event)
        callback, release = _build_winner_callback(
            leased,
            winner_callback_factory,
            hit_signal,
            manager,
            pow_key_host,
            routing,
            codes,
            scales,
        )

        _start_launch_completion(
            signal_event,
            callback,
            release,
            manager=manager,
            credited_hashes=credit,
            state=state,
            device=device,
            holds_completion_lease=holds_completion,
        )
        holds_completion = False
        leased = False
        completion_published = True
        return _MinedLaunch(c, routing, result)
    finally:
        try:
            if (
                work_may_be_enqueued
                and not completion_published
                and _fence_abandoned_launch(state, published, holds_completion)
            ):
                leased = False
                holds_completion = False
        finally:
            if leased:
                WINNER_LEASES.release()
            if holds_completion:
                _COMPLETION_LEASES.release()


def _mine_launch(
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


def mine_launch(
    state: LayerState,
    ctx: JobContext,
    m_bucket: int,
    m_tokens: int,
    fill: Callable[[torch.Tensor], object],
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
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
) -> _MinedLaunch | None:
    if not state.experts:
        raise ValueError(f"dense layer {state.layer_name} launches through mine_launch")
    return _mine_launch(
        state, ctx, m_bucket, m_tokens, fill, bias=None, topk_ids=topk_ids, tail=tail
    )


def run_mining_forward(
    state: LayerState,
    ctx: JobContext,
    x2d: torch.Tensor,
    bias: torch.Tensor | None,
    m_bucket: int,
) -> torch.Tensor | None:
    m_tokens = x2d.shape[0]
    c = mine_launch(state, ctx, m_bucket, m_tokens, lambda a: a[:m_tokens].copy_(x2d), bias=bias)
    return None if c is None else c[:m_tokens]


def run_mining_moe_forward(
    state: LayerState,
    ctx: JobContext,
    x2d: torch.Tensor,
    topk_ids: torch.Tensor,
    m_bucket: int,
    tail: MoeTail | None = None,
) -> object | None:
    m_tokens = x2d.shape[0]
    launched = mine_moe_launch(
        state, ctx, m_bucket, m_tokens, lambda a: a[:m_tokens].copy_(x2d), topk_ids, tail=tail
    )
    if launched is None:
        return None
    if launched.routing is None:
        raise RuntimeError("an admitted MoE serving launch emits its routing")
    return (launched.output, launched.routing) if tail is None else launched.result


def warmup_layer_variants(state: LayerState) -> bool:
    """Compile all configured m_bucket pipeline variants for ``state`` off the serving path."""
    assert state.buffers is not None
    device = state.weight.device
    config = mining_configuration(state.k, state.n, state.experts, device=state.committed_device)
    for m_bucket in runtime_settings().m_buckets:
        if not mining_producer_admission_open():
            return False
        variant = _variant_key(m_bucket, state)
        with _variant_lock:
            if variant in _ready_variants or variant in _failed_variants:
                continue
        try:
            dummy = torch.zeros(m_bucket, state.k, dtype=torch.bfloat16, device=device)
            routing = (
                round_robin_routing(m_bucket, state.experts, state.top_k, state.buffers.key_a_dev)
                if state.experts
                else None
            )
            _launch_stages(
                state.buffers,
                config,
                dummy,
                _hit_signal_for(device),
                layer_id=state.layer_id,
                record_hits=False,
                routing=routing,
            )
            ev = torch.cuda.Event()
            ev.record()
            ev.synchronize()
            if not mining_producer_admission_open():
                return False
        except torch.cuda.OutOfMemoryError:
            ev = torch.cuda.Event()
            ev.record()
            ev.synchronize()
            raise
        except Exception:
            ev = torch.cuda.Event()
            ev.record()
            ev.synchronize()
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
