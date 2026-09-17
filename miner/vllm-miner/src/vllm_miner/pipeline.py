"""Per-forward A-side pipeline: four launch-only stages over the padded
activation (``pre_quant -> tensor_hash_plus_stats -> noisy_quant ->
mixed_gemm``), then an async status-check callback on the AsyncLoopManager.

m pads up to a small bucket set (bounds CuTe JIT variants; zero pad rows are
committed as part of A, which is protocol-valid). A-side tensors are fresh
per call: the winner callback needs the committed planes after the forward.
"""

import math
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from miner_base.async_loop_manager import AsyncLoopManager, MiningLaunchDecision
from miner_utils import get_logger

from .capture import (
    mining_launches_suspended,
    mining_producer_admission_open,
    register_mining_completion,
)
from .health import report_device_oom
from .mining_config import (
    effective_work_per_matmul,
    max_mineable_k,
    mining_configuration,
    select_tile,
)
from .mining_state import get_async_manager
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

_LOGGER = get_logger(__name__)


# (M, n, k) pipeline variants compiled by warmup / first use.
_ready_variants: set[tuple[int, int, int]] = set()
_failed_variants: set[tuple[int, int, int]] = set()
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
    credited_hashes: int = 0
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
        if work.manager is not None and work.credited_hashes:
            work.manager.increment_credited_hashes(work.credited_hashes)
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
    credited_hashes: int = 0,
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


def _launch_credit(m_bucket: int, state: LayerState) -> int:
    """Protocol credit for one padded launch against this layer.

    Difficulty-normalized effective work (m*n*k): the only crediting unit, so
    the gateway's total_hash/hashrate is comparable across shapes/k and to
    network difficulty. effective_work_per_matmul derives the committed tile
    from (n, k), so a tall-tile (16x32) layer is credited against that tile.
    """
    return effective_work_per_matmul(m_bucket, state.n, state.k)


def variant_ready(m_bucket: int, n: int, k: int) -> bool:
    with _variant_lock:
        return (m_bucket, n, k) in _ready_variants


def _mixed_gemm_record_is_legal(m: int, n: int, k: int, kwargs: dict) -> bool:
    """Whether one saved record preserves protocol geometry and can launch."""
    from pearl_gemm import MixedGemmConfig, validate_mixed_gemm_config

    tile = select_tile(n, k)
    if tile is None:
        return False
    if kwargs.get("ltile_cols", tile.cols) != tile.cols:
        return False
    if kwargs.get("ltile_rows", tile.rows) != tile.rows:
        return False
    try:
        config = MixedGemmConfig(**{**kwargs, "ltile_cols": tile.cols, "ltile_rows": tile.rows})
        validate_mixed_gemm_config(m, n, k, config)
    except (TypeError, ValueError):
        return False
    return True


@lru_cache(maxsize=256)
def _configs(config_name: str, m: int, n: int, k: int):
    """Resolve and validate one device/shape pipeline configuration."""
    from pearl_gemm import (
        MixedGemmConfig,
        NoisyQuantConfig,
        PreQuantConfig,
        TensorHashConfig,
        tensor_hash_plus_stats_record_is_legal,
        validate_mixed_gemm_config,
        validate_noisy_quant_config,
    )
    from pearl_gemm.autotune import route_small_m_mixed_gemm

    tile = select_tile(n, k)
    if tile is None:
        raise ValueError(f"({n}, {k}) is not mineable by any committed lottery tile")

    prequant = PreQuantConfig(**tuned(config_name, "pre_quant", m=m, k=k))
    # Overlay the cert-v4 1024-byte leaf on autotune knobs. Every m bucket
    # of the layer must build the same A tree; the verifier can only open 1024.
    committed_leaf = mining_configuration(k, n).a_chunk_size
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
    prepare = NoisyQuantConfig(**tuned(config_name, "noisy_quant", m=m, k=k))
    validate_noisy_quant_config(m, k, prepare)

    def gemm_legal(kwargs: dict) -> bool:
        return _mixed_gemm_record_is_legal(m, n, k, kwargs)

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
    gemm = MixedGemmConfig(**{**gemm_kwargs, "ltile_cols": tile.cols, "ltile_rows": tile.rows})
    # Saved records are filtered first, but this public validator remains the
    # final safety boundary for both tuned and default configurations.
    validate_mixed_gemm_config(m, n, k, gemm)
    return prequant, commit, prepare, gemm


def _launch_stages(
    ctx: BOperands,
    config: "MiningConfiguration",
    a_padded: torch.Tensor,
    hit_signal: "HitSignal",
    *,
    layer_id: int,
    record_hits: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch every stage (no D2H sync); returns ``(codes, scales, a_keys, c)``.

    ``a_keys`` is the finalize's 96-byte ``seedA || noise-line keyA || jackpot
    key`` (``miner_base.commitment_hash.AKeys``).
    """
    from pearl_gemm import (
        LABEL_F1,
        R,
        mixed_gemm,
        noise_lines,
        noisy_quant,
        pre_quant,
        pre_quant_output_shapes,
        tensor_hash_plus_stats,
        tensor_hash_workspace_bytes,
    )

    device = a_padded.device
    m, k = a_padded.shape
    n = ctx.b_prime.shape[0]
    prequant_cfg, commit_cfg, prepare_cfg, gemm_cfg = _configs(
        device_config_name(device),
        m,
        n,
        k,
    )

    codes_shape, scales_shape = pre_quant_output_shapes(m, k)
    codes = torch.empty(codes_shape, dtype=torch.int8, device=device)
    scales = torch.empty(scales_shape, dtype=torch.bfloat16, device=device)
    pre_quant(a_padded, codes, scales, config=prequant_cfg)

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
    )
    noise_key_a = a_keys[32:64]
    pow_key = a_keys[64:96]

    # F_A is A's own basis, keyed by this launch's seedA: k lines drawn per
    # launch. At R == PACKED_NOISE_K the (k, R) draw viewed as int8 is already
    # noisy_quant's packed blob (pack_noise_factor).
    f1_lines = torch.empty(k, R, dtype=torch.float8_e4m3fn, device=device)
    noise_lines(noise_key_a, LABEL_F1, f1_lines)

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
        f1_lines.view(torch.int8),
        ctx.f2,
        alpha_a,
        beta_a,
        e1,
        a_prime,
        a_peel,
        config=prepare_cfg,
    )

    c = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    b_peel = torch.empty(n, 2 * R, dtype=torch.bfloat16, device=device)
    _b_peel_for_launch(ctx, f1_lines, b_peel)
    mixed_gemm(
        a_prime,
        ctx.b_prime,
        a_peel,
        b_peel,
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


def _b_peel_for_launch(ctx: BOperands, f1_lines: torch.Tensor, out: torch.Tensor) -> None:
    """B's peel for this launch's ``F_A`` (``pearl_gemm.b_peel_for_a``) into the
    launch-owned ``out``: the mid half is a per-launch ``(n, k) x (k, R)`` FP8
    product over ``B'`` -- one extra streaming read of the weight per mined
    forward that emits ``C``. Like every other per-launch buffer here, ``out``
    and the helper's FP32 intermediates come from the caching allocator;
    ``memory._launch_bytes`` budgets them."""
    from pearl_gemm import b_peel_for_a

    b_peel_for_a(ctx.b_prime, ctx.e2, ctx.f2, ctx.beta_b, ctx.b_peel, f1_lines, out=out)


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
    """Enqueue and credit one launch while its captured job stays current.

    Returns the launch's ``(m_bucket, n)`` output, or None when declined.
    """
    if mining_launches_suspended():
        return None
    manager = get_async_manager()
    with manager.mining_launch_admission(ctx.job) as decision:
        if not decision.launch:
            return None
        launched, output = _mine_admitted_launch(
            manager,
            decision,
            state,
            ctx,
            m_bucket,
            m_tokens,
            fill,
            bias=bias,
        )
        return output if launched else None


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


def _winner_continuation(
    *,
    retained: bool,
    callback_factory: WinnerCallbackFactory | None,
    hit_signal: "HitSignal",
    event: torch.cuda.Event,
    manager: AsyncLoopManager,
) -> tuple[Callable[[], None], Callable[[], None]]:
    if not retained:
        return _noop, _noop
    assert callback_factory is not None
    callback: Callable[[], None] = callback_factory(
        signal=hit_signal,
        event=event,
        leases=WINNER_LEASES,
        manager=manager,
    )
    return callback, WINNER_LEASES.release


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
) -> tuple[bool, torch.Tensor | None]:
    """Own completion/accounting and winner retention for one enqueue.

    Returns ``(launched, output)``.
    """
    # Every credited launch, retained or not, needs the lifecycle completion
    # worker. Start it before any GPU allocation so thread exhaustion fails
    # without leaving unowned kernels.
    _ensure_fallback_runner_started()
    leases = _acquire_launch_leases(decision)
    if leases is None:
        return False, None
    completion_lease, leased, winner_callback_factory = leases

    signal_event: torch.cuda.Event | None = None
    work_may_be_enqueued = False
    completion_published = False
    try:
        a_padded = torch.empty(
            m_bucket,
            state.k,
            dtype=torch.bfloat16,
            device=state.weight.device,
        )
        # ``fill`` may partially enqueue before raising. From this point onward
        # every escape must publish an exact completion or fence this stream.
        work_may_be_enqueued = True
        # Live serving rows are written exactly once; only protocol padding
        # needs explicit zeroes. Idle fill initializes its full synthetic input.
        fill(a_padded[:m_tokens])
        if m_tokens < m_bucket:
            a_padded[m_tokens:].zero_()
        hit_signal = _hit_signal_for(state.weight.device)
        _, _, _, c = _launch_stages(
            ctx,
            ctx.config,
            a_padded,
            hit_signal,
            layer_id=state.layer_id,
            record_hits=leased,
        )
        if bias is not None:
            # Keep this serving operation inside the launch's exact terminal
            # fence. In-place addition avoids a second m*n allocation and makes
            # completion-lease memory accounting cover the whole mined path.
            c[:m_tokens].add_(bias)

        signal_event = _record_stream_event()
        _register_launch_event(signal_event)
        callback, release = _winner_continuation(
            retained=leased,
            callback_factory=winner_callback_factory,
            hit_signal=hit_signal,
            event=signal_event,
            manager=manager,
        )
        # The worker proves event success before crediting. Its host token also
        # keeps capture/drain closed through rare winner D2H/reset/proof work.
        _start_launch_completion(
            signal_event,
            callback,
            release,
            manager=manager,
            credited_hashes=_launch_credit(m_bucket, state),
            state=state,
            device=state.weight.device,
            completion_lease=completion_lease,
        )
        completion_lease = None  # event-gated worker now owns this launch slot
        if leased:
            leased = False  # the event-gated callback now owns the winner lease
        completion_published = True
        return True, c
    finally:
        if leased:
            WINNER_LEASES.release()
        try:
            if work_may_be_enqueued and not completion_published:
                # A partial/failed launch may still have queued work. Prefer
                # another exact event; if event creation/record also fails,
                # synchronize only this launch stream before returning.
                try:
                    cleanup_event = _record_stream_event()
                    _register_launch_event(cleanup_event)
                except BaseException:
                    _synchronize_launch_stream(state.weight.device)
        finally:
            if completion_lease is not None:
                completion_lease.release()


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
    config = mining_configuration(state.k, state.n)
    for m_bucket in runtime_settings().m_buckets:
        if (cancelled is not None and cancelled()) or not mining_producer_admission_open():
            return False
        variant = (m_bucket, state.n, state.k)
        with _variant_lock:
            if variant in _ready_variants or variant in _failed_variants:
                continue
        try:
            dummy = torch.zeros(m_bucket, state.k, dtype=torch.bfloat16, device=device)
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
    return all(
        variant_ready(m_bucket, state.n, state.k) for m_bucket in runtime_settings().m_buckets
    )
