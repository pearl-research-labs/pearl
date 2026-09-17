"""Capture gate: blocks new GPU-touching mining work and waits for in-flight
work to finish, so CUDA-graph capture never races a mining producer.

Framework compilation and kernel warmup use the narrower
:func:`suspend_mining_launches`: credited launches stop while startup
preparation (variant warmup, B preparation) may still run. That gate does not
prove quiescence and is never sufficient for capture.

Mining launches do host<->device work (events, host callbacks, pinned signals,
allocations) that is illegal during capture and pointless during the warmup
preceding it. Capture raises the gate with :func:`graph_setup_no_mining`, then
waits with :func:`wait_for_mining_producers_idle`. Every GPU-touching mining
section runs inside :func:`gpu_mining_producer`, which refuses to start once
the gate is up. Both operate under one condition variable, so a producer can
never slip in between capture's raise and its wait.

``torch.cuda.is_current_stream_capturing()`` misses eager graph-break ops that
run while capture is in progress, so readers combine it with this gate. The
depth is a lock-guarded process global because capture applies process-wide:
the framework raises it on the worker thread while completion continuations
run on their own threads. It is reentrant.
"""

import collections
import contextlib
import itertools
import math
import threading
import time
from collections import deque
from collections.abc import Iterator
from typing import Protocol

from miner_utils import get_logger

_LOGGER = get_logger(__name__)
# The drain polls the idle predicate several times a second; report at most this
# often so a stuck drain is explained without flooding the log.
_IDLE_REPORT_INTERVAL_S = 5.0
_idle_report_after = 0.0

# One condition guards both the gate depth and the in-flight producer count:
# checking the gate and registering as a producer must be a single atomic step.
_STATE = threading.Condition()
_process_global_depth = 0
_launch_suspension_depth = 0
_launch_suspension_ids: set[int] = set()
_next_launch_suspension_id = itertools.count()
_producer_suspension_ids: set[int] = set()
_next_producer_suspension_id = itertools.count()
_active_producers = 0
# Held producer slots per thread: a slot that is never released is the hard case
# to diagnose, and the holder's name says which subsystem leaked it. Counted, not
# a bare name, because one thread can hold nested slots and releasing the inner
# one must not erase a holder the outer one still owns.
_producer_slots: collections.Counter[str] = collections.Counter()


class _CompletionEvent(Protocol):
    def query(self) -> bool: ...


class _PendingCompletion:
    """A recorded completion event plus whatever its launch must outlive.

    ``keepalive`` is an object the queued kernels read through a raw device
    pointer, so freeing it while they run would be a use-after-free. The
    registry holds the last reference once the launching frame returns and drops
    it when this event's own query proves the kernels are done -- the same fence
    capture and drain already wait on. Reclamation is therefore O(1) per launch
    and needs no process-wide quiet window.
    """

    __slots__ = ("event", "keepalive")

    def __init__(self, event: _CompletionEvent, keepalive: object) -> None:
        self.event = event
        self.keepalive = keepalive

    def query(self) -> bool:
        return self.event.query()


# Recorded mining GPU work whose kernels may still be running after the host
# producer slot was released. Capture and drain must wait for these too.
_pending_completions: deque[_CompletionEvent] = deque()
_COMPLETION_POLL_S = 0.005
DEFAULT_CAPTURE_QUIESCE_TIMEOUT_S = 60.0
# Closed while a reusable flush or runtime teardown owns producer admission.
_admission_closed = False


@contextlib.contextmanager
def graph_setup_no_mining() -> Iterator[None]:
    """Raise the capture gate for the enclosed scope (mining disabled).

    Blocks new producers immediately; callers that need in-flight producers
    finished must also call :func:`wait_for_mining_producers_idle`.
    """
    global _process_global_depth
    with _STATE:
        _process_global_depth += 1
    try:
        yield
    finally:
        with _STATE:
            _process_global_depth -= 1
            _STATE.notify_all()


def in_graph_setup_no_mining() -> bool:
    """Whether the capture gate is currently up."""
    # A GIL-atomic read; the lock is only needed for the paired
    # check-and-register in gpu_mining_producer.
    return _process_global_depth > 0


@contextlib.contextmanager
def suspend_mining_launches() -> Iterator[None]:
    """Suppress credited mined launches while startup preparation may continue."""
    global _launch_suspension_depth
    with _STATE:
        _launch_suspension_depth += 1
    try:
        yield
    finally:
        with _STATE:
            _launch_suspension_depth -= 1


def suspend_mining_launches_until_resumed(reason: str = "") -> int:
    """Suspend launches across separate framework hook calls.

    Returns an id for :func:`resume_mining_launches`; suspensions are reentrant
    and remain active until every id has been released.
    """
    suspension_id = next(_next_launch_suspension_id)
    with _STATE:
        _launch_suspension_ids.add(suspension_id)
    _LOGGER.debug(f"Mining launches suspended (id={suspension_id}, reason={reason!r})")
    return suspension_id


def resume_mining_launches(suspension_id: int) -> None:
    """Release one cross-hook suspension; unknown/released ids are no-ops."""
    with _STATE:
        removed = suspension_id in _launch_suspension_ids
        _launch_suspension_ids.discard(suspension_id)
    if removed:
        _LOGGER.debug(f"Mining launches resumed (id={suspension_id})")


def mining_launches_suspended() -> bool:
    """Whether framework initialization permits actual mining launches."""
    return (
        _launch_suspension_depth > 0
        or bool(_launch_suspension_ids)
        or bool(_producer_suspension_ids)
    )


def suspend_mining_producers_until_resumed(reason: str = "") -> int:
    """Block every GPU-touching producer across separate lifecycle hooks."""
    suspension_id = next(_next_producer_suspension_id)
    with _STATE:
        _producer_suspension_ids.add(suspension_id)
        _STATE.notify_all()
    _LOGGER.debug(f"Mining producers suspended (id={suspension_id}, reason={reason!r})")
    return suspension_id


def resume_mining_producers(suspension_id: int) -> None:
    """Release one cross-hook producer gate; unknown/released ids are no-ops."""
    with _STATE:
        removed = suspension_id in _producer_suspension_ids
        _producer_suspension_ids.discard(suspension_id)
        _STATE.notify_all()
    if removed:
        _LOGGER.debug(f"Mining producers resumed (id={suspension_id})")


def mining_producer_admission_open() -> bool:
    """Whether a long producer should continue optional preparation work."""
    return not _admission_closed and _process_global_depth == 0 and not _producer_suspension_ids


@contextlib.contextmanager
def gpu_mining_producer() -> Iterator[bool]:
    """Enter a GPU-touching mining section.

    Yields False (and registers nothing) when the capture gate is up, so the
    caller must skip its GPU work. Yields True while holding a producer slot
    that :func:`wait_for_mining_producers_idle` waits on.
    """
    global _active_producers
    with _STATE:
        admitted = mining_producer_admission_open()
        if admitted:
            _active_producers += 1
            _producer_slots[threading.current_thread().name] += 1
    if not admitted:
        yield False
        return
    try:
        yield True
    finally:
        with _STATE:
            _active_producers -= 1
            name = threading.current_thread().name
            _producer_slots[name] -= 1
            if _producer_slots[name] <= 0:
                del _producer_slots[name]
            _STATE.notify_all()


def register_mining_completion(event: _CompletionEvent, keepalive: object = None) -> None:
    """Record GPU work that outlives its producer slot (see the module docstring).

    ``keepalive`` is released once ``event`` completes: pass anything the queued
    kernels dereference but the caller stops referencing when it returns (see
    :class:`_PendingCompletion`).
    """
    with _STATE:
        _pending_completions.append(
            event if keepalive is None else _PendingCompletion(event, keepalive)
        )
        # Work on a stream completes in order, so reclaiming the finished prefix
        # keeps this hot path amortized O(1) instead of a scan per launch.
        while _pending_completions and _pending_completions[0].query():
            _pending_completions.popleft()


def close_mining_admission() -> None:
    """Refuse every future mining producer until a new runtime lifecycle opens one."""
    global _admission_closed
    with _STATE:
        _admission_closed = True
        _STATE.notify_all()


def open_mining_admission() -> None:
    """Admit mining producers after a flush or new runtime initialization."""
    global _admission_closed
    with _STATE:
        _admission_closed = False


def _prune_completions_locked() -> None:
    """Full sweep for the waiters: two streams can retire out of order, so the
    idle checks must not be fooled by a lagging event at the front."""
    survivors = [e for e in _pending_completions if not e.query()]
    _pending_completions.clear()
    _pending_completions.extend(survivors)


def mining_gpu_work_idle() -> bool:
    """Whether no mining section or recorded launch is running.

    As a drain predicate, it logs the unmet cause at a throttled rate.
    """
    global _idle_report_after
    with _STATE:
        producers, completions = _active_producers, 0
        holders = sorted(f"{name}x{count}" for name, count in _producer_slots.items())
        if not producers:
            _prune_completions_locked()
            completions = len(_pending_completions)
        if not producers and not completions:
            # Re-arm after idle so the next stall, including after restart, reports once.
            _idle_report_after = 0.0
            return True
        # Serialize the throttle because drain predicates can be polled concurrently.
        now = time.monotonic()
        if now < _idle_report_after:
            return False
        _idle_report_after = now + _IDLE_REPORT_INTERVAL_S
    _LOGGER.warning(
        f"Mining GPU work not idle: active_producers={producers} "
        f"pending_completions={completions} holders={holders or 'none'}"
    )
    return False


def wait_for_mining_producers_idle(timeout: float) -> bool:
    """Wait until mining is fully quiescent on both host and GPU, or time out.

    Host sections are waited on by condition; recorded completion events are
    then polled, because a producer releases its slot once its kernels are
    merely enqueued.
    """
    if not math.isfinite(timeout):
        raise ValueError(f"timeout must be finite, got {timeout!r}")
    deadline = time.monotonic() + timeout
    with _STATE:
        if not _STATE.wait_for(lambda: _active_producers == 0, timeout):
            return False
    while True:
        with _STATE:
            _prune_completions_locked()
            if not _pending_completions and not _active_producers:
                return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_COMPLETION_POLL_S, remaining))


@contextlib.contextmanager
def mining_suspended_for_capture(
    reason: str,
    *,
    framework: str,
    timeout: float = DEFAULT_CAPTURE_QUIESCE_TIMEOUT_S,
) -> Iterator[None]:
    """Raise the process gate, then prove all old mining GPU work complete.

    An unsuccessful boundary is fail-closed. A timeout leaves old GPU ownership
    unresolved, while an exception from the yielded framework scope can leave a
    partial graph epoch. In either case a durable producer suspension replaces
    the temporary graph gate before it is lowered; only a worker-process restart
    may admit mining again.
    """
    with graph_setup_no_mining():
        try:
            if not wait_for_mining_producers_idle(timeout):
                raise TimeoutError(
                    f"Mining GPU work did not quiesce within {timeout}s before "
                    f"{framework} CUDA-graph capture ({reason=}); refusing to enter capture"
                )
            yield
        except BaseException:
            # Install the durable gate while graph_setup_no_mining still owns
            # admission, so exception unwinding never exposes a producer gap.
            suspension_id = suspend_mining_producers_until_resumed(
                f"failed {framework} capture boundary: {reason}"
            )
            _LOGGER.critical(
                f"{framework} CUDA-graph setup failed ({reason=}); mining producer "
                f"admission remains closed until worker restart (id={suspension_id})"
            )
            raise
