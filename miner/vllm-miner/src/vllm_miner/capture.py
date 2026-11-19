"""Process-wide mining admission, CUDA-graph capture quiescence, and OOM isolation.

Every GPU-touching mining section enters ``gpu_mining_producer()``, which shares
one condition variable with ``graph_setup_no_mining()`` and
``wait_for_mining_producers_idle()`` so CUDA-graph capture never races a mining
kernel or completion continuation.
"""

import collections
import contextlib
import math
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import torch
from miner_utils import get_logger

from .settings import runtime_settings

_LOGGER = get_logger(__name__)
_IDLE_REPORT_INTERVAL_S = 5.0
_COMPLETION_POLL_S = 0.005
DEFAULT_CAPTURE_QUIESCE_TIMEOUT_S = 60.0

_STATE = threading.Condition()
_process_global_depth = 0
_launch_suspension_depth = 0
_capture_launch_suspended = False
_producer_permanently_suspended = False
_admission_closed = False
_active_producers = 0
_producer_slots: collections.Counter[str] = collections.Counter()
_idle_report_after = 0.0


class _CompletionEvent(Protocol):
    def query(self) -> bool: ...


_pending_completions: deque[_CompletionEvent] = deque()


def cuda_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        return -1
    return device.index if device.index is not None else torch.cuda.current_device()


# ---------------------------------------------------------------------------
# CUDA-graph capture quiescence & admission gates
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def graph_setup_no_mining() -> Iterator[None]:
    """Raise the process-wide capture gate for the enclosed scope."""
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
    return _process_global_depth > 0


@contextlib.contextmanager
def suspend_mining_launches() -> Iterator[None]:
    """Suppress credited mined launches while startup preparation may still run."""
    global _launch_suspension_depth
    with _STATE:
        _launch_suspension_depth += 1
    try:
        yield
    finally:
        with _STATE:
            _launch_suspension_depth -= 1


def _will_capture_cuda_graphs() -> bool:
    try:
        from vllm.config import get_current_vllm_config
        from vllm.config.compilation import CUDAGraphMode

        config = get_current_vllm_config()
        if bool(config.model_config.enforce_eager):
            return False
        mode = config.compilation_config.cudagraph_mode
        return mode is not None and mode != CUDAGraphMode.NONE
    except Exception:
        _LOGGER.warning(
            "Could not resolve vLLM CUDA-graph mode; not pre-suspending mining.",
            exc_info=True,
        )
        return False


def suspend_submissions_until_capture_complete() -> None:
    """Suspend mined launches across model-load -> capture_model when graphs are enabled."""
    global _capture_launch_suspended
    if not _will_capture_cuda_graphs():
        return
    with _STATE:
        _capture_launch_suspended = True


def resume_submissions_after_capture() -> None:
    """Resume mined launches after capture_model or on the first real execute step."""
    global _capture_launch_suspended
    with _STATE:
        _capture_launch_suspended = False


def mining_launches_suspended() -> bool:
    return (
        _launch_suspension_depth > 0 or _capture_launch_suspended or _producer_permanently_suspended
    )


def mining_producer_admission_open() -> bool:
    return (
        not _admission_closed and _process_global_depth == 0 and not _producer_permanently_suspended
    )


@contextlib.contextmanager
def gpu_mining_producer() -> Iterator[bool]:
    """Atomically check admission and hold a producer slot that capture waits on."""
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


def register_mining_completion(event: _CompletionEvent) -> None:
    """Record GPU/continuation work that outlives its producer slot."""
    with _STATE:
        _pending_completions.append(event)
        while _pending_completions and _pending_completions[0].query():
            _pending_completions.popleft()


def close_mining_admission() -> None:
    global _admission_closed
    with _STATE:
        _admission_closed = True
        _STATE.notify_all()


def open_mining_admission() -> None:
    global _admission_closed
    with _STATE:
        _admission_closed = False


def _prune_completions_locked() -> None:
    survivors = [e for e in _pending_completions if not e.query()]
    _pending_completions.clear()
    _pending_completions.extend(survivors)


def mining_gpu_work_idle() -> bool:
    global _idle_report_after
    with _STATE:
        producers, completions = _active_producers, 0
        holders = sorted(f"{name}x{count}" for name, count in _producer_slots.items())
        if not producers:
            _prune_completions_locked()
            completions = len(_pending_completions)
        if not producers and not completions:
            _idle_report_after = 0.0
            return True
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
    timeout: float = DEFAULT_CAPTURE_QUIESCE_TIMEOUT_S,
) -> Iterator[None]:
    """Raise the gate and prove all mining GPU work complete; fail closed on any error."""
    global _producer_permanently_suspended
    with graph_setup_no_mining():
        try:
            if not wait_for_mining_producers_idle(timeout):
                raise TimeoutError(
                    f"Mining GPU work did not quiesce within {timeout}s before "
                    f"vLLM CUDA-graph capture ({reason=}); refusing to enter capture"
                )
            yield
        except BaseException:
            with _STATE:
                _producer_permanently_suspended = True
                _STATE.notify_all()
            _LOGGER.critical(
                f"vLLM CUDA-graph setup failed ({reason=}); mining producer "
                "admission remains closed until worker restart"
            )
            raise


# ---------------------------------------------------------------------------
# Device OOM circuit breaker
# ---------------------------------------------------------------------------


@dataclass
class _DeviceState:
    cooldown_until: float = 0.0
    probe_in_flight: bool = False
    disabled_reason: str | None = None


class DeviceMiningAttempt:
    def __init__(
        self,
        breaker: "DeviceCircuitBreaker",
        device_index: int,
        *,
        allowed: bool,
        probe: bool,
        cooldown_s: float,
    ) -> None:
        self._breaker = breaker
        self._device_index = device_index
        self.allowed = allowed
        self.probe = probe
        self._cooldown_s = cooldown_s
        self._resolved = not allowed
        self._succeeded = False

    def __bool__(self) -> bool:
        return self.allowed

    def __enter__(self) -> "DeviceMiningAttempt":
        return self

    def mark_success(self, successful: bool = True) -> None:
        self._succeeded = self._succeeded or (self.allowed and successful)

    def __exit__(self, exc_type, _exc, _tb) -> bool:
        if self._resolved:
            return False
        self._resolved = True
        if exc_type is not None and issubclass(exc_type, torch.cuda.OutOfMemoryError):
            self._breaker._record_oom(self._device_index, self._cooldown_s)
        elif exc_type is None and self._succeeded:
            self._breaker._record_success(self._device_index, probe=self.probe)
        else:
            self._breaker._release_probe(self._device_index, probe=self.probe)
        return False


class DeviceCircuitBreaker:
    """Cooldown after OOM, then admit exactly one bounded recovery probe."""

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._states: dict[int, _DeviceState] = {}

    def begin(self, device_index: int, cooldown_s: float) -> DeviceMiningAttempt:
        with self._lock:
            state = self._states.setdefault(device_index, _DeviceState())
            if state.disabled_reason is not None:
                return DeviceMiningAttempt(
                    self, device_index, allowed=False, probe=False, cooldown_s=cooldown_s
                )
            if state.cooldown_until > 0.0:
                if self._clock() < state.cooldown_until or state.probe_in_flight:
                    return DeviceMiningAttempt(
                        self, device_index, allowed=False, probe=False, cooldown_s=cooldown_s
                    )
                state.probe_in_flight = True
                return DeviceMiningAttempt(
                    self, device_index, allowed=True, probe=True, cooldown_s=cooldown_s
                )
            return DeviceMiningAttempt(
                self, device_index, allowed=True, probe=False, cooldown_s=cooldown_s
            )

    def _record_oom(self, device_index: int, cooldown_s: float) -> None:
        with self._lock:
            state = self._states.setdefault(device_index, _DeviceState())
            state.cooldown_until = self._clock() + cooldown_s
            state.probe_in_flight = False

    def _record_success(self, device_index: int, *, probe: bool) -> None:
        if not probe:
            return
        with self._lock:
            state = self._states.setdefault(device_index, _DeviceState())
            state.cooldown_until = 0.0
            state.probe_in_flight = False

    def _release_probe(self, device_index: int, *, probe: bool) -> None:
        if not probe:
            return
        with self._lock:
            self._states.setdefault(device_index, _DeviceState()).probe_in_flight = False

    def disable(self, device_index: int, reason: str) -> None:
        with self._lock:
            state = self._states.setdefault(device_index, _DeviceState())
            if state.disabled_reason is None:
                state.disabled_reason = reason
            state.probe_in_flight = False

    def disabled_reason(self, device_index: int) -> str | None:
        with self._lock:
            return self._states.get(device_index, _DeviceState()).disabled_reason


_BREAKER = DeviceCircuitBreaker()


def mining_attempt(device: torch.device) -> DeviceMiningAttempt:
    return _BREAKER.begin(cuda_device_index(device), runtime_settings().oom_cooldown_s)


def report_device_oom(device: torch.device) -> None:
    _BREAKER._record_oom(cuda_device_index(device), runtime_settings().oom_cooldown_s)


def disable_device_mining(device: torch.device, reason: str) -> None:
    _BREAKER.disable(cuda_device_index(device), reason)
