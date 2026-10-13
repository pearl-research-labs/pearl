"""Device-wide mining failure isolation that always preserves serving fallback."""

import threading
import time
from dataclasses import dataclass

import torch

from .settings import runtime_settings


@dataclass
class _DeviceState:
    cooldown_until: float = 0.0
    probe_in_flight: bool = False
    disabled_reason: str | None = None


class DeviceMiningAttempt:
    """One admitted ordinary attempt or the sole post-cooldown probe."""

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
        """Record whether this permit completed actual mining GPU work."""
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
            # A stale/missing context or a declined launch is not evidence that
            # the GPU recovered. Release only the exclusive probe ownership.
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
                    self,
                    device_index,
                    allowed=False,
                    probe=False,
                    cooldown_s=cooldown_s,
                )
            now = self._clock()
            if state.cooldown_until > 0.0:
                if now < state.cooldown_until or state.probe_in_flight:
                    return DeviceMiningAttempt(
                        self,
                        device_index,
                        allowed=False,
                        probe=False,
                        cooldown_s=cooldown_s,
                    )
                state.probe_in_flight = True
                return DeviceMiningAttempt(
                    self,
                    device_index,
                    allowed=True,
                    probe=True,
                    cooldown_s=cooldown_s,
                )
            return DeviceMiningAttempt(
                self,
                device_index,
                allowed=True,
                probe=False,
                cooldown_s=cooldown_s,
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

    def reset_for_tests(self) -> None:
        with self._lock:
            self._states.clear()


_BREAKER = DeviceCircuitBreaker()


def _device_index(device: torch.device) -> int:
    if device.type != "cuda":
        return -1  # host-only policy tests; production mining states are CUDA
    return device.index if device.index is not None else torch.cuda.current_device()


def mining_attempt(device: torch.device) -> DeviceMiningAttempt:
    return _BREAKER.begin(_device_index(device), runtime_settings().oom_cooldown_s)


def report_device_oom(device: torch.device) -> None:
    _BREAKER._record_oom(_device_index(device), runtime_settings().oom_cooldown_s)


def disable_device_mining(device: torch.device, reason: str) -> None:
    _BREAKER.disable(_device_index(device), reason)
