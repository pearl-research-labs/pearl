"""Portable policy tests for the device-wide mining OOM circuit breaker."""

import threading

import pytest
import torch
from vllm_miner.health import DeviceCircuitBreaker


def _open_cooldown(breaker: DeviceCircuitBreaker, device: int, cooldown: float) -> None:
    with pytest.raises(torch.cuda.OutOfMemoryError), breaker.begin(device, cooldown) as allowed:
        assert allowed
        raise torch.cuda.OutOfMemoryError("synthetic OOM")


def test_cooldown_admits_exactly_one_concurrent_probe():
    now = [10.0]
    breaker = DeviceCircuitBreaker(clock=lambda: now[0])
    _open_cooldown(breaker, device=0, cooldown=5.0)
    assert not breaker.begin(0, 5.0).allowed
    now[0] = 15.0

    start = threading.Barrier(9)
    release_probe = threading.Event()
    all_started = threading.Event()
    results: list[bool] = []
    results_lock = threading.Lock()

    def contend() -> None:
        start.wait()
        with breaker.begin(0, 5.0) as attempt:
            with results_lock:
                results.append(bool(attempt))
                if len(results) == 8:
                    all_started.set()
            if attempt:
                assert release_probe.wait(timeout=5)
                attempt.mark_success()

    threads = [threading.Thread(target=contend) for _ in range(8)]
    for thread in threads:
        thread.start()
    start.wait()
    assert all_started.wait(timeout=5)
    assert results.count(True) == 1
    assert results.count(False) == 7
    release_probe.set()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    # The successful probe closed the breaker.
    assert breaker.begin(0, 5.0).allowed


def test_probe_oom_restarts_cooldown_and_non_oom_failure_releases_probe():
    now = [0.0]
    breaker = DeviceCircuitBreaker(clock=lambda: now[0])
    _open_cooldown(breaker, device=0, cooldown=3.0)
    now[0] = 3.0

    with pytest.raises(RuntimeError, match="deterministic"), breaker.begin(0, 3.0) as allowed:
        assert allowed
        raise RuntimeError("deterministic probe failure")

    # A non-OOM failure does not strand the sole probe permit.
    with pytest.raises(torch.cuda.OutOfMemoryError), breaker.begin(0, 3.0) as allowed:
        assert allowed
        raise torch.cuda.OutOfMemoryError("probe OOM")
    assert not breaker.begin(0, 3.0).allowed
    now[0] = 6.0
    assert breaker.begin(0, 3.0).allowed


def test_noop_probe_does_not_reopen_ordinary_admission():
    now = [0.0]
    breaker = DeviceCircuitBreaker(clock=lambda: now[0])
    _open_cooldown(breaker, device=0, cooldown=3.0)
    now[0] = 3.0

    with breaker.begin(0, 3.0) as attempt:
        assert attempt  # stale/missing work returns without mark_success()

    retry = breaker.begin(0, 3.0)
    assert retry.allowed
    assert retry.probe
    with retry as attempt:
        attempt.mark_success()
    assert breaker.begin(0, 3.0).allowed


def test_devices_are_isolated_and_permanent_disable_is_fail_closed():
    breaker = DeviceCircuitBreaker(clock=lambda: 1.0)
    _open_cooldown(breaker, device=0, cooldown=10.0)

    assert not breaker.begin(0, 10.0).allowed
    assert breaker.begin(1, 10.0).allowed

    breaker.disable(1, "poisoned hit signal")
    assert not breaker.begin(1, 10.0).allowed
    assert breaker.disabled_reason(1) == "poisoned hit signal"
