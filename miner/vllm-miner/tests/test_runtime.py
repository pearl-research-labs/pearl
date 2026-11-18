import threading
from unittest.mock import Mock

import pytest
from miner_base.settings import MinerSettings
from vllm_miner import mining_state as runtime


def test_init_async_manager_honors_env_rpc_transport(monkeypatch):
    manager = Mock()
    manager.started = True
    constructor = Mock(return_value=manager)
    monkeypatch.setattr(runtime, "AsyncLoopManager", constructor)
    monkeypatch.setattr(runtime, "_async_manager", None)
    monkeypatch.setattr(runtime, "_runtime_poisoned", False)
    monkeypatch.setattr(runtime.config, "settings", runtime.config.settings)
    monkeypatch.setenv("MINER_RPC_TRANSPORT", "tcp")
    monkeypatch.setenv("MINER_RPC_HOST", "gateway.internal")
    monkeypatch.setenv("MINER_RPC_PORT", "18445")

    runtime.init_async_manager()

    rpc_config, settings = constructor.call_args.args
    assert rpc_config.transport == "tcp"
    assert rpc_config.host == "gateway.internal"
    assert rpc_config.port == 18445
    manager.start.assert_called_once_with()
    assert runtime.config.settings is settings


def test_concurrent_initializers_publish_exactly_one_manager(monkeypatch):
    entered_start = threading.Event()
    second_attempting_lock = threading.Event()
    release_start = threading.Event()
    manager = Mock(started=False)

    class ObservedLifecycleLock:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._count_lock = threading.Lock()
            self._attempts = 0

        def __enter__(self):
            with self._count_lock:
                self._attempts += 1
                if self._attempts == 2:
                    second_attempting_lock.set()
            self._lock.acquire()
            return self

        def __exit__(self, *_exc_info) -> None:
            self._lock.release()

    def start() -> None:
        entered_start.set()
        assert release_start.wait(timeout=5)
        manager.started = True

    manager.start.side_effect = start
    constructor = Mock(return_value=manager)
    monkeypatch.setattr(runtime, "AsyncLoopManager", constructor)
    monkeypatch.setattr(runtime, "_RUNTIME_LIFECYCLE_LOCK", ObservedLifecycleLock())
    monkeypatch.setattr(runtime, "_async_manager", None)
    monkeypatch.setattr(runtime, "_runtime_poisoned", False)
    monkeypatch.setattr(runtime, "_runtime_startup_disabled", False)

    errors: list[BaseException] = []

    def initialize() -> None:
        try:
            runtime.init_async_manager(MinerSettings(no_gateway=True))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=initialize)
    second = threading.Thread(target=initialize)
    first.start()
    assert entered_start.wait(timeout=5)
    second.start()
    assert second_attempting_lock.wait(timeout=5), "second initializer never attempted startup"
    assert constructor.call_count == 1
    release_start.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not errors
    assert not first.is_alive() and not second.is_alive()
    assert constructor.call_count == 1
    manager.start.assert_called_once_with()
    assert runtime._async_manager is manager


def test_init_replaces_a_cleanly_stopped_runtime_for_wake(monkeypatch):
    stopped = Mock(started=False)
    replacement = Mock(started=True)
    constructor = Mock(return_value=replacement)
    monkeypatch.setattr(runtime, "_async_manager", stopped)
    monkeypatch.setattr(runtime, "_runtime_poisoned", False)
    monkeypatch.setattr(runtime, "AsyncLoopManager", constructor)

    runtime.init_async_manager()

    assert runtime._async_manager is replacement
    replacement.start.assert_called_once_with()
    replacement.register_drain_barrier.assert_called_once_with(runtime.close_mining_admission)
    replacement.register_drain_release.assert_called_once_with(runtime.open_mining_admission)


@pytest.mark.parametrize(
    ("producer_idle", "continuation_idle", "expected"),
    [(True, True, True), (False, True, False), (True, False, False)],
)
def test_gpu_runtime_release_waits_for_memory_safety(
    monkeypatch, producer_idle, continuation_idle, expected
):
    now = [100.0]
    producer_timeout = []
    continuation_timeout = []

    def wait_for_producers(timeout):
        producer_timeout.append(timeout)
        now[0] += 0.25
        return producer_idle

    def wait_for_continuations(timeout):
        continuation_timeout.append(timeout)
        return continuation_idle

    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        "vllm_miner.capture.wait_for_mining_producers_idle",
        wait_for_producers,
    )
    monkeypatch.setattr(
        "vllm_miner.pipeline.wait_for_mining_continuations_idle",
        wait_for_continuations,
    )

    safe, error = runtime._gpu_runtime_safe_to_release(1.0)

    assert safe is expected
    assert (error is None) is expected
    assert producer_timeout == [1.0]
    assert continuation_timeout == ([] if not producer_idle else [pytest.approx(0.75)])


def test_delete_state_cannot_reopen_admission_from_concurrent_flush(monkeypatch):
    manager = runtime.AsyncLoopManager(Mock(), MinerSettings(no_gateway=True))
    manager._submission_executor = Mock()
    manager._accepting_submissions = True
    phase_two_entered = threading.Event()
    release_phase_two = threading.Event()
    close_admission = Mock()
    open_admission = Mock()

    def submissions_and_hashes_idle() -> bool:
        phase_two_entered.set()
        assert release_phase_two.wait(timeout=5)
        return True

    manager._submissions_drained = submissions_and_hashes_idle
    manager.register_drain_barrier(close_admission)
    manager.register_drain_release(open_admission)

    monkeypatch.setattr(runtime, "_async_manager", manager)
    monkeypatch.setattr(runtime, "_runtime_poisoned", False)
    monkeypatch.setattr(runtime, "close_mining_admission", close_admission)
    monkeypatch.setattr(runtime, "_gpu_runtime_safe_to_release", lambda _timeout: (True, None))

    flush_result: list[bool] = []
    flush = threading.Thread(
        target=lambda: flush_result.append(manager.wait_until_drained(timeout=5))
    )
    try:
        flush.start()
        assert phase_two_entered.wait(timeout=5)

        runtime.delete_state()
        release_phase_two.set()
        flush.join(timeout=5)

        assert not flush.is_alive()
        assert flush_result == [False]
        open_admission.assert_not_called()
        assert close_admission.call_count == 2
    finally:
        release_phase_two.set()
        flush.join(timeout=5)


def test_delete_state_poisons_when_safety_query_raises(monkeypatch):
    manager = Mock()
    monkeypatch.setattr(runtime, "_async_manager", manager)
    monkeypatch.setattr(runtime, "_runtime_poisoned", False)
    monkeypatch.setattr(
        runtime,
        "_gpu_runtime_safe_to_release",
        Mock(side_effect=RuntimeError("CUDA query failed")),
    )

    with pytest.raises(RuntimeError, match="CUDA query failed"):
        runtime.delete_state()

    manager.stop.assert_called_once_with(wait_for_submissions=False)
    assert runtime._async_manager is manager
    assert runtime._runtime_poisoned
