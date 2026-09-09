"""The capture gate must refuse new mining GPU work and let capture wait for
work already in flight. Pure host logic: no GPU needed."""

import threading
import time
import weakref
from collections import deque

import pytest
import vllm_miner.capture as gcs
from miner_base.async_loop_manager import AsyncLoopManager
from miner_base.settings import MinerSettings
from pearl_gateway.config import MinerRpcConfig
from vllm_miner.capture import (
    gpu_mining_producer,
    graph_setup_no_mining,
    in_graph_setup_no_mining,
    mining_launches_suspended,
    suspend_mining_launches,
    suspend_mining_producers_until_resumed,
    wait_for_mining_producers_idle,
)


def test_gate_refuses_producers_while_raised():
    with gpu_mining_producer() as admitted:
        assert admitted  # open by default

    with graph_setup_no_mining():
        assert in_graph_setup_no_mining()
        with gpu_mining_producer() as admitted:
            assert not admitted  # refused: capture owns the GPU

    with gpu_mining_producer() as admitted:
        assert admitted  # released


def test_shared_capture_scope_gates_before_wait_and_holds_through_capture(monkeypatch):
    events = []

    def wait(timeout):
        events.append(("wait", timeout, in_graph_setup_no_mining()))
        return True

    monkeypatch.setattr(gcs, "wait_for_mining_producers_idle", wait)
    with gcs.mining_suspended_for_capture("lazy-vision", framework="vLLM", timeout=7.0):
        events.append(("capture", in_graph_setup_no_mining()))

    assert events == [("wait", 7.0, True), ("capture", True)]
    assert not in_graph_setup_no_mining()


def test_shared_capture_scope_timeout_retains_durable_gate(_open_admission, monkeypatch):
    monkeypatch.setattr(gcs, "wait_for_mining_producers_idle", lambda _timeout: False)

    with (
        pytest.raises(TimeoutError, match="vLLM CUDA-graph capture"),
        gcs.mining_suspended_for_capture("lazy-vision", framework="vLLM"),
    ):
        pytest.fail("capture began before mining quiesced")

    # The temporary graph-depth marker unwinds, but unresolved CUDA ownership
    # leaves producer admission durably closed until the worker is restarted.
    assert not in_graph_setup_no_mining()
    assert mining_launches_suspended()
    with gpu_mining_producer() as admitted:
        assert not admitted


def test_shared_capture_scope_exception_retains_durable_gate(_open_admission, monkeypatch):
    monkeypatch.setattr(gcs, "wait_for_mining_producers_idle", lambda _timeout: True)

    with (
        pytest.raises(RuntimeError, match="partial capture"),
        gcs.mining_suspended_for_capture("decode", framework="vLLM"),
    ):
        raise RuntimeError("partial capture")

    assert not in_graph_setup_no_mining()
    assert mining_launches_suspended()
    with gpu_mining_producer() as admitted:
        assert not admitted


def test_launch_suspension_does_not_block_context_preparation(_open_admission):
    """Framework dummy forwards must fall back without blocking the startup
    B preparation that runs under the same producer gate during model warmup."""
    with suspend_mining_launches():
        assert mining_launches_suspended()
        with gpu_mining_producer() as admitted:
            assert admitted
    assert not mining_launches_suspended()


def test_cross_hook_producer_suspensions_are_nested(_open_admission):
    first = suspend_mining_producers_until_resumed("sleep")
    second = suspend_mining_producers_until_resumed("reload")
    assert mining_launches_suspended()
    with gpu_mining_producer() as admitted:
        assert not admitted

    gcs.resume_mining_producers(first)
    with gpu_mining_producer() as admitted:
        assert not admitted
    gcs.resume_mining_producers(second)

    assert not mining_launches_suspended()
    with gpu_mining_producer() as admitted:
        assert admitted


def test_capture_waits_for_inflight_producer():
    entered = threading.Event()
    release = threading.Event()
    admitted_seen: list[bool] = []

    def producer():
        with gpu_mining_producer() as admitted:
            admitted_seen.append(admitted)
            entered.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=producer)
    thread.start()
    try:
        assert entered.wait(timeout=5)
        with graph_setup_no_mining():
            # The producer started before the gate: capture must not proceed.
            assert not wait_for_mining_producers_idle(0.2)
            release.set()
            assert wait_for_mining_producers_idle(5)
    finally:
        release.set()
        thread.join(timeout=5)
    assert admitted_seen == [True]


def test_producer_slot_released_on_exception():
    with pytest.raises(RuntimeError, match="boom"), gpu_mining_producer() as admitted:
        assert admitted
        raise RuntimeError("boom")
    assert wait_for_mining_producers_idle(0.1)


class _FakeEvent:
    def __init__(self, done: bool = False):
        self.done = done
        self.queries = 0

    def query(self) -> bool:
        self.queries += 1
        return self.done


@pytest.fixture
def _open_admission(monkeypatch):
    """close_mining_admission is sticky by design; isolate it per test."""
    monkeypatch.setattr(gcs, "_admission_closed", False)
    monkeypatch.setattr(gcs, "_process_global_depth", 0)
    monkeypatch.setattr(gcs, "_launch_suspension_depth", 0)
    monkeypatch.setattr(gcs, "_launch_suspension_ids", set())
    monkeypatch.setattr(gcs, "_producer_suspension_ids", set())
    monkeypatch.setattr(gcs, "_pending_completions", deque())


def test_pending_completion_blocks_idle(_open_admission):
    event = _FakeEvent()
    gcs.register_mining_completion(event)
    assert not gcs.mining_gpu_work_idle()
    assert not gcs.wait_for_mining_producers_idle(timeout=0.05)
    event.done = True
    assert gcs.wait_for_mining_producers_idle(timeout=1.0)
    assert gcs.mining_gpu_work_idle()


def test_completion_poll_does_not_oversleep_timeout(_open_admission, monkeypatch):
    event = _FakeEvent()
    gcs.register_mining_completion(event)
    clock = [100.0]
    sleeps: list[float] = []
    monkeypatch.setattr(gcs.time, "monotonic", lambda: clock[0])

    def advance(duration: float) -> None:
        sleeps.append(duration)
        clock[0] += duration

    monkeypatch.setattr(gcs.time, "sleep", advance)

    assert not gcs.wait_for_mining_producers_idle(timeout=0.003)
    assert sum(sleeps) == pytest.approx(0.003)
    assert max(sleeps) < gcs._COMPLETION_POLL_S


def test_registration_reclaims_completed_work_without_growing(_open_admission):
    """The registry must not accumulate CUDA event handles for the life of the
    process between captures."""
    for _ in range(500):
        gcs.register_mining_completion(_FakeEvent(done=True))
    assert len(gcs._pending_completions) == 0


class _Keepalive:
    """Stands in for device storage the registered launch's kernels still read."""


def test_keepalive_is_held_until_its_own_event_completes(_open_admission):
    """A launch's fence owns whatever the launch dereferences: the registry is
    the last reference once the launching frame returns (a retired
    ``threshold_dev``), so it must not drop it while the event is pending, and
    must drop it when the event completes -- not wait for global quiescence.
    """
    event = _FakeEvent()
    held = _Keepalive()
    keepalive = weakref.ref(held)
    gcs.register_mining_completion(event, keepalive=held)
    del held  # as when the launching frame returns: the registry holds the last ref
    assert keepalive() is not None

    # An unrelated launch retiring must not free work still in flight.
    gcs.register_mining_completion(_FakeEvent(done=True))
    assert keepalive() is not None

    event.done = True
    gcs.register_mining_completion(_FakeEvent(done=True))
    assert keepalive() is None, "the fenced keepalive outlived its completion event"


def test_keepalives_are_released_by_the_idle_sweep(_open_admission):
    """Capture and drain prune with a full sweep, so a lagging event at the
    front must not pin a later launch's keepalive either."""
    slow, fast = _FakeEvent(), _FakeEvent(done=True)
    gcs.register_mining_completion(slow, keepalive=_Keepalive())
    held = _Keepalive()
    fast_keepalive = weakref.ref(held)
    gcs.register_mining_completion(fast, keepalive=held)
    del held

    assert not gcs.mining_gpu_work_idle()  # the sweep ran, slow is still pending
    assert fast_keepalive() is None, "a completed launch's keepalive was pinned behind a slow one"


def test_closed_admission_refuses_producers(_open_admission):
    with gpu_mining_producer() as admitted:
        assert admitted
    gcs.close_mining_admission()
    with gpu_mining_producer() as admitted:
        assert not admitted
    # Sticky: still closed for the next producer.
    with gpu_mining_producer() as admitted:
        assert not admitted


def test_closed_admission_still_waits_out_registered_work(_open_admission):
    event = _FakeEvent()
    gcs.register_mining_completion(event)
    gcs.close_mining_admission()
    assert not gcs.wait_for_mining_producers_idle(timeout=0.05)
    event.done = True
    assert gcs.wait_for_mining_producers_idle(timeout=1.0)


def test_a_new_runtime_lifecycle_reopens_admission(_open_admission):
    """Terminality is per-lifecycle: a drained-and-stopped runtime must not
    leave the process unable to mine (in-process restart, per-test fixtures)."""
    gcs.close_mining_admission()
    with gpu_mining_producer() as admitted:
        assert not admitted
    gcs.open_mining_admission()
    with gpu_mining_producer() as admitted:
        assert admitted


def test_registration_does_not_rescan_an_unfinished_backlog(_open_admission):
    """A backlog that has not completed must not be re-queried by every later
    registration (the front-reclaim bound)."""
    pending = [_FakeEvent() for _ in range(400)]
    for event in pending:
        gcs.register_mining_completion(event)
    queries_before = sum(e.queries for e in pending)
    for _ in range(50):
        gcs.register_mining_completion(_FakeEvent())
    rescans = sum(e.queries for e in pending) - queries_before
    assert rescans <= 50, f"registration rescanned the backlog ({rescans} queries)"


def test_waiters_see_idle_even_when_a_lagging_event_is_at_the_front(_open_admission):
    """Two streams retire out of order; the idle checks do a full sweep so a
    slow event at the front cannot mask real quiescence."""
    slow, fast = _FakeEvent(), _FakeEvent(done=True)
    gcs.register_mining_completion(slow)
    gcs.register_mining_completion(fast)
    assert not gcs.mining_gpu_work_idle()
    slow.done = True
    assert gcs.mining_gpu_work_idle()


def test_gate_depth_is_visible_across_threads(_open_admission):
    """Capture applies process-wide: the framework raises the gate on the worker
    thread while producers may run on other threads."""
    seen = []
    with graph_setup_no_mining():
        t = threading.Thread(target=lambda: seen.append(in_graph_setup_no_mining()))
        t.start()
        t.join()
    assert seen == [True]


def _drain_manager() -> AsyncLoopManager:
    """A real manager, unstarted: __init__ sets every field the drain path reads.

    Constructed rather than stubbed so future manager state cannot silently
    bypass these tests.
    """
    return AsyncLoopManager(MinerRpcConfig(), MinerSettings(no_gateway=True))


def test_flush_completes_while_a_producer_loop_keeps_running(_open_admission):
    """A flush must not require a global gap in producer admission.

    Serving forwards take a producer slot per launch, so a drain that waits
    for ``_active_producers == 0`` while leaving admission open can wait for a
    window that never opens under load. The flush closes admission, waits, and
    re-opens.
    """
    manager = _drain_manager()
    manager.register_drain_predicate(gcs.mining_gpu_work_idle)
    manager.register_drain_barrier(gcs.close_mining_admission)
    manager.register_drain_release(gcs.open_mining_admission)

    stop = threading.Event()
    admitted_after_flush = []

    def producer_loop() -> None:
        while not stop.is_set():
            with gcs.gpu_mining_producer() as admitted:
                if admitted:
                    time.sleep(0.001)

    driver = threading.Thread(target=producer_loop, daemon=True)
    driver.start()
    try:
        assert manager.wait_until_drained(timeout=5.0), (
            "flush timed out while a producer loop was running"
        )
        # Re-opened: the loop must be able to mine again after a flush.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not admitted_after_flush:
            with gcs.gpu_mining_producer() as admitted:
                if admitted:
                    admitted_after_flush.append(True)
        assert admitted_after_flush, "flush left mining admission closed"
    finally:
        stop.set()
        driver.join(timeout=5.0)


def test_overlapping_flushes_run_complete_serial_cycles(_open_admission):
    """Each concurrent caller owns one complete close/wait/reopen cycle."""
    manager = _drain_manager()
    lock = threading.Lock()
    barrier_count = 0
    release_count = 0

    def barrier() -> None:
        nonlocal barrier_count
        gcs.close_mining_admission()
        with lock:
            barrier_count += 1

    def release() -> None:
        nonlocal release_count
        with lock:
            release_count += 1
        gcs.open_mining_admission()

    manager.register_drain_barrier(barrier)
    manager.register_drain_release(release)
    results: list[bool] = []
    flushes = [
        threading.Thread(target=lambda: results.append(manager.wait_until_drained(timeout=5.0)))
        for _ in range(2)
    ]
    for flush in flushes:
        flush.start()
    for flush in flushes:
        flush.join(timeout=5.0)

    assert all(not flush.is_alive() for flush in flushes)
    assert results == [True, True]
    assert barrier_count == 2
    assert release_count == 2
    with gpu_mining_producer() as admitted:
        assert admitted


def test_nested_producer_slots_keep_the_holder_named(_open_admission):
    """One thread can hold nested slots; releasing the inner one must not erase a
    holder the outer one still owns, or the diagnostic reports no holder while
    the count says otherwise -- exactly when it is needed most."""
    with gpu_mining_producer() as outer:
        assert outer
        with gpu_mining_producer() as inner:
            assert inner
            assert not gcs.mining_gpu_work_idle()
        assert not gcs.mining_gpu_work_idle(), "the outer slot is still held"
        assert gcs._producer_slots, "the outer holder must still be named"
    assert gcs.mining_gpu_work_idle()
    assert not gcs._producer_slots, "every slot released, so no holder remains"
