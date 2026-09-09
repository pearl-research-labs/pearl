import math
import shutil
import socket as socket_module
import tempfile
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from miner_base.async_loop_manager import AsyncLoopManager
from miner_base.gateway_client import MinerRpcConfig, MiningClient
from miner_base.settings import MinerSettings
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.dataclasses import MiningJob


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Keep Linux AF_UNIX listener paths below the 108-byte kernel limit."""
    path = Path(tempfile.mkdtemp(prefix="miner-base-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def miner_config():
    return MinerRpcConfig(transport="uds", socket_path="/temp/gateway.pipe")


@pytest.fixture
def mock_mining_job():
    mock_job = Mock(spec=MiningJob)
    mock_job.incomplete_header_bytes = b"test_bytes"
    mock_job.target = "test_target"
    mock_job.cert_version = CertificateVersion.PLAIN_FP8
    return mock_job


class TestAsyncLoopManagerInit:
    def test_init_with_default_settings(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)

        assert isinstance(manager._conf, MinerSettings)
        assert manager._client_config == miner_config
        assert manager._mining_job is None
        assert not manager.started

    def test_init_multiple_instances_independent(self):
        config1 = MinerRpcConfig(transport="uds", socket_path="/temp/gateway1.pipe")
        config2 = MinerRpcConfig(transport="uds", socket_path="/temp/gateway2.pipe")

        manager1 = AsyncLoopManager(config1, MinerSettings())
        manager2 = AsyncLoopManager(config2, MinerSettings())

        assert manager1._stop_event is not manager2._stop_event
        assert manager1._submission_condition is not manager2._submission_condition


class TestAsyncLoopManagerStartStop:
    @patch("miner_base.async_loop_manager._make_client")
    def test_start_initializes_components(self, mock_make_client, miner_config, mock_mining_job):
        mock_client = Mock(spec=MiningClient)
        mock_client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = mock_client

        manager = AsyncLoopManager(miner_config, None)
        manager.start()

        assert manager._thread is not None
        assert manager._thread.daemon is True
        assert manager.started
        assert manager._client is mock_client

        manager.stop()

    @patch("miner_base.async_loop_manager._make_client")
    def test_failed_start_rolls_back_and_can_retry(
        self, mock_make_client, miner_config, mock_mining_job
    ):
        failed_client = Mock(spec=MiningClient)
        failed_client.get_mining_info.side_effect = RuntimeError("gateway unavailable")
        healthy_client = Mock(spec=MiningClient)
        healthy_client.get_mining_info.return_value = mock_mining_job
        mock_make_client.side_effect = [failed_client, healthy_client]
        manager = AsyncLoopManager(miner_config, None)

        with pytest.raises(RuntimeError, match="gateway unavailable"):
            manager.start()

        failed_client.close.assert_called_once_with()
        assert manager._client is None
        assert manager._submission_executor is None
        assert manager._thread is None
        assert not manager.started

        manager.start()
        assert manager._client is healthy_client
        assert manager.started
        manager.stop()

    @patch("miner_base.async_loop_manager._make_client")
    def test_polling_thread_start_failure_unwinds_and_can_retry(
        self, mock_make_client, miner_config, mock_mining_job, monkeypatch
    ):
        failed_client = Mock(spec=MiningClient)
        failed_client.get_mining_info.return_value = mock_mining_job
        healthy_client = Mock(spec=MiningClient)
        healthy_client.get_mining_info.return_value = mock_mining_job
        mock_make_client.side_effect = [failed_client, healthy_client]
        manager = AsyncLoopManager(miner_config, None)

        real_start = threading.Thread.start
        fail_next_start = True

        def fail_once(thread):
            nonlocal fail_next_start
            if fail_next_start:
                fail_next_start = False
                raise RuntimeError("thread exhaustion")
            return real_start(thread)

        monkeypatch.setattr(threading.Thread, "start", fail_once)

        with pytest.raises(RuntimeError, match="thread exhaustion"):
            manager.start()

        failed_client.close.assert_called_once_with()
        assert manager._client is None
        assert manager._submission_executor is None
        assert manager._thread is None
        assert not manager.started
        manager.stop()  # idempotent cleanup must not join an unstarted Thread

        manager.start()
        try:
            assert manager._client is healthy_client
            assert manager._thread is not None and manager._thread.is_alive()
        finally:
            manager.stop()

    @patch("miner_base.async_loop_manager._make_client")
    def test_polling_thread_start_failure_never_opens_submission_admission(
        self, mock_make_client, miner_config, mock_mining_job, monkeypatch
    ):
        failed_client = Mock(spec=MiningClient)
        failed_client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = failed_client
        manager = AsyncLoopManager(miner_config, None)

        polling_start_entered = threading.Event()
        release_failure = threading.Event()
        real_start = threading.Thread.start

        def fail_polling_start(thread):
            if getattr(thread, "_target", None) == manager._run_async_loop:
                polling_start_entered.set()
                assert release_failure.wait(timeout=5)
                raise RuntimeError("thread exhaustion")
            return real_start(thread)

        monkeypatch.setattr(threading.Thread, "start", fail_polling_start)
        start_errors = []

        def start_manager():
            try:
                manager.start()
            except Exception as exc:
                start_errors.append(exc)

        starter = threading.Thread(target=start_manager)
        starter.start()
        assert polling_start_entered.wait(timeout=5)
        try:
            assert manager.get_mining_job() is mock_mining_job
            assert not manager.handle_submit_block(Mock(), mock_mining_job)
            assert manager._pending_submission_count() == 0
            assert not manager._accepting_submissions
        finally:
            release_failure.set()
            starter.join(timeout=5)

        assert not starter.is_alive()
        assert len(start_errors) == 1
        assert "thread exhaustion" in str(start_errors[0])
        assert manager._submission_executor is None
        assert manager._client is None

    @patch("miner_base.async_loop_manager._make_client")
    def test_failed_start_blocks_retry_until_rollback_finishes(
        self, mock_make_client, miner_config, mock_mining_job, monkeypatch
    ):
        failed_client = Mock(spec=MiningClient)
        failed_client.get_mining_info.return_value = mock_mining_job
        healthy_client = Mock(spec=MiningClient)
        healthy_client.get_mining_info.return_value = mock_mining_job
        mock_make_client.side_effect = [failed_client, healthy_client]
        manager = AsyncLoopManager(miner_config, None)

        real_thread_start = threading.Thread.start
        fail_polling_start = True

        def fail_once(thread):
            nonlocal fail_polling_start
            if fail_polling_start and getattr(thread, "_target", None) == manager._run_async_loop:
                fail_polling_start = False
                raise RuntimeError("thread exhaustion")
            return real_thread_start(thread)

        monkeypatch.setattr(threading.Thread, "start", fail_once)
        rollback_stop_entered = threading.Event()
        release_rollback = threading.Event()
        real_manager_stop = manager.stop

        def block_failure_rollback(*args, **kwargs):
            if threading.current_thread().name == "failing-start-owner":
                rollback_stop_entered.set()
                assert release_rollback.wait(timeout=5)
            return real_manager_stop(*args, **kwargs)

        monkeypatch.setattr(manager, "stop", block_failure_rollback)
        start_errors = []
        retry_errors = []
        retry_started = threading.Event()

        def failing_start():
            try:
                manager.start()
            except Exception as exc:
                start_errors.append(exc)

        def retry_start():
            try:
                manager.start()
                retry_started.set()
            except Exception as exc:
                retry_errors.append(exc)

        failing = threading.Thread(target=failing_start, name="failing-start-owner")
        retry = threading.Thread(target=retry_start)
        failing.start()
        assert rollback_stop_entered.wait(timeout=5)
        retry.start()
        assert not retry_started.wait(timeout=0.05), (
            "retry entered a new lifecycle before failed-start rollback completed"
        )

        release_rollback.set()
        failing.join(timeout=5)
        retry.join(timeout=5)
        try:
            assert not failing.is_alive()
            assert not retry.is_alive()
            assert len(start_errors) == 1
            assert "thread exhaustion" in str(start_errors[0])
            assert retry_errors == []
            assert retry_started.is_set()
            assert manager._client is healthy_client
            assert manager.started
        finally:
            manager.stop()

    @patch("miner_base.async_loop_manager._make_client")
    def test_polling_thread_start_failure_racing_stop_has_one_cleanup_owner(
        self, mock_make_client, miner_config, mock_mining_job, monkeypatch
    ):
        failed_client = Mock(spec=MiningClient)
        failed_client.get_mining_info.return_value = mock_mining_job
        healthy_client = Mock(spec=MiningClient)
        healthy_client.get_mining_info.return_value = mock_mining_job
        mock_make_client.side_effect = [failed_client, healthy_client]
        manager = AsyncLoopManager(miner_config, None)

        polling_start_entered = threading.Event()
        release_failure = threading.Event()
        real_start = threading.Thread.start
        failed = False

        def fail_polling_start(thread):
            nonlocal failed
            if not failed and getattr(thread, "_target", None) == manager._run_async_loop:
                polling_start_entered.set()
                assert release_failure.wait(timeout=5)
                failed = True
                raise RuntimeError("thread exhaustion")
            return real_start(thread)

        monkeypatch.setattr(threading.Thread, "start", fail_polling_start)
        start_errors = []
        stop_errors = []

        def _record_error(action, errors):
            try:
                action()
            except Exception as exc:
                errors.append(exc)

        starter = threading.Thread(target=lambda: _record_error(manager.start, start_errors))
        stopper = threading.Thread(target=lambda: _record_error(manager.stop, stop_errors))

        starter.start()
        assert polling_start_entered.wait(timeout=5)
        stopper.start()
        release_failure.set()
        starter.join(timeout=5)
        stopper.join(timeout=5)

        assert not starter.is_alive()
        assert not stopper.is_alive()
        assert len(start_errors) == 1
        assert "thread exhaustion" in str(start_errors[0])
        assert stop_errors == []
        assert manager._thread is None
        assert manager._submission_executor is None
        assert manager._client is None

        manager.start()
        try:
            assert manager._client is healthy_client
            assert manager.started
        finally:
            manager.stop()

    @patch("miner_base.async_loop_manager._make_client")
    def test_start_raises_if_already_started(self, mock_make_client, miner_config, mock_mining_job):
        mock_client = Mock(spec=MiningClient)
        mock_client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = mock_client

        manager = AsyncLoopManager(miner_config, None)
        manager.start()

        with pytest.raises(RuntimeError, match="Already started"):
            manager.start()

        manager.stop()

    @patch("miner_base.async_loop_manager._make_client")
    def test_stop_cleans_up_components(self, mock_make_client, miner_config, mock_mining_job):
        mock_client = Mock(spec=MiningClient)
        mock_client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = mock_client

        manager = AsyncLoopManager(miner_config, None)
        manager.start()
        time.sleep(0.1)
        manager.stop()

        assert manager._thread is None
        assert not manager.started
        mock_client.close.assert_called_once()

    @patch("miner_base.async_loop_manager._make_client")
    def test_stop_joins_blocked_default_executor_poll(
        self, mock_make_client, miner_config, mock_mining_job
    ):
        poll_entered = threading.Event()
        release_poll = threading.Event()
        poll_exited = threading.Event()
        calls = 0
        client = Mock(spec=MiningClient)

        def get_mining_info():
            nonlocal calls
            calls += 1
            if calls == 1:
                return mock_mining_job
            poll_entered.set()
            assert release_poll.wait(timeout=5)
            poll_exited.set()
            return mock_mining_job

        client.get_mining_info.side_effect = get_mining_info
        client.close.side_effect = release_poll.set
        mock_make_client.return_value = client
        manager = AsyncLoopManager(miner_config, None)
        manager.start()
        assert poll_entered.wait(timeout=5)

        manager.stop()

        assert poll_exited.is_set()
        assert not manager.started
        client.close.assert_called_once_with()
        mock_make_client.assert_called_once_with(manager._conf, miner_config)

    @patch("miner_base.async_loop_manager._make_client")
    def test_stop_waits_for_first_connect_and_closes_late_client(
        self, mock_make_client, miner_config
    ):
        late_client = Mock(spec=MiningClient)
        connect_entered = threading.Event()
        release_connect = threading.Event()

        def blocked_connect(*_args):
            connect_entered.set()
            assert release_connect.wait(timeout=5)
            return late_client

        mock_make_client.side_effect = blocked_connect
        manager = AsyncLoopManager(miner_config, None)
        stop_published = threading.Event()
        original_take = manager._take_polling_client

        def observed_take(**kwargs):
            client = original_take(**kwargs)
            if kwargs.get("stopping", False):
                stop_published.set()
            return client

        manager._take_polling_client = observed_take
        start_errors: list[Exception] = []
        stop_errors: list[Exception] = []

        def start_manager():
            try:
                manager.start()
            except Exception as exc:
                start_errors.append(exc)

        def stop_manager():
            try:
                manager.stop()
            except Exception as exc:
                stop_errors.append(exc)

        starter = threading.Thread(target=start_manager)
        stopper = threading.Thread(target=stop_manager)
        starter.start()
        assert connect_entered.wait(timeout=5)
        stopper.start()
        try:
            assert stop_published.wait(timeout=5)
            release_connect.set()
            starter.join(timeout=5)
            stopper.join(timeout=5)
            assert not starter.is_alive()
            assert not stopper.is_alive()
            assert start_errors == []
            assert stop_errors == []
            assert not manager.started
            late_client.close.assert_called_once_with()
            late_client.get_mining_info.assert_not_called()
        finally:
            release_connect.set()
            starter.join(timeout=5)
            stopper.join(timeout=5)

    @patch("miner_base.async_loop_manager._make_client")
    def test_stop_cannot_have_startup_reopen_submission_admission(
        self, mock_make_client, miner_config, mock_mining_job
    ):
        client = Mock(spec=MiningClient)
        client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = client
        manager = AsyncLoopManager(miner_config, None)
        admission_gate_entered = threading.Event()
        release_admission_gate = threading.Event()
        stop_published = threading.Event()
        original_open_admission = manager._open_submission_admission_if_running
        original_take = manager._take_polling_client

        def delayed_open_admission():
            admission_gate_entered.set()
            assert release_admission_gate.wait(timeout=5)
            return original_open_admission()

        def observed_take(**kwargs):
            polling_client = original_take(**kwargs)
            if kwargs.get("stopping", False):
                stop_published.set()
            return polling_client

        manager._open_submission_admission_if_running = delayed_open_admission
        manager._take_polling_client = observed_take
        start_errors: list[Exception] = []
        stop_errors: list[Exception] = []

        def start_manager():
            try:
                manager.start()
            except Exception as exc:
                start_errors.append(exc)

        def stop_manager():
            try:
                manager.stop()
            except Exception as exc:
                stop_errors.append(exc)

        starter = threading.Thread(target=start_manager)
        stopper = threading.Thread(target=stop_manager)
        starter.start()
        assert admission_gate_entered.wait(timeout=5)
        stopper.start()
        try:
            assert stop_published.wait(timeout=5)
            assert not manager.handle_submit_block(Mock(), mock_mining_job)
            release_admission_gate.set()
            starter.join(timeout=5)
            stopper.join(timeout=5)
            assert not starter.is_alive()
            assert not stopper.is_alive()
            assert start_errors == []
            assert stop_errors == []
            assert manager._pending_submission_count() == 0
            assert not manager.started
        finally:
            release_admission_gate.set()
            starter.join(timeout=5)
            stopper.join(timeout=5)

    @patch("miner_base.async_loop_manager._make_client")
    def test_stop_owns_client_that_finishes_connecting_during_shutdown(
        self, mock_make_client, miner_config, mock_mining_job
    ):
        initial = Mock(spec=MiningClient)
        initial.get_mining_info.side_effect = [
            mock_mining_job,
            RuntimeError("connection retired"),
        ]
        replacement = Mock(spec=MiningClient)
        connect_entered = threading.Event()
        release_connect = threading.Event()
        calls = 0

        def make_client(*_args):
            nonlocal calls
            calls += 1
            if calls == 1:
                return initial
            connect_entered.set()
            assert release_connect.wait(timeout=5)
            return replacement

        mock_make_client.side_effect = make_client
        manager = AsyncLoopManager(miner_config, None)
        manager.start()
        assert connect_entered.wait(timeout=5)

        stopped = threading.Event()
        stop_published = threading.Event()
        original_take = manager._take_polling_client

        def observed_take(**kwargs):
            client = original_take(**kwargs)
            if kwargs.get("stopping", False):
                stop_published.set()
            return client

        manager._take_polling_client = observed_take
        stopper = threading.Thread(target=lambda: (manager.stop(), stopped.set()))
        stopper.start()
        try:
            # The constructor is not yet holding a published socket, so stop
            # cannot close it; the configured connection timeout bounds this
            # interval in production. Once construction returns, stopping
            # must make publication fail and close the replacement immediately.
            assert stop_published.wait(timeout=5)
            release_connect.set()
            stopper.join(timeout=5)
            assert stopped.is_set()
            assert not manager.started
            replacement.close.assert_called_once_with()
            replacement.get_mining_info.assert_not_called()
        finally:
            release_connect.set()
            stopper.join(timeout=5)

    def test_stop_without_start_is_safe(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)
        manager.stop()

    def test_ordered_flush_keeps_submission_open_until_continuations_finish(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)
        continuation_pending = True
        barrier_ran = False

        def barrier() -> None:
            nonlocal barrier_ran
            barrier_ran = True
            assert manager._accepting_submissions

        def continuation_idle() -> bool:
            nonlocal continuation_pending
            assert barrier_ran
            assert manager._accepting_submissions
            continuation_pending = False
            return True

        manager._submission_executor = Mock()
        manager._accepting_submissions = True
        manager.register_drain_barrier(barrier)
        manager.register_drain_predicate(continuation_idle)

        assert manager.wait_until_drained(timeout=1.0)
        assert not continuation_pending
        assert manager._accepting_submissions

    def test_ordered_flush_closes_submission_before_final_quiescence(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)
        phase_two_observed = False

        def submissions_and_hashes_idle() -> bool:
            nonlocal phase_two_observed
            phase_two_observed = True
            assert not manager._accepting_submissions
            return True

        manager._submission_executor = Mock()
        manager._accepting_submissions = True
        manager._submissions_drained = submissions_and_hashes_idle

        assert manager.wait_until_drained(timeout=1.0)
        assert phase_two_observed
        assert manager._accepting_submissions

    def test_ordered_flush_queues_two_winners_with_one_submission_slot(
        self, miner_config, mock_mining_job
    ):
        manager = AsyncLoopManager(
            miner_config,
            MinerSettings(submission_inflight_limit=1),
        )
        manager._mining_job = mock_mining_job
        manager._accepting_submissions = True
        executor = ThreadPoolExecutor(max_workers=1)
        manager._submission_executor = executor
        first_entered = threading.Event()
        release_first = threading.Event()
        submitted: list[object] = []

        def submit(opening, _job) -> bool:
            submitted.append(opening)
            if len(submitted) == 1:
                first_entered.set()
                assert release_first.wait(timeout=5)
            return True

        manager._submit_block = submit
        continuations_done = threading.Event()

        def enqueue_winners() -> None:
            assert manager.handle_submit_block(Mock(), mock_mining_job)
            assert manager.handle_submit_block(Mock(), mock_mining_job)
            continuations_done.set()

        continuation = threading.Thread(target=enqueue_winners)
        result: list[bool] = []
        flush = threading.Thread(
            target=lambda: result.append(manager.wait_until_drained(timeout=5))
        )
        try:
            continuation.start()
            assert first_entered.wait(timeout=5)
            manager.register_drain_predicate(continuations_done.is_set)
            flush.start()
            time.sleep(0.05)
            assert flush.is_alive()
            assert manager._accepting_submissions

            release_first.set()
            continuation.join(timeout=5)
            flush.join(timeout=5)

            assert not continuation.is_alive()
            assert not flush.is_alive()
            assert result == [True]
            assert len(submitted) == 2
            assert manager._pending_submission_count() == 0
        finally:
            release_first.set()
            continuation.join(timeout=5)
            flush.join(timeout=5)
            executor.shutdown(wait=True, cancel_futures=True)

    def test_stop_during_flush_does_not_reopen_admission(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)
        phase_two_entered = threading.Event()
        release_phase_two = threading.Event()
        reopen = Mock()

        def submissions_and_hashes_idle() -> bool:
            assert not manager._accepting_submissions
            phase_two_entered.set()
            assert release_phase_two.wait(timeout=5)
            return True

        manager._submission_executor = Mock()
        manager._accepting_submissions = True
        manager._submissions_drained = submissions_and_hashes_idle
        manager.register_drain_release(reopen)
        flush = threading.Thread(target=manager.wait_until_drained)
        flush.start()
        assert phase_two_entered.wait(timeout=5)

        manager.stop(wait_for_submissions=False)
        release_phase_two.set()
        flush.join(timeout=5)

        assert not flush.is_alive()
        assert not manager._accepting_submissions
        reopen.assert_not_called()

    @patch("miner_base.async_loop_manager._make_client")
    def test_active_flush_crossing_restart_cannot_report_success(
        self, mock_make_client, miner_config, mock_mining_job
    ):
        client = Mock(spec=MiningClient)
        client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = client
        manager = AsyncLoopManager(miner_config, None)
        predicate_entered = threading.Event()
        release_predicate = threading.Event()

        def blocked_predicate() -> bool:
            predicate_entered.set()
            assert release_predicate.wait(timeout=5)
            return True

        manager.register_drain_predicate(blocked_predicate)
        manager.start()
        result: list[bool] = []
        flush = threading.Thread(target=lambda: result.append(manager.wait_until_drained()))
        flush.start()
        assert predicate_entered.wait(timeout=5)
        try:
            manager.stop()
            manager.start()
        finally:
            release_predicate.set()
            flush.join(timeout=5)
            manager.stop()

        assert not flush.is_alive()
        assert result == [False]

    @patch("miner_base.async_loop_manager._make_client")
    def test_flush_waiting_across_restart_does_not_enter_new_lifecycle(
        self, mock_make_client, miner_config, mock_mining_job
    ):
        client = Mock(spec=MiningClient)
        client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = client
        manager = AsyncLoopManager(miner_config, None)
        manager.start()
        manager._drain_lock.acquire()
        barrier = Mock()
        manager.register_drain_barrier(barrier)
        result: list[bool] = []
        flush = threading.Thread(target=lambda: result.append(manager.wait_until_drained()))
        flush.start()
        try:
            manager.stop()
            manager.start()
        finally:
            manager._drain_lock.release()
            flush.join(timeout=5)
            manager.stop()

        assert not flush.is_alive()
        assert result == [False]
        barrier.assert_not_called()

    def test_nonwaiting_stop_owns_submission_abort(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)
        manager.abort_pending_submissions = Mock()

        manager.stop(wait_for_submissions=False)

        manager.abort_pending_submissions.assert_called_once_with()
        assert not manager._restart_permitted

    def test_wait_until_drained_returns_true_when_drained(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)
        assert manager.wait_until_drained() is True

    def test_wait_until_drained_returns_false_when_stopping(self, miner_config):
        # Unreported hashes plus a set stop event: the drain must report
        # failure (False) rather than a false "done".
        manager = AsyncLoopManager(miner_config, None)
        manager.increment_credited_hashes(5)
        manager._stop_event.set()
        assert manager.wait_until_drained() is False

    @pytest.mark.parametrize("timeout", [0.0, -1.0])
    def test_finite_expired_drain_timeouts_still_return_immediately(self, miner_config, timeout):
        manager = AsyncLoopManager(miner_config, None)
        manager.increment_credited_hashes(1)
        barrier = Mock()
        manager.register_drain_barrier(barrier)

        assert not manager.wait_until_drained(timeout=timeout)
        barrier.assert_not_called()

    def test_drain_predicate_completion_after_deadline_is_failure(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)
        predicate_entered = threading.Event()
        now = [10.0]

        def predicate_that_consumes_the_deadline() -> bool:
            predicate_entered.set()
            now[0] = 11.1
            return True

        manager.register_drain_predicate(predicate_that_consumes_the_deadline)

        with patch(
            "miner_base.async_loop_manager.time.monotonic",
            side_effect=lambda: now[0],
        ):
            assert not manager.wait_until_drained(timeout=1.0)
        assert predicate_entered.is_set()

    @pytest.mark.parametrize("timeout", [math.nan, math.inf, -math.inf])
    def test_manager_boundaries_reject_nonfinite_timeouts(
        self, miner_config, mock_mining_job, timeout
    ):
        manager = AsyncLoopManager(miner_config, None)

        with pytest.raises(ValueError, match="timeout must be finite"):
            manager.wait_until_drained(timeout=timeout)
        with pytest.raises(ValueError, match="timeout must be finite"):
            manager.handle_submit_block(Mock(), mock_mining_job, timeout=timeout)

    @patch("miner_base.async_loop_manager._make_client")
    def test_restart_after_stop(self, mock_make_client, miner_config, mock_mining_job):
        # A cleanly stopped manager must be restartable with fresh loop state.
        mock_client = Mock(spec=MiningClient)
        mock_client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = mock_client

        manager = AsyncLoopManager(miner_config, None)
        manager.start()
        manager.stop()

        manager.start()
        try:
            assert manager._thread is not None and manager._thread.is_alive()
            assert not manager._stop_event.is_set()
        finally:
            manager.stop()


class TestAsyncLoopManagerGetMiningJob:
    def test_get_mining_job_returns_current_job(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)

        assert manager.get_mining_job() is None

        mock_job = Mock(spec=MiningJob)
        manager._mining_job = mock_job

        assert manager.get_mining_job() is mock_job

    def test_get_mining_job_concurrent_access_no_crash(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)
        results = []

        def reader():
            for _ in range(100):
                results.append(manager.get_mining_job())
                time.sleep(0.001)

        def writer():
            for _ in range(50):
                manager._mining_job = Mock(spec=MiningJob)
                time.sleep(0.002)

        reader_thread = threading.Thread(target=reader)
        writer_thread = threading.Thread(target=writer)

        reader_thread.start()
        writer_thread.start()
        reader_thread.join()
        writer_thread.join()

        assert len(results) == 100


class TestAsyncLoopManagerCreditedHashes:
    def test_accumulates_and_rejects_negative(self, miner_config):
        manager = AsyncLoopManager(miner_config, MinerSettings())

        manager.increment_credited_hashes(7)
        manager.increment_credited_hashes(11)
        assert manager._inner_hash_counter == 18

        with pytest.raises(ValueError, match="non-negative"):
            manager.increment_credited_hashes(-1)

    def test_zero_is_a_noop(self, miner_config):
        manager = AsyncLoopManager(miner_config, MinerSettings())
        manager.increment_credited_hashes(0)
        assert manager._inner_hash_counter == 0

    def test_concurrent_increments_are_exact(self, miner_config):
        manager = AsyncLoopManager(miner_config, MinerSettings())
        per_thread, threads_n = 1000, 8

        def add():
            for _ in range(per_thread):
                manager.increment_credited_hashes(3)

        threads = [threading.Thread(target=add) for _ in range(threads_n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert manager._inner_hash_counter == 3 * per_thread * threads_n


class TestAsyncLoopManagerMiningJobCallbacks:
    def test_register_multiple_callbacks(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)

        callback1 = Mock()
        callback2 = Mock()

        manager.register_mining_job_changed_callback(callback1)
        manager.register_mining_job_changed_callback(callback2)

        assert len(manager._mining_job_changed_callbacks) == 2
        assert callback1 in manager._mining_job_changed_callbacks
        assert callback2 in manager._mining_job_changed_callbacks


_SUBMISSION_DRAIN_TIMEOUT_S = 10.0
_DRAIN_RELEASE_TIMEOUT_S = 2.0


class TestAsyncLoopManagerSubmissionDrain:
    @patch("miner_base.async_loop_manager._make_client")
    def test_submission_capacity_backpressures_without_growing_the_queue(
        self, mock_make_client, miner_config, mock_mining_job
    ):
        mock_client = Mock(spec=MiningClient)
        mock_client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = mock_client
        manager = AsyncLoopManager(miner_config, MinerSettings(submission_inflight_limit=2))
        entered = threading.Semaphore(0)
        release = threading.Semaphore(0)

        def _blocked_submit(*_args) -> bool:
            entered.release()
            assert release.acquire(timeout=_SUBMISSION_DRAIN_TIMEOUT_S)
            return True

        manager._submit_block = _blocked_submit
        manager.start()
        try:
            assert manager.handle_submit_block(Mock(), mock_mining_job)
            assert manager.handle_submit_block(Mock(), mock_mining_job)
            assert entered.acquire(timeout=_SUBMISSION_DRAIN_TIMEOUT_S)
            assert entered.acquire(timeout=_SUBMISSION_DRAIN_TIMEOUT_S)

            accepted: list[bool] = []
            third = threading.Thread(
                target=lambda: accepted.append(manager.handle_submit_block(Mock(), mock_mining_job))
            )
            third.start()
            time.sleep(0.1)
            assert third.is_alive()
            assert manager._pending_submission_count() == 2

            release.release()
            third.join(timeout=_SUBMISSION_DRAIN_TIMEOUT_S)
            assert not third.is_alive()
            assert accepted == [True]
            assert entered.acquire(timeout=_SUBMISSION_DRAIN_TIMEOUT_S)
            assert manager._pending_submission_count() == 2

            release.release()
            release.release()
            assert manager.wait_until_drained(timeout=_SUBMISSION_DRAIN_TIMEOUT_S)
            assert manager.blocks_submitted == 3
        finally:
            manager.abort_pending_submissions()
            release.release()
            release.release()
            manager.stop()

    @patch("miner_base.async_loop_manager._make_client")
    def test_launch_admission_does_not_consume_submission_capacity(
        self, mock_make_client, miner_config, mock_mining_job
    ):
        mock_client = Mock(spec=MiningClient)
        mock_client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = mock_client
        manager = AsyncLoopManager(miner_config, MinerSettings(submission_inflight_limit=1))
        manager.start()
        try:
            for _ in range(3):
                with manager.mining_launch_admission(mock_mining_job) as decision:
                    assert decision.launch and decision.retain_winner
            assert manager._pending_submission_count() == 0
        finally:
            manager.stop(wait_for_submissions=False)

    @patch("miner_base.async_loop_manager._make_client")
    def test_bounded_handoff_timeout_transfers_no_ownership(
        self, mock_make_client, miner_config, mock_mining_job
    ):
        mock_client = Mock(spec=MiningClient)
        mock_client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = mock_client
        manager = AsyncLoopManager(miner_config, MinerSettings(submission_inflight_limit=1))
        entered = threading.Event()
        release = threading.Event()

        def _blocked_submit(*_args) -> bool:
            entered.set()
            assert release.wait(timeout=_SUBMISSION_DRAIN_TIMEOUT_S)
            return True

        manager._submit_block = _blocked_submit
        manager.start()
        try:
            assert manager.handle_submit_block(Mock(), mock_mining_job)
            assert entered.wait(timeout=_SUBMISSION_DRAIN_TIMEOUT_S)
            with patch("miner_base.async_loop_manager._LOGGER") as logger:
                assert not manager.handle_submit_block(Mock(), mock_mining_job, timeout=0.02)
            assert manager._pending_submission_count() == 1
            assert "accounting may lag" in logger.warning.call_args.args[0]
            assert "candidate was not queued" in logger.error.call_args.args[0]
        finally:
            release.set()
            manager.stop()

    @pytest.mark.parametrize(
        ("outcome", "expected_log"),
        [
            ("capacity", "capacity recovered"),
            ("job", "mining job replaced"),
            ("closed", "admission closed by drain or shutdown"),
        ],
    )
    def test_submission_capacity_wait_exit_is_logged(
        self,
        miner_config,
        mock_mining_job,
        outcome,
        expected_log,
    ):
        manager = AsyncLoopManager(
            miner_config,
            MinerSettings(submission_inflight_limit=1),
        )
        executor = Mock()
        manager._mining_job = mock_mining_job
        manager._submission_executor = executor
        manager._accepting_submissions = True
        manager._pending_submissions = 1
        result = []

        with patch("miner_base.async_loop_manager._LOGGER") as logger:
            waiter = threading.Thread(
                target=lambda: result.append(
                    manager._reserve_submission(mock_mining_job, timeout=1.0)
                )
            )
            waiter.start()
            deadline = time.monotonic() + _SUBMISSION_DRAIN_TIMEOUT_S
            while not logger.warning.called and time.monotonic() < deadline:
                time.sleep(0.01)
            assert logger.warning.called, "capacity waiter did not report saturation"

            if outcome == "capacity":
                with manager._submission_condition:
                    manager._pending_submissions = 0
                    manager._submission_condition.notify_all()
            elif outcome == "job":
                manager._replace_mining_job(Mock(spec=MiningJob))
            else:
                with manager._submission_condition:
                    manager._accepting_submissions = False
                    manager._submission_condition.notify_all()

            waiter.join(timeout=_SUBMISSION_DRAIN_TIMEOUT_S)
            assert not waiter.is_alive()
            assert expected_log in logger.info.call_args.args[0]

        if outcome == "capacity":
            assert result[0] is executor
        else:
            assert result == [None]

    @patch("miner_base.async_loop_manager._make_client")
    def test_submission_backpressure_releases_job_publication_and_revalidates(
        self, mock_make_client, miner_config, mock_mining_job
    ):
        replacement_job = Mock(spec=MiningJob)
        mock_client = Mock(spec=MiningClient)
        gateway_poll_release = threading.Event()
        gateway_poll_calls = 0

        def controlled_gateway_poll():
            nonlocal gateway_poll_calls
            gateway_poll_calls += 1
            if gateway_poll_calls == 1:
                return mock_mining_job
            assert gateway_poll_release.wait(timeout=_SUBMISSION_DRAIN_TIMEOUT_S)
            return replacement_job

        mock_client.get_mining_info.side_effect = controlled_gateway_poll
        mock_make_client.return_value = mock_client
        manager = AsyncLoopManager(miner_config, MinerSettings(submission_inflight_limit=1))
        submission_entered = threading.Event()
        submission_release = threading.Event()

        def _blocked_submit(*_args) -> bool:
            submission_entered.set()
            assert submission_release.wait(timeout=_SUBMISSION_DRAIN_TIMEOUT_S)
            return True

        manager._submit_block = _blocked_submit
        manager.start()
        waiting_result: list[bool] = []
        publication_result: list[bool] = []
        capacity_wait_entered = threading.Event()
        original_condition_wait = manager._submission_condition.wait

        def observed_condition_wait(timeout=None):
            capacity_wait_entered.set()
            return original_condition_wait(timeout=timeout)

        manager._submission_condition.wait = observed_condition_wait
        waiter = threading.Thread(
            target=lambda: waiting_result.append(
                manager.handle_submit_block(Mock(), mock_mining_job)
            )
        )
        publisher = threading.Thread(
            target=lambda: publication_result.append(manager._publish_mining_job(replacement_job))
        )
        try:
            assert manager.handle_submit_block(Mock(), mock_mining_job)
            assert submission_entered.wait(timeout=_SUBMISSION_DRAIN_TIMEOUT_S)
            waiter.start()
            assert capacity_wait_entered.wait(timeout=_SUBMISSION_DRAIN_TIMEOUT_S), (
                "second winner never entered submission backpressure"
            )
            assert waiter.is_alive()

            publisher.start()
            publisher.join(timeout=_DRAIN_RELEASE_TIMEOUT_S)
            assert not publisher.is_alive(), (
                "submission backpressure blocked mining-job publication"
            )
            assert publication_result == [True]

            waiter.join(timeout=_DRAIN_RELEASE_TIMEOUT_S)
            assert not waiter.is_alive()
            assert waiting_result == [False], "stale winner reserved capacity after job replacement"
            assert manager.get_mining_job() is replacement_job
            assert manager._pending_submission_count() == 1
        finally:
            manager._submission_condition.wait = original_condition_wait
            submission_release.set()
            gateway_poll_release.set()
            if waiter.ident is not None:
                waiter.join(timeout=_SUBMISSION_DRAIN_TIMEOUT_S)
            if publisher.ident is not None:
                publisher.join(timeout=_SUBMISSION_DRAIN_TIMEOUT_S)
            manager.stop()


def test_submission_failure_log_does_not_render_proof_or_job_locals(miner_config):
    manager = AsyncLoopManager(miner_config, None)
    manager._pending_submissions = 1
    future: Future[bool] = Future()
    secret = "proof-and-job-payload-sentinel"
    future.set_exception(RuntimeError(secret))

    with patch("miner_base.async_loop_manager._LOGGER") as logger:
        manager._submission_finished(future)

    logger.error.assert_called_once_with("Block submission failed (RuntimeError).")
    assert secret not in str(logger.mock_calls)
    assert manager._pending_submission_count() == 0


def test_abort_pending_submissions_interrupts_a_blocked_submission(tmp_path):
    """Closing an in-flight client must wake an unbounded gateway read."""
    sock_path = str(tmp_path / "gw.sock")
    server = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(1)
    accepted: list[socket_module.socket] = []

    def _accept_and_go_silent() -> None:
        conn, _ = server.accept()
        accepted.append(conn)

    accepter = threading.Thread(target=_accept_and_go_silent, daemon=True)
    accepter.start()

    config = MinerRpcConfig(transport="uds", socket_path=sock_path)
    client = MiningClient(config)
    manager = AsyncLoopManager(config, None)
    with manager._inflight_lock:
        manager._inflight_clients.add(client)

    raised: list[str] = []

    def _blocked_submission() -> None:
        try:
            client.client.call("submitPlainProof", {})
        except Exception as exc:
            raised.append(type(exc).__name__)

    caller = threading.Thread(target=_blocked_submission, daemon=True)
    caller.start()
    time.sleep(0.3)
    assert caller.is_alive(), "the submission should be blocked on the silent gateway"

    try:
        manager.abort_pending_submissions()
        caller.join(timeout=_SUBMISSION_DRAIN_TIMEOUT_S)
        assert not caller.is_alive(), "abort did not interrupt the blocked submission"
        assert raised, "the interrupted read should surface as an error"
    finally:
        for conn in accepted:
            conn.close()
        server.close()


def test_no_gateway_refuses_before_proof_construction(miner_config):
    manager = AsyncLoopManager(miner_config, MinerSettings(no_gateway=True))
    job = Mock(spec=MiningJob)

    with (
        patch(
            "miner_base.async_loop_manager.submit_opened_block",
            side_effect=AssertionError("offline mode constructed a proof"),
        ),
        patch(
            "miner_base.async_loop_manager._make_client",
            side_effect=AssertionError("offline mode opened a submission client"),
        ),
    ):
        assert not manager.handle_submit_block(Mock(), job)
        assert not manager._submit_block(Mock(), job)

    assert manager.blocks_submitted == 0
    assert manager._pending_submission_count() == 0


def test_submission_registered_after_abort_is_refused_not_left_blocking(miner_config):
    """A submission arriving after the abort snapshot must refuse its own client."""
    manager = AsyncLoopManager(miner_config, None)
    manager.abort_pending_submissions()
    closed: list[bool] = []

    class _RecordingClient:
        def submit_plain_proof(self, *args, **kwargs):
            raise AssertionError("submit_plain_proof ran after abort began")

        def close(self):
            closed.append(True)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return None

    job = Mock(spec=MiningJob)
    job.incomplete_header_bytes = b"x"
    with (
        patch("miner_base.async_loop_manager._make_client", return_value=_RecordingClient()),
        patch("miner_base.async_loop_manager.submit_opened_block", return_value=Mock()),
    ):
        result = manager._submit_block(Mock(), job)

    assert result is False
    assert closed == [True]
    with manager._inflight_lock:
        assert not manager._inflight_clients
