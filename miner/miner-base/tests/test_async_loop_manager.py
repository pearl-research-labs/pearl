"""Behavioral tests for AsyncLoopManager polling, admission, backpressure, and drain."""

import math
import shutil
import socket as socket_module
import tempfile
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from miner_base.async_loop_manager import AsyncLoopManager
from miner_base.gateway_client import MiningClient
from miner_base.settings import MinerSettings
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.dataclasses import MiningJob
from pearl_gateway.config import MinerRpcConfig

_TIMEOUT_S = 5.0


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Keep Linux AF_UNIX listener paths below the 108-byte kernel limit."""
    path = Path(tempfile.mkdtemp(prefix="miner-base-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def miner_config() -> MinerRpcConfig:
    return MinerRpcConfig(transport="uds", socket_path="/temp/gateway.pipe")


@pytest.fixture
def mock_mining_job() -> Mock:
    job = Mock(spec=MiningJob)
    job.incomplete_header_bytes = b"test_bytes"
    job.target = 12345
    job.cert_version = CertificateVersion.PLAIN_FP8
    job.ancestor_headers = [b"parent_header"]
    return job


class TestLifecycle:
    @patch("miner_base.async_loop_manager._make_client")
    def test_start_stop_and_restart(self, mock_make_client, miner_config, mock_mining_job):
        client = Mock(spec=MiningClient)
        client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = client

        manager = AsyncLoopManager(miner_config, None)
        assert not manager.started
        manager.stop()  # stop before start is safe

        manager.start()
        assert manager.started
        assert manager.get_mining_job() is mock_mining_job
        with pytest.raises(RuntimeError, match="Already started"):
            manager.start()

        manager.stop()
        assert not manager.started
        client.close.assert_called()

        # Clean stop permits restart
        manager.start()
        assert manager.started
        manager.stop()

    @patch("miner_base.async_loop_manager._make_client")
    def test_failed_start_rolls_back_and_can_retry(
        self, mock_make_client, miner_config, mock_mining_job
    ):
        failed = Mock(spec=MiningClient)
        failed.get_mining_info.side_effect = RuntimeError("gateway unavailable")
        healthy = Mock(spec=MiningClient)
        healthy.get_mining_info.return_value = mock_mining_job
        mock_make_client.side_effect = [failed, healthy]

        manager = AsyncLoopManager(miner_config, None)
        with pytest.raises(RuntimeError, match="gateway unavailable"):
            manager.start()

        failed.close.assert_called_once_with()
        assert not manager.started
        assert manager._submission_executor is None

        manager.start()
        try:
            assert manager.started
            assert manager.get_mining_job() is mock_mining_job
        finally:
            manager.stop()

    @patch("miner_base.async_loop_manager._make_client")
    def test_startup_fail_open_continues_without_job(self, mock_make_client, miner_config):
        mock_make_client.side_effect = ConnectionRefusedError("gateway down")
        manager = AsyncLoopManager(miner_config, None, startup_fail_open=True)
        manager.start()
        try:
            assert manager.started
            assert manager.get_mining_job() is None
        finally:
            manager.stop()

    @patch("miner_base.async_loop_manager._make_client")
    def test_abandoned_shutdown_forbids_restart(
        self, mock_make_client, miner_config, mock_mining_job
    ):
        client = Mock(spec=MiningClient)
        client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = client
        manager = AsyncLoopManager(miner_config, None)
        manager.start()
        manager.stop(wait_for_submissions=False)

        with pytest.raises(RuntimeError, match="Cannot restart"):
            manager.start()

    @patch("miner_base.async_loop_manager._make_client")
    def test_stale_job_retired_after_repeated_poll_failures(
        self, mock_make_client, miner_config, mock_mining_job, monkeypatch
    ):
        monkeypatch.setattr("miner_base.async_loop_manager._POLL_INTERVAL_SECONDS", 0.01)
        client = Mock(spec=MiningClient)
        client.get_mining_info.side_effect = [
            mock_mining_job,
            RuntimeError("e1"),
            RuntimeError("e2"),
            RuntimeError("e3"),
            RuntimeError("e4"),
        ]
        mock_make_client.return_value = client

        changed = threading.Event()
        manager = AsyncLoopManager(miner_config, None)
        manager.register_mining_job_changed_callback(
            lambda: changed.set() if manager.get_mining_job() is None else None
        )
        manager.start()
        try:
            assert changed.wait(timeout=_TIMEOUT_S)
            assert manager.get_mining_job() is None
        finally:
            manager.stop()


class TestAdmissionAndAccounting:
    def test_credited_hashes(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)
        assert manager.credited_hashes == 0
        manager.increment_credited_hashes(100)
        manager.increment_credited_hashes(50)
        assert manager.credited_hashes == 150
        with pytest.raises(ValueError, match="non-negative"):
            manager.increment_credited_hashes(-1)

    @patch("miner_base.async_loop_manager._make_client")
    def test_launch_admission_modes(self, mock_make_client, miner_config, mock_mining_job):
        client = Mock(spec=MiningClient)
        client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = client

        for settings, expected_retain in [
            (MinerSettings(no_gateway=False, skip_block_submission=False), True),
            (MinerSettings(no_gateway=False, skip_block_submission=True), False),
        ]:
            manager = AsyncLoopManager(miner_config, settings)
            manager.start()
            try:
                with manager.mining_launch_admission(mock_mining_job) as d:
                    assert d.launch is True
                    assert d.retain_winner is expected_retain
                with manager.mining_launch_admission(Mock(spec=MiningJob)) as d:
                    assert d.launch is False and d.retain_winner is False
            finally:
                manager.stop()

    def test_no_gateway_refuses_before_proof_construction(self, miner_config):
        manager = AsyncLoopManager(miner_config, MinerSettings(no_gateway=True))
        manager.start()
        try:
            job = manager.get_mining_job()
            assert job is not None
            with manager.mining_launch_admission(job) as d:
                assert d.launch is True and d.retain_winner is False
            with (
                patch("miner_base.async_loop_manager.submit_opened_block") as mock_submit,
                patch("miner_base.async_loop_manager._make_client") as mock_client,
            ):
                assert not manager.handle_submit_block(Mock(), job)
            mock_submit.assert_not_called()
            mock_client.assert_not_called()
            assert manager.submission_acknowledgements == 0
        finally:
            manager.stop()

    def test_non_finite_timeout_rejected(self, miner_config, mock_mining_job):
        manager = AsyncLoopManager(miner_config, MinerSettings(no_gateway=True))
        for bad in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="finite"):
                manager.handle_submit_block(Mock(), mock_mining_job, timeout=bad)
            with pytest.raises(ValueError, match="finite"):
                manager.wait_until_drained(timeout=bad)


class TestSubmissionBackpressureAndDrain:
    @patch("miner_base.async_loop_manager._make_client")
    def test_capacity_backpressures_and_drains(
        self, mock_make_client, miner_config, mock_mining_job
    ):
        client = Mock(spec=MiningClient)
        client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = client
        manager = AsyncLoopManager(miner_config, MinerSettings(submission_inflight_limit=2))

        entered = threading.Semaphore(0)
        release = threading.Semaphore(0)

        def _blocked_submit(*_args) -> bool:
            entered.release()
            assert release.acquire(timeout=_TIMEOUT_S)
            return True

        manager._submit_block = _blocked_submit
        manager.start()
        try:
            assert manager.handle_submit_block(Mock(), mock_mining_job)
            assert manager.handle_submit_block(Mock(), mock_mining_job)
            assert entered.acquire(timeout=_TIMEOUT_S)
            assert entered.acquire(timeout=_TIMEOUT_S)

            # Bounded timeout drops without consuming ownership.
            assert not manager.handle_submit_block(Mock(), mock_mining_job, timeout=0.02)

            # Unbounded third handoff blocks until a slot frees.
            accepted: list[bool] = []
            third = threading.Thread(
                target=lambda: accepted.append(manager.handle_submit_block(Mock(), mock_mining_job))
            )
            third.start()
            time.sleep(0.05)
            assert third.is_alive()

            release.release()
            third.join(timeout=_TIMEOUT_S)
            assert not third.is_alive()
            assert accepted == [True]
            assert entered.acquire(timeout=_TIMEOUT_S)

            release.release()
            release.release()
            assert manager.wait_until_drained(timeout=_TIMEOUT_S)
            assert manager.submission_acknowledgements == 3
        finally:
            release.release()
            release.release()
            manager.stop()

    @patch("miner_base.async_loop_manager._make_client")
    def test_job_change_wakes_capacity_waiter(
        self, mock_make_client, miner_config, mock_mining_job
    ):
        client = Mock(spec=MiningClient)
        client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = client
        manager = AsyncLoopManager(miner_config, MinerSettings(submission_inflight_limit=1))

        entered = threading.Event()
        release = threading.Event()
        manager._submit_block = lambda *_: (entered.set(), release.wait(_TIMEOUT_S), True)[2]
        manager.start()
        try:
            assert manager.handle_submit_block(Mock(), mock_mining_job)
            assert entered.wait(timeout=_TIMEOUT_S)

            result: list[bool] = []
            waiter = threading.Thread(
                target=lambda: result.append(
                    manager.handle_submit_block(Mock(), mock_mining_job, timeout=_TIMEOUT_S)
                )
            )
            waiter.start()
            time.sleep(0.05)
            assert waiter.is_alive()

            manager._publish_mining_job(Mock(spec=MiningJob))
            waiter.join(timeout=_TIMEOUT_S)
            assert not waiter.is_alive()
            assert result == [False]
        finally:
            release.set()
            manager.stop()

    @patch("miner_base.async_loop_manager._make_client")
    def test_ordered_drain_phases(self, mock_make_client, miner_config, mock_mining_job):
        client = Mock(spec=MiningClient)
        client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = client
        manager = AsyncLoopManager(miner_config, None)
        manager._submit_block = lambda *_: True

        events: list[str] = []
        predicate_done = threading.Event()
        manager.register_drain_barrier(lambda: events.append("barrier"))
        manager.register_drain_predicate(predicate_done.is_set)
        manager.register_drain_release(lambda: events.append("release"))

        manager.start()
        try:

            def finish_continuation():
                time.sleep(0.05)
                # Proof submission must remain open while phase 1 waits on continuations.
                assert manager.handle_submit_block(Mock(), mock_mining_job)
                events.append("continuation_submitted")
                predicate_done.set()

            t = threading.Thread(target=finish_continuation)
            t.start()
            assert manager.wait_until_drained(timeout=_TIMEOUT_S)
            t.join(timeout=_TIMEOUT_S)
            assert events == ["barrier", "continuation_submitted", "release"]
            assert manager.submission_acknowledgements == 1
        finally:
            manager.stop()

    def test_submission_failure_log_does_not_render_locals(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)
        manager._pending_submissions = 1
        failed: Future[bool] = Future()
        failed.set_exception(RuntimeError("secret proof bytes"))

        with patch("miner_base.async_loop_manager._LOGGER") as logger:
            manager._submission_finished(failed)

        logger.error.assert_called_once_with("Block submission failed (RuntimeError).")
        logger.exception.assert_not_called()

    def test_abort_pending_submissions_unblocks_socket_and_refuses_new(
        self, tmp_path, miner_config, mock_mining_job
    ):
        socket_path = str(tmp_path / "gw.sock")
        server = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
        server.bind(socket_path)
        server.listen(1)
        accepted = threading.Event()
        peers: list[socket_module.socket] = []

        def accept_once():
            conn, _ = server.accept()
            peers.append(conn)
            accepted.set()

        acceptor = threading.Thread(target=accept_once, daemon=True)
        acceptor.start()

        cfg = MinerRpcConfig(transport="uds", socket_path=socket_path)
        manager = AsyncLoopManager(cfg, MinerSettings(no_gateway=False))
        done = threading.Event()
        errors: list[Exception] = []

        def blocked_submit():
            try:
                with patch(
                    "miner_base.async_loop_manager.submit_opened_block",
                    side_effect=lambda _o, _j, client: client.client.call("submitPlainProof", {}),
                ):
                    manager._submit_block(Mock(), mock_mining_job)
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()

        worker = threading.Thread(target=blocked_submit)
        worker.start()
        try:
            assert accepted.wait(timeout=_TIMEOUT_S)
            manager.abort_pending_submissions()
            assert done.wait(timeout=_TIMEOUT_S)
            assert len(errors) == 1
            # Submissions after abort are dropped immediately.
            fake_client = Mock(spec=MiningClient)
            with patch("miner_base.async_loop_manager._make_client", return_value=fake_client):
                assert manager._submit_block(Mock(), mock_mining_job) is False
            fake_client.close.assert_called_once_with()
        finally:
            for peer in peers:
                peer.close()
            server.close()
            worker.join(timeout=_TIMEOUT_S)
