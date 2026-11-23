"""Framework-neutral runtime for gateway polling, hash accounting, and bounded proof submission."""

from __future__ import annotations

import concurrent.futures
import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass

from miner_utils import get_logger
from pearl_gateway.comm.dataclasses import MiningJob
from pearl_gateway.config import MinerRpcConfig

from .block_submission import OpenedBlockInfo, is_plain_fp8_job, submit_opened_block
from .gateway_client import DummyMiningClient, MiningClient
from .settings import MinerSettings

_LOGGER = get_logger(__name__)
_DRAIN_PROGRESS_LOG_INTERVAL_SECONDS = 5.0
_MAX_STALE_JOB_POLLS = 3
_POLL_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class MiningLaunchDecision:
    """Whether work may launch and inspect a possible winner."""

    launch: bool
    retain_winner: bool


def _deadline_from_timeout(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    if not math.isfinite(timeout):
        raise ValueError(f"timeout must be finite, got {timeout!r}")
    return time.monotonic() + timeout


def _make_client(
    miner_settings: MinerSettings,
    config: MinerRpcConfig,
    *,
    stop_event: threading.Event | None = None,
) -> MiningClient:
    return (
        DummyMiningClient()
        if miner_settings.no_gateway
        else MiningClient(config, stop_event=stop_event)
    )


class AsyncLoopManager:
    def __init__(
        self,
        miner_rpc_config: MinerRpcConfig,
        miner_settings: MinerSettings | None,
        *,
        startup_fail_open: bool = False,
    ) -> None:
        self._conf = miner_settings if miner_settings is not None else MinerSettings()
        self._client_config = miner_rpc_config
        self._startup_fail_open = startup_fail_open

        self._lifecycle_cond = threading.Condition(threading.Lock())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._client: MiningClient | None = None
        self._stopping = False
        self._starting = False
        self._starting_thread_id: int | None = None
        self._startup_rollback = False
        self._stop_in_progress = False
        self._restart_permitted = True
        self._lifecycle_generation = 0

        self._mining_job: MiningJob | None = None
        self._mining_job_lock = threading.RLock()
        self._submission_condition = threading.Condition(self._mining_job_lock)
        self._mining_job_changed_callbacks: list[Callable[[], None]] = []
        self._consecutive_job_failures = 0

        self._inner_hash_counter = 0
        self._hash_counter_lock = threading.Lock()

        self._submission_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._pending_submissions = 0
        self._accepting_submissions = False
        self.submission_acknowledgements = 0

        self._inflight_clients: set[MiningClient] = set()
        self._inflight_lock = threading.Lock()
        self._aborting = False
        self._abort_event = threading.Event()

        self._drain_predicates: list[Callable[[], bool]] = []
        self._drain_barriers: list[Callable[[], None]] = []
        self._drain_releases: list[Callable[[], None]] = []
        self._drain_lock = threading.Lock()

    def start(self) -> None:
        with self._lifecycle_cond:
            self._lifecycle_cond.wait_for(
                lambda: not self._stop_in_progress and not self._startup_rollback
            )
            if not self._restart_permitted:
                raise RuntimeError("Cannot restart after an abandoned submission shutdown")
            if (
                self._starting
                or self._thread is not None
                or self._submission_executor is not None
                or self._client is not None
            ):
                raise RuntimeError("Already started?")
            self._starting = True
            self._starting_thread_id = threading.get_ident()
            self._stopping = False
            self._lifecycle_generation += 1
            stop_event = threading.Event()
            self._stop_event = stop_event

        try:
            self._start(stop_event)
        except BaseException:
            with self._lifecycle_cond:
                self._startup_rollback = True
                self._starting = False
                self._starting_thread_id = None
                self._lifecycle_cond.notify_all()
            rollback_ok = False
            try:
                self.stop(wait_for_submissions=True)
                rollback_ok = True
            finally:
                with self._lifecycle_cond:
                    if not rollback_ok:
                        self._restart_permitted = False
                    self._startup_rollback = False
                    self._lifecycle_cond.notify_all()
            raise
        else:
            with self._lifecycle_cond:
                self._starting = False
                self._starting_thread_id = None
                self._lifecycle_cond.notify_all()

    def _get_initial_job(self, stop_event: threading.Event) -> tuple[bool, MiningJob | None]:
        try:
            connected = _make_client(self._conf, self._client_config, stop_event=stop_event)
            with self._lifecycle_cond:
                if self._stopping:
                    publish = False
                else:
                    self._client = connected
                    publish = True
            if not publish:
                with suppress(Exception):
                    connected.close()
                return False, None
            return True, connected.get_mining_info()
        except Exception:
            self._discard_client()
            with self._lifecycle_cond:
                if self._stopping:
                    return False, None
            if not self._startup_fail_open:
                raise
            _LOGGER.opt(exception=True).warning(
                "No mining job at startup; serving continues without mining."
            )
            return True, None

    def _start(self, stop_event: threading.Event) -> None:
        self._consecutive_job_failures = 0
        with self._submission_condition:
            self._pending_submissions = 0
            self._accepting_submissions = False
            self.submission_acknowledgements = 0
        with self._inflight_lock:
            self._inflight_clients.clear()
            self._aborting = False
            self._abort_event = threading.Event()

        self._submission_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._conf.submission_inflight_limit,
            thread_name_prefix="block-submit",
        )
        continue_start, initial_job = self._get_initial_job(stop_event)
        if not continue_start:
            return

        self._publish_mining_job(initial_job)
        thread = threading.Thread(
            target=self._poll_loop, args=(stop_event,), name="gateway-poll", daemon=True
        )
        with self._lifecycle_cond, self._submission_condition:
            if self._stopping:
                return
            self._thread = thread
            thread.start()
            self._accepting_submissions = True

    def stop(self, *, wait_for_submissions: bool = True) -> None:
        """Stop polling and release executor, thread, and gateway connections."""
        with self._lifecycle_cond:
            self._lifecycle_cond.wait_for(lambda: not self._stop_in_progress)
            self._stop_in_progress = True
        try:
            self._stop(wait_for_submissions=wait_for_submissions)
        finally:
            with self._lifecycle_cond:
                self._stop_in_progress = False
                self._lifecycle_cond.notify_all()

    def _detach_polling_client(
        self,
        *,
        stopping: bool = False,
        close_admission: bool = False,
        abandon_restart: bool = False,
    ) -> MiningClient | None:
        with self._lifecycle_cond:
            if stopping:
                self._stopping = True
                self._stop_event.set()
            if close_admission:
                with self._submission_condition:
                    self._accepting_submissions = False
                    if abandon_restart:
                        self._restart_permitted = False
                    self._submission_condition.notify_all()
            client, self._client = self._client, None
            return client

    def _stop(self, *, wait_for_submissions: bool) -> None:
        client = self._detach_polling_client(
            stopping=True,
            close_admission=True,
            abandon_restart=not wait_for_submissions,
        )
        if client is not None:
            with suppress(Exception):
                client.close()

        with self._lifecycle_cond:
            if self._starting_thread_id != threading.get_ident():
                self._lifecycle_cond.wait_for(lambda: not self._starting)

        if not wait_for_submissions:
            self.abort_pending_submissions()

        executor, self._submission_executor = self._submission_executor, None
        if executor is not None:
            executor.shutdown(wait=wait_for_submissions, cancel_futures=True)

        thread, self._thread = self._thread, None
        if thread is not None and thread.ident is not None:
            self._stop_event.set()
            if thread is not threading.current_thread():
                thread.join()

        with self._lifecycle_cond:
            self._client = None

    @property
    def started(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def abort_pending_submissions(self) -> None:
        """Close sockets whose blocking RPC reads could stall shutdown."""
        polling_client = self._detach_polling_client(stopping=True, close_admission=True)
        with self._inflight_lock:
            self._aborting = True
            self._abort_event.set()
            clients = list(self._inflight_clients)
        if polling_client is not None:
            clients.append(polling_client)
        for client in clients:
            with suppress(Exception):
                client.close()

    def get_mining_job(self) -> MiningJob | None:
        with self._mining_job_lock:
            return self._mining_job

    def _publish_mining_job(self, mining_job: MiningJob | None) -> bool:
        with self._submission_condition:
            if mining_job == self._mining_job:
                return False
            self._mining_job = mining_job
            self._submission_condition.notify_all()
        for callback in self._mining_job_changed_callbacks:
            try:
                callback()
            except Exception:
                _LOGGER.opt(exception=True).warning(
                    "Mining-job-changed callback raised; continuing."
                )
        return True

    def register_mining_job_changed_callback(self, callback: Callable[[], None]) -> None:
        self._mining_job_changed_callbacks.append(callback)

    def increment_credited_hashes(self, count: int) -> None:
        if count < 0:
            raise ValueError(f"credited hash count must be non-negative, got {count}")
        with self._hash_counter_lock:
            self._inner_hash_counter += count

    @property
    def credited_hashes(self) -> int:
        with self._hash_counter_lock:
            return self._inner_hash_counter

    def _reserve_submission(
        self,
        mining_job: MiningJob,
        *,
        timeout: float | None,
    ) -> concurrent.futures.ThreadPoolExecutor | None:
        deadline = _deadline_from_timeout(timeout)
        wait_started: float | None = None
        outcome = "not-started"
        try:
            with self._submission_condition:
                while True:
                    if mining_job != self._mining_job:
                        outcome = "job-replaced"
                        return None
                    executor = self._submission_executor
                    if not self._accepting_submissions or executor is None:
                        outcome = "admission-closed"
                        return None
                    if self._pending_submissions < self._conf.submission_inflight_limit:
                        self._pending_submissions += 1
                        outcome = "capacity-recovered"
                        return executor

                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        outcome = "timeout"
                        return None
                    if wait_started is None:
                        wait_started = time.monotonic()
                        budget = (
                            "without a deadline" if timeout is None else f"for up to {timeout:.3f}s"
                        )
                        _LOGGER.warning(
                            f"Proof submission capacity saturated; handoff waiting {budget} "
                            f"(pending={self._pending_submissions}/{self._conf.submission_inflight_limit})."
                        )
                    self._submission_condition.wait(timeout=remaining)
        finally:
            if wait_started is not None:
                elapsed = time.monotonic() - wait_started
                if outcome == "timeout":
                    _LOGGER.error(
                        f"Proof submission capacity wait timed out (waited={elapsed:.3f}s)."
                    )
                else:
                    _LOGGER.info(
                        f"Proof submission capacity wait ended: {outcome} (waited={elapsed:.3f}s)."
                    )

    def _release_pending_submission(self, *, acknowledged: bool = False) -> None:
        with self._submission_condition:
            if self._pending_submissions <= 0:
                raise RuntimeError("proof-submission ownership underflow")
            self._pending_submissions -= 1
            if acknowledged:
                self.submission_acknowledgements += 1
            self._submission_condition.notify_all()

    def handle_submit_block(
        self,
        opened_block_info: OpenedBlockInfo,
        mining_job: MiningJob,
        *,
        timeout: float | None = None,
    ) -> bool:
        """Reserve bounded capacity, snapshot inputs, and queue proof construction + RPC."""
        _deadline_from_timeout(timeout)
        if (
            self._conf.no_gateway
            or self._conf.skip_block_submission
            or not is_plain_fp8_job(mining_job)
        ):
            return False

        executor = self._reserve_submission(mining_job, timeout=timeout)
        if executor is None:
            return False
        try:
            owned_opening = opened_block_info.owned_copy()
            owned_job = MiningJob(
                incomplete_header_bytes=bytes(mining_job.incomplete_header_bytes),
                target=mining_job.target,
                cert_version=mining_job.cert_version,
                ancestor_headers=[bytes(h) for h in mining_job.ancestor_headers],
            )
        except BaseException:
            self._release_pending_submission()
            raise

        try:
            future = executor.submit(self._submit_block, owned_opening, owned_job)
        except RuntimeError:
            self._release_pending_submission()
            _LOGGER.warning("Dropping found block: submission runtime is shutting down.")
            return False
        future.add_done_callback(self._submission_finished)
        return True

    def _submission_finished(self, future: concurrent.futures.Future[bool]) -> None:
        acknowledged = False
        try:
            acknowledged = future.result()
        except concurrent.futures.CancelledError:
            _LOGGER.warning("Queued block submission was cancelled during shutdown.")
        except Exception as exc:
            # Omit traceback so Loguru diagnose=True cannot dump proof/job locals.
            _LOGGER.error(f"Block submission failed ({type(exc).__name__}).")
        finally:
            self._release_pending_submission(acknowledged=acknowledged)

    def _submit_block(self, opened_block_info: OpenedBlockInfo, mining_job: MiningJob) -> bool:
        if self._conf.no_gateway or self._conf.skip_block_submission:
            return False
        with self._inflight_lock:
            if self._aborting:
                _LOGGER.warning("Dropping found block: async loop manager is shutting down.")
                return False
            abort_event = self._abort_event
        client = _make_client(self._conf, self._client_config, stop_event=abort_event)
        with self._inflight_lock:
            if self._aborting:
                client.close()
                _LOGGER.warning("Dropping found block: async loop manager is shutting down.")
                return False
            self._inflight_clients.add(client)
        try:
            with client:
                proof = submit_opened_block(opened_block_info, mining_job, client)
        finally:
            with self._inflight_lock:
                self._inflight_clients.discard(client)
        if proof is None:
            _LOGGER.info("Lottery winner was not jackpot-admissible; filtered before RPC.")
            return False
        _LOGGER.info("FP8 proof handed to the gateway submission queue.")
        return True

    @contextmanager
    def mining_launch_admission(self, job: MiningJob) -> Iterator[MiningLaunchDecision]:
        """Keep job and submission admission stable through launch registration."""
        with self._submission_condition:
            if job != self._mining_job:
                decision = MiningLaunchDecision(False, False)
            elif (
                self._conf.no_gateway
                or self._conf.skip_block_submission
                or not is_plain_fp8_job(job)
            ):
                decision = MiningLaunchDecision(True, False)
            else:
                retain = self._accepting_submissions and self._submission_executor is not None
                decision = MiningLaunchDecision(retain, retain)
            yield decision

    def register_drain_predicate(self, predicate: Callable[[], bool]) -> None:
        self._drain_predicates.append(predicate)

    def register_drain_barrier(self, barrier: Callable[[], None]) -> None:
        self._drain_barriers.append(barrier)

    def register_drain_release(self, release: Callable[[], None]) -> None:
        self._drain_releases.append(release)

    def _is_current_lifecycle(self, generation: int, stop_event: threading.Event) -> bool:
        return (
            not self._stopping
            and generation == self._lifecycle_generation
            and stop_event is self._stop_event
            and not stop_event.is_set()
        )

    def _run_drain_barriers(self) -> bool:
        for barrier in self._drain_barriers:
            try:
                barrier()
            except Exception:
                _LOGGER.opt(exception=True).error("Drain barrier failed; flush aborted.")
                return False
        return True

    def _release_drain_admission(self, generation: int) -> None:
        with self._lifecycle_cond, self._submission_condition:
            if self._stopping or generation != self._lifecycle_generation:
                return
            if self._submission_executor is not None:
                self._accepting_submissions = True
                self._submission_condition.notify_all()
            for release in self._drain_releases:
                try:
                    release()
                except Exception:
                    _LOGGER.opt(exception=True).error("Drain release failed; mining stays closed.")

    def wait_until_drained(self, timeout: float | None = None) -> bool:
        """Flush admitted GPU continuations, then close and drain pending proof submissions."""
        deadline = _deadline_from_timeout(timeout)
        if deadline is not None and time.monotonic() >= deadline:
            return False
        with self._lifecycle_cond:
            if self._stopping:
                return False
            generation = self._lifecycle_generation

        with self._drain_lock:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            with self._lifecycle_cond:
                if self._stopping or generation != self._lifecycle_generation:
                    return False
                stop_event = self._stop_event
            try:
                if not self._run_drain_barriers():
                    return False
                if not self._wait_predicates(deadline, timeout, generation, stop_event):
                    return False
                with self._lifecycle_cond, self._submission_condition:
                    if not self._is_current_lifecycle(generation, stop_event):
                        return False
                    self._accepting_submissions = False
                    self._submission_condition.notify_all()
                return self._wait_submissions_empty(deadline, timeout, generation, stop_event)
            finally:
                self._release_drain_admission(generation)

    def _wait_predicates(
        self,
        deadline: float | None,
        timeout: float | None,
        generation: int,
        stop_event: threading.Event,
    ) -> bool:
        started = time.monotonic()
        next_report = started + _DRAIN_PROGRESS_LOG_INTERVAL_SECONDS
        while not stop_event.is_set():
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                _LOGGER.warning(f"Flush timed out after {timeout}s waiting for drain predicates.")
                return False
            if all(p() for p in self._drain_predicates):
                with self._lifecycle_cond:
                    return self._is_current_lifecycle(generation, stop_event) and (
                        deadline is None or time.monotonic() < deadline
                    )
            if now >= next_report:
                next_report = now + _DRAIN_PROGRESS_LOG_INTERVAL_SECONDS
                _LOGGER.warning(
                    f"Flush has been waiting {now - started:.0f}s for drain predicates."
                )
            sleep_s = 0.05 if deadline is None else min(0.05, max(0.0, deadline - time.monotonic()))
            if sleep_s:
                stop_event.wait(sleep_s)
        return False

    def _wait_submissions_empty(
        self,
        deadline: float | None,
        timeout: float | None,
        generation: int,
        stop_event: threading.Event,
    ) -> bool:
        with self._submission_condition:
            while not stop_event.is_set() and self._pending_submissions > 0:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    _LOGGER.warning(
                        f"Flush timed out after {timeout}s with "
                        f"{self._pending_submissions} pending submissions."
                    )
                    return False
                wait_s = (
                    _DRAIN_PROGRESS_LOG_INTERVAL_SECONDS
                    if remaining is None
                    else min(_DRAIN_PROGRESS_LOG_INTERVAL_SECONDS, remaining)
                )
                self._submission_condition.wait(timeout=wait_s)
        with self._lifecycle_cond:
            return self._is_current_lifecycle(generation, stop_event) and (
                deadline is None or time.monotonic() < deadline
            )

    def _discard_client(self, expected: MiningClient | None = None) -> None:
        with self._lifecycle_cond:
            if expected is not None and self._client is not expected:
                return
            client, self._client = self._client, None
        if client is not None:
            with suppress(Exception):
                client.close()

    def _polling_client(self, stop_event: threading.Event) -> MiningClient | None:
        with self._lifecycle_cond:
            if self._stopping:
                return None
            if self._client is not None:
                return self._client

        connected = _make_client(self._conf, self._client_config, stop_event=stop_event)
        with self._lifecycle_cond:
            if self._stopping:
                published = None
            elif self._client is None:
                self._client = connected
                return connected
            else:
                published = self._client
        with suppress(Exception):
            connected.close()
        return published

    def _poll_loop(self, stop_event: threading.Event) -> None:
        try:
            while not stop_event.is_set():
                client: MiningClient | None = None
                try:
                    client = self._polling_client(stop_event)
                    if client is None:
                        return
                    new_job = client.get_mining_info()
                    self._consecutive_job_failures = 0
                except Exception:
                    if stop_event.is_set():
                        if client is not None:
                            self._discard_client(client)
                        return
                    self._consecutive_job_failures += 1
                    _LOGGER.exception("Failed to get mining info")
                    if client is not None:
                        self._discard_client(client)
                    if (
                        self.get_mining_job() is not None
                        and self._consecutive_job_failures >= _MAX_STALE_JOB_POLLS
                    ):
                        _LOGGER.warning("Retiring a stale mining job after repeated poll failures.")
                        self._publish_mining_job(None)
                else:
                    self._publish_mining_job(new_job)
                stop_event.wait(_POLL_INTERVAL_SECONDS)
        except Exception:
            _LOGGER.exception("Async loop manager thread crashed")
            client = self._detach_polling_client(stopping=True, close_admission=True)
            if client is not None:
                with suppress(Exception):
                    client.close()
            with suppress(Exception):
                self._publish_mining_job(None)
