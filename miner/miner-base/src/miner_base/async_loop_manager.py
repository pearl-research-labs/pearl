"""Framework-neutral async runtime for plain-peel mining.

Owns gateway job polling, credited-hash accounting, and bounded proof
submission for one miner process. GPU owners may register framework-neutral
drain predicates and barriers, but CUDA events, streams, and callback
execution remain outside this package.
"""

import asyncio
import concurrent.futures
import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass

from miner_utils import get_logger
from pearl_gateway.comm.dataclasses import MiningJob

from .block_submission import (
    OpenedBlockInfo,
    is_plain_fp8_job,
    submit_opened_block,
)
from .gateway_client import DummyMiningClient, MinerRpcConfig, MiningClient
from .settings import MinerSettings

_LOGGER = get_logger(__name__)
_DRAIN_PROGRESS_LOG_INTERVAL_SECONDS = 5.0
_MAX_STALE_JOB_POLLS = 3


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


def _make_client(miner_settings: MinerSettings, config: MinerRpcConfig) -> MiningClient:
    if miner_settings.no_gateway:
        return DummyMiningClient()
    return MiningClient(config)


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

        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event = asyncio.Event()
        self._thread: threading.Thread | None = None
        # The polling connection is read by the loop and closed by lifecycle
        # owners. Publication under this lock makes every reconnect visible to
        # stop()/abort before it can begin an unbounded response read.
        self._client: MiningClient | None = None
        self._client_lock = threading.Lock()
        self._client_condition = threading.Condition(self._client_lock)
        self._stopping = False
        self._starting = False
        self._starting_thread_id: int | None = None
        self._startup_rollback = False
        self._stop_in_progress = False
        self._lifecycle_generation = 0

        self._mining_job: MiningJob | None = None
        self._mining_job_lock = threading.RLock()
        self._mining_job_changed_callbacks: list[Callable[[], None]] = []
        self._consecutive_job_failures = 0

        self._inner_hash_counter = 0
        self._hash_counter_lock = threading.Lock()

        self._submission_executor: concurrent.futures.ThreadPoolExecutor | None = None
        # Job publication and submission capacity share one condition lock so
        # a capacity wait releases job publication, then revalidates freshness
        # atomically when it wakes to reserve the slot.
        self._submission_condition = threading.Condition(self._mining_job_lock)
        # Counts queued/running proof RPCs. Reusable flush waits for GPU/host
        # continuations to hand off winners before closing this admission.
        self._pending_submissions = 0
        self._accepting_submissions = False
        self._restart_permitted = True
        self.submission_acknowledgements = 0

        # Submission RPC reads have no socket timeout. Track their one-shot
        # clients so bounded shutdown can close the sockets and wake the reads.
        self._inflight_clients: set[MiningClient] = set()
        self._inflight_lock = threading.Lock()
        self._aborting = False

        # GPU packages contribute only generic lifecycle hooks. Their CUDA
        # objects never cross this package boundary.
        self._drain_predicates: list[Callable[[], bool]] = []
        self._drain_barriers: list[Callable[[], None]] = []
        self._drain_releases: list[Callable[[], None]] = []
        # One flush owns admission at a time; concurrent callers wait here and
        # each receive a complete close/wait/reopen cycle.
        self._drain_lock = threading.Lock()

        # start() does not open producer/submission admission until the loop is
        # running.
        self._async_startup_ready = threading.Event()
        self._async_startup_error: BaseException | None = None

    def start(self) -> None:
        with self._client_condition:
            self._client_condition.wait_for(
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

        try:
            self._start()
        except BaseException:
            # Any startup stage may have published clients, executors, or a
            # Thread object. First let a concurrent stop finish waiting for the
            # startup attempt, but retain a distinct rollback flag so no retry
            # can begin between that stop and this failure owner's cleanup pass.
            with self._client_condition:
                self._startup_rollback = True
                self._starting = False
                self._starting_thread_id = None
                self._client_condition.notify_all()
            rollback_succeeded = False
            try:
                # Serialize rollback through the ordinary stop owner. A
                # concurrent stop completes first; this then performs an
                # idempotent second pass before admitting a retry.
                self.stop(wait_for_submissions=True)
                rollback_succeeded = True
            finally:
                with self._client_condition:
                    if not rollback_succeeded:
                        self._restart_permitted = False
                    self._startup_rollback = False
                    self._client_condition.notify_all()
            raise
        else:
            with self._client_condition:
                self._starting = False
                self._starting_thread_id = None
                self._client_condition.notify_all()

    def _get_initial_job(self) -> tuple[bool, MiningJob | None]:
        try:
            connected = _make_client(self._conf, self._client_config)
            with self._client_condition:
                publish_client = not self._stopping
                if publish_client:
                    self._client = connected
            if not publish_client:
                with suppress(Exception):
                    connected.close()
                return False, None
            return True, connected.get_mining_info()
        except Exception:
            self._discard_client()
            with self._client_lock:
                stopped_during_start = self._stopping
            if stopped_during_start:
                return False, None
            if not self._startup_fail_open:
                self._submission_executor.shutdown(wait=True, cancel_futures=True)
                self._submission_executor = None
                raise
            _LOGGER.opt(exception=True).warning(
                "No mining job at startup; serving continues without mining."
            )
            return True, None

    def _start(self) -> None:
        self._async_startup_ready.clear()
        self._async_startup_error = None
        self._stop_event = asyncio.Event()
        self._consecutive_job_failures = 0
        with self._submission_condition:
            self._pending_submissions = 0
            self._accepting_submissions = False
            self.submission_acknowledgements = 0
        with self._inflight_lock:
            self._inflight_clients.clear()
            self._aborting = False

        self._submission_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._conf.submission_inflight_limit,
            thread_name_prefix="block-submit",
        )
        continue_start, initial_job = self._get_initial_job()
        if not continue_start:
            return

        self._publish_mining_job(initial_job)
        thread = threading.Thread(target=self._run_async_loop, daemon=True)
        with self._client_condition:
            if self._stopping:
                launch_thread = False
            else:
                self._thread = thread
                thread.start()
                launch_thread = True
        if not launch_thread:
            return
        self._async_startup_ready.wait()
        if self._async_startup_error is not None:
            raise self._async_startup_error
        # No winner can enter an executor until the polling thread has a
        # successful OS-thread owner. If startup raises, rollback never admits
        # new proof/mining ownership.
        self._open_submission_admission_if_running()

    def _open_submission_admission_if_running(self) -> bool:
        # Terminality and admission share one linearization point. The lock
        # order is always client -> submission when both are required.
        with self._client_condition, self._submission_condition:
            if self._stopping:
                return False
            self._accepting_submissions = True
            return True

    def _wait_for_startup_to_finish(self) -> None:
        with self._client_condition:
            if self._starting_thread_id != threading.get_ident():
                self._client_condition.wait_for(lambda: not self._starting)

    def stop(self, *, wait_for_submissions: bool = True) -> None:
        """Stop the runtime and release its executor, loop, socket, and client.

        Owners call the default only after a successful drain. A bounded owner
        that already timed out may pass ``wait_for_submissions=False``; this
        aborts gateway clients before abandoning executor waiting.
        """
        with self._client_condition:
            self._client_condition.wait_for(lambda: not self._stop_in_progress)
            self._stop_in_progress = True
        try:
            self._stop(wait_for_submissions=wait_for_submissions)
        finally:
            with self._client_condition:
                self._stop_in_progress = False
                self._client_condition.notify_all()

    def _request_loop_stop(self, thread: threading.Thread) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(self._stop_event.set)
                return
            except RuntimeError:
                # The loop closed after is_running(); direct publication still
                # lets _stop() proceed to the bounded thread join.
                pass
        if thread.is_alive():
            _LOGGER.warning("Thread is alive but loop is dead?")
            self._stop_event.set()

    def _stop(self, *, wait_for_submissions: bool) -> None:
        # Close admission before detaching the polling client. Startup cannot
        # reopen admission behind this point.
        polling_client = self._take_polling_client(
            stopping=True,
            close_admission=True,
            abandon_restart=not wait_for_submissions,
        )
        if polling_client is not None:
            with suppress(Exception):
                polling_client.close()

        # A first connection may still be under construction and therefore not
        # published yet. Its configured connection deadline bounds this wait;
        # start() observes shutdown and closes any late result itself.
        self._wait_for_startup_to_finish()

        if not wait_for_submissions:
            self.abort_pending_submissions()

        executor, self._submission_executor = self._submission_executor, None
        if executor is not None:
            executor.shutdown(wait=wait_for_submissions, cancel_futures=True)

        # The polling transport was closed above to wake loop-owned
        # ``asyncio.to_thread`` work. The loop joins its default executor before
        # advertising termination.
        if self._thread is not None:
            thread = self._thread
            # Thread.start() can fail after the object is published. An object
            # with no ident was never started and Python forbids joining it.
            if getattr(thread, "ident", None) is not None:
                self._request_loop_stop(thread)
                if thread is not threading.current_thread():
                    thread.join()
                else:
                    _LOGGER.debug("Called `stop()` from managed thread?!")
            self._thread = None

        with self._client_lock:
            self._client = None
        self._loop = None

    @property
    def started(self) -> bool:
        return self._thread is not None

    @property
    def blocks_submitted(self) -> int:
        """Backward-compatible name for gateway queue acknowledgements."""
        return self.submission_acknowledgements

    def abort_pending_submissions(self) -> None:
        """Close clients whose unbounded RPC reads are blocking shutdown."""
        polling_client = self._take_polling_client(
            stopping=True,
            close_admission=True,
        )
        with self._inflight_lock:
            self._aborting = True
            clients = list(self._inflight_clients)
        if polling_client is not None:
            clients.append(polling_client)
        for client in clients:
            with suppress(Exception):
                client.close()

    def get_mining_job(self) -> MiningJob | None:
        with self._mining_job_lock:
            return self._mining_job

    def _replace_mining_job(self, mining_job: MiningJob | None) -> bool:
        with self._submission_condition:
            if mining_job == self._mining_job:
                return False
            self._mining_job = mining_job
            # Wake capacity waiters so stale winners are rejected immediately
            # instead of retaining their proof buffers until an RPC completes.
            self._submission_condition.notify_all()
            return True

    def _publish_mining_job(self, mining_job: MiningJob | None) -> bool:
        if not self._replace_mining_job(mining_job):
            return False
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
        """Add a protocol-computed count of completed, creditable hashes.

        The count is a local statistic (see ``credited_hashes``); the upstream
        gateway RPC has no hash-rate reporting method.
        """
        if count < 0:
            raise ValueError(f"credited hash count must be non-negative, got {count}")
        with self._hash_counter_lock:
            self._inner_hash_counter += count

    @property
    def credited_hashes(self) -> int:
        with self._hash_counter_lock:
            return self._inner_hash_counter

    def _pending_submission_count(self) -> int:
        with self._submission_condition:
            return self._pending_submissions

    def _finish_submission_capacity_wait(
        self,
        wait_started: float | None,
        outcome: str,
    ) -> None:
        if wait_started is None:
            return
        elapsed = time.monotonic() - wait_started
        detail = f"waited={elapsed:.3f}s"
        if outcome == "timeout":
            _LOGGER.error(
                f"Proof submission capacity wait timed out; candidate was not queued ({detail})."
            )
        elif outcome == "capacity-recovered":
            _LOGGER.info(f"Proof submission capacity recovered ({detail}).")
        elif outcome == "job-replaced":
            _LOGGER.info(f"Proof submission capacity wait ended: mining job replaced ({detail}).")
        elif outcome == "admission-closed":
            _LOGGER.info(
                "Proof submission capacity wait ended: submission admission closed "
                f"by drain or shutdown ({detail})."
            )

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
                            "Proof submission capacity saturated; handoff is waiting "
                            f"{budget} (pending={self._pending_submissions}/"
                            f"{self._conf.submission_inflight_limit}, "
                            "mining completion accounting may lag)."
                        )
                    # Condition.wait releases the shared mining-job lock completely.
                    # Publication, completion, or a flush can proceed and wake the
                    # waiter to revalidate freshness and admission before reserving.
                    self._submission_condition.wait(timeout=remaining)
        finally:
            self._finish_submission_capacity_wait(wait_started, outcome)

    def _release_pending_submission_ownership(self) -> None:
        with self._submission_condition:
            if self._pending_submissions <= 0:
                raise RuntimeError("proof-submission ownership underflow")
            self._pending_submissions -= 1
            self._submission_condition.notify_all()

    def _enqueue_submission(
        self,
        opened_block_info: OpenedBlockInfo,
        mining_job: MiningJob,
        *,
        timeout: float | None,
        on_complete: Callable[[], None] | None = None,
    ) -> bool:
        # Validate API boundaries even when policy will refuse the handoff.
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
            )
        except BaseException:
            self._release_pending_submission_ownership()
            raise

        try:
            future = executor.submit(
                self._submit_block,
                owned_opening,
                owned_job,
            )
        except RuntimeError:
            self._release_pending_submission_ownership()
            _LOGGER.warning("Dropping found block: submission runtime is shutting down.")
            return False
        future.add_done_callback(
            lambda completed: self._submission_finished(completed, on_complete)
        )
        return True

    def handle_submit_block(
        self,
        opened_block_info: OpenedBlockInfo,
        mining_job: MiningJob,
        *,
        timeout: float | None = None,
    ) -> bool:
        """Reserve bounded capacity, applying backpressure before ownership transfer."""
        return self._enqueue_submission(
            opened_block_info,
            mining_job,
            timeout=timeout,
        )

    def _submission_finished(
        self,
        future: concurrent.futures.Future[bool],
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        acknowledged = False
        try:
            acknowledged = future.result()
        except concurrent.futures.CancelledError:
            _LOGGER.warning("Queued block submission was cancelled during shutdown.")
        except Exception as exc:
            # Proof/job values live in the submitter frame. Do not attach a
            # Loguru traceback here: diagnose=True would render those locals.
            _LOGGER.error(f"Block submission failed ({type(exc).__name__}).")
        finally:
            if on_complete is not None:
                try:
                    on_complete()
                except Exception:
                    _LOGGER.error("Block-submission completion callback failed.")
            with self._submission_condition:
                if self._pending_submissions <= 0:
                    raise RuntimeError("proof-submission completion underflow")
                self._pending_submissions -= 1
                if acknowledged:
                    self.submission_acknowledgements += 1
                self._submission_condition.notify_all()

    def _submit_block(
        self,
        opened_block_info: OpenedBlockInfo,
        mining_job: MiningJob,
    ) -> bool:
        if self._conf.no_gateway or self._conf.skip_block_submission:
            return False
        client = _make_client(self._conf, self._client_config)
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

    def _classify_mining_launch_locked(self, job: MiningJob) -> MiningLaunchDecision:
        if job != self._mining_job:
            return MiningLaunchDecision(False, False)
        if self._conf.no_gateway or self._conf.skip_block_submission or not is_plain_fp8_job(job):
            # The useful runtime may still execute and credit work, but it must
            # not inspect, construct, verify, or submit a winner.
            return MiningLaunchDecision(True, False)
        # Deliberately do not gate launches on current proof/RPC occupancy.
        # submission_inflight_limit bounds actual handoffs; a retained winner
        # applies bounded backpressure while ordinary mining remains creditable.
        retain_winner = self._accepting_submissions and self._submission_executor is not None
        return MiningLaunchDecision(retain_winner, retain_winner)

    @contextmanager
    def mining_launch_admission(self, job: MiningJob) -> Iterator[MiningLaunchDecision]:
        """Keep job and submission admission stable through launch registration."""
        with self._submission_condition:
            yield self._classify_mining_launch_locked(job)

    def register_drain_predicate(self, predicate: Callable[[], bool]) -> None:
        self._drain_predicates.append(predicate)

    def register_drain_barrier(self, barrier: Callable[[], None]) -> None:
        self._drain_barriers.append(barrier)

    def register_drain_release(self, release: Callable[[], None]) -> None:
        self._drain_releases.append(release)

    def _continuations_drained(self) -> bool:
        return all(predicate() for predicate in self._drain_predicates)

    def _submissions_drained(self) -> bool:
        return self._pending_submission_count() == 0

    def _pending_work_description(self) -> str:
        unmet = [index for index, predicate in enumerate(self._drain_predicates) if not predicate()]
        return (
            f"pending_submissions={self._pending_submission_count()} "
            f"unmet_drain_predicates={unmet or 'none'} of={len(self._drain_predicates)} "
            f"hashes_counted={self.credited_hashes}"
        )

    def _run_drain_barriers(self) -> bool:
        for barrier in self._drain_barriers:
            try:
                barrier()
            except Exception:
                _LOGGER.opt(exception=True).error("Drain barrier failed; flush aborted.")
                return False
        return True

    def wait_until_drained(self, timeout: float | None = None) -> bool:
        """Flush admitted GPU work, then its proof submissions."""
        return self._wait_until_drained_with_deadline(
            deadline=_deadline_from_timeout(timeout),
            timeout=timeout,
        )

    def _wait_until_drained_with_deadline(
        self,
        *,
        deadline: float | None,
        timeout: float | None,
    ) -> bool:
        """Run one ordered reusable flush against one absolute deadline."""
        if deadline is not None and time.monotonic() >= deadline:
            return False
        with self._client_condition:
            if self._stopping:
                return False
            generation = self._lifecycle_generation
        with self._drain_lock:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            with self._client_condition:
                if self._stopping or generation != self._lifecycle_generation:
                    return False
                stop_event = self._stop_event
            try:
                # Phase 1 closes GPU producer admission only. Proof submission
                # stays open until every pre-close host continuation has had a
                # chance to enqueue all winners it observed.
                if not self._run_drain_barriers():
                    return False
                if not self._wait_for_drain_phase(
                    self._continuations_drained,
                    deadline,
                    timeout,
                    generation,
                    stop_event,
                ):
                    return False

                # Phase 2 closes proof admission only after continuation
                # quiescence, then waits for the resulting submission RPCs.
                with self._client_condition, self._submission_condition:
                    if not self._lifecycle_is_current(generation, stop_event):
                        return False
                    self._accepting_submissions = False
                    self._submission_condition.notify_all()
                return self._wait_for_drain_phase(
                    self._submissions_drained,
                    deadline,
                    timeout,
                    generation,
                    stop_event,
                )
            finally:
                self._release_drain_admission(generation)

    def _lifecycle_is_current(
        self,
        generation: int,
        stop_event: asyncio.Event,
    ) -> bool:
        return (
            not self._stopping
            and generation == self._lifecycle_generation
            and stop_event is self._stop_event
            and not stop_event.is_set()
        )

    def _wait_for_drain_phase(
        self,
        complete: Callable[[], bool],
        deadline: float | None,
        timeout: float | None,
        generation: int,
        stop_event: asyncio.Event,
    ) -> bool:
        started = time.monotonic()
        next_report = started + _DRAIN_PROGRESS_LOG_INTERVAL_SECONDS
        while not stop_event.is_set():
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                _LOGGER.warning(
                    f"Flush did not complete within {timeout}s: {self._pending_work_description()}"
                )
                return False
            if complete():
                with self._client_condition:
                    current = self._lifecycle_is_current(generation, stop_event)
                return current and (deadline is None or time.monotonic() < deadline)
            if now >= next_report:
                next_report = now + _DRAIN_PROGRESS_LOG_INTERVAL_SECONDS
                _LOGGER.warning(
                    f"Flush has been waiting {now - started:.0f}s: "
                    f"{self._pending_work_description()}"
                )
            sleep_s = 0.05
            if deadline is not None:
                sleep_s = min(sleep_s, max(0.0, deadline - time.monotonic()))
            if sleep_s:
                time.sleep(sleep_s)
        return False

    def _release_drain_admission(self, generation: int) -> None:
        with self._client_condition, self._submission_condition:
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

    def _run_async_loop(self) -> None:
        loop: asyncio.AbstractEventLoop | None = None
        try:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_main())
        except BaseException as exc:
            self._async_startup_error = exc
            if self._async_startup_ready.is_set():
                _LOGGER.opt(exception=True).error("Async loop manager thread crashed")
        finally:
            self._async_startup_ready.set()
            if loop is not None:
                with suppress(Exception):
                    loop.run_until_complete(loop.shutdown_default_executor())
                loop.close()
            self._loop = None

    async def _async_main(self) -> None:
        update_task: asyncio.Task[None] | None = None
        try:
            update_task = asyncio.create_task(self._update_gateway_loop(), name="update_gateway")
            update_task.add_done_callback(self._on_worker_done)
            self._async_startup_ready.set()
            await self._stop_event.wait()
        finally:
            if update_task is not None:
                update_task.cancel()
                await asyncio.gather(update_task, return_exceptions=True)

    @staticmethod
    def _on_worker_done(task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (exc := task.exception()) is not None:
            _LOGGER.error(f"Async task {task.get_name()} crashed: {exc!r}")

    def _take_polling_client(
        self,
        *,
        stopping: bool = False,
        close_admission: bool = False,
        abandon_restart: bool = False,
    ) -> MiningClient | None:
        """Atomically detach the polling connection from lifecycle ownership."""
        with self._client_condition:
            if stopping:
                self._stopping = True
            if close_admission:
                with self._submission_condition:
                    self._accepting_submissions = False
                    if abandon_restart:
                        self._restart_permitted = False
                    self._submission_condition.notify_all()
            client, self._client = self._client, None
            return client

    def _discard_client(self, expected: MiningClient | None = None) -> None:
        """Close the published polling client, unless it has already changed."""
        with self._client_lock:
            if expected is not None and self._client is not expected:
                return
            client, self._client = self._client, None
        if client is not None:
            with suppress(Exception):
                client.close()

    def _discard_finished_connector(self, task: asyncio.Task[MiningClient]) -> None:
        """Consume and close a connector result detached by cancellation."""
        if task.cancelled():
            return
        try:
            connected = task.result()
        except Exception:
            return
        with suppress(Exception):
            connected.close()

    async def _polling_client(self) -> MiningClient | None:
        """Return or connect the one stop-owned polling client off the loop."""
        with self._client_lock:
            if self._stopping:
                return None
            current = self._client
        if current is not None:
            return current

        connect_task = asyncio.create_task(
            asyncio.to_thread(_make_client, self._conf, self._client_config)
        )
        try:
            connected = await asyncio.shield(connect_task)
        except asyncio.CancelledError:
            # asyncio cannot cancel a connector already running in a worker
            # thread. Detach it and close any late result without delaying loop
            # shutdown beyond the configured connection deadline.
            connect_task.add_done_callback(self._discard_finished_connector)
            raise
        with self._client_lock:
            if self._stopping:
                published = None
            elif self._client is None:
                self._client = connected
                return connected
            else:
                published = self._client
        # stop() began, or an existing connection won publication. In both
        # cases this newly connected socket must not escape the ownership set.
        with suppress(Exception):
            connected.close()
        return published

    async def _update_gateway_loop(self) -> None:
        while not self._stop_event.is_set():
            client: MiningClient | None = None
            try:
                client = await self._polling_client()
                if client is None:
                    return
                new_mining_job = await asyncio.to_thread(client.get_mining_info)
                self._consecutive_job_failures = 0
            except Exception:
                self._consecutive_job_failures += 1
                _LOGGER.exception("Failed to get mining info")
                if client is not None:
                    self._discard_client(client)
                if (
                    self.get_mining_job() is not None
                    and self._consecutive_job_failures >= _MAX_STALE_JOB_POLLS
                ):
                    _LOGGER.warning("Retiring a stale mining job after repeated poll failures.")
                    await asyncio.to_thread(self._publish_mining_job, None)
            else:
                # Admission holders can briefly own the publication lock across
                # CUDA enqueue/registration. Keep that wait off the event loop.
                await asyncio.to_thread(self._publish_mining_job, new_mining_job)
            await asyncio.sleep(1.0)
