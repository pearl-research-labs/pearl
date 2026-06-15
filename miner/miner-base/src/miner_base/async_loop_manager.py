"""
Manages an async thread for this miner process, which handles several tasks:

1. Getting new block headers from the gateway
2. Reporting new blocks and inner hash counters to the gateway
3. Calculating proofs for new blocks
4. Supporting async status checking from CUDA kernels
"""

import asyncio
import threading
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Thread

import torch
from miner_utils import get_logger
from pearl_gateway.comm.dataclasses import MiningJob, OpenedBlockInfo
from pearl_mining import (
    IncompleteBlockHeader,
    check_cert_version_eligible,
    verify_plain_proof_for_cert_version,
)

from .block_submission import create_proof
from .gateway_client import DummyMiningClient, MinerRpcConfig, MiningClient
from .settings import MinerSettings

_LOGGER = get_logger(__name__)


@dataclass
class CudaEventQueueItem:
    """Represents a pending status check with a CUDA event and callback."""

    cuda_event: torch.cuda.Event
    callback: Callable[[], None]


def _make_client(miner_settings: MinerSettings, config: MinerRpcConfig) -> MiningClient:
    if miner_settings.no_gateway:
        return DummyMiningClient()
    return MiningClient(config)


class AsyncLoopManager:
    def __init__(
        self, miner_rpc_config: MinerRpcConfig, miner_settings: MinerSettings | None
    ) -> None:
        self._conf = miner_settings if miner_settings is not None else MinerSettings()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._mining_job: MiningJob | None = None
        self._stop_event = asyncio.Event()
        self._client_config = miner_rpc_config

        # pool and queue are for calculating proof infos
        self._block_results: list[asyncio.Future[bool]] = []

        # Queue and thread for async status checking
        self._cuda_event_queue: asyncio.Queue[CudaEventQueueItem] = asyncio.Queue()

        # Attrs initialized by AsyncLoopManager.start()
        self._thread: Thread | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._client: MiningClient | None = None

        self._mining_job_changed_callbacks: list[Callable[[], None]] = []

        # Counter for submitted blocks
        self.blocks_submitted = 0

        # Tracks CUDA events that have been scheduled but not yet fully processed
        # (i.e. the callback hasn't fired yet). Used by wait_until_done_submitting_blocks
        # to avoid returning before in-flight events have a chance to call handle_submit_block.
        self._pending_cuda_events = 0

    def start(self) -> None:
        """Start the async loop manager."""
        if self._thread is not None or self._pool is not None or self._client is not None:
            raise RuntimeError("Already started?")

        self._pool = ThreadPoolExecutor()
        self._client = _make_client(self._conf, self._client_config)
        # initialize the mining job synchronously for the first time
        self._mining_job = self._client.get_mining_info()
        self._thread = Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the async loop manager, wait for the thread to finish."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

        if self._thread is not None:
            if self._loop is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._stop_event.set)
            else:
                _LOGGER.warning("Thread is alive but loop is dead?")
                self._stop_event.set()
            if self._thread is not threading.current_thread():
                self._thread.join()
            else:
                _LOGGER.debug("Called `stop()` from managed thread?!")
            self._thread = None

        if self._client is not None:
            self._client.close()
            self._client = None

    def get_mining_job(self) -> MiningJob:
        return self._mining_job  # Simple read is atomic for Python objects

    def done_submitting_blocks(self) -> bool:
        return len(self._block_results) == 0

    def handle_submit_block(
        self, opened_block_info: OpenedBlockInfo, mining_job: MiningJob
    ) -> None:
        """
        Submit a block (in a separate async thread)
        """

        if self._loop is None:
            raise AssertionError("Async loop is not started")

        if self._pool is None:
            raise AssertionError("Thread Pool Executor is not initialized")

        def on_block_submitted() -> None:
            self.blocks_submitted += 1

        future = self._loop.run_in_executor(
            self._pool,
            self._submit_block,
            self._conf,
            opened_block_info,
            mining_job,
            self._client_config,
            on_block_submitted,
        )

        self._block_results.append(future)

    def schedule_status_check(
        self, cuda_event: torch.cuda.Event, callback: Callable[[], None]
    ) -> None:
        """Queue a status check for async processing after kernel completion.

        Args:
            cuda_event: CUDA event to monitor for completion
            callback: Function to call when the event completes
        """
        if not self._conf.enable_async_cuda_event_processing:
            warnings.warn(
                "Async CUDA event processing disabled, not scheduling callback", stacklevel=1
            )
            return
        assert self._cuda_event_queue, "Status check queue not initialized"
        self._pending_cuda_events += 1
        self._loop.call_soon_threadsafe(
            self._cuda_event_queue.put_nowait,
            CudaEventQueueItem(cuda_event=cuda_event, callback=callback),
        )

    def wait_until_done_submitting_blocks(self) -> None:
        while not self.done_submitting_blocks() or self._pending_cuda_events > 0:
            b = next(iter(self._block_results), None)
            if b is None or b.done():
                # sleep a bit to let the async loop catch up
                time.sleep(1.0)
                continue

            while not b.done():
                time.sleep(0.1)

    def register_mining_job_changed_callback(self, callback: Callable[[], None]) -> None:
        self._mining_job_changed_callbacks.append(callback)

    def _run_async_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_main())
        finally:
            if self._loop is not None:  # don't throw exception in finally
                self._loop.close()
            self._loop = None

    async def _async_main(self) -> None:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._update_gateway_loop(), name="update_gateway")

            if self._conf.enable_async_cuda_event_processing:
                tg.create_task(self._process_cuda_events_loop(), name="async_cuda_event_processing")

    async def _update_gateway_loop(self) -> None:
        assert self._client is not None, "Not started? Bug?"
        while not self._stop_event.is_set():
            try:
                while self._block_results and self._block_results[0].done():
                    # If there's an exception, we find it in the next line
                    await self._block_results.pop(0)
            except Exception:
                _LOGGER.exception("Failed to reap async results.")

            try:
                new_mining_job = await asyncio.to_thread(self._client.get_mining_info)
                if new_mining_job != self._mining_job:
                    if self._conf.print_header_hash:
                        _LOGGER.info(
                            f"Got mining job - Header Bytes: {self._mining_job.incomplete_header_bytes.hex()}, "
                            f"Target: {self._mining_job.target}"
                        )
                    for c in self._mining_job_changed_callbacks:
                        c()
                self._mining_job = new_mining_job
            except Exception:
                _LOGGER.exception("Failed to get mining info")
            await asyncio.sleep(1.0)  # Update every second

    async def _process_cuda_events_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = await asyncio.wait_for(self._cuda_event_queue.get(), timeout=10)
            except TimeoutError:
                # check the stop event
                continue

            while not item.cuda_event.query():
                await asyncio.sleep(0.01)

            # callback should be fast (setup and call `handle_submit_block`)
            try:
                item.callback(self.handle_submit_block)
            except Exception:
                _LOGGER.exception("Caught exception in CUDA event callback")
            finally:
                self._pending_cuda_events -= 1

        _LOGGER.info("Status check polling loop stopped")

    def __del__(self) -> None:
        """Cleanup when MiningState is destroyed."""
        self.stop()

    @staticmethod
    def _submit_block(
        miner_settings: MinerSettings,
        opened_block_info: OpenedBlockInfo,
        mining_job: MiningJob,
        miner_rpc_config: MinerRpcConfig,
        on_block_submitted: Callable[[], None],
    ) -> bool:
        """Submit a block to the gateway.

        The PoW check has already been performed in the kernel during mining.
        We directly create the proof and submit it.
        """
        _LOGGER.info("Block found, creating proof for submission.")

        # Create PlainProof from OpenedBlockInfo using non-noised A and B
        plain_proof = create_proof(opened_block_info, mining_job.incomplete_header_bytes)
        _LOGGER.debug("Created plain proof")

        # Crossover guard: an MoE proof cannot be certified before the fork.
        try:
            check_cert_version_eligible(mining_job.cert_version, plain_proof)
        except ValueError as e:
            _LOGGER.warning("Skipping block submission: %s", e)
            return False

        if miner_settings.debug:
            _LOGGER.debug("Verifying plain proof in debug mode")
            block_header = IncompleteBlockHeader.from_bytes(mining_job.incomplete_header_bytes)
            is_valid, message = verify_plain_proof_for_cert_version(
                mining_job.cert_version, block_header, plain_proof
            )
            if not is_valid:
                raise AssertionError(f"Plain proof verification failed: {message}")
            _LOGGER.debug("Plain proof verified")

        with _make_client(miner_settings, miner_rpc_config) as client:
            client.submit_plain_proof(plain_proof, mining_job)

        on_block_submitted()

        _LOGGER.info("Submitted plain proof to gateway")
        _LOGGER.debug(f"Proof info: {opened_block_info=}, {mining_job.to_dict()=}, {plain_proof=}")

        return True
