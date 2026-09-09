"""Runs ZK proving in a dedicated, warmed worker process, off the event loop."""

import asyncio
import multiprocessing
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor

from miner_utils import get_logger

from pearl_gateway import proof_worker

logger = get_logger(__name__)


class ProofPool:
    """Runs ``proof_worker`` functions in a dedicated, warmed worker process."""

    def __init__(self) -> None:
        # spawn, not fork: the native prover (Plonky2/rayon) spawns its own
        # threads, and forking a multithreaded process risks deadlock. spawn also
        # inherits the parent environment, so the worker can read
        # PEARL_GATEWAY_WARMUP_SHAPE.
        self._mp_context = multiprocessing.get_context("spawn")
        self._executor: ProcessPoolExecutor | None = None
        self._restart_lock = asyncio.Lock()

    async def start(self) -> None:
        """Create and warm the worker so no proof is ever served on a cold worker."""
        self._executor = self._new_executor()
        await self._warm_worker(self._executor)

    async def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    async def prove(
        self,
        cert_version: int,
        incomplete_header_bytes: bytes,
        plain_proof_b64: str,
        debug: bool = False,
    ) -> tuple[bytes, bytes]:
        """Prove in the worker process; on worker death, recreate the pool and drop this proof."""
        try:
            return await self._run_prove(
                cert_version, incomplete_header_bytes, plain_proof_b64, debug
            )
        except BrokenExecutor:
            logger.warning("Proof worker died; recreating pool and dropping this proof")
            await self._restart(self._executor)
            raise

    async def _run_prove(
        self,
        cert_version: int,
        incomplete_header_bytes: bytes,
        plain_proof_b64: str,
        debug: bool,
    ) -> tuple[bytes, bytes]:
        if self._executor is None:
            raise RuntimeError("ProofPool not started")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            proof_worker.prove,
            cert_version,
            incomplete_header_bytes,
            plain_proof_b64,
            debug,
        )

    def _new_executor(self) -> ProcessPoolExecutor:
        # One worker: proof generation is already internally multi-threaded, so more
        # worker processes would just oversubscribe cores.
        return ProcessPoolExecutor(
            max_workers=1,
            mp_context=self._mp_context,
            initializer=proof_worker.worker_init,
        )

    async def _restart(self, broken: ProcessPoolExecutor | None) -> None:
        """Replace a broken executor with a fresh, warmed one (idempotent under concurrency)."""
        async with self._restart_lock:
            if self._executor is not broken:
                return
            if broken is not None:
                broken.shutdown(wait=False, cancel_futures=True)
            self._executor = self._new_executor()
            await self._warm_worker(self._executor)

    async def _warm_worker(self, executor: ProcessPoolExecutor) -> None:
        """Spawn the worker and wait for its warmup initializer to finish."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, proof_worker.ready_probe)
        logger.info("ZK proof worker ready")
