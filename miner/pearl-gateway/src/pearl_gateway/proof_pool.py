"""Runs ZK proving in a dedicated worker process, off the event loop."""

import asyncio
import multiprocessing
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor

from miner_utils import get_logger

from pearl_gateway import proof_worker

logger = get_logger(__name__)


class ProofPool:
    def __init__(self) -> None:
        # spawn, not fork: the native prover (Plonky2/rayon) spawns its own
        # threads, and forking a multithreaded process risks deadlock.
        self._mp_context = multiprocessing.get_context("spawn")
        self._executor: ProcessPoolExecutor | None = None
        self._restart_lock = asyncio.Lock()

    async def start(self) -> None:
        self._executor = self._new_executor()
        # Spawn the worker now: with spawn, interpreter boot plus the native prover
        # import take seconds, which the first proof would otherwise wait out.
        await asyncio.get_running_loop().run_in_executor(self._executor, proof_worker.prewarm)

    async def stop(self) -> None:
        if self._executor is not None:
            # shutdown(wait=False) does not interrupt a proof already running, and the
            # worker is not a daemon, so interpreter exit would block until it finishes.
            # An in-flight proof is dropped on stop anyway, so kill the worker
            # (SIGKILL: it cannot be caught or delayed).
            for process in list((self._executor._processes or {}).values()):
                process.kill()
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    async def prove(
        self,
        cert_version: int,
        incomplete_header_bytes: bytes,
        plain_proof_b64: str,
        debug: bool = False,
    ) -> tuple[bytes, bytes]:
        """Prove in the worker process.

        If the worker is dead (it may have died while idle, before this job was
        dispatched), recreate it and retry once. A proof that kills the worker itself
        is tried twice and then dropped.
        """
        for attempt in range(2):
            # Snapshot: stop() or a concurrent restart may replace _executor while we wait.
            executor = self._executor
            try:
                return await self._run_prove(
                    executor, cert_version, incomplete_header_bytes, plain_proof_b64, debug
                )
            except BrokenExecutor:
                logger.warning("Proof worker died; recreating it")
                await self._restart(executor)
                if attempt:
                    raise
        raise AssertionError("unreachable")

    async def _run_prove(
        self,
        executor: ProcessPoolExecutor | None,
        cert_version: int,
        incomplete_header_bytes: bytes,
        plain_proof_b64: str,
        debug: bool,
    ) -> tuple[bytes, bytes]:
        if executor is None:
            raise RuntimeError("ProofPool not started")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            proof_worker.prove,
            cert_version,
            incomplete_header_bytes,
            plain_proof_b64,
            debug,
        )

    def _new_executor(self) -> ProcessPoolExecutor:
        # One worker: proof generation is already internally multi-threaded, so more
        # worker processes would just oversubscribe cores.
        return ProcessPoolExecutor(max_workers=1, mp_context=self._mp_context)

    async def _restart(self, broken: ProcessPoolExecutor | None) -> None:
        """Replace a broken executor with a fresh one (idempotent under concurrency)."""
        async with self._restart_lock:
            # None: the pool was stopped. Not current: someone already replaced it.
            if broken is None or self._executor is not broken:
                return
            broken.shutdown(wait=False, cancel_futures=True)
            self._executor = self._new_executor()
