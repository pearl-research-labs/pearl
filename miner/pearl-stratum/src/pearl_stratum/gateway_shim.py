"""Threadsafe shim that masquerades as `miner_base.gateway_client.MiningClient`.

`AsyncLoopManager` calls `MiningClient.__init__(MinerRpcConfig)` and expects
`get_mining_info()` / `submit_plain_proof(...)` / `close()` to be synchronous.
Internally it spawns worker threads via `ThreadPoolExecutor` and re-instantiates
the client per submission (see `async_loop_manager._submit_block`).

We therefore present `MiningClient` as a thin wrapper around a module-scoped
`SharedState` singleton:

  * `SharedState.latest_job` is updated by the stratum read-loop (asyncio thread)
    on every `mining.notify`. It exposes `as_mining_job()` to synthesize the
    `MiningJob` the orchestrator expects.
  * `submit_plain_proof()` on a worker thread takes the `MiningJob` it was given
    (which we'll have produced earlier from `get_mining_info`), looks up the
    matching stratum job, and submits via `run_coroutine_threadsafe` on the
    stratum loop.

Threading invariants:
  * Reads/writes to `latest_job` are guarded by an `RLock` plus an atomic pointer
    swap (`object` reference assignment is atomic in CPython). The lock matters
    only for the (latest_job, mapping_dict) bundle being consistent together.
  * The stratum client lives on a single asyncio loop running in its own thread.
  * Worker threads only ever call:
      - `SharedState.snapshot_mining_job()`  (lock-protected read)
      - `SharedState.submit_plain_proof_blocking()`  (run_coroutine_threadsafe)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Any

try:
    # Real production types
    from pearl_gateway.comm.dataclasses import MiningJob
except ImportError:  # pragma: no cover - fallback for partial install paths
    MiningJob = None  # type: ignore[assignment]

from .job import Job
from .stratum_client import StratumClient, SubmitResult

logger = logging.getLogger(__name__)


@dataclass
class _JobMapping:
    """Bookkeeping that lets us find the stratum job_id for a given MiningJob.

    `MiningJob` only carries `incomplete_header_bytes` + `target`. The pool
    needs `job_id` on submit. We key the lookup on `incomplete_header_bytes`
    since that's the unique-per-job blob.
    """

    by_header: dict[bytes, str]
    """Map incomplete_header_bytes -> stratum job_id. Bounded; trimmed on each new job."""

    MAX_RETAINED = 16
    """Keep this many recent jobs so worker threads catching up on stale work can still resolve."""


class SharedState:
    """Process-global handle to the stratum loop.

    `StratumGatewayClient` (the MiningClient shim) reads from / writes to this.
    `bootstrap()` is called once at startup from the orchestrator's main thread
    before `AsyncLoopManager.start()`.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._latest_job: Job | None = None
        self._mapping = _JobMapping(by_header={})
        self._client: StratumClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stratum_thread: threading.Thread | None = None
        # Fires when the first mining.notify lands — used by get_mining_info()
        # so the first AsyncLoopManager.start() doesn't see a None job.
        self._first_job_event = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def bootstrap(self, client: StratumClient) -> None:
        """Attach a StratumClient and start its asyncio loop in a daemon thread.

        Wires `client.on_new_job` to our `_on_new_job` so jobs land in latest_job.
        Caller still owns the client object.
        """
        with self._lock:
            if self._client is not None:
                raise RuntimeError("SharedState.bootstrap called twice")
            self._client = client

        prev_on_new_job = client.on_new_job
        def chained_on_new_job(job: Job) -> None:
            self._on_new_job(job)
            if prev_on_new_job is not None:
                prev_on_new_job(job)
        client.on_new_job = chained_on_new_job

        self._loop = asyncio.new_event_loop()
        self._stratum_thread = threading.Thread(
            target=self._run_stratum_loop, name="pearl-stratum-asyncio", daemon=True
        )
        self._stratum_thread.start()

    def shutdown(self) -> None:
        if self._loop is None:
            return
        client = self._client
        loop = self._loop

        async def _stop() -> None:
            if client is not None:
                await client.stop()

        try:
            fut = asyncio.run_coroutine_threadsafe(_stop(), loop)
            fut.result(timeout=5)
        except Exception:
            logger.debug("Stratum loop did not stop cleanly", exc_info=True)
        loop.call_soon_threadsafe(loop.stop)
        if self._stratum_thread is not None:
            self._stratum_thread.join(timeout=5)
        try:
            loop.close()
        except Exception:
            pass

    def wait_for_first_job(self, timeout: float = 30.0) -> bool:
        return self._first_job_event.wait(timeout=timeout)

    # -- threadsafe accessors ---------------------------------------------

    def snapshot_mining_job(self) -> Any:
        """Return a MiningJob for the current head (synthesized from latest stratum job).

        Returns a brand-new MiningJob each call (so equality checks in
        AsyncLoopManager work correctly across notifies). Callers from worker
        threads land here.
        """
        with self._lock:
            job = self._latest_job
        if job is None:
            raise RuntimeError("No mining.notify received yet from stratum pool")
        if MiningJob is None:  # pragma: no cover
            raise RuntimeError("pearl_gateway.comm.dataclasses.MiningJob not importable")
        return MiningJob(
            incomplete_header_bytes=job.incomplete_header_bytes,
            target=job.target,
        )

    def lookup_job_id(self, incomplete_header_bytes: bytes) -> str | None:
        with self._lock:
            return self._mapping.by_header.get(incomplete_header_bytes)

    def submit_plain_proof_blocking(
        self, plain_proof_b64: str, incomplete_header_bytes: bytes
    ) -> SubmitResult:
        """Run `StratumClient.submit_plain_proof` on the stratum loop from a worker thread.

        Maps `incomplete_header_bytes` -> stratum job_id. If the mapping is
        missing (job already trimmed or never seen), we send the stale-job
        error code synthetically so the caller can log it the same way.
        """
        job_id = self.lookup_job_id(incomplete_header_bytes)
        if job_id is None:
            logger.warning(
                "submit_plain_proof: no stratum job_id matches header (size=%d) — dropping share",
                len(incomplete_header_bytes),
            )
            return SubmitResult(
                accepted=False, latency_ms=0.0,
                error="No matching stratum job_id", error_code=21,
            )

        if self._client is None or self._loop is None:
            raise RuntimeError("SharedState not bootstrapped")

        # NOTE: alphapool rejects `submitPlainProof` with error 25 "unknown
        # method"; all proofs (share AND block) are submitted via `mining.submit`.
        # We discovered this empirically on 2026-05-18 — see wave-6 report.
        coro = self._client.submit_share(job_id, plain_proof_b64)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=60)
        except Exception as e:
            logger.exception("submit_plain_proof crashed")
            return SubmitResult(accepted=False, latency_ms=0.0, error=str(e))

    # -- internals --------------------------------------------------------

    def _run_stratum_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._client.run())  # type: ignore[union-attr]
        except Exception:
            logger.exception("Stratum loop exited with exception")

    def _on_new_job(self, job: Job) -> None:
        """Called on the asyncio thread when `mining.notify` parses successfully."""
        with self._lock:
            self._latest_job = job
            self._mapping.by_header[job.incomplete_header_bytes] = job.job_id
            # Trim oldest entries if we're past the retention cap.
            if len(self._mapping.by_header) > _JobMapping.MAX_RETAINED:
                # dict preserves insertion order; drop the oldest N.
                excess = len(self._mapping.by_header) - _JobMapping.MAX_RETAINED
                for k in list(self._mapping.by_header.keys())[:excess]:
                    self._mapping.by_header.pop(k, None)
        self._first_job_event.set()


# Module-level singleton. `init_shared_state` must be called once before the
# AsyncLoopManager starts.
_SHARED: SharedState | None = None


def init_shared_state(client: StratumClient) -> SharedState:
    global _SHARED
    if _SHARED is not None:
        raise RuntimeError("pearl_stratum.gateway_shim.init_shared_state called twice")
    _SHARED = SharedState()
    _SHARED.bootstrap(client)
    return _SHARED


def get_shared_state() -> SharedState:
    if _SHARED is None:
        raise RuntimeError(
            "pearl_stratum SharedState not initialized — call init_shared_state() first"
        )
    return _SHARED


def reset_shared_state() -> None:
    """Test-only: tear down and clear the module singleton."""
    global _SHARED
    if _SHARED is not None:
        _SHARED.shutdown()
        _SHARED = None


# -- MiningClient shim -------------------------------------------------------


class StratumGatewayClient(AbstractContextManager):
    """Implements `miner_base.gateway_client.MiningClient` against a stratum pool.

    Constructor signature MUST match `MiningClient.__init__(MinerRpcConfig)`
    because `AsyncLoopManager._make_client` calls it that way. The argument
    is ignored — we get our state from the module-level singleton.
    """

    def __init__(self, miner_rpc_config: Any = None) -> None:
        self._state = get_shared_state()

    def get_mining_info(self) -> Any:
        return self._state.snapshot_mining_job()

    def submit_plain_proof(self, plain_proof: Any, mining_job: Any) -> None:
        """Submit a PlainProof to the pool.

        `plain_proof` is a `pearl_mining.PlainProof` with `to_base64()`.
        `mining_job` is a `pearl_gateway.comm.dataclasses.MiningJob` carrying
        `incomplete_header_bytes` + `target`. We look up the stratum job_id
        from the latter and submit.
        """
        b64 = plain_proof.to_base64()
        header_bytes = mining_job.incomplete_header_bytes
        result = self._state.submit_plain_proof_blocking(b64, header_bytes)
        if not result.accepted:
            # Match the real MiningClient's behavior: raise on rejection so the
            # AsyncLoopManager logs the exception. Stale (21) is logged but
            # NOT escalated to an exception — they're routine.
            if result.error_code == 21:
                logger.info(
                    "Share dropped (stale, job advanced): %s (latency=%.1fms)",
                    result.error, result.latency_ms,
                )
                return
            raise RuntimeError(
                f"submitPlainProof rejected: code={result.error_code} {result.error}"
            )

    def close(self) -> None:
        # Per-instance close is a no-op; the shared state owns the connection.
        # The real shutdown is via `reset_shared_state()` at process exit.
        return

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.close()
        return None


# -- helpers ------------------------------------------------------------------


def encode_plain_proof_b64(plain_proof: Any) -> str:
    """`plain_proof.to_base64()`, with a fallback for tests that pass raw bytes."""
    if hasattr(plain_proof, "to_base64"):
        return plain_proof.to_base64()
    if isinstance(plain_proof, (bytes, bytearray)):
        return base64.b64encode(plain_proof).decode("ascii")
    raise TypeError(f"plain_proof must be PlainProof or bytes, got {type(plain_proof).__name__}")
