"""Hot-spare failover wrapper for StratumClient.

PROBLEM:
  Single-socket `StratumClient` loses 100% of submission capacity during a TCP
  reconnect storm. Production baseline (2026-05-19) observed `[Errno 104]
  Connection reset by peer` mid-submit + 62 second backoff before reconnect
  completed handshake — every share found during that window was unsubmittable.

  alpha-miner accidentally gets fault isolation by opening one socket per GPU:
  if GPU3's socket RSTs, GPUs 0/1/2 keep submitting. We don't have that since
  pearl-stratum runs a single shared connection.

SOLUTION:
  Maintain N parallel `StratumClient` instances to the same pool, authorized
  with different worker suffixes (`worker`, `worker-spare`, ...). Both/all
  receive the same `mining.notify` stream (pool job_ids are global; see
  STRATUM_CAPTURE.md §3f). Primary handles every submit; on any non-stale
  failure (RST, timeout, unexpected error code) the failover immediately
  retries on the next healthy peer with a tight per-attempt deadline.

WHY job_id failover works:
  Pearl/v1 has no per-connection extranonce (STRATUM_CAPTURE.md line 58:
  "pearl/v1 has no client-side nonce-rolling at the stratum layer"). The
  `job_id` returned in `mining.notify` is a pool-global `BLOCK-SEQ` string;
  identical across simultaneous connections. A share computed against the
  current job is submittable on any socket that's authorized.

DOES NOT FIX:
  Correlated outages (pool-side ban, our IP rate-limited, both sockets RST
  simultaneously). Those are logged explicitly via `Stats.both_failed`.

CONSTRAINTS:
  - Max 2 sockets per pool to avoid alphapool per-IP rate-limit.
  - Worker names MUST differ (`worker` vs `worker-spare`) — alphapool does
    NOT explicitly forbid duplicate workers, but credit attribution is
    cleaner with distinct workers and total-to-address is unaffected.
  - We start both StratumClients eagerly; first job from EITHER socket fires
    `on_new_job` to the consumer. Subsequent duplicates (same job_id) are
    suppressed to avoid double-firing the driver.

API SURFACE (must match enough of `StratumClient` for `gateway_shim.bootstrap`):
  - `on_new_job`, `on_set_difficulty`, `on_set_mining_params` settable
  - `current_job`, `mining_params`, `stats`, `connected`
  - `async run()` — fires up all peers, returns when all are stopped
  - `async stop()` — stop all peers
  - `async submit_share(job_id, plain_proof_b64) -> SubmitResult` — failover-aware
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .job import Job
from .stratum_client import (
    StaleShareError,
    StratumClient,
    StratumProtocolError,
    StratumStats,
    SubmitResult,
)

logger = logging.getLogger(__name__)


# Per-attempt deadline for a single peer's submit. The primary gets
# PRIMARY_SUBMIT_TIMEOUT_S; if that elapses (or returns a non-stale failure)
# we cut over to the spare within FAILOVER_BUDGET_S total wall-time.
PRIMARY_SUBMIT_TIMEOUT_S = 5.0
SPARE_SUBMIT_TIMEOUT_S = 5.0
# Target wall-time for the failover handoff itself (peer selection + write).
# We aim for sub-50ms but allow up to 500ms before logging a slow-failover event.
FAILOVER_TARGET_MS = 50.0
FAILOVER_LOG_THRESHOLD_MS = 500.0


@dataclass
class FailoverStats:
    """Aggregate failover bookkeeping. Per-peer stats live on each StratumClient."""

    primary_accepts: int = 0
    primary_failures: int = 0
    spare_accepts: int = 0
    spare_failures: int = 0
    """Submits where the spare also failed (or hadn't connected). Includes stale-on-spare."""
    failover_attempts: int = 0
    failover_successes: int = 0
    """Failover catches: primary failed (non-stale), spare succeeded."""
    both_failed: int = 0
    """Correlated outages — no peer could submit. The metric you watch for ‘failover didn't help'."""
    stale: int = 0
    """Pool returned 21 on the primary — we do NOT retry stale shares on the spare."""
    failover_latencies_ms: list[float] = field(default_factory=list)
    """Wall-time from primary-fail to spare-accept. For p50/p99 reporting."""

    def total_accepts(self) -> int:
        return self.primary_accepts + self.spare_accepts

    def total_submits(self) -> int:
        return (
            self.primary_accepts + self.primary_failures + self.stale + self.both_failed
        )


class FailoverStratumClient:
    """N-peer hot-spare wrapper around `StratumClient`.

    Construct with same pool args + a list of worker-name overrides per peer.
    Index 0 is the PRIMARY; the rest are spares queried in order.

    The wrapper is itself a small asyncio object — does not require its own
    event loop. Callers should drive it from the same loop the StratumClients
    use (the gateway_shim manages this).
    """

    def __init__(
        self,
        host: str,
        port: int,
        address: str,
        worker: str,
        password: str = "x",
        user_agent: str = "alpha-miner/0.1",
        *,
        n_peers: int = 2,
        worker_suffix_for_spare: str = "-spare",
        on_new_job: Callable[[Job], None] | None = None,
        on_set_difficulty: Callable[[float], None] | None = None,
        on_set_mining_params: Callable[[dict[str, Any]], None] | None = None,
        on_disconnect: Callable[[str], None] | None = None,
    ) -> None:
        if n_peers < 1:
            raise ValueError(f"n_peers must be >= 1, got {n_peers}")
        if n_peers > 2:
            # Hard guardrail: alphapool rate-limits per IP. >2 simultaneous
            # connections from one IP risks a ban (per memory:
            # "us2 had rate-limited CPU01's IP — us1 is the clean test endpoint").
            raise ValueError(
                f"n_peers > 2 risks pool-side IP rate-limit; got {n_peers}"
            )

        self.host = host
        self.port = port
        self.address = address
        self.worker = worker
        self.password = password
        self.user_agent = user_agent
        self.n_peers = n_peers

        # User-supplied callbacks. We chain into peer-level callbacks below so
        # only the FIRST peer to deliver a given event fires the consumer's
        # callback — avoids double-notify.
        self._user_on_new_job = on_new_job
        self._user_on_set_difficulty = on_set_difficulty
        self._user_on_set_mining_params = on_set_mining_params
        self._user_on_disconnect = on_disconnect

        # Job-id dedup: when N=2, both peers receive the same notify; we want
        # the consumer's on_new_job to fire once per unique job_id only.
        self._seen_job_ids: set[str] = set()
        self._seen_job_ids_lock: asyncio.Lock | None = None

        # Aggregate / forwarded state (so existing consumers can read `.current_job`)
        self.current_job: Job | None = None
        self.mining_params: dict[str, Any] | None = None
        self.stats = StratumStats()  # primary's stats; per-peer stats also kept

        self.failover_stats = FailoverStats()

        # Build the peer clients.
        self.peers: list[StratumClient] = []
        for i in range(n_peers):
            peer_worker = worker if i == 0 else f"{worker}{worker_suffix_for_spare}"
            peer = StratumClient(
                host=host,
                port=port,
                address=address,
                worker=peer_worker,
                password=password,
                user_agent=user_agent,
                on_new_job=self._make_peer_on_new_job(i),
                on_set_difficulty=self._make_peer_on_set_difficulty(i),
                on_set_mining_params=self._make_peer_on_set_mining_params(i),
                on_disconnect=self._make_peer_on_disconnect(i),
            )
            self.peers.append(peer)

        # The asyncio Lock used in submit_share to serialize cutover. Lazily
        # created on the running loop the first time submit_share is called.
        self._submit_lock: asyncio.Lock | None = None

        # Per-peer "had at least one successful submit" — for picking healthy
        # spare. A peer with zero successful submits is degraded until proven.
        # (We initialize all peers to True so cold-start spare can be tried.)
        self._peer_healthy: list[bool] = [True] * n_peers

    # ---- properties matching StratumClient API ---------------------------

    @property
    def worker_name(self) -> str:
        """The primary's worker name (what the consumer expects)."""
        return self.peers[0].worker_name

    @property
    def connected(self) -> bool:
        """True if ANY peer is connected."""
        return any(p.connected for p in self.peers)

    @property
    def on_new_job(self) -> Callable[[Job], None] | None:
        return self._user_on_new_job

    @on_new_job.setter
    def on_new_job(self, fn: Callable[[Job], None] | None) -> None:
        # The gateway_shim chains its own on_new_job ON TOP of the existing
        # callback (see SharedState.bootstrap). We support that pattern.
        self._user_on_new_job = fn

    # ---- lifecycle -------------------------------------------------------

    # Stagger time between successive peer.run() task launches. Observed
    # empirically against us1.alphapool.tech 2026-05-20: simultaneous TCP
    # connect attempts from the same IP get burst-rejected with `[Errno 104]
    # Connection reset by peer` within 60ms. Staggering peer start by 500ms
    # avoids the burst-protection without slowing first-job-from-pool.
    PEER_START_STAGGER_S = 0.5

    async def run(self) -> int:
        """Run all peers concurrently. Returns when all have exited.

        Each peer has its own reconnect loop; we don't interfere. The overall
        wrapper exits with 0 if ANY peer ran cleanly to .stop(); 2 if all
        tripped their circuit breaker (mirrors single-client semantics).

        Peer tasks are started with a small inter-launch stagger so the pool's
        per-IP burst-protection doesn't RST both connections simultaneously.
        """
        if self._submit_lock is None:
            self._submit_lock = asyncio.Lock()
        if self._seen_job_ids_lock is None:
            self._seen_job_ids_lock = asyncio.Lock()

        tasks: list[asyncio.Task] = []
        for i, p in enumerate(self.peers):
            if i > 0:
                await asyncio.sleep(self.PEER_START_STAGGER_S)
            tasks.append(asyncio.create_task(p.run(), name=f"stratum-peer-{i}"))
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise
        # Aggregate: nonzero if ALL peers reported nonzero (circuit breaker).
        exit_codes = [r for r in results if isinstance(r, int)]
        if exit_codes and all(c != 0 for c in exit_codes):
            return 2
        return 0

    async def stop(self) -> None:
        """Stop all peers in parallel."""
        await asyncio.gather(*(p.stop() for p in self.peers), return_exceptions=True)

    # ---- peer-callback factories ----------------------------------------

    def _make_peer_on_new_job(self, peer_idx: int) -> Callable[[Job], None]:
        def _cb(job: Job) -> None:
            # Dedup: only the FIRST peer to deliver this job_id wakes the
            # consumer. The set is bounded by trimming (see _trim_seen).
            if job.job_id in self._seen_job_ids:
                logger.debug("peer=%d duplicate notify job=%s (dedup'd)", peer_idx, job.job_id)
                return
            self._seen_job_ids.add(job.job_id)
            self._trim_seen()
            self.current_job = job
            if self._user_on_new_job is not None:
                try:
                    self._user_on_new_job(job)
                except Exception:
                    logger.exception("user on_new_job callback raised (peer=%d)", peer_idx)
        return _cb

    def _make_peer_on_set_difficulty(self, peer_idx: int) -> Callable[[float], None]:
        def _cb(diff: float) -> None:
            # Mirror the primary's diff into wrapper-level stats (legacy consumers).
            if peer_idx == 0:
                self.stats.last_diff = diff
            if self._user_on_set_difficulty is not None:
                try:
                    self._user_on_set_difficulty(diff)
                except Exception:
                    logger.exception("user on_set_difficulty callback raised (peer=%d)", peer_idx)
        return _cb

    def _make_peer_on_set_mining_params(self, peer_idx: int) -> Callable[[dict[str, Any]], None]:
        def _cb(params: dict[str, Any]) -> None:
            self.mining_params = params
            if self._user_on_set_mining_params is not None:
                try:
                    self._user_on_set_mining_params(params)
                except Exception:
                    logger.exception("user on_set_mining_params callback raised (peer=%d)", peer_idx)
        return _cb

    def _make_peer_on_disconnect(self, peer_idx: int) -> Callable[[str], None]:
        def _cb(reason: str) -> None:
            logger.warning("peer=%d disconnect: %s", peer_idx, reason)
            self._peer_healthy[peer_idx] = False  # marked unhealthy until next accept
            if self._user_on_disconnect is not None:
                try:
                    self._user_on_disconnect(reason)
                except Exception:
                    logger.exception("user on_disconnect callback raised")
        return _cb

    def _trim_seen(self, max_keep: int = 64) -> None:
        if len(self._seen_job_ids) > max_keep:
            # Drop a chunk of the oldest. set() doesn't preserve insertion order
            # cross-platform; for our purposes the trim is best-effort.
            excess = len(self._seen_job_ids) - max_keep
            for j in list(self._seen_job_ids)[:excess]:
                self._seen_job_ids.discard(j)

    # ---- submit with failover -------------------------------------------

    async def submit_share(
        self,
        job_id: str,
        plain_proof_b64: str,
    ) -> SubmitResult:
        """Try primary; on non-stale failure, retry on first healthy spare.

        Stale (error 21) is NEVER retried on the spare. The pool determined the
        share is for an old block height; submitting it on another socket
        won't change that. Stale is a per-share verdict, not a per-connection
        verdict.

        For all other failures (RST, timeout, error 23 low-diff, unknown
        codes) we re-submit on the spare. We tolerate the worst case (both
        peers fail / spare not yet authorized) by returning the spare's
        SubmitResult so the caller can log accordingly.

        Returns SubmitResult exactly as `StratumClient.submit_share` does;
        consumers can keep their existing accept-or-not branching.
        """
        # Lazy lock init (we may be called from worker threads via
        # run_coroutine_threadsafe; the lock is loop-local).
        if self._submit_lock is None:
            self._submit_lock = asyncio.Lock()

        t_wall_start = time.monotonic()
        primary = self.peers[0]

        # Phase 1: try primary with a per-call deadline.
        try:
            primary_result = await asyncio.wait_for(
                primary.submit_share(job_id, plain_proof_b64),
                timeout=PRIMARY_SUBMIT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "primary submit_share timed out after %.1fs; failing over",
                PRIMARY_SUBMIT_TIMEOUT_S,
            )
            primary_result = SubmitResult(
                accepted=False,
                latency_ms=PRIMARY_SUBMIT_TIMEOUT_S * 1000,
                error=f"primary timeout after {PRIMARY_SUBMIT_TIMEOUT_S}s",
                error_code=None,
            )
        except StratumProtocolError as e:
            # _wait_for_connection or call() bubbled a socket-level error.
            logger.warning("primary submit_share raised StratumProtocolError: %s", e)
            primary_result = SubmitResult(
                accepted=False, latency_ms=0.0,
                error=str(e), error_code=getattr(e, "code", None),
            )
        except Exception as e:
            # Catch-all for connection-dropped / unexpected. Same intent as
            # StratumProtocolError but we treat it as a recoverable per-peer
            # failure rather than crashing the wrapper.
            logger.warning("primary submit_share raised %s: %s", type(e).__name__, e)
            primary_result = SubmitResult(
                accepted=False, latency_ms=0.0,
                error=f"{type(e).__name__}: {e}",
                error_code=None,
            )

        # Accept: done.
        if primary_result.accepted:
            self.failover_stats.primary_accepts += 1
            self._peer_healthy[0] = True
            return primary_result

        # Stale (21): NEVER retry on spare. The share is genuinely dead.
        if primary_result.error_code == 21:
            self.failover_stats.stale += 1
            return primary_result

        # Phase 2: failover. Try each spare in order until one accepts or we
        # exhaust them.
        self.failover_stats.primary_failures += 1
        self.failover_stats.failover_attempts += 1
        self._peer_healthy[0] = False

        t_fail_start = time.monotonic()
        spare_result: SubmitResult | None = None
        for spare_idx in range(1, self.n_peers):
            spare = self.peers[spare_idx]
            if not spare.connected:
                logger.warning(
                    "spare peer=%d not connected; skipping", spare_idx,
                )
                continue
            try:
                spare_result = await asyncio.wait_for(
                    spare.submit_share(job_id, plain_proof_b64),
                    timeout=SPARE_SUBMIT_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "spare peer=%d submit timed out after %.1fs",
                    spare_idx, SPARE_SUBMIT_TIMEOUT_S,
                )
                spare_result = SubmitResult(
                    accepted=False, latency_ms=SPARE_SUBMIT_TIMEOUT_S * 1000,
                    error=f"spare-{spare_idx} timeout",
                    error_code=None,
                )
            except StratumProtocolError as e:
                logger.warning(
                    "spare peer=%d raised StratumProtocolError: %s", spare_idx, e,
                )
                spare_result = SubmitResult(
                    accepted=False, latency_ms=0.0,
                    error=str(e), error_code=getattr(e, "code", None),
                )
            except Exception as e:
                logger.warning(
                    "spare peer=%d raised %s: %s",
                    spare_idx, type(e).__name__, e,
                )
                spare_result = SubmitResult(
                    accepted=False, latency_ms=0.0,
                    error=f"{type(e).__name__}: {e}",
                    error_code=None,
                )

            if spare_result.accepted:
                failover_ms = (time.monotonic() - t_fail_start) * 1000
                self.failover_stats.failover_successes += 1
                self.failover_stats.spare_accepts += 1
                self.failover_stats.failover_latencies_ms.append(failover_ms)
                self._peer_healthy[spare_idx] = True
                if failover_ms > FAILOVER_LOG_THRESHOLD_MS:
                    logger.warning(
                        "SLOW failover: peer=%d caught share in %.1fms (target %.1fms)",
                        spare_idx, failover_ms, FAILOVER_TARGET_MS,
                    )
                else:
                    logger.info(
                        "failover: peer=%d caught share in %.1fms",
                        spare_idx, failover_ms,
                    )
                # Total wall-time including primary attempt
                total_ms = (time.monotonic() - t_wall_start) * 1000
                return SubmitResult(
                    accepted=True,
                    latency_ms=total_ms,
                    error=None,
                    error_code=None,
                )
            # Spare also failed — try the next one (if any), else fall through.
            if spare_result.error_code == 21:
                # Stale-on-spare means the share was genuinely too old
                # (block advanced between primary attempt and spare attempt).
                # No point trying further spares.
                self.failover_stats.stale += 1
                return spare_result
            self.failover_stats.spare_failures += 1

        # No spare accepted — return the spare's last result, or the primary's
        # if no spare was available at all.
        self.failover_stats.both_failed += 1
        if spare_result is not None:
            logger.warning(
                "BOTH FAILED: primary+all spares rejected/errored; primary_err=%r spare_err=%r",
                primary_result.error, spare_result.error,
            )
            return spare_result
        logger.warning(
            "NO SPARE AVAILABLE: primary failed and no spare was connected; primary_err=%r",
            primary_result.error,
        )
        return primary_result
