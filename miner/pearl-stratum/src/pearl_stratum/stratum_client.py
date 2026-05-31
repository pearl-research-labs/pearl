"""Asyncio Pearl/v1 stratum client.

Talks plain-TCP line-delimited JSON-RPC to a pool (us2.alphapool.tech:5566 in
prod). Implements the dialogue RE'd from alpha-miner v1.4.0:

    C -> mining.configure  [["pearl/v1"], {}]
    S -> {pearl/v1: true, ...}
    C -> mining.subscribe  ["pearl-stratum/0.1"]
    S -> [[...subscriptions...], extranonce1, extranonce2_size]
    S -> pearl.set_mining_params {m,n,k,rank,rows_pattern,cols_pattern,mma_type}   (unsolicited)
    C -> mining.authorize  ["ADDRESS.WORKER", "x;d=1048576"]
    S -> true
    S -> mining.set_difficulty [diff]
    S -> mining.notify [job_id, prevhash, incomplete_header_bytes, ntime, nbits, ver, clean_jobs]
    ...
    C -> mining.submit       [worker, job_id, plain_proof_b64]              (per share)
    C -> submitPlainProof    {plain_proof: <b64>, mining_job: {...}}        (full block)
    S -> {result: true}                                                      (accept)
    S -> {error: [21, "chain advanced ...", null]}                           (stale — DROP, not reconnect)

The bug-fix point per `C:/Source/pearl-investigation/alphafix.c`: error code 21
must NOT trigger socket close. alpha-miner v1.4.0 incorrectly tears down TCP on
21 and rebuilds the whole session, costing ~490 ms per stale share. We treat 21
strictly as a per-share reject (`StaleShareError`) and keep the socket open.

Threading model: this module runs entirely inside one asyncio event loop. The
`GatewayShim` in gateway_shim.py is the threadsafe boundary that lets the
synchronous `AsyncLoopManager` worker threads call into us.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import blake3

from .job import Job, parse_notify

logger = logging.getLogger(__name__)


# pearl.challenge: server-side DDoS pacer added in alphapool v1.5 (2026-05-19).
# After TCP connect, server sends `pearl.challenge` (unsolicited, id=null) and
# expects a `pearl.challenge_response` containing a u64 nonce such that
# `blake3(seed_bytes || nonce.to_bytes(8, "little"))` has `difficulty` leading
# zero bits. Server may re-issue mid-session.
#
# Wire-format and PoW verified bit-exact against 9 captured (seed, nonce) tuples
# in pearl-investigation/wave16-domination/58_pearl_challenge_protocol.md.
PEARL_CHALLENGE_METHOD = "pearl.challenge"
PEARL_CHALLENGE_RESPONSE_METHOD = "pearl.challenge_response"

# Native C solvers live next to this module (compiled by build script on Linux).
# We prefer the AVX-512 16-way batched solver (`pearl_challenge_solver_simd`,
# ~1.5 GH/s on Zen4 -> diff=32 median ~3 s); fall back to the OpenMP scalar
# solver (~160 MH/s -> diff=32 median ~30 s); fall back to pure Python.
# When present, used for difficulty >= _C_SOLVER_THRESHOLD. Pure-Python solver
# below is kept as fallback for Windows / test environments and for tiny diffs
# where subprocess overhead would dominate.
_C_SOLVER_SIMD_BIN = os.path.join(os.path.dirname(__file__), "pearl_challenge_solver_simd")
_C_SOLVER_REF_BIN = os.path.join(os.path.dirname(__file__), "pearl_challenge_solver")
_C_SOLVER_THRESHOLD = 20  # below this, Python is fast enough and subprocess fork is overkill
_C_SOLVER_TIMEOUT_S = 600.0


def _pick_solver_bin() -> str | None:
    """Return path to the fastest available native solver, or None for Python.

    Resolution order: AVX-512 SIMD > scalar OpenMP > None.
    """
    if os.path.exists(_C_SOLVER_SIMD_BIN):
        return _C_SOLVER_SIMD_BIN
    if os.path.exists(_C_SOLVER_REF_BIN):
        return _C_SOLVER_REF_BIN
    return None


def _solve_pearl_challenge_python(seed_hex: str, difficulty: int) -> str:
    """Pure-Python reference solver. ~1.6 MH/s. Difficulty=32 takes ~40 min.

    Kept as the cross-platform fallback (Windows / test fixtures) and for very
    small difficulties where subprocess overhead would dominate.
    """
    seed = bytes.fromhex(seed_hex)
    if len(seed) != 32:
        raise ValueError(f"expected 32-byte seed, got {len(seed)}")
    full_zero_bytes, leftover_bits = divmod(difficulty, 8)
    leftover_mask = 0xFF ^ ((1 << (8 - leftover_bits)) - 1) if leftover_bits else 0
    prefix = b"\x00" * full_zero_bytes
    nonce = 0
    while True:
        h = blake3.blake3(seed + nonce.to_bytes(8, "little")).digest()
        if h[:full_zero_bytes] == prefix:
            if leftover_bits == 0 or (h[full_zero_bytes] & leftover_mask) == 0:
                return f"{nonce:016x}"
        nonce += 1


def _solve_pearl_challenge(seed_hex: str, difficulty: int) -> str:
    """Find a u64 nonce such that blake3(seed||nonce.to_bytes(8,'little')) has
    `difficulty` leading zero bits.

    Returns the nonce as a 16-hex-char string. Uses the bundled C solver
    (AVX-512 SIMD when available, OpenMP scalar otherwise); falls back to a
    pure-Python loop.

    Caller MUST run this in a worker thread (asyncio.to_thread) to avoid
    blocking the event loop — even the C solver takes seconds at difficulty=32.
    """
    seed = bytes.fromhex(seed_hex)
    if len(seed) != 32:
        raise ValueError(f"expected 32-byte seed, got {len(seed)}")

    solver_bin = _pick_solver_bin()
    if difficulty >= _C_SOLVER_THRESHOLD and solver_bin is not None:
        try:
            r = subprocess.run(
                [solver_bin, seed_hex, str(difficulty)],
                capture_output=True,
                text=True,
                timeout=_C_SOLVER_TIMEOUT_S,
            )
            if r.returncode != 0:
                logger.warning(
                    "C solver %s failed (rc=%d), falling back to Python: %s",
                    solver_bin, r.returncode, r.stderr.strip(),
                )
            else:
                out = r.stdout.strip()
                if len(out) == 16:
                    try:
                        int(out, 16)
                        return out
                    except ValueError:
                        logger.warning("C solver returned non-hex output %r; falling back", out)
                else:
                    logger.warning("C solver returned wrong-length output %r; falling back", out)
        except subprocess.TimeoutExpired:
            logger.warning("C solver timed out after %.0fs; falling back to Python", _C_SOLVER_TIMEOUT_S)
        except (OSError, ValueError) as e:
            logger.warning("C solver invocation error %r; falling back to Python", e)

    return _solve_pearl_challenge_python(seed_hex, difficulty)


# Exponential reconnect backoff (seconds). Cap at 60s.
RECONNECT_BACKOFF = [1.0, 2.0, 5.0, 10.0, 30.0, 60.0]

# Consecutive connect failures before we give up and exit nonzero so the
# supervisor (mfarm-agent) can restart the process.
CIRCUIT_BREAKER_FAILURES = 10

# Per-call response timeout for configure/subscribe/authorize (seconds).
HANDSHAKE_TIMEOUT_S = 30.0

# Initial-frame timeout for `pearl.challenge` detection. The server is expected
# to push the challenge within ~1 ms of TCP accept (no client traffic
# required). We use a short timeout so legacy v1.4 pools (no challenge) don't
# hold us up for the full handshake budget.
INITIAL_CHALLENGE_TIMEOUT_S = 2.0

# Per-submit response timeout. The capture shows the pool can take up to ~2s
# to validate a 270 KB plain_proof — give a generous margin.
SUBMIT_TIMEOUT_S = 30.0


class StaleShareError(Exception):
    """Pool returned error code 21. Per-share reject only — do NOT reconnect."""

    def __init__(self, message: str, job_id: str | None = None):
        super().__init__(message)
        self.job_id = job_id


class StratumProtocolError(Exception):
    """Malformed pool message or handshake failure."""


@dataclass
class SubmitResult:
    accepted: bool
    latency_ms: float
    error: str | None = None
    error_code: int | None = None


@dataclass
class StratumStats:
    accepted: int = 0
    rejected: int = 0
    dropped_stale_jobid: int = 0
    """Submits the pool returned error 21 (stale / chain advanced) for."""

    challenges_solved: int = 0
    """Count of `pearl.challenge` PoW puzzles successfully solved + ack'd."""

    last_diff: float = 0.0
    ema_hr: float = 0.0
    """Exponential-moving average accept-rate (accepts/sec, alpha=0.1)."""

    _last_accept_at: float = field(default=0.0, repr=False)

    def note_accept(self) -> None:
        """Update ema_hr from inter-arrival time. Call AFTER incrementing `accepted`."""
        now = time.time()
        if self._last_accept_at > 0:
            dt = max(now - self._last_accept_at, 1e-3)
            inst = 1.0 / dt
            self.ema_hr = 0.1 * inst + 0.9 * self.ema_hr
        self._last_accept_at = now


@dataclass
class _PendingRequest:
    """Slot for a JSON-RPC response future, keyed by id."""

    future: asyncio.Future
    method: str
    sent_at: float


def parse_pool_url(url: str) -> tuple[str, int]:
    """Accept `stratum+tcp://host:port`, `tcp://host:port`, or bare `host:port`."""
    if "://" in url:
        parsed = urlparse(url)
        if parsed.scheme not in ("stratum+tcp", "tcp"):
            raise ValueError(f"Unsupported pool URL scheme: {parsed.scheme!r}")
        if not parsed.hostname or not parsed.port:
            raise ValueError(f"Pool URL needs host and port: {url!r}")
        return parsed.hostname, parsed.port
    if ":" not in url:
        raise ValueError(f"Pool URL needs `host:port`: {url!r}")
    host, port_s = url.rsplit(":", 1)
    return host, int(port_s)


class StratumClient:
    """Persistent stratum-pool connection. Single event loop, single asyncio.Lock for writes.

    Concurrency contract:
      * Construct on the asyncio event loop; never share across loops.
      * `run()` is the main coroutine; spawn it as a task and let it loop forever.
      * `submit_share` / `submit_plain_proof` are coroutines that block on a Future
        until the pool answers (or we time out / reconnect).
      * Callbacks fire on the event loop thread; threadsafe handoff is the caller's
        responsibility (see GatewayShim).
    """

    def __init__(
        self,
        host: str,
        port: int,
        address: str,
        worker: str,
        password: str = "x",
        user_agent: str = "alpha-miner/0.1",
        on_new_job: Callable[[Job], None] | None = None,
        on_set_difficulty: Callable[[float], None] | None = None,
        on_set_mining_params: Callable[[dict[str, Any]], None] | None = None,
        on_disconnect: Callable[[str], None] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.address = address
        self.worker = worker
        self.password = password
        self.user_agent = user_agent

        self.on_new_job = on_new_job
        self.on_set_difficulty = on_set_difficulty
        self.on_set_mining_params = on_set_mining_params
        self.on_disconnect = on_disconnect

        self.stats = StratumStats()

        # Latest pool-pushed mining-params dict (from `pearl.set_mining_params`).
        # Echoed back inside `mining_job` on `submitPlainProof`.
        self.mining_params: dict[str, Any] | None = None

        # Most recent job. None until the first mining.notify arrives.
        self.current_job: Job | None = None

        # JSON-RPC plumbing
        self._next_id = 1
        self._pending: dict[int, _PendingRequest] = {}
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._write_lock = asyncio.Lock()
        self._connected_event = asyncio.Event()
        self._stop = asyncio.Event()

        # Reconnect bookkeeping
        self._consecutive_failures = 0

    # ---- public API ------------------------------------------------------

    @property
    def worker_name(self) -> str:
        """The full `ADDRESS.WORKER` string sent on mining.authorize and mining.submit."""
        if self.worker:
            return f"{self.address}.{self.worker}"
        return self.address

    @property
    def connected(self) -> bool:
        return self._connected_event.is_set()

    async def stop(self) -> None:
        self._stop.set()
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass

    async def run(self) -> int:
        """Main reconnect loop. Returns nonzero on circuit-breaker trip.

        The supervisor (mfarm-agent) is expected to restart the process when
        we exit nonzero; we never silently spin past the breaker.
        """
        while not self._stop.is_set():
            try:
                await self._open_and_handshake()
                self._consecutive_failures = 0
                await self._read_loop()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("Stratum session error: %s", e)
                if self.on_disconnect is not None:
                    try:
                        self.on_disconnect(str(e))
                    except Exception:
                        logger.exception("on_disconnect callback raised")
            finally:
                await self._close_socket()
                self._connected_event.clear()
                self._fail_pending("connection dropped")

            if self._stop.is_set():
                break

            self._consecutive_failures += 1
            if self._consecutive_failures >= CIRCUIT_BREAKER_FAILURES:
                logger.error(
                    "Stratum circuit breaker tripped after %d consecutive failures, exiting",
                    self._consecutive_failures,
                )
                return 2

            backoff_idx = min(self._consecutive_failures - 1, len(RECONNECT_BACKOFF) - 1)
            backoff = RECONNECT_BACKOFF[backoff_idx]
            logger.info("Reconnecting in %.1fs (failure #%d)", backoff, self._consecutive_failures)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass

        return 0

    async def submit_share(
        self,
        job_id: str,
        plain_proof_b64: str,
    ) -> SubmitResult:
        """Send `mining.submit` (positional params, alpha-miner-compatible wire form).

        Returns SubmitResult. Error 21 is reported with accepted=False; we never
        raise StaleShareError out of here — callers care about pool latency stats
        regardless of accept/reject and can branch on `error_code`.

        Wave-7 instrumentation: logs the exact pool response value (True / None /
        other) so we can disambiguate "accepted protocol-level but rejected by
        backend" from "accepted everywhere". Pool-side silent rejection (returns
        `{"result": true}` but never credits) is the failure mode wave-6 hit.
        """
        await self._wait_for_connection()
        params = [self.worker_name, job_id, plain_proof_b64]
        t0 = time.monotonic()
        try:
            raw_result = await self._call("mining.submit", params)
            latency_ms = (time.monotonic() - t0) * 1000
            # WAVE-7: log raw response — we suspect pool returns `result: true`
            # protocol-level but silently rejects the share later. Capturing
            # the raw value (True vs None vs object) helps diagnose backend
            # rejection patterns. NEEDS: subsequent shares24h delta check.
            logger.info(
                "mining.submit OK: job_id=%s latency=%.1fms raw_result=%r proof_b64_len=%d",
                job_id, latency_ms, raw_result, len(plain_proof_b64),
            )
            self.stats.accepted += 1
            self.stats.note_accept()
            return SubmitResult(accepted=True, latency_ms=latency_ms)
        except StaleShareError as e:
            latency_ms = (time.monotonic() - t0) * 1000
            logger.warning(
                "mining.submit STALE (21): job_id=%s latency=%.1fms msg=%s",
                job_id, latency_ms, e,
            )
            self.stats.dropped_stale_jobid += 1
            return SubmitResult(
                accepted=False, latency_ms=latency_ms, error=str(e), error_code=21
            )
        except StratumProtocolError as e:
            latency_ms = (time.monotonic() - t0) * 1000
            logger.warning(
                "mining.submit REJECTED: job_id=%s latency=%.1fms code=%r msg=%s",
                job_id, latency_ms, getattr(e, "code", None), e,
            )
            self.stats.rejected += 1
            return SubmitResult(
                accepted=False, latency_ms=latency_ms, error=str(e),
                error_code=getattr(e, "code", None),
            )

    async def submit_plain_proof(
        self,
        job_id: str,
        plain_proof_b64: str,
        incomplete_header_hex: str,
    ) -> SubmitResult:
        """Send `submitPlainProof` with the `mining_job` envelope.

        Used when we have a full block (PoW satisfies network target), not just
        a pool share. The pool needs the structured envelope to forward to the
        Pearl network.
        """
        await self._wait_for_connection()
        mining_job: dict[str, Any] = {
            "job_id": job_id,
            "incomplete_header_bytes": incomplete_header_hex,
        }
        # Echo the pool-pushed mining params verbatim if we have them.
        if self.mining_params is not None:
            mining_job["mining_params"] = self.mining_params

        params = {"plain_proof": plain_proof_b64, "mining_job": mining_job}
        t0 = time.monotonic()
        try:
            await self._call("submitPlainProof", params)
            latency_ms = (time.monotonic() - t0) * 1000
            self.stats.accepted += 1
            self.stats.note_accept()
            return SubmitResult(accepted=True, latency_ms=latency_ms)
        except StaleShareError as e:
            latency_ms = (time.monotonic() - t0) * 1000
            self.stats.dropped_stale_jobid += 1
            return SubmitResult(
                accepted=False, latency_ms=latency_ms, error=str(e), error_code=21
            )
        except StratumProtocolError as e:
            latency_ms = (time.monotonic() - t0) * 1000
            self.stats.rejected += 1
            return SubmitResult(
                accepted=False, latency_ms=latency_ms, error=str(e),
                error_code=getattr(e, "code", None),
            )

    # ---- internals -------------------------------------------------------

    async def _open_and_handshake(self) -> None:
        logger.info("Connecting to stratum %s:%d as worker=%s", self.host, self.port, self.worker_name)
        self._reader, self._writer = await asyncio.open_connection(self.host, self.port)
        try:
            sock = self._writer.get_extra_info("socket")
            if sock is not None:
                import socket as _socket
                sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
                sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE, 1)
        except Exception:
            logger.debug("Could not set TCP_NODELAY/SO_KEEPALIVE on socket", exc_info=True)

        # alphapool v1.5+ sends an unsolicited `pearl.challenge` BEFORE the
        # client may send anything; solve & ack synchronously here so the
        # mining.configure that follows isn't dropped. See protocol memo
        # 58_pearl_challenge_protocol.md.
        await self._handle_initial_challenge_if_any()

        # Spawn the read-loop pump BEFORE handshake calls so _call's futures resolve.
        # We do this by structuring _read_loop so the handshake runs on top of it:
        # configure -> subscribe -> authorize all share the same pending-id machinery,
        # and the _read_loop task starts once handshake is complete. Therefore we
        # implement a small "drain-one" helper for the handshake phase.
        await self._handshake()
        self._connected_event.set()

    async def _handle_initial_challenge_if_any(self) -> None:
        """Drain ONE frame from the socket; if it's `pearl.challenge`, solve
        and reply synchronously and await `result:true`. If the first frame is
        anything else, dispatch it via the regular path (handshake will then
        run on top).

        Tolerates pools that don't issue a challenge (v1.4 or downgraded
        deploys) by short-circuiting on read timeout.
        """
        assert self._reader is not None
        try:
            line = await asyncio.wait_for(
                self._reader.readline(), timeout=INITIAL_CHALLENGE_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            # No unsolicited frame within 2 s -> assume legacy v1.4 / no challenge.
            logger.info(
                "No pearl.challenge from pool within %.1fs; proceeding to handshake",
                INITIAL_CHALLENGE_TIMEOUT_S,
            )
            return
        if not line:
            raise StratumProtocolError("Connection closed before any frame received")
        msg = self._parse_line(line)
        if msg is None:
            return
        method = msg.get("method")
        if method != PEARL_CHALLENGE_METHOD:
            # Not a challenge — dispatch normally (e.g. pearl.set_mining_params
            # arriving before subscribe; or a late response).
            self._dispatch_frame(msg)
            return
        await self._solve_and_respond_challenge(msg, blocking=True)

    async def _solve_and_respond_challenge(self, msg: dict, *, blocking: bool) -> None:
        """Run the PoW solver in a worker thread and send the response.

        When `blocking=True` (pre-handshake path), also wait for the pool's ack
        synchronously by reading the socket directly. When False (mid-session,
        from `_dispatch_frame`), we send via the regular write_lock path and
        let the read-loop deliver the ack into the future.
        """
        params = msg.get("params") or {}
        seed = params.get("seed")
        difficulty = params.get("difficulty")
        if not isinstance(seed, str) or not isinstance(difficulty, int):
            raise StratumProtocolError(
                f"Malformed pearl.challenge params: {params!r}"
            )
        logger.info("pearl.challenge: seed=%s... difficulty=%d (blocking=%s)",
                    seed[:8], difficulty, blocking)
        t0 = time.monotonic()
        nonce_hex = await asyncio.to_thread(_solve_pearl_challenge, seed, difficulty)
        solve_ms = (time.monotonic() - t0) * 1000
        logger.info("pearl.challenge solved: nonce=%s in %.1fms", nonce_hex, solve_ms)

        response_params = {"seed": seed, "nonce": nonce_hex}
        if blocking:
            # Send directly and read the ack inline (read-loop not yet running).
            rid = self._send_request_unlocked(
                PEARL_CHALLENGE_RESPONSE_METHOD, response_params
            )
            # Drain frames until our ack arrives.
            t1 = time.monotonic()
            while True:
                if time.monotonic() - t1 > HANDSHAKE_TIMEOUT_S:
                    raise StratumProtocolError(
                        "Timeout waiting for pearl.challenge_response ack"
                    )
                line = await asyncio.wait_for(
                    self._reader.readline(), timeout=HANDSHAKE_TIMEOUT_S
                )
                if not line:
                    raise StratumProtocolError(
                        "Pool closed connection awaiting pearl.challenge ack"
                    )
                ack = self._parse_line(line)
                if ack is None:
                    continue
                if ack.get("id") == rid:
                    if ack.get("error") is not None:
                        raise StratumProtocolError(
                            f"pearl.challenge_response rejected: {ack['error']!r}"
                        )
                    if ack.get("result") is not True:
                        raise StratumProtocolError(
                            f"pearl.challenge_response unexpected result: {ack.get('result')!r}"
                        )
                    self.stats.challenges_solved += 1
                    logger.info(
                        "pearl.challenge ACK ok: rid=%s total_solved=%d",
                        rid, self.stats.challenges_solved,
                    )
                    return
                # Some other frame arrived first; dispatch via the normal path
                # (e.g. a stray pearl.set_mining_params right after challenge).
                self._dispatch_frame(ack)
        else:
            # Mid-session: send via the same channel as `_call` so write_lock
            # serialization holds. We don't await the ack here because the
            # read-loop will route it through _complete_pending -> our future,
            # which we don't strictly need (pool never returns false), but we
            # still want the rid in `_pending` so the ack isn't logged as
            # "unknown rid". Use a fire-and-forget future.
            try:
                await self._call(
                    PEARL_CHALLENGE_RESPONSE_METHOD, response_params
                )
                self.stats.challenges_solved += 1
                logger.info(
                    "mid-session pearl.challenge ACK ok: total_solved=%d",
                    self.stats.challenges_solved,
                )
            except StratumProtocolError as e:
                logger.warning("mid-session pearl.challenge_response failed: %s", e)

    async def _handshake(self) -> None:
        """Run the configure/subscribe/authorize triad on a stopped-loop socket.

        Reads each response synchronously since `_read_loop` hasn't started yet.
        Any unsolicited frames (e.g. `pearl.set_mining_params`) that arrive
        between subscribe and authorize are dispatched inline.
        """
        configure_resp = await self._handshake_call(
            "mining.configure", [["pearl/v1"], {}]
        )
        if not isinstance(configure_resp, dict) or configure_resp.get("pearl/v1") is not True:
            raise StratumProtocolError(
                f"Pool did not accept pearl/v1: {configure_resp!r}"
            )

        subscribe_resp = await self._handshake_call(
            "mining.subscribe", [self.user_agent]
        )
        if not isinstance(subscribe_resp, list) or len(subscribe_resp) < 1:
            raise StratumProtocolError(f"Malformed subscribe response: {subscribe_resp!r}")

        authorize_resp = await self._handshake_call(
            "mining.authorize", [self.worker_name, self.password]
        )
        if authorize_resp is not True:
            raise StratumProtocolError(f"Authorize rejected: {authorize_resp!r}")

        logger.info(
            "Stratum handshake OK (worker=%s, params=%s)",
            self.worker_name,
            "received" if self.mining_params is not None else "none-yet",
        )

    async def _handshake_call(self, method: str, params: Any) -> Any:
        """Issue a request and read responses inline until our id is answered.

        Any non-matching responses (e.g. notifications, params push) are
        dispatched to handlers in-line.
        """
        rid = self._send_request_unlocked(method, params)
        t0 = time.monotonic()
        while True:
            if time.monotonic() - t0 > HANDSHAKE_TIMEOUT_S:
                raise StratumProtocolError(f"Handshake timeout waiting for {method}")
            line = await asyncio.wait_for(
                self._reader.readline(), timeout=HANDSHAKE_TIMEOUT_S
            )
            if not line:
                raise StratumProtocolError("Connection closed during handshake")
            msg = self._parse_line(line)
            if msg is None:
                continue
            if "id" in msg and msg["id"] == rid:
                if "error" in msg and msg["error"] is not None:
                    raise StratumProtocolError(
                        f"{method} returned error: {msg['error']!r}"
                    )
                return msg.get("result")
            # Out-of-order frame; dispatch via the same path the read-loop uses.
            self._dispatch_frame(msg)

    def _send_request_unlocked(self, method: str, params: Any) -> int:
        """Write a request directly to the socket (handshake-time only).

        Wire format note: alpha-miner / alphapool's pearl/v1 dialect uses
        **JSON-RPC 1.x style requests** (no `"jsonrpc": "2.0"` field on
        outgoing; pool responses DO carry it but that's their choice).
        Confirmed via tcpdump in pearl-investigation/STRATUM_CAPTURE.md §3a-d.
        Including `"jsonrpc": "2.0"` on requests causes the pool to drop the
        TCP connection with "connection reset by peer" mid-handshake.
        """
        rid = self._next_id
        self._next_id += 1
        frame = {"id": rid, "method": method, "params": params}
        payload = (json.dumps(frame) + "\n").encode("utf-8")
        assert self._writer is not None
        self._writer.write(payload)
        # WAVE-16 DBG: log every TX frame (handshake-time path).
        logger.info("TX[%d bytes] %s", len(payload), payload[:200].decode('utf-8', errors='replace').rstrip())
        # Flush is implicit on next await; for handshake we read immediately after.
        return rid

    async def _call(self, method: str, params: Any) -> Any:
        """Threadsafe-ish (within the loop) async request/response.

        Behavior on stale-job rejects:
          * `mining.submit` / `submitPlainProof` with error code 21:
            raise StaleShareError. Do NOT close the socket.
        Other JSON-RPC errors raise StratumProtocolError.
        """
        rid = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[rid] = _PendingRequest(future=fut, method=method, sent_at=time.monotonic())

        # JSON-RPC 1.x style on the wire (matches alpha-miner / pcap).
        # See _send_request_unlocked for the wire-format rationale.
        frame = {"id": rid, "method": method, "params": params}
        payload = (json.dumps(frame) + "\n").encode("utf-8")

        try:
            async with self._write_lock:
                if self._writer is None or self._writer.is_closing():
                    raise StratumProtocolError("Socket is closed")
                self._writer.write(payload)
                await self._writer.drain()
            # WAVE-16 DBG: log every TX frame (post-handshake path).
            logger.info("TX[%d bytes] %s", len(payload), payload[:200].decode('utf-8', errors='replace').rstrip())

            return await asyncio.wait_for(fut, timeout=SUBMIT_TIMEOUT_S)
        finally:
            self._pending.pop(rid, None)

    async def _read_loop(self) -> None:
        assert self._reader is not None
        logger.info("WAVE-16 _read_loop ENTERED (handshake handed over; now monitoring socket for unsolicited frames)")
        while not self._stop.is_set():
            line = await self._reader.readline()
            if not line:
                raise StratumProtocolError("Pool closed the connection (EOF)")
            # WAVE-16 DBG: log every raw RX frame to disambiguate "pool sent
            # nothing" from "we silently dropped a frame". Truncate body.
            logger.info("RX[%d bytes] %s", len(line), line[:200].decode('utf-8', errors='replace').rstrip())
            msg = self._parse_line(line)
            if msg is None:
                continue
            self._dispatch_frame(msg)

    def _parse_line(self, line: bytes) -> dict | None:
        try:
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                return None
            obj = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Pool sent non-JSON line (ignored): %r", line[:200])
            return None
        if not isinstance(obj, dict):
            logger.warning("Pool sent non-object frame (ignored): %r", obj)
            return None
        return obj

    def _dispatch_frame(self, msg: dict) -> None:
        """Route a parsed JSON-RPC frame to either a pending response or a notification handler."""
        # Response to a request we sent: has `id`, no `method`.
        rid = msg.get("id")
        method = msg.get("method")
        if rid is not None and method is None:
            self._complete_pending(rid, msg)
            return

        # Notification or server-initiated request.
        if method == "mining.notify":
            self._handle_notify(msg.get("params") or [])
        elif method == "mining.set_difficulty":
            self._handle_set_difficulty(msg.get("params") or [])
        elif method == "pearl.set_mining_params":
            self._handle_set_mining_params(msg.get("params") or [])
        elif method == PEARL_CHALLENGE_METHOD:
            # Mid-session DDoS pacer. Spawn solver+responder as a task so we
            # don't block other frame dispatch (in particular, mining.submit
            # callers waiting on _call futures must keep getting their acks).
            asyncio.create_task(
                self._solve_and_respond_challenge(msg, blocking=False)
            )
        elif method == "mining.set_extranonce":
            # Pearl/v1 has no client-side nonce-rolling, but we accept and log
            # this for safety since the pool may push it.
            logger.info("mining.set_extranonce (ignored, pearl/v1): %r", msg.get("params"))
        elif method == "client.reconnect":
            # Pool requested we reconnect. Drop the socket cleanly; our run-loop
            # will reconnect.
            logger.info("Pool sent client.reconnect; dropping session")
            if self._writer is not None and not self._writer.is_closing():
                self._writer.close()
        elif method == "client.show_message":
            params = msg.get("params") or []
            text = params[0] if params else ""
            logger.info("Pool message: %s", text)
        else:
            logger.debug("Unhandled stratum frame: method=%r", method)

    def _complete_pending(self, rid: Any, msg: dict) -> None:
        pending = self._pending.get(rid)
        if pending is None:
            logger.debug("Response to unknown rid=%r (likely a late stale)", rid)
            return

        # WAVE-7 instrumentation: when responding to a mining.submit, dump
        # the raw frame so the post-mortem has the exact bytes the pool sent.
        # Useful for diagnosing silent rejection / nonstandard error codes.
        if pending.method == "mining.submit":
            try:
                logger.info("mining.submit response (rid=%s): %s", rid, json.dumps(msg)[:512])
            except Exception:
                logger.info("mining.submit response (rid=%s): %r", rid, msg)

        err = msg.get("error")
        if err is not None:
            # Stratum error tuples are [code, message, traceback].
            code: int | None = None
            message: str = repr(err)
            if isinstance(err, list) and len(err) >= 2:
                code = err[0] if isinstance(err[0], int) else None
                message = str(err[1])
            elif isinstance(err, dict):
                code = err.get("code")
                message = str(err.get("message", err))

            if code == 21:
                # CRITICAL: do not close socket. alpha-miner v1.4.0 bug fixed here.
                exc: Exception = StaleShareError(message)
            else:
                proto_err = StratumProtocolError(message)
                proto_err.code = code  # type: ignore[attr-defined]
                exc = proto_err
            if not pending.future.done():
                pending.future.set_exception(exc)
            return

        result = msg.get("result")
        if not pending.future.done():
            pending.future.set_result(result)

    def _handle_notify(self, params: list[Any]) -> None:
        try:
            job = parse_notify(params)
        except ValueError as e:
            logger.warning("Bad mining.notify (ignored): %s; params=%r", e, params)
            return
        self.current_job = job
        if self.on_new_job is not None:
            try:
                self.on_new_job(job)
            except Exception:
                logger.exception("on_new_job callback raised")

    def _handle_set_difficulty(self, params: list[Any]) -> None:
        if not params or not isinstance(params[0], (int, float)):
            logger.warning("Bad mining.set_difficulty (ignored): %r", params)
            return
        diff = float(params[0])
        self.stats.last_diff = diff
        if self.on_set_difficulty is not None:
            try:
                self.on_set_difficulty(diff)
            except Exception:
                logger.exception("on_set_difficulty callback raised")

    def _handle_set_mining_params(self, params: list[Any]) -> None:
        # Pearl form: params is a single-element list of one dict (per STRATUM_CAPTURE §3c).
        if not params or not isinstance(params[0], dict):
            logger.warning("Bad pearl.set_mining_params (ignored): %r", params)
            return
        self.mining_params = params[0]
        logger.info(
            "pearl.set_mining_params: m=%s n=%s k=%s rank=%s mma=%s",
            self.mining_params.get("m"),
            self.mining_params.get("n"),
            self.mining_params.get("k"),
            self.mining_params.get("rank"),
            self.mining_params.get("mma_type"),
        )
        if self.on_set_mining_params is not None:
            try:
                self.on_set_mining_params(self.mining_params)
            except Exception:
                logger.exception("on_set_mining_params callback raised")

    def _fail_pending(self, reason: str) -> None:
        """Cancel all in-flight requests after a socket drop."""
        for pending in list(self._pending.values()):
            if not pending.future.done():
                pending.future.set_exception(StratumProtocolError(f"{pending.method}: {reason}"))
        self._pending.clear()

    async def _wait_for_connection(self) -> None:
        await self._connected_event.wait()

    async def _close_socket(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None


# ---- Helpers ---------------------------------------------------------------


def default_worker_name() -> str:
    """`$HOSTNAME` or fall back to a sentinel. Used by the CLI."""
    return os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME") or "unknown-worker"
