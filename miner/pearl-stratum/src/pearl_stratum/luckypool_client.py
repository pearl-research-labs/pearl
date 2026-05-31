"""LuckyPool / lpminer stratum dialect adapter.

LuckyPool (pearl-ca1.luckypool.io:3360, served by `lpminer` 0.1.9) speaks a
DIFFERENT stratum dialect than alphapool.tech. The alphapool client lives in
`stratum_client.py` and uses **positional** JSON-RPC params; LuckyPool uses
**object (named)** params. Rather than fork the existing client, this module
provides the LuckyPool dialect as a small standalone client that shares the
same JSON-RPC framing (newline-delimited JSON-RPC 1.x over TCP).

Wire format (extracted from the lpminer 0.1.9 binary strings,
`C:/Source/_lpminer_re/lpminer.strings.txt:8544-8653`; see report
`re_2026_05_30/reports/07_integration_gap.md` §"Stratum gap"):

    C -> mining.subscribe   {agent}                            (object)
    C -> mining.authorize   {wallet, worker, agent}            (object)
    S -> true
    S -> mining.notify      {job_id, header, target, height}   (object, push)
    C -> mining.submit      {job_id, plain_proof, hs}          (object, per share)
    S -> {result: true}                                         (accept)
    S -> {error: [code, msg, null]}                             (reject)

Notes vs alphapool:
  * NO `mining.configure` / `pearl/v1` capability negotiation.
  * NO unsolicited `pearl.set_mining_params` — the mining geometry is fixed /
    derived (the miner reads it from the calibrated `MiningConfiguration` and
    the `header`). LuckyPool does not push m/n/k/rank.
  * NO `pearl.challenge` DDoS pacer.
  * `mining.notify.header` is the 76-byte incomplete block header (hex). The
    header already embeds `nbits` (the *block* difficulty). `nbits` is read
    out of the header by `IncompleteBlockHeader.from_bytes`.
  * `mining.notify.target` is the *share* threshold sent DIRECTLY (not nbits).
    Endianness: lpminer sends it as a hex string of the 256-bit target; we parse
    it big-endian (most-significant first, the human-readable form) by default.
    If a live capture shows little-endian, flip `TARGET_HEX_BIG_ENDIAN`.

    IMPORTANT — this raw wire target is NOT the on-device comparison threshold.
    The verifier (`zk-pow/src/api/sanity_checks.rs::extract_difficulty_bound`,
    enforced by `verify_plain_proof`) accepts a share iff
        int.from_bytes(jackpot_hash, "little") <= target * (h*w*k)
    where the difficulty_adjustment_factor `h*w*k` = rows_pattern.size (h=8) *
    cols_pattern.size (w=16) * dot_product_length (k - k%rank). The GPU binary
    `pearl_miner_sm89.cu` applies this `* h*w*k` multiply when it converts the
    wire target to the 32 LE `pow_target` words; this module therefore passes the
    raw share target through unchanged. (Confirmed against the live diff=262144
    job: wire_target = 2^206, target*difficulty = 0xffff<<208 = diff1, and
    wire_target*h*w*k = 2^225 — the empirically-correct ~2^225..2^232 threshold.)
  * `mining.submit.hs` is the claimed hashrate (float, hashes/s) — telemetry
    only; the pool does not gate on it.

This module is intentionally dependency-light (stdlib + asyncio) so it can be
unit-tested offline with a mock server. The proof bytes it submits are produced
elsewhere (the GPU driver + `pearl_mining.PlainProof.to_base64`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket as _socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Default LuckyPool Pearl endpoint (OVH Quebec; no US endpoint exists).
DEFAULT_POOL_HOST = "pearl-ca1.luckypool.io"
DEFAULT_POOL_PORT = 3360

# lpminer sends the notify `target` as a hex string of the 256-bit threshold.
# We treat it as big-endian (human-readable, MSB-first) by default. Flip to
# False only if a live capture proves little-endian on the wire.
TARGET_HEX_BIG_ENDIAN = True

HANDSHAKE_TIMEOUT_S = 30.0
SUBMIT_TIMEOUT_S = 30.0

# A `mining.submit` frame carries the full plain_proof base64 (~137 KB for the
# production 131072² job). asyncio's default StreamReader line limit is 64 KB,
# which truncates the *response* read if the pool ever echoes the proof, and
# more importantly bounds our own readline on large server pushes. Raise it
# generously so no single JSON-RPC line is ever clipped.
STREAM_LIMIT_BYTES = 4 * 1024 * 1024

RECONNECT_BACKOFF = [1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
CIRCUIT_BREAKER_FAILURES = 10


# ---------------------------------------------------------------------------
# Job model + notify parsing
# ---------------------------------------------------------------------------


def parse_target_hex(target_hex: str, big_endian: bool = TARGET_HEX_BIG_ENDIAN) -> int:
    """Parse the LuckyPool `target` hex string into a 256-bit int.

    Accepts an optional `0x` prefix. The hex digits encode a 256-bit threshold;
    `big_endian=True` reads them MSB-first (the standard human-readable form).
    """
    s = target_hex.strip()
    if s.lower().startswith("0x"):
        s = s[2:]
    if not s:
        raise ValueError("empty target hex")
    try:
        raw = bytes.fromhex(s.zfill(len(s) + (len(s) & 1)))
    except ValueError as e:
        raise ValueError(f"target is not valid hex: {e}") from e
    return int.from_bytes(raw, "big" if big_endian else "little")


def target_int_to_le_bytes(target: int) -> bytes:
    """32-byte little-endian target — the kernel `pow_target` input form."""
    target &= (1 << 256) - 1
    return target.to_bytes(32, "little")


@dataclass
class LuckyPoolJob:
    """A parsed LuckyPool `mining.notify`."""

    job_id: str
    header_bytes: bytes
    """Hex-decoded 76-byte incomplete block header (the `header` field)."""

    target: int
    """Full 256-bit share target (sent directly by the pool, NOT nbits)."""

    target_le: bytes
    """32-byte little-endian `target` — suitable for the kernel `pow_target`."""

    height: int | None
    """Block height (telemetry / staleness). Optional — some jobs omit it."""

    received_at: float
    raw_params: dict[str, Any] = field(default_factory=dict)


def parse_luckypool_notify(params: Any) -> LuckyPoolJob:
    """Parse LuckyPool `mining.notify` OBJECT params into a LuckyPoolJob.

    Expected shape: ``{"job_id", "header", "target", "height"}``. Raises
    ValueError on malformed input. Tolerates `height` absence.
    """
    if not isinstance(params, dict):
        raise ValueError(
            f"LuckyPool mining.notify params must be an object, got {type(params).__name__}"
        )

    job_id = params.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError(f"job_id must be a non-empty string, got {job_id!r}")

    header_hex = params.get("header")
    if not isinstance(header_hex, str):
        raise ValueError(f"header must be a hex string, got {type(header_hex).__name__}")
    try:
        header_bytes = bytes.fromhex(header_hex)
    except ValueError as e:
        raise ValueError(f"header is not valid hex: {e}") from e
    if len(header_bytes) != 76:
        raise ValueError(f"header must decode to 76 bytes, got {len(header_bytes)}")

    target_field = params.get("target")
    if isinstance(target_field, str):
        target = parse_target_hex(target_field)
    elif isinstance(target_field, int):
        target = target_field
    else:
        raise ValueError(f"target must be hex string or int, got {type(target_field).__name__}")
    if not (0 < target < (1 << 256)):
        raise ValueError(f"target out of 256-bit range: {target:#x}")

    height_field = params.get("height")
    height = int(height_field) if isinstance(height_field, (int, float)) else None

    return LuckyPoolJob(
        job_id=job_id,
        header_bytes=header_bytes,
        target=target,
        target_le=target_int_to_le_bytes(target),
        height=height,
        received_at=time.time(),
        raw_params=dict(params),
    )


def build_submit_params(job_id: str, plain_proof_b64: str, hashrate: float) -> dict[str, Any]:
    """Format `mining.submit` OBJECT params for LuckyPool.

    ``{"job_id", "plain_proof", "hs"}`` — `plain_proof` is the base64 of the
    proof.bin layout (`PlainProof.to_base64()`), `hs` is the claimed hashrate.
    """
    return {"job_id": job_id, "plain_proof": plain_proof_b64, "hs": float(hashrate)}


# ---------------------------------------------------------------------------
# Result / stats
# ---------------------------------------------------------------------------


@dataclass
class SubmitResult:
    accepted: bool
    latency_ms: float
    error: str | None = None
    error_code: int | None = None


@dataclass
class LuckyPoolStats:
    accepted: int = 0
    rejected: int = 0
    last_target: int = 0


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


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


class LuckyPoolStratumError(Exception):
    """Malformed pool message, handshake failure, or share reject."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


class LuckyPoolStratumClient:
    """Persistent LuckyPool stratum connection (object-param dialect).

    Single asyncio event loop. `run()` is the reconnect loop; `submit_share`
    blocks on the pool's response. New jobs are delivered to `on_new_job`.
    """

    def __init__(
        self,
        host: str = DEFAULT_POOL_HOST,
        port: int = DEFAULT_POOL_PORT,
        *,
        wallet: str,
        worker: str,
        agent: str = "lpminer/0.1.9-552bdfe",
        on_new_job: Callable[[LuckyPoolJob], None] | None = None,
        on_disconnect: Callable[[str], None] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.wallet = wallet
        self.worker = worker
        self.agent = agent
        self.on_new_job = on_new_job
        self.on_disconnect = on_disconnect

        self.stats = LuckyPoolStats()
        self.current_job: LuckyPoolJob | None = None

        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._write_lock = asyncio.Lock()
        self._connected_event = asyncio.Event()
        self._stop = asyncio.Event()
        self._consecutive_failures = 0

    # ---- public API ------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected_event.is_set()

    async def stop(self) -> None:
        self._stop.set()
        await self._close_socket()

    async def run(self) -> int:
        """Reconnect loop. Returns nonzero on circuit-breaker trip."""
        while not self._stop.is_set():
            try:
                await self._open_and_handshake()
                self._consecutive_failures = 0
                await self._read_loop()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("LuckyPool session error: %s", e)
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
                    "LuckyPool circuit breaker tripped after %d failures, exiting",
                    self._consecutive_failures,
                )
                return 2
            idx = min(self._consecutive_failures - 1, len(RECONNECT_BACKOFF) - 1)
            backoff = RECONNECT_BACKOFF[idx]
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
        hashrate: float = 0.0,
    ) -> SubmitResult:
        """Send `mining.submit` with OBJECT params `{job_id, plain_proof, hs}`."""
        await self._connected_event.wait()
        params = build_submit_params(job_id, plain_proof_b64, hashrate)
        t0 = time.monotonic()
        try:
            raw_result = await self._call("mining.submit", params)
            latency_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "mining.submit OK: job_id=%s latency=%.1fms raw_result=%r proof_b64_len=%d",
                job_id, latency_ms, raw_result, len(plain_proof_b64),
            )
            self.stats.accepted += 1
            return SubmitResult(accepted=True, latency_ms=latency_ms)
        except LuckyPoolStratumError as e:
            latency_ms = (time.monotonic() - t0) * 1000
            logger.warning(
                "mining.submit REJECTED: job_id=%s latency=%.1fms code=%r msg=%s",
                job_id, latency_ms, e.code, e,
            )
            self.stats.rejected += 1
            return SubmitResult(
                accepted=False, latency_ms=latency_ms, error=str(e), error_code=e.code
            )

    # ---- internals -------------------------------------------------------

    async def _open_and_handshake(self) -> None:
        logger.info("Connecting to LuckyPool %s:%d wallet=%s worker=%s",
                    self.host, self.port, self.wallet, self.worker)
        self._reader, self._writer = await asyncio.open_connection(
            self.host, self.port, limit=STREAM_LIMIT_BYTES
        )
        try:
            sock = self._writer.get_extra_info("socket")
            if sock is not None:
                sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
                sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE, 1)
        except Exception:
            logger.debug("Could not set TCP_NODELAY/SO_KEEPALIVE", exc_info=True)

        # Captured from lpminer 0.1.9 (strace, 2026-05-31): NO mining.subscribe;
        # authorize is the FIRST message (id:1). The `wallet` field carries the
        # worker appended after a dot, alongside a separate `worker` field.
        auth_wallet = (
            self.wallet
            if self.wallet.endswith(f".{self.worker}")
            else f"{self.wallet}.{self.worker}"
        )
        auth = await self._handshake_call(
            "mining.authorize",
            {"wallet": auth_wallet, "worker": self.worker, "agent": self.agent},
        )
        if auth is not True:
            raise LuckyPoolStratumError(f"authorize rejected: {auth!r}")
        logger.info("LuckyPool handshake OK (wallet=%s worker=%s)", self.wallet, self.worker)
        self._connected_event.set()

    async def _handshake_call(self, method: str, params: Any) -> Any:
        rid = self._send_request_unlocked(method, params)
        t0 = time.monotonic()
        assert self._reader is not None
        while True:
            if time.monotonic() - t0 > HANDSHAKE_TIMEOUT_S:
                raise LuckyPoolStratumError(f"handshake timeout waiting for {method}")
            line = await asyncio.wait_for(self._reader.readline(), timeout=HANDSHAKE_TIMEOUT_S)
            if not line:
                raise LuckyPoolStratumError("connection closed during handshake")
            msg = self._parse_line(line)
            if msg is None:
                continue
            if "id" in msg and msg["id"] == rid:
                err = msg.get("error")
                if err is not None:
                    raise LuckyPoolStratumError(f"{method} error: {err!r}")
                return msg.get("result")
            # Out-of-order push (e.g. an early mining.notify) — dispatch it.
            self._dispatch_frame(msg)

    def _send_request_unlocked(self, method: str, params: Any) -> int:
        rid = self._next_id
        self._next_id += 1
        frame = {"id": rid, "method": method, "params": params}
        payload = (json.dumps(frame) + "\n").encode("utf-8")
        assert self._writer is not None
        self._writer.write(payload)
        logger.debug("TX %s", payload[:200].decode("utf-8", errors="replace").rstrip())
        return rid

    async def _call(self, method: str, params: Any) -> Any:
        rid = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[rid] = fut
        frame = {"id": rid, "method": method, "params": params}
        payload = (json.dumps(frame) + "\n").encode("utf-8")
        try:
            async with self._write_lock:
                if self._writer is None or self._writer.is_closing():
                    raise LuckyPoolStratumError("socket is closed")
                self._writer.write(payload)
                await self._writer.drain()
            logger.debug("TX %s", payload[:200].decode("utf-8", errors="replace").rstrip())
            return await asyncio.wait_for(fut, timeout=SUBMIT_TIMEOUT_S)
        finally:
            self._pending.pop(rid, None)

    async def _read_loop(self) -> None:
        assert self._reader is not None
        while not self._stop.is_set():
            line = await self._reader.readline()
            if not line:
                raise LuckyPoolStratumError("pool closed the connection (EOF)")
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
            logger.warning("non-JSON line (ignored): %r", line[:200])
            return None
        if not isinstance(obj, dict):
            logger.warning("non-object frame (ignored): %r", obj)
            return None
        return obj

    def _dispatch_frame(self, msg: dict) -> None:
        rid = msg.get("id")
        method = msg.get("method")
        if rid is not None and method is None:
            self._complete_pending(rid, msg)
            return
        if method == "mining.notify":
            self._handle_notify(msg.get("params"))
        elif method == "mining.set_target":
            # Some lpminer deployments push target separately; fold it into the
            # current job if present.
            self._handle_set_target(msg.get("params"))
        elif method == "client.reconnect":
            logger.info("Pool sent client.reconnect; dropping session")
            if self._writer is not None and not self._writer.is_closing():
                self._writer.close()
        else:
            logger.debug("Unhandled LuckyPool frame: method=%r", method)

    def _complete_pending(self, rid: Any, msg: dict) -> None:
        fut = self._pending.get(rid)
        if fut is None:
            logger.debug("response to unknown rid=%r", rid)
            return
        err = msg.get("error")
        if err is not None:
            code: int | None = None
            message = repr(err)
            if isinstance(err, list) and len(err) >= 2:
                code = err[0] if isinstance(err[0], int) else None
                message = str(err[1])
            elif isinstance(err, dict):
                code = err.get("code")
                message = str(err.get("message", err))
            if not fut.done():
                fut.set_exception(LuckyPoolStratumError(message, code=code))
            return
        if not fut.done():
            fut.set_result(msg.get("result"))

    def _handle_notify(self, params: Any) -> None:
        try:
            job = parse_luckypool_notify(params)
        except ValueError as e:
            logger.warning("bad mining.notify (ignored): %s; params=%r", e, params)
            return
        self.current_job = job
        self.stats.last_target = job.target
        if self.on_new_job is not None:
            try:
                self.on_new_job(job)
            except Exception:
                logger.exception("on_new_job callback raised")

    def _handle_set_target(self, params: Any) -> None:
        if not isinstance(params, dict):
            return
        tgt = params.get("target")
        if tgt is None or self.current_job is None:
            return
        try:
            t = parse_target_hex(tgt) if isinstance(tgt, str) else int(tgt)
        except ValueError:
            return
        self.current_job.target = t
        self.current_job.target_le = target_int_to_le_bytes(t)
        self.stats.last_target = t

    def _fail_pending(self, reason: str) -> None:
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(LuckyPoolStratumError(reason))
        self._pending.clear()

    async def _close_socket(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None
