"""Asyncio Pearlhash pool client.

Talks the proprietary Pearlhash wire protocol (host 84.32.220.219:9000 in
production) using the cipher RE'd in `51_pearlhash_cipher_re.md`. Wire format:

    on_wire = ascii_hex(inner) + '\\n'
    inner = htonl(rand_N) || XOR(plaintext, KEY_MSG[N])

Where `rand_N` is the Nth output of glibc's `rand()` after `srand(0)` (called
once at TCP connection start). `KEY_MSG[N]` is a 48-byte periodic keystream
recovered by known-plaintext attack — currently 16 keystreams (N=0..15) are
known; submit-frame (N>=16) keystreams need one paired capture each.

This is a **partial client**: it can perform login + 15 keepalive frames, then
exhausts its keystream table. Reconnection (which resets N to 0) lets it keep
going. Share submission raises `NotImplementedError` until the submit keystream
is recovered (memo 51 §4.1, the "1 paired capture" follow-up).

Design parallels `stratum_client.StratumClient` so the same `AsyncLoopManager`
dispatcher can drive either one. The `MiningClient`-shaped surface is:

    * `run()`     — main reconnect loop; spawn as a task and forget.
    * `submit_share(...)` — currently raises; placeholder for after recapture.
    * `stats`     — `PearlhashStats` dataclass; mirrors `StratumStats`.
    * `connected` — asyncio.Event-backed property.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .pearlhash_cipher import KeyNotKnownError, decrypt, encrypt
from .pearlhash_framing import decode_frame, encode_frame
from .pearlhash_keys import COUNTER_TO_FRAME_INDEX, MAX_KNOWN_FRAME_INDEX
from .pearlhash_rand import GlibcRand

logger = logging.getLogger(__name__)


# Reconnect backoff (seconds); cap at 60.
RECONNECT_BACKOFF = [1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
CIRCUIT_BREAKER_FAILURES = 10

# Pearl-miner v2 emits a `report_info` keepalive every 5 seconds (memo 36 §1.5).
KEEPALIVE_INTERVAL_S = 5.0

# Protocol-level connect/login timeout.
LOGIN_TIMEOUT_S = 30.0

# `--user` doesn't accept dotted worker names (memo 36 §1.6), and the
# version field is observed as the literal string "0.5".
PEARLHASH_CLIENT_VERSION = "0.5"


class PearlhashProtocolError(Exception):
    """Generic protocol failure (malformed frame, unexpected EOF, etc.)."""


class SubmitNotYetSupportedError(NotImplementedError):
    """Submit keystream not yet recovered. See PEARLHASH_README.md."""


@dataclass
class PearlhashStats:
    frames_sent: int = 0
    frames_received: int = 0
    frames_dropped_unknown_key: int = 0
    """Frames where the frame index exceeded `MAX_KNOWN_FRAME_INDEX`."""
    frames_dropped_unparsable: int = 0
    last_hashrate_reported: float = 0.0
    connect_count: int = 0
    last_connect_at: float = 0.0


@dataclass
class _OutboundFrame:
    frame_index: int
    plaintext: bytes


@dataclass
class _DecodedFrame:
    frame_index: int
    plaintext: bytes


class PearlhashClient:
    """Persistent Pearlhash pool connection.

    Single asyncio event loop. Mirrors `StratumClient` lifecycle:
      * construct
      * `await client.run()` as a task
      * call `await client.stop()` to drain

    Hashrate reporting is push-based: the caller is expected to update
    `current_hashrate` (e.g. via `set_hashrate()`) and the keepalive coroutine
    snapshots that value on each tick.
    """

    def __init__(
        self,
        host: str,
        port: int,
        wallet: str,
        *,
        gpu_name: str = "NVIDIA GeForce RTX 4070 Ti SUPER",
        password: str = "",
        version: str = PEARLHASH_CLIENT_VERSION,
        on_message: Callable[[str, Any], None] | None = None,
        on_disconnect: Callable[[str], None] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.wallet = wallet
        self.gpu_name = gpu_name
        self.password = password
        self.version = version

        self.on_message = on_message
        self.on_disconnect = on_disconnect

        self.stats = PearlhashStats()
        self.current_hashrate: float = 0.0

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._write_lock = asyncio.Lock()
        self._connected_event = asyncio.Event()
        self._stop = asyncio.Event()

        self._rand: GlibcRand | None = None
        self._out_index = 0  # next outbound frame index N (0 = login)
        self._in_count = 0   # number of S->C frames received this connection

        self._keepalive_task: asyncio.Task | None = None
        self._consecutive_failures = 0

    # ---- public API ------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected_event.is_set()

    def set_hashrate(self, hashrate: float) -> None:
        """Update the value reported in subsequent keepalive frames (hashes/sec)."""
        self.current_hashrate = float(hashrate)

    async def stop(self) -> None:
        self._stop.set()
        if self._keepalive_task is not None and not self._keepalive_task.done():
            self._keepalive_task.cancel()
        await self._close_socket()

    async def run(self) -> int:
        """Reconnect loop. Returns nonzero when the circuit breaker trips."""
        while not self._stop.is_set():
            try:
                await self._open_and_login()
                self._consecutive_failures = 0
                self._keepalive_task = asyncio.create_task(self._send_keepalive_loop())
                await self._receive_loop()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("Pearlhash session error: %s", e)
                if self.on_disconnect is not None:
                    try:
                        self.on_disconnect(str(e))
                    except Exception:
                        logger.exception("on_disconnect callback raised")
            finally:
                if self._keepalive_task is not None and not self._keepalive_task.done():
                    self._keepalive_task.cancel()
                    try:
                        await self._keepalive_task
                    except (asyncio.CancelledError, Exception):
                        pass
                self._keepalive_task = None
                await self._close_socket()
                self._connected_event.clear()

            if self._stop.is_set():
                break

            self._consecutive_failures += 1
            if self._consecutive_failures >= CIRCUIT_BREAKER_FAILURES:
                logger.error(
                    "Pearlhash circuit breaker tripped after %d consecutive failures",
                    self._consecutive_failures,
                )
                return 2

            backoff_idx = min(self._consecutive_failures - 1, len(RECONNECT_BACKOFF) - 1)
            backoff = RECONNECT_BACKOFF[backoff_idx]
            logger.info("Reconnecting Pearlhash in %.1fs (fail #%d)",
                        backoff, self._consecutive_failures)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass

        return 0

    async def submit_share(self, *args: Any, **kwargs: Any) -> Any:
        """Submit a share. NOT YET IMPLEMENTED.

        The submit-frame keystream (KEY_MSG[16] and beyond) has not been
        recovered. Recovery requires one paired capture of pearl-miner v2
        emitting a submit frame, paired with the in-memory plaintext template
        (see `51_pearlhash_cipher_re.md` §4.1).
        """
        raise SubmitNotYetSupportedError(
            "submit keystream not yet recovered — capture needed; "
            "see 51_pearlhash_cipher_re.md §4.1"
        )

    # ---- connect / login ------------------------------------------------

    async def _open_and_login(self) -> None:
        logger.info("Connecting to Pearlhash %s:%d", self.host, self.port)
        self._reader, self._writer = await asyncio.open_connection(self.host, self.port)
        try:
            sock = self._writer.get_extra_info("socket")
            if sock is not None:
                import socket as _socket
                sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
                sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE, 1)
        except Exception:
            logger.debug("Could not set TCP_NODELAY/SO_KEEPALIVE", exc_info=True)

        # Reset cipher state for the new connection: srand(0) per memo 36 §1.3.
        self._rand = GlibcRand(seed=0)
        self._out_index = 0
        self._in_count = 0

        # Login frame: msg index 0, fixed JSON template.
        login_pt = self._build_login_plaintext().encode("ascii")
        await self._send_encrypted_frame(login_pt)
        self.stats.connect_count += 1
        self.stats.last_connect_at = time.time()

        # Login response is multi-frame (login resp A 37B, B 67B, C 525B per
        # memo 36 §1.5). S->C keystreams are NOT recovered, so we cannot decode
        # these — we just observe their arrival to confirm the pool accepted us.
        # We give the pool LOGIN_TIMEOUT_S to send *something*. Receipt of any
        # S->C frame within the window is treated as "login accepted".
        try:
            await asyncio.wait_for(self._await_first_inbound(), timeout=LOGIN_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise PearlhashProtocolError(
                f"no response from pool within {LOGIN_TIMEOUT_S}s after login"
            ) from None

        logger.info("Pearlhash login accepted (wallet=%s)", self.wallet)
        self._connected_event.set()

    def _build_login_plaintext(self) -> str:
        """Construct the login JSON-RPC frame body.

        Format observed in cap_long.pcap (memo 51 §3):
            {"id":0,"method":"login","params":["<wallet>","<password>","<version>"]}

        We must NOT include `jsonrpc:"2.0"` (matches the alphapool dialect note;
        pearl-miner v2 also omits it) and there must be no whitespace inside
        the JSON (json.dumps default with separators=(",", ":")).
        """
        msg = {
            "id": 0,
            "method": "login",
            "params": [self.wallet, self.password, self.version],
        }
        return json.dumps(msg, separators=(",", ":"))

    async def _await_first_inbound(self) -> None:
        """Wait for the first byte of pool response after login.

        Doesn't decode — S->C keystreams are unknown. Just confirms the pool
        didn't FIN,ACK us (which is its rejection mode per memo 36 §1.6).
        """
        assert self._reader is not None
        line = await self._reader.readline()
        if not line:
            raise PearlhashProtocolError(
                "pool closed connection immediately after login "
                "(likely wallet rejected; see memo 36 §1.6)"
            )
        # Record the frame for stats but don't try to decode (S->C keystreams TBD).
        self._in_count += 1
        self.stats.frames_received += 1

    # ---- frame I/O ------------------------------------------------------

    async def _send_encrypted_frame(self, plaintext: bytes) -> None:
        """Construct + send one C->S frame using the current frame index.

        Counter = htonl(rand_N), body = XOR(plaintext, KEY_MSG[N]), wire =
        ascii_hex(counter||body) + '\\n'.

        Drops (logs + returns) if the keystream for the current frame index is
        unknown; the connection stays open so future reconnects retry from N=0.
        """
        assert self._writer is not None
        assert self._rand is not None
        n = self._out_index
        # Consume one rand even if we're going to drop the frame — keeps the
        # counter sequence aligned for any later recovered keys at higher N.
        counter = self._rand.next_u31()
        self._out_index += 1

        try:
            body = encrypt(plaintext, n)
        except KeyNotKnownError:
            logger.warning(
                "keystream not available for frame index %d, dropping frame "
                "(%d bytes plaintext); consider reconnecting to reset index",
                n, len(plaintext),
            )
            self.stats.frames_dropped_unknown_key += 1
            return

        # Counter is a 32-bit BE quantity (htonl). glibc rand returns 31 bits;
        # the high bit is always 0. struct.pack(">I", ...) gives the right form.
        inner = struct.pack(">I", counter & 0xFFFFFFFF) + body
        wire = encode_frame(inner)

        async with self._write_lock:
            if self._writer is None or self._writer.is_closing():
                raise PearlhashProtocolError("socket closed before frame send")
            self._writer.write(wire)
            await self._writer.drain()
        self.stats.frames_sent += 1
        logger.debug("pearlhash send N=%d ctr=0x%08x inner_len=%d",
                     n, counter, len(inner))

    async def _receive_loop(self) -> None:
        """Read frames until EOF. Decoding requires S->C keystreams (not recovered)."""
        assert self._reader is not None
        while not self._stop.is_set():
            line = await self._reader.readline()
            if not line:
                raise PearlhashProtocolError("pool closed connection (EOF)")
            try:
                inner = decode_frame(line)
            except ValueError as e:
                logger.warning("malformed Pearlhash wire line: %s", e)
                self.stats.frames_dropped_unparsable += 1
                continue
            self._in_count += 1
            self.stats.frames_received += 1
            self._on_inbound_frame(inner)

    def _on_inbound_frame(self, inner: bytes) -> None:
        """Hand off a decoded inner-bytes frame to the on_message callback.

        We currently DON'T try to decrypt — S->C keystreams are unknown. Each
        inbound frame surfaces as a raw ciphertext blob and the caller (or
        future logic) can attempt known-plaintext recovery offline.
        """
        if self.on_message is not None:
            try:
                self.on_message("inbound_raw", inner)
            except Exception:
                logger.exception("on_message callback raised")

    # ---- keepalive ------------------------------------------------------

    async def _send_keepalive_loop(self) -> None:
        """Emit a `report_info` frame every 5s with the current hashrate.

        Frame index advances each tick. Once index passes
        `MAX_KNOWN_FRAME_INDEX` (15 currently) every send drops; we keep the
        loop running so the timing pattern matches pearl-miner v2 (visible
        on the wire even when payloads are missing), and so production users
        notice the dropped-frame counter climbing.
        """
        try:
            while not self._stop.is_set():
                await asyncio.sleep(KEEPALIVE_INTERVAL_S)
                if self._stop.is_set():
                    break
                if self._out_index > MAX_KNOWN_FRAME_INDEX:
                    logger.debug(
                        "skipping keepalive: frame index %d > MAX_KNOWN_FRAME_INDEX %d",
                        self._out_index, MAX_KNOWN_FRAME_INDEX,
                    )
                    # Bump out_index/rand to stay aligned even when dropping.
                    self.stats.frames_dropped_unknown_key += 1
                    continue
                pt = self._build_report_info_plaintext().encode("ascii")
                try:
                    await self._send_encrypted_frame(pt)
                except PearlhashProtocolError:
                    return
        except asyncio.CancelledError:
            return

    def _build_report_info_plaintext(self) -> str:
        """Build a `report_info` keepalive JSON body for the next outgoing id.

        The id field must equal the frame index (per memo 51 §6 observation
        that login resp pairs id=0; the global counter is the same as the
        frame index N). Hashrate format is Python `repr(float)` — pearl-miner
        v2 uses Python-style float repr too (memo 51 §2.2).
        """
        msg = {
            "id": self._out_index,
            "method": "report_info",
            "params": [
                {
                    "name": self.gpu_name,
                    "hashrate": float(self.current_hashrate),
                }
            ],
        }
        return json.dumps(msg, separators=(",", ":"))

    # ---- teardown -------------------------------------------------------

    async def _close_socket(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None


__all__ = [
    "KEEPALIVE_INTERVAL_S",
    "PearlhashClient",
    "PearlhashProtocolError",
    "PearlhashStats",
    "SubmitNotYetSupportedError",
]
