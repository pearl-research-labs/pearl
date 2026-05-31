"""Long-lived upstream connections, keyed by ``worker_name``.

# Per-upstream-pool design (one upstream per GPU client)

Alpha-miner opens **6** TCP sockets to the pool — one per GPU — and each
authorises with a distinct ``worker.gpuN`` name (STRATUM_CAPTURE.md §2).
Pool state (set_difficulty, current job, share-id sequence) is per
connection, not per worker-name, so we MUST NOT multiplex 6 clients onto a
shared upstream — share/job state divergence would mis-route results and
silently corrupt accounting.

The pool key is therefore the full ``worker_name`` string (eg
``prl1pja266...rig03v2.gpu2``). When a client reconnects (after the
alpha-miner's reconnect_drop_share bug fires) with the SAME worker name,
we reuse the matching persistent upstream and replay cached responses
locally so the GPU is back at work in <10 ms instead of 490-2480 ms.

# What we cache per upstream

Per ``CachedHandshake``:

  - configure response (§3a)
  - subscribe response (§3b)
  - authorize response (§3d)
  - pearl.set_mining_params notification (§3c — byte-identical across
    reconnects, hence safe to replay)
  - most recent mining.set_difficulty (§3e)
  - most recent mining.notify (§3f)

# Lifetime

A ``PersistentUpstream`` is created on the first connection for a worker
and reused across as many alpha-miner reconnects as the upstream remains
healthy. If the upstream pool socket itself dies (FIN/RST from server,
read timeout) we tear it down and the next client connect rebuilds the
full handshake from scratch.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from .error21_interceptor import LineFramer, StratumMessage, parse_line

logger = logging.getLogger(__name__)


# We give the upstream a generous idle ceiling. Pool sends a notify
# every ~5-15 s in practice, so a half-minute of silence is anomalous.
UPSTREAM_IDLE_TIMEOUT_S = 60.0


@dataclass(slots=True)
class CachedHandshake:
    """Per-worker handshake state we can replay to a reconnecting client.

    Each field stores the *full* JSON-RPC line (with trailing ``\\n``)
    exactly as it came off the upstream socket. The proxy writes these
    bytes verbatim back to the new client when it reconnects.
    """

    configure_response: bytes | None = None
    subscribe_response: bytes | None = None
    authorize_response: bytes | None = None
    set_mining_params: bytes | None = None  # pearl.set_mining_params notification
    last_set_difficulty: bytes | None = None
    last_notify: bytes | None = None

    def is_complete(self) -> bool:
        """True once we've captured the minimum to bootstrap a new client.

        The pearl protocol bootstrap is configure -> subscribe ->
        authorize, with pearl.set_mining_params pushed between subscribe
        and authorize (§3c). A new client can start mining as soon as it
        has set_difficulty + a notify.
        """
        return (
            self.configure_response is not None
            and self.subscribe_response is not None
            and self.authorize_response is not None
        )


@dataclass
class PersistentUpstream:
    """One long-lived TCP connection to the pool, keyed by worker name.

    The upstream is opened the first time a client authorises with this
    ``worker_name``, and reused across as many alpha-miner reconnect
    cycles as it stays alive.
    """

    worker_name: str
    upstream_host: str
    upstream_port: int

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter

    cache: CachedHandshake = field(default_factory=CachedHandshake)

    # Refcount of currently-attached client connections. Normally 0 or 1
    # (one alpha-miner GPU thread per worker). >1 would be a bug —
    # multiplexing 2 clients onto one upstream breaks share accounting.
    _refcount: int = 0

    # Forward task: reads from the upstream and pushes lines to whichever
    # client is currently attached. Stays alive across client reconnects.
    _forward_task: asyncio.Task[None] | None = None

    # Currently attached client writer (None when the client has FIN'd
    # and we're waiting for its reconnect). When None, we BUFFER notifies
    # into _pending_to_client so the next attach sees them immediately.
    _client_writer: asyncio.StreamWriter | None = None

    # Bounded buffer for notifications received while no client is
    # attached. Bounded to avoid unbounded memory growth if the alpha-
    # miner never reconnects (eg killed).
    _pending_to_client: list[bytes] = field(default_factory=list)
    _PENDING_LIMIT: int = 64

    # Lock around (cache mutations, client_writer swap, pending buffer)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # ------------------------------------------------------------------
    # Lifecycle

    @classmethod
    async def connect(
        cls, worker_name: str, host: str, port: int
    ) -> "PersistentUpstream":
        """Open a fresh TCP connection to the pool and return a manager.

        Raises whatever ``asyncio.open_connection`` raises (OSError on
        DNS / refused / timeout). Callers handle those by returning a
        protocol error to the alpha-miner.
        """
        reader, writer = await asyncio.open_connection(host, port)
        up = cls(
            worker_name=worker_name,
            upstream_host=host,
            upstream_port=port,
            reader=reader,
            writer=writer,
        )
        up._forward_task = asyncio.create_task(
            up._upstream_reader_loop(), name=f"upstream-rdr[{worker_name[-12:]}]"
        )
        logger.info(
            "upstream OPEN worker=%s peer=%s:%d", worker_name[-20:], host, port
        )
        return up

    async def close(self) -> None:
        """Tear down the upstream. Idempotent."""
        if self._forward_task is not None:
            self._forward_task.cancel()
            try:
                await self._forward_task
            except (asyncio.CancelledError, Exception):
                pass
            self._forward_task = None
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except Exception:
            pass
        logger.info("upstream CLOSE worker=%s", self.worker_name[-20:])

    @property
    def is_alive(self) -> bool:
        """True iff the upstream socket is still usable.

        We treat (a) the writer being closed, or (b) the reader loop having
        finished/errored as "dead". The proxy checks this before reusing
        an upstream for a reconnect; on death it builds a fresh one.
        """
        if self.writer.is_closing():
            return False
        if self._forward_task is None or self._forward_task.done():
            return False
        return True

    # ------------------------------------------------------------------
    # Client attach / detach

    async def attach_client(self, client_writer: asyncio.StreamWriter) -> list[bytes]:
        """Make this client the destination for future upstream lines.

        Returns the buffered bytes the caller should immediately write
        to the new client (replays of cached set_difficulty / notify that
        arrived while no client was attached).

        Increments the refcount and asserts <= 1: we should NEVER attach
        a second client. If we do, that's a code bug elsewhere and we'd
        rather crash loudly than silently corrupt share routing.
        """
        async with self._lock:
            self._refcount += 1
            if self._refcount > 1:
                self._refcount -= 1
                raise RuntimeError(
                    f"upstream for worker={self.worker_name} has "
                    f"refcount>1 — multiplexing would corrupt share state"
                )
            self._client_writer = client_writer
            pending = self._pending_to_client
            self._pending_to_client = []
            return pending

    async def detach_client(self) -> None:
        """Mark the client as gone. Holds the upstream open."""
        async with self._lock:
            self._client_writer = None
            if self._refcount > 0:
                self._refcount -= 1

    @property
    def has_attached_client(self) -> bool:
        return self._client_writer is not None

    # ------------------------------------------------------------------
    # Upstream write — caller-driven, mostly pass-through.

    async def send_to_upstream(self, line: bytes) -> None:
        """Write a single framed JSON-RPC line to the pool socket."""
        self.writer.write(line)
        await self.writer.drain()

    # ------------------------------------------------------------------
    # Cache mutation, called from the reader loop.

    def _maybe_cache(self, msg: StratumMessage) -> None:
        """Decide whether to cache this server-to-client message.

        We cache notifications by method, and responses by — well, we
        can't tell what method a response correlates to without external
        state (the in-flight request map lives in the per-client
        ClientSession, not here). So responses for caching purposes are
        identified opportunistically by what's inside `result`.
        """
        if msg.is_notification:
            if msg.method == "mining.set_difficulty":
                self.cache.last_set_difficulty = msg.raw
            elif msg.method == "mining.notify":
                self.cache.last_notify = msg.raw
            elif msg.method == "pearl.set_mining_params":
                self.cache.set_mining_params = msg.raw
        # Response caching is driven by the client session as it sees
        # request ids fly past (configure, subscribe, authorize). The
        # client session calls cache_response_for() below.

    def cache_response_for(self, method: str, raw: bytes) -> None:
        """Manually cache a response that the per-client session has
        correlated to a known method."""
        if method == "mining.configure":
            self.cache.configure_response = raw
        elif method == "mining.subscribe":
            self.cache.subscribe_response = raw
        elif method == "mining.authorize":
            self.cache.authorize_response = raw

    # ------------------------------------------------------------------
    # Upstream reader loop.

    async def _upstream_reader_loop(self) -> None:
        """Read lines from the pool, cache notifies, forward to client.

        When no client is attached, notifies go into ``_pending_to_client``
        (bounded). When the client reconnects, ``attach_client()`` drains
        the buffer in one go.

        Exits on EOF / cancellation / socket error. After exit, ``is_alive``
        returns False.
        """
        framer = LineFramer()
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        self.reader.read(65536), timeout=UPSTREAM_IDLE_TIMEOUT_S
                    )
                except asyncio.TimeoutError:
                    # Idle ceiling exceeded. Pool isn't pushing notifies.
                    # We treat this as "upstream is probably dead"; the
                    # alpha-miner has its own ping/notify expectation and
                    # would reconnect anyway.
                    logger.warning(
                        "upstream idle timeout worker=%s", self.worker_name[-20:]
                    )
                    return
                if not chunk:
                    # Clean EOF from the pool — connection done.
                    logger.info("upstream EOF worker=%s", self.worker_name[-20:])
                    return
                lines = framer.feed(chunk)
                for line in lines:
                    await self._handle_upstream_line(line)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "upstream reader error worker=%s err=%r",
                self.worker_name[-20:],
                exc,
            )

    async def _handle_upstream_line(self, line: bytes) -> None:
        """Cache, then forward to the attached client (or buffer)."""
        try:
            msg = parse_line(line)
        except Exception:
            # Unknown line — pass it through unparsed. Pool may have
            # extended the protocol; we shouldn't drop bytes.
            msg = None

        if msg is not None:
            self._maybe_cache(msg)

        async with self._lock:
            cw = self._client_writer
            if cw is None or cw.is_closing():
                # No live client attached. Buffer.
                if len(self._pending_to_client) >= self._PENDING_LIMIT:
                    # Drop oldest. Worst-case the client sees stale jobs;
                    # they'll be superseded by the next notify anyway.
                    self._pending_to_client.pop(0)
                self._pending_to_client.append(line)
                return
            client_writer = cw

        # Outside the lock to avoid holding it across drain()
        # (drain can suspend on backpressure).
        try:
            client_writer.write(line)
            await client_writer.drain()
        except Exception as exc:
            logger.debug(
                "client write failed worker=%s err=%r — detaching",
                self.worker_name[-20:],
                exc,
            )
            # Client died on us — detach so subsequent lines buffer.
            async with self._lock:
                self._client_writer = None
                if self._refcount > 0:
                    self._refcount -= 1


class UpstreamPool:
    """Maps ``worker_name`` -> ``PersistentUpstream``.

    All access is awaitable so we serialise creation: two simultaneous
    client connections with the same worker name (a transient state
    during the alpha-miner reconnect race) should share one upstream,
    not race to open two.
    """

    def __init__(self, upstream_host: str, upstream_port: int) -> None:
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self._by_worker: dict[str, PersistentUpstream] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, worker_name: str) -> PersistentUpstream:
        """Return a healthy upstream for ``worker_name``, creating it
        if absent or replacing it if dead."""
        async with self._lock:
            existing = self._by_worker.get(worker_name)
            if existing is not None and existing.is_alive:
                return existing
            if existing is not None:
                # Dead — clean up before replacing.
                await existing.close()
                self._by_worker.pop(worker_name, None)
            up = await PersistentUpstream.connect(
                worker_name, self.upstream_host, self.upstream_port
            )
            self._by_worker[worker_name] = up
            return up

    def lookup(self, worker_name: str) -> Optional[PersistentUpstream]:
        """Non-creating lookup, used by tests / introspection."""
        return self._by_worker.get(worker_name)

    async def close_all(self) -> None:
        async with self._lock:
            ups = list(self._by_worker.values())
            self._by_worker.clear()
        for up in ups:
            await up.close()

    def __len__(self) -> int:
        return len(self._by_worker)
