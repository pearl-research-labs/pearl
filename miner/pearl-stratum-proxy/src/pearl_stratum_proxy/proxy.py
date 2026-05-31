"""Pearl Stratum Proxy — main asyncio TCP server.

# Why this exists

Alpha-miner has a closed-source bug: on every stale-share rejection
(``error[21]``) it tears down the TCP socket and reconnects. Over 60 s of
production capture (STRATUM_CAPTURE.md §4) this wasted ~42% of share
submissions to TCP teardown + reconnect handshake.

This proxy sits between alpha-miner and the upstream alphapool. It runs
on the same host as the miner, listens on 127.0.0.1:5567, and forwards
to ``us2.alphapool.tech:5566``. The alpha-miner is reconfigured with
``--pool stratum+tcp://127.0.0.1:5567`` and is otherwise unchanged.

# What it does

1. Accepts the alpha-miner client connection.
2. Reads the ``mining.authorize`` request to extract the worker name.
3. Looks up (or creates) a persistent upstream connection for that
   worker. Each GPU's worker.gpuN gets its own upstream — one connection
   per alpha-miner socket — so per-pool share/job state never mixes.
4. Replays the cached handshake responses to the client locally (so the
   GPU is back at work in <10 ms instead of 490-2480 ms).
5. When the client FINs after an error-21 response, the proxy DOES NOT
   propagate the close to the upstream. The upstream stays open; the
   pool keeps sending notifies that the proxy buffers. On the next
   client reconnect (new ephemeral port, same worker name) the buffer
   drains immediately.

# Why we can do the swallow-close trick

``asyncio.StreamReader.read()`` returning ``b""`` is just "the client
TCP half-closed". We're free to NOT close the upstream — that's a
totally separate TCP connection. The "trick" is purely "don't propagate
the close". The persistent upstream's reader loop notices the client
has detached and buffers further notifies into ``_pending_to_client``.

# Why per-worker, not per-connection

See ``persistent_upstream.py`` module docstring. TL;DR: pool state is
per-TCP-connection on the server side, so mapping N client connections
to <N upstream connections would scramble share-id sequences and break
accounting.

# What we DON'T do (out of scope for the skeleton)

- TLS termination (alphapool is plain TCP).
- Vardiff smoothing (we forward set_difficulty verbatim).
- Pool failover (one upstream host:port; for failover, restart the
  proxy with a different ``--upstream``).
- mining.submit dedup (alpha-miner doesn't double-submit; we forward
  submits as-is).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .error21_interceptor import (
    LineFramer,
    StratumMessage,
    classify_response_id_to_method,
    parse_line,
)
from .persistent_upstream import PersistentUpstream, UpstreamPool

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Per-client session.

# After a client sends mining.authorize but before we receive the response,
# we keep its requests buffered locally instead of forwarding upstream when
# we plan to short-circuit. The short-circuit list:
SHORT_CIRCUITABLE_METHODS = frozenset(
    {"mining.configure", "mining.subscribe", "mining.authorize"}
)


@dataclass
class ClientSession:
    """One alpha-miner client connection.

    Owns the client-side TCP socket and a reference to its
    ``PersistentUpstream`` (which may outlive the session if the client
    reconnects).
    """

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    peer: str  # for logging only

    pool: UpstreamPool

    # Discovered from the authorize call. Until known, we buffer.
    worker_name: str | None = None

    # The upstream this session is attached to. Set after authorize.
    upstream: PersistentUpstream | None = None

    # Pending request-id -> method, so we can correlate responses we want
    # to cache (configure / subscribe / authorize).
    pending_requests: dict[int | str, str] = field(default_factory=dict)

    # If we short-circuited, we still need to remember the configure /
    # subscribe / authorize request ids so we can answer locally with the
    # cached responses. Maps request-id -> method.
    short_circuited_requests: dict[int | str, str] = field(default_factory=dict)

    # Diagnostic counters.
    n_short_circuit_replies: int = 0
    n_buffered_replays_sent: int = 0
    n_swallowed_closes: int = 0
    started_at: float = field(default_factory=time.monotonic)

    # ------------------------------------------------------------------
    # Driver: one task per client.

    async def run(self) -> None:
        """Read from the client until EOF; handle the worker-name discovery
        and either short-circuit or forward upstream."""
        framer = LineFramer()
        logger.info("client OPEN peer=%s", self.peer)
        try:
            while True:
                try:
                    chunk = await self.reader.read(65536)
                except Exception as exc:
                    logger.debug("client read err peer=%s err=%r", self.peer, exc)
                    break
                if not chunk:
                    # Client FIN. This is where the swallow-close trick lives.
                    self.n_swallowed_closes += 1
                    logger.info(
                        "client FIN peer=%s worker=%s — holding upstream open",
                        self.peer,
                        (self.worker_name or "?")[-20:],
                    )
                    break
                for line in framer.feed(chunk):
                    await self._handle_client_line(line)
        finally:
            # Detach from the upstream (don't close it!) and close client.
            if self.upstream is not None:
                await self.upstream.detach_client()
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass
            logger.info(
                "client CLOSE peer=%s worker=%s short=%d replays=%d swallowed=%d",
                self.peer,
                (self.worker_name or "?")[-20:],
                self.n_short_circuit_replies,
                self.n_buffered_replays_sent,
                self.n_swallowed_closes,
            )

    # ------------------------------------------------------------------
    # Per-line dispatch.

    async def _handle_client_line(self, line: bytes) -> None:
        try:
            msg = parse_line(line)
        except Exception as exc:
            logger.warning(
                "client sent unparseable line peer=%s err=%r — forwarding raw",
                self.peer,
                exc,
            )
            if self.upstream is not None:
                await self.upstream.send_to_upstream(line)
            return

        # Pre-authorize: we don't yet know which upstream to forward to.
        # Worker name comes from authorize.params[0].
        if msg.method == "mining.authorize":
            await self._handle_authorize(msg)
            return

        # Other handshake calls before authorize: hold them and replay
        # once we know the worker.
        if msg.method in SHORT_CIRCUITABLE_METHODS and self.worker_name is None:
            # Stash; reply locally after we have an upstream.
            self.short_circuited_requests[msg.msg_id] = msg.method  # type: ignore[index]
            self.pending_requests[msg.msg_id] = msg.method  # type: ignore[index]
            return

        # Post-authorize: forward to upstream.
        if self.upstream is None:
            logger.warning(
                "client sent method=%s before authorize peer=%s — buffering anyway",
                msg.method,
                self.peer,
            )
            self.pending_requests[msg.msg_id] = msg.method or ""  # type: ignore[index]
            return

        if msg.is_request and msg.method is not None:
            self.pending_requests[msg.msg_id] = msg.method  # type: ignore[index]

        await self.upstream.send_to_upstream(line)

    # ------------------------------------------------------------------
    # Authorize: pick or build upstream, then short-circuit.

    async def _handle_authorize(self, msg: StratumMessage) -> None:
        """Process mining.authorize from the client.

        params: [worker_name, password]. See STRATUM_CAPTURE.md §3d.
        """
        worker_name = self._extract_worker_name(msg)
        if worker_name is None:
            # Without a worker name we can't route to an upstream. Send a
            # synthetic error response and let alpha-miner give up.
            await self._send_local_error(
                msg.msg_id,
                code=-32602,
                message="invalid authorize params: missing worker name",
            )
            return

        self.worker_name = worker_name
        # The auth request also goes into short_circuited so we reply
        # locally when the cache is warm; and into pending_requests so
        # the cold-path sniffer correlates the upstream auth response
        # to the right method for caching.
        self.short_circuited_requests[msg.msg_id] = "mining.authorize"  # type: ignore[index]
        self.pending_requests[msg.msg_id] = "mining.authorize"  # type: ignore[index]

        try:
            upstream = await self.pool.get_or_create(worker_name)
        except Exception as exc:
            logger.error(
                "failed to open upstream worker=%s err=%r",
                worker_name[-20:],
                exc,
            )
            await self._send_local_error(
                msg.msg_id,
                code=-32000,
                message=f"upstream connect failed: {exc!r}",
            )
            return

        self.upstream = upstream

        # Attach client to upstream (this is what makes future notifies flow).
        try:
            buffered = await upstream.attach_client(self.writer)
        except RuntimeError as exc:
            # Another client already attached for this worker.
            logger.error(
                "refused to attach client peer=%s worker=%s reason=%r",
                self.peer,
                worker_name[-20:],
                exc,
            )
            await self._send_local_error(
                msg.msg_id,
                code=-32000,
                message=str(exc),
            )
            return

        # If the upstream cache is COLD (this is the very first client for
        # this worker since the proxy started), we need to forward the
        # handshake upstream so the pool sees an authorise. We'll then
        # capture responses for replay later.
        if not upstream.cache.is_complete():
            await self._cold_handshake_forward(msg)
        else:
            # Warm cache. Replay everything locally — no upstream round-trip.
            await self._warm_replay()

        # Replay any buffered notifies the upstream collected while there
        # was no client attached.
        for line in buffered:
            self.writer.write(line)
            self.n_buffered_replays_sent += 1
        if buffered:
            await self.writer.drain()
            logger.info(
                "replayed n=%d cached msgs to client peer=%s worker=%s",
                len(buffered),
                self.peer,
                worker_name[-20:],
            )

    async def _cold_handshake_forward(self, auth_msg: StratumMessage) -> None:
        """First client for this worker: forward the queued handshake
        requests upstream so the pool sees a real authorise.

        IMPORTANT: install the response sniffer BEFORE forwarding
        anything to the upstream. Otherwise the pool's response races
        against us and the configure/subscribe responses arrive at the
        client writer before the sniffer is monkey-patched in — meaning
        we'd never cache them.
        """
        assert self.upstream is not None

        # 1. Install sniffer first so we catch every reply.
        self._install_response_sniffer()

        # 2. Forward the queued configure / subscribe in id-order.
        # alpha-miner's pattern: configure (id=46), subscribe (id=47),
        # authorize (id=48).
        stashed = sorted(self.short_circuited_requests.items(), key=lambda kv: _id_sort_key(kv[0]))
        for req_id, method in stashed:
            if method == "mining.authorize":
                continue  # handled below
            await self._forward_stashed_request(req_id, method)

        # 3. Forward the live authorize verbatim.
        await self.upstream.send_to_upstream(auth_msg.raw)

        # Now: future writes from upstream go through the sniffer, which
        # caches configure/subscribe/authorize responses on the
        # PersistentUpstream by correlating ids in self.pending_requests.

    async def _forward_stashed_request(self, req_id: int | str, method: str) -> None:
        """Re-synthesise a stashed request (configure/subscribe) for
        forwarding upstream during the cold handshake."""
        assert self.upstream is not None
        # We don't have the original bytes; reconstruct minimal canonical
        # forms. This loses anything custom (extensions, extra params)
        # but matches what alpha-miner sends per STRATUM_CAPTURE.md.
        if method == "mining.configure":
            obj = {"id": req_id, "method": "mining.configure", "params": [["pearl/v1"], {}]}
        elif method == "mining.subscribe":
            obj = {"id": req_id, "method": "mining.subscribe", "params": ["alpha-miner/0.1"]}
        else:
            return
        line = (json.dumps(obj, separators=(", ", ": ")) + "\n").encode("utf-8")
        await self.upstream.send_to_upstream(line)

    async def _warm_replay(self) -> None:
        """Cache is complete — answer the queued configure / subscribe /
        authorize requests locally with the cached responses."""
        assert self.upstream is not None
        cache = self.upstream.cache

        # The cached responses are bytes captured with their ORIGINAL
        # request ids embedded. The new client used different ids, so
        # we must re-stamp the id field before sending. Parse, swap id,
        # re-serialise.
        for req_id, method in sorted(
            self.short_circuited_requests.items(), key=lambda kv: _id_sort_key(kv[0])
        ):
            cached_bytes: bytes | None
            if method == "mining.configure":
                cached_bytes = cache.configure_response
            elif method == "mining.subscribe":
                cached_bytes = cache.subscribe_response
            elif method == "mining.authorize":
                cached_bytes = cache.authorize_response
            else:
                cached_bytes = None
            if cached_bytes is None:
                continue
            try:
                obj = json.loads(cached_bytes)
            except Exception:
                # Cache is corrupt; fall back to forwarding upstream.
                logger.warning(
                    "cache parse fail for %s — forwarding upstream", method
                )
                await self._forward_stashed_request(req_id, method)
                continue
            obj["id"] = req_id
            framed = (json.dumps(obj) + "\n").encode("utf-8")
            self.writer.write(framed)
            self.n_short_circuit_replies += 1

        # pearl.set_mining_params is a NOTIFICATION (no id). Replay verbatim.
        if cache.set_mining_params is not None:
            self.writer.write(cache.set_mining_params)
            self.n_short_circuit_replies += 1

        # Now any cached set_difficulty / notify — push them too so the
        # GPU starts work immediately.
        if cache.last_set_difficulty is not None:
            self.writer.write(cache.last_set_difficulty)
            self.n_short_circuit_replies += 1
        if cache.last_notify is not None:
            self.writer.write(cache.last_notify)
            self.n_short_circuit_replies += 1

        await self.writer.drain()

        # Clear the short-circuit queue.
        self.short_circuited_requests.clear()

    # ------------------------------------------------------------------
    # Response sniffer.

    def _install_response_sniffer(self) -> None:
        """During cold handshake, snoop on the upstream-to-client byte
        flow to capture configure/subscribe/authorize responses for the
        cache.

        Implementation: monkey-wrap ``self.writer.write`` so we observe
        every line the upstream reader pushes to us. We parse each line
        and, if it correlates to a pending handshake request, cache it.
        """
        assert self.upstream is not None
        upstream = self.upstream
        original_write = self.writer.write
        framer = LineFramer()
        sess = self

        def sniffing_write(data: bytes) -> None:
            # Forward to the real client writer first.
            original_write(data)
            # Then attempt to parse & cache.
            for line in framer.feed(data):
                logger.debug("sniffer line: %r", line[:120])
                try:
                    msg = parse_line(line)
                except Exception as exc:
                    logger.debug("sniffer parse err: %r", exc)
                    continue
                method = classify_response_id_to_method(sess.pending_requests, msg)
                logger.debug(
                    "sniffer classify msg_id=%r method=%r pending_left=%r",
                    msg.msg_id,
                    method,
                    list(sess.pending_requests.keys()),
                )
                if method is not None:
                    upstream.cache_response_for(method, line)
                    if upstream.cache.is_complete():
                        logger.info(
                            "cache WARM worker=%s — future reconnects will short-circuit",
                            (sess.worker_name or "?")[-20:],
                        )
                        # We can uninstall after the cache is complete to
                        # avoid the per-line overhead on the hot mining
                        # path. Restore the original writer.
                        try:
                            sess.writer.write = original_write  # type: ignore[method-assign]
                        except Exception:
                            pass

        try:
            self.writer.write = sniffing_write  # type: ignore[method-assign]
            logger.debug("sniffer INSTALLED worker=%s", (self.worker_name or "?")[-20:])
        except Exception:
            # If monkey-patching fails (some StreamWriter implementations
            # don't allow it) we just lose the cache warmup — the proxy
            # still forwards correctly, just no short-circuit on reconnect.
            logger.warning("response sniffer install failed — caching disabled for this session")

    # ------------------------------------------------------------------
    # Helpers.

    def _extract_worker_name(self, msg: StratumMessage) -> str | None:
        params = msg.obj.get("params")
        if not isinstance(params, list) or not params:
            return None
        wn = params[0]
        if not isinstance(wn, str) or not wn:
            return None
        return wn

    async def _send_local_error(
        self, msg_id: int | str | None, code: int, message: str
    ) -> None:
        """Synthesise a JSON-RPC error response and send to client."""
        obj = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": code, "message": message},
        }
        line = (json.dumps(obj) + "\n").encode("utf-8")
        try:
            self.writer.write(line)
            await self.writer.drain()
        except Exception:
            pass


def _id_sort_key(req_id: int | str) -> tuple[int, int | str]:
    """Stable sort across int/str ids (don't crash if alpha-miner mixes)."""
    if isinstance(req_id, int):
        return (0, req_id)
    return (1, req_id)


# ======================================================================
# Server entry-point.


class ProxyServer:
    """Bind a TCP listener and spawn a ``ClientSession`` per accept."""

    def __init__(
        self,
        listen_host: str,
        listen_port: int,
        upstream_host: str,
        upstream_port: int,
    ) -> None:
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.pool = UpstreamPool(upstream_host, upstream_port)
        self._server: asyncio.base_events.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._on_client, host=self.listen_host, port=self.listen_port
        )
        sockets = self._server.sockets or ()
        for s in sockets:
            logger.info("proxy LISTEN %s", s.getsockname())

    async def serve_forever(self) -> None:
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        await self.pool.close_all()

    @property
    def actual_port(self) -> int:
        """Real bound port (useful when listening on port 0 in tests)."""
        assert self._server is not None
        sockets = self._server.sockets or ()
        if not sockets:
            return self.listen_port
        return sockets[0].getsockname()[1]

    async def _on_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        peer_repr = f"{peer[0]}:{peer[1]}" if peer else "?"
        session = ClientSession(
            reader=reader, writer=writer, peer=peer_repr, pool=self.pool
        )
        await session.run()


__all__ = [
    "ClientSession",
    "ProxyServer",
]
