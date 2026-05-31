"""End-to-end proxy tests.

Topology:

    [simulated alpha-miner client] <-TCP-> [ProxyServer] <-TCP-> [MockPool]

Each test starts a ``ProxyServer`` on an ephemeral port pointing at the
fixture-provided ``MockPool``, then drives one or more "client" coroutines
that play the alpha-miner role.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from pearl_stratum_proxy.error21_interceptor import LineFramer
from pearl_stratum_proxy.proxy import ProxyServer

from .conftest import (
    make_set_difficulty,
    make_notify,
    make_submit_stale_error,
)


WORKER_BASE = (
    "prl1pja266dfa7kcg0xdagaacy0y7x60h7qrw3tcau4enx4gwnmmyxxvs7ep7ad.rig03v2"
)


# ----------------------------------------------------------------------
# Helpers


async def _start_proxy(mock_pool) -> ProxyServer:
    server = ProxyServer(
        listen_host="127.0.0.1",
        listen_port=0,
        upstream_host=mock_pool.host,
        upstream_port=mock_pool.port,
    )
    await server.start()
    return server


class FakeMinerClient:
    """A test-only stand-in for alpha-miner's stratum client.

    Drives the standard configure -> subscribe -> authorize handshake and
    exposes a queue of received lines for assertion.
    """

    def __init__(self, host: str, port: int, worker_name: str) -> None:
        self.host = host
        self.port = port
        self.worker_name = worker_name
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self.received: list[bytes] = []
        self._reader_task: asyncio.Task[None] | None = None
        self._framer = LineFramer()
        self._next_id = 46

    async def connect(self) -> None:
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        self._reader_task = asyncio.create_task(self._reader_loop())

    async def _reader_loop(self) -> None:
        assert self.reader is not None
        try:
            while True:
                try:
                    chunk = await self.reader.read(65536)
                except Exception:
                    break
                if not chunk:
                    break
                for line in self._framer.feed(chunk):
                    self.received.append(line)
        except Exception:
            pass

    async def send_raw(self, line: bytes) -> None:
        assert self.writer is not None
        self.writer.write(line)
        await self.writer.drain()

    async def send(self, obj: dict[str, Any]) -> None:
        await self.send_raw((json.dumps(obj) + "\n").encode("utf-8"))

    def alloc_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    async def configure(self) -> int:
        i = self.alloc_id()
        await self.send({"id": i, "method": "mining.configure", "params": [["pearl/v1"], {}]})
        return i

    async def subscribe(self) -> int:
        i = self.alloc_id()
        await self.send({"id": i, "method": "mining.subscribe", "params": ["alpha-miner/0.1"]})
        return i

    async def authorize(self) -> int:
        i = self.alloc_id()
        await self.send(
            {
                "id": i,
                "method": "mining.authorize",
                "params": [self.worker_name, "x;d=1048576"],
            }
        )
        return i

    async def full_handshake(self) -> tuple[int, int, int]:
        cid = await self.configure()
        sid = await self.subscribe()
        aid = await self.authorize()
        return cid, sid, aid

    async def submit(self, job_id: str, proof: str = "AAAA") -> int:
        i = self.alloc_id()
        await self.send(
            {
                "id": i,
                "method": "mining.submit",
                "params": [self.worker_name, job_id, proof],
            }
        )
        return i

    async def close_client_socket(self) -> None:
        """Simulate alpha-miner's reconnect_drop_share FIN."""
        if self.writer is not None:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass
            self.writer = None
            self.reader = None
        if self._reader_task is not None:
            try:
                await asyncio.wait_for(self._reader_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
            self._reader_task = None

    def parsed_objects(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for line in self.received:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out

    def lines_with_method(self, method: str) -> list[dict[str, Any]]:
        return [o for o in self.parsed_objects() if o.get("method") == method]

    def response_with_id(self, msg_id: int) -> dict[str, Any] | None:
        for o in self.parsed_objects():
            if o.get("id") == msg_id and "method" not in o:
                return o
        return None


async def _wait_for(predicate, timeout: float = 2.0, interval: float = 0.02) -> bool:
    """Poll until predicate() returns truthy or timeout. Returns False on timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


# ======================================================================
# Test 1: Normal forwarding works.


async def test_normal_handshake_and_submit_forwards(mock_pool) -> None:
    server = await _start_proxy(mock_pool)
    try:
        client = FakeMinerClient("127.0.0.1", server.actual_port, f"{WORKER_BASE}.gpu0")
        await client.connect()
        try:
            cid, sid, aid = await client.full_handshake()

            # Wait for proxy/pool to round-trip the responses.
            ok = await _wait_for(
                lambda: client.response_with_id(aid) is not None
                and client.lines_with_method("mining.notify"),
                timeout=2.0,
            )
            assert ok, f"didn't see auth response and notify; got {client.parsed_objects()}"

            # Configure / subscribe / authorize responses present.
            assert client.response_with_id(cid) == {"jsonrpc": "2.0", "id": cid, "result": {"pearl/v1": True, "pearl/v1.share_format": "base64"}}
            assert client.response_with_id(sid) is not None
            assert client.response_with_id(aid) == {"jsonrpc": "2.0", "id": aid, "result": True}

            # And the pool got configure / subscribe / authorize requests too.
            assert len(mock_pool.connections) == 1
            pool_recv = mock_pool.connections[0].received
            methods = [o.get("method") for o in pool_recv]
            assert methods.count("mining.configure") == 1
            assert methods.count("mining.subscribe") == 1
            assert methods.count("mining.authorize") == 1

            # Now a submit should forward and get a success response.
            submit_id = await client.submit("0000d446-3061")
            ok = await _wait_for(
                lambda: client.response_with_id(submit_id) is not None, timeout=2.0
            )
            assert ok
            assert client.response_with_id(submit_id) == {"jsonrpc": "2.0", "id": submit_id, "result": True}
        finally:
            await client.close_client_socket()
    finally:
        await server.stop()


# ======================================================================
# Test 2: Error 21 path — upstream stays open across client FIN.


async def test_error21_does_not_close_upstream(mock_pool) -> None:
    # Override submit handler to return error 21.
    async def stale_handler(conn, obj):
        return make_submit_stale_error(obj["id"])

    mock_pool.submit_handler = stale_handler

    server = await _start_proxy(mock_pool)
    try:
        worker = f"{WORKER_BASE}.gpu1"
        client = FakeMinerClient("127.0.0.1", server.actual_port, worker)
        await client.connect()
        _cid, _sid, aid = await client.full_handshake()
        ok = await _wait_for(lambda: client.response_with_id(aid) is not None, timeout=2.0)
        assert ok

        submit_id = await client.submit("0000d446-3060")
        ok = await _wait_for(
            lambda: client.response_with_id(submit_id) is not None
            and client.response_with_id(submit_id).get("error") is not None,
            timeout=2.0,
        )
        assert ok
        err = client.response_with_id(submit_id)
        assert err["error"][0] == 21

        # Now alpha-miner does the buggy thing: close its TCP socket.
        # Upstream should remain open!
        assert len(mock_pool.connections) == 1
        pool_conn = mock_pool.connections[0]

        await client.close_client_socket()

        # Give the proxy time to process the FIN.
        await asyncio.sleep(0.2)

        # Mock pool should NOT see the upstream socket close — it only
        # closes when the proxy initiates close. The pool reader_loop
        # would have exited on EOF, but we can directly check whether
        # the pool's writer thinks it's still up by trying to push.
        try:
            pool_conn.writer.write(b'{"method":"mining.notify","params":["STILL_HERE","p","c",1,"n","t",true]}\n')
            await pool_conn.writer.drain()
            still_open = True
        except Exception:
            still_open = False
        assert still_open, "proxy propagated client FIN to upstream — bug!"

        # The persistent upstream should still be tracked in the pool.
        ups = server.pool.lookup(worker)
        assert ups is not None
        assert ups.is_alive
    finally:
        await server.stop()


# ======================================================================
# Test 3: Reconnect path — second client connect drains cached state instantly.


async def test_reconnect_replays_cached_state(mock_pool) -> None:
    server = await _start_proxy(mock_pool)
    try:
        worker = f"{WORKER_BASE}.gpu2"

        # First connect: warms the cache.
        c1 = FakeMinerClient("127.0.0.1", server.actual_port, worker)
        await c1.connect()
        _c, _s, a1 = await c1.full_handshake()
        ok = await _wait_for(
            lambda: c1.response_with_id(a1) is not None
            and c1.lines_with_method("mining.notify"),
            timeout=2.0,
        )
        assert ok

        # Cache should now be warm.
        ups = server.pool.lookup(worker)
        assert ups is not None
        assert ups.cache.is_complete(), "cache should be complete after first handshake"
        assert ups.cache.last_notify is not None
        assert ups.cache.set_mining_params is not None

        # Simulate the reconnect: close client socket cleanly.
        await c1.close_client_socket()
        await asyncio.sleep(0.1)  # let proxy detach

        # New client connects with SAME worker name.
        c2 = FakeMinerClient("127.0.0.1", server.actual_port, worker)
        await c2.connect()
        _c2, _s2, a2 = await c2.full_handshake()

        # The reconnect should be answered locally with cached responses.
        # We expect to see set_mining_params + set_difficulty + notify
        # AS PART OF THE REPLAY, not from a fresh pool round-trip.
        ok = await _wait_for(
            lambda: c2.response_with_id(a2) is not None
            and c2.lines_with_method("mining.notify")
            and c2.lines_with_method("pearl.set_mining_params"),
            timeout=2.0,
        )
        assert ok, f"reconnect didn't replay cached state; got {c2.parsed_objects()}"

        # Crucially, the pool should NOT have seen a second authorize.
        # Pool connection count should still be 1 (one upstream per worker).
        assert len(mock_pool.connections) == 1
        methods = [o.get("method") for o in mock_pool.connections[0].received]
        # First handshake = configure+subscribe+authorize, plus possibly
        # the submit. Reconnect should add zero handshake messages.
        assert methods.count("mining.authorize") == 1
        assert methods.count("mining.configure") == 1
        assert methods.count("mining.subscribe") == 1

        # The replayed authorize response carries our NEW id.
        ar = c2.response_with_id(a2)
        assert ar == {"jsonrpc": "2.0", "id": a2, "result": True}

        await c2.close_client_socket()
    finally:
        await server.stop()


async def test_reconnect_delivers_notify_buffered_during_outage(mock_pool) -> None:
    """A notify pushed by the pool WHILE the client was offline should
    be delivered to the new client immediately on attach."""
    server = await _start_proxy(mock_pool)
    try:
        worker = f"{WORKER_BASE}.gpu3"

        c1 = FakeMinerClient("127.0.0.1", server.actual_port, worker)
        await c1.connect()
        await c1.full_handshake()
        ok = await _wait_for(
            lambda: c1.lines_with_method("mining.notify"), timeout=2.0
        )
        assert ok
        await c1.close_client_socket()
        await asyncio.sleep(0.1)

        # Push a fresh notify while NOBODY is attached.
        late_notify = (
            b'{"method":"mining.notify","params":["LATE_JOB","p","c",1,"n","t",true]}\n'
        )
        await mock_pool.push(mock_pool.connections[0], late_notify)
        await asyncio.sleep(0.1)

        # Reconnect.
        c2 = FakeMinerClient("127.0.0.1", server.actual_port, worker)
        await c2.connect()
        await c2.full_handshake()
        ok = await _wait_for(
            lambda: any(b'"LATE_JOB"' in line for line in c2.received), timeout=2.0
        )
        assert ok, f"buffered late notify wasn't delivered on reconnect; got {c2.received}"

        await c2.close_client_socket()
    finally:
        await server.stop()


# ======================================================================
# Test 4: Multi-GPU — 6 simultaneous clients, distinct workers, distinct upstreams.


async def test_six_simultaneous_workers_six_upstreams(mock_pool) -> None:
    server = await _start_proxy(mock_pool)
    try:
        clients = []
        for gpu in range(6):
            c = FakeMinerClient(
                "127.0.0.1", server.actual_port, f"{WORKER_BASE}.gpu{gpu}"
            )
            await c.connect()
            clients.append(c)

        # Each does its handshake.
        ids = []
        for c in clients:
            ids.append(await c.full_handshake())

        # Wait for all 6 to receive auth + notify.
        async def all_ready():
            for c, (_, _, aid) in zip(clients, ids):
                if c.response_with_id(aid) is None:
                    return False
                if not c.lines_with_method("mining.notify"):
                    return False
            return True

        ok = await _wait_for(all_ready, timeout=3.0)
        if not ok:
            details = []
            for i, (c, (_, _, aid)) in enumerate(zip(clients, ids)):
                resp = c.response_with_id(aid)
                notifies = c.lines_with_method("mining.notify")
                details.append(
                    f"gpu{i}: auth_resp={bool(resp)} notifies={len(notifies)} total_lines={len(c.received)}"
                )
            assert False, "not all 6 clients completed handshake\n" + "\n".join(details)

        # Pool should have seen exactly 6 distinct connections.
        assert len(mock_pool.connections) == 6, (
            f"expected 6 pool connections (one per worker), got {len(mock_pool.connections)}"
        )

        # Proxy pool should have 6 distinct upstreams.
        assert len(server.pool) == 6

        # Each connection should see its OWN worker name only (no cross-talk).
        for conn in mock_pool.connections:
            auth_msgs = [o for o in conn.received if o.get("method") == "mining.authorize"]
            assert len(auth_msgs) == 1
            wn = auth_msgs[0]["params"][0]
            assert wn.startswith(f"{WORKER_BASE}.gpu")

        # Workers seen across all 6 connections should be the 6 distinct gpuN.
        seen_workers = set()
        for conn in mock_pool.connections:
            auth = next(o for o in conn.received if o.get("method") == "mining.authorize")
            seen_workers.add(auth["params"][0])
        assert seen_workers == {f"{WORKER_BASE}.gpu{i}" for i in range(6)}

        for c in clients:
            await c.close_client_socket()
    finally:
        await server.stop()


# ======================================================================
# Sanity / regression


async def test_invalid_authorize_returns_error_no_upstream(mock_pool) -> None:
    """Authorize without a worker name should fail-fast locally; we
    shouldn't open a useless upstream connection."""
    server = await _start_proxy(mock_pool)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.actual_port)
        try:
            # Send an authorize with empty params.
            writer.write(b'{"id":1,"method":"mining.authorize","params":[]}\n')
            await writer.drain()
            data = await asyncio.wait_for(reader.readline(), timeout=2.0)
            obj = json.loads(data)
            assert obj["id"] == 1
            assert "error" in obj
            # No upstream should have been opened (mock pool sees 0 conns).
            await asyncio.sleep(0.1)
            assert len(mock_pool.connections) == 0
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
    finally:
        await server.stop()
