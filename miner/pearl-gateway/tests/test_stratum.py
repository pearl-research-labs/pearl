"""Tests for the pool-mode Stratum transport (comm/stratum.py): URL dialect,
extranonce2 size validation, the JSON-RPC multiplexer, and the connection
lifecycle over real sockets."""

import asyncio
import json

import pytest
from pearl_gateway.comm.stratum import (
    AsyncJSONRPCClient,
    AsyncStratumConnection,
    JSONRPCError,
    parse_extranonce2_size,
    stratum_host,
    stratum_port,
    stratum_uses_tls,
)


@pytest.mark.parametrize(
    ("url", "host", "port", "tls"),
    [
        ("stratum+ssl://pool.example.com:3443", "pool.example.com", 3443, True),
        ("stratum+tls://pool.example.com", "pool.example.com", 3333, True),
        ("stratum+tcp://10.0.0.1:3333", "10.0.0.1", 3333, False),
        ("stratum://pool.example.com:4444", "pool.example.com", 4444, False),
    ],
)
def test_stratum_url_dialect(url, host, port, tls):
    assert stratum_host(url) == host
    assert stratum_port(url) == port
    assert stratum_uses_tls(url) is tls


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (4, 4),
        (32, 32),  # upper bound
        (33, 0),  # over bound => off
        (-1, 0),  # negative => off
        (1.5, 0),  # float (not int) => off
        (4.0, 0),  # whole-number float still rejected — pool must send int
        ("4", 0),  # string => off
        (None, 0),
        (True, 0),  # bool is not int here => off
    ],
)
def test_parse_extranonce2_size_validates_and_caps(value, expected):
    assert parse_extranonce2_size(value) == expected


class FakeStreamWriter:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        await asyncio.sleep(0)


class TestAsyncJSONRPCClient:
    @pytest.mark.asyncio
    async def test_call_sends_request_and_resolves_matching_response(self):
        writer = FakeStreamWriter()
        client = AsyncJSONRPCClient(writer)

        call_task = asyncio.create_task(client.call("mining.subscribe", ["pearl-gateway/1.0"]))
        await asyncio.sleep(0)

        request = json.loads(writer.writes[0].decode())
        assert request == {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "mining.subscribe",
            "params": ["pearl-gateway/1.0"],
        }

        client.handle_response({"id": 1, "result": "subscribed"})

        assert await call_task == "subscribed"

    @pytest.mark.asyncio
    async def test_call_includes_jsonrpc_version(self):
        writer = FakeStreamWriter()
        client = AsyncJSONRPCClient(writer)

        call_task = asyncio.create_task(client.call("getwork"))
        await asyncio.sleep(0)

        request = json.loads(writer.writes[0].decode())
        assert request["jsonrpc"] == "2.0"
        assert request["params"] == []

        client.handle_response({"id": 1, "result": "ok"})

        assert await call_task == "ok"

    @pytest.mark.asyncio
    async def test_fail_all_pending_rejects_waiting_call(self):
        writer = FakeStreamWriter()
        client = AsyncJSONRPCClient(writer)

        call_task = asyncio.create_task(client.call("mining.authorize", []))
        await asyncio.sleep(0)

        client.fail_all_pending("connection lost")

        with pytest.raises(ConnectionError, match="connection lost"):
            await call_task

    @pytest.mark.asyncio
    async def test_error_response_raises_json_rpc_error(self):
        writer = FakeStreamWriter()
        client = AsyncJSONRPCClient(writer)

        call_task = asyncio.create_task(client.call("mining.authorize", []))
        await asyncio.sleep(0)

        client.handle_response(
            {"id": 1, "error": {"code": -32000, "message": "authorization failed"}}
        )

        with pytest.raises(JSONRPCError, match="JSON-RPC error -32000: authorization failed"):
            await call_task

    @pytest.mark.asyncio
    async def test_stratum_error_array_raises_json_rpc_error(self):
        writer = FakeStreamWriter()
        client = AsyncJSONRPCClient(writer)

        call_task = asyncio.create_task(client.call("mining.submit", []))
        await asyncio.sleep(0)

        client.handle_response({"id": 1, "result": None, "error": [23, "below share difficulty"]})

        with pytest.raises(JSONRPCError, match="JSON-RPC error 23: below share difficulty"):
            await call_task

    @pytest.mark.asyncio
    async def test_cancelled_call_does_not_consume_later_matching_response(self):
        writer = FakeStreamWriter()
        client = AsyncJSONRPCClient(writer)

        cancelled_task = asyncio.create_task(client.call("mining.subscribe", []))
        await asyncio.sleep(0)

        cancelled_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_task

        next_task = asyncio.create_task(client.call("mining.authorize", []))
        await asyncio.sleep(0)

        client.handle_response({"id": 1, "result": "late"})
        assert not next_task.done()

        client.handle_response({"id": 2, "result": "authorized"})
        assert await next_task == "authorized"

    def test_is_response_message_ignores_notifications(self):
        assert AsyncJSONRPCClient.is_response_message({"id": 1, "result": True})
        assert not AsyncJSONRPCClient.is_response_message({"method": "mining.notify", "params": []})


class FakeStratumServer:
    """Accepts connections, answers every request with result=True, and can
    drop the current connection to trigger client reconnects."""

    def __init__(self) -> None:
        self.server: asyncio.Server | None = None
        self.connections = 0
        self.requests: list[dict] = []
        self.drop_before_replying = 0  # close instead of answering, N times
        self._writer: asyncio.StreamWriter | None = None

    async def start(self) -> int:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        self._writer = writer
        while line := await reader.readline():
            msg = json.loads(line)
            self.requests.append(msg)
            if self.drop_before_replying > 0:
                self.drop_before_replying -= 1
                break
            writer.write(
                (json.dumps({"id": msg["id"], "result": True, "error": None}) + "\n").encode()
            )
            await writer.drain()
        # Close our side on EOF, or 3.12's Server.wait_closed() waits forever
        # for the still-attached transport.
        writer.close()

    def drop_connection(self) -> None:
        assert self._writer is not None
        self._writer.close()

    async def stop(self) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise TimeoutError("condition not met in time")


class TestAsyncStratumConnection:
    @pytest.mark.asyncio
    async def test_connects_handshakes_and_reconnects_after_drop(self):
        server = FakeStratumServer()
        port = await server.start()

        handshakes = 0
        disconnects = 0
        notifications: list[tuple[str, list]] = []

        conn: AsyncStratumConnection | None = None

        async def on_disconnect() -> None:
            nonlocal disconnects
            disconnects += 1

        async def handshake() -> None:
            nonlocal handshakes
            result = await conn.call("mining.subscribe", ["test/1.0"])
            assert result is True
            handshakes += 1

        async def on_notification(method: str, params: list) -> None:
            notifications.append((method, params))

        conn = AsyncStratumConnection(
            "127.0.0.1",
            port,
            tls=False,
            reconnect_delay_seconds=0,
            max_reconnect_delay_seconds=1,
            handshake=handshake,
            on_notification=on_notification,
            on_disconnect=on_disconnect,
        )
        try:
            await conn.start()
            await _wait_for(lambda: handshakes == 1)
            assert conn.is_connected
            assert disconnects == 0
            assert server.requests[0]["method"] == "mining.subscribe"

            # Peer drops the connection: the loop must reconnect and re-run
            # the handshake on the new connection.
            server.drop_connection()
            await _wait_for(lambda: handshakes == 2)
            assert server.connections == 2
            assert disconnects == 1  # the owner is told each session ended

            # disconnect() forces the same reconnect path from our side.
            conn.disconnect()
            await _wait_for(lambda: handshakes == 3)
        finally:
            await conn.stop()
            await server.stop()

        assert not conn.is_connected

    @pytest.mark.asyncio
    async def test_peer_close_mid_handshake_fails_fast_and_reconnects(self):
        # A clean peer close while the handshake awaits a response must fail
        # the pending call immediately (not dangle until the 30s RPC timeout,
        # wedging the reconnect loop).
        server = FakeStratumServer()
        server.drop_before_replying = 1
        port = await server.start()

        handshakes = 0

        async def handshake() -> None:
            nonlocal handshakes
            await conn.call("mining.subscribe", ["test/1.0"])
            handshakes += 1

        async def on_notification(method: str, params: list) -> None:
            pass

        conn = AsyncStratumConnection(
            "127.0.0.1",
            port,
            tls=False,
            reconnect_delay_seconds=0,
            max_reconnect_delay_seconds=1,
            handshake=handshake,
            on_notification=on_notification,
        )
        try:
            await conn.start()
            await _wait_for(lambda: handshakes == 1)  # second attempt succeeded
            assert server.connections == 2
        finally:
            await conn.stop()
            await server.stop()

    @pytest.mark.asyncio
    async def test_malformed_lines_do_not_kill_the_connection(self):
        server = FakeStratumServer()
        port = await server.start()

        handshakes = 0
        notifications: list[tuple[str, list]] = []

        async def handshake() -> None:
            nonlocal handshakes
            await conn.call("mining.subscribe", ["test/1.0"])
            handshakes += 1

        async def on_notification(method: str, params: list) -> None:
            notifications.append((method, params))

        conn = AsyncStratumConnection(
            "127.0.0.1",
            port,
            tls=False,
            reconnect_delay_seconds=0,
            max_reconnect_delay_seconds=1,
            handshake=handshake,
            on_notification=on_notification,
        )
        try:
            await conn.start()
            await _wait_for(lambda: handshakes == 1)

            for line in (
                b"42\n",  # valid JSON, not an object
                b'{"method": "mining.notify", "params": null}\n',
                b'{"method": "mining.notify", "params": {"job": 1}}\n',
                b'{"method": "mining.notify", "params": ["job-ok"]}\n',
            ):
                server._writer.write(line)
            await server._writer.drain()

            await _wait_for(lambda: notifications == [("mining.notify", ["job-ok"])])
            assert server.connections == 1  # nothing tore the connection down
            assert conn.is_connected
        finally:
            await conn.stop()
            await server.stop()

    @pytest.mark.asyncio
    async def test_call_without_connection_raises(self):
        async def handshake() -> None:
            pass

        async def on_notification(method: str, params: list) -> None:
            pass

        conn = AsyncStratumConnection(
            "127.0.0.1",
            1,
            tls=False,
            reconnect_delay_seconds=0,
            max_reconnect_delay_seconds=1,
            handshake=handshake,
            on_notification=on_notification,
        )
        with pytest.raises(ConnectionError):
            await conn.call("mining.subscribe", [])

    @pytest.mark.asyncio
    async def test_failed_handshake_retries_with_backoff(self):
        server = FakeStratumServer()
        port = await server.start()

        attempts = 0

        async def failing_then_ok_handshake() -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConnectionError("authorization failed")

        async def on_notification(method: str, params: list) -> None:
            pass

        conn = AsyncStratumConnection(
            "127.0.0.1",
            port,
            tls=False,
            reconnect_delay_seconds=0,
            max_reconnect_delay_seconds=1,
            handshake=failing_then_ok_handshake,
            on_notification=on_notification,
        )
        try:
            await conn.start()
            await _wait_for(lambda: attempts == 2)  # first failed, second succeeded
            assert server.connections == 2
            assert conn.is_connected
        finally:
            await conn.stop()
            await server.stop()
