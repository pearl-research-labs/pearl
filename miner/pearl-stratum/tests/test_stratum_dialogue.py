"""End-to-end tests against an asyncio fake pool.

Covers:
  - mining.configure / subscribe / authorize sequence and pearl/v1 requirement
  - mining.notify parsing -> on_new_job callback fires
  - mining.set_difficulty -> stats.last_diff and on_set_difficulty
  - mining.submit accept path -> stats.accepted increments
  - mining.submit error code 21 -> StaleShareError reported, socket stays open,
    next submit on the same socket works (the alphafix.c bug-fix invariant)
  - subscribe rejection (server returns wrong shape) is fatal at handshake time
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from pearl_stratum.stratum_client import (
    HANDSHAKE_TIMEOUT_S,
    StratumClient,
    StratumProtocolError,
    parse_pool_url,
)


pytestmark = pytest.mark.asyncio


# ---- fake pool helper ------------------------------------------------------


class FakePool:
    """Bare-minimum stratum responder for tests.

    The scripted behavior is driven by self.script which is a list of coroutines
    appended by the test; on each client request the pool dispatches by method.
    """

    def __init__(
        self,
        *,
        accept_pearl_v1: bool = True,
        push_set_mining_params: bool = True,
        push_notify_after_authorize: bool = True,
        notify_params: list | None = None,
        submit_response: str = "accept",  # "accept" | "stale" | "low_diff" | "second_call_stale"
        diff_after_notify: float | None = 1048576.0,
    ) -> None:
        self.accept_pearl_v1 = accept_pearl_v1
        self.push_set_mining_params = push_set_mining_params
        self.push_notify_after_authorize = push_notify_after_authorize
        self.notify_params = notify_params or [
            "0000d446-3061",
            "46b849bae7551681283f02a20080cd3f0fd0dfad5e320b09b6af901291bfc554",
            "0000402054c5bf911290afb6090b325eaddfd00f3fcd8000a2023f28811655e7ba49b846d262d62ab2f3dbbf2ddd73a2c00a9ccd9838264c4298998096ef5602b0bfec3b6130096a99a00618",
            54342,
            "6a093061",
            "1a0ffff0",
            True,
        ]
        self.submit_response = submit_response
        self.diff_after_notify = diff_after_notify

        # Test inspection
        self.requests: list[dict] = []
        self.submit_count = 0
        self.connection_count = 0
        self.host = "127.0.0.1"
        self.port = 0  # set after start

        self._server: asyncio.base_events.Server | None = None
        self._connected = asyncio.Event()
        # The most recent client writer, so the test can shove unsolicited
        # frames at the client.
        self.last_writer: asyncio.StreamWriter | None = None
        self._handler_tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_client, host=self.host, port=0
        )
        sockets = self._server.sockets or ()
        if not sockets:
            raise RuntimeError("fake pool failed to bind")
        self.port = sockets[0].getsockname()[1]

    async def stop(self) -> None:
        # Close all open client writers first so handler readers see EOF
        # quickly (Windows IOCP can be lazy about delivering FIN).
        for task in self._handler_tasks:
            if not task.done():
                task.cancel()
        # Wait for handlers to exit (or finish their cancel).
        for task in self._handler_tasks:
            try:
                await asyncio.wait_for(task, timeout=2)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
        self._handler_tasks.clear()

        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=2)
            except asyncio.TimeoutError:
                pass
            self._server = None

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.connection_count += 1
        self.last_writer = writer
        # Register so stop() can cancel us promptly on Windows IOCP.
        self._handler_tasks.append(asyncio.current_task())  # type: ignore[arg-type]
        self._connected.set()
        try:
            while True:
                line = await reader.readline()
                if not line:
                    return
                msg = json.loads(line)
                self.requests.append(msg)
                method = msg.get("method")
                rid = msg.get("id")
                if method == "mining.configure":
                    if self.accept_pearl_v1:
                        await self._send(writer, {
                            "jsonrpc": "2.0", "id": rid,
                            "result": {"pearl/v1": True, "pearl/v1.share_format": "base64"},
                        })
                    else:
                        await self._send(writer, {
                            "jsonrpc": "2.0", "id": rid,
                            "result": {},
                        })
                elif method == "mining.subscribe":
                    await self._send(writer, {
                        "jsonrpc": "2.0", "id": rid,
                        "result": [
                            [["mining.set_difficulty", "conn-test"],
                             ["mining.notify", "conn-test"]],
                            "", 0,
                        ],
                    })
                elif method == "mining.authorize":
                    if self.push_set_mining_params:
                        await self._send(writer, {
                            "method": "pearl.set_mining_params",
                            "params": [{
                                "m": 131072, "n": 131072, "k": 4096, "rank": 128,
                                "rows_pattern": [0, 32],
                                "cols_pattern": list(range(64)),
                                "mma_type": "Int7xInt7ToInt32",
                            }],
                        })
                    await self._send(writer, {
                        "jsonrpc": "2.0", "id": rid, "result": True,
                    })
                    if self.diff_after_notify is not None:
                        await self._send(writer, {
                            "method": "mining.set_difficulty",
                            "params": [self.diff_after_notify],
                        })
                    if self.push_notify_after_authorize:
                        await self._send(writer, {
                            "method": "mining.notify",
                            "params": self.notify_params,
                        })
                elif method == "mining.submit":
                    self.submit_count += 1
                    if self.submit_response == "accept":
                        await self._send(writer, {"jsonrpc": "2.0", "id": rid, "result": True})
                    elif self.submit_response == "stale":
                        await self._send(writer, {
                            "jsonrpc": "2.0", "id": rid,
                            "error": [21, "chain advanced — share points to old block", None],
                        })
                    elif self.submit_response == "low_diff":
                        await self._send(writer, {
                            "jsonrpc": "2.0", "id": rid,
                            "error": [23, "Low difficulty share", None],
                        })
                    elif self.submit_response == "second_call_stale":
                        if self.submit_count == 1:
                            await self._send(writer, {
                                "jsonrpc": "2.0", "id": rid,
                                "error": [21, "chain advanced", None],
                            })
                        else:
                            await self._send(writer, {
                                "jsonrpc": "2.0", "id": rid, "result": True,
                            })
                elif method == "submitPlainProof":
                    self.submit_count += 1
                    await self._send(writer, {"jsonrpc": "2.0", "id": rid, "result": True})
                else:
                    # ignore unknown
                    pass
        except (ConnectionResetError, asyncio.IncompleteReadError, asyncio.CancelledError):
            return
        except Exception:  # pragma: no cover
            import traceback
            traceback.print_exc()
        finally:
            try:
                writer.close()
            except Exception:
                pass

    @staticmethod
    async def _send(writer: asyncio.StreamWriter, obj: dict) -> None:
        writer.write((json.dumps(obj) + "\n").encode("utf-8"))
        await writer.drain()


# ---- fixtures --------------------------------------------------------------


async def _spin_client(
    pool: FakePool,
    *,
    notify_event: asyncio.Event | None = None,
    on_new_job=None,
    on_set_difficulty=None,
) -> tuple[StratumClient, asyncio.Task]:
    """Build and start a client connected to `pool`."""
    callbacks = {}
    if on_new_job is not None:
        callbacks["on_new_job"] = on_new_job
    if on_set_difficulty is not None:
        callbacks["on_set_difficulty"] = on_set_difficulty

    client = StratumClient(
        host=pool.host,
        port=pool.port,
        address="prl1testtesttest",
        worker="testworker",
        password="x",
        **callbacks,
    )
    task = asyncio.create_task(client.run())
    # Wait until first job is parsed (server pushed it post-authorize).
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if client.current_job is not None:
            break
        await asyncio.sleep(0.01)
    return client, task


# ---- tests -----------------------------------------------------------------


async def test_handshake_sequence_in_order() -> None:
    pool = FakePool()
    await pool.start()
    try:
        client, task = await _spin_client(pool)
        try:
            assert client.connected
            # The first three requests must be configure, subscribe, authorize, in order.
            methods = [r["method"] for r in pool.requests[:3]]
            assert methods == ["mining.configure", "mining.subscribe", "mining.authorize"]
            # configure params: [["pearl/v1"], {}]
            cfg = pool.requests[0]
            assert cfg["params"] == [["pearl/v1"], {}]
            # subscribe params: [user_agent]
            sub = pool.requests[1]
            assert sub["params"] == ["pearl-stratum/0.1"]
            # authorize params: [worker_name, password]
            auth = pool.requests[2]
            assert auth["params"] == ["prl1testtesttest.testworker", "x"]
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pool.stop()


async def test_mining_notify_dispatches_callback() -> None:
    pool = FakePool()
    await pool.start()
    received: list = []
    try:
        client, task = await _spin_client(pool, on_new_job=received.append)
        try:
            assert len(received) == 1
            assert received[0].job_id == "0000d446-3061"
            assert client.current_job is not None
            assert client.current_job.job_id == "0000d446-3061"
            assert client.mining_params is not None
            assert client.mining_params["m"] == 131072
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pool.stop()


async def test_set_difficulty_updates_stats_and_callback() -> None:
    pool = FakePool(diff_after_notify=2048.0)
    await pool.start()
    diffs: list[float] = []
    try:
        client, task = await _spin_client(pool, on_set_difficulty=diffs.append)
        try:
            # Wait briefly for set_difficulty to be processed.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not diffs:
                await asyncio.sleep(0.01)
            assert diffs == [2048.0]
            assert client.stats.last_diff == 2048.0
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pool.stop()


async def test_submit_share_accept_path() -> None:
    pool = FakePool()
    await pool.start()
    try:
        client, task = await _spin_client(pool)
        try:
            result = await client.submit_share("0000d446-3061", "AAAA==")
            assert result.accepted is True
            assert result.latency_ms > 0
            assert client.stats.accepted == 1
            assert client.stats.dropped_stale_jobid == 0
            # Submit frame on the wire was positional.
            sub = next(r for r in pool.requests if r.get("method") == "mining.submit")
            assert sub["params"] == ["prl1testtesttest.testworker", "0000d446-3061", "AAAA=="]
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pool.stop()


async def test_error21_does_not_reconnect() -> None:
    """The alphafix.c invariant: error 21 reports as stale, socket stays open."""
    pool = FakePool(submit_response="second_call_stale")
    await pool.start()
    try:
        client, task = await _spin_client(pool)
        try:
            r1 = await client.submit_share("0000d446-3061", "AAAA==")
            assert r1.accepted is False
            assert r1.error_code == 21
            assert client.stats.dropped_stale_jobid == 1
            assert client.stats.accepted == 0

            # The next submit on the SAME socket must work — proving we didn't reconnect.
            assert pool.connection_count == 1
            r2 = await client.submit_share("0000d446-3061", "BBBB==")
            assert r2.accepted is True
            assert pool.connection_count == 1, (
                "client must NOT reconnect on error 21 — "
                f"saw {pool.connection_count} connections"
            )
            assert client.stats.accepted == 1
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pool.stop()


async def test_submit_low_diff_rejected_distinctly_from_stale() -> None:
    pool = FakePool(submit_response="low_diff")
    await pool.start()
    try:
        client, task = await _spin_client(pool)
        try:
            result = await client.submit_share("0000d446-3061", "AAAA==")
            assert result.accepted is False
            assert result.error_code == 23
            assert client.stats.rejected == 1
            assert client.stats.dropped_stale_jobid == 0
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pool.stop()


async def test_pearl_v1_rejection_fails_handshake() -> None:
    """Pool that doesn't accept pearl/v1 must trip a StratumProtocolError."""
    pool = FakePool(accept_pearl_v1=False)
    await pool.start()
    try:
        client = StratumClient(
            host=pool.host, port=pool.port,
            address="prl1testtesttest", worker="testworker", password="x",
        )
        # We can't easily inject a "stop the run-loop on first failure" without
        # special wiring; instead verify the handshake call directly via _handshake.
        # The full run-loop would attempt reconnects, which is correct production
        # behavior but inconvenient to assert on.
        reader, writer = await asyncio.open_connection(pool.host, pool.port)
        client._reader = reader
        client._writer = writer
        try:
            with pytest.raises(StratumProtocolError, match="pearl/v1"):
                await client._handshake()
        finally:
            writer.close()
            await writer.wait_closed()
    finally:
        await pool.stop()


async def test_client_reconnect_method_drops_socket() -> None:
    """`client.reconnect` from the pool should close the socket; the run-loop reconnects."""
    pool = FakePool()
    await pool.start()
    try:
        client, task = await _spin_client(pool)
        try:
            assert pool.connection_count == 1
            # Push client.reconnect at the client.
            assert pool.last_writer is not None
            pool.last_writer.write(
                (json.dumps({"method": "client.reconnect", "params": []}) + "\n").encode()
            )
            await pool.last_writer.drain()
            # Wait for reconnection.
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and pool.connection_count < 2:
                await asyncio.sleep(0.05)
            assert pool.connection_count >= 2
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=3)
    finally:
        await pool.stop()


async def test_malformed_json_ignored_gracefully() -> None:
    pool = FakePool()
    await pool.start()
    try:
        client, task = await _spin_client(pool)
        try:
            # Send garbage on the socket; the client must keep working.
            assert pool.last_writer is not None
            pool.last_writer.write(b"not json at all\n")
            pool.last_writer.write(b"{not closed json\n")
            await pool.last_writer.drain()
            # Subsequent submit still works.
            result = await client.submit_share("0000d446-3061", "AAAA==")
            assert result.accepted is True
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pool.stop()


