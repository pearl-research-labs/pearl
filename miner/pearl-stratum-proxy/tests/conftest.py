"""Shared pytest fixtures: mock alphapool TCP server.

The mock server speaks just enough of the pearl/v1 stratum dialect for
the proxy to exercise its short-circuit / replay / swallow-close paths.
Each test gets a fresh server instance on a random ephemeral port.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import pytest
import pytest_asyncio


logging.basicConfig(level=logging.DEBUG)


# Canonical pool responses, structurally matching STRATUM_CAPTURE.md §3.
def make_configure_response(req_id: int | str) -> bytes:
    return (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"pearl/v1": True, "pearl/v1.share_format": "base64"},
            }
        )
        + "\n"
    ).encode("utf-8")


def make_subscribe_response(req_id: int | str, session_id: str = "conn-test-1") -> bytes:
    return (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": [
                    [
                        ["mining.set_difficulty", session_id],
                        ["mining.notify", session_id],
                    ],
                    "",
                    0,
                ],
            }
        )
        + "\n"
    ).encode("utf-8")


def make_authorize_response(req_id: int | str, ok: bool = True) -> bytes:
    return (
        json.dumps({"jsonrpc": "2.0", "id": req_id, "result": ok}) + "\n"
    ).encode("utf-8")


def make_submit_ok(req_id: int | str) -> bytes:
    return make_authorize_response(req_id, ok=True)


def make_submit_stale_error(req_id: int | str) -> bytes:
    return (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": [21, "chain advanced - share points to old block", None],
            }
        )
        + "\n"
    ).encode("utf-8")


def make_set_mining_params() -> bytes:
    # Byte-identical across reconnects per STRATUM_CAPTURE §3c.
    return (
        json.dumps(
            {
                "method": "pearl.set_mining_params",
                "params": [
                    {
                        "m": 131072,
                        "n": 131072,
                        "k": 4096,
                        "rank": 128,
                    }
                ],
            }
        )
        + "\n"
    ).encode("utf-8")


def make_set_difficulty(diff: int = 1048576) -> bytes:
    return (
        json.dumps({"method": "mining.set_difficulty", "params": [diff]}) + "\n"
    ).encode("utf-8")


def make_notify(job_id: str = "0000d446-3061") -> bytes:
    return (
        json.dumps(
            {
                "method": "mining.notify",
                "params": [job_id, "prevhash", "coinbase", 54342, "nbits", "ntime", True],
            }
        )
        + "\n"
    ).encode("utf-8")


# ----------------------------------------------------------------------
# Mock pool server.


@dataclass
class PoolConnection:
    """Tracks one inbound client (proxy) -> pool TCP connection."""

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    received: list[dict[str, Any]] = field(default_factory=list)
    """Every parsed inbound JSON-RPC object, in order received."""


class MockPool:
    """In-process stand-in for ``us2.alphapool.tech:5566``.

    The default behaviour matches the captured pool dialogue from
    STRATUM_CAPTURE.md. Tests can override responses by setting hooks.
    """

    def __init__(self) -> None:
        self.host: str = "127.0.0.1"
        self.port: int = 0  # filled in by start()
        self._server: asyncio.base_events.Server | None = None
        self.connections: list[PoolConnection] = []

        # Pluggable hook: called after auth-OK with (PoolConnection, auth_msg).
        # Default sends set_mining_params, set_difficulty, notify.
        self.on_post_auth: Callable[
            [PoolConnection, dict[str, Any]], Awaitable[None]
        ] = self._default_post_auth

        # Pluggable hook: response to mining.submit. Default OK.
        self.submit_handler: Callable[
            [PoolConnection, dict[str, Any]], Awaitable[bytes]
        ] = self._default_submit_handler

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._on_conn, host=self.host, port=0
        )
        sockets = self._server.sockets or ()
        if sockets:
            self.port = sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for conn in self.connections:
            try:
                conn.writer.close()
                await conn.writer.wait_closed()
            except Exception:
                pass

    async def push(self, conn: PoolConnection, line: bytes) -> None:
        """Send an arbitrary line to a connected client (used in tests
        to simulate unsolicited notifies)."""
        conn.writer.write(line)
        await conn.writer.drain()

    # -----------------------------------------------------------------
    # Default behaviour mimicking the captured pool.

    async def _default_post_auth(
        self, conn: PoolConnection, auth_msg: dict[str, Any]
    ) -> None:
        # Per §3 ordering: set_mining_params (unsolicited) -> auth response ->
        # set_difficulty -> first notify. We send mining_params first.
        await self.push(conn, make_set_mining_params())
        await self.push(conn, make_authorize_response(auth_msg["id"]))
        await self.push(conn, make_set_difficulty())
        await self.push(conn, make_notify())

    async def _default_submit_handler(
        self, conn: PoolConnection, msg: dict[str, Any]
    ) -> bytes:
        return make_submit_ok(msg["id"])

    # -----------------------------------------------------------------
    # Wire loop.

    async def _on_conn(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        conn = PoolConnection(reader=reader, writer=writer)
        self.connections.append(conn)
        from pearl_stratum_proxy.error21_interceptor import LineFramer

        framer = LineFramer()
        try:
            while True:
                try:
                    chunk = await reader.read(65536)
                except Exception:
                    break
                if not chunk:
                    break
                for line in framer.feed(chunk):
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    conn.received.append(obj)
                    await self._dispatch(conn, obj)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _dispatch(self, conn: PoolConnection, obj: dict[str, Any]) -> None:
        method = obj.get("method")
        msg_id = obj.get("id")
        if method == "mining.configure":
            await self.push(conn, make_configure_response(msg_id))
        elif method == "mining.subscribe":
            await self.push(conn, make_subscribe_response(msg_id))
        elif method == "mining.authorize":
            await self.on_post_auth(conn, obj)
        elif method == "mining.submit":
            reply = await self.submit_handler(conn, obj)
            await self.push(conn, reply)


@pytest_asyncio.fixture
async def mock_pool() -> "MockPool":  # type: ignore[type-arg]
    pool = MockPool()
    await pool.start()
    try:
        yield pool
    finally:
        await pool.stop()
