"""Stratum v1 client transport for the gateway's pool mode.

- URL dialect: stratum+ssl:// and stratum+tls:// mean TLS, stratum+tcp:// and
  stratum:// mean plaintext; the port defaults to 3333.
- AsyncJSONRPCClient multiplexes requests/responses over one stream.
- AsyncStratumConnection owns connect (TLS or TCP), reconnect with backoff, and
  the listen loop that routes responses to pending calls and notifications to
  the owner.
"""

import asyncio
import contextlib
import json
import ssl
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

from miner_utils import get_logger

logger = get_logger(__name__)

DEFAULT_STRATUM_PORT = 3333


def stratum_host(url: str) -> str | None:
    """The endpoint's hostname, or None when the URL has no host."""
    return urlsplit(url).hostname


def stratum_port(url: str) -> int:
    return urlsplit(url).port or DEFAULT_STRATUM_PORT


def stratum_uses_tls(url: str) -> bool:
    return urlsplit(url).scheme.endswith(("ssl", "tls"))


# The extranonce lives in the coinbase scriptSig (≤100 bytes alongside the
# height push and pool flags), so 32 is already generous. A larger value is
# either a buggy or hostile pool; clamping to 0 disables the feature rather
# than letting it force huge allocations downstream.
MAX_EXTRANONCE2_SIZE = 32


def parse_extranonce2_size(value: Any) -> int:
    """Return a safe extranonce2 byte size, or 0 (feature off) for anything
    out of range. JSON ints arrive as Python ``int``; anything else means the
    pool is broken, so treat it as feature-off rather than guessing."""
    # type() is intentional: bool/float/str are all "broken pool" → 0.
    if type(value) is not int:
        return 0
    if not 0 <= value <= MAX_EXTRANONCE2_SIZE:
        logger.warning(
            f"Pool advertised extranonce2_size={value}, ignoring "
            f"(valid range 0..{MAX_EXTRANONCE2_SIZE})"
        )
        return 0
    return value


def build_request(request_id: int, method: str, params: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": method, "id": request_id, "params": params}


class JSONRPCError(Exception):
    def __init__(self, error: Any) -> None:
        self.error = error
        if isinstance(error, dict):
            code = error.get("code")
            message = error.get("message", str(error))
        elif isinstance(error, list | tuple):
            code = error[0] if len(error) > 0 else None
            message = error[1] if len(error) > 1 else str(error)
        else:
            code = None
            message = str(error)
        super().__init__(f"JSON-RPC error {code}: {message}")


class AsyncJSONRPCClient:
    """Async JSON-RPC request helper for an existing newline-delimited stream."""

    def __init__(
        self,
        writer: asyncio.StreamWriter,
        *,
        request_timeout_seconds: int = 30,
    ) -> None:
        self._writer = writer
        self._request_timeout_seconds = request_timeout_seconds
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future[Any]] = {}

    @staticmethod
    def is_response_message(message: dict[str, Any]) -> bool:
        return "id" in message and message["id"] is not None and "method" not in message

    async def call(self, method: str, params: Any = None) -> Any:
        """Send a JSON-RPC request and wait for its matching response."""
        self._request_id += 1
        request_id = self._request_id
        payload = (
            json.dumps(
                build_request(
                    request_id,
                    method,
                    params if params is not None else [],
                )
            )
            + "\n"
        )

        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending_requests[request_id] = future

        try:
            self._writer.write(payload.encode())
            await self._writer.drain()
        except Exception:
            self._pending_requests.pop(request_id, None)
            raise

        try:
            return await asyncio.wait_for(future, timeout=self._request_timeout_seconds)
        except TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise TimeoutError(f"JSON-RPC call {method} timed out") from None
        except asyncio.CancelledError:
            self._pending_requests.pop(request_id, None)
            raise

    def handle_response(self, message: dict[str, Any]) -> None:
        """Resolve a pending request from a JSON-RPC response message."""
        request_id = message["id"]
        future = self._pending_requests.pop(request_id, None)
        if future is None or future.done():
            return
        if message.get("error"):
            future.set_exception(JSONRPCError(message["error"]))
        else:
            future.set_result(message.get("result"))

    def fail_all_pending(self, reason: str) -> None:
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(ConnectionError(reason))
        self._pending_requests.clear()


# Bound a single Stratum line in either direction. A ZKCertificate submit is
# ~80 KB base16; 512 KB leaves generous headroom while capping what one line
# can allocate.
READER_BUFFER_LIMIT = 2**19

# Runs after every (re)connect, with the listener already routing responses,
# so it can issue calls through the connection. Raising aborts the attempt
# and the connection loop reconnects with backoff.
HandshakeCallback = Callable[[], Awaitable[None]]

# Receives every server-initiated notification as (method, raw params).
NotificationCallback = Callable[[str, list[Any]], Awaitable[None]]


class AsyncStratumConnection:
    """Persistent newline-delimited JSON-RPC connection with reconnect.

    start() launches the connection loop: connect, run the owner's handshake,
    then stay connected until the peer drops or an error occurs, reconnecting
    with exponential backoff.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        tls: bool,
        reconnect_delay_seconds: int,
        max_reconnect_delay_seconds: int,
        handshake: HandshakeCallback,
        on_notification: NotificationCallback,
    ) -> None:
        self._host = host
        self._port = port
        self._tls = tls
        self._reconnect_delay_seconds = reconnect_delay_seconds
        self._max_reconnect_delay_seconds = max_reconnect_delay_seconds
        self._handshake = handshake
        self._on_notification = on_notification

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._rpc_client: AsyncJSONRPCClient | None = None
        self._loop_task: asyncio.Task | None = None
        self._running = False

    @property
    def is_connected(self) -> bool:
        return self._running and self._writer is not None and not self._writer.is_closing()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._connection_loop())

    async def stop(self) -> None:
        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None
        await self._close_connection()

    async def call(self, method: str, params: list[Any]) -> Any:
        """Send a JSON-RPC request and wait for the matching response."""
        if self._rpc_client is None:
            raise ConnectionError("Stratum connection is not established")
        return await self._rpc_client.call(method, params)

    def disconnect(self) -> None:
        """Drop the current connection; the connection loop reconnects."""
        if self._writer is not None and not self._writer.is_closing():
            self._writer.close()

    async def _connection_loop(self) -> None:
        backoff = self._reconnect_delay_seconds

        while self._running:
            try:
                await self._connect()
                # The listener starts first so handshake calls can have their
                # responses routed; the group then holds the connection open
                # until the listener returns (peer closed) or either side
                # fails, which cancels the other.
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._listen())
                    await self._handshake()
                    backoff = self._reconnect_delay_seconds
            except* Exception:
                logger.exception("Pool connection error, reconnecting")
            finally:
                await self._close_connection()

            if not self._running:
                break

            logger.info(f"Reconnecting to pool in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._max_reconnect_delay_seconds)

    async def _connect(self) -> None:
        ssl_ctx = ssl.create_default_context() if self._tls else None
        server_hostname = self._host if self._tls else None
        transport = "TLS" if self._tls else "plain TCP"

        logger.info(f"Connecting to pool at {self._host}:{self._port} ({transport})")
        self._reader, self._writer = await asyncio.open_connection(
            self._host,
            self._port,
            ssl=ssl_ctx,
            server_hostname=server_hostname,
            limit=READER_BUFFER_LIMIT,
        )
        logger.info(f"{transport} connection established")
        self._rpc_client = AsyncJSONRPCClient(self._writer)

    async def _close_connection(self) -> None:
        if self._rpc_client is not None:
            self._rpc_client.fail_all_pending("connection closed")
        if self._writer is not None and not self._writer.is_closing():
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
        self._writer = None
        self._reader = None
        self._rpc_client = None

    async def _listen(self) -> None:
        """Read lines from the peer and dispatch responses / notifications."""
        assert self._reader is not None
        while True:
            line = await self._reader.readline()
            if not line:
                logger.warning("Pool connection closed by remote")
                if self._rpc_client is not None:
                    # A handshake awaiting a response in the parent task would
                    # otherwise dangle until its RPC timeout: the TaskGroup
                    # only exits (and cleanup only runs) once that await
                    # resolves, and nobody else fails it on a clean EOF.
                    self._rpc_client.fail_all_pending("connection closed by remote")
                return

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Ignoring non-JSON line from pool: {line!r}")
                continue
            if not isinstance(msg, dict):
                logger.warning(f"Ignoring non-object message from pool: {line!r}")
                continue

            if AsyncJSONRPCClient.is_response_message(msg):
                if self._rpc_client is not None:
                    self._rpc_client.handle_response(msg)
            elif "method" in msg:
                await self._dispatch_notification(msg, line)
            else:
                logger.debug(f"Unhandled pool message: {msg}")

    async def _dispatch_notification(self, msg: dict[str, Any], line: bytes) -> None:
        params = msg.get("params", [])
        if not isinstance(params, list):
            # Stratum params are positional; an explicit null/object payload
            # would blow up the owner's handlers.
            logger.warning(f"Ignoring notification with non-list params: {line!r}")
            return
        try:
            await self._on_notification(msg["method"], params)
        except Exception:
            # A bad message would otherwise drop the connection, and the pool
            # replays it on every reconnect: log it and keep reading.
            logger.exception(f"Ignoring pool notification that failed to handle: {line!r}")
