"""Direct pearld JSON-RPC client with long-poll support.

`pearl-gateway.PearlNodeClient.get_block_template()` always polls — it doesn't
include `longpollid` in the request and doesn't surface it in the response.
That gives us a stale-share window of up to `poll_interval` seconds on every
chain advance.

This module talks to pearld directly via aiohttp so we can:
  - Pass `longpollid` to getblocktemplate, having pearld block until the
    template changes (or `timeout_secs` elapses).
  - Surface the new `longpollid` from the response so we can immediately
    re-arm the next long-poll call.

We reuse pearl-gateway's `BlockTemplate.from_get_block_template` + raw schema
for the dataclass conversion — no duplication of coinbase / witness commitment
assembly logic.

Behavior:
  - First call: no longpollid, returns immediately with the current template.
  - Subsequent calls: pass the previous longpollid, pearld holds the connection
    open until tip changes; we return as soon as a new template is available.
  - On HTTP error or RPC error: caller's responsibility to back off and retry.
    We don't bake in retry policy here so the poller stays in control.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

_LOGGER = logging.getLogger(__name__)


class PearldRpc:
    """Minimal async pearld JSON-RPC client. Use as `async with`."""

    def __init__(
        self,
        rpc_url: str,
        rpc_user: str,
        rpc_password: str,
        request_timeout_secs: float = 60.0,
    ):
        self.url = rpc_url
        self._auth = aiohttp.BasicAuth(rpc_user, rpc_password)
        self._request_timeout = aiohttp.ClientTimeout(total=request_timeout_secs)
        self._session: aiohttp.ClientSession | None = None
        self._req_id = 0

    async def __aenter__(self) -> "PearldRpc":
        # connector with no SSL verification by default — pearld on LAN uses
        # self-signed certs. For prod-grade TLS, set verify_ssl externally.
        connector = aiohttp.TCPConnector(ssl=False)
        self._session = aiohttp.ClientSession(
            auth=self._auth, timeout=self._request_timeout, connector=connector
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def call(self, method: str, params: list | None = None) -> Any:
        """Make a JSON-RPC call. Raises on HTTP or RPC error."""
        if self._session is None:
            raise RuntimeError("PearldRpc must be used inside `async with`")
        self._req_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or [],
            "id": self._req_id,
        }
        async with self._session.post(self.url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RpcError(f"pearld HTTP {resp.status}: {text[:200]}")
            data = await resp.json()
            if data.get("error"):
                raise RpcError(f"pearld RPC error: {data['error']}")
            return data["result"]


class RpcError(Exception):
    """Raised on HTTP or pearld-side RPC failure. Caller decides retry policy."""


class LongPollingTemplateFetcher:
    """Owns the longpollid handshake state. Returns raw template dicts; the
    caller converts to pearl-gateway's BlockTemplate.
    """

    def __init__(
        self,
        rpc: PearldRpc,
        long_poll_timeout_secs: float = 30.0,
    ):
        self.rpc = rpc
        self.long_poll_timeout = long_poll_timeout_secs
        self._longpollid: str | None = None

    async def fetch(self) -> dict:
        """Fetch the next template.

        First call: returns the current template immediately.
        Subsequent calls: long-poll — pearld holds connection open until tip
        changes OR until `long_poll_timeout_secs` elapses (whichever first).
        On timeout pearld still returns the (unchanged) template; the caller
        can de-dup on `previousblockhash`.
        """
        req: dict[str, Any] = {
            "capabilities": ["coinbasevalue", "workid", "coinbase/append", "longpoll"],
            "rules": ["segwit"],
        }
        if self._longpollid is not None:
            req["longpollid"] = self._longpollid

        template = await self.rpc.call("getblocktemplate", [req])

        # Update longpollid for the next call. Some pearld configs may not
        # return it (e.g., long-poll disabled); in that case we just keep
        # polling without it, and the caller's sleep-between-calls kicks in.
        new_lpid = template.get("longpollid")
        if new_lpid:
            self._longpollid = new_lpid
        else:
            _LOGGER.debug("pearld did not return longpollid; will keep polling")
            self._longpollid = None

        return template

    def reset(self) -> None:
        """Forget the cached longpollid. Use after an RPC error so the next
        call doesn't hang on a longpollid pearld no longer recognizes."""
        self._longpollid = None
