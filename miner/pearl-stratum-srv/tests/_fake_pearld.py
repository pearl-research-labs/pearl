"""Shared fake pearld HTTP server. Imported by test_node_rpc and
test_long_poll_integration. Lives outside the test_*.py glob so pytest
doesn't try to collect tests from it.
"""

from __future__ import annotations

import asyncio
import json


class FakePearld:
    """Minimal HTTP/1.1 JSON-RPC responder simulating pearld's getblocktemplate."""

    def __init__(self):
        self.received: list[dict] = []
        self.next_response: dict | None = None
        self._block_event = asyncio.Event()
        self._block_event.set()  # default: don't block
        self.host = "127.0.0.1"
        self.port = 0
        self._server: asyncio.base_events.Server | None = None
        self.template = {
            "version": 0x20000000,
            "previousblockhash": "ab" * 32,
            "merkleroot": "00" * 32,
            "transactions": [],
            "coinbaseaux": {"flags": "2f503253482f706561726c642f"},
            "coinbasevalue": 275_039_000_000_000,
            "longpollid": "lpid-v1",
            "target": "0" * 8 + "f" * 56,
            "mintime": 0,
            "mutable": ["time", "transactions", "prevblock"],
            "noncerange": "00000000ffffffff",
            "vsizelimit": 4_000_000,
            "curtime": 1_778_987_105,
            "bits": "1a0ffff0",
            "height": 54_374,
            "default_witness_commitment": "6a24aa21a9ed" + "00" * 32,
        }

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, host=self.host, port=0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not request_line:
                return
            content_length = 0
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b""):
                    break
                if line.lower().startswith(b"content-length:"):
                    content_length = int(line.split(b":", 1)[1].strip())
            body = await reader.readexactly(content_length) if content_length else b""
            payload = json.loads(body)
            self.received.append(payload)
            await self._block_event.wait()
            response = self.next_response or {
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "result": dict(self.template),
                "error": None,
            }
            body_out = json.dumps(response).encode("utf-8")
            head = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(body_out)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii")
            writer.write(head + body_out)
            await writer.drain()
        except Exception:
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    def block_long_poll(self) -> None:
        self._block_event.clear()

    def release_long_poll(self) -> None:
        self._block_event.set()
