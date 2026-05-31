"""JSON-RPC line parsing + error-21 detection.

The alpha-miner stratum dialect is line-delimited JSON (`\n` framing). Each
line is a single JSON-RPC 1.0/2.0 object. We need to:

1. Frame raw bytes into complete lines (handle partial reads / TCP coalescing).
2. Parse each line as JSON.
3. Classify it: request | response | notification, and for responses extract
   `id`, `result`, `error` per JSON-RPC.
4. Spot stale-share rejections (``error[21]``) so the proxy can take action
   *before* the alpha-miner's FIN reaches the upstream socket.

Wire reference: ``C:/Source/pearl-investigation/STRATUM_CAPTURE.md`` §3.

This module is **pure** (no I/O, no asyncio). The proxy layer calls into it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Sentinel for the stale-share error code from alphapool.
# See STRATUM_CAPTURE.md §3i:
#   {"jsonrpc":"2.0","id":45,"error":[21,"chain advanced — share points to old block",null]}
ERROR_STALE_SHARE = 21


@dataclass(slots=True)
class StratumMessage:
    """A parsed JSON-RPC message from either direction.

    Exactly one of (request, response, notification) is true. We don't
    sub-class because dispatch is small and a single dataclass keeps the
    proxy hot-path branch-friendly.
    """

    raw: bytes
    """Original line including trailing ``\\n`` — kept so we can forward
    byte-identical to the other side (preserves whatever whitespace /
    field-ordering the pool happens to use)."""

    obj: dict[str, Any]
    """Parsed JSON object."""

    # ------------------------------------------------------------------
    # JSON-RPC classification (cached on parse so the hot path is O(1)).

    msg_id: int | str | None = None
    """Top-level ``id``. None means notification."""

    method: str | None = None
    """Top-level ``method``. None means response."""

    error_code: int | None = None
    """If this is an error response, the error code (first element of
    ``error`` array). None otherwise. JSON-RPC 1.0 style errors-as-arrays
    are what alphapool uses per §3i."""

    @property
    def is_request(self) -> bool:
        return self.method is not None and self.msg_id is not None

    @property
    def is_notification(self) -> bool:
        return self.method is not None and self.msg_id is None

    @property
    def is_response(self) -> bool:
        return self.method is None and self.msg_id is not None

    @property
    def is_stale_share_error(self) -> bool:
        return self.error_code == ERROR_STALE_SHARE


def _classify(obj: dict[str, Any]) -> tuple[int | str | None, str | None, int | None]:
    """Extract (id, method, error_code) from a parsed JSON-RPC object."""
    msg_id = obj.get("id")  # may be None for notifications
    method = obj.get("method")
    err = obj.get("error")
    error_code: int | None = None
    # alphapool uses JSON-RPC 1.0 array errors: [code, message, data].
    # Be tolerant of the 2.0 object form {"code": ..., "message": ...} too.
    if isinstance(err, list) and err and isinstance(err[0], int):
        error_code = err[0]
    elif isinstance(err, dict):
        code = err.get("code")
        if isinstance(code, int):
            error_code = code
    return msg_id, method, error_code


def parse_line(raw: bytes) -> StratumMessage:
    """Parse a single JSON-RPC line (with or without trailing newline).

    Raises ``json.JSONDecodeError`` if the line isn't valid JSON. Callers
    in the proxy hot-path catch this and forward the bytes anyway (better
    to be a pass-through than to drop messages we don't understand).
    """
    stripped = raw.rstrip(b"\r\n")
    obj = json.loads(stripped)
    if not isinstance(obj, dict):
        raise json.JSONDecodeError("top-level JSON must be an object", stripped.decode("utf-8", "replace"), 0)
    msg_id, method, error_code = _classify(obj)
    # Normalise framing: ensure outgoing bytes have exactly one trailing \n.
    framed = stripped + b"\n"
    return StratumMessage(raw=framed, obj=obj, msg_id=msg_id, method=method, error_code=error_code)


@dataclass(slots=True)
class LineFramer:
    """Accumulates bytes from a stream and yields complete JSON-RPC lines.

    Stratum is plain line-delimited JSON. We can't assume reads correspond
    to message boundaries — one read might contain three messages and half
    of a fourth.
    """

    _buf: bytearray = field(default_factory=bytearray)

    def feed(self, chunk: bytes) -> list[bytes]:
        """Append ``chunk`` to the buffer and return any complete lines.

        Each returned element ends with ``\\n``. The trailing fragment (if
        any) is kept for the next ``feed()`` call.
        """
        if not chunk:
            return []
        self._buf.extend(chunk)
        out: list[bytes] = []
        while True:
            nl = self._buf.find(b"\n")
            if nl < 0:
                break
            line = bytes(self._buf[: nl + 1])
            del self._buf[: nl + 1]
            out.append(line)
        return out

    def pending(self) -> bytes:
        """Return any unframed trailing bytes. Used at connection close to
        salvage a message that didn't end in ``\\n`` (none observed in the
        capture, but defensive)."""
        return bytes(self._buf)


def classify_response_id_to_method(
    pending_requests: dict[int | str, str],
    msg: StratumMessage,
) -> str | None:
    """Look up which method a response correlates to.

    JSON-RPC responses don't include the method, only the ``id``. The proxy
    keeps a small map of in-flight request ids -> method names so it can
    notice that an error[21] is a response to ``mining.submit`` and treat
    accordingly. Pops the entry from ``pending_requests`` on match.
    """
    if not msg.is_response or msg.msg_id is None:
        return None
    return pending_requests.pop(msg.msg_id, None)
