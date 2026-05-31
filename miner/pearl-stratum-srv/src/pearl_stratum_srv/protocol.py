"""Stratum-v1 JSON-RPC framing.

Server side mirrors the wire format alphapool serves (see
C:/Source/pearl-investigation/STRATUM_CAPTURE.md and the client-side
encoder in pearl_stratum.stratum_client lines 675-795).

Key invariants from observed alphapool behaviour:
  - Each frame is a single JSON object terminated by '\\n'.
  - Server REPLIES include "jsonrpc": "2.0"; server PUSHES (notify,
    set_difficulty, set_mining_params) typically omit jsonrpc and id.
  - Errors are 3-tuples: [code, message, null].
  - Stale share (chain advanced) is code 21, message "Job not found".
    Socket MUST stay open; miner must keep submitting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

STALE_SHARE_CODE = 21
LOW_DIFF_CODE = 23
UNKNOWN_METHOD_CODE = 25
INVALID_PARAMS_CODE = -32602


@dataclass(slots=True)
class Request:
    """An inbound JSON-RPC request from a miner."""

    id: Any
    method: str
    params: Any


def encode_response(req_id: Any, result: Any) -> bytes:
    """Successful reply to a client request."""
    payload = {"jsonrpc": "2.0", "id": req_id, "result": result, "error": None}
    return _encode(payload)


def encode_error(req_id: Any, code: int, message: str) -> bytes:
    """Error reply. error is the 3-tuple [code, message, null] alphapool uses."""
    payload = {"jsonrpc": "2.0", "id": req_id, "result": None, "error": [code, message, None]}
    return _encode(payload)


def encode_notification(method: str, params: Any) -> bytes:
    """Server push (notify / set_difficulty / set_mining_params)."""
    payload = {"id": None, "method": method, "params": params}
    return _encode(payload)


def _encode(payload: dict) -> bytes:
    # separators=(",", ":") to match alphapool's tight encoding; trailing \n.
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def parse_request(line: bytes) -> Request:
    """Parse one line of inbound JSON.

    Raises ValueError on malformed input; callers should reply with code -32602
    and keep the connection open (miner may recover by re-sending).
    """
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed JSON: {e}") from e

    if not isinstance(obj, dict):
        raise ValueError(f"frame must be a JSON object, got {type(obj).__name__}")

    method = obj.get("method")
    if not isinstance(method, str):
        raise ValueError("frame missing 'method' string")

    return Request(id=obj.get("id"), method=method, params=obj.get("params", []))
