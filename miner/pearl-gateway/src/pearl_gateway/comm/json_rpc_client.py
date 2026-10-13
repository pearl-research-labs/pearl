import json
import socket
from contextlib import suppress
from typing import Any, TextIO

from miner_utils import get_logger
from pearl_gateway.config import MinerRpcConfig

logger = get_logger(__name__)


class JSONRPCClient:
    """Simple JSON-RPC client for newline-terminated messages"""

    def __init__(self, gateway_params: MinerRpcConfig):
        self._socket = self._connect_socket(gateway_params)

        self._reader: TextIO = self._socket.makefile("r", encoding="utf-8")
        self._writer: TextIO = self._socket.makefile("w", encoding="utf-8")
        self._request_id = 0

    def _connect_socket(self, gateway_params: MinerRpcConfig) -> socket.socket:
        """Establish connection based on MinerRpcConfig"""
        try:
            if gateway_params.transport == "tcp":
                _socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                host = gateway_params.host
                port = gateway_params.port
                _socket.connect((host, port))
                logger.info(f"Connected to {host}:{port}")
            else:  # UDS (Unix Domain Socket)
                _socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                _socket.connect(gateway_params.socket_path)
                logger.info(f"Connected to {gateway_params.socket_path}")
        except Exception as e:
            logger.warning(f"Failed to connect: {e}")
            raise
        return _socket

    def close(self):
        """Close the file objects and socket.

        The socket is shut down first so a ``recv_message`` blocked in another
        thread wakes up instead of holding the connection open.
        """
        with suppress(OSError):
            self._socket.shutdown(socket.SHUT_RDWR)
        self._reader.close()
        self._writer.close()
        self._socket.close()

    @property
    def is_connected(self) -> bool:
        """Check if connection is active"""
        return self._socket is not None and self._socket.fileno() != -1

    def send_message(self, message: dict[str, Any]) -> None:
        """Send a JSON message"""
        json_str = json.dumps(message)
        self._writer.write(json_str + "\n")
        self._writer.flush()

    def recv_message(self) -> dict[str, Any]:
        """Receive a single JSON message (blocks until message received)"""
        line = self._reader.readline()
        if not line:
            raise ConnectionError("Connection closed by remote host")
        return json.loads(line.strip())

    def call(self, method: str, params: Any = None) -> Any:
        """Make a JSON-RPC request and return the result"""
        self._request_id += 1

        request = {"jsonrpc": "2.0", "method": method, "id": self._request_id}
        request["params"] = params if params is not None else {}

        # Send request
        self.send_message(request)

        # Receive response
        response = self.recv_message()

        # Check for errors
        if "error" in response:
            error = response["error"]
            raise Exception(
                f"JSON-RPC error {error.get('code')}: {error.get('message')}",
            )

        # Verify response ID
        if response.get("id") != self._request_id:
            raise Exception(
                f"Response ID mismatch: expected {self._request_id}, got {response.get('id')}",
            )

        return response.get("result")
