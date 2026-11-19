"""JSON-RPC client for talking to pearl-gateway over UDS or TCP."""

from __future__ import annotations

import json
import socket
from contextlib import AbstractContextManager, suppress
from types import TracebackType
from typing import Any, TextIO

from miner_utils import get_logger
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.dataclasses import MiningJob
from pearl_gateway.config import MinerRpcConfig
from pearl_mining import BlockHeader, IncompleteBlockHeader, PlainProofV4

from .protocol import bits_to_target

_LOGGER = get_logger(__name__)

# Valid 76-byte header + linked parent for MINER_NO_GATEWAY offline runs;
# hard difficulty so benchmarks do not constantly hit the lottery.
_DUMMY_NBITS = 0x1D00FFFF
_DUMMY_PARENT = BlockHeader(
    IncompleteBlockHeader(
        version=1,
        prev_block=b"\xde\xad\xba\xbe" * 8,
        merkle_root=b"\x00" * 32,
        timestamp=0,
        nbits=_DUMMY_NBITS,
    ),
    bytes(32),
)
_DUMMY_HEADER_BYTES = bytes(
    IncompleteBlockHeader(
        version=1,
        prev_block=bytes(_DUMMY_PARENT.block_hash()),
        merkle_root=b"\x00" * 32,
        timestamp=0,
        nbits=_DUMMY_NBITS,
    ).to_bytes()
)


class JSONRPCClient:
    """Newline-delimited JSON-RPC 2.0 client over UDS or TCP."""

    def __init__(self, gateway_params: MinerRpcConfig) -> None:
        self._socket = self._connect_socket(gateway_params)
        self._reader: TextIO = self._socket.makefile("r", encoding="utf-8")
        self._writer: TextIO = self._socket.makefile("w", encoding="utf-8")
        self._request_id = 0

    def _connect_socket(self, gateway_params: MinerRpcConfig) -> socket.socket:
        try:
            if gateway_params.transport == "tcp":
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((gateway_params.host, gateway_params.port))
                _LOGGER.info(f"Connected to {gateway_params.host}:{gateway_params.port}")
            else:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(gateway_params.socket_path)
                _LOGGER.info(f"Connected to {gateway_params.socket_path}")
        except Exception as e:
            _LOGGER.warning(f"Failed to connect: {e}")
            raise
        return sock

    def close(self) -> None:
        # Shut down first so any blocked readline() in another thread wakes immediately.
        with suppress(OSError):
            self._socket.shutdown(socket.SHUT_RDWR)
        self._reader.close()
        self._writer.close()
        self._socket.close()

    def call(self, method: str, params: Any = None) -> Any:
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self._request_id,
            "params": params if params is not None else {},
        }
        self._writer.write(json.dumps(request) + "\n")
        self._writer.flush()

        line = self._reader.readline()
        if not line:
            raise ConnectionError("Connection closed by remote host")
        response = json.loads(line.strip())
        if "error" in response:
            error = response["error"]
            raise Exception(f"JSON-RPC error {error.get('code')}: {error.get('message')}")
        if response.get("id") != self._request_id:
            raise Exception(
                f"Response ID mismatch: expected {self._request_id}, got {response.get('id')}"
            )
        return response.get("result")


class MiningClient(AbstractContextManager):
    """Persistent gateway connection for job polling and proof submission."""

    def __init__(self, miner_rpc_config: MinerRpcConfig) -> None:
        self.client = JSONRPCClient(miner_rpc_config)

    def get_mining_info(self) -> MiningJob:
        return MiningJob.from_dict(self.client.call("getMiningInfo"))

    def submit_plain_proof(self, plain_proof: PlainProofV4, mining_job: MiningJob) -> None:
        self.client.call(
            "submitPlainProof",
            {"plain_proof": plain_proof.to_base64(), "mining_job": mining_job.to_dict()},
        )

    def close(self) -> None:
        self.client.close()

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class DummyMiningClient(MiningClient):
    """Offline client for ``MINER_NO_GATEWAY=true`` runs."""

    def __init__(self) -> None:
        pass

    def get_mining_info(self) -> MiningJob:
        return MiningJob(
            _DUMMY_HEADER_BYTES,
            bits_to_target(_DUMMY_NBITS),
            CertificateVersion.PLAIN_FP8,
            [bytes(_DUMMY_PARENT.to_bytes())],
        )

    def submit_plain_proof(self, plain_proof: PlainProofV4, mining_job: MiningJob) -> None:
        raise RuntimeError("DummyMiningClient does not submit proofs")

    def close(self) -> None:
        pass
