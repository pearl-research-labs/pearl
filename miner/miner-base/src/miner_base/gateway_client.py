from contextlib import AbstractContextManager
from types import TracebackType

from miner_utils import get_logger
from pearl_gateway.blockchain_utils.blockchain_utils import bits_to_target
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.dataclasses import MiningJob
from pearl_gateway.comm.json_rpc_client import JSONRPCClient
from pearl_gateway.config import MinerRpcConfig
from pearl_mining import IncompleteBlockHeader, PlainProof, PlainProofV4

_LOGGER = get_logger(__name__)

# The dummy (no-gateway) job: plain-peel miners parse the header with
# IncompleteBlockHeader.from_bytes on every first matmul, so it must be a
# valid serialized 76-byte header. Hard difficulty so offline runs
# (MINER_NO_GATEWAY=1 benchmarks) do not constantly "win".
_DUMMY_NBITS = 0x1D00FFFF
_DUMMY_HEADER_BYTES = bytes(
    IncompleteBlockHeader(
        version=1,
        prev_block=b"\xde\xad\xba\xbe" * 8,
        merkle_root=b"\x00" * 32,
        timestamp=0,
        nbits=_DUMMY_NBITS,
    ).to_bytes()
)


class MiningClient(AbstractContextManager):
    """
    A wrapper around JSONRPCClient that provides mining-specific methods.
    This class manages a persistent connection to the gateway.
    """

    def __init__(self, miner_rpc_config: MinerRpcConfig) -> None:
        self.client = JSONRPCClient(miner_rpc_config)

    def get_mining_info(self) -> MiningJob:
        """
        Send a getMiningInfo request to the Pearl Gateway.
        Returns a MiningJob object containing the mining information.
        """
        result = self.client.call("getMiningInfo")
        return MiningJob.from_dict(result)

    def submit_plain_proof(
        self, plain_proof: PlainProof | PlainProofV4, mining_job: MiningJob
    ) -> None:
        """Submit a plain proof to the gateway.

        Args:
            plain_proof: PlainProof (int7 certs) or PlainProofV4 (cert v4) with the proof data
            mining_job: MiningJob associated with this proof
        """
        self.client.call(
            "submitPlainProof",
            {"plain_proof": plain_proof.to_base64(), "mining_job": mining_job.to_dict()},
        )

    def close(self) -> None:
        """Close the underlying client connection."""
        self.client.close()

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Close the client on exit."""
        self.close()
        return None


class DummyRPCClient:
    def call(self, method: str, args: dict | None = None) -> None:
        _LOGGER.debug('DummyRPCClient.call("{}", ...)', method)

    def close(self) -> None:
        _LOGGER.debug("DummyRPCClient.close()")


class DummyMiningClient(MiningClient):
    def __init__(self) -> None:
        self.client = DummyRPCClient()

    def get_mining_info(self) -> MiningJob:
        return MiningJob(
            _DUMMY_HEADER_BYTES, bits_to_target(_DUMMY_NBITS), CertificateVersion.PLAIN_FP8
        )
