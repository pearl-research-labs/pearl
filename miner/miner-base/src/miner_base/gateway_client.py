from contextlib import AbstractContextManager
from types import TracebackType

from miner_utils import get_logger
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.dataclasses import MiningJob
from pearl_gateway.comm.json_rpc_client import JSONRPCClient
from pearl_gateway.config import MinerRpcConfig
from pearl_mining import PlainProof

_LOGGER = get_logger(__name__)


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

    def submit_plain_proof(self, plain_proof: PlainProof, mining_job: MiningJob) -> None:
        """Submit a PlainProof to the gateway.

        Args:
            plain_proof: PlainProof object containing the proof data
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
        return MiningJob(b"\xde\xad\xba\xbe" * 8, 1, cert_version=CertificateVersion.ZK_MOE)
