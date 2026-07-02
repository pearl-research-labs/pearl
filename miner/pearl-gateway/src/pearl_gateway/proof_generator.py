from copy import copy

from miner_utils import get_logger
from pearl_mining import ZKProof

from pearl_gateway.blockchain_utils.pearl_block import PearlBlock
from pearl_gateway.blockchain_utils.zk_certificate import ZKCertificate
from pearl_gateway.comm.dataclasses import BlockTemplate

_LOGGER = get_logger(__name__)


class ProofGenerator:
    """Assembles a complete block from a worker-generated ZK proof and the cached template.

    The CPU-bound proving step runs in a separate process (see ``proof_worker`` /
    ``ProofPool``); this class only does the cheap, parent-side assembly from the
    proof bytes the worker returns.
    """

    @classmethod
    def build_block(
        cls, public_data: bytes, proof_data: bytes, template: BlockTemplate
    ) -> PearlBlock:
        """Build a complete block from the worker's proof bytes and the template."""
        _LOGGER.debug("Building block from ZK proof")

        # The certificate version is dictated by the block height via the template.
        cert_version = template.required_cert_version
        zk_proof = ZKProof(public_data, proof_data)

        # We need to copy because ZKCertificate assigns the proof_commitment to the header
        header = copy(template.header)
        zk_certificate = ZKCertificate.from_pearl_header(
            header, zk_proof, cert_version=cert_version
        )
        block = PearlBlock(
            header=header,
            raw_txns=template.get_raw_transactions(),
            zk_certificate=zk_certificate,
        )
        _LOGGER.debug("Built block")
        return block
