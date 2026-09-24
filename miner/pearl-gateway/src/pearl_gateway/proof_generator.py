from copy import copy

from miner_utils import get_logger

from pearl_gateway.blockchain_utils.pearl_block import PearlBlock
from pearl_gateway.blockchain_utils.zk_certificate import CertificateProof, ZKCertificate
from pearl_gateway.comm.dataclasses import BlockTemplate
from pearl_gateway.proof_worker import prove

_LOGGER = get_logger(__name__)


class ProofGenerator:
    """Assembles a complete block from a worker-generated ZK proof and the cached template.

    The CPU-bound proving step runs in a separate process (see ``proof_worker`` /
    ``ProofPool``); this class only does the cheap, parent-side assembly from the
    proof bytes the worker returns. ``generate_block`` is the in-process fallback
    used by tests that do not start a ``ProofPool``.
    """

    @classmethod
    def build_block(
        cls, public_data: bytes, proof_data: bytes, template: BlockTemplate
    ) -> PearlBlock:
        """Build a complete block from the worker's proof bytes and the template."""
        _LOGGER.debug("Building block from ZK proof")

        # The certificate version is dictated by the block height via the template.
        cert_version = template.required_cert_version
        zk_proof = CertificateProof(public_data, proof_data)

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

    @classmethod
    def generate_block(cls, plain_proof, template: BlockTemplate, debug_mode: bool = False):
        """Prove in-process and assemble the block (test / no-pool path)."""
        public_data, proof_data = prove(
            int(template.required_cert_version),
            template.header.serialize_without_proof_commitment(),
            plain_proof.to_base64(),
            debug_mode,
        )
        return cls.build_block(public_data, proof_data, template)
