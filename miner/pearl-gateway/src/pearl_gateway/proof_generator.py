from copy import copy

from miner_utils import get_logger
from pearl_mining import PlainProof, generate_proof, verify_proof

from pearl_gateway.blockchain_utils.pearl_block import PearlBlock, ZKCertificate
from pearl_gateway.comm.dataclasses import BlockTemplate

_LOGGER = get_logger(__name__)


class ProofGenerator:
    """Builds a complete block from miner-supplied PlainProof and the cached block template."""

    @classmethod
    def generate_block(
        cls, plain_proof: PlainProof, template: BlockTemplate, debug_mode: bool = False
    ) -> PearlBlock:
        """Generate a complete block from PlainProof and BlockTemplate.
        Returns the PearlBlock for submission to the Pearl node.
        """
        _LOGGER.debug("Generating block from PlainProof")

        zk_proof = generate_proof(template.header.incomplete_header, plain_proof)
        _LOGGER.debug("Generated ZK proof")

        if debug_mode:
            _LOGGER.info("verifying ZK proof")
            result, msg = verify_proof(template.header.incomplete_header, zk_proof)
            if not result:
                raise AssertionError(f"Failed to verify proof: {msg}")
            _LOGGER.info("verified ZK proof")

        # We need to copy because ZKCertificate assigns the proof_commitment to the header
        header = copy(template.header)
        zk_certificate = ZKCertificate.from_pearl_header(header, zk_proof)
        block = PearlBlock(
            header=header,
            raw_txns=template.get_raw_transactions(),
            zk_certificate=zk_certificate,
        )
        _LOGGER.debug("Generated block")
        return block
