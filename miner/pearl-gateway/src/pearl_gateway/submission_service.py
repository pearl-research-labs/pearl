import asyncio
from typing import TYPE_CHECKING, Any

from miner_utils import get_logger
from pearl_mining import PlainProof, check_cert_version_eligible

from pearl_gateway.comm.dataclasses import BlockTemplate
from pearl_gateway.pearl_client import PearlNodeClient
from pearl_gateway.proof_generator import ProofGenerator

if TYPE_CHECKING:
    from pearl_gateway.blockchain_utils.pearl_block import PearlBlock
    from pearl_gateway.proof_pool import ProofPool

logger = get_logger(__name__)


class SubmissionService:
    """
    Handles block submissions from miners to the Pearl node.
    Receives PlainProof from miners and generates complete blocks.
    """

    def __init__(
        self, pearl_client: PearlNodeClient, proof_pool: "ProofPool", debug_mode: bool = False
    ):
        self.pearl_client = pearl_client
        self.proof_pool = proof_pool
        self.submission_lock = asyncio.Lock()  # Ensure serialized submissions
        self.submission_log = set()
        self.debug_mode = debug_mode
        self.submitted_blocks = 0
        self.accepted_blocks = 0
        self.rejected_blocks = 0

    async def _build_block(self, plain_proof: PlainProof, template: BlockTemplate) -> "PearlBlock":
        """Prove in a worker process and assemble the block from the returned bytes."""
        public_data, proof_data = await self.proof_pool.prove(
            int(template.required_cert_version),
            template.header.serialize_without_proof_commitment(),
            plain_proof.to_base64(),
            self.debug_mode,
        )
        return ProofGenerator.build_block(public_data, proof_data, template)

    async def submit_plain_proof(
        self, plain_proof: PlainProof, template: BlockTemplate
    ) -> dict[str, Any]:
        """
        Submit a block built from PlainProof and the current template.
        Returns the result of the submission.
        """
        async with self.submission_lock:
            try:
                if template.header.serialize_without_proof_commitment() in self.submission_log:
                    logger.warning("Block already submitted, skipping")
                    return {"status": "already_submitted"}

                logger.info(
                    f"Received PlainProof submission for template time {template.header.timestamp}"
                )

                # Reject proofs that cannot be certified at the version the block requires.
                try:
                    check_cert_version_eligible(template.required_cert_version, plain_proof)
                except ValueError as e:
                    logger.warning(f"Rejecting proof: {e}")
                    return {"status": f"error: {e}"}

                block = await self._build_block(plain_proof, template)

                # Submit to the Pearl node
                self.submitted_blocks += 1
                result = await self.pearl_client.submit_block(block.serialize().hex())
                # Update counters based on result
                if result == "accepted":
                    self.accepted_blocks += 1
                    self.submission_log.add(template.header.serialize_without_proof_commitment())
                    logger.info("Block accepted by node!")
                else:
                    self.rejected_blocks += 1
                    logger.warning(f"Block rejected: {result}")

                # Return result to miner
                return {"status": result}

            except Exception as e:
                logger.exception(
                    f"Error submitting block: {e=}, {type(e)=}, {plain_proof=}, {template=}"
                )
                return {"status": f"error: {str(e)}"}
