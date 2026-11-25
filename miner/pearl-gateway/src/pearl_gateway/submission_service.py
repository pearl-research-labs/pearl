from collections import deque
from copy import copy
from typing import TYPE_CHECKING, Any

from miner_utils import get_logger
from pearl_mining import PlainProof, PlainProofV4, check_cert_version_eligible

from pearl_gateway.blockchain_utils.pearl_block import PearlBlock
from pearl_gateway.blockchain_utils.pearl_header import PearlHeader
from pearl_gateway.blockchain_utils.zk_certificate import CertificateProof, ZKCertificate
from pearl_gateway.comm.dataclasses import BlockTemplate
from pearl_gateway.pearl_client import PearlNodeClient
from pearl_gateway.proof_worker import prove as prove_in_process

if TYPE_CHECKING:
    from pearl_gateway.pool_stratum_client import PoolStratumClient
    from pearl_gateway.proof_pool import ProofPool
    from pearl_gateway.work_cache import WorkCache

logger = get_logger(__name__)
_MAX_SUBMISSION_LOG = 64


def build_block(
    public_data: bytes,
    proof_data: bytes,
    template: BlockTemplate,
    ancestor_headers: tuple[bytes, ...] = (),
) -> PearlBlock:
    """Assemble a certified block from proof bytes and a copy of the template header."""
    header = copy(template.header)
    zk_certificate = ZKCertificate.from_pearl_header(
        header,
        CertificateProof(public_data, proof_data),
        cert_version=template.required_cert_version,
        ancestor_headers=[PearlHeader.deserialize(h) for h in ancestor_headers],
    )
    return PearlBlock(
        header=header,
        raw_txns=template.get_raw_transactions(),
        zk_certificate=zk_certificate,
    )


class SubmissionService:
    """Certifies incoming plain proofs and submits assembled blocks to pearld."""

    def __init__(
        self,
        pearl_client: PearlNodeClient | None,
        proof_pool: "ProofPool | None" = None,
        debug_mode: bool = False,
    ) -> None:
        self.pearl_client = pearl_client
        self.proof_pool = proof_pool
        self.debug_mode = debug_mode
        self.submission_log: deque[bytes] = deque(maxlen=_MAX_SUBMISSION_LOG)
        self.submitted_blocks = 0
        self.accepted_blocks = 0
        self.rejected_blocks = 0

    async def _build_block(
        self, plain_proof: PlainProof | PlainProofV4, template: BlockTemplate
    ) -> PearlBlock:
        ancestor_headers = (
            tuple(bytes(h.to_bytes()) for h in plain_proof.ancestor_chain)
            if isinstance(plain_proof, PlainProofV4)
            else ()
        )
        args = (
            int(template.required_cert_version),
            template.header.serialize_without_proof_commitment(),
            plain_proof.to_base64(),
            self.debug_mode,
        )
        if self.proof_pool is not None:
            public_data, proof_data = await self.proof_pool.prove(*args)
        else:
            public_data, proof_data = prove_in_process(*args)
        return build_block(public_data, proof_data, template, ancestor_headers)

    async def submit_plain_proof(
        self, plain_proof: PlainProof | PlainProofV4, template: BlockTemplate
    ) -> dict[str, Any]:
        try:
            header_bytes = template.header.serialize_without_proof_commitment()
            if header_bytes in self.submission_log:
                logger.warning("Block already submitted, skipping")
                return {"status": "already_submitted"}

            logger.info(
                f"Received PlainProof submission for template time {template.header.timestamp}"
            )
            try:
                check_cert_version_eligible(template.required_cert_version, plain_proof)
            except ValueError as e:
                logger.warning(f"Rejecting proof: {e}")
                return {"status": f"error: {e}"}

            block = await self._build_block(plain_proof, template)
            assert self.pearl_client is not None
            self.submitted_blocks += 1
            result = await self.pearl_client.submit_block(block.serialize().hex())
            if result == "accepted":
                self.accepted_blocks += 1
                self.submission_log.append(header_bytes)
                logger.info("Block accepted by node!")
            else:
                self.rejected_blocks += 1
                logger.warning(f"Block rejected: {result}")
            return {"status": result}
        except Exception as e:
            logger.exception(f"Error submitting block: {e=}, {type(e)=}")
            return {"status": f"error: {e}"}


class PoolSubmissionService(SubmissionService):
    """Pool mode: certifies PlainProof and submits ZKCertificate to the pool as a share."""

    def __init__(
        self,
        pool_client: "PoolStratumClient",
        proof_pool: "ProofPool",
        work_cache: "WorkCache",
    ) -> None:
        # No block-level --debug verify here: it checks the network target, which
        # every share that is not also a block fails. MinerRpcServer checks each
        # proof against the share target before proving instead.
        super().__init__(pearl_client=None, proof_pool=proof_pool)
        self.pool_client = pool_client
        self.work_cache = work_cache

    async def submit_plain_proof(
        self, plain_proof: PlainProof | PlainProofV4, template: BlockTemplate
    ) -> dict[str, Any]:
        """Generate a ZK proof and submit the certificate to the pool.

        zkcert_hex is ZKCertificate.serialize() hex-encoded: the same leading
        bytes of PearlBlock.serialize() for a solo submit.
        """
        if template.job_id is None:
            logger.error("Cannot submit to pool: template has no job_id")
            return {"status": "error: missing job_id"}

        try:
            try:
                check_cert_version_eligible(template.required_cert_version, plain_proof)
            except ValueError as e:
                logger.warning(f"Rejecting proof: {e}")
                return {"status": f"error: {e}"}

            block = await self._build_block(plain_proof, template)

            # Same object or stale: the pool keeps a job id across reconnects, but a
            # new session's template carries a different extranonce1.
            if self.work_cache.current_template is not template:
                logger.info("Job changed during proving; skipping stale submission")
                return {"status": "skipped: stale job"}

            self.submitted_blocks += 1
            accepted = await self.pool_client.submit_zk_cert(
                job_id=template.job_id,
                ntime_hex=f"{template.header.timestamp:08x}",
                zkcert_hex=block.zk_certificate.serialize().hex(),
                extranonce2_hex=template.extranonce2.hex()
                if template.extranonce2 is not None
                else None,
            )

            if accepted:
                self.accepted_blocks += 1
                logger.info("Share accepted by pool")
                return {"status": "accepted"}
            self.rejected_blocks += 1
            logger.warning("Share rejected by pool")
            return {"status": "rejected"}

        except Exception as e:
            logger.exception(f"Error submitting proof to pool: {e}")
            return {"status": f"error: {e}"}
