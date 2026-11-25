"""
Test submission service functionality.
"""

import dataclasses
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pearl_mining
import pytest
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion, ZKCertificate
from pearl_gateway.submission_service import PoolSubmissionService, SubmissionService


@pytest.fixture
def mock_pearl_client():
    """Create a mock PearlNodeClient."""
    client = AsyncMock()
    client.submit_block = AsyncMock()
    return client


@pytest.fixture
def fake_proof_pool():
    """A stand-in ProofPool whose prove() returns dummy proof bytes."""
    pool = MagicMock()
    pool.prove = AsyncMock(return_value=(b"public-data", b"proof-data"))
    return pool


@pytest.fixture
def submission_service(mock_pearl_client, fake_proof_pool):
    """Create a SubmissionService instance for testing."""
    return SubmissionService(mock_pearl_client, fake_proof_pool)


def create_mock_block(hex_data: str):
    """Create a mock block object with a serialize method."""
    mock_block = MagicMock()
    mock_block.serialize.return_value = bytes.fromhex(hex_data)
    return mock_block


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("count", [0, 3])
async def test_build_block_preserves_the_proof_chain(
    mock_pearl_client, sample_block_template, pooled, count
):
    headers = tuple(bytes([i + 1]) * 108 for i in range(count))
    plain_proof = Mock(
        spec=pearl_mining.PlainProofV4,
        ancestor_chain=[pearl_mining.BlockHeader.from_bytes(h) for h in headers],
    )
    template = dataclasses.replace(
        sample_block_template, required_cert_version=CertificateVersion.PLAIN_FP8
    )
    pool = Mock(prove=AsyncMock(return_value=(b"public", b"proof"))) if pooled else None
    service = SubmissionService(mock_pearl_client, proof_pool=pool)
    with patch(
        "pearl_gateway.submission_service.prove_in_process", return_value=(b"public", b"proof")
    ):
        block = await service._build_block(plain_proof, template)
    certificate = ZKCertificate.deserialize(block.zk_certificate.serialize())
    assert tuple(h.serialize() for h in certificate.ancestor_headers) == headers


class TestBlockSubmission:
    """Test block submission functionality."""

    @pytest.mark.asyncio
    async def test_submit_block_accepted(
        self,
        submission_service,
        mock_pearl_client,
        fake_proof_pool,
        sample_plain_proof,
        sample_block_template,
        sample_pearl_block,
    ):
        """Test successful block submission that gets accepted."""
        mock_pearl_client.submit_block.return_value = "accepted"

        with patch(
            "pearl_gateway.submission_service.build_block",
            return_value=sample_pearl_block,
        ) as mock_build_block:
            result = await submission_service.submit_plain_proof(
                sample_plain_proof, sample_block_template
            )

        assert result["status"] == "accepted"

        # The proof is generated in the worker via the serialized boundary args.
        fake_proof_pool.prove.assert_awaited_once_with(
            int(sample_block_template.required_cert_version),
            sample_block_template.header.serialize_without_proof_commitment(),
            sample_plain_proof.to_base64(),
            False,
        )
        # The block is assembled in-process from the worker's proof bytes.
        mock_build_block.assert_called_once_with(
            b"public-data", b"proof-data", sample_block_template, ()
        )
        mock_pearl_client.submit_block.assert_called_once_with(sample_pearl_block.serialize().hex())

    @pytest.mark.asyncio
    async def test_submit_block_rejected(
        self,
        submission_service,
        mock_pearl_client,
        sample_plain_proof,
        sample_block_template,
        sample_pearl_block,
    ):
        """Test block submission that gets rejected."""
        mock_pearl_client.submit_block.return_value = "rejected: invalid proof"

        with patch(
            "pearl_gateway.submission_service.build_block",
            return_value=sample_pearl_block,
        ):
            result = await submission_service.submit_plain_proof(
                sample_plain_proof, sample_block_template
            )

        assert result["status"] == "rejected: invalid proof"

    @pytest.mark.asyncio
    async def test_submit_block_pearl_client_error(
        self,
        submission_service,
        mock_pearl_client,
        sample_plain_proof,
        sample_block_template,
        sample_pearl_block,
    ):
        """Test block submission when pearl client raises an exception."""
        mock_pearl_client.submit_block.side_effect = ConnectionError("Connection failed")

        with patch(
            "pearl_gateway.submission_service.build_block",
            return_value=sample_pearl_block,
        ):
            result = await submission_service.submit_plain_proof(
                sample_plain_proof, sample_block_template
            )

        assert result["status"] == "error: Connection failed"


class TestCrossoverEnforcement:
    """Test certificate-version crossover enforcement at submission time."""

    @pytest.mark.asyncio
    async def test_moe_proof_before_crossover_rejected(
        self,
        submission_service,
        mock_pearl_client,
        sample_block_template,
        dummy_moe_proof,
    ):
        """An MoE proof submitted for a V1-required block is rejected clearly."""
        v1_template = dataclasses.replace(
            sample_block_template,
            required_cert_version=CertificateVersion.ZK_DENSE,
        )

        result = await submission_service.submit_plain_proof(dummy_moe_proof, v1_template)

        assert result["status"].startswith("error:")
        assert "crossover" in result["status"].lower()
        # The proof must never reach the node.
        mock_pearl_client.submit_block.assert_not_called()

    @pytest.mark.asyncio
    async def test_dense_proof_before_crossover_accepted(
        self,
        submission_service,
        mock_pearl_client,
        sample_plain_proof,
        sample_block_template,
        sample_pearl_block,
    ):
        """A dense proof is eligible for a V1-required block."""
        v1_template = dataclasses.replace(
            sample_block_template,
            required_cert_version=CertificateVersion.ZK_DENSE,
        )
        mock_pearl_client.submit_block.return_value = "accepted"

        with patch(
            "pearl_gateway.submission_service.build_block",
            return_value=sample_pearl_block,
        ) as mock_generate:
            result = await submission_service.submit_plain_proof(sample_plain_proof, v1_template)

        assert result["status"] == "accepted"
        # The block must be assembled for the V1-required template
        # (build_block signature: public_data, proof_data, template, ancestor_headers).
        passed_template = mock_generate.call_args.args[2]
        assert passed_template.required_cert_version == CertificateVersion.ZK_DENSE


class TestSubmissionServiceIntegration:
    """Test SubmissionService integration scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_submissions_workflow(
        self,
        submission_service,
        mock_pearl_client,
        sample_plain_proof,
        sample_block_template,
        sample_pearl_block,
    ):
        """Test a complete submission workflow with multiple blocks."""
        mock_pearl_client.submit_block.side_effect = [
            "accepted",
            "rejected: duplicate",
            "accepted",
            ConnectionError("Network error"),
        ]

        results = []

        # Submit 4 blocks - clear submission_log each time to simulate different blocks
        for _ in range(4):
            submission_service.submission_log.clear()
            with patch(
                "pearl_gateway.submission_service.build_block",
                return_value=sample_pearl_block,
            ):
                try:
                    result = await submission_service.submit_plain_proof(
                        sample_plain_proof, sample_block_template
                    )
                    results.append(result)
                except Exception:
                    pass

        assert len(results) == 4
        assert results[0]["status"] == "accepted"
        assert results[1]["status"] == "rejected: duplicate"
        assert results[2]["status"] == "accepted"
        assert results[3]["status"].startswith("error:")


class TestPoolSubmission:
    """Pool mode: the certificate goes to the pool as a Stratum share."""

    @pytest.fixture
    def pool_client(self):
        client = AsyncMock()
        client.submit_zk_cert = AsyncMock(return_value=True)
        return client

    def _service(self, pool_client, fake_proof_pool, current_template):
        work_cache = MagicMock()
        work_cache.current_template = current_template
        return PoolSubmissionService(pool_client, fake_proof_pool, work_cache)

    async def test_pool_submission_sends_generated_certificate(
        self,
        pool_client,
        fake_proof_pool,
        sample_plain_proof,
        sample_block_template,
        sample_pearl_block,
    ):
        sample_block_template.job_id = "job-1"
        service = self._service(pool_client, fake_proof_pool, sample_block_template)

        with patch(
            "pearl_gateway.submission_service.build_block",
            return_value=sample_pearl_block,
        ) as mock_build_block:
            result = await service.submit_plain_proof(sample_plain_proof, sample_block_template)

        assert result == {"status": "accepted"}
        fake_proof_pool.prove.assert_awaited_once_with(
            int(sample_block_template.required_cert_version),
            sample_block_template.header.serialize_without_proof_commitment(),
            sample_plain_proof.to_base64(),
            False,
        )
        mock_build_block.assert_called_once_with(
            b"public-data", b"proof-data", sample_block_template, ()
        )
        _, kwargs = pool_client.submit_zk_cert.call_args
        assert kwargs["job_id"] == "job-1"
        assert kwargs["ntime_hex"] == f"{sample_block_template.header.timestamp:08x}"
        assert kwargs["zkcert_hex"] == sample_pearl_block.zk_certificate.serialize().hex()
        assert kwargs["extranonce2_hex"] is None
        assert service.accepted_blocks == 1

    async def test_pool_submission_echoes_extranonce2(
        self,
        pool_client,
        fake_proof_pool,
        sample_plain_proof,
        sample_block_template,
        sample_pearl_block,
    ):
        sample_block_template.job_id = "job-1"
        sample_block_template.extranonce2 = bytes.fromhex("aabbccdd")
        service = self._service(pool_client, fake_proof_pool, sample_block_template)

        with patch(
            "pearl_gateway.submission_service.build_block",
            return_value=sample_pearl_block,
        ):
            await service.submit_plain_proof(sample_plain_proof, sample_block_template)

        _, kwargs = pool_client.submit_zk_cert.call_args
        assert kwargs["extranonce2_hex"] == "aabbccdd"

    async def test_pool_submission_skipped_when_job_changed_during_proof(
        self,
        pool_client,
        fake_proof_pool,
        sample_plain_proof,
        sample_block_template,
        sample_pearl_block,
    ):
        sample_block_template.job_id = "job-1"
        service = self._service(pool_client, fake_proof_pool, MagicMock(job_id="job-2"))

        with patch(
            "pearl_gateway.submission_service.build_block",
            return_value=sample_pearl_block,
        ):
            result = await service.submit_plain_proof(sample_plain_proof, sample_block_template)

        assert result["status"].startswith("skipped")

    async def test_pool_submission_skipped_when_template_replaced_under_same_job_id(
        self,
        pool_client,
        fake_proof_pool,
        sample_plain_proof,
        sample_block_template,
        sample_pearl_block,
    ):
        # After a reconnect the pool re-sends the same job id with a new extranonce1.
        sample_block_template.job_id = "job-1"
        service = self._service(pool_client, fake_proof_pool, MagicMock(job_id="job-1"))

        with patch(
            "pearl_gateway.submission_service.build_block",
            return_value=sample_pearl_block,
        ):
            result = await service.submit_plain_proof(sample_plain_proof, sample_block_template)

        assert result["status"].startswith("skipped")
        pool_client.submit_zk_cert.assert_not_called()

    async def test_pool_rejection_is_counted(
        self,
        pool_client,
        fake_proof_pool,
        sample_plain_proof,
        sample_block_template,
        sample_pearl_block,
    ):
        pool_client.submit_zk_cert.return_value = False
        sample_block_template.job_id = "job-1"
        service = self._service(pool_client, fake_proof_pool, sample_block_template)

        with patch(
            "pearl_gateway.submission_service.build_block",
            return_value=sample_pearl_block,
        ):
            result = await service.submit_plain_proof(sample_plain_proof, sample_block_template)

        assert result == {"status": "rejected"}
        assert service.rejected_blocks == 1

    async def test_pool_submission_without_job_id_is_an_error(
        self, pool_client, fake_proof_pool, sample_plain_proof, sample_block_template
    ):
        service = self._service(pool_client, fake_proof_pool, sample_block_template)

        result = await service.submit_plain_proof(sample_plain_proof, sample_block_template)

        assert result["status"].startswith("error")
        fake_proof_pool.prove.assert_not_called()
