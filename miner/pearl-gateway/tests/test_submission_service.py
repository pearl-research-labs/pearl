"""
Test submission service functionality.
"""

import asyncio
import dataclasses
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.submission_service import SubmissionService


@pytest.fixture
def mock_pearl_client():
    """Create a mock PearlNodeClient."""
    client = AsyncMock()
    client.submit_block = AsyncMock()
    return client


@pytest.fixture
def submission_service(mock_pearl_client):
    """Create a SubmissionService instance for testing."""
    return SubmissionService(mock_pearl_client)


def create_mock_block(hex_data: str):
    """Create a mock block object with a serialize method."""
    mock_block = MagicMock()
    mock_block.serialize.return_value = bytes.fromhex(hex_data)
    return mock_block


class TestBlockSubmission:
    """Test block submission functionality."""

    @pytest.mark.asyncio
    async def test_submit_block_accepted(
        self,
        submission_service,
        mock_pearl_client,
        sample_plain_proof,
        sample_block_template,
        sample_pearl_block,
    ):
        """Test successful block submission that gets accepted."""
        mock_pearl_client.submit_block.return_value = "accepted"

        with patch(
            "pearl_gateway.proof_generator.ProofGenerator.generate_block",
            return_value=sample_pearl_block,
        ):
            result = await submission_service.submit_plain_proof(
                sample_plain_proof, sample_block_template
            )

        assert result["status"] == "accepted"

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
            "pearl_gateway.proof_generator.ProofGenerator.generate_block",
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

        with (
            patch(
                "pearl_gateway.proof_generator.ProofGenerator.generate_block",
                return_value=sample_pearl_block,
            ),
        ):
            result = await submission_service.submit_plain_proof(
                sample_plain_proof, sample_block_template
            )

        assert result["status"] == "error: Connection failed"

    @pytest.mark.asyncio
    async def test_submit_block_thread_safety(
        self,
        submission_service,
        mock_pearl_client,
        sample_plain_proof,
        sample_block_template,
        sample_pearl_block,
    ):
        """Test that block submissions are serialized."""
        mock_pearl_client.submit_block.return_value = "accepted"

        # Track call order
        call_order = []

        async def mock_submit_with_delay(block_hex):
            call_order.append(f"start_{block_hex[:8]}")
            await asyncio.sleep(0.1)  # Simulate network delay
            call_order.append(f"end_{block_hex[:8]}")
            return "accepted"

        mock_pearl_client.submit_block.side_effect = mock_submit_with_delay

        with patch(
            "pearl_gateway.proof_generator.ProofGenerator.generate_block",
            side_effect=[sample_pearl_block, sample_pearl_block],
        ):
            # Submit two blocks concurrently with same template
            results = await asyncio.gather(
                submission_service.submit_plain_proof(sample_plain_proof, sample_block_template),
                submission_service.submit_plain_proof(sample_plain_proof, sample_block_template),
            )

        # First should be accepted, second should be detected as duplicate
        statuses = [r["status"] for r in results]
        assert "accepted" in statuses
        assert "already_submitted" in statuses

        # Only one actual submission should happen due to deduplication
        assert len(call_order) == 2
        assert call_order[0].startswith("start_")
        assert call_order[1].startswith("end_")


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
            "pearl_gateway.proof_generator.ProofGenerator.generate_block",
            return_value=sample_pearl_block,
        ) as mock_generate:
            result = await submission_service.submit_plain_proof(sample_plain_proof, v1_template)

        assert result["status"] == "accepted"
        # The block must be generated for the V1-required template.
        passed_template = mock_generate.call_args.args[1]
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
        # Configure different responses
        mock_pearl_client.submit_block.side_effect = [
            "accepted",
            "rejected: duplicate",
            "accepted",
            ConnectionError("Network error"),
        ]

        results = []

        # Submit 4 blocks - clear submission_log each time to simulate different blocks
        for _ in range(4):
            submission_service.submission_log.clear()  # Clear to simulate different blocks
            with patch(
                "pearl_gateway.proof_generator.ProofGenerator.generate_block",
                return_value=sample_pearl_block,
            ):
                try:
                    result = await submission_service.submit_plain_proof(
                        sample_plain_proof, sample_block_template
                    )
                    results.append(result)
                except Exception:
                    # The service should handle exceptions internally
                    pass

        # Check final metrics

        # Check results
        assert len(results) == 4
        assert results[0]["status"] == "accepted"
        assert results[1]["status"] == "rejected: duplicate"
        assert results[2]["status"] == "accepted"
        assert results[3]["status"].startswith("error:")
