"""
Integration tests for Miner RPC communication.
Tests complete workflows and real network communication.
"""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock

import pytest
from pearl_gateway.comm.dataclasses import MiningJob, MiningPausedError
from pearl_gateway.config import MinerRpcConfig
from pearl_gateway.miner_rpc.server import MinerRpcServer
from pearl_gateway.submission_service import SubmissionService
from pearl_gateway.work_cache import WorkCache

from .mock_miner_client import MockMinerClient


@pytest.fixture
def submit_block_data(sample_plain_proof, sample_block_template):
    return {
        "plain_proof": sample_plain_proof.to_base64(),
        "mining_job": MiningJob.from_template(sample_block_template).to_dict(),
    }


@pytest.mark.integration
class TestMinerRpcIntegrationTcp:
    """Integration tests for TCP transport."""

    @pytest.fixture
    async def mock_services(self):
        """Mock services for testing."""
        work_cache = AsyncMock(spec=WorkCache)
        submission_service = AsyncMock(spec=SubmissionService)
        return work_cache, submission_service

    @pytest.fixture
    async def running_tcp_server(self, mock_services, sample_block_template):
        """Start a TCP server for testing."""
        work_cache, submission_service = mock_services

        # Setup mock responses
        mining_job = MiningJob.from_template(sample_block_template)
        work_cache.get_mining_job.return_value = mining_job
        work_cache.current_template = sample_block_template
        submission_service.submit_plain_proof.return_value = {"status": "accepted"}

        config = MinerRpcConfig(
            transport="tcp",
            socket_path="",  # Not used for TCP
            port=18446,
        )

        server = MinerRpcServer(work_cache, submission_service, config)
        await server.start()

        yield server, work_cache, submission_service

        await server.stop()

    @pytest.fixture
    async def running_auth_server(self, mock_services, sample_block_template):
        """Start a TCP server with authentication for testing."""
        work_cache, submission_service = mock_services

        # Setup mock responses
        mining_job = MiningJob.from_template(sample_block_template)
        work_cache.get_mining_job.return_value = mining_job
        work_cache.current_template = sample_block_template
        submission_service.submit_plain_proof.return_value = {"status": "accepted"}

        config = MinerRpcConfig(
            transport="tcp",
            socket_path="",
            port=18447,
        )

        server = MinerRpcServer(work_cache, submission_service, config)
        await server.start()

        yield server, work_cache, submission_service

        await server.stop()

    async def test_full_mining_workflow_tcp(self, running_tcp_server, submit_block_data):
        """Test complete mining workflow via TCP."""
        server, work_cache, submission_service = running_tcp_server

        async with MockMinerClient(transport="tcp", port=18446) as client:
            # Step 1: Get mining info
            status, response = await client.get_mining_info()

            assert status == 200
            assert response["jsonrpc"] == "2.0"
            assert response.get("error") is None
            assert "result" in response

            result = response["result"]
            assert "incomplete_header_bytes" in result
            assert "target" in result

            work_cache.get_mining_job.assert_called()

            # Step 2: Submit a block
            status, response = await client.submit_plain_proof(**submit_block_data)

            assert status == 200
            assert response["jsonrpc"] == "2.0"
            assert response.get("error") is None
            assert response["result"] == "submitted"

        # Verify submission service was called (may take a moment due to background task)
        await asyncio.sleep(0.1)
        submission_service.submit_plain_proof.assert_called()

    async def test_authentication_tcp(self, running_auth_server):
        """Test that authentication is not required (local communication only)."""
        server, work_cache, submission_service = running_auth_server

        # Test without authentication - should succeed (no auth required)
        async with MockMinerClient(transport="tcp", port=18447) as client:
            status, response = await client.get_mining_info()

            assert status == 200
            assert response["jsonrpc"] == "2.0"
            assert response.get("error") is None
            assert "result" in response

        # Test with authentication token - should also succeed (token ignored)
        async with MockMinerClient(
            transport="tcp",
            port=18447,
        ) as client:
            status, response = await client.get_mining_info()

            assert status == 200
            assert response["jsonrpc"] == "2.0"
            assert response.get("error") is None
            assert "result" in response

    async def test_concurrent_requests_tcp(self, running_tcp_server):
        """Test handling concurrent requests via TCP."""
        server, work_cache, submission_service = running_tcp_server

        async def make_request():
            async with MockMinerClient(transport="tcp", port=18446) as client:
                return await client.get_mining_info()

        # Send 5 concurrent requests
        tasks = [make_request() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        for status, response in results:
            assert status == 200
            assert response["jsonrpc"] == "2.0"
            assert response.get("error") is None

        # Work cache should have been called 5 times
        assert work_cache.get_mining_job.call_count == 5

    async def test_error_handling_tcp(self, running_tcp_server):
        """Test error handling via TCP."""
        server, work_cache, submission_service = running_tcp_server

        # Test with mining paused error
        work_cache.get_mining_job.side_effect = MiningPausedError("Node not ready")

        async with MockMinerClient(transport="tcp", port=18446) as client:
            status, response = await client.get_mining_info()

            assert status == 200
            assert response["error"]["code"] == -32001
            assert "mining_paused" in response["error"]["message"]

    async def test_invalid_requests_tcp(self, running_tcp_server):
        """Test handling of invalid requests via TCP."""
        server, work_cache, submission_service = running_tcp_server

        async with MockMinerClient(transport="tcp", port=18446) as client:
            # Test invalid method
            status, response = await client.send_request("invalidMethod")

            assert status == 200
            assert response["error"]["code"] == -32601
            assert "Method invalidMethod not found" in response["error"]["message"]

            # Test invalid parameters for getMiningInfo
            status, response = await client.send_request(
                "getMiningInfo", {"invalid_param": "invalid_value"}
            )

            assert status == 200
            assert response["error"]["code"] == -32602
            assert response["error"]["message"] == "Invalid params"


@pytest.mark.integration
class TestMinerRpcIntegrationUds:
    """Integration tests for Unix Domain Socket transport."""

    @pytest.fixture
    async def mock_services(self):
        """Mock services for testing."""
        work_cache = AsyncMock(spec=WorkCache)
        submission_service = AsyncMock(spec=SubmissionService)
        return work_cache, submission_service

    @pytest.fixture
    async def running_uds_server(self, mock_services, sample_block_template):
        """Start a UDS server for testing."""
        work_cache, submission_service = mock_services

        # Setup mock responses
        mining_job = MiningJob.from_template(sample_block_template)
        work_cache.get_mining_job.return_value = mining_job
        work_cache.current_template = sample_block_template
        submission_service.submit_plain_proof.return_value = {"status": "accepted"}

        socket_path = tempfile.mktemp(suffix=".sock")
        config = MinerRpcConfig(
            transport="uds",
            socket_path=socket_path,
        )

        server = MinerRpcServer(work_cache, submission_service, config)
        await server.start()

        yield server, work_cache, submission_service, socket_path

        await server.stop()

        # Cleanup socket file if it still exists
        if os.path.exists(socket_path):
            os.unlink(socket_path)

    async def test_uds_socket_creation(self, running_uds_server):
        """Test that UDS socket is created with correct permissions."""
        server, work_cache, submission_service, socket_path = running_uds_server

        # Check socket file exists
        assert os.path.exists(socket_path)

        # Check permissions
        stat_info = os.stat(socket_path)
        permissions = stat_info.st_mode & 0o777
        assert permissions == 0o600

    async def test_full_mining_workflow_uds(self, running_uds_server, submit_block_data):
        """Test complete mining workflow via UDS."""
        server, work_cache, submission_service, socket_path = running_uds_server

        async with MockMinerClient(transport="uds", socket_path=socket_path) as client:
            # Step 1: Get mining info
            status, response = await client.get_mining_info()

            assert status == 200
            assert response["jsonrpc"] == "2.0"
            assert response.get("error") is None
            assert "result" in response

            result = response["result"]
            assert "incomplete_header_bytes" in result
            assert "target" in result

            work_cache.get_mining_job.assert_called()

            # Step 2: Submit a block
            status, response = await client.submit_plain_proof(**submit_block_data)

            assert status == 200
            assert response["jsonrpc"] == "2.0"
            assert response.get("error") is None
            assert response["result"] == "submitted"

            # Verify submission service was called
            await asyncio.sleep(0.1)
            submission_service.submit_plain_proof.assert_called()


@pytest.mark.integration
class TestMinerRpcStressTests:
    """Stress tests for miner RPC communication."""

    @pytest.fixture
    async def stress_test_server(self, sample_block_template):
        """Server configured for stress testing."""
        work_cache = AsyncMock(spec=WorkCache)
        submission_service = AsyncMock(spec=SubmissionService)

        # Fast responses for stress testing
        mining_job = MiningJob.from_template(sample_block_template)
        work_cache.get_mining_job.return_value = mining_job
        work_cache.current_template = sample_block_template
        submission_service.submit_plain_proof.return_value = {"status": "accepted"}

        config = MinerRpcConfig(transport="tcp", socket_path="", port=18448)

        server = MinerRpcServer(work_cache, submission_service, config)
        await server.start()

        yield server, work_cache, submission_service

        await server.stop()

    async def test_rapid_fire_requests(self, stress_test_server):
        """Test rapid consecutive requests."""
        server, work_cache, submission_service = stress_test_server

        # Create separate connections for each request
        async def make_single_request():
            async with MockMinerClient(transport="tcp", port=18448) as client:
                return await client.get_mining_info()

        # Send 50 rapid requests with separate connections
        tasks = [make_single_request() for _ in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful responses
        successful = 0
        for result in results:
            if not isinstance(result, Exception):
                status, response = result
                if status == 200 and response.get("error") is None:
                    successful += 1

        # Most requests should succeed (allow some tolerance for system limits)
        assert successful >= 45

    async def test_mixed_request_types(self, stress_test_server, submit_block_data):
        """Test mixing getMiningInfo and submitPlainProof requests."""

        async def get_info():
            async with MockMinerClient(transport="tcp", port=18448) as client:
                return await client.get_mining_info()

        async def submit_proof():
            async with MockMinerClient(transport="tcp", port=18448) as client:
                return await client.submit_plain_proof(**submit_block_data)

        # Mix of request types
        tasks = []
        for i in range(20):
            if i % 2 == 0:
                tasks.append(get_info())
            else:
                tasks.append(submit_proof())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful responses
        successful = 0
        for result in results:
            if not isinstance(result, Exception):
                status, response = result
                if status == 200 and response.get("error") is None:
                    successful += 1

        # Most requests should succeed
        assert successful >= 18


@pytest.mark.integration
class TestMinerRpcErrorScenarios:
    """Test various error scenarios in integration."""

    @pytest.fixture
    async def error_test_server(self, sample_block_template):
        """Server for testing error scenarios."""
        work_cache = AsyncMock(spec=WorkCache)
        submission_service = AsyncMock(spec=SubmissionService)

        config = MinerRpcConfig(transport="tcp", socket_path="", port=18449)

        server = MinerRpcServer(work_cache, submission_service, config)
        await server.start()

        yield server, work_cache, submission_service

        await server.stop()

    async def test_work_cache_failure(self, error_test_server):
        """Test handling of work cache failures."""
        _, work_cache, _ = error_test_server

        # Simulate work cache failure
        work_cache.get_mining_job.side_effect = Exception("Work cache down")

        async with MockMinerClient(transport="tcp", port=18449) as client:
            status, response = await client.get_mining_info()

            assert status == 200
            assert response["error"]["code"] == -32000
            assert "Work cache down" in response["error"]["message"]

    async def test_submission_service_failure(
        self, error_test_server, sample_block_template, submit_block_data
    ):
        """Test handling of submission service failures."""
        _, work_cache, submission_service = error_test_server

        # Setup work cache to succeed
        mining_job = MiningJob.from_template(sample_block_template)
        work_cache.get_mining_job.return_value = mining_job
        work_cache.current_template = sample_block_template

        # Simulate submission service failure
        submission_service.submit_plain_proof.side_effect = Exception("Submission failed")

        async with MockMinerClient(transport="tcp", port=18449) as client:
            # submitPlainProof should still return "submitted" (background task)
            status, response = await client.submit_plain_proof(**submit_block_data)

            assert status == 200
            assert response["result"] == "submitted"

            # But the background task should have failed
            await asyncio.sleep(0.2)  # Wait for background task
            submission_service.submit_plain_proof.assert_called()

    async def test_malformed_json_request(self, error_test_server):
        """Test handling of malformed JSON requests."""
        server, work_cache, submission_service = error_test_server

        # Send malformed JSON directly via raw socket
        reader, writer = await asyncio.open_connection("127.0.0.1", 18449)

        try:
            # Send malformed JSON
            malformed_json = "{ invalid json\n"
            writer.write(malformed_json.encode())
            await writer.drain()

            # Read response
            response_line = await reader.readline()
            response = json.loads(response_line.decode().strip())

            assert response["error"]["code"] == -32700
            assert response["error"]["message"] == "Parse error"

        finally:
            writer.close()
            await writer.wait_closed()
