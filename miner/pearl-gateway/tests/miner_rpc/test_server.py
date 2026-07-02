"""
Unit tests for MinerRpcServer.
"""

import asyncio
import contextlib
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from pearl_gateway.comm.dataclasses import MiningJob, MiningPausedError
from pearl_gateway.config import MinerRpcConfig
from pearl_gateway.miner_rpc.server import ClientInfo, MinerRpcServer
from pearl_gateway.submission_service import SubmissionService
from pearl_gateway.work_cache import WorkCache


@pytest.fixture
def mock_work_cache():
    """Mock WorkCache for testing."""
    cache = AsyncMock(spec=WorkCache)
    return cache


@pytest.fixture
def mock_submission_service():
    """Mock SubmissionService for testing."""
    service = AsyncMock(spec=SubmissionService)
    return service


@pytest.fixture
def tcp_config():
    """TCP transport configuration for testing."""
    return MinerRpcConfig(
        transport="tcp",
        socket_path="",  # Not used for TCP
        port=18443,
    )


@pytest.fixture
def uds_config():
    """UDS transport configuration for testing."""
    return MinerRpcConfig(
        transport="uds",
        socket_path="/tmp/test_miner_rpc.sock",
        port=0,  # Not used for UDS
    )


@pytest.fixture
def no_auth_config():
    """Configuration without authentication."""
    return MinerRpcConfig(
        transport="tcp",
        socket_path="",  # Not used for TCP
        port=18444,
    )


@pytest.fixture
def sample_mining_job(sample_block_template):
    """Create a sample MiningJob for testing."""
    return MiningJob.from_template(sample_block_template)


class TestMinerRpcServerInit:
    """Test MinerRpcServer initialization."""

    def test_init_with_tcp_config(self, mock_work_cache, mock_submission_service, tcp_config):
        """Test server initialization with TCP config."""
        server = MinerRpcServer(
            mock_work_cache, mock_submission_service, tcp_config, max_pending_submissions=64
        )

        assert server.work_cache == mock_work_cache
        assert server.submission_service == mock_submission_service
        assert server.config == tcp_config
        assert server.server is None  # Not started yet
        assert server.clients == {}  # Empty client dict initially

    def test_init_with_uds_config(self, mock_work_cache, mock_submission_service, uds_config):
        """Test server initialization with UDS config."""
        server = MinerRpcServer(
            mock_work_cache, mock_submission_service, uds_config, max_pending_submissions=64
        )

        assert server.work_cache == mock_work_cache
        assert server.submission_service == mock_submission_service
        assert server.config == uds_config
        assert server.server is None  # Not started yet
        assert server.clients == {}  # Empty client dict initially

    def test_init_without_auth(self, mock_work_cache, mock_submission_service, no_auth_config):
        """Test server initialization without authentication."""
        server = MinerRpcServer(
            mock_work_cache, mock_submission_service, no_auth_config, max_pending_submissions=64
        )

        assert server.work_cache == mock_work_cache
        assert server.submission_service == mock_submission_service
        assert server.config == no_auth_config
        assert server.server is None  # Not started yet
        assert server.clients == {}  # Empty client dict initially


class TestMinerRpcServerTransport:
    """Test server transport functionality."""

    @pytest.fixture
    def server(self, mock_work_cache, mock_submission_service, tcp_config):
        return MinerRpcServer(
            mock_work_cache, mock_submission_service, tcp_config, max_pending_submissions=64
        )

    @pytest.fixture
    def uds_server(self, mock_work_cache, mock_submission_service):
        config = MinerRpcConfig(
            transport="uds",
            socket_path=tempfile.mktemp(suffix=".sock"),
        )
        return MinerRpcServer(
            mock_work_cache, mock_submission_service, config, max_pending_submissions=64
        )

    async def test_start_stop_tcp(self, server):
        """Test starting and stopping TCP server."""
        await server.start()
        assert server.server is not None
        assert hasattr(server.server, "close")  # asyncio server has close method

        await server.stop()
        assert server.server is None

    async def test_start_stop_uds(self, uds_server):
        """Test starting and stopping UDS server."""
        socket_path = uds_server.config.socket_path

        await uds_server.start()
        assert os.path.exists(socket_path)
        assert uds_server.server is not None
        assert hasattr(uds_server.server, "close")  # asyncio server has close method

        await uds_server.stop()
        assert not os.path.exists(socket_path)
        assert uds_server.server is None

    async def test_uds_socket_permissions(self, uds_server):
        """Test UDS socket file permissions."""
        socket_path = uds_server.config.socket_path

        await uds_server.start()

        # Check that socket file has correct permissions (0o600)
        stat_info = os.stat(socket_path)
        permissions = stat_info.st_mode & 0o777
        assert permissions == 0o600

        await uds_server.stop()


class TestMinerRpcServerHandlers:
    """Test individual handler methods."""

    @pytest.fixture
    def server(self, mock_work_cache, mock_submission_service, no_auth_config):
        return MinerRpcServer(
            mock_work_cache, mock_submission_service, no_auth_config, max_pending_submissions=64
        )

    async def test_handle_submit_plain_proof(
        self, server, sample_plain_proof, sample_mining_job, sample_block_template
    ):
        """Test submitPlainProof handler."""
        server.work_cache.current_template = sample_block_template
        server.submission_service.submit_plain_proof.return_value = {"status": "accepted"}

        # This is a background task, so it doesn't return a value
        await server.handle_submit_plain_proof(sample_plain_proof, sample_mining_job)

        # Verify the submission service was called with correct arguments
        server.submission_service.submit_plain_proof.assert_called_once_with(
            sample_plain_proof, sample_block_template
        )

    async def test_handle_submit_plain_proof_drops_stale_job(
        self, server, sample_plain_proof, sample_block_template
    ):
        """A submission whose job no longer matches the current template is dropped."""
        server.work_cache.current_template = sample_block_template
        stale_job = MiningJob(incomplete_header_bytes=b"\x00" * 76, target=1)

        await server.handle_submit_plain_proof(sample_plain_proof, stale_job)

        server.submission_service.submit_plain_proof.assert_not_called()

    async def test_handle_submit_plain_proof_drops_without_template(
        self, server, sample_plain_proof, sample_mining_job
    ):
        """A queued submission is dropped (not raised) when no template is available."""
        server.work_cache.current_template = None

        await server.handle_submit_plain_proof(sample_plain_proof, sample_mining_job)

        server.submission_service.submit_plain_proof.assert_not_called()

    async def test_submission_consumer_processes_queued_submissions(
        self, server, sample_plain_proof, sample_mining_job, sample_block_template
    ):
        """The consumer drains the queue and hands each submission to the service."""
        server.work_cache.current_template = sample_block_template
        server.submission_service.submit_plain_proof.return_value = {"status": "accepted"}

        server._submission_queue.put_nowait((sample_plain_proof, sample_mining_job))
        consumer = asyncio.create_task(server._consume_submissions())
        try:
            await asyncio.wait_for(server._submission_queue.join(), timeout=5)
        finally:
            consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer

        server.submission_service.submit_plain_proof.assert_called_once_with(
            sample_plain_proof, sample_block_template
        )

    async def test_submission_consumer_survives_handler_errors(
        self, server, sample_plain_proof, sample_mining_job, sample_block_template
    ):
        """One failing submission must not stall the consumer for later ones."""
        server.work_cache.current_template = sample_block_template
        server.submission_service.submit_plain_proof.side_effect = [
            RuntimeError("proof worker died"),
            {"status": "accepted"},
        ]

        server._submission_queue.put_nowait((sample_plain_proof, sample_mining_job))
        server._submission_queue.put_nowait((sample_plain_proof, sample_mining_job))
        consumer = asyncio.create_task(server._consume_submissions())
        try:
            await asyncio.wait_for(server._submission_queue.join(), timeout=5)
        finally:
            consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer

        assert server.submission_service.submit_plain_proof.call_count == 2


class TestMinerRpcServerJsonRpc:
    """Test JSON-RPC request handling."""

    @pytest.fixture
    def server(self, mock_work_cache, mock_submission_service, no_auth_config):
        return MinerRpcServer(
            mock_work_cache, mock_submission_service, no_auth_config, max_pending_submissions=64
        )

    @pytest.fixture
    def mock_client(self):
        """Create a mock ClientInfo for testing."""
        return ClientInfo(client_id=0, writer=MagicMock())

    async def test_valid_get_mining_info_request(self, server, sample_mining_job, mock_client):
        """Test valid getMiningInfo JSON-RPC request."""
        server.work_cache.get_mining_job.return_value = sample_mining_job

        request_line = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "getMiningInfo",
                "params": {},
                "id": 1,
            }
        )

        response = await server._process_request(request_line, mock_client)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert response.get("error") is None
        assert "result" in response

    async def test_valid_submit_block_request(
        self,
        server,
        sample_mining_job,
        sample_block_template,
        submit_plain_proof_params,
        mock_client,
    ):
        """Test valid submitPlainProof JSON-RPC request."""
        server.work_cache.get_mining_job.return_value = sample_mining_job
        server.work_cache.current_template = sample_block_template
        server.submission_service.submit_plain_proof.return_value = {"status": "accepted"}

        request_line = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "submitPlainProof",
                "params": submit_plain_proof_params,
                "id": 2,
            }
        )

        response = await server._process_request(request_line, mock_client)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert response.get("error") is None
        assert response["result"] == "submitted"

    async def test_submit_plain_proof_rejected_when_queue_full(
        self, server, sample_block_template, submit_plain_proof_params, mock_client
    ):
        """A flood of submissions is rejected once the queue capacity is reached."""
        server.work_cache.current_template = sample_block_template
        while not server._submission_queue.full():
            server._submission_queue.put_nowait((None, None))

        request_line = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "submitPlainProof",
                "params": submit_plain_proof_params,
                "id": 3,
            }
        )

        response = await server._process_request(request_line, mock_client)

        assert response["id"] == 3
        assert response.get("result") is None
        assert response["error"]["message"] == "proof submission queue full"
        # Rejected before anything was enqueued for processing.
        server.submission_service.submit_plain_proof.assert_not_called()

    async def test_invalid_json_request(self, server, mock_client):
        """Test handling of invalid JSON."""
        request_line = "{ invalid json"

        response = await server._process_request(request_line, mock_client)

        assert response["error"]["code"] == -32700
        assert response["error"]["message"] == "Parse error"

    async def test_invalid_jsonrpc_envelope(self, server, mock_client):
        """Test handling of invalid JSON-RPC envelope."""
        request_line = json.dumps(
            {
                "jsonrpc": "1.0",  # Invalid version
                "method": "getMiningInfo",
                "id": 1,
            }
        )

        response = await server._process_request(request_line, mock_client)

        assert response["error"]["code"] == -32600
        assert response["error"]["message"] == "Invalid Request"

    async def test_unknown_method(self, server, mock_client):
        """Test handling of unknown method."""
        request_line = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "unknownMethod",
                "params": {},
                "id": 1,
            }
        )

        response = await server._process_request(request_line, mock_client)

        assert response["error"]["code"] == -32601
        assert "Method unknownMethod not found" in response["error"]["message"]

    async def test_invalid_method_params(self, server, mock_client):
        """Test handling of invalid method parameters."""
        request_line = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "getMiningInfo",
                "params": {
                    "invalid_param": "should_not_be_here"
                },  # getMiningInfo should be empty object
                "id": 1,
            }
        )

        response = await server._process_request(request_line, mock_client)

        assert response["error"]["code"] == -32602
        assert response["error"]["message"] == "Invalid params"

    async def test_mining_paused_error(self, server, mock_client):
        """Test handling of MiningPausedError."""
        server.work_cache.get_mining_job.side_effect = MiningPausedError("Node not ready")

        request_line = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "getMiningInfo",
                "params": {},
                "id": 1,
            }
        )

        response = await server._process_request(request_line, mock_client)

        assert response["error"]["code"] == -32001
        assert "mining_paused" in response["error"]["message"]

    async def test_general_exception_handling(self, server, mock_client):
        """Test handling of general exceptions."""
        server.work_cache.get_mining_job.side_effect = Exception("Unexpected error")

        request_line = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "getMiningInfo",
                "params": {},
                "id": 1,
            }
        )

        response = await server._process_request(request_line, mock_client)

        assert response["error"]["code"] == -32000
        assert "Unexpected error" in response["error"]["message"]


@pytest.mark.integration
class TestMinerRpcServerIntegration:
    """Integration tests requiring actual server startup."""

    @pytest.fixture
    async def running_server(self, mock_work_cache, mock_submission_service):
        """Start a test server and clean up after test."""
        config = MinerRpcConfig(
            transport="tcp",
            socket_path="",  # Required field
            port=18445,  # Use different port to avoid conflicts
        )
        server = MinerRpcServer(
            mock_work_cache, mock_submission_service, config, max_pending_submissions=64
        )
        await server.start()
        yield server
        await server.stop()

    async def test_real_socket_request(self, running_server, sample_mining_job):
        """Test actual raw JSON-RPC request to running server."""
        running_server.work_cache.get_mining_job.return_value = sample_mining_job

        # Connect to the server using raw socket
        reader, writer = await asyncio.open_connection("127.0.0.1", 18445)

        try:
            # Send JSON-RPC request
            request = {
                "jsonrpc": "2.0",
                "method": "getMiningInfo",
                "params": {},
                "id": 1,
            }
            request_line = json.dumps(request) + "\n"
            writer.write(request_line.encode())
            await writer.drain()

            # Read response
            response_line = await reader.readline()
            data = json.loads(response_line.decode().strip())

            assert data["jsonrpc"] == "2.0"
            assert data["id"] == 1
            assert data.get("error") is None
            assert "result" in data

        finally:
            writer.close()
            await writer.wait_closed()
