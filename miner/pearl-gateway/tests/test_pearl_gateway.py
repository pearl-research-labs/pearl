"""
Test main application functionality.
"""

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from pearl_gateway.pearl_gateway import PearlGateway


@pytest.fixture
def mock_pearl_gateway_components():
    """Mock all PearlGateway components for comprehensive testing."""
    with (
        patch("pearl_gateway.pearl_gateway.WorkCache") as mock_work_cache_cls,
        patch("pearl_gateway.pearl_gateway.PearlNodeClient") as mock_pearl_client_cls,
        patch("pearl_gateway.pearl_gateway.ProofPool") as mock_proof_pool_cls,
        patch("pearl_gateway.pearl_gateway.SubmissionService") as mock_submission_service_cls,
        patch("pearl_gateway.pearl_gateway.MinerRpcServer") as mock_miner_rpc_cls,
        patch("pearl_gateway.pearl_gateway.TemplateScheduler") as mock_scheduler_cls,
        patch("pearl_gateway.pearl_gateway.get_logger") as mock_get_logger,
    ):
        # Create mock instances
        mock_work_cache = AsyncMock()
        mock_pearl_client = AsyncMock()
        mock_pearl_client.__aenter__ = AsyncMock(return_value=mock_pearl_client)
        mock_pearl_client.__aexit__ = AsyncMock(return_value=None)
        mock_proof_pool = AsyncMock()
        mock_submission_service = AsyncMock()
        mock_miner_rpc = AsyncMock()
        mock_scheduler = AsyncMock()
        mock_logger = MagicMock()

        # Configure class constructors to return our mocks
        mock_work_cache_cls.return_value = mock_work_cache
        mock_pearl_client_cls.return_value = mock_pearl_client
        mock_proof_pool_cls.return_value = mock_proof_pool
        mock_submission_service_cls.return_value = mock_submission_service
        mock_miner_rpc_cls.return_value = mock_miner_rpc
        mock_scheduler_cls.return_value = mock_scheduler
        mock_get_logger.return_value = mock_logger

        yield {
            "work_cache": mock_work_cache,
            "pearl_client": mock_pearl_client,
            "proof_pool": mock_proof_pool,
            "submission_service": mock_submission_service,
            "miner_rpc": mock_miner_rpc,
            "scheduler": mock_scheduler,
            "logger": mock_logger,
            "classes": {
                "WorkCache": mock_work_cache_cls,
                "PearlNodeClient": mock_pearl_client_cls,
                "ProofPool": mock_proof_pool_cls,
                "SubmissionService": mock_submission_service_cls,
                "MinerRpcServer": mock_miner_rpc_cls,
                "TemplateScheduler": mock_scheduler_cls,
                "get_logger": mock_get_logger,
            },
        }


class TestPearlGateway:
    """Test PearlGateway class functionality."""

    def test_pearl_gateway_initialization(self, mock_pearl_gateway_components, config_env_vars):
        """Test PearlGateway initialization."""
        gateway = PearlGateway()

        # Verify configuration is loaded
        assert gateway.config is not None
        assert gateway.config.pearl.rpc_user == "test_user"

        # Verify logger setup
        mock_pearl_gateway_components["classes"]["get_logger"].assert_called_once()

        # Verify components are initialized
        mock_pearl_gateway_components["classes"]["WorkCache"].assert_called_once()
        mock_pearl_gateway_components["classes"]["PearlNodeClient"].assert_called_once_with(
            gateway.config.pearl
        )
        mock_pearl_gateway_components["classes"]["SubmissionService"].assert_called_once()
        mock_pearl_gateway_components["classes"]["MinerRpcServer"].assert_called_once()
        mock_pearl_gateway_components["classes"]["TemplateScheduler"].assert_called_once()

        # Verify initial state
        assert gateway.running is False

        # Verify logger message
        mock_pearl_gateway_components["logger"].info.assert_called_with("PearlGateway initialized")

    def test_pearl_gateway_initialization_no_config_path(self, mock_pearl_gateway_components):
        """Test PearlGateway initialization without explicit config path."""
        with patch("pearl_gateway.pearl_gateway.load_config") as mock_load_config:
            mock_load_config.return_value = MagicMock()

            gateway = PearlGateway()

            # Should call load_config with no arguments
            mock_load_config.assert_called_once_with()
            assert gateway.config is not None

    @pytest.mark.asyncio
    async def test_pearl_gateway_start(self, mock_pearl_gateway_components, config_env_vars):
        """Test PearlGateway start method."""
        gateway = PearlGateway()

        await gateway.start()

        # Verify running state
        assert gateway.running is True

        # Verify components are started in order
        mock_pearl_gateway_components["pearl_client"].__aenter__.assert_called_once()
        mock_pearl_gateway_components["scheduler"].start.assert_called_once()
        mock_pearl_gateway_components["proof_pool"].start.assert_called_once()
        mock_pearl_gateway_components["miner_rpc"].start.assert_called_once()

        # Verify log messages
        calls = mock_pearl_gateway_components["logger"].info.call_args_list
        assert call("Starting PearlGateway") in calls
        assert call("PearlGateway started successfully") in calls

    @pytest.mark.asyncio
    async def test_pearl_gateway_start_already_running(
        self, mock_pearl_gateway_components, config_env_vars
    ):
        """Test PearlGateway start method when already running."""
        gateway = PearlGateway()
        gateway.running = True

        await gateway.start()

        # Should not call any start methods
        mock_pearl_gateway_components["pearl_client"].__aenter__.assert_not_called()
        mock_pearl_gateway_components["scheduler"].start.assert_not_called()
        mock_pearl_gateway_components["miner_rpc"].start.assert_not_called()

    @pytest.mark.asyncio
    async def test_pearl_gateway_stop(self, mock_pearl_gateway_components, config_env_vars):
        """Test PearlGateway stop method."""
        gateway = PearlGateway()
        gateway.running = True

        await gateway.stop()

        # Verify running state
        assert gateway.running is False

        # Verify components are stopped in reverse order
        mock_pearl_gateway_components["miner_rpc"].stop.assert_called_once()
        mock_pearl_gateway_components["proof_pool"].stop.assert_called_once()
        mock_pearl_gateway_components["scheduler"].stop.assert_called_once()
        mock_pearl_gateway_components["pearl_client"].__aexit__.assert_called_once_with(
            None, None, None
        )

        # Verify log messages
        calls = mock_pearl_gateway_components["logger"].info.call_args_list
        assert call("Stopping PearlGateway") in calls
        assert call("PearlGateway stopped") in calls

    @pytest.mark.asyncio
    async def test_pearl_gateway_stop_not_running(
        self, mock_pearl_gateway_components, config_env_vars
    ):
        """Test PearlGateway stop method when not running."""
        gateway = PearlGateway()
        gateway.running = False

        await gateway.stop()

        # Should not call any stop methods
        mock_pearl_gateway_components["miner_rpc"].stop.assert_not_called()
        mock_pearl_gateway_components["scheduler"].stop.assert_not_called()
        mock_pearl_gateway_components["pearl_client"].__aexit__.assert_not_called()

    @pytest.mark.asyncio
    async def test_pearl_gateway_start_pearl_client_error(
        self, mock_pearl_gateway_components, config_env_vars
    ):
        """Test PearlGateway start method with PearlClient error."""
        gateway = PearlGateway()

        # Make pearl client start fail
        mock_pearl_gateway_components["pearl_client"].__aenter__.side_effect = ConnectionError(
            "Connection failed"
        )

        with pytest.raises(ConnectionError, match="Connection failed"):
            await gateway.start()

        # Should still be not running
        assert gateway.running is False  # Error prevents running from being set to True

    @pytest.mark.asyncio
    async def test_pearl_gateway_start_scheduler_error(
        self, mock_pearl_gateway_components, config_env_vars
    ):
        """Test PearlGateway start method with scheduler error."""
        gateway = PearlGateway()

        # Make scheduler start fail
        mock_pearl_gateway_components["scheduler"].start.side_effect = RuntimeError(
            "Scheduler failed"
        )

        with pytest.raises(RuntimeError, match="Scheduler failed"):
            await gateway.start()

    @pytest.mark.asyncio
    async def test_pearl_gateway_start_miner_rpc_error(
        self, mock_pearl_gateway_components, config_env_vars
    ):
        """Test PearlGateway start method with miner RPC error."""
        gateway = PearlGateway()

        # Make miner RPC start fail
        mock_pearl_gateway_components["miner_rpc"].start.side_effect = OSError("Port in use")

        with pytest.raises(OSError, match="Port in use"):
            await gateway.start()

    @pytest.mark.asyncio
    async def test_pearl_gateway_full_lifecycle(
        self, mock_pearl_gateway_components, config_env_vars
    ):
        """Test full PearlGateway lifecycle: init -> start -> stop."""
        gateway = PearlGateway()

        # Initial state
        assert gateway.running is False

        # Start
        await gateway.start()
        assert gateway.running is True

        # Stop
        await gateway.stop()
        assert gateway.running is False

        # Verify all components were properly managed
        mock_pearl_gateway_components["pearl_client"].__aenter__.assert_called_once()
        mock_pearl_gateway_components["scheduler"].start.assert_called_once()
        mock_pearl_gateway_components["proof_pool"].start.assert_called_once()
        mock_pearl_gateway_components["miner_rpc"].start.assert_called_once()

        mock_pearl_gateway_components["miner_rpc"].stop.assert_called_once()
        mock_pearl_gateway_components["proof_pool"].stop.assert_called_once()
        mock_pearl_gateway_components["scheduler"].stop.assert_called_once()
        mock_pearl_gateway_components["pearl_client"].__aexit__.assert_called_once()
