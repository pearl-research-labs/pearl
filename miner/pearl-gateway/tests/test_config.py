"""
Test configuration loading and validation.
"""

import os
from unittest.mock import patch

import pytest
from pearl_gateway.config import (
    MinerRpcConfig,
    PearlConfig,
    PearlGatewayConfig,
    PoolConfig,
    load_config,
)
from pydantic import ValidationError


class TestPearlConfig:
    """Test PearlConfig."""

    def test_pearl_config_defaults(self, mining_address):
        """Test PearlConfig with default values."""
        config = PearlConfig(mining_address=mining_address)

        assert config.rpc_url == "http://0.0.0.0:44107"
        assert config.rpc_user == "user"
        assert config.rpc_password == "pass"
        assert config.refresh_interval_seconds == 1
        assert config.mining_address == mining_address

    def test_solo_mode_requires_mining_address(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValidationError, match="PEARLD_MINING_ADDRESS is required"),
        ):
            load_config()

    def test_pearl_config_custom_values(self, mining_address):
        """Test creating PearlConfig with custom values."""
        config = PearlConfig(
            rpc_url="https://127.0.0.1:18334",
            rpc_user="test_user",
            rpc_password="test_password",
            refresh_interval_seconds=5,
            mining_address=mining_address,
        )

        assert config.rpc_url == "https://127.0.0.1:18334"
        assert config.rpc_user == "test_user"
        assert config.rpc_password == "test_password"
        assert config.refresh_interval_seconds == 5
        assert config.mining_address == mining_address

    def test_pearl_config_env_override(self, mining_address):
        """Test PearlConfig with environment variable overrides."""
        with patch.dict(
            os.environ,
            {
                "PEARLD_RPC_URL": "https://production:8334",
                "PEARLD_RPC_USER": "prod_user",
                "PEARLD_RPC_PASSWORD": "prod_pass",
                "PEARLD_MINING_ADDRESS": mining_address,
            },
        ):
            config = PearlConfig()

            assert config.rpc_url == "https://production:8334"
            assert config.rpc_user == "prod_user"
            assert config.rpc_password == "prod_pass"
            assert config.mining_address == mining_address


class TestMinerRpcConfig:
    """Test MinerRpcConfig."""

    def test_miner_rpc_config_defaults(self):
        """Test MinerRpcConfig with default values."""
        config = MinerRpcConfig()

        assert config.transport == "uds"
        assert config.socket_path == "/tmp/pearlgw.sock"
        assert config.port == 8337
        assert config.host == "localhost"
        assert config.max_pending == 64

    def test_miner_rpc_config_rejects_non_positive_max_pending(self):
        with pytest.raises(ValidationError, match="max_pending"):
            MinerRpcConfig(max_pending=0)

    def test_miner_rpc_config_max_pending_env_override(self):
        with patch.dict(os.environ, {"MINER_RPC_MAX_PENDING": "8"}):
            assert MinerRpcConfig().max_pending == 8

    def test_miner_rpc_config_uds_transport(self):
        """Test MinerRpcConfig with UDS transport."""
        config = MinerRpcConfig(
            transport="uds",
            socket_path="/tmp/test.sock",
        )

        assert config.transport == "uds"
        assert config.socket_path == "/tmp/test.sock"

    def test_miner_rpc_config_tcp_transport(self):
        """Test MinerRpcConfig with TCP transport."""
        config = MinerRpcConfig(
            transport="tcp",
            port=9000,
            host="192.168.1.1",
        )

        assert config.transport == "tcp"
        assert config.port == 9000
        assert config.host == "192.168.1.1"

    def test_miner_rpc_config_invalid_transport(self):
        """Test MinerRpcConfig with invalid transport."""
        with pytest.raises(ValidationError, match="Input should be 'uds' or 'tcp'"):
            MinerRpcConfig(transport="invalid")

    def test_miner_rpc_config_tcp_without_port(self):
        """Test MinerRpcConfig TCP transport without port."""
        with pytest.raises(ValueError, match="TCP transport requires port"):
            MinerRpcConfig(transport="tcp", port=None)

    def test_miner_rpc_config_uds_without_socket_path(self):
        """Test MinerRpcConfig UDS transport without socket_path."""
        with pytest.raises(ValueError, match="UDS transport requires socket_path"):
            MinerRpcConfig(transport="uds", socket_path=None)

    def test_miner_rpc_config_env_override(self):
        """Test MinerRpcConfig with environment variable overrides."""
        with patch.dict(
            os.environ,
            {
                "MINER_RPC_TRANSPORT": "tcp",
                "MINER_RPC_PORT": "9000",
                "MINER_RPC_HOST": "192.168.1.100",
            },
        ):
            config = MinerRpcConfig()

            assert config.transport == "tcp"
            assert config.port == 9000
            assert config.host == "192.168.1.100"


class TestLoadConfig:
    """Test load_config function."""

    def test_load_config_defaults(self, mining_address):
        """Test loading configuration with all defaults."""
        with patch.dict(
            os.environ,
            {"PEARLD_MINING_ADDRESS": mining_address},
        ):
            config = load_config()

        assert isinstance(config, PearlGatewayConfig)
        assert isinstance(config.pearl, PearlConfig)
        assert isinstance(config.miner_rpc, MinerRpcConfig)

        # Verify some defaults
        assert config.pearl.rpc_url == "http://0.0.0.0:44107"
        assert config.miner_rpc.transport == "uds"
        assert config.miner_rpc.max_pending == 64

    def test_load_config_with_env_overrides(self, mining_address):
        """Test loading configuration with environment variable overrides."""
        with patch.dict(
            os.environ,
            {
                "PEARLD_RPC_URL": "https://custom:8334",
                "PEARLD_RPC_USER": "custom_user",
                "PEARLD_MINING_ADDRESS": mining_address,
            },
        ):
            config = load_config()

            assert config.pearl.rpc_url == "https://custom:8334"
            assert config.pearl.rpc_user == "custom_user"
            assert config.pearl.mining_address == mining_address


class TestPoolConfig:
    """Test PoolConfig and pool-mode selection."""

    def test_pool_config_parses_stratum_url(self):
        config = PoolConfig(stratum_url="stratum+ssl://pool.example.com:3443")

        assert config.stratum_host == "pool.example.com"
        assert config.stratum_port == 3443
        assert config.stratum_tls is True
        assert config.address is None
        assert config.worker == "gateway"
        assert config.password == "x"
        assert config.difficulty is None

    def test_pool_config_plain_tcp_default_port(self):
        config = PoolConfig(stratum_url="stratum+tcp://pool.example.com")

        assert config.stratum_port == 3333
        assert config.stratum_tls is False

    def test_pool_config_rejects_url_without_host(self):
        with pytest.raises(ValidationError, match="must include a host"):
            PoolConfig(stratum_url="stratum+tcp://")

    @pytest.mark.parametrize("value", [0, -1, 0.5, "0.5", "1e3", float("inf"), float("nan")])
    def test_pool_config_difficulty_rejects_invalid(self, value):
        with pytest.raises(ValidationError, match="difficulty"):
            PoolConfig(stratum_url="stratum+tcp://pool.example.com", difficulty=value)

    def test_load_config_pool_mode(self, mining_address):
        with patch.dict(
            os.environ,
            {
                "PEARLD_MINING_ADDRESS": mining_address,
                "POOL_STRATUM_URL": "stratum+tcp://pool.example.com:3333",
                "POOL_WORKER": "rig-1",
                "POOL_DIFFICULTY": "128",
            },
        ):
            config = load_config()

        assert config.pool is not None
        assert config.pool.worker == "rig-1"
        assert config.pool.difficulty == 128

    def test_load_config_pool_mode_without_pearld(self, mining_address):
        env = {"POOL_STRATUM_URL": "stratum+tcp://pool.example.com", "POOL_ADDRESS": mining_address}
        with patch.dict(os.environ, env, clear=True):
            config = load_config()

        assert config.pearl.mining_address is None
        assert config.pool is not None
        assert config.pool.address == mining_address

    def test_load_config_pool_mode_requires_an_address(self):
        env = {"POOL_STRATUM_URL": "stratum+tcp://pool.example.com"}
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(ValidationError, match="POOL_ADDRESS"),
        ):
            load_config()

    def test_load_config_solo_mode_without_pool_url(self, mining_address):
        with patch.dict(os.environ, {"PEARLD_MINING_ADDRESS": mining_address}):
            os.environ.pop("POOL_STRATUM_URL", None)
            config = load_config()

        assert config.pool is None
