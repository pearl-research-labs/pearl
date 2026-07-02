from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PearlConfig(BaseSettings):
    """Pearl node connection configuration.

    Environment variables override defaults:
    - PEARLD_RPC_URL
    - PEARLD_RPC_USER
    - PEARLD_RPC_PASSWORD
    - PEARLD_REFRESH_INTERVAL_SECONDS
    - PEARLD_MINING_ADDRESS
    """

    model_config = SettingsConfigDict(env_prefix="PEARLD_")

    rpc_url: str = "http://0.0.0.0:44107"
    rpc_user: str = "user"
    rpc_password: str = "pass"
    refresh_interval_seconds: int = 1
    mining_address: str


class MinerRpcConfig(BaseSettings):
    """Miner RPC server configuration.

    Environment variables override defaults:
    - MINER_RPC_TRANSPORT
    - MINER_RPC_SOCKET_PATH
    - MINER_RPC_PORT
    - MINER_RPC_HOST
    """

    model_config = SettingsConfigDict(env_prefix="MINER_RPC_")

    transport: Literal["uds", "tcp"] = "uds"
    socket_path: str | None = "/tmp/pearlgw.sock"
    port: int | None = 8337
    host: str | None = "localhost"

    def model_post_init(self, __context) -> None:
        """Validate configuration after initialization."""
        if self.transport == "tcp" and not self.port:
            raise ValueError("TCP transport requires port to be specified")
        elif self.transport == "uds" and not self.socket_path:
            raise ValueError("UDS transport requires socket_path")


class ProofConfig(BaseSettings):
    """ZK proof generation worker configuration.

    Proof generation is CPU-bound and holds the GIL, so it runs in a warmed
    worker process off the gateway's event loop.

    Environment variables override defaults:
    - GATEWAY_PROOF_MAX_PENDING
    """

    model_config = SettingsConfigDict(env_prefix="GATEWAY_PROOF_")

    # Max proof submissions queued before new ones are rejected (enforced at
    # the RPC boundary), bounding gateway memory when miners outpace the worker.
    max_pending: int = Field(default=64, ge=1)


class PearlGatewayConfig(BaseModel):
    """Complete PearlGateway configuration.

    All nested configs support environment variable overrides.
    See individual config classes for available environment variables.
    """

    pearl: PearlConfig
    miner_rpc: MinerRpcConfig
    proof: ProofConfig


def load_config() -> PearlGatewayConfig:
    """Load configuration from environment variables with defaults.

    Priority (highest to lowest):
    1. Environment variables (e.g., PEARLD_RPC_URL)
    2. Default values defined in the config classes

    Returns:
        PearlGatewayConfig: Fully configured gateway settings
    """
    return PearlGatewayConfig(
        pearl=PearlConfig(),
        miner_rpc=MinerRpcConfig(),
        proof=ProofConfig(),
    )
