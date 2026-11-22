import os
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pearl_gateway.comm.stratum import stratum_host, stratum_port, stratum_uses_tls


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
    # Required in solo mode; in pool mode it is the fallback for POOL_ADDRESS.
    mining_address: str | None = None


class PoolConfig(BaseSettings):
    """Pool Stratum connection configuration (pool mode).

    Setting POOL_STRATUM_URL switches the gateway from solo to pool mode.

    Environment variables override defaults:
    - POOL_STRATUM_URL (e.g. stratum+ssl://host:3443)
    - POOL_ADDRESS (payout address; defaults to PEARLD_MINING_ADDRESS)
    - POOL_WORKER
    - POOL_PASSWORD
    - POOL_RECONNECT_DELAY_SECONDS
    - POOL_MAX_RECONNECT_DELAY_SECONDS
    - POOL_DIFFICULTY
    """

    model_config = SettingsConfigDict(env_prefix="POOL_")

    stratum_url: str
    address: str | None = None
    worker: str = "gateway"
    password: str = "x"
    reconnect_delay_seconds: int = 1
    max_reconnect_delay_seconds: int = 60
    # Start share difficulty in the pool's unit, sent as d=N in the password.
    difficulty: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_stratum_url(self) -> "PoolConfig":
        if stratum_host(self.stratum_url) is None:
            raise ValueError("POOL_STRATUM_URL must include a host, e.g. stratum+ssl://host:3333")
        return self

    @property
    def stratum_host(self) -> str:
        return stratum_host(self.stratum_url)  # type: ignore[return-value]

    @property
    def stratum_port(self) -> int:
        return stratum_port(self.stratum_url)

    @property
    def stratum_tls(self) -> bool:
        return stratum_uses_tls(self.stratum_url)


class MinerRpcConfig(BaseSettings):
    """Miner RPC server configuration.

    Environment variables override defaults:
    - MINER_RPC_TRANSPORT
    - MINER_RPC_SOCKET_PATH
    - MINER_RPC_PORT
    - MINER_RPC_HOST
    - MINER_RPC_MAX_PENDING
    """

    model_config = SettingsConfigDict(env_prefix="MINER_RPC_")

    transport: Literal["uds", "tcp"] = "uds"
    socket_path: str | None = "/tmp/pearlgw.sock"
    port: int | None = 8337
    host: str | None = "localhost"
    # Max proof submissions queued before new ones are rejected at the RPC boundary.
    max_pending: int = Field(default=64, ge=1)

    def model_post_init(self, __context) -> None:
        """Validate configuration after initialization."""
        if self.transport == "tcp" and not self.port:
            raise ValueError("TCP transport requires port to be specified")
        elif self.transport == "uds" and not self.socket_path:
            raise ValueError("UDS transport requires socket_path")


class PearlGatewayConfig(BaseModel):
    """Complete PearlGateway configuration.

    All nested configs support environment variable overrides.
    See individual config classes for available environment variables.
    """

    pearl: PearlConfig
    miner_rpc: MinerRpcConfig
    # Set => pool mode (work from a Stratum pool); None => solo mode (pearld).
    pool: PoolConfig | None = None

    @model_validator(mode="after")
    def validate_mining_address(self) -> "PearlGatewayConfig":
        if self.pool is None:
            if not self.pearl.mining_address:
                raise ValueError("PEARLD_MINING_ADDRESS is required in solo mode")
        elif not (self.pool.address or self.pearl.mining_address):
            raise ValueError("Pool mode needs POOL_ADDRESS (or PEARLD_MINING_ADDRESS)")
        return self


def load_config() -> PearlGatewayConfig:
    """Load configuration from environment variables with defaults.

    Priority (highest to lowest):
    1. Environment variables (e.g., PEARLD_RPC_URL)
    2. Default values defined in the config classes

    Setting POOL_STRATUM_URL enables pool mode.

    Returns:
        PearlGatewayConfig: Fully configured gateway settings
    """
    return PearlGatewayConfig(
        pearl=PearlConfig(),
        miner_rpc=MinerRpcConfig(),
        pool=PoolConfig() if os.environ.get("POOL_STRATUM_URL") else None,
    )
