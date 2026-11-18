from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MinerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="miner_")

    no_gateway: bool = False
    no_mining: bool = False
    skip_block_submission: bool = False
    # Bound proof construction plus gateway RPC work accepted from GPU owners.
    submission_inflight_limit: int = Field(default=8, ge=1)
    register_quantization: bool = True
