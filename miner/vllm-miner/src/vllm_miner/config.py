import os

import yaml
from miner_base.settings import MinerSettings


class Config:
    _config: dict = {}

    def __init__(self) -> None:
        self._load_config()
        self.settings = MinerSettings()

    def _load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        try:
            with open(config_path) as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise RuntimeError(
                "config.yaml file not found. Please create it with required configuration.",
            ) from None

    @property
    def gateway_socket_path(self) -> str:
        """Gateway socket path (``MINER_RPC_SOCKET_PATH`` env var overrides config.yaml)."""
        return os.environ.get("MINER_RPC_SOCKET_PATH", self._config["gateway_socket_path"])


# Singleton instance shared by the mining runtime.
config = Config()
