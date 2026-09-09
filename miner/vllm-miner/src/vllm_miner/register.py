"""Pearl mining plugin registration for vLLM."""

from miner_base.settings import MinerSettings
from vllm.model_executor.layers.quantization import register_quantization_config

from .engine_lifecycle import install_engine_reload_boundary
from .vllm_pearl_config import PearlConfig
from .worker_hooks import install_pearl_worker_runtime_hook


def register_pearl_miner_layer() -> None:
    """Register the Pearl quantization method and the worker runtime hooks."""
    install_engine_reload_boundary()

    # Defer CUDA-sensitive setup until the worker binds its device.
    install_pearl_worker_runtime_hook()

    if MinerSettings().register_quantization:
        register_quantization_config("pearl")(PearlConfig)
