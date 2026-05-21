# Top-level re-exports are gated on the optional vllm / compressed-tensors
# stack so that consumers who only want the lightweight mining_state module
# (e.g. pearl-stratum's _miner_driver_*) can import this package without
# pulling the entire LLM toolchain. If you actually need PearlKernel /
# register_pearl_miner_layer / the gemm_operators glue, install the full
# `[vllm]` extra: pip install -e '.[vllm]'.
try:
    from .gemm_operators import pearl_gemm_noisy, pearl_gemm_vanilla
    from .register import register_pearl_miner_layer
    from .vllm_kernels import PearlKernel

    __all__ = [
        "register_pearl_miner_layer",
        "PearlKernel",
        "pearl_gemm_vanilla",
        "pearl_gemm_noisy",
    ]
except ImportError as _e:  # noqa: F841
    # vllm / compressed-tensors / transformers / similar stack missing.
    # mining_state.py and config.py still import fine via
    # `from vllm_miner.mining_state import ...`.
    __all__ = []
