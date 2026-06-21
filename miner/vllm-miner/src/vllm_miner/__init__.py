__all__ = [
    "register_pearl_miner_layer",
    "PearlKernel",
    "pearl_gemm_vanilla",
    "pearl_gemm_noisy",
]


def __getattr__(name: str):
    if name in {"pearl_gemm_noisy", "pearl_gemm_vanilla"}:
        from .gemm_operators import pearl_gemm_noisy, pearl_gemm_vanilla

        return {
            "pearl_gemm_noisy": pearl_gemm_noisy,
            "pearl_gemm_vanilla": pearl_gemm_vanilla,
        }[name]
    if name == "register_pearl_miner_layer":
        from .register import register_pearl_miner_layer

        return register_pearl_miner_layer
    if name == "PearlKernel":
        from .vllm_kernels import PearlKernel

        return PearlKernel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
