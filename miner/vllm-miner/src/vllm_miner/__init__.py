"""Pearl miner vLLM plugin package.

vLLM loads this package before the worker binds its CUDA device, so top-level
imports stay strictly CUDA-free via PEP 562 lazy lookup.
"""

import importlib

__all__ = ["register_pearl_miner_layer"]


def __getattr__(name: str) -> object:
    if name == "register_pearl_miner_layer":
        return importlib.import_module(".plugin", __name__).register_pearl_miner_layer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
