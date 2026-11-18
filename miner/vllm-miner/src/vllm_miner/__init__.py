"""Pearl miner vLLM plugin package.

vLLM loads this package (via ``vllm.general_plugins``) inside the worker process
before the worker selects its CUDA device, so importing it must stay CUDA-free.
Public names are resolved lazily (PEP 562) to keep import side effects minimal.
"""

import importlib

# name -> (module, attribute), resolved relative to this package.
_LAZY_EXPORTS = {
    "register_pearl_miner_layer": (".register", "register_pearl_miner_layer"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> object:
    try:
        module, attr = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module, __name__), attr)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
