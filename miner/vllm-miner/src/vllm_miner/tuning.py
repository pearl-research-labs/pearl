"""Cached device-specific autotune records shared by A and B preparation."""

from collections.abc import Callable
from functools import cache, lru_cache

import torch


@lru_cache(maxsize=32)
def _device_config_name(index: int) -> str:
    return torch.cuda.get_device_name(index).lower().replace(" ", "_").replace("-", "_")


def device_config_name(device: torch.device) -> str:
    """Return the checked-in autotune filename stem for a concrete device."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _device_config_name(index)


@cache
def _load_config(name: str) -> dict:
    from pearl_gemm.autotune import load_config

    return load_config(name) or {}


def tuned(
    config_name: str,
    kernel: str,
    *,
    legal: Callable[[dict], bool] | None = None,
    require_legal: bool = False,
    exact: bool = False,
    **shape: int,
) -> dict:
    """Resolve one record from the cached config for a concrete device family."""
    from pearl_gemm.autotune import get_tuned

    return get_tuned(
        kernel,
        _load_config(config_name),
        legal=legal,
        require_legal=require_legal,
        exact=exact,
        **shape,
    )


def with_committed_leaf(kwargs: dict, chunk_size: int) -> dict:
    """Keep autotune launch knobs, pin ``chunk_size`` to the committed leaf.

    Production overlays the cert-v4 1024-byte leaf so GPU hashing matches
    what pearld can open. ``thread_load_size`` is clamped when it would not
    divide that leaf.
    """
    from pearl_gemm.tensor_hash_plus_stats._merkle_host import SUPPORTED_THREAD_LOAD_SIZES

    pinned = dict(kwargs)
    pinned["chunk_size"] = chunk_size
    load = pinned.get("thread_load_size", 128)
    if chunk_size % load:
        pinned["thread_load_size"] = max(
            size for size in SUPPORTED_THREAD_LOAD_SIZES if chunk_size % size == 0
        )
    return pinned
