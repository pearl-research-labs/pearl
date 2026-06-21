from __future__ import annotations

import weakref
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock

import torch


@dataclass(frozen=True, slots=True)
class PreparedBMiningState:
    b_hash: torch.Tensor
    commitment_b: torch.Tensor
    ebl_r_major: torch.Tensor
    ebl_k_major: torch.Tensor
    ebr: torch.Tensor
    ebr_fp16: torch.Tensor

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.b_hash,
                self.commitment_b,
                self.ebl_r_major,
                self.ebl_k_major,
                self.ebr,
                self.ebr_fp16,
            )
        )


@dataclass(slots=True)
class _CacheEntry:
    weak_weight: weakref.ReferenceType[torch.Tensor]
    state: PreparedBMiningState


@dataclass(slots=True)
class PreparedBCacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    oversize: int = 0
    disabled: int = 0


_CACHE: OrderedDict[tuple, _CacheEntry] = OrderedDict()
_STATS = PreparedBCacheStats()
_CACHE_BYTES = 0
_CACHE_LOCK = RLock()


def _cache_key(weight: torch.Tensor, job_key_bytes: bytes, noise_rank: int) -> tuple:
    return (
        weight.device.type,
        weight.device.index,
        weight.data_ptr(),
        tuple(weight.shape),
        tuple(weight.stride()),
        weight.dtype,
        bytes(job_key_bytes),
        int(noise_rank),
    )


def _evict_key_locked(cache_key: tuple) -> None:
    global _CACHE_BYTES
    entry = _CACHE.pop(cache_key, None)
    if entry is not None:
        _CACHE_BYTES -= entry.state.nbytes
        _STATS.evictions += 1


def _evict_lru_until_fits_locked(required_bytes: int, max_cache_bytes: int) -> None:
    while _CACHE and _CACHE_BYTES + required_bytes > max_cache_bytes:
        cache_key, _ = next(iter(_CACHE.items()))
        _evict_key_locked(cache_key)


def get_or_prepare_b_mining_state(
    weight: torch.Tensor,
    job_key_bytes: bytes,
    noise_rank: int,
    max_cache_bytes: int,
    prepare: Callable[[], PreparedBMiningState],
) -> PreparedBMiningState:
    """Return immutable B-side mining state for one exact weight/job/rank tuple."""
    if max_cache_bytes <= 0:
        with _CACHE_LOCK:
            _STATS.disabled += 1
        return prepare()

    cache_key = _cache_key(weight, job_key_bytes, noise_rank)
    with _CACHE_LOCK:
        entry = _CACHE.get(cache_key)
        if entry is not None:
            if entry.weak_weight() is weight:
                _CACHE.move_to_end(cache_key)
                _STATS.hits += 1
                return entry.state
            _evict_key_locked(cache_key)

        _STATS.misses += 1
    state = prepare()
    state_bytes = state.nbytes
    if state_bytes > max_cache_bytes:
        with _CACHE_LOCK:
            _STATS.oversize += 1
        return state

    def _finalize(ref: weakref.ReferenceType[torch.Tensor], key: tuple = cache_key) -> None:
        with _CACHE_LOCK:
            entry = _CACHE.get(key)
            if entry is not None and entry.weak_weight is ref:
                _evict_key_locked(key)

    try:
        weak_weight = weakref.ref(weight, _finalize)
    except TypeError:
        return state

    with _CACHE_LOCK:
        entry = _CACHE.get(cache_key)
        if entry is not None:
            if entry.weak_weight() is weight:
                _CACHE.move_to_end(cache_key)
                return entry.state
            _evict_key_locked(cache_key)

        _evict_lru_until_fits_locked(state_bytes, max_cache_bytes)
        global _CACHE_BYTES
        _CACHE[cache_key] = _CacheEntry(weak_weight=weak_weight, state=state)
        _CACHE_BYTES += state_bytes
    return state


def clear_prepared_b_cache(*, reset_stats: bool = True) -> None:
    global _CACHE_BYTES, _STATS
    with _CACHE_LOCK:
        _CACHE.clear()
        _CACHE_BYTES = 0
        if reset_stats:
            _STATS = PreparedBCacheStats()


def reset_prepared_b_cache_stats() -> None:
    global _STATS
    with _CACHE_LOCK:
        _STATS = PreparedBCacheStats()


def prepared_b_cache_stats() -> PreparedBCacheStats:
    with _CACHE_LOCK:
        return PreparedBCacheStats(
            hits=_STATS.hits,
            misses=_STATS.misses,
            evictions=_STATS.evictions,
            oversize=_STATS.oversize,
            disabled=_STATS.disabled,
        )


def prepared_b_cache_size() -> int:
    with _CACHE_LOCK:
        return len(_CACHE)


def prepared_b_cache_bytes() -> int:
    with _CACHE_LOCK:
        return _CACHE_BYTES
