import threading

import torch
from vllm_miner.prepared_b_mining_state import (
    PreparedBMiningState,
    clear_prepared_b_cache,
    get_or_prepare_b_mining_state,
    prepared_b_cache_bytes,
    prepared_b_cache_size,
    prepared_b_cache_stats,
)


def _state(fill: int, elems: int = 4) -> PreparedBMiningState:
    def tensor(multiplier: int = 1) -> torch.Tensor:
        return torch.full((elems * multiplier,), fill, dtype=torch.uint8)

    return PreparedBMiningState(
        b_hash=tensor(),
        commitment_b=tensor(),
        ebl_r_major=tensor(2),
        ebl_k_major=tensor(2),
        ebr=tensor(2),
        ebr_fp16=torch.full((elems,), fill, dtype=torch.float16),
    )


def setup_function() -> None:
    clear_prepared_b_cache()


def teardown_function() -> None:
    clear_prepared_b_cache()


def test_prepared_b_cache_hits_exact_weight_job_and_rank() -> None:
    weight = torch.empty((4, 4), dtype=torch.int8)
    calls = 0

    def prepare() -> PreparedBMiningState:
        nonlocal calls
        calls += 1
        return _state(calls)

    first = get_or_prepare_b_mining_state(weight, b"job-a", 128, 10_000, prepare)
    second = get_or_prepare_b_mining_state(weight, b"job-a", 128, 10_000, prepare)

    assert first is second
    assert calls == 1
    assert prepared_b_cache_size() == 1
    stats = prepared_b_cache_stats()
    assert stats.misses == 1
    assert stats.hits == 1


def test_prepared_b_cache_misses_on_job_rank_and_exact_identity() -> None:
    weight = torch.empty((4, 4), dtype=torch.int8)
    same_storage_view = weight.as_strided(weight.shape, weight.stride())
    calls = 0

    def prepare() -> PreparedBMiningState:
        nonlocal calls
        calls += 1
        return _state(calls)

    base = get_or_prepare_b_mining_state(weight, b"job-a", 128, 100_000, prepare)
    new_job = get_or_prepare_b_mining_state(weight, b"job-b", 128, 100_000, prepare)
    new_rank = get_or_prepare_b_mining_state(weight, b"job-a", 64, 100_000, prepare)
    new_identity = get_or_prepare_b_mining_state(
        same_storage_view, b"job-a", 128, 100_000, prepare
    )

    assert len({id(base), id(new_job), id(new_rank), id(new_identity)}) == 4
    assert calls == 4
    assert prepared_b_cache_stats().misses == 4


def test_prepared_b_cache_enforces_byte_budget_with_lru_eviction() -> None:
    weight_a = torch.empty((4, 4), dtype=torch.int8)
    weight_b = torch.empty((4, 4), dtype=torch.int8)
    one_entry_bytes = _state(1).nbytes

    get_or_prepare_b_mining_state(weight_a, b"job", 128, one_entry_bytes + 1, lambda: _state(1))
    get_or_prepare_b_mining_state(weight_b, b"job", 128, one_entry_bytes + 1, lambda: _state(2))

    assert prepared_b_cache_size() == 1
    assert prepared_b_cache_bytes() <= one_entry_bytes + 1
    assert prepared_b_cache_stats().evictions == 1


def test_concurrent_same_key_misses_do_not_double_count_cache_bytes() -> None:
    weight = torch.empty((4, 4), dtype=torch.int8)
    barrier = threading.Barrier(2, timeout=5)
    calls = 0
    calls_lock = threading.Lock()
    errors: list[BaseException] = []
    results: list[PreparedBMiningState] = []
    one_entry_bytes = _state(1).nbytes

    def prepare() -> PreparedBMiningState:
        nonlocal calls
        with calls_lock:
            calls += 1
            fill = calls
        barrier.wait()
        return _state(fill)

    def worker() -> None:
        try:
            results.append(
                get_or_prepare_b_mining_state(
                    weight,
                    b"job",
                    128,
                    100_000,
                    prepare,
                )
            )
        except BaseException as exc:  # pragma: no cover - surfaced by assertion below.
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert calls == 2
    assert len(results) == 2
    assert prepared_b_cache_size() == 1
    assert prepared_b_cache_bytes() == one_entry_bytes


def test_prepared_b_cache_skips_oversize_and_disabled_entries() -> None:
    weight = torch.empty((4, 4), dtype=torch.int8)
    state = _state(1)

    get_or_prepare_b_mining_state(weight, b"job", 128, state.nbytes - 1, lambda: state)
    assert prepared_b_cache_size() == 0
    assert prepared_b_cache_stats().oversize == 1

    get_or_prepare_b_mining_state(weight, b"job", 128, 0, lambda: state)
    assert prepared_b_cache_size() == 0
    assert prepared_b_cache_stats().disabled == 1
