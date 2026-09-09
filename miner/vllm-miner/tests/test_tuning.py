"""Portable legality and hot-path cache tests for runtime tuning."""

from collections import Counter

import pytest
import vllm_miner.pipeline as pipeline
import vllm_miner.state as state
from pearl_gemm.autotune import get_tuned


def test_mixed_gemm_selection_skips_illegal_nearest_and_preserves_tile_n(monkeypatch):
    records = {
        "mixed_gemm": [
            {
                "shape": {"m": 384, "n": 256, "k": 2048},
                "runtime": 1.0,
                "tile_n": 256,
                "cluster_m": 2,
                "cluster_n": 1,
            },
            {
                "shape": {"m": 512, "n": 256, "k": 2048},
                "runtime": 2.0,
                "tile_n": 256,
                "cluster_m": 1,
                "cluster_n": 1,
            },
        ]
    }

    def resolve(_name, kernel, *, legal=None, require_legal=False, exact=False, **shape):
        return get_tuned(
            kernel,
            records,
            legal=legal,
            require_legal=require_legal,
            exact=exact,
            **shape,
        )

    monkeypatch.setattr(pipeline, "tuned", resolve)
    pipeline._configs.cache_clear()

    *_, gemm = pipeline._configs("synthetic", 384, 256, 2048)

    assert gemm.cluster_m == 1
    assert gemm.tile_n == 256
    # n=256,k=2048 commits the preferred 4x64 tile.
    assert gemm.ltile_cols == pipeline.select_tile(256, 2048).cols == 64

    records["mixed_gemm"] = records["mixed_gemm"][:1]
    pipeline._configs.cache_clear()
    with pytest.raises(ValueError, match="no legal mixed_gemm autotune record"):
        pipeline._configs("synthetic", 384, 256, 2048)


def test_with_committed_leaf_pins_chunk_size_and_clamps_load():
    """A-side autotune knobs stay; ``chunk_size`` is overwritten to the
    weight-side leaf hashed into ``MiningConfiguration`` (and
    ``thread_load_size`` is clamped when it would not divide that leaf)."""
    from vllm_miner.tuning import with_committed_leaf

    pinned = with_committed_leaf(
        {"chunk_size": 256, "thread_load_size": 128, "threads_per_block": 128},
        64,
    )
    assert pinned["chunk_size"] == 64
    assert pinned["thread_load_size"] == 64
    assert pinned["threads_per_block"] == 128
    already = with_committed_leaf({"chunk_size": 256, "thread_load_size": 256}, 1024)
    assert already["chunk_size"] == 1024
    assert already["thread_load_size"] == 256


def test_small_m_routes_to_the_64_row_tile_without_records(monkeypatch):
    """m < 256 on a 4x64-committed layer (the preferred tile) uses the decode
    tile (64, 64) when no exact mixed_gemm record exists."""

    def resolve(_name, kernel, *, legal=None, require_legal=False, exact=False, **shape):
        if kernel == "mixed_gemm":
            return {}
        return {}

    monkeypatch.setattr(pipeline, "tuned", resolve)
    pipeline._configs.cache_clear()

    assert pipeline.select_tile(4096, 4096).cols == 64
    *_, gemm = pipeline._configs("synthetic", 64, 4096, 4096)

    assert (gemm.tile_m, gemm.tile_n) == (64, 64)
    assert (gemm.cluster_m, gemm.cluster_n) == (1, 1)
    assert (gemm.ltile_rows, gemm.ltile_cols) == (4, 64)


def test_gemm_resolution_orders_exact_then_heuristic_then_nearest(monkeypatch):
    """The three resolution tiers, observable through real record ranking
    (``get_tuned``): an exact (m, n, k) record beats the 64-row heuristic;
    without an exact record the heuristic beats a legal nearest record; and
    a shape the heuristic declines (16x32 lottery) falls back to the
    nearest record."""
    records = {
        "mixed_gemm": [
            {
                "shape": {"m": 64, "n": 256, "k": 2048},
                "runtime": 1.0,
                "tile_m": 128,
                "tile_n": 128,
                "cluster_m": 1,
                "cluster_n": 1,
            },
            {
                "shape": {"m": 256, "n": 512, "k": 2048},
                "runtime": 1.0,
                "tile_m": 128,
                "tile_n": 128,
                "cluster_m": 1,
                "cluster_n": 1,
            },
            {
                "shape": {"m": 256, "n": 96, "k": 2048},
                "runtime": 1.0,
                "tile_m": 128,
                "tile_n": 96,
                "cluster_m": 1,
                "cluster_n": 1,
            },
        ]
    }

    def resolve(_name, kernel, *, legal=None, require_legal=False, exact=False, **shape):
        return get_tuned(
            kernel, records, legal=legal, require_legal=require_legal, exact=exact, **shape
        )

    monkeypatch.setattr(pipeline, "tuned", resolve)
    pipeline._configs.cache_clear()

    # Exact record wins over the heuristic inside the small-m window.
    *_, exact_gemm = pipeline._configs("synthetic", 64, 256, 2048)
    assert (exact_gemm.tile_m, exact_gemm.tile_n) == (128, 128)

    # No exact record: the heuristic beats the legal nearest record.
    *_, routed = pipeline._configs("synthetic", 64, 512, 2048)
    assert (routed.tile_m, routed.tile_n) == (64, 64)

    # 16x32 lottery: the heuristic declines and the nearest record is used.
    assert pipeline.select_tile(96, 2048).rows == 16
    *_, tall = pipeline._configs("synthetic", 64, 96, 2048)
    assert (tall.tile_m, tall.ltile_rows) == (128, 16)


def test_exact_mixed_gemm_record_wins_over_small_m_route(monkeypatch):
    """An exact (m, n, k) record is used even inside the small-m window."""

    def resolve(_name, kernel, *, legal=None, require_legal=False, exact=False, **shape):
        if kernel == "mixed_gemm" and exact:
            return {
                "tile_m": 256,
                "tile_n": 64,
                "cluster_m": 2,
                "cluster_n": 1,
            }
        if kernel == "mixed_gemm":
            raise AssertionError("nearest mixed_gemm lookup should not run when exact hits")
        return {}

    monkeypatch.setattr(pipeline, "tuned", resolve)
    pipeline._configs.cache_clear()

    *_, gemm = pipeline._configs("synthetic", 256, 256, 2048)
    assert (gemm.tile_m, gemm.tile_n) == (256, 64)
    assert (gemm.cluster_m, gemm.cluster_n) == (2, 1)
    assert (gemm.ltile_rows, gemm.ltile_cols) == (4, 64)


def test_small_m_keeps_records_for_the_tall_tile(monkeypatch):
    """A 16x32-committed layer (n % 32 but not 64-aligned) cannot run the
    64-row tile: small m still resolves through the saved records."""
    gemm_lookups = []

    def resolve(_name, kernel, *, legal=None, require_legal=False, exact=False, **shape):
        if kernel == "mixed_gemm":
            gemm_lookups.append(shape)
            return {"cluster_m": 1, "cluster_n": 1}
        return {}

    monkeypatch.setattr(pipeline, "tuned", resolve)
    pipeline._configs.cache_clear()

    assert pipeline.select_tile(96, 2048).rows == 16
    *_, gemm = pipeline._configs("synthetic", 64, 96, 2048)

    assert gemm_lookups == [{"m": 64, "n": 96, "k": 2048}]
    assert gemm.ltile_rows == 16
    assert gemm.tile_m != 64


def test_pipeline_configs_are_resolved_once_per_device_shape(monkeypatch):
    calls = Counter()

    def resolve(name, kernel, *, legal=None, require_legal=False, exact=False, **shape):
        calls[(name, kernel, tuple(sorted(shape.items())))] += 1
        kwargs = {}
        if kernel == "mixed_gemm":
            kwargs = {"cluster_m": 1, "cluster_n": 1}
            if not exact:
                assert require_legal
            assert legal is not None and legal(kwargs)
        return kwargs

    monkeypatch.setattr(pipeline, "tuned", resolve)
    pipeline._configs.cache_clear()

    first = pipeline._configs("device_a", 256, 256, 2048)
    assert pipeline._configs("device_a", 256, 256, 2048) is first
    pipeline._configs("device_b", 256, 256, 2048)
    pipeline._configs("device_a", 512, 256, 2048)

    # Each unique (device, shape) consults mixed_gemm once (exact lookup)
    # plus the three non-gemm kernels: 4 + 4 + 4.
    assert sum(calls.values()) == 12
    assert all(count == 1 for count in calls.values())


def test_pipeline_rejects_invalid_noisy_quant_record(monkeypatch):
    """Config-time validation used to live in ``noisy_quant_workspace_shapes``."""

    def resolve(_name, kernel, *, legal=None, require_legal=False, exact=False, **shape):
        if kernel == "noisy_quant":
            return {"noise_rows": 48}
        if kernel == "mixed_gemm":
            kwargs = {"cluster_m": 1, "cluster_n": 1}
            if legal is not None and not legal(kwargs):
                return {}
            return kwargs
        return {}

    monkeypatch.setattr(pipeline, "tuned", resolve)
    pipeline._configs.cache_clear()
    with pytest.raises(ValueError, match="noise_rows"):
        pipeline._configs("synthetic", 256, 256, 2048)


def test_b_configs_rejects_invalid_noisy_quant_record(monkeypatch):
    def resolve(_name, kernel, *, legal=None, require_legal=False, exact=False, **shape):
        if kernel == "noisy_quant":
            return {"noise_rows": 48}
        return {}

    monkeypatch.setattr(state, "tuned", resolve)
    state._b_configs.cache_clear()
    with pytest.raises(ValueError, match="noise_rows"):
        state._b_configs("synthetic", 256, 2048)


def test_unmineable_shape_is_rejected_before_routing(monkeypatch):
    """A shape no committed tile accepts must not resolve a launchable config."""

    def resolve(_name, kernel, *, legal=None, require_legal=False, exact=False, **shape):
        raise AssertionError("unmineable shapes must not consult autotune records")

    monkeypatch.setattr(pipeline, "tuned", resolve)
    pipeline._configs.cache_clear()
    with pytest.raises(ValueError, match="not mineable by any committed lottery tile"):
        pipeline._configs("synthetic", 64, 256, 512)
