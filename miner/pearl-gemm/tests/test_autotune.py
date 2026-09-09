"""Autotune selection and persistence tests."""

import json
from dataclasses import fields

import pytest

from pearl_gemm import TensorHashConfig, get_tensor_hash_plus_stats_config
from pearl_gemm.autotune import (
    SMALL_M_LIMIT,
    TUNERS,
    _stats_chunk_sizes,
    _sweep,
    autotune,
    get_tuned,
    merge_records,
    mixed_gemm_space,
    mixed_gemm_sweep_space,
    route_small_m_mixed_gemm,
    tensor_hash_plus_stats_space,
)
from pearl_gemm.tensor_hash_plus_stats import _blake3, tensor_hash_plus_stats_record_is_legal
from pearl_gemm.tensor_hash_plus_stats._merkle_host import (
    SUPPORTED_LEAVES_PER_MT_BLOCK,
    SUPPORTED_NUM_STAGES,
    SUPPORTED_THREAD_LOAD_SIZES,
    SUPPORTED_THREADS_PER_BLOCK,
    _effective_chunks_per_thread,
    _validate_stats_compatible,
    supported_chunk_sizes,
)
from pearl_gemm.tensor_hash_plus_stats._merkle_tree_roots_kernel import (
    SUPPORTED_CHUNKS_PER_THREAD,
)


def test_mixed_gemm_space_filters_tile_n_by_lottery_width():
    wide = mixed_gemm_space(4, 128)
    assert {cfg["tile_n"] for cfg in wide} == {128, 256}
    assert all("ltile_cols" not in cfg for cfg in wide)

    narrow = mixed_gemm_space(4, 64)
    assert {cfg["tile_n"] for cfg in narrow} == {64, 128, 192, 256}
    assert all(cfg["ltile_cols"] == 64 for cfg in narrow)
    assert all(cfg["tile_n"] % 64 == 0 for cfg in narrow)

    tall = mixed_gemm_space(16, 32)
    assert {cfg["tile_n"] for cfg in tall} == {32, 64, 96, 128, 160, 192, 224, 256}
    assert all(cfg["ltile_cols"] == 32 and cfg["ltile_rows"] == 16 for cfg in tall)
    assert all(cfg["tile_n"] % 32 == 0 for cfg in tall)


def test_small_m_sweep_includes_routed_64_row_tile():
    routed = route_small_m_mixed_gemm(64, 576, 6144, ltile_rows=4, ltile_cols=64)
    assert routed is not None
    space = mixed_gemm_sweep_space(64, 576, 6144)
    assert dict(routed) in space
    assert all(cfg.get("tile_m") != 64 for cfg in mixed_gemm_space(4, 64))
    # Prefill m is outside the route, so the 64-row tile stays out.
    assert all(cfg.get("tile_m") != 64 for cfg in mixed_gemm_sweep_space(4096, 576, 6144))


def test_autotune_kernels_subset_skips_other_tuners(monkeypatch, tmp_path):
    import pearl_gemm.autotune as module

    output = tmp_path / "tuned.json"
    output.write_text(
        json.dumps(
            {
                "pre_quant": [{"shape": {"m": 256, "k": 512}, "runtime": 1.0, "kept": True}],
                "mixed_gemm": [],
            }
        )
    )
    calls: list[str] = []

    def mixed(m, n, k):
        calls.append("mixed_gemm")
        return [{"shape": {"m": m, "n": n, "k": k}, "runtime": 1.5, "tile_m": 256}]

    def forbidden(*_args):
        raise AssertionError("non-selected tuner must not run")

    monkeypatch.setattr(
        module,
        "TUNERS",
        {
            "pre_quant": forbidden,
            "mixed_gemm": mixed,
        },
    )
    result = autotune([(256, 128, 512)], output, kernels=["mixed_gemm"])
    assert calls == ["mixed_gemm"]
    assert result["pre_quant"] == [{"shape": {"m": 256, "k": 512}, "runtime": 1.0, "kept": True}]
    assert result["mixed_gemm"][0]["tile_m"] == 256


def test_route_small_m_mixed_gemm_routes_to_the_64_row_tile():
    """Small-m routing: (64, 64) for the 64-wide lottery, (64, ltile_cols)
    otherwise, and None whenever the committed geometry or shape cannot run it."""
    routed = route_small_m_mixed_gemm(64, 2752, 6144, ltile_rows=4, ltile_cols=64)
    assert routed == {
        "tile_m": 64,
        "tile_n": 64,
        "cluster_m": 1,
        "cluster_n": 1,
        "ltile_rows": 4,
        "ltile_cols": 64,
    }
    # 128-wide committed lottery: the narrowest tile_n it divides.
    wide = route_small_m_mixed_gemm(128, 4096, 4096, ltile_rows=4, ltile_cols=128)
    assert (wide["tile_m"], wide["tile_n"], wide["ltile_cols"]) == (64, 128, 128)
    # The limit is inclusive; above it the saved records stay in charge.
    at_limit = route_small_m_mixed_gemm(SMALL_M_LIMIT, 2752, 6144, ltile_rows=4, ltile_cols=64)
    assert (at_limit["tile_m"], at_limit["tile_n"]) == (64, 64)
    assert (
        route_small_m_mixed_gemm(SMALL_M_LIMIT + 64, 2752, 6144, ltile_rows=4, ltile_cols=64)
        is None
    )
    # The 16-row lottery family cannot run a 64-row kernel tile.
    assert route_small_m_mixed_gemm(64, 2752, 6144, ltile_rows=16, ltile_cols=32) is None
    # Shapes the routed config cannot launch fall back to the records.
    assert route_small_m_mixed_gemm(64, 100, 6144, ltile_rows=4, ltile_cols=64) is None
    assert route_small_m_mixed_gemm(64, 2752, 1000, ltile_rows=4, ltile_cols=64) is None
    # Hot-path LRU: repeated resolution returns the cached object, which is
    # therefore shared and immutable (a mutation would poison the cache).
    assert route_small_m_mixed_gemm(64, 2752, 6144, ltile_rows=4, ltile_cols=64) is routed
    with pytest.raises(TypeError):
        routed["tile_m"] = 128


def test_get_tuned_prefers_exact_then_nearest_shape():
    config = {
        "mixed_gemm": [
            {
                "shape": {"m": 256, "n": 128, "k": 512},
                "runtime": 2.0,
                "tile_m": 256,
            },
            {
                "shape": {"m": 4096, "n": 8192, "k": 4096},
                "runtime": 1.0,
                "tile_m": 128,
            },
        ]
    }
    assert get_tuned("mixed_gemm", config, m=256, n=128, k=512) == {"tile_m": 256}
    assert get_tuned("mixed_gemm", config, m=2048, n=8192, k=4096) == {"tile_m": 128}
    assert get_tuned("missing", config, m=1) == {}
    assert get_tuned("mixed_gemm", config, exact=True, m=256, n=128, k=512) == {"tile_m": 256}
    assert get_tuned("mixed_gemm", config, exact=True, m=2048, n=8192, k=4096) == {}


def test_get_tuned_drops_retired_noisy_quant_ksplit():
    """Historical ``noise_ksplit`` must not reach ``NoisyQuantConfig``."""
    config = {
        "noisy_quant": [
            {
                "shape": {"m": 256, "k": 4096},
                "runtime": 1.0,
                "noise_rows": 32,
                "noise_bk": 256,
                "noise_ksplit": 2,
            }
        ]
    }
    assert get_tuned("noisy_quant", config, m=256, k=4096) == {
        "noise_rows": 32,
        "noise_bk": 256,
    }


def test_merge_records_strips_retired_noisy_quant_ksplit():
    saved = {
        "noisy_quant": [
            {
                "shape": {"m": 256, "k": 4096},
                "runtime": 1.0,
                "noise_rows": 32,
                "noise_ksplit": 4,
            }
        ]
    }
    merged = merge_records(saved, {})
    assert merged["noisy_quant"][0] == {
        "shape": {"m": 256, "k": 4096},
        "runtime": 1.0,
        "noise_rows": 32,
    }


def test_get_tuned_skips_illegal_nearest_record():
    config = {
        "mixed_gemm": [
            {
                "shape": {"m": 384, "n": 256, "k": 1024},
                "runtime": 1.0,
                "cluster_m": 2,
            },
            {
                "shape": {"m": 1024, "n": 256, "k": 1024},
                "runtime": 2.0,
                "cluster_m": 1,
            },
        ]
    }

    selected = get_tuned(
        "mixed_gemm",
        config,
        legal=lambda kwargs: kwargs["cluster_m"] == 1,
        m=384,
        n=256,
        k=1024,
    )

    assert selected == {"cluster_m": 1}
    assert (
        get_tuned(
            "mixed_gemm",
            config,
            legal=lambda _kwargs: False,
            m=384,
            n=256,
            k=1024,
        )
        == {}
    )
    with pytest.raises(ValueError, match="no legal mixed_gemm"):
        get_tuned(
            "mixed_gemm",
            config,
            legal=lambda _kwargs: False,
            require_legal=True,
            m=384,
            n=256,
            k=1024,
        )


def test_sweep_compiles_every_candidate_before_timing(monkeypatch):
    """JIT clock-drop is paid for the whole space before any timed replay."""
    import pearl_gemm.autotune as module

    events: list[str] = []

    def fake_prime(call_slot):
        events.append("compile")
        call_slot(0)

    def fake_time(call_slot):
        events.append("time")
        return 1e-6

    monkeypatch.setattr(module, "_prime_compile", fake_prime)
    monkeypatch.setattr(module, "time_graph", fake_time)

    def build_launch(cfg):
        events.append(f"build:{cfg['x']}")
        if cfg["x"] == 2:
            raise AssertionError("invalid")
        return lambda _i: None

    records = _sweep(
        [{"x": 1}, {"x": 2}, {"x": 3}],
        {"m": 1, "k": 1},
        build_launch,
    )
    assert [r["x"] for r in records] == [1, 3]
    assert events == [
        "build:1",
        "compile",
        "build:2",
        "build:3",
        "compile",
        "build:1",
        "time",
        "build:3",
        "time",
    ]


def test_autotune_writes_only_fastest_records(monkeypatch, tmp_path):
    import pearl_gemm.autotune as module

    def mk_tuner(m, k):
        return [
            {"shape": {"m": m, "k": k}, "runtime": 2.0, "choice": "slow"},
            {"shape": {"m": m, "k": k}, "runtime": 1.0, "choice": "fast"},
        ]

    def mnk_tuner(m, n, k):
        return [
            {"shape": {"m": m, "n": n, "k": k}, "runtime": 3.0, "choice": "slow"},
            {"shape": {"m": m, "n": n, "k": k}, "runtime": 1.5, "choice": "fast"},
        ]

    monkeypatch.setattr(
        module,
        "TUNERS",
        {
            "tensor_hash_plus_stats": mk_tuner,
            "noisy_quant": mk_tuner,
            "mixed_gemm": mnk_tuner,
        },
    )
    output = tmp_path / "tuned.json"
    result = autotune([(256, 128, 512)], output)

    assert all(len(records) == 1 for records in result.values())
    assert {records[0]["choice"] for records in result.values()} == {"fast"}
    assert json.loads(output.read_text()) == result


def test_autotune_subset_keeps_unswept_and_empty_sweep_records(monkeypatch, tmp_path):
    """A retune must cost only the shapes it sweeps.

    ``untouched`` stands for every shape left out of a subset retune.
    ``narrow`` is an explicitly synthetic no-winner: the tuner returns none
    for that shape even though n=576 is in the committed 64-wide space
    (see ``test_mixed_gemm_space_filters_tile_n_by_lottery_width``).
    """
    import pearl_gemm.autotune as module

    saved = {
        "mixed_gemm": [
            {"shape": {"m": 8192, "n": 576, "k": 6144}, "runtime": 3.0, "ltile_cols": 64},
            {"shape": {"m": 8192, "n": 8192, "k": 4096}, "runtime": 2.0, "choice": "untouched"},
            {"shape": {"m": 256, "n": 128, "k": 512}, "runtime": 9.0, "choice": "stale"},
        ],
        "pre_quant": [{"shape": {"m": 4096, "k": 2048}, "runtime": 1.0, "choice": "untouched"}],
    }
    output = tmp_path / "tuned.json"
    output.write_text(json.dumps(saved))

    def mnk_tuner(m, n, k):
        if n == 576:  # synthetic no-winner; n=576 is launchable in production
            return []
        return [{"shape": {"m": m, "n": n, "k": k}, "runtime": 1.5, "choice": "fresh"}]

    monkeypatch.setattr(module, "TUNERS", {"mixed_gemm": mnk_tuner})
    result = autotune([(8192, 576, 6144), (256, 128, 512)], output)

    by_shape = {
        (r["shape"]["m"], r["shape"]["n"], r["shape"]["k"]): r for r in result["mixed_gemm"]
    }
    assert by_shape[(8192, 576, 6144)]["ltile_cols"] == 64
    assert by_shape[(8192, 8192, 4096)]["choice"] == "untouched"
    assert by_shape[(256, 128, 512)]["choice"] == "fresh"
    assert result["pre_quant"] == saved["pre_quant"]
    assert json.loads(output.read_text()) == result


def test_tensor_hash_space_matches_stats_path_legal_grid():
    """The sweep is the full grid over the record-legal knobs: sync legs pin
    their dead knobs, and non-1024 leaves stay out (``commit_planes`` /
    ``MatrixMerkleProof`` are fixed to 1024, so their records could not be
    opened; other leaves are a protocol/verifier experiment)."""
    assert set(supported_chunk_sizes()) == set(range(64, 4096 + 1, 64))
    # The kernel's stats gate: sub-block leaves plus whole multiples of one.
    assert set(_stats_chunk_sizes()) == set(range(64, 512, 64)) | set(range(512, 4096 + 1, 512))
    space = tensor_hash_plus_stats_space()
    assert {cfg["threads_per_block"] for cfg in space} == set(SUPPORTED_THREADS_PER_BLOCK)
    assert {cfg["num_stages"] for cfg in space} == set(SUPPORTED_NUM_STAGES)
    assert {cfg["thread_load_size"] for cfg in space} == set(SUPPORTED_THREAD_LOAD_SIZES)
    assert {cfg["leaves_per_mt_block"] for cfg in space} == set(SUPPORTED_LEAVES_PER_MT_BLOCK)
    assert {cfg["mad_rot"] for cfg in space} == {False, True}
    assert {cfg["sync_loads"] for cfg in space} == {False, True}
    assert {cfg["chunks_per_thread"] for cfg in space} == set(SUPPORTED_CHUNKS_PER_THREAD)
    assert {cfg["chunk_size"] for cfg in space} == {_blake3.CHUNK_SIZE}
    # The sync path has no TMA ring and never dispatches the rotate offload,
    # so its legs pin those knobs rather than re-time identical kernels.
    sync_legs = [cfg for cfg in space if cfg["sync_loads"]]
    assert sync_legs and all(not cfg["mad_rot"] for cfg in sync_legs)
    assert len({(cfg["num_stages"], cfg["thread_load_size"]) for cfg in sync_legs}) == 1
    for cfg in space:
        TensorHashConfig(**cfg)
    assert all(tensor_hash_plus_stats_record_is_legal(cfg) for cfg in space)


def test_tensor_hash_space_covers_every_config_field_or_documents_why_not():
    """Every ``TensorHashConfig`` field is swept, and the lifted stats gates
    hold: both load paths and every sub-block or whole-multiple leaf fuse
    stats, and coarsening rides the stats path at whole-block leaves while
    sub-block leaves force k=1 (their cross-thread fold stages one pair per
    chunk message block)."""
    swept = set().union(*(cfg.keys() for cfg in tensor_hash_plus_stats_space()))
    config_fields = {field.name for field in fields(TensorHashConfig)}
    assert config_fields - swept == set()

    for chunk_size in (64, 448, 512, 1536, _blake3.CHUNK_SIZE, 4096):
        _validate_stats_compatible(chunk_size)
    for chunk_size in (576, 1088, 4032):  # above a stats block, not a multiple
        with pytest.raises(ValueError, match="stats requires"):
            _validate_stats_compatible(chunk_size)
    blob_bytes = (32 * 1024 * 1024, 8 * 1024 * 1024)  # coarsening-friendly sizes
    for chunks_per_thread in SUPPORTED_CHUNKS_PER_THREAD:
        assert (
            _effective_chunks_per_thread(
                chunks_per_thread, blob_bytes, _blake3.CHUNK_SIZE, with_stats=True
            )
            == chunks_per_thread
        )
        assert (
            _effective_chunks_per_thread(chunks_per_thread, blob_bytes, 256, with_stats=True) == 1
        )


def test_get_tuned_skips_malformed_record_envelopes():
    """A corrupt persisted record (missing or non-numeric shape) is skipped
    instead of crashing distance ranking."""
    config = {
        "tensor_hash_plus_stats": [
            {"runtime": 1.0, "threads_per_block": 512},  # no shape at all
            {"shape": {"m": "big", "k": 4096}, "runtime": 1.0, "threads_per_block": 512},
            {"shape": {"m": 256, "k": 4096}, "runtime": 2.0, "threads_per_block": 256},
        ]
    }
    assert get_tuned("tensor_hash_plus_stats", config, m=256, k=4096) == {"threads_per_block": 256}
    config["tensor_hash_plus_stats"] = config["tensor_hash_plus_stats"][:2]
    assert get_tuned("tensor_hash_plus_stats", config, m=256, k=4096) == {}


def test_get_tensor_hash_config_prefers_exact_then_nearest_legal():
    records = {
        "tensor_hash_plus_stats": [
            {
                "shape": {"m": 256, "k": 4096},
                "runtime": 1.0,
                "threads_per_block": 128,
                "num_stages": 4,
                "thread_load_size": 64,
                "chunk_size": 1024,
            },
            {
                "shape": {"m": 8192, "k": 512},
                "runtime": 2.0,
                "threads_per_block": 256,
                "num_stages": 3,
                "thread_load_size": 128,
                "chunk_size": 1024,
            },
        ]
    }
    exact = get_tensor_hash_plus_stats_config(256, 4096, records=records)
    assert exact == TensorHashConfig(
        threads_per_block=128,
        num_stages=4,
        thread_load_size=64,
        chunk_size=1024,
    )
    assert get_tensor_hash_plus_stats_config(300, 4000, records=records).thread_load_size == 64
    assert get_tensor_hash_plus_stats_config(8000, 600, records=records).threads_per_block == 256


def test_get_tensor_hash_config_skips_illegal_record():
    """A record the kernel cannot construct is skipped for the nearest
    stats-legal neighbour. Non-1024 leaves are legal when constructible."""
    records = {
        "tensor_hash_plus_stats": [
            {
                "shape": {"m": 256, "k": 4096},
                "runtime": 1.0,
                "threads_per_block": 256,
                "chunk_size": 96,
            },
            {
                "shape": {"m": 8192, "k": 512},
                "runtime": 2.0,
                "threads_per_block": 128,
                "chunk_size": 1024,
            },
        ]
    }
    selected = get_tensor_hash_plus_stats_config(256, 4096, records=records)
    assert selected.chunk_size == 1024
    assert selected.threads_per_block == 128


def test_get_tensor_hash_config_accepts_non_protocol_leaf():
    """Miners launch the JSON record's leaf; 64 is stats-legal."""
    records = {
        "tensor_hash_plus_stats": [
            {
                "shape": {"m": 256, "k": 4096},
                "runtime": 1.0,
                "threads_per_block": 256,
                "thread_load_size": 64,
                "chunk_size": 64,
            },
        ]
    }
    selected = get_tensor_hash_plus_stats_config(256, 4096, records=records)
    assert selected.chunk_size == 64
    assert selected.threads_per_block == 256


def test_get_tuned_skips_unrankable_shape_envelopes():
    """A record whose shape dict is not exactly the requested dimensions with
    positive integer values -- missing a dimension (which ranking would
    otherwise treat as an exact match), carrying a stale extra one (which
    would win a zero-distance tie), or holding a boolean / non-positive /
    non-integer value -- is skipped for the nearest rankable record."""
    records = {
        "tensor_hash_plus_stats": [
            {"shape": {"m": 256}, "runtime": 1.0, "threads_per_block": 512},
            {"shape": {"m": 256, "k": 4096, "n": 64}, "runtime": 1.0, "threads_per_block": 512},
            {"shape": {"m": True, "k": 4096}, "runtime": 1.0, "threads_per_block": 512},
            {"shape": {"m": 256, "k": 0}, "runtime": 1.0, "threads_per_block": 512},
            {"shape": {"m": 256.0, "k": 4096}, "runtime": 1.0, "threads_per_block": 512},
            {"shape": {"m": 8192, "k": 512}, "runtime": 2.0, "threads_per_block": 128},
        ]
    }
    selected = get_tensor_hash_plus_stats_config(256, 4096, records=records)
    assert selected.threads_per_block == 128


def test_get_tuned_falls_back_on_malformed_containers():
    """A file whose top level is not a mapping, or whose kernel collection is
    not a list, degrades to the fail-closed empty dict instead of raising."""
    assert get_tuned("tensor_hash_plus_stats", [{"shape": {"m": 1}}], m=256, k=4096) == {}
    assert get_tuned("tensor_hash_plus_stats", {"tensor_hash_plus_stats": None}, m=256) == {}
    assert (
        get_tuned("tensor_hash_plus_stats", {"tensor_hash_plus_stats": {"shape": {}}}, m=256) == {}
    )


def test_record_legality_shares_the_tune_space_smem_estimator():
    """Persisted-record legality uses the same shared-memory estimator as
    tune-space generation, so a constructible record no B200 CTA can launch
    is rejected up front (and skipped for a farther launchable record)."""
    oversubscribed = {
        "threads_per_block": 512,
        "num_stages": 4,
        "thread_load_size": 128,
        "chunk_size": 1024,
    }
    assert TensorHashConfig(**oversubscribed)  # constructible, yet over the smem cap
    assert not tensor_hash_plus_stats_record_is_legal(oversubscribed)
    # The sync path has no TMA staging ring, so the same knobs fit there.
    assert tensor_hash_plus_stats_record_is_legal({**oversubscribed, "sync_loads": True})
    records = {
        "tensor_hash_plus_stats": [
            {"shape": {"m": 256, "k": 4096}, "runtime": 1.0, **oversubscribed},
            {"shape": {"m": 8192, "k": 512}, "runtime": 2.0, "threads_per_block": 128},
        ]
    }
    selected = get_tensor_hash_plus_stats_config(256, 4096, records=records)
    assert selected.threads_per_block == 128


def test_autotune_merges_over_a_malformed_saved_file(monkeypatch, tmp_path):
    """Malformed-but-parseable saved data cannot abort a retune: non-list
    kernel collections and unkeyable records drop to the valid subset, and a
    non-mapping top level drops entirely, while fresh winners still publish."""
    import pearl_gemm.autotune as module

    saved = {
        "pre_quant": "not-a-list",
        "noisy_quant": [
            "not-a-record",
            {"runtime": 1.0},  # no shape mapping
            {"shape": {"m": [1]}, "runtime": 1.0},  # unhashable dimension
            {"shape": {"m": 64, "k": 128}, "runtime": 1.0, "choice": "kept"},
        ],
    }
    output = tmp_path / "tuned.json"
    output.write_text(json.dumps(saved))
    monkeypatch.setattr(
        module,
        "TUNERS",
        {"pre_quant": lambda m, k: [{"shape": {"m": m, "k": k}, "runtime": 0.5}]},
    )
    result = autotune([(4, 8, 16)], output)
    assert result["pre_quant"] == [{"shape": {"m": 4, "k": 16}, "runtime": 0.5}]
    assert result["noisy_quant"] == [
        {"shape": {"m": 64, "k": 128}, "runtime": 1.0, "choice": "kept"}
    ]
    assert json.loads(output.read_text()) == result

    output.write_text(json.dumps([1, 2, 3]))  # non-mapping top level
    result = autotune([(4, 8, 16)], output)
    assert result == {"pre_quant": [{"shape": {"m": 4, "k": 16}, "runtime": 0.5}]}


def test_autotune_publishes_atomically(monkeypatch, tmp_path):
    """A failed publish leaves the previous generation intact: the merged
    JSON lands in a staging file and replaces the target in one step."""
    import pearl_gemm.autotune as module

    saved = {"pre_quant": [{"shape": {"m": 1, "k": 2}, "runtime": 1.0}]}
    output = tmp_path / "tuned.json"
    output.write_text(json.dumps(saved))
    monkeypatch.setattr(
        module,
        "TUNERS",
        {"pre_quant": lambda m, k: [{"shape": {"m": m, "k": k}, "runtime": 0.5}]},
    )

    def failing_replace(src, dst):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="publish failure"):
        autotune([(4, 8, 16)], output)
    assert json.loads(output.read_text()) == saved


def test_records_with_non_bool_flags_are_rejected():
    """A truthy string or int in ``sync_loads``/``mad_rot`` must not reach the
    Constexpr[bool] compile parameters; the record falls back to a legal one."""
    for bad in ("true", 1, 0):
        assert not tensor_hash_plus_stats_record_is_legal({"sync_loads": bad})
        assert not tensor_hash_plus_stats_record_is_legal({"mad_rot": bad})
    records = {
        "tensor_hash_plus_stats": [
            {"shape": {"m": 256, "k": 4096}, "runtime": 1.0, "sync_loads": "true"},
            {"shape": {"m": 8192, "k": 512}, "runtime": 2.0, "threads_per_block": 256},
        ]
    }
    selected = get_tensor_hash_plus_stats_config(256, 4096, records=records)
    assert selected.threads_per_block == 256
    assert selected.sync_loads is False


def test_get_tensor_hash_config_lru_caches_device_lookup(monkeypatch):
    from pearl_gemm.tensor_hash_plus_stats import _host as host

    host._device_config.cache_clear()
    loads = []

    def fake_load(name=None, device=None):
        loads.append((name, device))
        return {
            "tensor_hash_plus_stats": [
                {"shape": {"m": 64, "k": 512}, "runtime": 1.0, "threads_per_block": 256}
            ]
        }

    monkeypatch.setattr("pearl_gemm.autotune.load_config", fake_load)
    first = get_tensor_hash_plus_stats_config(64, 512)
    second = get_tensor_hash_plus_stats_config(64, 512)
    assert first is second
    assert first.threads_per_block == 256
    assert len(loads) == 1


def test_main_rejects_separator_only_kernels(monkeypatch):
    import sys

    import pearl_gemm.autotune as module

    monkeypatch.setattr(sys, "argv", ["autotune", "--kernels", ", ,"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2


def test_kernels_help_lists_tuner_names(capsys, monkeypatch):
    import sys

    import pearl_gemm.autotune as module

    monkeypatch.setattr(sys, "argv", ["autotune", "--help"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for name in TUNERS:
        assert name in help_text
    assert "untouched" in help_text
