"""Autotuning for the CuTeDSL kernels: sweep launch/tiling params, keep the best.

A "config" is host-constructor kwargs. ``_sweep`` compiles every candidate
first (fills the process ``cute.compile`` cache; JIT may drop SM clocks
to idle) and only then times, so a monolithic grid search stays
comparable. One JSON per device family ships in ``autotune_configs/``:

    {"noisy_quant": [{"shape": {"m":.., "k":..}, "runtime": .., **cfg}, ...], ...}

Timing is CUDA-graph based (``steps_per_graph`` kernel launches per replay)
over ``NUM_SLOTS`` rotating input sets so L2 does not flatter small shapes.
Inputs are random with the right dtypes/shapes. The lottery threshold never
wins, so signal publication does not affect timing.

Run ``python -m pearl_gemm.autotune --shapes 4096x4096x4096,...`` on the
target device to (re)generate its config. Winners merge into the existing
file by ``(kernel, shape)``, so tuning a subset leaves the rest alone.
``tensor_hash_plus_stats`` reads those records at launch when ``config`` is
omitted (nearest legal ``(m, k)`` if the shape was not swept).
"""

import argparse
import fcntl
import gc
import itertools
import json
import math
import os
from collections.abc import Callable, Mapping
from functools import lru_cache
from importlib import resources
from pathlib import Path
from types import MappingProxyType

import torch

from .protocol_constants import R
from .tensor_hash_plus_stats import _blake3, _host
from .tensor_hash_plus_stats._merkle_host import (
    SUPPORTED_LEAVES_PER_MT_BLOCK,
    SUPPORTED_NUM_STAGES,
    SUPPORTED_THREAD_LOAD_SIZES,
    SUPPORTED_THREADS_PER_BLOCK,
    _effective_chunks_per_thread,
    _validate_stats_compatible,
    supported_chunk_sizes,
    tensor_hash_smem_fits,
)
from .tensor_hash_plus_stats._merkle_tree_roots_kernel import SUPPORTED_CHUNKS_PER_THREAD

NUM_SLOTS = 10  # rotating input sets (defeats L2 residency)
WARMUP = 25
REPEAT = 50  # graph replays; each replay runs STEPS_PER_GRAPH calls
STEPS_PER_GRAPH = 10

# --------------------------------------------------------------------------- #
# Tune spaces
# --------------------------------------------------------------------------- #

PRE_QUANT_SPACE: list[dict] = [{"threads_per_block": threads} for threads in (128, 256, 512)]


def _stats_chunk_sizes() -> tuple[int, ...]:
    """The Merkle leaves the stats fusion can launch (sub-block leaves and
    whole multiples of a stats block). Derived from the kernel gate, so a
    lifted stats restriction widens the sweep without touching the space."""
    legal = []
    for chunk in supported_chunk_sizes():
        try:
            _validate_stats_compatible(chunk)
        except ValueError:
            continue
        legal.append(chunk)
    return tuple(legal)


def tensor_hash_launch_space() -> list[dict]:
    """Launch knobs only (implicit 1024-byte leaf): the smem-fitting grid.

    The plus-stats tuner expands this grid. Generated on demand, so a
    serving process that imports this module for record lookup builds no
    candidate tables.
    """
    return [
        {
            "threads_per_block": threads,
            "num_stages": stages,
            "leaves_per_mt_block": 256,
            "thread_load_size": load,
        }
        for threads, stages, load in itertools.product(
            SUPPORTED_THREADS_PER_BLOCK,
            SUPPORTED_NUM_STAGES,
            SUPPORTED_THREAD_LOAD_SIZES,
        )
        if tensor_hash_smem_fits(threads, stages, load)
    ]


def tensor_hash_plus_stats_space() -> list[dict]:
    """The production stats-path tune space; the no-stats/raw hash is not
    tuned here.

    The full grid over the record-legal knobs (``test_autotune`` pins the
    structure):

    - chunk_size stays at the 1024-byte leaf in this production space.
      Other leaves change every digest and are a protocol/verifier
      decision, not a per-device autotune record. Miners launch whatever
      the record carries (``tensor_hash_plus_stats_record_is_legal`` gates
      constructibility and smem, not a fixed leaf).
    - sync_loads=True legs pin num_stages / thread_load_size (no TMA ring, so
      both are dead there) and mad_rot (``_mad_rot_dispatch`` keeps the sync
      path on SHF), so the sync sweep only walks the live knobs.
    - chunks_per_thread legs that a shape's blobs cannot support are dropped
      per shape in ``tune_tensor_hash_plus_stats`` (the launch would silently
      fall back and re-time the k=1 kernel).
    """
    return [
        {
            **cfg,
            "leaves_per_mt_block": leaves,
            "chunk_size": _blake3.CHUNK_SIZE,
            "chunks_per_thread": chunks_per_thread,
            "mad_rot": mad_rot,
            "sync_loads": False,
        }
        for cfg, leaves, chunks_per_thread, mad_rot in itertools.product(
            tensor_hash_launch_space(),
            SUPPORTED_LEAVES_PER_MT_BLOCK,
            SUPPORTED_CHUNKS_PER_THREAD,
            (False, True),
        )
        if tensor_hash_smem_fits(
            cfg["threads_per_block"],
            cfg["num_stages"],
            cfg["thread_load_size"],
            stats_chunk=_blake3.CHUNK_SIZE,
        )
    ] + [
        {
            "threads_per_block": threads,
            "num_stages": 2,
            "leaves_per_mt_block": leaves,
            "thread_load_size": 128,
            "chunk_size": _blake3.CHUNK_SIZE,
            "chunks_per_thread": chunks_per_thread,
            "mad_rot": False,
            "sync_loads": True,
        }
        for threads, leaves, chunks_per_thread in itertools.product(
            SUPPORTED_THREADS_PER_BLOCK,
            SUPPORTED_LEAVES_PER_MT_BLOCK,
            SUPPORTED_CHUNKS_PER_THREAD,
        )
    ]


# The config families that win on the standard shapes (see the standard-shape
# records in autotune_configs/): rows=64 with the tcgen05 kernel on large shapes,
# MERGED bk=256 on mid shapes, smaller row blocks (denser grids) on small-m.
# Non-standard shapes sweep over these plus the incumbents.
NOISY_QUANT_SPACE: list[dict] = [
    {
        "noise_rows": 64,
        "noise_bk": 256,
        "noise_stages": 2,
        "noise_out_stages": 3,
        "noise_load_mode": "merged",
    },
    {
        "noise_rows": 64,
        "noise_bk": 256,
        "noise_stages": 3,
        "noise_out_stages": 2,
        "noise_load_mode": "merged",
    },
    {
        "noise_rows": 64,
        "noise_bk": 128,
        "noise_stages": 2,
        "noise_out_stages": 3,
        "noise_load_mode": "merged",
    },
    {
        "noise_rows": 32,
        "noise_bk": 128,
        "noise_stages": 3,
        "noise_out_stages": 2,
        "noise_load_mode": "ring",
    },
    {
        "noise_rows": 32,
        "noise_bk": 256,
        "noise_stages": 3,
        "noise_out_stages": 3,
        "noise_load_mode": "merged",
    },
    {
        "noise_rows": 32,
        "noise_bk": 256,
        "noise_stages": 4,
        "noise_out_stages": 2,
        "noise_load_mode": "merged",
    },
    {
        "noise_rows": 16,
        "noise_bk": 128,
        "noise_stages": 2,
        "noise_out_stages": 2,
        "noise_load_mode": "merged",
    },
    {
        "noise_rows": 16,
        "noise_bk": 256,
        "noise_stages": 3,
        "noise_out_stages": 2,
        "noise_load_mode": "ring",
    },
    {
        "noise_rows": 16,
        "noise_bk": 256,
        "noise_stages": 4,
        "noise_out_stages": 2,
        "noise_load_mode": "merged",
    },
]

# Historical 128-wide shortlist kept as the @slow bit-identical gate. The
# tuner walks ``mixed_gemm_space`` at the shape's committed lottery instead.
MIXED_GEMM_SPACE: list[dict] = [
    {"tile_m": 256, "tile_n": 128, "cluster_m": 2, "cluster_n": 1},
    {"tile_m": 256, "tile_n": 256, "cluster_m": 2, "cluster_n": 1},
    {"tile_m": 256, "tile_n": 128, "cluster_m": 2, "cluster_n": 2},
    {"tile_m": 128, "tile_n": 256, "cluster_m": 2, "cluster_n": 1},
    {"tile_m": 128, "tile_n": 128, "cluster_m": 2, "cluster_n": 1},
    {"tile_m": 128, "tile_n": 128, "cluster_m": 1, "cluster_n": 1},
]

# Must stay in sync with ``vllm_miner.mining_config.select_tile``:
# 4x64 wherever it fits, 16x32 next, 4x128 fallback. ltile_cols is a job-key
# commitment, not a free knob; the sweep injects it so n=576 / n=2752 are
# launchable.
_LOTTERY_4X64_MAX_K = 30720
_LOTTERY_16X32_MAX_K = 43520
_LOTTERY_MIN_K = 1024

_MIXED_GEMM_TILE_M = (128, 256)
_MIXED_GEMM_CLUSTERS = ((1, 1), (2, 1), (2, 2))
_MIXED_GEMM_TILE_N_ALIGN = 32
_MIXED_GEMM_TILE_N_MAX = 256


def committed_lottery(n: int, k: int) -> tuple[int, int]:
    """Return the committed ``(ltile_rows, ltile_cols)`` for ``(n, k)``."""
    if n % 64 == 0 and _LOTTERY_MIN_K <= k <= _LOTTERY_4X64_MAX_K:
        return 4, 64
    if n % 32 == 0 and _LOTTERY_MIN_K <= k <= _LOTTERY_16X32_MAX_K:
        return 16, 32
    return 4, 128


def mixed_gemm_space(ltile_rows: int = 4, ltile_cols: int = 128) -> list[dict]:
    """Tile/cluster lattice legal for a committed lottery width.

    ``tile_n`` steps by ``ltile_cols`` across the host-legal range (multiples
    of 32 in ``[32, 256]``), so a 16x32 job can pick an exact-width 96-column
    CTA rather than a padded 64/128 neighbour. Non-default lottery fields are
    stored on the record so ``get_tuned``'s legal filter can keep 64-wide and
    128-wide jobs from inheriting each other.
    """
    space = []
    tile_ns = range(ltile_cols, _MIXED_GEMM_TILE_N_MAX + 1, ltile_cols)
    for tile_m, tile_n, (cluster_m, cluster_n) in itertools.product(
        _MIXED_GEMM_TILE_M, tile_ns, _MIXED_GEMM_CLUSTERS
    ):
        if tile_n % _MIXED_GEMM_TILE_N_ALIGN:
            continue
        cfg = {
            "tile_m": tile_m,
            "tile_n": tile_n,
            "cluster_m": cluster_m,
            "cluster_n": cluster_n,
        }
        if ltile_cols != 128:
            cfg["ltile_cols"] = ltile_cols
        if ltile_rows != 4:
            cfg["ltile_rows"] = ltile_rows
        space.append(cfg)
    return space


# --------------------------------------------------------------------------- #
# CUDA-graph timing
# --------------------------------------------------------------------------- #


def _prime_compile(call_slot) -> None:
    """One untimed launch so ``cute.compile`` fills the process cache.

    JIT can drop SM clocks to idle for seconds. Paying that for every
    candidate before any timed replay keeps a monolithic sweep fair.
    """
    call_slot(0)
    torch.cuda.synchronize()


def time_graph(call_slot) -> float:
    """Median seconds per call of ``call_slot(slot_idx)``, CUDA-graph timed."""
    counter = [0]

    def step():
        call_slot(counter[0] % NUM_SLOTS)
        counter[0] += 1

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(STEPS_PER_GRAPH):
            step()

    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(REPEAT):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3 / STEPS_PER_GRAPH)
    times.sort()
    return times[len(times) // 2]


def _rand_bf16(m: int, k: int, device="cuda") -> torch.Tensor:
    return torch.randn(m, k, dtype=torch.bfloat16, device=device)


# --------------------------------------------------------------------------- #
# Per-op sweeps: build one instance per config, rotate inputs, keep the best
# --------------------------------------------------------------------------- #


def _sweep(space, shape, build_launch, *, catch=(AssertionError,)) -> list[dict]:
    """Time every config in ``space``.

    Two passes: compile every launchable candidate, then time from the
    cache. ``build_launch(cfg)`` allocates the op's outputs and returns the
    slot-indexed launch closure, or raises one of ``catch`` for a tunable
    that is invalid for this shape/device.
    """
    viable = []
    for cfg in space:
        print(f"[autotune] compile {shape} {cfg}", flush=True)
        try:
            _prime_compile(build_launch(cfg))
        except catch:
            print(f"[autotune] skip {shape} {cfg}", flush=True)
            continue
        viable.append(cfg)
    gc.collect()
    records = []
    for cfg in viable:
        print(f"[autotune] trying {shape} {cfg}", flush=True)
        try:
            rt = time_graph(build_launch(cfg))
        except catch:
            print(f"[autotune] skip {shape} {cfg}", flush=True)
            continue
        records.append({"shape": shape, "runtime": rt, **cfg})
        print(f"[autotune] {rt * 1e6:.1f} us  {shape} {cfg}", flush=True)
    return records


def tune_pre_quant(m: int, k: int) -> list[dict]:
    from .pre_quant import PreQuantConfig, pre_quant, pre_quant_output_shapes

    slots = [_rand_bf16(m, k) for _ in range(NUM_SLOTS)]
    codes_shape, scales_shape = pre_quant_output_shapes(m, k)

    def build_launch(cfg):
        config = PreQuantConfig(**cfg)
        codes = torch.zeros(codes_shape, dtype=torch.int8, device="cuda")
        scales = torch.zeros(scales_shape, dtype=torch.bfloat16, device="cuda")
        return lambda i: pre_quant(slots[i], codes, scales, config=config)

    return _sweep(PRE_QUANT_SPACE, {"m": m, "k": k}, build_launch)


def tune_tensor_hash_plus_stats(m: int, k: int) -> list[dict]:
    from .pre_quant import pre_quant_output_shapes
    from .tensor_hash_plus_stats import (
        TensorHashConfig,
        tensor_hash_plus_stats,
        tensor_hash_workspace_bytes,
    )
    from .tensor_hash_plus_stats._finalize_kernel import A_KEYS_BYTES, P_A_BYTES

    # Hash timing is data-independent: fixed keyA / seedB / pA stand in for
    # the header, B side and committed layout; random blobs for the pre_quant
    # outputs.
    key_a, seed_b, p_a = b"\x01" * 32, b"\x02" * 32, bytes(P_A_BYTES)
    codes_shape, scales_shape = pre_quant_output_shapes(m, k)
    slots = [
        (
            torch.randint(-127, 128, codes_shape, dtype=torch.int8, device="cuda"),
            torch.rand(scales_shape, dtype=torch.bfloat16, device="cuda"),
        )
        for _ in range(NUM_SLOTS)
    ]
    key = torch.frombuffer(bytearray(key_a), dtype=torch.uint8).cuda()
    seed_b_dev = torch.frombuffer(bytearray(seed_b), dtype=torch.uint8).cuda()

    def build_launch(cfg):
        config = TensorHashConfig(**cfg)
        root_codes = torch.zeros(32, dtype=torch.uint8, device="cuda")
        root_scales = torch.zeros(32, dtype=torch.uint8, device="cuda")
        roots = torch.zeros(
            tensor_hash_workspace_bytes(m, k, config),
            dtype=torch.uint8,
            device="cuda",
        )
        a_keys = torch.zeros(A_KEYS_BYTES, dtype=torch.uint8, device="cuda")
        stats = torch.zeros(2 * (m * k // 512), dtype=torch.float32, device="cuda")
        return lambda i: tensor_hash_plus_stats(
            *slots[i],
            key,
            seed_b_dev,
            root_codes,
            root_scales,
            roots,
            a_keys,
            stats,
            p_a=p_a,
            config=config,
        )

    # Drop the legs that would re-time another leg's kernel at this shape:
    # below the per-blob size gate mad_rot dispatches to the same SHF kernel
    # (``_mad_rot_dispatch``; the codes blob, m*k bytes, is the larger one),
    # and a chunks_per_thread the blobs cannot support silently falls back
    # to the k=1 launch (``_effective_chunks_per_thread``).
    blobs_bytes = (m * k, m * k // 4)
    space = [
        cfg
        for cfg in tensor_hash_plus_stats_space()
        if not (cfg["mad_rot"] and m * k < _blake3.MAD_ROT_MIN_BYTES)
        and cfg["chunks_per_thread"]
        == _effective_chunks_per_thread(
            cfg["chunks_per_thread"], blobs_bytes, cfg["chunk_size"], with_stats=True
        )
    ]
    return _sweep(
        space,
        {"m": m, "k": k},
        build_launch,
        catch=(AssertionError, ValueError),
    )


def tune_noisy_quant(m: int, k: int) -> list[dict]:
    from .api import (
        NoisyQuantConfig,
        noisy_quant,
        pack_noise_factor,
        pre_quant,
        pre_quant_output_shapes,
    )

    # Noising timing is data-independent, but the blob slots come from
    # pre_quant so the staged scales stay finite.
    codes_shape, scales_shape = pre_quant_output_shapes(m, k)
    slots = []
    for _ in range(NUM_SLOTS):
        codes = torch.zeros(codes_shape, dtype=torch.int8, device="cuda")
        scales = torch.zeros(scales_shape, dtype=torch.bfloat16, device="cuda")
        pre_quant(_rand_bf16(m, k), codes, scales)
        slots.append((codes, scales))
    cA = torch.randint(0, 256, (32,), dtype=torch.uint8, device="cuda")
    # plausible commit partials (values do not affect kernel timing)
    stats = torch.rand(2 * (m * k // 512), device="cuda") * 512 + 1
    # Noise-line-like values (+-[0.5, ~200]): pack_noise_factor asserts the
    # Noiser value grid, which plain randn codes do not satisfy.
    f1 = (
        (torch.rand(R, k, device="cuda") * 200 + 0.5)
        * (torch.randint(0, 2, (R, k), device="cuda") * 2 - 1)
    ).to(torch.float8_e4m3fn)
    f2 = torch.randn(R, k, device="cuda").to(torch.float8_e4m3fn)
    f1_hl = pack_noise_factor(f1)

    def build_launch(cfg):
        config = NoisyQuantConfig(**cfg)
        alpha = torch.zeros(m, dtype=torch.bfloat16, device="cuda")
        beta = torch.zeros_like(alpha)
        e1 = torch.zeros(m, R, dtype=torch.float8_e4m3fn, device="cuda")
        a_prime = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device="cuda")
        a_peel = torch.zeros(m, 2 * R, dtype=torch.bfloat16, device="cuda")
        return lambda i: noisy_quant(
            *slots[i],
            cA,
            stats,
            f1_hl,
            f2,
            alpha,
            beta,
            e1,
            a_prime,
            a_peel,
            config=config,
        )

    return _sweep(
        NOISY_QUANT_SPACE,
        {"m": m, "k": k},
        build_launch,
        catch=(AssertionError, ValueError),
    )


def tune_mixed_gemm(m: int, n: int, k: int) -> list[dict]:
    from .api import HitSignal, HitSignalConfig, MixedGemmConfig, mixed_gemm

    a_slots = [
        (
            torch.randn(m, k, device="cuda").clamp(-2, 2).to(torch.float8_e4m3fn),
            torch.randn(m, 2 * R, dtype=torch.bfloat16, device="cuda"),
            torch.rand(m, dtype=torch.bfloat16, device="cuda") + 0.5,
        )
        for _ in range(NUM_SLOTS)
    ]
    b = (
        torch.randn(n, k, device="cuda").clamp(-2, 2).to(torch.float8_e4m3fn),
        torch.randn(n, 2 * R, dtype=torch.bfloat16, device="cuda"),
        torch.rand(n, dtype=torch.bfloat16, device="cuda") + 0.5,
    )
    pow_key = torch.randint(0, 256, (32,), dtype=torch.uint8, device="cuda")
    threshold = torch.zeros(32, dtype=torch.uint8, device="cuda")  # never wins
    inv_alpha_b = torch.reciprocal(b[2].float()).contiguous()
    # Full-capacity signal so the timed kernel is the production snapshot
    # variant (the copy code never executes with the never-win threshold).
    hit_signal = HitSignal(HitSignalConfig(max_m=m, max_k=k))
    a_codes = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
    a_scales = torch.rand(m, k // 8, dtype=torch.bfloat16, device="cuda")
    commitment_hash_b = torch.randint(0, 256, (32,), dtype=torch.uint8, device="cuda")

    def build_launch(cfg):
        config = MixedGemmConfig(**cfg)
        out = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
        return lambda i: mixed_gemm(
            a_slots[i][0],
            b[0],
            a_slots[i][1],
            b[1],
            a_slots[i][2],
            inv_alpha_b,
            pow_key,
            threshold,
            out,
            hit_signal,
            a_codes,
            a_scales,
            commitment_hash_b,
            config=config,
        )

    return _sweep(
        mixed_gemm_sweep_space(m, n, k),
        {"m": m, "n": n, "k": k},
        build_launch,
        catch=(AssertionError, ValueError),
    )


TUNERS = {
    "pre_quant": tune_pre_quant,
    "tensor_hash_plus_stats": tune_tensor_hash_plus_stats,
    "noisy_quant": tune_noisy_quant,
    "mixed_gemm": tune_mixed_gemm,
}


# --------------------------------------------------------------------------- #
# Persistence + runtime lookup
# --------------------------------------------------------------------------- #


def device_config_name(device: "torch.device | int | None" = None) -> str:
    """Config-file stem for ``device`` (current CUDA device when omitted)."""
    return torch.cuda.get_device_name(device).lower().replace(" ", "_").replace("-", "_")


def config_path(name: str | None = None, device: "torch.device | int | None" = None) -> Path:
    name = name or device_config_name(device)
    return Path(resources.files("pearl_gemm") / "autotune_configs" / f"{name}.json")


def _shape_key(shape: dict) -> tuple:
    return tuple(sorted(shape.items()))


# Constructor fields the kernel no longer accepts. Dropped at read/merge so a
# historical JSON still resolves; the rest of the record is kept as-is.
_RETIRED_RECORD_FIELDS = {
    "noisy_quant": frozenset({"noise_ksplit"}),
}


def _without_retired_fields(kernel: str, record: dict) -> dict:
    retired = _RETIRED_RECORD_FIELDS.get(kernel)
    if not retired or not any(key in record for key in retired):
        return record
    return {key: value for key, value in record.items() if key not in retired}


def _record_kwargs(kernel: str, record: dict) -> dict:
    retired = _RETIRED_RECORD_FIELDS.get(kernel, ())
    return {
        key: value
        for key, value in record.items()
        if key not in ("shape", "runtime") and key not in retired
    }


def merge_records(saved: dict, swept: dict) -> dict:
    """Overlay freshly swept winners onto ``saved``, keyed by (kernel, shape).

    Saved order is preserved and new shapes append, so a retune shows up as a
    minimal diff rather than a reshuffled file. A malformed saved file (a
    non-mapping top level, non-list kernel collections, records without a
    keyable shape mapping) degrades to its valid subset, so corrupt data
    cannot abort a publish -- ``get_tuned`` skips the same records at read
    time.
    """

    def mergeable(records) -> list[dict]:
        if not isinstance(records, list):
            return []
        kept = []
        for record in records:
            if not (isinstance(record, dict) and isinstance(record.get("shape"), dict)):
                continue
            try:
                hash(_shape_key(record["shape"]))
            except TypeError:  # unhashable dimension values
                continue
            kept.append(record)
        return kept

    saved = saved if isinstance(saved, dict) else {}
    merged = {
        kernel: [_without_retired_fields(kernel, record) for record in mergeable(records)]
        for kernel, records in saved.items()
    }
    for kernel, records in swept.items():
        by_shape = {_shape_key(record["shape"]): record for record in merged.get(kernel, [])}
        by_shape.update({_shape_key(record["shape"]): record for record in records})
        merged[kernel] = [_without_retired_fields(kernel, record) for record in by_shape.values()]
    return merged


def autotune(
    shapes: list[tuple[int, int, int]],
    out_path: Path | None = None,
    kernels: list[str] | None = None,
) -> dict:
    """Sweep selected ops over ``shapes`` ((m, n, k) triples; k-only ops
    dedupe on (m, k)) and merge the per-shape winners into the device's JSON,
    which is written and returned.

    ``kernels`` defaults to every tuner. Two kinds of record survive a run
    untouched: shapes it did not sweep, so retuning a subset costs only the
    shapes in it, and shapes it swept without finding a single launchable
    config. An empty sweep means the space missed the shape, not that the
    saved record went bad.

    ``mixed_gemm`` injects the committed lottery for ``(n, k)`` and walks
    ``mixed_gemm_sweep_space`` at that width (the committed lattice plus the
    routed 64-row decode tile when it applies), so GLM n=576 / n=2752 are
    in-space.

    A shape that did produce a winner is replaced by it even if the saved
    runtime was lower. A winner can also be faster at the shape it swept while
    regressing a nearby shape that inherits it through ``get_tuned``, so bench
    the inherited shapes too before adopting a retune.
    """
    if kernels is None:
        selected = TUNERS
    else:
        unknown = [name for name in kernels if name not in TUNERS]
        if unknown:
            known = ", ".join(TUNERS)
            raise ValueError(f"unknown kernel(s) {unknown}; known: {known}")
        selected = {name: TUNERS[name] for name in kernels}
    best: dict[str, list[dict]] = {name: [] for name in selected}
    mk_done: set[tuple] = set()
    for m, n, k in shapes:
        for name, tuner in selected.items():
            if name == "mixed_gemm":
                shape = {"m": m, "n": n, "k": k}
                records = tuner(m, n, k)
            else:
                if (name, m, k) in mk_done:
                    continue
                mk_done.add((name, m, k))
                shape = {"m": m, "k": k}
                records = tuner(m, k)
            if records:
                win = min(records, key=lambda r: r["runtime"])
                best[name].append(win)
                print(f"[autotune] {name} {win['shape']}: {win['runtime'] * 1e6:.1f} us  {win}")
            else:
                print(f"[autotune] {name} {shape}: no launchable config; keeping saved record")
    out_path = out_path or config_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize read-merge-write across concurrent tuners (each keeps the
    # other's records) and publish atomically so a reader never sees a torn
    # file.
    with open(out_path.with_name(out_path.name + ".lock"), "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        saved = json.loads(out_path.read_text()) if out_path.exists() else {}
        merged = merge_records(saved, best)
        updated = sum(len(records) for records in best.values())
        total = sum(len(records) for records in merged.values())
        staging = out_path.with_name(out_path.name + ".tmp")
        staging.write_text(json.dumps(merged, indent=2))
        os.replace(staging, out_path)
    # In-process readers cache resolved configs; a retune must not leave them
    # serving the pre-sweep records.
    _host._device_config.cache_clear()
    print(f"[autotune] wrote {out_path}: {updated} record(s) from this sweep, {total} in the file")
    return merged


def load_config(name: str | None = None, device: "torch.device | int | None" = None) -> dict | None:
    path = config_path(name, device)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def get_tuned(
    kernel: str,
    config: dict | None = None,
    *,
    legal: Callable[[dict], bool] | None = None,
    require_legal: bool = False,
    exact: bool = False,
    **shape,
) -> dict:
    """Return the nearest shape record whose constructor kwargs are legal.

    ``legal`` is evaluated before distance ranking, so an illegal nearest
    cluster can never displace a farther configuration that the requested
    runtime shape can actually launch. A record whose envelope ranking
    cannot consume -- no ``shape`` dict, or one whose dimensions are not
    exactly the requested ones with positive integer values -- is skipped
    the same way, so a corrupt file degrades to the fallback instead of
    crashing ranking, silently treating a missing dimension as an exact
    match, or letting a stale extra-dimension record win a zero-distance
    tie. Non-mapping files and non-list kernel collections degrade the same
    way. ``exact=True`` keeps only a record whose shape values equal the
    request (no nearest-neighbor fallback). An empty dict is the fail-closed
    signal when no saved record is usable; public host validation remains
    the final boundary after callers construct their config object.
    """
    config = config if config is not None else load_config()
    records = config.get(kernel) if isinstance(config, dict) else None
    records = records if isinstance(records, list) else []

    def kwargs(record: dict) -> dict:
        return _record_kwargs(kernel, record)

    def rankable(record) -> bool:
        if not isinstance(record, dict) or not isinstance(record.get("shape"), dict):
            return False
        envelope = record["shape"]
        # Exact dimension-key match: bool is excluded by the exact int type.
        return set(envelope) == set(shape) and all(
            type(value) is int and value > 0 for value in envelope.values()
        )

    candidates = [
        record
        for record in records
        if rankable(record) and (legal is None or legal(kwargs(record)))
    ]
    if exact:
        candidates = [
            record
            for record in candidates
            if all(record["shape"][dimension] == value for dimension, value in shape.items())
        ]
    if not candidates:
        if records and require_legal:
            raise ValueError(f"no legal {kernel} autotune record for requested shape {shape}")
        return {}

    def dist(record: dict) -> float:
        return sum(
            abs(math.log2(record["shape"][dimension]) - math.log2(max(value, 1)))
            for dimension, value in shape.items()
        )

    return kwargs(min(candidates, key=dist))


# For m <= SMALL_M_LIMIT with a 4-row lottery, route to the 64-row kernel
# tile: tile_n = max(64, ltile_cols), cluster 1x1. Callers that have an
# exact autotune record should prefer that record (see pipeline._configs).
SMALL_M_LIMIT = 256
_SMALL_TILE_M = 64


@lru_cache(maxsize=256)
def route_small_m_mixed_gemm(
    m: int, n: int, k: int, *, ltile_rows: int, ltile_cols: int
) -> Mapping | None:
    """Heuristic ``MixedGemmConfig`` kwargs for a small-m (decode) launch.

    For ``m <= SMALL_M_LIMIT`` and a 4-row lottery, route to the 64-row
    kernel tile: tile (64, 64) when the committed lottery width is 64, and
    (64, ltile_cols) otherwise (the narrowest tile_n the committed width
    divides). Returns ``None`` -- keep the saved records -- when the
    committed geometry cannot run a 64-row tile (the 16-row lottery family
    splits every accumulator row between two TMEM-load threads) or when the
    routed config cannot launch the shape. Cached: config resolution sits
    on the launch hot path, so the returned mapping is shared across callers
    and read-only (an immutable proxy: mutating it would poison every later
    resolution of the shape).
    """
    if m > SMALL_M_LIMIT or ltile_rows != 4:
        return None
    from .mixed_gemm import MixedGemmConfig, validate_mixed_gemm_config

    kwargs = {
        "tile_m": _SMALL_TILE_M,
        "tile_n": max(_SMALL_TILE_M, ltile_cols),
        "cluster_m": 1,
        "cluster_n": 1,
        "ltile_rows": ltile_rows,
        "ltile_cols": ltile_cols,
    }
    try:
        validate_mixed_gemm_config(m, n, k, MixedGemmConfig(**kwargs))
    except (TypeError, ValueError):
        return None
    return MappingProxyType(kwargs)


def mixed_gemm_sweep_space(m: int, n: int, k: int) -> list[dict]:
    """Committed lottery lattice plus the routed 64-row decode tile when it applies.

    Exact autotune records outrank the small-m heuristic at lookup, so a
    small-m sweep that never timed the routed tile would replace it with a
    128/256-row winner the heuristic was meant to beat. Include it so a
    replacement has to win the comparison.
    """
    ltile_rows, ltile_cols = committed_lottery(n, k)
    space = mixed_gemm_space(ltile_rows, ltile_cols)
    routed = route_small_m_mixed_gemm(m, n, k, ltile_rows=ltile_rows, ltile_cols=ltile_cols)
    if routed is None:
        return space
    routed_cfg = dict(routed)
    if routed_cfg in space:
        return space
    return [routed_cfg, *space]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        default="4096x4096x4096,8192x8192x8192",
        help="comma-separated MxNxK triples",
    )
    parser.add_argument("--out", default=None, help="output JSON path (default: per-device)")
    parser.add_argument(
        "--kernels",
        default=None,
        help=(
            "comma-separated kernel names from "
            f"{', '.join(TUNERS)} (default: all). Unknown names raise "
            "ValueError. A subset retunes only those kernels; other tuner "
            "records are left untouched."
        ),
    )
    args = parser.parse_args()
    shapes = [tuple(int(d) for d in s.split("x")) for s in args.shapes.split(",")]
    kernels = None
    if args.kernels is not None:
        kernels = [name.strip() for name in args.kernels.split(",") if name.strip()]
        if not kernels:
            parser.error(f"--kernels must name at least one kernel; known: {', '.join(TUNERS)}")
    autotune(shapes, Path(args.out) if args.out else None, kernels=kernels)


if __name__ == "__main__":
    main()
