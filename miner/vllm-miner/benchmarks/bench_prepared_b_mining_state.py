"""Benchmark dense noisy GEMM with and without PreparedBMiningState hits.

Example:
    uv run --package vllm-miner python \
        miner/vllm-miner/benchmarks/bench_prepared_b_mining_state.py
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from miner_base.settings import MinerSettings
from pearl_gateway.comm.dataclasses import MiningJob
from vllm_miner.config import config
from vllm_miner.gemm_operators import pearl_gemm_noisy, pearl_gemm_vanilla
from vllm_miner.mining_state import (
    delete_state,
    get_async_manager,
    init_async_manager,
    init_pinned_pool,
)
from vllm_miner.prepared_b_mining_state import (
    clear_prepared_b_cache,
    prepared_b_cache_bytes,
    prepared_b_cache_size,
    prepared_b_cache_stats,
    reset_prepared_b_cache_stats,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark dense GEMM with vanilla and noisy PreparedBMiningState modes."
    )
    parser.add_argument(
        "--m-values",
        type=str,
        default="256,512,1024,2048",
        help="Comma-separated M values. Use this to model batch/speculative-width pressure.",
    )
    parser.add_argument("--n", type=int, default=28672)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--cache-bytes", type=int, default=1 << 30)
    parser.add_argument(
        "--modes",
        type=str,
        default="vanilla,disabled,cold,warm,job_transition",
        help="Comma-separated modes: vanilla,disabled,cold,warm,job_transition.",
    )
    parser.add_argument("--allow-non-h100", action="store_true")
    args = parser.parse_args()
    args.m_values = _parse_int_list(args.m_values, "--m-values")
    args.modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    valid_modes = {"vanilla", "disabled", "cold", "warm", "job_transition"}
    unknown_modes = set(args.modes) - valid_modes
    if unknown_modes:
        raise ValueError(f"unknown --modes entries: {sorted(unknown_modes)}")
    return args


def _parse_int_list(raw: str, flag_name: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{flag_name} must contain at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError(f"{flag_name} must contain only positive integers")
    return values


def _require_gpu(allow_non_h100: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    name = torch.cuda.get_device_name(0)
    if "H100" not in name and not allow_non_h100:
        raise RuntimeError(f"Expected H100; got {name}. Pass --allow-non-h100 to override.")


def _make_inputs(m: int, args: argparse.Namespace) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    a = torch.randint(-63, 64, (m, args.k), dtype=torch.int8, device="cuda")
    b = torch.randint(-63, 64, (args.n, args.k), dtype=torch.int8, device="cuda")
    scale_a = torch.ones((m,), dtype=torch.float32, device="cuda")
    scale_b = torch.ones((args.n,), dtype=torch.float32, device="cuda")
    return a, b, scale_a, scale_b


def _set_job(sequence: int) -> None:
    header = sequence.to_bytes(8, byteorder="little", signed=False) * 4
    get_async_manager()._mining_job = MiningJob(header, 1)


def _one_noisy_call(
    a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor
) -> None:
    out = pearl_gemm_noisy(a, b, scale_a, scale_b, torch.bfloat16, submit_block=False)
    # Keep the output live until after kernel launch.
    del out


def _one_vanilla_call(
    a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor
) -> None:
    out = pearl_gemm_vanilla(a, b, scale_a, scale_b, torch.bfloat16)
    # Keep the output live until after kernel launch.
    del out


def _one_call(
    mode: str, a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor
) -> None:
    if mode == "vanilla":
        _one_vanilla_call(a, b, scale_a, scale_b)
        return
    _one_noisy_call(a, b, scale_a, scale_b)


def _measure(  # noqa: C901
    mode: str,
    args: argparse.Namespace,
    m: int,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> dict[str, float | int | str]:
    clear_prepared_b_cache()
    if mode in {"vanilla", "disabled"}:
        config.settings.prepared_b_cache_bytes = 0
    else:
        config.settings.prepared_b_cache_bytes = args.cache_bytes

    if mode == "warm":
        _set_job(0)
        _one_call(mode, a, b, scale_a, scale_b)
        torch.cuda.synchronize()
        reset_prepared_b_cache_stats()
    elif mode not in {"vanilla", "disabled", "cold", "job_transition"}:
        raise ValueError(f"unknown mode: {mode}")

    gpu_samples = []
    wall_samples = []
    sequence = 1

    for idx in range(args.warmup):
        if mode == "cold":
            clear_prepared_b_cache(reset_stats=False)
        if mode == "vanilla":
            pass
        elif mode == "warm":
            _set_job(0)
        elif mode == "job_transition":
            _set_job(sequence)
            sequence += 1
        else:
            _set_job(idx)
        _one_call(mode, a, b, scale_a, scale_b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(args.iterations):
        if mode == "cold":
            clear_prepared_b_cache(reset_stats=False)
        if mode == "warm":
            _set_job(0)
        elif mode == "job_transition":
            _set_job(sequence)
            sequence += 1
        start.record()
        wall_start = time.perf_counter()
        _one_call(mode, a, b, scale_a, scale_b)
        end.record()
        torch.cuda.synchronize()
        wall_samples.append((time.perf_counter() - wall_start) * 1000)
        gpu_samples.append(start.elapsed_time(end))

    stats = prepared_b_cache_stats()
    cache_entries = prepared_b_cache_size()
    cache_bytes = prepared_b_cache_bytes()
    state_bytes = cache_bytes // cache_entries if cache_entries else 0
    return {
        "mode": mode,
        "m": m,
        "n": args.n,
        "k": args.k,
        "avg_gpu_ms": statistics.mean(gpu_samples),
        "median_gpu_ms": statistics.median(gpu_samples),
        "avg_wall_ms": statistics.mean(wall_samples),
        "median_wall_ms": statistics.median(wall_samples),
        "cache_entries": cache_entries,
        "cache_bytes": cache_bytes,
        "b_state_bytes": state_bytes,
        "avoided_b_state_bytes": stats.hits * state_bytes,
        "hits": stats.hits,
        "misses": stats.misses,
        "evictions": stats.evictions,
        "oversize": stats.oversize,
        "disabled": stats.disabled,
        "cache_budget_bytes": args.cache_bytes,
    }


def _print_rows(rows: list[dict[str, float | int | str]]) -> None:
    print(
        "mode,m,n,k,avg_gpu_ms,median_gpu_ms,avg_wall_ms,median_wall_ms,"
        "cache_entries,cache_bytes,b_state_bytes,avoided_b_state_bytes,"
        "hits,misses,evictions,oversize,disabled,cache_budget_bytes",
        flush=True,
    )
    for row in rows:
        print(
            f"{row['mode']},{row['m']},{row['n']},{row['k']},"
            f"{row['avg_gpu_ms']:.3f},{row['median_gpu_ms']:.3f},"
            f"{row['avg_wall_ms']:.3f},{row['median_wall_ms']:.3f},"
            f"{row['cache_entries']},{row['cache_bytes']},{row['b_state_bytes']},"
            f"{row['avoided_b_state_bytes']},{row['hits']},{row['misses']},"
            f"{row['evictions']},{row['oversize']},{row['disabled']},"
            f"{row['cache_budget_bytes']}",
            flush=True,
        )


def main() -> None:
    args = _parse_args()
    _require_gpu(args.allow_non_h100)
    settings = MinerSettings(
        no_gateway=True,
        skip_block_submission=True,
        prepared_b_cache_bytes=args.cache_bytes,
    )
    init_async_manager(settings)
    init_pinned_pool(settings.pinned_pool_size)
    try:
        rows = []
        for m in args.m_values:
            a, b, scale_a, scale_b = _make_inputs(m, args)
            for mode in args.modes:
                rows.append(_measure(mode, args, m, a, b, scale_a, scale_b))
        _print_rows(rows)
    finally:
        clear_prepared_b_cache()
        delete_state()


if __name__ == "__main__":
    main()
