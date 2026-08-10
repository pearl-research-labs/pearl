"""Benchmark MoE expert-local mining gate break-even on a RunPod H100.

Example:
    uv run --package vllm-miner python \
        miner/vllm-miner/benchmarks/bench_moe_expert_gate.py
"""

from __future__ import annotations

import argparse
import os
import statistics
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from compressed_tensors.quantization import QuantizationArgs
from miner_base.settings import MinerSettings
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.v1.worker.workspace import init_workspace_manager, is_workspace_manager_initialized
from vllm_miner.config import config as pearl_config
from vllm_miner.mining_state import delete_state, init_async_manager, init_pinned_pool
from vllm_miner.pearl_moe_experts import (
    _EXPERT_LOCAL_MIN_M_ENV,
    _EXPERT_LOCAL_MINING_ENV,
)
from vllm_miner.pearl_moe_method import PearlMoEMethod

_DOWN_WEIGHT_QUANT = QuantizationArgs(
    num_bits=8,
    type="float",
    strategy="block",
    block_structure=[128, 128],
    symmetric=True,
    dynamic=False,
)
_DOWN_INPUT_QUANT = QuantizationArgs(
    num_bits=8,
    type="float",
    strategy="group",
    group_size=128,
    dynamic=True,
    symmetric=True,
)


@contextmanager
def _temporary_env(name: str, value: str | None) -> Iterator[None]:
    old_value = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_value


def _global_min_m() -> int:
    return int(pearl_config.matrix_multiplication_config["use_simplified_gemm"]["min_m"])


def _make_topk_ids_from_counts(
    counts: list[int],
    top_k: int,
    device: torch.device,
) -> torch.Tensor:
    total_slots = sum(counts)
    if total_slots % top_k != 0:
        raise ValueError(f"sum(counts)={total_slots} must be divisible by top_k={top_k}")
    flat = torch.cat(
        [
            torch.full((count,), expert_id, dtype=torch.int32, device=device)
            for expert_id, count in enumerate(counts)
        ]
    )
    return flat.reshape(total_slots // top_k, top_k).contiguous()


def _make_cases(num_experts: int, min_m: int, top_k: int) -> list[tuple[str, list[int]]]:
    cold = min_m - 1

    def counts_with(hot_counts: list[int]) -> list[int]:
        counts = [cold] * num_experts
        for expert_id, count in enumerate(hot_counts):
            if expert_id >= num_experts:
                raise ValueError(f"num_experts={num_experts} is too small for benchmark cases")
            counts[expert_id] = count
        remainder = sum(counts) % top_k
        if remainder:
            counts[-1] -= remainder
            if counts[-1] < 0:
                raise ValueError(f"top_k={top_k} is too large for generated counts")
        return counts

    return [
        ("0-hot-all-cold", counts_with([])),
        ("1-hot-exact", counts_with([min_m])),
        ("2-hot-exact", counts_with([min_m, min_m])),
        ("several-hot-mixed", counts_with([min_m - 1, min_m, min_m + 1, min_m * 2])),
        ("below-above-edge", counts_with([min_m - 1, min_m + 1])),
    ]


def _build_layer(
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    device: torch.device,
) -> tuple[PearlMoEMethod, torch.nn.Module]:
    if not is_workspace_manager_initialized():
        init_workspace_manager(device)

    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        activation=MoEActivation.SILU,
        device=str(device),
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
    )
    moe_method = PearlMoEMethod(moe_config, _DOWN_WEIGHT_QUANT, _DOWN_INPUT_QUANT)
    layer = torch.nn.Module()
    moe_method.create_weights(
        layer,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        params_dtype=torch.bfloat16,
    )

    weight_scale = 1.0 / 64.0
    layer.w13_weight.data.copy_(
        torch.randint(
            -63,
            64,
            (num_experts, 2 * intermediate_size, hidden_size),
            dtype=torch.int8,
            device=device,
        )
    )
    layer.w2_weight.data.copy_(
        (torch.randn(num_experts, hidden_size, intermediate_size, device=device) * weight_scale).to(
            torch.float8_e4m3fn
        )
    )
    layer.w13_weight_scale.data.fill_(weight_scale)
    layer.w2_weight_scale.data.fill_(1.0)

    layer.to(device)
    init_pinned_pool()
    moe_method.process_weights_after_loading(layer)
    layer.activation = MoEActivation.SILU
    layer.global_num_experts = num_experts
    layer.apply_router_weight_on_input = False
    layer.expert_map = None
    return moe_method, layer


def _reset_mining_state(no_mining: bool) -> None:
    delete_state()
    init_async_manager(
        MinerSettings(no_gateway=True, no_mining=no_mining, skip_block_submission=True)
    )
    init_pinned_pool()


def _time_case(
    moe_method: PearlMoEMethod,
    layer: torch.nn.Module,
    counts: list[int],
    *,
    top_k: int,
    hidden_size: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> float:
    topk_ids = _make_topk_ids_from_counts(counts, top_k, device)
    num_tokens = topk_ids.shape[0]
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    topk_weights = torch.full((num_tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device)

    for _ in range(warmup):
        moe_method.apply(layer, hidden_states, topk_weights, topk_ids, None)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        moe_method.apply(layer, hidden_states, topk_weights, topk_ids, None)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def _run_mode(
    mode: str,
    no_mining: bool,
    expert_local_env: str | None,
    moe_method: PearlMoEMethod,
    layer: torch.nn.Module,
    cases: list[tuple[str, list[int]]],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    rows = []
    _reset_mining_state(no_mining=no_mining)
    with _temporary_env(_EXPERT_LOCAL_MINING_ENV, expert_local_env):
        for case_name, counts in cases:
            samples = [
                _time_case(
                    moe_method,
                    layer,
                    counts,
                    top_k=args.top_k,
                    hidden_size=args.hidden_size,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device=device,
                )
                for _ in range(args.repeats)
            ]
            avg_ms = statistics.mean(samples)
            total_slots = sum(counts)
            if no_mining:
                mined_slots = 0
            elif expert_local_env == "0":
                mined_slots = sum(count for count in counts if count > 0)
            else:
                mined_slots = sum(count for count in counts if count >= args.min_m)
            rows.append(
                {
                    "mode": mode,
                    "case": case_name,
                    "qualifying": sum(count >= args.min_m for count in counts),
                    "slots": total_slots,
                    "mined_slots": mined_slots,
                    "tokens": total_slots // args.top_k,
                    "avg_ms": avg_ms,
                    "slots_per_s": total_slots / (avg_ms / 1000.0),
                    "mined_slots_per_s": mined_slots / (avg_ms / 1000.0),
                }
            )
    return rows


def _print_rows(rows: list[dict[str, float | int | str]]) -> None:
    print(
        "mode,case,qualifying_experts,routed_slots,mined_slots,tokens,"
        "avg_ms,slots_per_s,mined_slots_per_s",
        flush=True,
    )
    for row in rows:
        print(
            f"{row['mode']},{row['case']},{row['qualifying']},{row['slots']},"
            f"{row['mined_slots']},{row['tokens']},{row['avg_ms']:.3f},"
            f"{row['slots_per_s']:.1f},{row['mined_slots_per_s']:.1f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--min-m", type=int, default=_global_min_m())
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--allow-non-h100", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(device)
    if "H100" not in device_name and not args.allow_non_h100:
        raise RuntimeError(f"Expected H100 for this benchmark, got {device_name!r}")

    with _temporary_env(_EXPERT_LOCAL_MIN_M_ENV, str(args.min_m)):
        _reset_mining_state(no_mining=True)
        moe_method, layer = _build_layer(
            num_experts=args.num_experts,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            top_k=args.top_k,
            device=device,
        )
        cases = _make_cases(args.num_experts, args.min_m, args.top_k)
        rows = []
        rows.extend(
            _run_mode(
                "grouped-vanilla",
                True,
                None,
                moe_method,
                layer,
                cases,
                args,
                device,
            )
        )
        rows.extend(
            _run_mode(
                "mine-all-active",
                False,
                "0",
                moe_method,
                layer,
                cases,
                args,
                device,
            )
        )
        rows.extend(
            _run_mode(
                "expert-local-gate",
                False,
                None,
                moe_method,
                layer,
                cases,
                args,
                device,
            )
        )
        _print_rows(rows)
    delete_state()


if __name__ == "__main__":
    main()
