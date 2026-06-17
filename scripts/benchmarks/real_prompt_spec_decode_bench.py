#!/usr/bin/env python3
"""Run real-prompt vLLM speculative decoding throughput benchmarks.

This benchmark is intended for Pearl/EAGLE mining experiments. It uses real
prompts from JSONL files (by default ``RedHatAI/speculator_benchmarks``), serves
one vLLM configuration at a time, sends OpenAI-compatible completion requests,
and records both throughput and vLLM speculative-decoding counters.

Why this exists: random-token vLLM throughput runs are useful for stress tests,
but they are the opposite of what EAGLE learns. EAGLE learns natural verifier
continuations, so acceptance and mining tradeoffs must be measured on real
prompt distributions.

Example:

    python scripts/benchmarks/real_prompt_spec_decode_bench.py \
      --model pearl-ai/Llama-3.1-8B-Instruct-pearl \
      --eagle-checkpoint /workspace/.../checkpoints_eagle3_scratch/3 \
      --output-dir /workspace/pearl-real-prompt-bench \
      --gpu 0 \
      --bucket-lengths 0:128,129:256,257:512,513:1024,1025:2048 \
      --samples-per-bucket 64 \
      --cases vanilla_no_mining,vanilla_mining,eagle_k2_mining,eagle_k3_mining
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SPEC_METRIC_NAMES = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)

DEFAULT_SUBSETS = (
    "qa,writing,summarization,math_reasoning,HumanEval,translation,question,rag,tool_call"
)
DEFAULT_CASES = "vanilla_no_mining,vanilla_mining,eagle_k2_mining,eagle_k3_mining"


@dataclass(frozen=True)
class PromptRecord:
    group: str
    subset: str
    prompt: str
    prompt_len: int | None = None


@dataclass(frozen=True)
class CaseConfig:
    name: str
    mining: bool
    spec_k: int = 0


class BenchmarkError(RuntimeError):
    """Raised when a benchmark case cannot complete."""


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def prompt_from_row(row: dict[str, Any]) -> str:
    """Extract a prompt string from common real-prompt JSONL schemas."""
    for key in ("prompt", "question", "text", "instruction"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value

    messages = row.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role", "")
                content = message.get("content", "")
                if isinstance(content, str) and content.strip():
                    parts.append(f"{role}: {content}" if role else content)
        if parts:
            return "\n".join(parts)

    # Last-resort fallback keeps the row deterministic and debuggable.
    return json.dumps(row, ensure_ascii=False)[:4000]


def parse_bucket_lengths(spec: str | None) -> list[tuple[str, int, int]]:
    if not spec:
        return []
    buckets: list[tuple[str, int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            lo_s, hi_s = part.split(":", 1)
            lo = int(lo_s)
            hi = int(hi_s)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid bucket {part!r}; expected '<lo>:<hi>'"
            ) from exc
        if lo < 0 or hi < lo:
            raise argparse.ArgumentTypeError(f"Invalid bucket range {part!r}")
        buckets.append((f"{lo:04d}_{hi:04d}", lo, hi))
    return buckets


def parse_case(case_name: str) -> CaseConfig:
    name = case_name.strip()
    if name == "vanilla_no_mining":
        return CaseConfig(name=name, mining=False, spec_k=0)
    if name == "vanilla_mining":
        return CaseConfig(name=name, mining=True, spec_k=0)
    match = re.fullmatch(r"(?:scratch_)?eagle_k(\d+)_mining", name)
    if match:
        return CaseConfig(name=name, mining=True, spec_k=int(match.group(1)))
    raise argparse.ArgumentTypeError(
        f"Unknown case {name!r}; use vanilla_no_mining, vanilla_mining, "
        "or eagle_k<N>_mining"
    )


def load_jsonl_prompts_from_hf(
    *, repo: str, subset: str, cache_dir: str | None = None
) -> list[str]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on remote env
        raise BenchmarkError(
            "huggingface_hub is required to load HF benchmark prompts"
        ) from exc

    path = hf_hub_download(
        repo_id=repo,
        filename=f"{subset}.jsonl",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    prompts: list[str] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            prompts.append(prompt_from_row(json.loads(line)))
    return prompts


def load_prompt_records(args: argparse.Namespace) -> list[PromptRecord]:  # noqa: C901
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    if not subsets:
        raise BenchmarkError("No subsets selected")

    tokenizer = None
    buckets = parse_bucket_lengths(args.bucket_lengths)
    if buckets:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on remote env
            raise BenchmarkError("transformers is required for length buckets") from exc
        tokenizer_name = args.tokenizer or args.model
        log(f"Loading tokenizer for length buckets: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    rng = random.Random(args.seed)
    grouped: dict[str, list[PromptRecord]] = {}

    for subset in subsets:
        prompts = load_jsonl_prompts_from_hf(
            repo=args.dataset_repo, subset=subset, cache_dir=args.hf_cache_dir
        )
        rng.shuffle(prompts)
        if buckets:
            for prompt in prompts:
                assert tokenizer is not None
                prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
                for bucket_name, lo, hi in buckets:
                    if lo <= prompt_len <= hi:
                        grouped.setdefault(bucket_name, []).append(
                            PromptRecord(
                                group=bucket_name,
                                subset=subset,
                                prompt=prompt,
                                prompt_len=prompt_len,
                            )
                        )
                        break
        else:
            take = prompts[: args.samples_per_subset]
            grouped.setdefault(subset, [])
            grouped[subset].extend(
                PromptRecord(group=subset, subset=subset, prompt=prompt)
                for prompt in take
            )

    selected: list[PromptRecord] = []
    if buckets:
        for bucket_name, _lo, _hi in buckets:
            records = grouped.get(bucket_name, [])
            rng.shuffle(records)
            take = records[: args.samples_per_bucket]
            if not take:
                log(f"WARNING: bucket {bucket_name} has no prompts")
                continue
            lengths = [r.prompt_len for r in take if r.prompt_len is not None]
            log(
                f"bucket={bucket_name} prompts={len(take)} "
                f"avg_len={sum(lengths) / len(lengths):.1f} "
                f"min={min(lengths)} max={max(lengths)}"
            )
            selected.extend(take)
    else:
        for subset in subsets:
            records = grouped.get(subset, [])
            if not records:
                log(f"WARNING: subset {subset} has no prompts")
                continue
            selected.extend(records[: args.samples_per_subset])
            log(f"subset={subset} prompts={len(records[: args.samples_per_subset])}")

    if not selected:
        raise BenchmarkError("No prompts selected")
    return selected


def http_get_json(url: str, timeout_s: float = 10) -> Any:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(base_url: str, process: subprocess.Popen[Any], timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise BenchmarkError(f"vLLM server exited early with rc={process.returncode}")
        try:
            http_get_json(f"{base_url}/v1/models", timeout_s=5)
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(2)
    raise BenchmarkError(f"Timed out waiting for {base_url}: {last_error}")


def parse_prometheus_metrics(text: str) -> dict[str, Any]:
    values = dict.fromkeys(SPEC_METRIC_NAMES, 0.0)
    per_pos: dict[int, float] = {}

    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([^\s{]+)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)", line)
        if not match:
            continue
        name, labels, value_s = match.groups()
        labels = labels or ""
        try:
            value = float(value_s)
        except ValueError:
            continue

        # Prometheus counters are exported as *_total, while vLLM source names omit
        # the suffix. Normalize so this script works across vLLM versions.
        base_name = name[:-6] if name.endswith("_total") else name
        if base_name in values:
            values[base_name] += value

        if "spec_decode" in name and "per_pos" in name:
            pos_match = re.search(r'(?:pos|position|index)="?(\d+)"?', labels)
            if pos_match:
                pos = int(pos_match.group(1))
                per_pos[pos] = per_pos.get(pos, 0.0) + value

    values["per_pos"] = per_pos
    return values


def fetch_metrics(base_url: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as response:  # noqa: S310
        return parse_prometheus_metrics(response.read().decode("utf-8"))


def metric_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    diff = {name: after.get(name, 0.0) - before.get(name, 0.0) for name in SPEC_METRIC_NAMES}
    before_pos = before.get("per_pos", {})
    after_pos = after.get("per_pos", {})
    diff["per_pos"] = {
        pos: after_pos.get(pos, 0.0) - before_pos.get(pos, 0.0)
        for pos in set(before_pos) | set(after_pos)
    }
    return diff


def request_completion(
    *,
    base_url: str,
    model_name: str,
    prompt: PromptRecord,
    max_tokens: int,
    request_timeout_s: int,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model_name,
            "prompt": prompt.prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    with urllib.request.urlopen(request, timeout=request_timeout_s) as response:  # noqa: S310
        data = json.loads(response.read().decode("utf-8"))
    usage = data.get("usage", {})
    choice = data.get("choices", [{}])[0]
    return {
        "group": prompt.group,
        "subset": prompt.subset,
        "prompt_len": prompt.prompt_len,
        "elapsed_s": time.time() - start,
        "completion_tokens": usage.get("completion_tokens") or 0,
        "prompt_tokens": usage.get("prompt_tokens") or 0,
        "finish_reason": choice.get("finish_reason"),
    }


def summarize_group(
    *,
    case: CaseConfig,
    group: str,
    records: list[PromptRecord],
    results: list[dict[str, Any]],
    elapsed_s: float,
    metrics: dict[str, Any],
    max_tokens: int,
    concurrency: int,
) -> dict[str, Any]:
    completion_tokens = sum(int(r["completion_tokens"]) for r in results)
    prompt_tokens = sum(int(r["prompt_tokens"]) for r in results)
    total_tokens = completion_tokens + prompt_tokens
    drafts = float(metrics.get("vllm:spec_decode_num_drafts", 0.0))
    draft_tokens = float(metrics.get("vllm:spec_decode_num_draft_tokens", 0.0))
    accepted_tokens = float(metrics.get("vllm:spec_decode_num_accepted_tokens", 0.0))
    per_pos = metrics.get("per_pos", {})

    lengths = [r.prompt_len for r in records if r.prompt_len is not None]
    finish_counts: dict[str, int] = {}
    for result in results:
        key = str(result.get("finish_reason"))
        finish_counts[key] = finish_counts.get(key, 0) + 1

    return {
        "case": case.name,
        "group": group,
        "mining": case.mining,
        "spec_k": case.spec_k,
        "num_requests": len(records),
        "avg_prompt_len": (sum(lengths) / len(lengths)) if lengths else None,
        "min_prompt_len": min(lengths) if lengths else None,
        "max_prompt_len": max(lengths) if lengths else None,
        "elapsed_s": elapsed_s,
        "requests_per_second": len(records) / elapsed_s,
        "output_tokens_per_second": completion_tokens / elapsed_s,
        "total_tokens_per_second": total_tokens / elapsed_s,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "max_tokens": max_tokens,
        "concurrency": concurrency,
        "num_drafts": drafts,
        "num_draft_tokens": draft_tokens,
        "num_accepted_tokens": accepted_tokens,
        "acceptance_length": 1 + accepted_tokens / drafts if drafts > 0 else None,
        "avg_draft_acceptance_rate": accepted_tokens / draft_tokens
        if draft_tokens > 0
        else None,
        "acceptance_at_pos": {
            str(pos): count / drafts for pos, count in sorted(per_pos.items())
        }
        if drafts > 0
        else {},
        "raw_metrics_diff": metrics,
        "finish_counts": finish_counts,
    }


def build_server_command(args: argparse.Namespace, case: CaseConfig, port: int) -> list[str]:
    command = [
        args.vllm_bin,
        "serve",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(port),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.disable_prefix_caching:
        command.append("--no-enable-prefix-caching")
    if case.spec_k > 0:
        if not args.eagle_checkpoint:
            raise BenchmarkError(f"{case.name} requires --eagle-checkpoint")
        command.extend(
            [
                "--speculative-config",
                json.dumps(
                    {
                        "method": args.eagle_method,
                        "model": args.eagle_checkpoint,
                        "num_speculative_tokens": case.spec_k,
                    },
                    separators=(",", ":"),
                ),
            ]
        )
    return command


def run_case(
    *,
    args: argparse.Namespace,
    case: CaseConfig,
    prompts_by_group: dict[str, list[PromptRecord]],
    port: int,
) -> dict[str, Any]:
    case_dir = Path(args.output_dir) / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = case_dir / "server.log"
    base_url = f"http://{args.host}:{port}"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.miner_no_gateway:
        env["MINER_NO_GATEWAY"] = "1"
    if args.skip_block_submission:
        env["MINER_SKIP_BLOCK_SUBMISSION"] = "1"
    if case.mining:
        env.pop("MINER_NO_MINING", None)
    else:
        env["MINER_NO_MINING"] = "1"

    command = build_server_command(args, case, port)
    log(f"case={case.name} starting vLLM on gpu={args.gpu} port={port}")
    log("server command: " + " ".join(command))
    with server_log_path.open("w", encoding="utf-8") as server_log:
        process = subprocess.Popen(  # noqa: S603 - benchmark harness starts vLLM intentionally
            command,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        try:
            wait_for_server(base_url, process, args.server_timeout_s)
            log(f"case={case.name} server ready")

            group_summaries: dict[str, dict[str, Any]] = {}
            all_results: list[dict[str, Any]] = []
            for group, records in prompts_by_group.items():
                before = fetch_metrics(base_url)
                start = time.time()
                workers = min(args.concurrency, len(records))
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [
                        pool.submit(
                            request_completion,
                            base_url=base_url,
                            model_name=args.served_model_name,
                            prompt=record,
                            max_tokens=args.max_tokens,
                            request_timeout_s=args.request_timeout_s,
                        )
                        for record in records
                    ]
                    results = [future.result() for future in futures]
                elapsed_s = time.time() - start
                after = fetch_metrics(base_url)
                metrics = metric_diff(before, after)
                summary = summarize_group(
                    case=case,
                    group=group,
                    records=records,
                    results=results,
                    elapsed_s=elapsed_s,
                    metrics=metrics,
                    max_tokens=args.max_tokens,
                    concurrency=workers,
                )
                group_summaries[group] = summary
                all_results.extend(results)
                log(
                    f"case={case.name} group={group} "
                    f"req/s={summary['requests_per_second']:.3f} "
                    f"out_tok/s={summary['output_tokens_per_second']:.1f} "
                    f"acc_len={summary['acceptance_length']}"
                )

            (case_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "case": asdict(case),
                        "groups": group_summaries,
                        "results": all_results,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return group_summaries
        finally:
            log(f"case={case.name} stopping vLLM")
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)


def _pct_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    return (value / baseline - 1) * 100


def add_primary_baseline_deltas(
    all_summaries: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Add deltas against the primary production baseline: vanilla no-mining.

    The PR claim is not "EAGLE beats mining". Vanilla mining is included to
    show the mining tax. The primary comparison is EAGLE+mining vs vanilla
    no-mining, because a win there means mining work is effectively hidden from
    the serving-throughput perspective.
    """
    enriched = json.loads(json.dumps(all_summaries))
    baseline_groups = enriched.get("vanilla_no_mining", {})
    for case_groups in enriched.values():
        for group, summary in case_groups.items():
            baseline = baseline_groups.get(group)
            if not baseline:
                continue
            summary["primary_baseline_case"] = "vanilla_no_mining"
            summary["requests_vs_vanilla_no_mining_pct"] = _pct_delta(
                summary.get("requests_per_second"), baseline.get("requests_per_second")
            )
            summary["output_tokens_vs_vanilla_no_mining_pct"] = _pct_delta(
                summary.get("output_tokens_per_second"),
                baseline.get("output_tokens_per_second"),
            )
            summary["total_tokens_vs_vanilla_no_mining_pct"] = _pct_delta(
                summary.get("total_tokens_per_second"),
                baseline.get("total_tokens_per_second"),
            )
    return enriched


def write_combined_outputs(output_dir: Path, all_summaries: dict[str, dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched = add_primary_baseline_deltas(all_summaries)
    (output_dir / "combined_summary.json").write_text(
        json.dumps(enriched, indent=2), encoding="utf-8"
    )

    fieldnames = [
        "case",
        "group",
        "mining",
        "spec_k",
        "num_requests",
        "avg_prompt_len",
        "requests_per_second",
        "output_tokens_per_second",
        "total_tokens_per_second",
        "primary_baseline_case",
        "requests_vs_vanilla_no_mining_pct",
        "output_tokens_vs_vanilla_no_mining_pct",
        "total_tokens_vs_vanilla_no_mining_pct",
        "acceptance_length",
        "avg_draft_acceptance_rate",
        "acceptance_at_pos_0",
        "acceptance_at_pos_1",
        "acceptance_at_pos_2",
        "acceptance_at_pos_3",
    ]
    with (output_dir / "combined_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for case_groups in enriched.values():
            for summary in case_groups.values():
                row = {key: summary.get(key) for key in fieldnames}
                acceptance_at_pos = summary.get("acceptance_at_pos", {})
                for pos in range(4):
                    row[f"acceptance_at_pos_{pos}"] = acceptance_at_pos.get(str(pos))
                writer.writerow(row)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-prompt vLLM speculative decoding throughput benchmark"
    )
    parser.add_argument("--model", required=True, help="Target model name/path for vLLM")
    parser.add_argument("--served-model-name", default="pearl")
    parser.add_argument("--eagle-checkpoint", default=None, help="EAGLE checkpoint path")
    parser.add_argument("--eagle-method", default="eagle3")
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-repo", default="RedHatAI/speculator_benchmarks")
    parser.add_argument("--subsets", default=DEFAULT_SUBSETS)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--bucket-lengths", default=None)
    parser.add_argument("--samples-per-subset", type=int, default=24)
    parser.add_argument("--samples-per-bucket", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port-base", type=int, default=8130)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--server-timeout-s", type=int, default=180)
    parser.add_argument("--request-timeout-s", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--disable-prefix-caching", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--miner-no-gateway",
        dest="miner_no_gateway",
        action="store_true",
        default=True,
        help="Set MINER_NO_GATEWAY=1 while serving benchmark cases (default)",
    )
    parser.add_argument(
        "--with-gateway",
        dest="miner_no_gateway",
        action="store_false",
        help="Do not set MINER_NO_GATEWAY; use the configured mining gateway",
    )
    parser.add_argument(
        "--skip-block-submission", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load/sample prompts and write prompt_manifest.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [parse_case(case) for case in args.cases.split(",") if case.strip()]
    prompts = load_prompt_records(args)
    prompts_by_group: dict[str, list[PromptRecord]] = {}
    for prompt in prompts:
        prompts_by_group.setdefault(prompt.group, []).append(prompt)

    (output_dir / "prompt_manifest.json").write_text(
        json.dumps([asdict(prompt) for prompt in prompts], indent=2),
        encoding="utf-8",
    )
    log(
        f"selected prompts={len(prompts)} groups="
        + ",".join(f"{group}:{len(rows)}" for group, rows in prompts_by_group.items())
    )
    if args.dry_run:
        log("dry run complete")
        return 0

    all_summaries: dict[str, dict[str, Any]] = {}
    for idx, case in enumerate(cases, start=1):
        port = args.port_base + idx
        summaries = run_case(
            args=args, case=case, prompts_by_group=prompts_by_group, port=port
        )
        all_summaries[case.name] = summaries
        write_combined_outputs(output_dir, all_summaries)

    log("combined results (primary comparator: vanilla_no_mining):")
    enriched = add_primary_baseline_deltas(all_summaries)
    for case_name, groups in enriched.items():
        for group, summary in groups.items():
            vs_no_mining = summary.get("requests_vs_vanilla_no_mining_pct")
            vs_text = "n/a" if vs_no_mining is None else f"{vs_no_mining:+.1f}%"
            log(
                f"{case_name}/{group}: req/s={summary['requests_per_second']:.3f} "
                f"vs_no_mining={vs_text} "
                f"out_tok/s={summary['output_tokens_per_second']:.1f} "
                f"acc_len={summary['acceptance_length']} "
                f"draft_accept={summary['avg_draft_acceptance_rate']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
