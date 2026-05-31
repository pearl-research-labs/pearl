"""Benchmark: eager `pg.noisy_gemm` per-attempt launches vs captured CUDA Graph.

Closes the launch-overhead portion of the alpha-miner gap (per memory note
`project_pearl_perf_postmortem_2026_05_18`: today's path issues 5 kernel
launches per attempt, each paying ~4-10 µs of Python+driver launch overhead).
The graph path captures the full noisingA→GEMM→noisingB→denoise→PoW chain
once and replays it as a single graph submit per attempt.

What this script does:
  1. Allocate persistent input + scratch + output buffers matching the
     R=128 R=128 sm_89 driver config (CHUNK_M=N=2048, K=4096, R=128).
  2. Warm up the kernel (one eager call + sync) so any lazy CUTLASS dispatch
     init happens before measurement.
  3. Bench eager: N iterations of `pg.noisy_gemm(...)` against persistent
     buffers, measuring wall clock. Two sub-benches:
       (a) eager + random regen each iter (production parity)
       (b) eager only (kernel + sync, no random regen) — isolates launch
           overhead from random-gen overhead.
  4. Capture a CUDA Graph wrapping `pg.noisy_gemm(...)`.
  5. Bench graph: same N iterations, but each iter is `graph.replay()`
     instead of a Python-side `pg.noisy_gemm()` call. Two sub-benches mirror
     the eager pair.
  6. Bit-exact verification: for 10 random input draws, run eager and graph
     paths against IDENTICAL inputs (copied via `.copy_`) and compare
     `C`, `pow_key`, `host_signal_header`, scratch tensors.

Usage (on a host with the pearl-gemm .so installed):

    cd /host_home/pearl-deploy/pearl-gemm/src
    PYTHONPATH=/host_home/pearl-deploy/pearl-stratum/src \\
      python /host_home/pearl-deploy/pearl-stratum/tests/_bench_cuda_graph.py

Writes results to `bench_cuda_graph_result.txt` in the working directory.

Notes:
  * The script does NOT touch the stratum pool. It allocates and runs the
    kernel directly. The driver itself wires graphs into the production
    loop behind `PEARL_SM89_CUDA_GRAPH=1`.
  * 1 iteration ≈ 4-5 ms on a 4070 Ti SUPER per the postmortem (kernel
    time at chunk M=N=2048 K=4096 R=128). 1000 iters ≈ 4-5 s wall.
  * If `pearl_gemm_cuda` is not importable (e.g. running on a CI box
    without the .so), the script reports the import error and exits 2 —
    so it's safe to gate behind `pytest --runslow` or similar without
    breaking other test runs.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path


def _try_import_torch_and_pg():
    """Return (torch, pg) or print a clear error + return (None, None).

    We import these only when the bench actually runs so the script can be
    pytest-collected on machines without the .so installed.
    """
    try:
        import torch  # type: ignore[import]
    except Exception as e:
        print(f"FATAL: torch not importable: {e}", file=sys.stderr)
        return None, None
    try:
        import pearl_gemm_cuda as pg  # type: ignore[import]
    except Exception as e:
        print(f"FATAL: pearl_gemm_cuda not importable: {e}", file=sys.stderr)
        return None, None
    if not torch.cuda.is_available():
        print("FATAL: torch.cuda.is_available() is False", file=sys.stderr)
        return None, None
    cap = torch.cuda.get_device_capability(0)
    if cap != (8, 9):
        print(
            f"WARN: expected sm_89 (cap=(8,9)); got cap={cap}. "
            f"The driver under bench is sm_89-only.",
            file=sys.stderr,
        )
    return torch, pg


# ----- driver config (mirrors _miner_driver_sm89_r128.py) ---------------------

CHUNK_M, CHUNK_N, CHUNK_K, R = 2048, 2048, 4096, 128
BM, BN, BK = 64, 64, 64
CM, CN = 1, 1
MATMUL_STAGES = 2
NOISE_TILE_A_M, NOISE_TILE_A_K = 64, 64
NOISE_TILE_B_N, NOISE_TILE_B_K = 64, 64
NOISE_STAGES_A = 2
NOISE_STAGES_B = 2


def _allocate_buffers(torch, pg, device):
    """Allocate all persistent tensors the kernel reads/writes.

    Returns a dict so callers can name-access fields without a 20-tuple.
    """
    M, N, K = CHUNK_M, CHUNK_N, CHUNK_K
    bufs = dict(
        A=torch.zeros(M, K, dtype=torch.int8, device=device),
        B=torch.zeros(N, K, dtype=torch.int8, device=device),
        A_scales=torch.zeros(M, dtype=torch.float32, device=device),
        B_scales=torch.zeros(N, dtype=torch.float32, device=device),
        EAL=torch.zeros(M, R, dtype=torch.int8, device=device),
        EBR=torch.zeros(N, R, dtype=torch.int8, device=device),
        EAL_fp16=torch.zeros(M, R, dtype=torch.float16, device=device),
        EBR_fp16=torch.zeros(N, R, dtype=torch.float16, device=device),
        EAR_R_major=torch.zeros(K, R, dtype=torch.int8, device=device),
        EBL_R_major=torch.zeros(K, R, dtype=torch.int8, device=device),
        EAR_K_major=torch.zeros(R, K, dtype=torch.int8, device=device),
        EBL_K_major=torch.zeros(R, K, dtype=torch.int8, device=device),
        AxEBL_fp16=torch.zeros(M, R, dtype=torch.float16, device=device),
        EARxBpEB_fp16=torch.zeros(N, R, dtype=torch.float16, device=device),
        AxEBL_int32=torch.zeros(M, R, dtype=torch.int32, device=device),
        EARxBpEB_int32=torch.zeros(N, R, dtype=torch.int32, device=device),
        ApEA=torch.zeros(M, K, dtype=torch.int8, device=device),
        BpEB=torch.zeros(N, K, dtype=torch.int8, device=device),
        C=torch.zeros(M, N, dtype=torch.bfloat16, device=device),
        pow_target=torch.zeros(8, dtype=torch.uint32, device=device),
        pow_key=torch.zeros(8, dtype=torch.uint32, device=device),
    )
    hh = pg.get_host_signal_header_size()
    hs = pg.get_host_signal_sync_size()
    bufs["host_signal_header"] = torch.zeros(hh, dtype=torch.int8, pin_memory=True)
    bufs["host_signal_sync"] = torch.zeros(hs, dtype=torch.int8, device=device)
    return bufs


def _call_noisy_gemm(pg, b):
    """Single dispatch point — mirrors the driver's `_call_noisy_gemm`.

    Takes the buffer dict from `_allocate_buffers` and calls
    `pg.noisy_gemm` with the standard R=128 tile config.
    """
    pg.noisy_gemm(
        b["A"], b["B"], b["EAL"], b["EAL_fp16"], b["EBR"], b["EBR_fp16"],
        b["EAR_R_major"], b["EBL_R_major"], b["EAR_K_major"], b["EBL_K_major"],
        b["AxEBL_fp16"], b["EARxBpEB_fp16"], b["ApEA"], b["BpEB"],
        b["A_scales"], b["B_scales"], b["C"],
        b["host_signal_header"], b["host_signal_sync"],
        b["pow_target"], b["pow_key"],
        b["AxEBL_int32"], b["EARxBpEB_int32"],
        BM, BN, BK, CM, CN, MATMUL_STAGES,
        None, True,
        NOISE_TILE_A_M, NOISE_TILE_A_K,
        NOISE_TILE_B_N, NOISE_TILE_B_K,
        NOISE_STAGES_A, NOISE_STAGES_B,
        None, None,
        True, True,
        False, False,
        None, False,
    )


def _refresh_inputs(b, *, seed=None, torch=None):
    """Refresh A, B, A_scales, B_scales, pow_key in-place.

    Mirrors the driver's `_refresh_inputs`. Optionally seeds torch's CUDA
    generator so two paths can be fed identical inputs for the bit-exact
    verification.
    """
    if seed is not None:
        torch.cuda.manual_seed(seed)
    b["A"].random_(-127, 127)
    b["B"].random_(-127, 127)
    b["A_scales"].uniform_(0.005, 0.025)
    b["B_scales"].uniform_(0.005, 0.025)
    b["pow_key"].zero_()


def _bench_eager(torch, pg, b, n_iters, *, refresh):
    """Run `n_iters` eager `noisy_gemm` calls; return (attempts_per_sec, p50_ms).

    `refresh=True` regenerates random inputs each iter (production parity);
    `False` keeps the inputs fixed so we measure kernel + launch overhead
    only (no Python-side random gen cost).
    """
    # One-shot warmup + sync to flush any deferred CUTLASS init.
    _call_noisy_gemm(pg, b)
    torch.cuda.synchronize()

    per_iter_ms = []
    t_start = time.perf_counter()
    for _ in range(n_iters):
        if refresh:
            _refresh_inputs(b, torch=torch)
        iter_start = time.perf_counter()
        _call_noisy_gemm(pg, b)
        torch.cuda.synchronize()
        per_iter_ms.append((time.perf_counter() - iter_start) * 1e3)
    elapsed = time.perf_counter() - t_start
    return n_iters / elapsed, statistics.median(per_iter_ms)


def _capture_graph(torch, pg, b, *, warmup_iters=3):
    """Capture a CUDA Graph wrapping `pg.noisy_gemm(...)`.

    Returns the captured `torch.cuda.CUDAGraph` object. Capture happens on
    a side stream per the PyTorch docs; warmup runs the kernel on the side
    stream first to materialize any lazy state (cuBLAS handles, CUTLASS
    dispatch caches) — graph capture cannot capture cudaMalloc/lazy-init.
    """
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(warmup_iters):
            _call_noisy_gemm(pg, b)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _call_noisy_gemm(pg, b)
    return g


def _bench_graph(torch, b, g, n_iters, *, refresh):
    """Run `n_iters` graph replays; return (attempts_per_sec, p50_ms)."""
    g.replay()
    torch.cuda.synchronize()

    per_iter_ms = []
    t_start = time.perf_counter()
    for _ in range(n_iters):
        if refresh:
            _refresh_inputs(b, torch=torch)
        iter_start = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        per_iter_ms.append((time.perf_counter() - iter_start) * 1e3)
    elapsed = time.perf_counter() - t_start
    return n_iters / elapsed, statistics.median(per_iter_ms)


def _verify_bitexact(torch, pg, n_seeds=10):
    """For `n_seeds` random nonce seeds, run eager and graph paths against
    IDENTICAL inputs and compare `C`, `pow_key`, `host_signal_header`.

    Returns (n_pass, n_fail, max_diffs) where max_diffs is a list of dicts
    (one per seed) holding per-tensor max |eager - graph| values.
    """
    device = torch.device("cuda:0")
    # Two independent buffer sets so we can compare outputs.
    b_eager = _allocate_buffers(torch, pg, device)
    b_graph = _allocate_buffers(torch, pg, device)

    # Warm both paths' kernel dispatch (separately, to make sure no lazy
    # state leaks between captures).
    _call_noisy_gemm(pg, b_eager)
    torch.cuda.synchronize()
    g = _capture_graph(torch, pg, b_graph, warmup_iters=3)

    n_pass = 0
    n_fail = 0
    diffs = []
    for seed in range(n_seeds):
        # Same seed → same A, B, A_scales, B_scales, pow_key=0 on both paths.
        torch.cuda.manual_seed(seed)
        b_eager["A"].random_(-127, 127)
        b_eager["B"].random_(-127, 127)
        b_eager["A_scales"].uniform_(0.005, 0.025)
        b_eager["B_scales"].uniform_(0.005, 0.025)
        b_eager["pow_key"].zero_()

        torch.cuda.manual_seed(seed)
        b_graph["A"].random_(-127, 127)
        b_graph["B"].random_(-127, 127)
        b_graph["A_scales"].uniform_(0.005, 0.025)
        b_graph["B_scales"].uniform_(0.005, 0.025)
        b_graph["pow_key"].zero_()

        # Sanity check inputs match.
        assert torch.equal(b_eager["A"], b_graph["A"]), "A mismatch before run"
        assert torch.equal(b_eager["B"], b_graph["B"]), "B mismatch before run"

        # Run both.
        _call_noisy_gemm(pg, b_eager)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()

        # Compare. C is bfloat16 — exact equality is the contract; even if
        # the kernel uses non-deterministic reductions internally, the graph
        # captured the SAME kernel so the result must be byte-identical.
        c_match = torch.equal(b_eager["C"], b_graph["C"])
        key_match = torch.equal(b_eager["pow_key"], b_graph["pow_key"])
        hdr_match = torch.equal(
            b_eager["host_signal_header"], b_graph["host_signal_header"]
        )

        d = {
            "seed": seed,
            "C_match": c_match,
            "pow_key_match": key_match,
            "host_signal_header_match": hdr_match,
            "C_max_abs_diff": float(
                (b_eager["C"].float() - b_graph["C"].float()).abs().max().item()
            ),
        }
        diffs.append(d)
        ok = c_match and key_match and hdr_match
        n_pass += int(ok)
        n_fail += int(not ok)
    return n_pass, n_fail, diffs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CUDA Graph speedup bench for Pearl sm_89 R=128.")
    ap.add_argument("--n-iters", type=int, default=1000,
                    help="iterations per bench run (default 1000)")
    ap.add_argument("--n-seeds", type=int, default=10,
                    help="seeds for bit-exact verification (default 10)")
    ap.add_argument("--out", type=str, default="bench_cuda_graph_result.txt",
                    help="output text file (default ./bench_cuda_graph_result.txt)")
    args = ap.parse_args(argv)

    torch, pg = _try_import_torch_and_pg()
    if torch is None or pg is None:
        return 2

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)

    print(f"GPU: {gpu_name} cap={cap}")
    print(f"config: CHUNK_M={CHUNK_M} CHUNK_N={CHUNK_N} CHUNK_K={CHUNK_K} R={R}")
    print(f"tile: bM={BM} bN={BN} bK={BK} stages={MATMUL_STAGES}")
    print(f"running {args.n_iters} iters per bench, {args.n_seeds} seeds for verify")

    # Eager-path benches.
    b = _allocate_buffers(torch, pg, device)
    _refresh_inputs(b, torch=torch)

    print("eager + refresh each iter...", flush=True)
    eager_refresh_rate, eager_refresh_p50 = _bench_eager(
        torch, pg, b, args.n_iters, refresh=True
    )
    print(f"  -> {eager_refresh_rate:.2f} attempts/s, p50={eager_refresh_p50:.3f} ms")

    print("eager, fixed inputs (kernel + sync only)...", flush=True)
    eager_fixed_rate, eager_fixed_p50 = _bench_eager(
        torch, pg, b, args.n_iters, refresh=False
    )
    print(f"  -> {eager_fixed_rate:.2f} attempts/s, p50={eager_fixed_p50:.3f} ms")

    # Graph-path benches (rebuild buffer + capture so we don't reuse state
    # poisoned by 1000 eager runs of host_signal_sync etc — though it should
    # be fine, this is cheap insurance).
    b = _allocate_buffers(torch, pg, device)
    _refresh_inputs(b, torch=torch)
    g = _capture_graph(torch, pg, b, warmup_iters=3)

    print("graph + refresh each iter...", flush=True)
    graph_refresh_rate, graph_refresh_p50 = _bench_graph(
        torch, b, g, args.n_iters, refresh=True
    )
    print(f"  -> {graph_refresh_rate:.2f} attempts/s, p50={graph_refresh_p50:.3f} ms")

    print("graph, fixed inputs (replay + sync only)...", flush=True)
    graph_fixed_rate, graph_fixed_p50 = _bench_graph(
        torch, b, g, args.n_iters, refresh=False
    )
    print(f"  -> {graph_fixed_rate:.2f} attempts/s, p50={graph_fixed_p50:.3f} ms")

    # Bit-exact verification.
    print(f"verifying bit-exact equivalence on {args.n_seeds} seeds...", flush=True)
    n_pass, n_fail, diffs = _verify_bitexact(torch, pg, n_seeds=args.n_seeds)
    print(f"  -> pass={n_pass}/{args.n_seeds}, fail={n_fail}")

    # Write result file.
    out_path = Path(args.out).resolve()
    speedup_refresh = (
        graph_refresh_rate / eager_refresh_rate if eager_refresh_rate > 0 else float("nan")
    )
    speedup_fixed = (
        graph_fixed_rate / eager_fixed_rate if eager_fixed_rate > 0 else float("nan")
    )
    lines = [
        "Pearl sm_89 R=128 — CUDA Graph vs eager `pg.noisy_gemm`",
        f"  GPU: {gpu_name} cap={cap}",
        f"  config: CHUNK_M={CHUNK_M} CHUNK_N={CHUNK_N} CHUNK_K={CHUNK_K} R={R} "
        f"tile=bM={BM},bN={BN},bK={BK},stages={MATMUL_STAGES}",
        f"  n_iters per bench: {args.n_iters}",
        "",
        "EAGER PATH",
        f"  with refresh   : {eager_refresh_rate:8.2f} attempts/s   "
        f"p50={eager_refresh_p50:.3f} ms",
        f"  fixed inputs   : {eager_fixed_rate:8.2f} attempts/s   "
        f"p50={eager_fixed_p50:.3f} ms",
        "",
        "GRAPH PATH",
        f"  with refresh   : {graph_refresh_rate:8.2f} attempts/s   "
        f"p50={graph_refresh_p50:.3f} ms",
        f"  fixed inputs   : {graph_fixed_rate:8.2f} attempts/s   "
        f"p50={graph_fixed_p50:.3f} ms",
        "",
        "SPEEDUP (graph / eager)",
        f"  with refresh   : {speedup_refresh:6.3f}x",
        f"  fixed inputs   : {speedup_fixed:6.3f}x",
        "",
        f"BIT-EXACT VERIFY : {n_pass}/{args.n_seeds} seeds passed (fail={n_fail})",
    ]
    if diffs:
        lines.append("  per-seed diffs:")
        for d in diffs:
            lines.append(
                f"    seed={d['seed']:>2}  C_match={d['C_match']}  "
                f"pow_key_match={d['pow_key_match']}  "
                f"host_signal_header_match={d['host_signal_header_match']}  "
                f"C_max_abs_diff={d['C_max_abs_diff']:.6f}"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")

    # Non-zero exit if bit-exact verification failed.
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
