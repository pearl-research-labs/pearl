# Pearl-GEMM sm_89 bench harness

ONE harness. Every other agent should use this to validate perf changes.
This replaces the scatter of `_bench_*.cu`, `_bench_*.py`, and `/tmp/perf_*.py`
files with slightly different shapes, PoW configs, and measurement methodologies
that have plagued past investigations (see `project_pearl_perf_postmortem_2026_05_18`).

There are two entry points; **prefer the Python one**:

* `tools/bench_pearl_gemm.py` — pybind path, what production actually exercises.
* `csrc/gemm/_bench_unified.cu` — C++ standalone, for cases where pybind
  overhead matters or you want to bench without a torch install.

Both write CSV-compatible output. Both produce the same TOPS numbers (within
measurement noise) at the same shapes and configs.

## Quick start (Python)

On any sm_89 box with `pearl_gemm_cuda` importable (e.g. CPU01/CPU02
in the `pearl-ab` container at `/opt/pearl-venv/bin/python`):

```bash
# Baseline: R=64 production config, hard PoW target, single stream.
# Expected: ~14.6 main_TOPS at 2048^3 (per postmortem).
python tools/bench_pearl_gemm.py \
    --shapes 2048 --configs r64-hard-1s --repeats 30

# Standard sweep — what to run for any kernel/driver change.
python tools/bench_pearl_gemm.py \
    --shapes 1024,2048,4096,8192,16384,4096x4096x8192,16384x4096x4096 \
    --configs r64-hard-1s,r64-disabled-1s \
    --repeats 30 --warmup 5

# Stream sweep for multi-stream chain overlap experiments.
python tools/bench_pearl_gemm.py \
    --shapes 2048,4096 \
    --configs r64-hard-1s,r64-hard-2s,r64-hard-4s \
    --repeats 30

# R=128 (the configuration alphapool currently sends).
python tools/bench_pearl_gemm.py \
    --shapes 2048,4096 --configs r128-hard-1s --repeats 30
```

CSV lands at `C:/Source/pearl-investigation/bench_<ISO-date>.csv` by default
(or `/tmp/bench_<date>.csv` if that directory doesn't exist on the host).
Override with `--out /path/to/file.csv`.

## Config grammar

Configs are `<rank>-<pow>-<streams>[-flag...]`:

| field     | values                       | notes                                              |
|-----------|------------------------------|----------------------------------------------------|
| rank      | `r64`, `r128`                | must match a tile spec the `.so` was built with    |
| pow       | `hard`, `disabled`           | NEVER `easy` — see "Why no easy target" below      |
| streams   | `1s`..`16s`                  | driver-side N CUDA streams running the chain       |
| flag      | `graph`                      | wrap the chain in a captured CUDA graph            |
| flag      | `persistent`                 | placeholder — kernel feature not yet shipped       |

Examples:

* `r64-hard-1s`       — production config, single stream.
* `r64-disabled-1s`   — same minus PoW accumulator (skip_reduction=true).
* `r64-hard-4s`       — driver-side 4-stream chain overlap.
* `r64-hard-4s-graph` — 4 streams + CUDA graph capture per stream.

## Shape grammar

Comma-separated shapes. `N` (single integer) means `NxNxN`; `MxNxK` is explicit.

Defaults: `1024,2048,4096,8192,16384,4096x4096x8192,16384x4096x4096`.

These match the production sweep — `16384x4096x4096` is the skinny case where
the L2-aware persistent-swizzle scheduler showed a 6.77× gain (see
`project_pearl_gemm_tier1a_l2_swizzle_2026_05_17`).

## What gets measured

Each (shape, config) pair runs `--warmup` chain iterations, then `--repeats`
timed iterations recorded via paired CUDA events. We report:

* `median_ms`, `p99_ms`, `min_ms`, `mean_ms` over the timed iterations
* `main_tops`  = `2*M*N*K * streams / median_seconds * 1e-12`  (the alpha-miner
                 comparable number)
* `full_tops`  = `(2*K*R*(M+N) + M*N*(K+2*R)) * streams / median_seconds * 1e-12`
                 (all MAC work in noisingA + noisingB + main GEMM)
* `attempts_per_s` = `streams / median_seconds`  (nonces per wall second)

Use **median** for headline numbers, **p99** to spot tail latency that would
hurt real mining (we lose shares on slow attempts).

## Why no easy target

`pow_target = 0xFFFFFFFF...` triggers `check_pow_target → write_host_signal_header`
on EVERY thread (`pow_utils.hpp:271`). That code path spins on a single
`atomicCAS(&host_signal_sync->global_lock, 0, 1)` — with 7680 threads contending
on one atomic, it serializes the entire kernel and reports nonsense TOPS
(0.04 vs 14.6, see `project_pearl_perf_postmortem_2026_05_18`).

The right way to bench is `--pow hard` (impossible target → no thread enters
the spin → kernel runs the actual hot path). The pool's real target is so far
below this that empirical pool-runtime PoW load is ≤1% — there's no "realistic
target between hard and easy" worth benching.

## Baseline (what you should recover)

The headline number for the current `.so` (post-Tier-1a swizzle, May 18 build)
on a 4070 Ti SUPER, with `--shapes 2048 --configs r64-hard-1s --repeats 100 --warmup 20`:

| metric        | expected | tolerance | source |
|---------------|---------:|:---------:|--------|
| main_tops     |   ~95    | ±10 TOPS  | tools/bench_pearl_gemm.py on CPU02 2026-05-18 |
| median_ms     |   ~0.18  | ±0.05 ms  | tools/bench_pearl_gemm.py on CPU02 2026-05-18 |

For the `--configs r64-disabled-1s` variant (PoW off):
* main_tops ≈ 105 (within 12% of PoW-on — accumulator is nearly free in mainloop).

> The earlier `project_pearl_perf_postmortem_2026_05_18` note reported 14.6 TOPS
> at 2048³ with the same config. That number predates the Tier-1a L2-aware swizzle
> ship (`project_pearl_gemm_tier1a_l2_swizzle_2026_05_17`) which brought 2048³
> to 107 TOPS. The harness recovers ~95-105 TOPS on the current `.so`, consistent
> with the post-Tier-1a measurement. If you see <50 TOPS at 2048³, your `.so` is
> pre-Tier-1a (or built without `PEARL_GEMM_TARGET_ARCH=89`).

## C++ standalone

For cases where you don't want torch/pybind in the loop:

```bash
# Build (run from a WSL CUDA 12.x environment or directly on a Linux rig).
bash csrc/gemm/_build_tier1a.sh   # also builds _bench_unified target if patched
# Or build it directly:
nvcc -gencode arch=compute_89,code=sm_89 -std=c++20 -O3 \
  -I csrc/gemm -I csrc -I third_party/cutlass/include \
  -I third_party/cutlass/tools/util/include \
  -I third_party/cutlass/examples/common \
  --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG \
  csrc/gemm/_bench_unified.cu \
  csrc/gemm/pearl_noisingA_sm89_inst.cu \
  csrc/gemm/pearl_noisingB_sm89_inst.cu \
  csrc/gemm/pearl_gemm_sm89_denoise_inst.cu \
  csrc/gemm/pearl_gemm_sm89_pow_inst.cu \
  -o build-sm89/_bench_unified

./build-sm89/_bench_unified --rank 64 --pow hard --streams 1 \
    --shapes 1024,2048,4096 --repeats 30 --out /tmp/bench_unified.csv
```

The C++ bench currently only wires R=64 chains; use the Python harness for
R=128. (R=128 .cu wiring is straightforward — see
`pearl_gemm_sm89_pow_inst.cu:157` for the C symbol — but isn't compiled into
`_bench_unified` by default to keep the link line short.)

## How to extend

When you add a new kernel variant (different tile, swizzle, scheduler, etc):

1. **Don't write a new `_bench_*.py` or `_bench_*.cu`.** Add it as a config
   here. Bench harness is the API; don't fork it.
2. If the new variant requires different kernel-traits parameters, add an
   entry to the `TILES` dict in `bench_pearl_gemm.py` keyed on a new
   pseudo-rank (e.g. `r64plus`) and parse it from the config name.
3. Run the standard sweep:
   ```
   python tools/bench_pearl_gemm.py --configs r64-hard-1s,<new-variant>
   ```
4. Diff the two CSVs; commit only if the new variant is non-regressing across
   the standard sweep.

## Failure modes

* `unsupported`: config asked for something the .so/binary doesn't support
  (e.g. R=128 in the C++ standalone, or `persistent` which isn't yet
  implemented). Status is recorded; numbers are `n/a`.
* `oom`: tensor allocation at the requested (M,N,K,R,streams) didn't fit in
  GPU memory. Drop `--streams` or skip the shape.
* Any CUDA error: bench aborts with `CUDA <file>:<line> <msg>` to stderr.
* `[bench_lock] DENIED`: another agent already holds the per-rig
  bench-exclusive lock at `/var/lock/pearl-bench-exclusive`. See
  `C:/Source/pearl-investigation/BENCH_LOCK_README.md` for the full
  convention. To override (rarely correct):
  `--force-bench-lock` steals the metadata claim (but can't override a
  same-rig peer python process holding an active `fcntl.flock`);
  `--no-lock` bypasses the mutex entirely (debugging only).

## Coordinating with other agents

This harness now auto-acquires `/var/lock/pearl-bench-exclusive` before any
CUDA work. Pair it with `pearl-bench-acquire.sh` from the controller:

```bash
# Controller side
python3 C:/Source/mfarm/scripts/pearl_bench_lock.py acquire CPU01 \
    --purpose "wave5-r64-sweep" --duration-min 10 --caller "agent-N@me"
# ... ssh CPU01 'python3 tools/bench_pearl_gemm.py --lock-caller "agent-N@me" ...'
python3 C:/Source/mfarm/scripts/pearl_bench_lock.py release CPU01 --caller "agent-N@me"
```

If you skip the controller-side acquire, the harness's preamble still
refuses to start if another agent is already holding the lock — so even
"forgetful" agents can't accidentally co-bench. See
`C:/Source/pearl-investigation/BENCH_LOCK_README.md` for the rationale
(Wave-3 / Wave-4 contention disasters).
