"""Unified Python bench harness for the sm_89 Pearl-GEMM pybind path.

ONE script that every other agent should use to validate perf changes. Goal: end
the scatter of `_bench_*.cu`, `_bench_*.py`, and `/tmp/perf_*.py` files with
slightly different shapes / PoW configs / measurement methodology that have
plagued past investigations.

Sweeps a fixed list of shapes (M, N, K) crossed with a fixed list of configs,
records median + p99 wall-clock, derives MAIN-GEMM TOPS and attempts/s, writes
a CSV under C:/Source/pearl-investigation/ (or `--out`).

Configs are expressed as a comma-separated list, e.g.

    python tools/bench_pearl_gemm.py \\
        --shapes 1024,2048,4096 \\
        --configs r64-hard-1s,r64-disabled-1s,r64-hard-4s \\
        --repeats 50 --warmup 5

Each config name is parsed as `<rank>-<pow>-<streams>[-flags]`:
  rank   = r64 | r128
  pow    = hard      (impossible target, no atomic contention — the right
                     way to measure perf; see project_pearl_perf_postmortem)
           disabled  (skip_reduction=True, no PoW accumulator at all)
  streams = Ns (N CUDA streams launching the chain concurrently)
  flags   = graph    (capture the chain into a CUDA graph and replay)
            persistent  (single-CTA-over-256-nonces mode — NOT YET IMPLEMENTED
                     in the kernel; the flag is a placeholder that will cause
                     the harness to skip that config and record `unsupported`).

The "easy target" (0xFFFFFFFF — every hash trivially passes) is INTENTIONALLY
NOT exposed: it triggers 7680-way atomic serialization in write_host_signal_header
and produces nonsense TOPS numbers (see postmortem 2026-05-18). Don't use it.

Baseline validation: running

    python tools/bench_pearl_gemm.py --shapes 2048 --configs r64-hard-1s --repeats 30

should recover ~14.6 main_TOPS at 2048^3 per the postmortem note.
"""
from __future__ import annotations

import argparse
import atexit
import csv
import datetime
import math
import os
import re
import signal
import socket
import statistics
import sys
import time
from dataclasses import dataclass

import torch

import pearl_gemm_cuda as pg  # noqa: F401 — registers torch.ops.pearl_gemm.*
from pearl_gemm import noisy_gemm  # high-level signature

# --------------------------------------------------------------------------
# Bench-exclusive lock (see C:/Source/pearl-investigation/BENCH_LOCK_README.md)
#
# Wave-3 + Wave-4 live-A/B aborted because parallel agents ran concurrent
# benches on the same rig, producing nonsense numbers (the wave-4 l2-cliff
# "9x gap" was contention noise). This preamble refuses to start if another
# agent holds the lock, and auto-acquires for the bench duration.
# --------------------------------------------------------------------------
LOCK_PATH = "/var/lock/pearl-bench-exclusive"
LOCK_HEARTBEAT_AGE_MAX_S = 900  # treat as stale if heartbeat > 15 min old
_LOCK_FD: int | None = None  # held open for process lifetime; None = not held


def _parse_lock_metadata(path: str) -> dict[str, str]:
    """Read `path` and parse KV pairs (one per line OR whitespace-separated)."""
    md: dict[str, str] = {}
    try:
        with open(path, "r") as f:
            content = f.read()
    except OSError:
        return md
    for line in content.splitlines():
        line = line.rstrip()
        if not line or "=" not in line:
            continue
        first_eq = line.index("=")
        key = line[:first_eq]
        rest = line[first_eq + 1:]
        # If rest looks like it contains additional `key=` patterns, treat as
        # tokenized (legacy single-line format).
        if re.search(r"\s[a-zA-Z_][a-zA-Z0-9_]*=", rest):
            for tok in line.split():
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    if k and k not in md:
                        md[k] = v
        else:
            if key and key not in md:
                md[key] = rest
    return md


def _is_pid_alive(pid_str: str) -> bool:
    try:
        pid = int(pid_str)
    except (ValueError, TypeError):
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        # PermissionError actually means pid exists but we can't signal it
        return isinstance(sys.exc_info()[1], PermissionError)
    except OSError:
        return False


def _heartbeat_age_secs(md: dict[str, str]) -> float:
    """Seconds since heartbeat (or ts if no heartbeat). Infinity on parse error."""
    ref = md.get("heartbeat") or md.get("ts") or ""
    if not ref:
        return float("inf")
    # Strip Z / +00:00 suffix and parse
    try:
        ref_clean = ref.replace("Z", "+00:00")
        # Python 3.11+ handles +00:00 natively; for older, fall back
        dt = datetime.datetime.fromisoformat(ref_clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        return (now - dt).total_seconds()
    except (ValueError, TypeError):
        return float("inf")


def _write_lock_metadata(fd: int, *, purpose: str, caller: str) -> None:
    """Truncate the lock file and write our metadata. Caller already holds flock."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    hostname = socket.gethostname() or "host"
    payload = (
        f"agent_pid={os.getpid()}\n"
        f"host={hostname}\n"
        f"ts={ts}\n"
        f"heartbeat={ts}\n"
        f"purpose={purpose}\n"
        f"duration_min=?\n"
        f"caller={caller}\n"
    )
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, payload.encode("utf-8"))


def _release_lock() -> None:
    """Best-effort cleanup. Closing the FD releases the kernel flock.

    IMPORTANT: we deliberately do NOT unlink the lock file. Unlinking creates
    a race where a new acquirer can open() a fresh inode while the previous
    process still holds flock on the unlinked-but-FD-pinned inode — the two
    flocks then refer to different inodes and don't serialize. Instead, we
    truncate the file to empty (so other agents see "no metadata" and know
    the lock is releasable) and close the FD to release the kernel flock.
    """
    global _LOCK_FD
    if _LOCK_FD is None:
        return
    try:
        try:
            os.lseek(_LOCK_FD, 0, os.SEEK_SET)
            os.ftruncate(_LOCK_FD, 0)
        except OSError:
            pass
        os.close(_LOCK_FD)
    except OSError:
        pass
    finally:
        _LOCK_FD = None


def _signal_handler(signum: int, _frame) -> None:  # noqa: ANN001
    _release_lock()
    # Re-raise as default action so we exit with the right code.
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def acquire_bench_lock(purpose: str, caller: str, *, force: bool = False,
                       skip: bool = False) -> None:
    """Acquire the per-rig bench-exclusive lock or exit(2) with a diagnostic.

    Behavior:
      - If lock file is missing/empty/stale (heartbeat > 15 min): we take it.
      - If lock file is held by a live process AND heartbeat is fresh: we
        check fcntl-flock state. If kernel-flocked by another process,
        refuse. If only the metadata is fresh but no flock, the holder is
        almost certainly the shell-script form (pearl-bench-acquire.sh) —
        refuse unless --force-bench-lock.
      - On acquire: hold the FD open + register an atexit/signal handler
        so the kernel releases on process exit even on crash.

    Args:
      purpose: free-form description (e.g. "wave5-r64-sweep-50r")
      caller: agent identifier (e.g. "agent-42@laptop")
      force:  override holder check (logs a STOLEN_FROM line)
      skip:   bypass locking entirely (--no-lock flag; debugging only)
    """
    global _LOCK_FD
    if skip:
        print("[bench_lock] WARNING: --no-lock specified, skipping mutex; "
              "concurrent benches may produce noise.", file=sys.stderr)
        return

    # The lock-file path is /var/lock/pearl-bench-exclusive — Linux convention.
    # On non-Linux dev machines, fall back to a no-op with a loud warning.
    if not sys.platform.startswith("linux") or not os.path.isdir("/var/lock"):
        print(f"[bench_lock] WARNING: {sys.platform} / no /var/lock; "
              "skipping lock. This is fine on dev laptops but if you're on "
              "an actual bench rig, something is wrong.", file=sys.stderr)
        return

    import fcntl  # linux-only

    # Step 1: probe existing lock metadata (without holding the flock yet).
    if os.path.exists(LOCK_PATH):
        md = _parse_lock_metadata(LOCK_PATH)
        if md:
            age = _heartbeat_age_secs(md)
            pid_alive = _is_pid_alive(md.get("agent_pid", ""))
            if age < LOCK_HEARTBEAT_AGE_MAX_S and not force:
                holder = md.get("caller", "?")
                hpid = md.get("agent_pid", "?")
                hpurp = md.get("purpose", "?")
                print(f"[bench_lock] DENIED — lock held by another agent on this rig.",
                      file=sys.stderr)
                print(f"[bench_lock]   holder caller : {holder}", file=sys.stderr)
                print(f"[bench_lock]   holder pid    : {hpid} (alive={pid_alive})", file=sys.stderr)
                print(f"[bench_lock]   holder purpose: {hpurp}", file=sys.stderr)
                print(f"[bench_lock]   heartbeat age : {age:.0f}s "
                      f"(stale at {LOCK_HEARTBEAT_AGE_MAX_S}s)", file=sys.stderr)
                print(f"[bench_lock] To override (NOT recommended): rerun with --force-bench-lock",
                      file=sys.stderr)
                print(f"[bench_lock] To inspect from the controller: "
                      f"python C:/Source/mfarm/scripts/pearl_bench_lock.py status",
                      file=sys.stderr)
                sys.exit(2)
            elif age >= LOCK_HEARTBEAT_AGE_MAX_S:
                print(f"[bench_lock] reclaiming stale lock (heartbeat {age:.0f}s old, "
                      f"prev_caller={md.get('caller', '?')})", file=sys.stderr)
            elif force:
                print(f"[bench_lock] WARNING: --force-bench-lock specified, stealing from "
                      f"caller={md.get('caller', '?')} pid={md.get('agent_pid', '?')} "
                      f"purpose={md.get('purpose', '?')}", file=sys.stderr)

    # Step 2: open + flock. O_CREAT so it works even if file is absent.
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
    except OSError as e:
        print(f"[bench_lock] ERROR opening {LOCK_PATH}: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Another process holds the kernel flock — definitely concurrent bench.
        md = _parse_lock_metadata(LOCK_PATH)
        os.close(fd)
        print(f"[bench_lock] DENIED — kernel flock held by another process on this rig.",
              file=sys.stderr)
        if md:
            print(f"[bench_lock]   holder metadata: {md}", file=sys.stderr)
        sys.exit(2)
    except OSError as e:
        os.close(fd)
        print(f"[bench_lock] ERROR flocking {LOCK_PATH}: {e}", file=sys.stderr)
        sys.exit(2)

    # Step 3: write our metadata, register cleanup.
    _write_lock_metadata(fd, purpose=purpose, caller=caller)
    _LOCK_FD = fd
    atexit.register(_release_lock)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            pass  # signal not available on this platform / inside thread
    print(f"[bench_lock] ACQUIRED on this rig (purpose={purpose} caller={caller} "
          f"pid={os.getpid()})", file=sys.stderr)

# --------------------------------------------------------------------------
# Config + shape parsing
# --------------------------------------------------------------------------

DEFAULT_SHAPES = ["1024", "2048", "4096", "8192", "16384", "4096x4096x8192", "16384x4096x4096"]
DEFAULT_CONFIGS = ["r64-hard-1s", "r64-disabled-1s"]


def parse_shape(spec: str) -> tuple[int, int, int]:
    """`1024` -> (1024,1024,1024).  `4096x4096x8192` -> (4096, 4096, 8192)."""
    if "x" in spec:
        parts = spec.split("x")
        if len(parts) != 3:
            raise ValueError(f"shape must be 'N' or 'MxNxK', got {spec!r}")
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    n = int(spec)
    return n, n, n


@dataclass(frozen=True)
class Config:
    rank: int
    pow_mode: str          # 'hard' or 'disabled'
    streams: int
    use_graph: bool
    persistent: bool
    raw: str

    @property
    def name(self) -> str:
        return self.raw


def parse_config(spec: str) -> Config:
    parts = spec.split("-")
    if len(parts) < 3:
        raise ValueError(
            f"config name must be '<rank>-<pow>-<streams>[-flag...]', got {spec!r}"
        )
    rank_s, pow_s, streams_s, *flags = parts

    if rank_s not in ("r64", "r128"):
        raise ValueError(f"rank must be r64 or r128, got {rank_s!r}")
    rank = 64 if rank_s == "r64" else 128

    if pow_s not in ("hard", "disabled"):
        raise ValueError(
            f"pow must be 'hard' or 'disabled' (NOT 'easy' — see postmortem), "
            f"got {pow_s!r}"
        )

    if not streams_s.endswith("s"):
        raise ValueError(f"streams part must end with 's' (e.g. 1s, 4s), got {streams_s!r}")
    streams = int(streams_s[:-1])
    if streams < 1 or streams > 16:
        raise ValueError(f"streams must be 1..16, got {streams}")

    use_graph = "graph" in flags
    persistent = "persistent" in flags
    unknown = set(flags) - {"graph", "persistent"}
    if unknown:
        raise ValueError(f"unknown config flags: {unknown}")
    return Config(
        rank=rank,
        pow_mode=pow_s,
        streams=streams,
        use_graph=use_graph,
        persistent=persistent,
        raw=spec,
    )


# --------------------------------------------------------------------------
# Tile config per rank — matches what the .so on disk is compiled with.
# Keep in sync with:
#   csrc/gemm/pearl_gemm_sm89_denoise_inst.cu  (R=64: 128x128x64 stages=3)
#                                              (R=128: 64x64x64   stages=2)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TileSpec:
    bM: int
    bN: int
    bK: int
    cM: int
    cN: int
    stages: int
    noise_bM: int  # noisingA tile (M, K)
    noise_bN: int  # noisingB tile (N, K)
    noise_bK: int
    noise_stages: int


TILES: dict[int, TileSpec] = {
    64:  TileSpec(bM=128, bN=128, bK=64, cM=1, cN=1, stages=3,
                  noise_bM=64, noise_bN=64, noise_bK=64, noise_stages=2),
    128: TileSpec(bM=64,  bN=64,  bK=64, cM=1, cN=1, stages=2,
                  noise_bM=64, noise_bN=64, noise_bK=64, noise_stages=2),
}


# --------------------------------------------------------------------------
# Tensor allocation — once per (M,N,K,R)
# --------------------------------------------------------------------------
class BenchTensors:
    """All buffers needed for one full noisy_gemm chain at (M, N, K, R)."""

    def __init__(self, M: int, N: int, K: int, R: int, device: torch.device):
        gen = torch.Generator(device=device).manual_seed(0)
        self.M, self.N, self.K, self.R = M, N, K, R

        self.A = torch.randint(
            -127, 127, (M, K), dtype=torch.int8, device=device, generator=gen,
        )
        self.B = torch.randint(
            -127, 127, (N, K), dtype=torch.int8, device=device, generator=gen,
        )
        self.A_scales = torch.rand(M, dtype=torch.float32, device=device, generator=gen) * 0.02 + 0.005
        self.B_scales = torch.rand(N, dtype=torch.float32, device=device, generator=gen) * 0.02 + 0.005

        # Noise residuals.
        self.EAL = torch.zeros(M, R, dtype=torch.int8, device=device)
        self.EBR = torch.zeros(N, R, dtype=torch.int8, device=device)
        self.EAL_fp16 = torch.zeros(M, R, dtype=torch.float16, device=device)
        self.EBR_fp16 = torch.zeros(N, R, dtype=torch.float16, device=device)

        # Sparse noise factors.
        self.EAR_R_major = torch.zeros(K, R, dtype=torch.int8, device=device)
        self.EBL_R_major = torch.zeros(K, R, dtype=torch.int8, device=device)
        self.EAR_K_major = torch.zeros(R, K, dtype=torch.int8, device=device)
        self.EBL_K_major = torch.zeros(R, K, dtype=torch.int8, device=device)

        # Denoise factors (fp16) and their int32 counterparts.
        self.AxEBL_fp16 = torch.zeros(M, R, dtype=torch.float16, device=device)
        self.EARxBpEB_fp16 = torch.zeros(N, R, dtype=torch.float16, device=device)
        self.AxEBL_int32 = torch.zeros(M, R, dtype=torch.int32, device=device)
        self.EARxBpEB_int32 = torch.zeros(N, R, dtype=torch.int32, device=device)

        # Noised operands and output.
        self.ApEA = torch.zeros(M, K, dtype=torch.int8, device=device)
        self.BpEB = torch.zeros(N, K, dtype=torch.int8, device=device)
        self.C = torch.zeros(M, N, dtype=torch.bfloat16, device=device)

        # PoW transcript scratch.
        hh_size = pg.get_host_signal_header_size()
        hs_size = pg.get_host_signal_sync_size()
        self.host_signal_header = torch.zeros(hh_size, dtype=torch.int8, pin_memory=True)
        self.host_signal_sync = torch.zeros(hs_size, dtype=torch.int8, device=device)

        # 'hard' (impossible) target — see postmortem.
        self.pow_target = torch.zeros(8, dtype=torch.uint32, device=device)
        self.pow_key = torch.zeros(8, dtype=torch.uint32, device=device)


# --------------------------------------------------------------------------
# One pipeline iteration
# --------------------------------------------------------------------------
def _run_chain(t: BenchTensors, tile: TileSpec, *, skip_reduction: bool) -> None:
    """One full noisy_gemm attempt: noisingA + noisingB + denoise+PoW GEMM."""
    noisy_gemm(
        A=t.A, B=t.B,
        EAL=t.EAL, EAL_fp16=t.EAL_fp16,
        EBR=t.EBR, EBR_fp16=t.EBR_fp16,
        EAR_R_major=t.EAR_R_major, EBL_R_major=t.EBL_R_major,
        EAR_K_major=t.EAR_K_major, EBL_K_major=t.EBL_K_major,
        AxEBL_fp16=t.AxEBL_fp16, EARxBpEB_fp16=t.EARxBpEB_fp16,
        ApEA=t.ApEA, BpEB=t.BpEB,
        A_scales=t.A_scales, B_scales=t.B_scales, C=t.C,
        host_signal_header_pinned=t.host_signal_header,
        host_signal_sync=t.host_signal_sync,
        pow_target=t.pow_target, pow_key=t.pow_key,
        AxEBL_int32=t.AxEBL_int32, EARxBpEB_int32=t.EARxBpEB_int32,
        tile_size_m=tile.bM, tile_size_n=tile.bN, tile_size_k=tile.bK,
        cluster_size_m=tile.cM, cluster_size_n=tile.cN,
        pipeline_stages=tile.stages,
        tile_size_m_noising_A=tile.noise_bM,
        tile_size_n_noising_B=tile.noise_bN,
        tile_size_k_noising_A=tile.noise_bK,
        tile_size_k_noising_B=tile.noise_bK,
        pipeline_stages_noising_A=tile.noise_stages,
        pipeline_stages_noising_B=tile.noise_stages,
        run_noising_A=True, run_noising_B=True,
        skip_reduction=skip_reduction,
        skip_denoising=False,
    )


# --------------------------------------------------------------------------
# Bench one (shape, config) pair
# --------------------------------------------------------------------------
@dataclass
class BenchResult:
    shape: str
    config: str
    rank: int
    pow_mode: str
    streams: int
    use_graph: bool
    iters: int
    median_ms: float
    p99_ms: float
    min_ms: float
    mean_ms: float
    attempts_per_s: float        # = streams / median_time
    main_tops: float             # 2*M*N*K / median_time * streams * 1e-12
    full_tops: float             # all MAC work / median_time * streams * 1e-12
    status: str
    note: str


def _full_macs(M: int, N: int, K: int, R: int) -> float:
    """Total MAC count for one chain iteration (noisingA + noisingB + denoise GEMM).

    See bench_sm89_noisy_gemm_e2e.cu for the derivation:
        noisingA:        2 * M * K * R   (A @ EBL + EAL @ EAR)
        noisingB:        2 * N * K * R   (symmetric)
        denoise GEMM:    M * N * K + 2 * M * N * R
        Total MACs    =  2*K*R*(M+N) + M*N*(K + 2*R)

    NOTE on units: `main_tops` uses 2*M*N*K (the FLOPS convention, alpha-miner's
    `tmac_s` headline number). `full_tops` uses the MAC count above WITHOUT
    a leading factor of 2 — this matches the units used by the existing
    `bench_sm89_noisy_gemm_e2e.cu`. Don't try to "fix" the asymmetry without
    also updating that bench, or you'll break comparison with prior CSVs.
    """
    M_, N_, K_, R_ = float(M), float(N), float(K), float(R)
    return 2.0 * K_ * R_ * (M_ + N_) + M_ * N_ * (K_ + 2.0 * R_)


def bench(shape: tuple[int, int, int], config: Config,
          warmup: int, repeats: int, device: torch.device) -> BenchResult:
    M, N, K = shape

    if config.persistent:
        return BenchResult(
            shape=f"{M}x{N}x{K}", config=config.name, rank=config.rank,
            pow_mode=config.pow_mode, streams=config.streams,
            use_graph=config.use_graph, iters=0,
            median_ms=float("nan"), p99_ms=float("nan"),
            min_ms=float("nan"), mean_ms=float("nan"),
            attempts_per_s=float("nan"), main_tops=float("nan"),
            full_tops=float("nan"), status="unsupported",
            note="persistent-CTA-over-N-nonces kernel not yet implemented",
        )

    if config.rank not in TILES:
        return BenchResult(
            shape=f"{M}x{N}x{K}", config=config.name, rank=config.rank,
            pow_mode=config.pow_mode, streams=config.streams,
            use_graph=config.use_graph, iters=0,
            median_ms=float("nan"), p99_ms=float("nan"),
            min_ms=float("nan"), mean_ms=float("nan"),
            attempts_per_s=float("nan"), main_tops=float("nan"),
            full_tops=float("nan"), status="unsupported",
            note=f"no tile spec for R={config.rank}",
        )

    tile = TILES[config.rank]
    skip_reduction = (config.pow_mode == "disabled")

    # Allocate `streams` independent tensor sets — same shape, different memory.
    # This is what alpha-miner does for its "persistent CTA over 256 nonces" —
    # but here we get the *driver-side* multi-stream version.
    try:
        tensor_sets = [BenchTensors(M, N, K, config.rank, device)
                       for _ in range(config.streams)]
    except torch.cuda.OutOfMemoryError as e:
        return BenchResult(
            shape=f"{M}x{N}x{K}", config=config.name, rank=config.rank,
            pow_mode=config.pow_mode, streams=config.streams,
            use_graph=config.use_graph, iters=0,
            median_ms=float("nan"), p99_ms=float("nan"),
            min_ms=float("nan"), mean_ms=float("nan"),
            attempts_per_s=float("nan"), main_tops=float("nan"),
            full_tops=float("nan"), status="oom",
            note=f"OOM allocating {config.streams} tensor sets at {M}x{N}x{K} R={config.rank}: {e}",
        )

    streams = [torch.cuda.Stream(device=device) for _ in range(config.streams)]

    # Warmup
    for _ in range(warmup):
        for s, ts in zip(streams, tensor_sets):
            with torch.cuda.stream(s):
                _run_chain(ts, tile, skip_reduction=skip_reduction)
        torch.cuda.synchronize(device)

    # ----------------------------------------------------------------
    # Optional CUDA graph capture
    # ----------------------------------------------------------------
    if config.use_graph:
        # We capture per-stream graphs and replay them. The pinned-memory write
        # in write_host_signal_header is graph-unsafe in general, but with our
        # 'hard' target it never fires, so capture works.
        graphs: list[torch.cuda.CUDAGraph] = []
        for s, ts in zip(streams, tensor_sets):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s):
                # extra warmup is required by CUDAGraph API
                _run_chain(ts, tile, skip_reduction=skip_reduction)
                torch.cuda.synchronize(device)
                with torch.cuda.graph(g, stream=s):
                    _run_chain(ts, tile, skip_reduction=skip_reduction)
            graphs.append(g)

        def launch_iter() -> None:
            for s, g in zip(streams, graphs):
                with torch.cuda.stream(s):
                    g.replay()
    else:
        def launch_iter() -> None:
            for s, ts in zip(streams, tensor_sets):
                with torch.cuda.stream(s):
                    _run_chain(ts, tile, skip_reduction=skip_reduction)

    # ----------------------------------------------------------------
    # Per-iteration cuda events; one event pair per iter, all on the
    # default stream (sync barriers will block until all per-stream
    # work has flushed). This gives us a clean per-iter wall time
    # in the multi-stream case.
    # ----------------------------------------------------------------
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    torch.cuda.synchronize(device)
    for i in range(repeats):
        starts[i].record()
        launch_iter()
        # Force ordering: each end is recorded after every per-stream chain
        # completes by waiting on all streams from the default stream.
        for s in streams:
            torch.cuda.current_stream(device).wait_stream(s)
        ends[i].record()
    torch.cuda.synchronize(device)

    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    median_ms = statistics.median(times_ms)
    p99_ms = sorted(times_ms)[max(0, int(0.99 * (len(times_ms) - 1)))]
    min_ms = min(times_ms)
    mean_ms = statistics.mean(times_ms)

    # Throughput: each `iter` launched config.streams chains; throughput is
    # `streams * (2*M*N*K) / iter_seconds` of main-gemm work.
    sec = median_ms / 1000.0
    attempts_per_s = config.streams / sec if sec > 0 else float("nan")
    main_macs = 2.0 * float(M) * float(N) * float(K) * float(config.streams)
    full_macs = _full_macs(M, N, K, config.rank) * float(config.streams)
    main_tops = main_macs / sec * 1e-12 if sec > 0 else float("nan")
    full_tops = full_macs / sec * 1e-12 if sec > 0 else float("nan")

    return BenchResult(
        shape=f"{M}x{N}x{K}", config=config.name, rank=config.rank,
        pow_mode=config.pow_mode, streams=config.streams,
        use_graph=config.use_graph, iters=repeats,
        median_ms=median_ms, p99_ms=p99_ms, min_ms=min_ms, mean_ms=mean_ms,
        attempts_per_s=attempts_per_s, main_tops=main_tops, full_tops=full_tops,
        status="ok", note="",
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--shapes", default=",".join(DEFAULT_SHAPES),
        help="comma-separated shapes; 'N' = NxNxN, 'MxNxK' = explicit",
    )
    p.add_argument(
        "--configs", default=",".join(DEFAULT_CONFIGS),
        help="comma-separated configs; '<rank>-<pow>-<streams>[-flag]'",
    )
    p.add_argument("--warmup", type=int, default=5,
                   help="warmup iterations (bump to 20+ for stable large-shape numbers)")
    p.add_argument("--repeats", type=int, default=30, help="measured iterations")
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument(
        "--out", default=None,
        help="output CSV path (default: C:/Source/pearl-investigation/bench_<date>.csv)",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="if --out not given, write to <out-dir>/bench_<date>.csv",
    )
    # Lock control. See BENCH_LOCK_README.md.
    p.add_argument("--lock-purpose", default=None,
                   help="bench lock purpose tag (default: auto from configs+shapes). "
                        "Use a short, descriptive value like 'wave5-r64-sweep'.")
    p.add_argument("--lock-caller", default=None,
                   help="bench lock caller id (default: $USER@$HOSTNAME). "
                        "Use the same string in pearl-bench-acquire.sh if you "
                        "pre-acquired from the controller.")
    p.add_argument("--force-bench-lock", action="store_true",
                   help="steal the bench-exclusive lock even if held by another agent "
                        "(NOT RECOMMENDED — this is what caused wave-3 / wave-4 to abort).")
    p.add_argument("--no-lock", action="store_true",
                   help="bypass the bench-exclusive lock entirely. ONLY for debugging on "
                        "an idle rig you're 100%% certain no other agent is using.")
    args = p.parse_args(argv)

    # ---- Acquire bench-exclusive lock BEFORE any CUDA work ----
    # Derive a default purpose tag from configs+shapes if user didn't supply.
    lock_purpose = args.lock_purpose or (
        f"bench_pearl_gemm:{args.configs[:30]}@{args.shapes[:30]}"
    )
    lock_caller = args.lock_caller or (
        f"{os.environ.get('USER', 'unknown')}@{socket.gethostname() or 'host'}"
    )
    acquire_bench_lock(
        purpose=lock_purpose,
        caller=lock_caller,
        force=args.force_bench_lock,
        skip=args.no_lock,
    )

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device available", file=sys.stderr)
        return 1
    device = torch.device(f"cuda:{args.device}")
    cap = torch.cuda.get_device_capability(args.device)
    name = torch.cuda.get_device_name(args.device)
    print(f"device {args.device}: {name}  sm_{cap[0]}{cap[1]}")
    if cap != (8, 9):
        print(f"  WARNING: harness is intended for sm_89, got {cap}")

    shapes = [parse_shape(s.strip()) for s in args.shapes.split(",") if s.strip()]
    configs = [parse_config(c.strip()) for c in args.configs.split(",") if c.strip()]
    print(f"  shapes: {[f'{m}x{n}x{k}' for m,n,k in shapes]}")
    print(f"  configs: {[c.name for c in configs]}")
    print(f"  warmup={args.warmup}  repeats={args.repeats}")
    print()

    results: list[BenchResult] = []

    # Pretty header
    hdr = (f"  {'shape':>16s}  {'config':>22s}  {'med ms':>9s}  "
           f"{'p99 ms':>9s}  {'main TOPS':>10s}  {'full TOPS':>10s}  "
           f"{'att/s':>8s}  status")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for shape in shapes:
        for cfg in configs:
            r = bench(shape, cfg, args.warmup, args.repeats, device)
            results.append(r)
            tops_str = (f"{r.main_tops:10.2f}" if math.isfinite(r.main_tops) else "       n/a")
            full_str = (f"{r.full_tops:10.2f}" if math.isfinite(r.full_tops) else "       n/a")
            atts_str = (f"{r.attempts_per_s:8.1f}" if math.isfinite(r.attempts_per_s) else "    n/a")
            med_str = (f"{r.median_ms:9.3f}" if math.isfinite(r.median_ms) else "      n/a")
            p99_str = (f"{r.p99_ms:9.3f}" if math.isfinite(r.p99_ms) else "      n/a")
            print(f"  {r.shape:>16s}  {r.config:>22s}  {med_str}  "
                  f"{p99_str}  {tops_str}  {full_str}  {atts_str}  {r.status}"
                  + (f"  ({r.note})" if r.note else ""))

    # CSV
    if args.out:
        out_path = args.out
    else:
        out_dir = args.out_dir or "C:/Source/pearl-investigation"
        if not os.path.exists(out_dir):
            # Local path may not exist on remote rig; fall back to /tmp.
            out_dir = "/tmp" if os.path.exists("/tmp") else "."
        out_path = os.path.join(
            out_dir,
            f"bench_{datetime.date.today().isoformat()}.csv",
        )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "shape", "config", "rank", "pow_mode", "streams", "use_graph",
            "iters", "median_ms", "p99_ms", "min_ms", "mean_ms",
            "attempts_per_s", "main_tops", "full_tops",
            "device_name", "device_cap", "ts_utc", "status", "note",
        ])
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        for r in results:
            w.writerow([
                r.shape, r.config, r.rank, r.pow_mode, r.streams,
                int(r.use_graph), r.iters,
                f"{r.median_ms:.4f}", f"{r.p99_ms:.4f}",
                f"{r.min_ms:.4f}", f"{r.mean_ms:.4f}",
                f"{r.attempts_per_s:.4f}", f"{r.main_tops:.4f}",
                f"{r.full_tops:.4f}",
                name, f"{cap[0]}.{cap[1]}", ts,
                r.status, r.note,
            ])
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
