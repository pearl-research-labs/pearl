#!/usr/bin/env python3
"""LuckyPool canary miner — single stable pool connection on THIS box, GPU
mining OFFLOADED to a rig over SSH (or run locally / on the Rust CPU reference).

Architecture (see report 07 + MEMORY luckypool-full-migration):

    THIS box (full env: miner_base + pearl_mining + numpy)
      - ONE stable LuckyPool stratum connection (luckypool_client, validated
        live: authorize-first handshake, wallet="<wallet>.<worker>",
        agent="lpminer/0.1.9-552bdfe").
      - On each mining.notify, drive a pluggable mining backend.
      - On a HIT: regenerate the seed-derived A/B operands (numpy splitmix64),
        build OpenedBlockInfo, serialize via miner_base.block_submission.
        create_proof -> PlainProof.to_base64, then pearl_mining.verify_plain_proof.
      - Only if verify is True (and --submit) do we send mining.submit.

    RIG (mini06 etc.): no torch / miner_base / pearl_mining. Just the standalone
      `pearl_miner_sm89_sm89` GPU binary at /tmp. We ssh in, feed it the stdin
      contract, and parse HIT/NOHIT. All proof serialization + verification
      happens back on THIS box.

Backends (`--backend`):
  * ssh-rig : ssh root@<rig> '<env> /tmp/pearl_miner_sm89_sm89' feeding the
              stdin contract; parse HIT/NOHIT; serialize+verify here.
  * local   : run the GPU binary on THIS box ($PEARL_MINER_SM89_BIN); same
              serialize+verify path. (No SSH; for a box with a local sm_89 GPU.)
  * rust    : pearl_mining.mine (the offline-validated CPU reference). Mines at
              the header's nbits and returns a PlainProof directly. Used by the
              OFFLINE validation (`--selftest`) with an EASY nbits override.

Safety: `--dry-run` (DEFAULT) never submits — it logs
"would submit, verify_plain_proof=<bool>". `--submit` is required to actually
send a share. verify_plain_proof=False is fail-safe: we refuse to submit.

OFFLINE self-test: `python run_canary.py --selftest` does NOT open a pool socket
or ssh anywhere. It feeds a REAL captured job header (with an easy nbits so the
CPU reference hits instantly) through the same job->mine->serialize->verify path
and asserts verify_plain_proof==True, plus exercises notify->job parsing on the
real captured notify params. This proves the runner's plumbing on a real header.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import os
import struct
import subprocess
import sys
from typing import Optional

logger = logging.getLogger("canary")

# ---------------------------------------------------------------------------
# Captured fixtures (a REAL LuckyPool job, see report 07 / live capture)
# ---------------------------------------------------------------------------

# The calibrated 52-byte MiningConfiguration the miner commits to. LuckyPool
# does NOT push this on the wire — it is fixed / derived. Parses to
# common_dim=4096, rank=256, mma=Int7xInt7ToInt32, hash_tile 8x16 (== the GPU
# contract k=4096 r=256).
DEFAULT_CONFIG_HEX = (
    "001000000001000007010103000000010f07"
    "00000000000000000000000000000000000000000000000000000000000000000000"
)

# A real captured mining.notify (header / target / height / job_id).
CAPTURED_HEADER_HEX = (
    "00004020a346c202d2a744391a8a4d646612dd203fbbf88a7660a3e0a4ff490ac41dc2de8"
    "6b170a7d49cbdf36bb27ebaca48790971963a35b94ca2a89963fee9155976f8b5361c6ad6"
    "9e0118"
)
CAPTURED_TARGET_HEX = "0000000000003fffc0000000000000000000000000000000000000000000000000"
CAPTURED_HEIGHT = 65076
CAPTURED_JOB_ID = "bfd4cb60_262144"

# Easy nbits for the offline self-test: huge target => the CPU reference hits on
# the first attempt. The header body (version/prev_block/merkle_root/timestamp,
# the first 72 bytes) is preserved; only the trailing 4-byte nbits is swapped.
SELFTEST_EASY_NBITS = 0x207FFFFF

# GPU binary stdin/env contract (full speed needs the swizzle/L2 env).
GPU_ENV = {
    # SWIZZLE=24 (no L2BLOCK) is the measured-best full-shape config: ~166 tmac_s
    # on a 4070 Ti SUPER, vs ~133 at SWIZZLE=2. Wide plain swizzle keeps the B
    # panel L2-resident at 131072^2; the L2Block scheduler HURTS the light kernel.
    "PEARL_SM89_SWIZZLE": "24",
    "PEARL_SM89_SWIZZLE_NMAJ": "1",
    "PEARL_SM89_NO_L2_POLICY": "1",
}
RIG_BIN_PATH = "/tmp/pearl_miner_sm89_sm89"

# Per-call GPU nonce window. LuckyPool rotates jobs every ~7s, so each mine call
# must return FAST: the binary returns on the first HIT, else NOHIT once the
# window is exhausted. A small window keeps the mine-loop responsive so a new
# `mining.notify` can preempt the in-flight job within ~1 window. The mine loop
# advances the nonce cursor across successive windows for the SAME job.
NONCE_WINDOW = 32

# Per-mine-attempt subprocess timeout. Sized so one short window comfortably
# completes on a ~130 TH/s rig (which clears tens of attempts/window in well
# under this) while still bounding a wedged ssh/binary.
MINE_TIMEOUT_S = 25.0

# --- land-share mode defaults --------------------------------------------
# A moderate per-mine nonce count: hits arrive ~nonce 8, so 48 nonces makes a
# HIT very likely while the binary still returns on the FIRST hit (~6s). The
# mine-timeout must cover a full 48-nonce sweep (worst case ~30s) so a no-hit
# sweep isn't cut off mid-way; default 45s.
LAND_SHARE_NONCE_COUNT = 48
LAND_SHARE_MINE_TIMEOUT_S = 45.0


# ---------------------------------------------------------------------------
# A/B regeneration (numpy splitmix64) — identical to gpu_sm89_mine in the driver
# ---------------------------------------------------------------------------


def _splitmix64_fill(n: int, seed: int):
    """Regenerate the GPU binary's seed-derived int8 operand stream.

    Bit-exact with fill_AB/splitmix64 in pearl_miner_sm89.cu: advance
    splitmix64, map r % 127 - 63 into [-63, 63]. B uses seed ^ MIX_CONST.

    VECTORIZED (numpy uint64, wraps mod 2^64): the per-element accumulator
    `s += GOLDEN` means element i (0-based) uses s = seed + (i+1)*GOLDEN, so the
    whole stream is computable without a Python loop. The pure-Python loop took
    MINUTES for the full M*K=536M operand (it hung the canary's proof build);
    this runs in ~1s.
    """
    import numpy as np

    GOLDEN = np.uint64(0x9E3779B97F4A7C15)
    C1 = np.uint64(0xBF58476D1CE4E5B9)
    C2 = np.uint64(0x94D049BB133111EB)
    seed_u = np.uint64(seed)
    out = np.empty(n, dtype=np.int8)
    # Chunked so the full M*K=536M operand fits in a rig's RAM (the one-shot
    # uint64 arrays would be ~20GB; 16M-element chunks peak ~0.5GB). Each element
    # is independent (s_i = seed + (i+1)*GOLDEN), so chunking is value-identical.
    CHUNK = 1 << 26
    with np.errstate(over="ignore"):  # uint64 overflow IS the mod-2^64 wrap
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            z = np.arange(start + 1, end + 1, dtype=np.uint64)
            z *= GOLDEN
            z += seed_u
            z = (z ^ (z >> np.uint64(30))) * C1
            z = (z ^ (z >> np.uint64(27))) * C2
            z ^= z >> np.uint64(31)
            out[start:end] = (z % np.uint64(127)).astype(np.int64) - 63
    return out


B_SEED_MIX = 0xD1B54A32D192ED03


# ---------------------------------------------------------------------------
# GPU-hit -> proof (shared by ssh-rig and local backends)
# ---------------------------------------------------------------------------


def _build_proof_from_hit(hit: dict, header_bytes: bytes, mining_config) -> bytes:
    """Turn a parsed GPU `HIT {...}` into proof.bin bytes via create_proof.

    Regenerates the FULL A / B^T operands the binary mined (so the merkle
    multiproof siblings are available), builds OpenedBlockInfo, and serializes
    with the authoritative miner_base serializer. Returns proof.bin bytes.
    """
    # Torch-free proof builder (rig-deployable: numpy + pearl_mining only).
    # Bit-exact with miner_base.create_proof; verify_plain_proof gates submit.
    from pearl_proof_numpy import OpenedBlockInfo, create_proof

    m = 131072
    n = 131072
    k = int(mining_config.common_dim)
    r = int(getattr(mining_config, "rank", 256))

    # jobmine reports the A/B operand seed as "ab_seed"; mine/serve report it as
    # "seed". Both name the SAME splitmix64 fill seed used to regenerate A/B^T.
    seed = int(hit.get("ab_seed", hit.get("seed")))
    a_rows = list(map(int, hit["a_rows"]))
    b_cols = list(map(int, hit["b_cols"]))

    A = _splitmix64_fill(m * k, seed).reshape(m, k)
    B_t = _splitmix64_fill(n * k, seed ^ B_SEED_MIX).reshape(n, k)

    opened = OpenedBlockInfo(
        A_row_indices=a_rows,
        B_column_indices=b_cols,
        A=A,
        B_t=B_t,
        commitment_hash=None,
        noise_rank=r,
    )
    # NOTE on the nonce: the GPU mutates header[72:76) per attempt to search, but
    # the proof/verify path uses the ORIGINAL job header here, byte-for-byte
    # identical to the validated `mine`-mode driver (luckypool_miner_driver.py).
    # Serve mode reuses this exact path so its proof semantics match mine-mode.
    proof = create_proof(opened, header_bytes)
    return base64.b64decode(proof.to_base64())


def _gpu_stdin(header_bytes: bytes, mining_config, target: int, nonce_start: int,
               nonce_count: int, dev: int = 0) -> bytes:
    """Build the GPU binary's stdin line per the fixed contract."""
    m = 131072
    n = 131072
    k = int(mining_config.common_dim)
    r = int(getattr(mining_config, "rank", 256))
    target_be_hex = int(target).to_bytes(32, "big").hex()
    # mode=jobmine is the POOL-VALID search: it keeps the job header UNCHANGED
    # (never mutates header[72:76)=nbits) and searches the A/B operand seed, so
    # the proof's header == the job header and the recomputed commitment matches
    # the pool. (mode=mine mutates the nonce into header[72:76), producing a
    # proof whose header != the job header -> the pool rejects with code 23.)
    line = (
        f"header={header_bytes.hex()} config={mining_config.to_bytes().hex()} "
        f"target={target_be_hex} m={m} n={n} k={k} r={r} mode=jobmine real_commit=1 "
        f"nonce_start={nonce_start} nonce_count={nonce_count} dev={dev}"
    )
    try:
        open("/tmp/last_stdin.txt", "w").write(line)
    except Exception:
        pass
    return line.encode()


def _parse_gpu_output(out: str) -> Optional[dict]:
    """Parse the GPU binary stdout. Returns the HIT dict, or None on NOHIT."""
    import json

    out = out.strip()
    if out.startswith("HIT"):
        return json.loads(out[len("HIT"):].strip())
    return None


# ---------------------------------------------------------------------------
# Serve-mode stdin contract (persistent binary; one JOB line per pool notify)
# ---------------------------------------------------------------------------

# Per-job nonce window cap (serve mode). Unused by the binary — serve mines an
# UNBOUNDED nonce stream of the current header until a newer JOB preempts it, so
# there is no per-window cursor to advance here.

SERVE_BIN_ARGS = ["mode=serve", "m=131072", "n=131072", "r=256"]


def _serve_argv_remote(bin_path: str, k: int, dev: int, config_hex: str) -> str:
    """The remote command for the persistent serve-mode binary (ssh-rig).

    serve mode still requires the 52-byte mining `config` on argv (the header
    arrives per-JOB on stdin, but config does not) — without it the binary
    exits with "bad config (need 52B hex)" before the serve loop starts.
    """
    env_prefix = " ".join(f"{kk}={vv}" for kk, vv in GPU_ENV.items())
    args = " ".join(SERVE_BIN_ARGS + [f"k={k}", f"dev={dev}", f"config={config_hex}"])
    return f"env {env_prefix} {bin_path} {args}"


def _serve_job_line(header_bytes: bytes, target: int) -> bytes:
    """One serve-mode stdin line: `JOB <header_76B_hex> <target_64hex>`.

    The target is the raw per-share wire target as big-endian hex (the binary
    re-applies the `* h*w*k` difficulty adjustment, same as mine-mode)."""
    target_be_hex = int(target).to_bytes(32, "big").hex()
    return f"JOB {header_bytes.hex()} {target_be_hex}\n".encode()


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _SubprocessMineBackend:
    """Shared base for the ssh-rig and local subprocess backends.

    Each call runs the GPU binary for ONE short nonce window via `Popen` so the
    in-flight process can be KILLED the instant a newer job arrives (see
    `cancel()`). The mine loop owns the nonce cursor — it passes an explicit
    `nonce_start`/`nonce_count` per window so we don't keep stale per-job state
    here.
    """

    def __init__(self, dev: int = 0, timeout_s: float = MINE_TIMEOUT_S):
        self.dev = dev
        self.timeout_s = timeout_s
        self._proc: Optional[subprocess.Popen] = None
        self._proc_lock = __import__("threading").Lock()
        self._cancelled = False

    # ---- cancellation ----------------------------------------------------

    def cancel(self) -> None:
        """Kill any in-flight mine subprocess (called from the notify handler
        on the event loop thread when a new job preempts the current one)."""
        with self._proc_lock:
            self._cancelled = True
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                logger.debug("backend cancel: proc.kill raised", exc_info=True)

    def reset_cancel(self) -> None:
        """Clear the cancel flag before starting a window for a fresh job."""
        with self._proc_lock:
            self._cancelled = False

    # ---- orphan cleanup --------------------------------------------------

    def kill_orphans(self) -> None:
        """Kill any stale GPU-binary process holding the device.

        Base impl is a no-op (the rust backend has no GPU process). Subprocess
        backends that spawn the sm_89 binary override this to reap orphans that
        a prior cancel()/timeout left behind — orphans hold the GPU so the next
        mine fails fast (NOHIT in <1s). Best-effort; never raises."""
        return None

    def _argv(self) -> list[str]:
        raise NotImplementedError

    def _popen_kwargs(self) -> dict:
        return {}

    def _run_window(self, header_bytes: bytes, mining_config, target: int,
                    nonce_start: int, nonce_count: int) -> Optional[bytes]:
        stdin = _gpu_stdin(header_bytes, mining_config, target, nonce_start,
                           nonce_count, dev=self.dev)
        argv = self._argv()
        proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, **self._popen_kwargs())
        with self._proc_lock:
            if self._cancelled:
                # Preempted between the cancel() call and starting this window.
                proc.kill()
                self._proc = None
                return None
            self._proc = proc
        try:
            out_b, err_b = proc.communicate(input=stdin, timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            logger.warning("mine window timed out after %.0fs", self.timeout_s)
            return None
        finally:
            with self._proc_lock:
                self._proc = None
        if self._cancelled:
            return None
        out = out_b.decode(errors="replace")
        try:
            with open("/tmp/last_out.txt", "w") as _f:
                _f.write(f"rc={proc.returncode}\n--STDOUT--\n{out}\n--STDERR--\n{err_b.decode(errors='replace')}")
        except Exception:
            pass
        if proc.returncode != 0 and not out.strip().startswith(("HIT", "NOHIT")):
            logger.warning("mine nonzero rc=%d stderr=%s",
                           proc.returncode, err_b.decode(errors="replace")[-400:])
            return None
        hit = _parse_gpu_output(out)
        if hit is None:
            logger.info("NOHIT (window exhausted)")
            return None
        return _build_proof_from_hit(hit, header_bytes, mining_config)

    def __call__(self, header_bytes: bytes, mining_config, target: int,
                 nonce_range: range, job_id: Optional[str] = None) -> Optional[bytes]:
        nonce_start = nonce_range.start
        nonce_count = len(nonce_range)
        logger.info("mine window: nonce=[%d,+%d) job=%s",
                    nonce_start, nonce_count, job_id)
        return self._run_window(header_bytes, mining_config, target,
                                nonce_start, nonce_count)


class SshRigBackend(_SubprocessMineBackend):
    """Offload GPU mining to a rig over SSH; serialize+verify on THIS box."""

    def __init__(self, rig: str, dev: int = 0, ssh_user: str = "root",
                 timeout_s: float = MINE_TIMEOUT_S, bin_path: str = RIG_BIN_PATH):
        super().__init__(dev=dev, timeout_s=timeout_s)
        self.rig = rig
        self.ssh_user = ssh_user
        self.bin_path = bin_path

    def _ssh_prefix(self) -> list[str]:
        return [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            f"{self.ssh_user}@{self.rig}",
        ]

    def _argv(self) -> list[str]:
        env_prefix = " ".join(f"{k}={v}" for k, v in GPU_ENV.items())
        remote = f"env {env_prefix} {self.bin_path}"
        return self._ssh_prefix() + [remote]

    def kill_orphans(self) -> None:
        """Reap any orphaned `pearl_miner_sm89` on the rig (best-effort).

        A cancelled/timed-out mine kills the local ssh client but leaves the
        REMOTE binary running and holding the GPU; the next mine then NOHITs in
        <1s. `pkill -9 -f pearl_miner_sm89` matches the binary regardless of the
        `env ...` prefix in its argv. rc=1 (no match) is the normal/healthy
        case, so we don't log it as an error."""
        argv = self._ssh_prefix() + ["pkill -9 -f pearl_miner_sm89; true"]
        try:
            subprocess.run(argv, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=15.0)
            logger.info("kill_orphans: pkill pearl_miner_sm89 on %s", self.rig)
        except Exception:
            logger.warning("kill_orphans: ssh pkill failed (continuing)",
                           exc_info=True)


class LocalGpuBackend(_SubprocessMineBackend):
    """Run the GPU binary on THIS box ($PEARL_MINER_SM89_BIN). Same hit->proof."""

    def __init__(self, dev: int = 0, timeout_s: float = MINE_TIMEOUT_S,
                 bin_path: Optional[str] = None):
        super().__init__(dev=dev, timeout_s=timeout_s)
        self.bin_path = bin_path or os.environ.get(
            "PEARL_MINER_SM89_BIN", "/opt/pearl/pearl_miner_sm89")

    def _argv(self) -> list[str]:
        return [self.bin_path]

    def _popen_kwargs(self) -> dict:
        env = dict(os.environ)
        env.update(GPU_ENV)
        return {"env": env}

    def kill_orphans(self) -> None:
        """Reap any orphaned local `pearl_miner_sm89` holding the GPU."""
        try:
            subprocess.run(["pkill", "-9", "-f", "pearl_miner_sm89"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=15.0)
            logger.info("kill_orphans: local pkill pearl_miner_sm89")
        except Exception:
            logger.warning("kill_orphans: local pkill failed (continuing)",
                           exc_info=True)


class RustCpuBackend:
    """The offline-validated CPU reference (pearl_mining.mine).

    Mines at the header's own nbits and returns a PlainProof's bytes. The
    `target` and `nonce_range` args are interface parity only — the Rust miner
    draws its own A/B and loops to the header difficulty.
    """

    def __init__(self, m: int = 256, n: int = 256):
        self.m = m
        self.n = n

    def __call__(self, header_bytes: bytes, mining_config, target: int,
                 nonce_range: range, job_id: Optional[str] = None) -> Optional[bytes]:
        import pearl_mining as pm

        bh = pm.IncompleteBlockHeader.from_bytes(header_bytes)
        k = int(mining_config.common_dim)
        proof = pm.mine(self.m, self.n, k, bh, mining_config, None, False)
        return base64.b64decode(proof.to_base64())


def make_backend(name: str, args: argparse.Namespace):
    if name == "ssh-rig":
        if not args.rig:
            raise SystemExit("--backend ssh-rig requires --rig <host>")
        return SshRigBackend(args.rig, dev=args.dev, ssh_user=args.ssh_user,
                             timeout_s=args.mine_timeout)
    if name == "local":
        return LocalGpuBackend(dev=args.dev, timeout_s=args.mine_timeout)
    if name == "rust":
        return RustCpuBackend()
    raise SystemExit(f"unknown backend: {name!r}")


# ---------------------------------------------------------------------------
# Share-acceptance gate
# ---------------------------------------------------------------------------


def meets_share_target(pm, bh, proof, wire_target: int) -> tuple[bool, int, int]:
    """The pool's SHARE check, replicated locally. Returns (ok, jackpot, bound).

    The pool accepts a share iff the recomputed jackpot hash (little-endian) is
    <= the SHARE bound = wire_target * h * w * dot_product_length — exactly the
    pearl-gateway `MiningJob.adjust_target` formula. `wire_target` is the share
    `target` from `mining.notify` (NOT the header's nbits).

    Gating on `verify_plain_proof` instead is WRONG: it derives its bound from
    the header's nbits (the much harder *block* target), so it rejects perfectly
    valid pool shares with the same "hash does not meet difficulty target"
    message the pool returns as stratum code 23. We recompute the jackpot via
    `dump_jackpot`, which uses the identical compute_noise -> compute_jackpot ->
    compute_jackpot_hash path as the verifier, so this match is bit-exact.
    """
    jh_le, h, w, dot, _nbits = pm.dump_jackpot(bh, proof)
    jackpot = int.from_bytes(jh_le, "little")
    bound = wire_target * h * w * dot
    if bound > (1 << 256) - 1:
        # Mirror adjust_target's "Target is too easy" clamp: a bound past 2^256
        # is rejected pool-side, so never submit against it.
        return False, jackpot, bound
    return jackpot <= bound, jackpot, bound


# ---------------------------------------------------------------------------
# Job handler (notify -> mine -> serialize -> verify -> [submit])
# ---------------------------------------------------------------------------


def handle_job(job, *, mining_config, backend, submit: bool,
               client=None, loop=None, nonce_range: Optional[range] = None,
               is_current=None) -> dict:
    """Mine ONE nonce window of `job`: returns a result dict.

    Runs a single short window through the backend, then (on HIT) verifies and
    optionally submits. Pure-sync; the driver runs it in a thread. The caller
    (the live mine loop) advances `nonce_range` across windows for the same job
    and preempts in-flight windows by calling `backend.cancel()` on a new job.

    `is_current(job_id) -> bool` is the staleness guard: it is checked AFTER the
    (potentially slow) proof build / verify and BEFORE submit, so a share whose
    job rotated mid-proof is dropped ("stale, skipping submit") instead of being
    submitted late. `client`/`loop` are only needed when submit=True.
    """
    import pearl_mining as pm

    if nonce_range is None:
        nonce_range = range(0, NONCE_WINDOW)
    if is_current is None:
        def is_current(_job_id):  # default: always current (selftest path)
            return True

    bh = pm.IncompleteBlockHeader.from_bytes(job.header_bytes)
    result = {"job_id": job.job_id, "verify": None, "submitted": False,
              "accepted": None, "error": None, "stale": False}

    try:
        proof_bytes = backend(job.header_bytes, mining_config, job.target,
                              nonce_range, job_id=job.job_id)
    except Exception as e:
        logger.exception("mining backend raised")
        result["error"] = f"backend: {e}"
        return result

    if proof_bytes is None:
        logger.info("job %s: NOHIT this window", job.job_id)
        return result

    proof = pm.PlainProof.from_base64(base64.b64encode(proof_bytes).decode())
    ok, jackpot, bound = meets_share_target(pm, bh, proof, job.target)
    result["verify"] = bool(ok)
    logger.info("job %s: meets_share_target=%s jackpot=2^%d share_bound=2^%d",
                job.job_id, ok, jackpot.bit_length(), bound.bit_length())

    if not ok:
        # Fail-safe: only submit shares that actually meet the SHARE target
        # (wire_target * h*w*k). A miss here means the backend returned a
        # non-winning candidate (or a stale/mismatched mining_config) — drop it
        # BEFORE we burn a submit slot.
        logger.error("job %s: NOT submitting (jackpot exceeds share target)",
                     job.job_id)
        return result

    # Staleness guard: a HIT for a job that has since rotated is worthless — the
    # pool rejects a stale job_id. Drop it rather than submit late.
    if not is_current(job.job_id):
        logger.warning("job %s: HIT but job rotated during proof-build — "
                       "stale, skipping submit", job.job_id)
        result["stale"] = True
        return result

    b64 = base64.b64encode(proof_bytes).decode()
    if not submit:
        logger.info("job %s: would submit, meets_share_target=%s (proof_b64_len=%d)",
                    job.job_id, ok, len(b64))
        return result

    if client is None or loop is None:
        logger.error("submit requested but no client/loop bound")
        result["error"] = "no client"
        return result

    fut = asyncio.run_coroutine_threadsafe(
        client.submit_share(job.job_id, b64, hashrate=0.0), loop)
    res = fut.result()
    result["submitted"] = True
    result["accepted"] = res.accepted
    if res.accepted:
        logger.info("job %s: SHARE ACCEPTED latency=%.1fms", job.job_id, res.latency_ms)
    else:
        logger.warning("job %s: SHARE REJECTED code=%s err=%s",
                       job.job_id, res.error_code, res.error)
    return result


# ---------------------------------------------------------------------------
# Live mine loop (preemptive: current-job mining with new-job preemption)
# ---------------------------------------------------------------------------


class CanaryMineLoop:
    """Drives the GPU backend against the CURRENT LuckyPool job, preemptively.

    A single background asyncio task runs short nonce windows against whatever
    `self._job` currently is. The stratum `on_new_job` callback (event-loop
    thread) just sets the new job, bumps `self._gen`, and `cancel()`s the
    in-flight backend subprocess so the GPU drops the stale job and restarts on
    the fresh one within ~one window. Results from an OLD generation are
    discarded; a HIT is only submitted if its job is STILL current.

    Lifecycle per window:
      1. snapshot (job, gen) under the lock
      2. run ONE short window in a worker thread (handle_job)
      3. if gen changed mid-window -> drop the result (stale)
      4. else: HIT -> verify -> (is_current guard) -> submit; NOHIT -> advance
         the nonce cursor and loop.
    """

    def __init__(self, *, mining_config, backend, submit: bool, client, loop,
                 window: int = NONCE_WINDOW):
        self.mining_config = mining_config
        self.backend = backend
        self.submit = submit
        self.client = client
        self.loop = loop
        self.window = window

        self._job = None
        self._gen = 0
        self._cursor = 0
        self._job_ready = asyncio.Event()
        self._stop = asyncio.Event()

    # ---- called from the stratum read loop (event-loop thread) ----------

    def on_new_job(self, job) -> None:
        logger.info("NEW JOB job_id=%s height=%s target=%#x header=%s... (gen->%d)",
                    job.job_id, job.height, job.target,
                    job.header_bytes.hex()[:32], self._gen + 1)
        self._job = job
        self._gen += 1
        self._cursor = 0
        # Drop the GPU's in-flight stale window so it restarts on the fresh job.
        cancel = getattr(self.backend, "cancel", None)
        if cancel is not None:
            cancel()
        self._job_ready.set()

    def current_gen(self) -> int:
        return self._gen

    def is_current(self, gen: int):
        """Return a predicate(job_id)->bool that is True iff `gen` is still the
        live generation (job_id is accepted for log symmetry)."""
        return lambda _job_id: self._gen == gen

    # ---- the background mine loop ---------------------------------------

    async def run(self) -> None:
        while not self._stop.is_set():
            if self._job is None:
                # Wait for the first job (or stop).
                await self._wait_job_or_stop()
                continue

            job = self._job
            gen = self._gen
            nonce_range = range(self._cursor, self._cursor + self.window)

            reset = getattr(self.backend, "reset_cancel", None)
            if reset is not None:
                reset()

            res = await asyncio.to_thread(
                handle_job, job, mining_config=self.mining_config,
                backend=self.backend, submit=self.submit,
                client=self.client, loop=self.loop,
                nonce_range=nonce_range, is_current=self.is_current(gen))

            if gen != self._gen:
                # A new job arrived during this window; the backend was
                # cancelled and on_new_job already reset self._cursor to 0 for
                # the new job. Drop whatever came back and re-loop on it.
                logger.info("window for job %s gen=%d superseded by gen=%d — dropped",
                            job.job_id, gen, self._gen)
                continue

            # Still the current job (HIT submitted/verified, NOHIT, or a stale
            # drop that raced just under the gen check): advance the nonce
            # cursor and keep mining successive windows of the SAME job until it
            # rotates.
            self._cursor += self.window

    async def _wait_job_or_stop(self) -> None:
        stop_task = asyncio.ensure_future(self._stop.wait())
        job_task = asyncio.ensure_future(self._job_ready.wait())
        try:
            await asyncio.wait({stop_task, job_task},
                               return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (stop_task, job_task):
                if not t.done():
                    t.cancel()
        self._job_ready.clear()

    def stop(self) -> None:
        self._stop.set()
        self._job_ready.set()
        cancel = getattr(self.backend, "cancel", None)
        if cancel is not None:
            cancel()


# ---------------------------------------------------------------------------
# Land-share mode (orphan-free serial mining; exploits pool grace)
# ---------------------------------------------------------------------------


def land_share_mine(job, *, mining_config, backend, submit, client, loop,
                    nonce_range, current_job_id=None):
    """Mine ONE one-shot window of `job`, then submit a verified HIT EVEN IF the
    job has since rotated (exploit pool grace). Returns a result dict.

    This is the orphan-free, grace-exploiting twin of `handle_job`: there is NO
    staleness guard — a verified proof for an already-rotated job is still
    submitted, because the whole point of this mode is to discover whether the
    pool grants grace for a recently-rotated job_id. `current_job_id` is the
    live job at submit time and is logged only so we can see HOW stale the mined
    job was. Pure-sync; the loop runs it in a worker thread.
    """
    import pearl_mining as pm

    bh = pm.IncompleteBlockHeader.from_bytes(job.header_bytes)
    result = {"job_id": job.job_id, "verify": None, "submitted": False,
              "accepted": None, "error": None, "stale": False,
              "nonce_start": nonce_range.start}

    try:
        proof_bytes = backend(job.header_bytes, mining_config, job.target,
                              nonce_range, job_id=job.job_id)
    except Exception as e:
        logger.exception("mining backend raised")
        result["error"] = f"backend: {e}"
        return result

    if proof_bytes is None:
        logger.info("job %s: NOHIT this window (nonce_start=%d)",
                    job.job_id, nonce_range.start)
        return result

    proof = pm.PlainProof.from_base64(base64.b64encode(proof_bytes).decode())
    ok, jackpot, bound = meets_share_target(pm, bh, proof, job.target)
    result["verify"] = bool(ok)
    logger.info("job %s: HIT -> meets_share_target=%s jackpot=2^%d share_bound=2^%d",
                job.job_id, ok, jackpot.bit_length(), bound.bit_length())

    if not ok:
        # The SHARE gate is wire_target * h*w*k (pool's adjust_target), NOT the
        # header-nbits block bound — gating on the latter (the old code-23 bug)
        # rejected valid shares. A miss here means a genuine non-winning
        # candidate; do NOT submit (avoid pool abuse flags).
        logger.error("job %s: NOT submitting (jackpot exceeds share target)",
                     job.job_id)
        return result

    # How stale is this HIT? Log mined job_id vs the current live job_id.
    stale = current_job_id is not None and current_job_id != job.job_id
    result["stale"] = stale
    if stale:
        logger.warning("job %s: job rotated to %s during mine — submitting ANYWAY "
                       "(land-share: exploit pool grace)", job.job_id, current_job_id)
    else:
        logger.info("job %s: still the current job at submit time", job.job_id)

    b64 = base64.b64encode(proof_bytes).decode()
    if not submit:
        logger.info("job %s: DRY-RUN would submit, meets_share_target=%s stale=%s "
                    "(proof_b64_len=%d)", job.job_id, ok, stale, len(b64))
        return result

    if client is None or loop is None:
        logger.error("submit requested but no client/loop bound")
        result["error"] = "no client"
        return result

    fut = asyncio.run_coroutine_threadsafe(
        client.submit_share(job.job_id, b64, hashrate=0.0), loop)
    res = fut.result()
    result["submitted"] = True
    result["accepted"] = res.accepted
    if res.accepted:
        logger.info("job %s: SHARE ACCEPTED latency=%.1fms (stale=%s)",
                    job.job_id, res.latency_ms, stale)
    else:
        # Distinguish a grace-miss (stale/job-not-found) from a real format error
        # so we learn whether grace exists at all.
        emsg = (res.error or "").lower()
        kind = ("STALE/job-not-found" if any(t in emsg for t in
                ("stale", "not found", "unknown job", "expired"))
                else "FORMAT/other")
        logger.warning("job %s: SHARE REJECTED [%s] code=%s err=%s (mined_stale=%s)",
                       job.job_id, kind, res.error_code, res.error, stale)
    return result


class LandShareLoop:
    """Serial, orphan-free mine loop that lands an ACCEPTED share.

    Distinct from CanaryMineLoop's preemptive design: there is NO preemption and
    only ONE GPU subprocess is ever in flight, so the bug that orphaned the
    remote binary (and starved subsequent windows) cannot occur.

    Per iteration:
      1. kill_orphans() on the backend (reap any stale binary holding the GPU).
      2. snapshot the CURRENT job (set by on_new_job); if none yet, wait.
      3. run ONE one-shot mine (moderate nonce_count, returns on first HIT).
         A new job arriving mid-mine does NOT cancel it — we let it finish.
      4. on a verified HIT: submit EVEN IF the job rotated (grace), then on
         SHARE ACCEPTED -> stop. On NOHIT, advance the nonce cursor for the
         same job (reset to 0 when the job changes) and loop.

    Stops on the first accepted share (sets self.accepted=True and signals the
    pool client to stop).
    """

    def __init__(self, *, mining_config, backend, submit, client, loop,
                 nonce_count, on_accepted=None):
        self.mining_config = mining_config
        self.backend = backend
        self.submit = submit
        self.client = client
        self.loop = loop
        self.nonce_count = nonce_count
        self.on_accepted = on_accepted

        self._job = None
        self._last_job_id = None
        self._cursor = 0
        self._job_ready = asyncio.Event()
        self._stop = asyncio.Event()
        self.accepted = False

    # ---- called from the stratum read loop (event-loop thread) ----------

    def on_new_job(self, job) -> None:
        # No cancel(): an in-flight one-shot mine is allowed to finish. We just
        # record the latest job; the loop picks it up on its next iteration.
        logger.info("NEW JOB job_id=%s height=%s target=%#x header=%s...",
                    job.job_id, job.height, job.target, job.header_bytes.hex()[:32])
        self._job = job
        self._job_ready.set()

    # ---- the background mine loop ---------------------------------------

    async def run(self) -> None:
        while not self._stop.is_set():
            if self._job is None:
                await self._wait_job_or_stop()
                continue

            job = self._job
            current_id = job.job_id
            if job.job_id != self._last_job_id:
                self._cursor = 0
                self._last_job_id = job.job_id

            # Reap any orphaned binary BEFORE mining so the GPU is free.
            ko = getattr(self.backend, "kill_orphans", None)
            if ko is not None:
                await asyncio.to_thread(ko)

            nonce_range = range(self._cursor, self._cursor + self.nonce_count)
            reset = getattr(self.backend, "reset_cancel", None)
            if reset is not None:
                reset()

            res = await asyncio.to_thread(
                land_share_mine, job, mining_config=self.mining_config,
                backend=self.backend, submit=self.submit, client=self.client,
                loop=self.loop, nonce_range=nonce_range,
                current_job_id=self._job.job_id)

            if res.get("accepted"):
                self.accepted = True
                logger.info("land-share: SHARE ACCEPTED on job %s — done", current_id)
                if self.on_accepted is not None:
                    self.on_accepted()
                self.stop()
                return

            # NOHIT or rejected: advance the cursor for the SAME job and retry.
            # (If the job changed meanwhile, the next iteration resets to 0.)
            self._cursor += self.nonce_count

    async def _wait_job_or_stop(self) -> None:
        stop_task = asyncio.ensure_future(self._stop.wait())
        job_task = asyncio.ensure_future(self._job_ready.wait())
        try:
            await asyncio.wait({stop_task, job_task},
                               return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (stop_task, job_task):
                if not t.done():
                    t.cancel()
        self._job_ready.clear()

    def stop(self) -> None:
        self._stop.set()
        self._job_ready.set()
        # Final orphan reap so we don't leave the GPU held on exit.
        ko = getattr(self.backend, "kill_orphans", None)
        if ko is not None:
            try:
                ko()
            except Exception:
                logger.debug("stop: kill_orphans raised", exc_info=True)


# ---------------------------------------------------------------------------
# Serve mode (DEFINITIVE non-stale miner: one persistent pool conn + one
# persistent ssh pipe to a serve-mode binary)
# ---------------------------------------------------------------------------


class ServeLoop:
    """Drive the PERSISTENT serve-mode binary over ONE persistent ssh pipe.

    Unlike the one-shot backends (which spawn a fresh ssh+binary per nonce
    window, paying CUDA init each time and mining a FIXED header that goes stale
    before it hits), this loop:

      * spawns ONE long-lived `ssh root@<rig> 'env ... <bin> mode=serve ...'`
        via a SYNCHRONOUS `subprocess.Popen` (bufsize=0, unbuffered), keeping
        stdin/stdout/stderr pipes open for the whole run;
      * on each `mining.notify`, writes a single `JOB <header> <target>` line to
        the ssh stdin and IMMEDIATELY `flush()`es it under a lock — exactly like
        the proven manual `( printf 'JOB ...'; sleep ) | ssh rig 'binary ...'`
        bash pipe. The fragile asyncio `proc.stdin.write` + fire-and-forget
        `create_task(drain)` transport NEVER flushed the JOB bytes through to
        the remote binary's non-blocking `poll` stdin, so the binary received
        nothing and emitted zero HITs. A synchronous write+flush fixes that.
      * a daemon HIT-reader THREAD does a blocking `for line in proc.stdout`,
        parses each `HIT {...}`, maps its echoed `header` back to the job_id, and
        marshals a proof-build+verify+submit back onto the event loop (where the
        async pool client lives) via `loop.call_soon_threadsafe`. Because the
        binary always mines the CURRENT header, a HIT is for a ~current job ->
        accepted (not stale).
      * a daemon stderr-reader THREAD drains the binary's stderr to the logger so
        `serve: ready` / `serve: new JOB` surface at INFO.

    Exits 0 on the first SHARE ACCEPTED. Kills the ssh proc + reaps the remote
    binary on stop / atexit.
    """

    def __init__(self, *, mining_config, submit, client, loop, rig, ssh_user,
                 bin_path, dev, on_accepted=None):
        import threading

        self.mining_config = mining_config
        self.submit = submit
        self.client = client
        self.loop = loop
        self.rig = rig
        self.ssh_user = ssh_user
        self.bin_path = bin_path
        self.dev = dev
        self.on_accepted = on_accepted

        # header_hex -> (job_id, target, received_at). The binary echoes the
        # ORIGINAL job header in each HIT, so an exact-hex lookup resolves the
        # job_id + the live target + when we received it (staleness).
        self._jobs: dict[str, tuple] = {}
        self._proc = None  # synchronous subprocess.Popen
        self.accepted = False
        self._hits_seen = 0  # HIT lines the reader thread has pulled off stdout
        self._stop = asyncio.Event()
        # Guards stdin write+flush so the (possible) HIT-thread resubmit path and
        # the notify path never interleave bytes on the pipe.
        self._stdin_lock = threading.Lock()
        self._threads: list = []

    # ---- ssh process lifecycle ------------------------------------------

    def _argv(self) -> list[str]:
        """The full argv for the persistent serve process (ssh + remote cmd).

        Overridable in tests to point at a LOCAL fake serve binary instead of
        ssh, so the real Popen+flush+pipe delivery is exercised with no rig."""
        k = int(self.mining_config.common_dim)
        remote = _serve_argv_remote(
            self.bin_path, k, self.dev, self.mining_config.to_bytes().hex()
        )
        return [
            "ssh", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ServerAliveInterval=30",
            f"{self.ssh_user}@{self.rig}", remote,
        ]

    async def start(self) -> None:
        import threading

        argv = self._argv()
        logger.info("serve: spawning persistent pipe: %s", " ".join(argv))
        # Synchronous, UNBUFFERED Popen (bufsize=0): a write+flush hits the OS
        # pipe immediately, mirroring the working bash pipe.
        self._proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0)
        t_out = threading.Thread(target=self._read_stdout, name="serve-stdout",
                                 daemon=True)
        t_err = threading.Thread(target=self._read_stderr, name="serve-stderr",
                                 daemon=True)
        self._threads = [t_out, t_err]
        t_out.start()
        t_err.start()

    # ---- called from the stratum read loop (event-loop thread) ----------

    def on_new_job(self, job) -> None:
        """Record the job and push a JOB line to the persistent binary.

        Runs on the event-loop thread. The synchronous `stdin.write + flush`
        under the lock is a quick OS write — safe to call directly here."""
        self._jobs[job.header_bytes.hex()] = (job.job_id, job.target,
                                              __import__("time").time())
        logger.info("NEW JOB job_id=%s height=%s target=%#x header=%s... -> JOB line",
                    job.job_id, job.height, job.target, job.header_bytes.hex()[:32])
        self._write_job_line(_serve_job_line(job.header_bytes, job.target))

    def _write_job_line(self, line: bytes) -> None:
        """Synchronous write+flush of one JOB line under the stdin lock.

        THE FIX: explicit `flush()` pushes the bytes to the remote binary's
        non-blocking `poll` stdin immediately (the old asyncio fire-and-forget
        drain never reliably did)."""
        proc = self._proc
        if proc is None or proc.stdin is None:
            logger.warning("serve: stdin not ready; dropping JOB push")
            return
        with self._stdin_lock:
            try:
                proc.stdin.write(line)
                proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                logger.warning("serve: stdin write/flush failed (binary gone?)",
                               exc_info=True)

    # ---- stdout HIT reader (daemon thread) ------------------------------

    def _read_stdout(self) -> None:
        """Blocking readline loop on the binary's stdout (own daemon thread).

        Marshals each HIT back to the event loop (where the async pool client
        lives) via `call_soon_threadsafe`; never touches the loop directly."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for raw in iter(proc.stdout.readline, b""):
            if self._stop.is_set() or self.accepted:
                break
            text = raw.decode(errors="replace").strip()
            if not text.startswith("HIT"):
                continue
            self._hits_seen += 1
            # Hand the HIT to the event loop: the proof-build/verify/submit path
            # uses the asyncio pool client and must run there.
            self.loop.call_soon_threadsafe(self._schedule_hit, text)
        else:
            logger.warning("serve: stdout EOF (binary exited?)")

    def _schedule_hit(self, text: str) -> None:
        """(event-loop thread) launch the async HIT handler as a task."""
        self.loop.create_task(self._handle_hit_guarded(text))

    async def _handle_hit_guarded(self, text: str) -> None:
        try:
            await self._handle_hit(text)
        except Exception:
            logger.exception("serve: error handling HIT line")

    def _read_stderr(self) -> None:
        """Drain the binary's stderr to the logger (own daemon thread) so
        `serve: ready` / `serve: new JOB` are visible at INFO."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for raw in iter(proc.stderr.readline, b""):
            if self._stop.is_set():
                break
            msg = raw.decode(errors="replace").rstrip()
            if msg:
                logger.info("serve[binary]: %s", msg)

    async def _handle_hit(self, text: str) -> None:
        import json
        import time as _time

        import pearl_mining as pm

        hit = json.loads(text[len("HIT"):].strip())
        header_hex = hit.get("header", "")
        entry = self._jobs.get(header_hex)
        if entry is None:
            logger.warning("serve: HIT for UNKNOWN header %s... (no job map) — drop",
                           header_hex[:32])
            return
        job_id, target, received_at = entry
        staleness = _time.time() - received_at
        header_bytes = bytes.fromhex(header_hex)

        # Build the proof (CPU; on THIS box) and gate on the SHARE target
        # (wire_target * h*w*k, the pool's adjust_target bound) — NOT the
        # header-nbits block bound. Fail-safe: a non-winning candidate is never
        # submitted.
        proof_bytes = await asyncio.to_thread(
            _build_proof_from_hit, hit, header_bytes, self.mining_config)
        bh = pm.IncompleteBlockHeader.from_bytes(header_bytes)
        proof = pm.PlainProof.from_base64(
            base64.b64encode(proof_bytes).decode())
        ok, jackpot, bound = await asyncio.to_thread(
            meets_share_target, pm, bh, proof, target)
        logger.info("serve: HIT job_id=%s nonce=%s staleness=%.1fs "
                    "meets_share_target=%s jackpot=2^%d share_bound=2^%d",
                    job_id, hit.get("nonce"), staleness, ok,
                    jackpot.bit_length(), bound.bit_length())
        if not ok:
            logger.error("serve: job %s NOT submitting (jackpot exceeds share "
                         "target)", job_id)
            return

        b64 = base64.b64encode(proof_bytes).decode()
        if not self.submit:
            logger.info("serve: job %s DRY-RUN would submit, verify=%s "
                        "staleness=%.1fs (proof_b64_len=%d)",
                        job_id, ok, staleness, len(b64))
            return

        res = await self.client.submit_share(job_id, b64, hashrate=0.0)
        if res.accepted:
            logger.info("serve: job %s SHARE ACCEPTED latency=%.1fms staleness=%.1fs",
                        job_id, res.latency_ms, staleness)
            self.accepted = True
            if self.on_accepted is not None:
                self.on_accepted()
            await self.stop()
        else:
            emsg = (res.error or "").lower()
            kind = ("STALE/job-not-found" if any(t in emsg for t in
                    ("stale", "not found", "unknown job", "expired"))
                    else "FORMAT/other")
            logger.warning("serve: job %s SHARE REJECTED [%s] code=%s err=%s "
                           "staleness=%.1fs", job_id, kind, res.error_code,
                           res.error, staleness)

    # ---- teardown --------------------------------------------------------

    async def stop(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()  # EOF -> binary exits cleanly
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass
        self.kill_remote_orphans()

    def kill_remote_orphans(self) -> None:
        """Reap the remote serve binary (best-effort) so it doesn't hold the GPU."""
        argv = [
            "ssh", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            f"{self.ssh_user}@{self.rig}",
            "pkill -9 -f pearl_miner_sm89; true",
        ]
        try:
            subprocess.run(argv, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=15.0)
            logger.info("serve: kill_remote_orphans pkill on %s", self.rig)
        except Exception:
            logger.warning("serve: kill_remote_orphans ssh pkill failed",
                           exc_info=True)


async def _run_serve(args, mining_config, loop, client_cls) -> int:
    """Drive the persistent ServeLoop against ONE stable pool connection.

    One persistent pool socket (no reconnect churn) + one persistent ssh pipe to
    the serve-mode binary (CUDA initialized ONCE). Exits 0 on first accepted
    share; reaps the remote binary on exit/atexit.
    """
    import atexit

    if not args.rig:
        raise SystemExit("--serve requires --backend ssh-rig and --rig <host>")

    serve = ServeLoop(
        mining_config=mining_config, submit=args.submit, client=None, loop=loop,
        rig=args.rig, ssh_user=args.ssh_user, bin_path=RIG_BIN_PATH,
        dev=args.dev)

    # Reap any stale remote binary BEFORE we start, and on exit.
    serve.kill_remote_orphans()
    atexit.register(lambda: serve.kill_remote_orphans())

    stop_box: dict = {}

    def _on_accepted():
        client = stop_box.get("client")
        if client is not None:
            loop.call_soon_threadsafe(lambda: loop.create_task(client.stop()))

    serve.on_accepted = _on_accepted

    client = client_cls(
        host=args.host, port=args.port, wallet=args.wallet, worker=args.worker,
        agent=args.agent, on_new_job=serve.on_new_job)
    stop_box["client"] = client
    serve.client = client

    mode = "SUBMIT" if args.submit else "DRY-RUN (no submit)"
    logger.info("Canary SERVE: pool=%s:%d rig=%s mode=%s wallet=%s worker=%s",
                args.host, args.port, args.rig, mode, args.wallet, args.worker)

    await serve.start()
    try:
        rc = await client.run()
    finally:
        await serve.stop()
    return 0 if serve.accepted else rc


# ---------------------------------------------------------------------------
# Live runner
# ---------------------------------------------------------------------------


async def run_live(args) -> int:
    import pearl_mining as pm

    from pearl_stratum.luckypool_client import LuckyPoolStratumClient

    mining_config = pm.MiningConfiguration.from_bytes(bytes.fromhex(args.config_hex))
    loop = asyncio.get_running_loop()

    # SERVE mode owns its own persistent ssh pipe (no per-window backend), so it
    # is dispatched before make_backend.
    if args.serve:
        return await _run_serve(args, mining_config, loop, LuckyPoolStratumClient)

    backend = make_backend(args.backend, args)

    if args.land_share:
        return await _run_land_share(args, mining_config, backend, loop,
                                     LuckyPoolStratumClient)

    mine_loop = CanaryMineLoop(
        mining_config=mining_config, backend=backend, submit=args.submit,
        client=None, loop=loop)

    client = LuckyPoolStratumClient(
        host=args.host, port=args.port, wallet=args.wallet, worker=args.worker,
        agent=args.agent, on_new_job=mine_loop.on_new_job)
    mine_loop.client = client

    mode = "SUBMIT" if args.submit else "DRY-RUN (no submit)"
    logger.info("Canary starting: pool=%s:%d backend=%s mode=%s wallet=%s worker=%s",
                args.host, args.port, args.backend, mode, args.wallet, args.worker)

    mine_task = loop.create_task(mine_loop.run())
    try:
        return await client.run()
    finally:
        mine_loop.stop()
        mine_task.cancel()


async def _run_land_share(args, mining_config, backend, loop, client_cls) -> int:
    """Drive the orphan-free LandShareLoop against ONE stable pool connection.

    Reaps GPU orphans at startup and on exit (atexit), runs serial one-shot
    mines, and stops the pool client the moment a share is ACCEPTED.
    """
    import atexit

    # 1) Kill orphans BEFORE anything else, and register a final reap on exit.
    ko = getattr(backend, "kill_orphans", None)
    if ko is not None:
        ko()
        atexit.register(lambda: _safe_kill_orphans(ko))

    mine_loop = LandShareLoop(
        mining_config=mining_config, backend=backend, submit=args.submit,
        client=None, loop=loop, nonce_count=args.nonce_count)

    stop_box: dict = {}

    def _on_accepted():
        client = stop_box.get("client")
        if client is not None:
            loop.call_soon_threadsafe(lambda: loop.create_task(client.stop()))

    mine_loop.on_accepted = _on_accepted

    client = client_cls(
        host=args.host, port=args.port, wallet=args.wallet, worker=args.worker,
        agent=args.agent, on_new_job=mine_loop.on_new_job)
    stop_box["client"] = client
    mine_loop.client = client

    mode = "SUBMIT" if args.submit else "DRY-RUN (no submit)"
    logger.info("Canary LAND-SHARE: pool=%s:%d backend=%s mode=%s nonce_count=%d "
                "wallet=%s worker=%s", args.host, args.port, args.backend, mode,
                args.nonce_count, args.wallet, args.worker)

    mine_task = loop.create_task(mine_loop.run())
    try:
        rc = await client.run()
    finally:
        mine_loop.stop()
        mine_task.cancel()
    # Exit 0 if we landed a share; otherwise propagate the client's rc.
    return 0 if mine_loop.accepted else rc


def _safe_kill_orphans(ko) -> None:
    try:
        ko()
    except Exception:
        logger.debug("atexit kill_orphans raised", exc_info=True)


# ---------------------------------------------------------------------------
# Offline self-test (NO pool, NO rig, NO GPU)
# ---------------------------------------------------------------------------


def run_selftest(args) -> int:
    """Offline validation against the REAL captured job header (rust backend,
    easy nbits). NO socket, NO ssh. Asserts verify_plain_proof==True and that
    notify->job parsing of the real captured notify works."""
    import pearl_mining as pm

    from pearl_stratum.luckypool_client import parse_luckypool_notify

    print("=== run_canary.py OFFLINE self-test ===")
    mining_config = pm.MiningConfiguration.from_bytes(bytes.fromhex(DEFAULT_CONFIG_HEX))
    print(f"[1] mining_config parsed: common_dim={mining_config.common_dim} "
          f"rank={mining_config.rank} mma={mining_config.mma_type}")

    # --- notify -> job parsing on the REAL captured notify params ---
    notify_params = {
        "job_id": CAPTURED_JOB_ID,
        "header": CAPTURED_HEADER_HEX,
        "target": CAPTURED_TARGET_HEX,
        "height": CAPTURED_HEIGHT,
    }
    job = parse_luckypool_notify(notify_params)
    assert job.job_id == CAPTURED_JOB_ID, job.job_id
    assert job.header_bytes == bytes.fromhex(CAPTURED_HEADER_HEX)
    assert job.height == CAPTURED_HEIGHT, job.height
    assert len(job.header_bytes) == 76
    print(f"[2] parse_luckypool_notify OK: job_id={job.job_id} height={job.height} "
          f"target={job.target:#x} header_len={len(job.header_bytes)}")

    # --- real header body + EASY nbits so the CPU reference hits instantly ---
    real = bytes.fromhex(CAPTURED_HEADER_HEX)
    easy_header = real[:-4] + struct.pack("<I", SELFTEST_EASY_NBITS)
    bh_real = pm.IncompleteBlockHeader.from_bytes(real)
    bh_easy = pm.IncompleteBlockHeader.from_bytes(easy_header)
    assert easy_header[:-4] == real[:-4], "header body must be preserved"
    print(f"[3] real nbits={bh_real.nbits:#x} -> easy nbits={bh_easy.nbits:#x} "
          f"(body 72B preserved={easy_header[:-4] == real[:-4]})")

    # Build the easy-target job object the SAME way the driver gets one, then
    # run it through the runner's job handler (rust backend, dry-run). The gate
    # is now the SHARE bound (wire_target * h*w*dot); the rust backend mines to
    # the easy header nbits, so size the wire target to saturate the bound near
    # 2^256 — i.e. make the share gate behave like the old easy-nbits pass.
    factor = (mining_config.hash_tile_h * mining_config.hash_tile_w
              * (mining_config.common_dim - mining_config.common_dim % mining_config.rank))
    max_wire = ((1 << 256) - 1) // factor
    easy_notify = dict(notify_params, header=easy_header.hex(), target=f"{max_wire:064x}")
    easy_job = parse_luckypool_notify(easy_notify)
    backend = RustCpuBackend()
    res = handle_job(easy_job, mining_config=mining_config, backend=backend,
                     submit=False)
    print(f"[4] handle_job(rust, easy nbits, dry-run): {res}")
    assert res["verify"] is True, f"verify must be True, got {res!r}"
    assert res["submitted"] is False, "dry-run must NOT submit"

    print()
    print("RESULT: OFFLINE VALIDATION PASSED")
    print("  - notify->job parsing of the REAL captured notify: OK")
    print("  - REAL captured header body -> mine -> create_proof -> "
          "verify_plain_proof == True (easy nbits): OK")
    print("  - dry-run fail-safe (no submit): OK")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LuckyPool canary miner")
    p.add_argument("--backend", choices=["ssh-rig", "local", "rust"],
                   default="ssh-rig")
    p.add_argument("--rig", help="rig host/IP for --backend ssh-rig (e.g. 192.168.70.6)")
    p.add_argument("--ssh-user", default="root")
    p.add_argument("--dev", type=int, default=0, help="GPU device index on the rig")
    p.add_argument("--mine-timeout", type=float, default=None,
                   help="per-mine subprocess timeout seconds. For --land-share "
                        "this must comfortably exceed a full nonce_count sweep "
                        "(default 45s); for the preempt loop it is short "
                        "(default 25s).")
    p.add_argument("--land-share", action="store_true", default=False,
                   help="orphan-free serial mode that lands an ACCEPTED share: "
                        "kill orphans, mine one-shot windows of the CURRENT job "
                        "(no preemption), and submit a verified HIT even if the "
                        "job rotated (exploit pool grace). Exits on first accept.")
    p.add_argument("--serve", action="store_true", default=False,
                   help="DEFINITIVE non-stale mode: ONE persistent pool conn + "
                        "ONE persistent ssh pipe to the serve-mode binary (CUDA "
                        "init ONCE). Each notify pushes a JOB line; HITs are "
                        "mapped to the live job, verified, and submitted. Always "
                        "mines the CURRENT header so the share is fresh. Requires "
                        "--backend ssh-rig + --rig. Exits 0 on first accept.")
    p.add_argument("--nonce-count", type=int, default=LAND_SHARE_NONCE_COUNT,
                   help="nonces per one-shot mine in --land-share mode (returns "
                        "on the first HIT; hits arrive ~nonce 8). Default 48.")

    p.add_argument("--host", default="pearl-ca1.luckypool.io")
    p.add_argument("--port", type=int, default=3360)
    p.add_argument("--wallet", help="Pearl wallet address (prl1...)")
    p.add_argument("--worker", default="cnry01")
    p.add_argument("--agent", default="lpminer/0.1.9-552bdfe")

    p.add_argument("--config-hex", default=DEFAULT_CONFIG_HEX,
                   help="52-byte MiningConfiguration hex (default: captured)")

    submit = p.add_mutually_exclusive_group()
    submit.add_argument("--dry-run", action="store_true", default=True,
                        help="(default) never submit; just verify + log")
    submit.add_argument("--submit", action="store_true", default=False,
                        help="actually submit verified shares to the pool")

    p.add_argument("--selftest", action="store_true",
                   help="OFFLINE validation (no pool, no rig); rust backend, "
                        "real captured header, easy nbits")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.selftest:
        return run_selftest(args)

    # Resolve the per-mine timeout default by mode (land-share needs a longer
    # window to fit a full nonce_count sweep; the preempt loop wants it short).
    if args.mine_timeout is None:
        args.mine_timeout = (LAND_SHARE_MINE_TIMEOUT_S if args.land_share
                             else MINE_TIMEOUT_S)

    if not args.wallet:
        raise SystemExit("--wallet is required for the live canary "
                         "(or use --selftest for offline validation)")
    # --submit overrides the default --dry-run.
    return asyncio.run(run_live(args))


if __name__ == "__main__":
    sys.exit(main())
