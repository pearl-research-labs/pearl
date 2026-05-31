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
    "PEARL_SM89_SWIZZLE": "2",
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


# ---------------------------------------------------------------------------
# A/B regeneration (numpy splitmix64) — identical to gpu_sm89_mine in the driver
# ---------------------------------------------------------------------------


def _splitmix64_fill(n: int, seed: int):
    """Regenerate the GPU binary's seed-derived int8 operand stream.

    Bit-exact with fill_AB/splitmix64 in pearl_miner_sm89.cu: advance
    splitmix64, map r % 127 - 63 into [-63, 63]. B uses seed ^ MIX_CONST.
    """
    import numpy as np

    MASK = (1 << 64) - 1
    out = np.empty(n, dtype=np.int8)
    s = seed & MASK
    for i in range(n):
        s = (s + 0x9E3779B97F4A7C15) & MASK
        z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        z = z ^ (z >> 31)
        out[i] = int(z % 127) - 63
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
    import torch
    from miner_base.block_submission import create_proof
    from pearl_gateway.comm.dataclasses import OpenedBlockInfo

    m = 131072
    n = 131072
    k = int(mining_config.common_dim)
    r = int(getattr(mining_config, "rank", 256))

    seed = int(hit["seed"])
    a_rows = list(map(int, hit["a_rows"]))
    b_cols = list(map(int, hit["b_cols"]))

    A = torch.from_numpy(_splitmix64_fill(m * k, seed)).reshape(m, k)
    B_t = torch.from_numpy(_splitmix64_fill(n * k, seed ^ B_SEED_MIX)).reshape(n, k)

    opened = OpenedBlockInfo(
        A_row_indices=a_rows,
        B_column_indices=b_cols,
        A=A,
        B_t=B_t,
        commitment_hash=None,
        noise_rank=r,
    )
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
    line = (
        f"header={header_bytes.hex()} config={mining_config.to_bytes().hex()} "
        f"target={target_be_hex} m={m} n={n} k={k} r={r} mode=mine "
        f"nonce_start={nonce_start} nonce_count={nonce_count} dev={dev}"
    )
    return line.encode()


def _parse_gpu_output(out: str) -> Optional[dict]:
    """Parse the GPU binary stdout. Returns the HIT dict, or None on NOHIT."""
    import json

    out = out.strip()
    if out.startswith("HIT"):
        return json.loads(out[len("HIT"):].strip())
    return None


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

    def _argv(self) -> list[str]:
        env_prefix = " ".join(f"{k}={v}" for k, v in GPU_ENV.items())
        remote = f"env {env_prefix} {self.bin_path}"
        return [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            f"{self.ssh_user}@{self.rig}",
            remote,
        ]


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
    ok, msg = pm.verify_plain_proof(bh, proof)
    result["verify"] = bool(ok)
    logger.info("job %s: verify_plain_proof=%s (%s)", job.job_id, ok, msg)

    if not ok:
        # Fail-safe: never submit an unverifiable proof. On the live pool this
        # is also how a stale/mismatched mining_config surfaces — verify fails
        # BEFORE we burn a submit slot.
        logger.error("job %s: NOT submitting (verify failed): %s", job.job_id, msg)
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
        logger.info("job %s: would submit, verify_plain_proof=%s (proof_b64_len=%d)",
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
# Live runner
# ---------------------------------------------------------------------------


async def run_live(args) -> int:
    import pearl_mining as pm

    from pearl_stratum.luckypool_client import LuckyPoolStratumClient

    mining_config = pm.MiningConfiguration.from_bytes(bytes.fromhex(args.config_hex))
    backend = make_backend(args.backend, args)
    loop = asyncio.get_running_loop()
    box: dict = {}

    mine_loop = CanaryMineLoop(
        mining_config=mining_config, backend=backend, submit=args.submit,
        client=None, loop=loop)

    client = LuckyPoolStratumClient(
        host=args.host, port=args.port, wallet=args.wallet, worker=args.worker,
        agent=args.agent, on_new_job=mine_loop.on_new_job)
    box["client"] = client
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
    # run it through the runner's job handler (rust backend, dry-run).
    easy_notify = dict(notify_params, header=easy_header.hex())
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
    p.add_argument("--mine-timeout", type=float, default=MINE_TIMEOUT_S,
                   help="per-mine-window subprocess timeout seconds (short, so a "
                        "new job can preempt the in-flight window)")

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

    if not args.wallet:
        raise SystemExit("--wallet is required for the live canary "
                         "(or use --selftest for offline validation)")
    # --submit overrides the default --dry-run.
    return asyncio.run(run_live(args))


if __name__ == "__main__":
    sys.exit(main())
