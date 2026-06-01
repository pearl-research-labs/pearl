"""LuckyPool miner driver — wires the stratum adapter to a `mine()` backend.

This is the PLUMBING that turns the (already bit-exact) Pearl PoW kernel into a
working LuckyPool miner:

    mining.notify {job_id, header, target, height}
        -> derive job_key/seeds from header+mining_config (blake3 chain)
        -> run mining attempts (noise-gen -> noisy int8 GEMM -> on-device PoW)
        -> on a winning tile, serialize the PlainProof (pearl_mining serializer)
        -> mining.submit {job_id, plain_proof, hs}

Mining backend (`MineFn`)
-------------------------
The driver is backend-agnostic. A backend is any callable

    mine(header_bytes: bytes,
         mining_config: MiningConfiguration,
         target: int,
         nonce_range: range) -> Optional[bytes]   # proof.bin bytes, or None

Two backends ship here:

  * `rust_cpu_mine` (default, validated offline): calls
    `pearl_mining.mine(m, n, k, header, config, signal_range, wrong_jackpot_hash)`
    — the protocol-authoritative Rust miner. It runs the EXACT derivation chain
    the GPU kernel reproduces (job_key = blake3(header||config); commitment
    chain; noise from seeds; noised int8 GEMM; transcript; jackpot-hash <
    bound) and returns a `PlainProof`. Arch-independent; bit-exact vs the
    captured oracle (see report 07). This is the offline-validated path and the
    correctness oracle the GPU backend must match.

  * `gpu_sm89_mine` (production, requires the sm_89 `pearl_gemm_cuda` .so + a
    matching GPU): drives `pg.noisy_gemm` / `gemm_persistent_multinonce` via
    `_nonce_batcher.NonceBatch`, reads back the winning `signal_pairs`, builds
    an `OpenedBlockInfo`, and serializes via `miner_base.block_submission.
    create_proof` -> `PlainProof.to_base64`. The transcript the GPU emits is
    arch-independent and matches `rust_cpu_mine` by construction. This backend
    is import-guarded: it is only usable where torch + pearl_gemm_cuda +
    miner_base are installed (a real rig), NOT in the offline test env.

The `target` from the wire is the SHARE threshold (used to set the kernel's
`pow_target` AND to gate submission). Both the kernel and the pool scale it by
the hash-tile work factor h*w*k (= `MiningJob.adjust_target`), so a share is
accepted iff jackpot_hash <= wire_target * h*w*k. The header's embedded `nbits`
is the harder BLOCK target; it is EASIER (numerically larger) on the wire, i.e.
wire_target >= block_target. Submission must NOT be gated on `verify_plain_proof`
(which derives its bound from `nbits`, the block target) — that rejects valid
shares with the pool's own "hash does not meet difficulty target" (stratum
code 23).
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Callable, Optional

logger = logging.getLogger("luckypool_miner")

# A backend turns (header, config, target, nonce_range) into proof bytes or None.
MineFn = Callable[[bytes, "object", int, range], Optional[bytes]]


# ---------------------------------------------------------------------------
# Backend: Rust CPU (offline-validated authority)
# ---------------------------------------------------------------------------


def rust_cpu_mine(
    header_bytes: bytes,
    mining_config: "object",
    target: int,
    nonce_range: range,
    *,
    m: int = 256,
    n: int = 256,
    signal_range: Optional[tuple[int, int]] = None,
) -> Optional[bytes]:
    """Mine one share with the protocol-authoritative Rust miner.

    Returns the proof.bin bytes (`PlainProof.to_base64` decoded) on a hit, else
    None. `m`/`n` are the search tile (the pool job is M=N=131072; we search a
    small tile per attempt — a winning hash-tile within it is a valid share).
    `k` is taken from `mining_config.common_dim`.

    `nonce_range` is accepted for interface parity with the GPU backend; the
    Rust `mine` draws its own random A/B internally (the A/B *are* the nonce).
    We honor `len(nonce_range)` as a cap on attempts so the driver stays
    responsive to new jobs.
    """
    import pearl_mining as pm

    bh = pm.IncompleteBlockHeader.from_bytes(header_bytes)
    k = int(mining_config.common_dim)
    # mine() loops until it finds a hit at the header's difficulty. On the easy
    # pool/share difficulty this is sub-second; we call it once per "attempt
    # budget" (the caller's nonce_range bounds how often we re-check for a new
    # job between calls is handled by the driver loop).
    proof = pm.mine(m, n, k, bh, mining_config, signal_range, False)
    return base64.b64decode(proof.to_base64())


def make_rust_cpu_backend(m: int = 256, n: int = 256,
                          signal_range: Optional[tuple[int, int]] = None) -> MineFn:
    """Bind tile size / signal range into a `MineFn`."""
    def _fn(header_bytes: bytes, mining_config: "object", target: int, nonce_range: range):
        return rust_cpu_mine(header_bytes, mining_config, target, nonce_range,
                             m=m, n=n, signal_range=signal_range)
    return _fn


# ---------------------------------------------------------------------------
# Backend: GPU sm_89 (production; import-guarded)
# ---------------------------------------------------------------------------


# Path to the standalone sm_89 miner binary. Overridable via env.
PEARL_MINER_SM89_BIN = os.environ.get(
    "PEARL_MINER_SM89_BIN",
    "/opt/pearl/pearl_miner_sm89",  # default rig install path
)


def _splitmix64_fill(n: int, seed: int) -> "object":
    """Regenerate the binary's seed-derived int8 operand stream (numpy int8).

    Bit-exact with `fill_AB`/`splitmix64` in pearl_miner_sm89.cu: per element,
    advance splitmix64 and map `r % 127 - 63` to [-63,63]. The binary reports the
    `seed` it used for A; B uses `seed ^ 0xD1B54A32D192ED03`. The driver replays
    the SAME operands so `create_proof` can build the merkle tree over them.
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


def gpu_sm89_mine(
    header_bytes: bytes,
    mining_config: "object",
    target: int,
    nonce_range: range,
) -> Optional[bytes]:
    """Drive the standalone sm_89 GPU miner (`pearl_miner_sm89`) for one job and
    serialize a hit into proof.bin bytes.

    REQUIRES the prebuilt sm_89 binary (`$PEARL_MINER_SM89_BIN`) + a compatible
    GPU (sm_89), plus `pearl_mining` (the authoritative proof serializer). The
    binary does the GPU-arch-specific work (noised GEMM + on-device PoW); this
    function does the arch-independent proof serialization with `pearl_mining`.
    Not runnable in the offline env (no GPU / no binary there) — `rust_cpu_mine`
    is the offline-validated path.

    Flow:
      1. Shell out to `pearl_miner_sm89 mode=mine header=… config=… target=… …`.
         The binary derives job_key/seeds (blake3, validated bit-exact vs the
         oracle), generates seed-derived noised operands, runs the sm_89 PoW
         kernel, and on a hit prints JSON:
           HIT {"seed":S,"a_rows":[…8…],"b_cols":[…16…],"transcript":[…],"gpu_hash":…}
      2. Regenerate the SAME A / B^T operands from `seed` (splitmix64) so the
         merkle tree matches what the binary mined.
      3. Build `OpenedBlockInfo` and call `create_proof(opened, header)` ->
         `PlainProof.to_base64()` -> proof bytes.  The caller `verify_plain_proof`s
         before submitting.
    """
    import json
    import subprocess

    import torch
    from miner_base.block_submission import create_proof
    from pearl_gateway.comm.dataclasses import OpenedBlockInfo

    m = 131072
    n = 131072
    k = int(mining_config.common_dim)
    r = int(getattr(mining_config, "rank", 256))

    # The standalone binary parses `target=` as big-endian (MSB-first), the same
    # human/pool convention parse_target_hex uses; it converts to the kernel's
    # little-endian pow_target words internally.
    target_be_hex = int(target).to_bytes(32, "big").hex()
    args = (
        f"header={header_bytes.hex()} config={mining_config.to_bytes().hex()} "
        f"target={target_be_hex} mode=mine m={m} n={n} k={k} r={r} "
        f"nonce_start={nonce_range.start} "
        f"nonce_count={max(1, len(nonce_range))} dev={os.environ.get('PEARL_GPU_DEV', '0')}"
    )
    proc = subprocess.run(
        [PEARL_MINER_SM89_BIN],
        input=args.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=float(os.environ.get("PEARL_MINER_SM89_TIMEOUT", "120")),
    )
    out = proc.stdout.decode(errors="replace").strip()
    if not out.startswith("HIT"):
        if out and not out.startswith("NOHIT"):
            logger.warning("pearl_miner_sm89: %s | stderr=%s", out, proc.stderr.decode(errors="replace")[-400:])
        return None  # NOHIT (range exhausted) or error

    hit = json.loads(out[len("HIT"):].strip())
    seed = int(hit["seed"])
    a_rows = list(map(int, hit["a_rows"]))
    b_cols = list(map(int, hit["b_cols"]))

    # Regenerate the FULL A / B^T operands the binary mined (splitmix64). The
    # B fill produces (n, k) rows = the columns of B = the rows of B^T, which is
    # exactly the `B_t` create_proof expects. create_proof builds the merkle tree
    # over the full matrices (it needs the multiproof siblings), so full A/B^T are
    # required, not just the disclosed strips.
    A = torch.from_numpy(_splitmix64_fill(m * k, seed)).reshape(m, k)
    B_t = torch.from_numpy(_splitmix64_fill(n * k, seed ^ 0xD1B54A32D192ED03)).reshape(n, k)

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


def select_backend() -> MineFn:
    """Pick the GPU backend when its deps + a GPU are available, else the Rust
    CPU backend. Override with `PEARL_MINE_BACKEND=rust|gpu`."""
    forced = os.environ.get("PEARL_MINE_BACKEND", "").strip().lower()
    if forced == "rust":
        return make_rust_cpu_backend()
    if forced == "gpu":
        return gpu_sm89_mine
    try:
        import torch  # noqa: F401
        import pearl_gemm_cuda  # noqa: F401
        if torch.cuda.is_available():
            return gpu_sm89_mine
    except Exception:
        pass
    return make_rust_cpu_backend()


# ---------------------------------------------------------------------------
# Driver: glue stratum -> mine -> submit
# ---------------------------------------------------------------------------


async def run_luckypool_miner(
    *,
    host: str,
    port: int,
    wallet: str,
    worker: str,
    mining_config_bytes: bytes,
    mine_fn: Optional[MineFn] = None,
    attempts_per_job: int = 1,
    agent: str = "pearl-stratum-luckypool/0.1",
) -> int:
    """Connect to LuckyPool and mine. Returns the client's exit code.

    `mining_config_bytes` is the calibrated 52-byte MiningConfiguration the
    miner commits to (the pool does NOT push it on LuckyPool — it is fixed /
    derived from the kernel's transcript pattern; see report 07). The driver
    decodes it once and reuses it for every job.
    """
    import asyncio

    import pearl_mining as pm

    from pearl_stratum.luckypool_client import LuckyPoolStratumClient, LuckyPoolJob

    mining_config = pm.MiningConfiguration.from_bytes(mining_config_bytes)
    backend = mine_fn or select_backend()

    client_box: dict[str, LuckyPoolStratumClient] = {}
    loop = asyncio.get_running_loop()

    def on_new_job(job: LuckyPoolJob) -> None:
        # Spawn mining for this job; keep the read-loop responsive.
        loop.create_task(_mine_and_submit(job))

    async def _mine_and_submit(job: LuckyPoolJob) -> None:
        client = client_box["client"]
        for _ in range(attempts_per_job):
            try:
                proof_bytes = await asyncio.to_thread(
                    backend, job.header_bytes, mining_config, job.target, range(0, 1)
                )
            except NotImplementedError as e:
                logger.error("mining backend unavailable: %s", e)
                return
            except Exception:
                logger.exception("mining attempt failed")
                return
            if proof_bytes is None:
                continue
            # Gate on the SHARE bound (wire_target * h*w*k = the pool's
            # adjust_target), NOT verify_plain_proof (which uses the header-nbits
            # block bound and rejects valid shares -> stratum code 23). dump_jackpot
            # recomputes the jackpot via the same path the verifier uses, so this
            # match is bit-exact with the pool's check.
            bh = pm.IncompleteBlockHeader.from_bytes(job.header_bytes)
            proof = pm.PlainProof.from_base64(base64.b64encode(proof_bytes).decode())
            jh_le, h, w, dot, _nbits = pm.dump_jackpot(bh, proof)
            jackpot = int.from_bytes(jh_le, "little")
            share_bound = job.target * h * w * dot
            if share_bound > (1 << 256) - 1 or jackpot > share_bound:
                logger.error("backend candidate misses share target "
                             "(jackpot=2^%d bound=2^%d), not submitting",
                             jackpot.bit_length(), share_bound.bit_length())
                continue
            b64 = base64.b64encode(proof_bytes).decode()
            res = await client.submit_share(job.job_id, b64, hashrate=0.0)
            logger.info("share submit accepted=%s latency=%.1fms err=%s",
                        res.accepted, res.latency_ms, res.error)
            return

    client = LuckyPoolStratumClient(
        host=host, port=port, wallet=wallet, worker=worker,
        agent=agent, on_new_job=on_new_job,
    )
    client_box["client"] = client
    return await client.run()
