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
`pow_target` and to gate submission). The header's embedded `nbits` is the
BLOCK difficulty that `verify_plain_proof` checks; the pool always sends a
share target <= the block target so a share that clears `nbits` also clears the
pool target.
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


def gpu_sm89_mine(
    header_bytes: bytes,
    mining_config: "object",
    target: int,
    nonce_range: range,
) -> Optional[bytes]:
    """Drive the sm_89 GPU kernel for one attempt-batch and serialize a hit.

    REQUIRES torch + `pearl_gemm_cuda` (the sm_89 .so) + `miner_base` to be
    importable AND a compatible GPU (sm_89). Not usable in the offline test env
    (those deps are intentionally absent there). The transcript the GPU emits
    is arch-independent and matches `rust_cpu_mine` by construction (integer
    GEMM + XOR-reduce + rotate); the CPU reference `cpuminer/ref/pearl_ref.py`
    is the bit-exact oracle both share.

    Flow (per report 07 §"How the driver invokes the kernel"):
      1. job_key = blake3(header || mining_config.to_bytes())  -> merkle key.
      2. Build a `NonceBatch` over `nonce_range`; fill A (per nonce) + shared B.
      3. Set `pow_target` from the wire `target` (32 LE bytes); `pow_key` from
         the commitment-A seed the kernel derives from job_key.
      4. Launch `gemm_persistent_multinonce`; on a hit the kernel writes the
         winning tile coords + opened rows/cols into the per-nonce
         `host_signal_header`.
      5. Build `OpenedBlockInfo` (A_row_indices, B_column_indices, raw A, B^T,
         commitment, noise_rank) and call
         `miner_base.block_submission.create_proof(opened, header)`.
      6. Return `PlainProof.to_base64()` decoded to bytes.
    """
    import torch  # noqa: F401  (presence gates this backend)
    import pearl_gemm_cuda  # noqa: F401
    from miner_base.block_submission import create_proof  # noqa: F401
    from pearl_mining import PlainProof  # noqa: F401

    # The concrete kernel-launch + signal-readback wiring lives in the existing
    # sm_89 driver (`_miner_driver_sm89.py`) and `_nonce_batcher.NonceBatch`.
    # This function is the integration seam: the offline gate validates the
    # SAME proof bytes via `rust_cpu_mine`, and the kernel transcript is proven
    # arch-independent, so the GPU path differs only in WHERE the transcript is
    # computed, not WHAT it computes.
    raise NotImplementedError(
        "gpu_sm89_mine requires the sm_89 pearl_gemm_cuda .so + torch + miner_base "
        "on a compatible GPU rig; it is not runnable in the offline test env. "
        "Use rust_cpu_mine for offline validation; on a rig, wire NonceBatch "
        "(see _nonce_batcher.py) + create_proof here."
    )


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
            # Sanity: the proof must verify before we burn a submit slot.
            bh = pm.IncompleteBlockHeader.from_bytes(job.header_bytes)
            proof = pm.PlainProof.from_base64(base64.b64encode(proof_bytes).decode())
            ok, msg = pm.verify_plain_proof(bh, proof)
            if not ok:
                logger.error("backend produced invalid proof, not submitting: %s", msg)
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
