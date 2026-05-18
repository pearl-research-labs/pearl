#!/usr/bin/env python3
"""Direct Akoya-shape GPU submitter for P1K-132.

This bypasses vLLM shape selection and mines exactly the Akoya baseline
configuration:

    m=8192, n=32768, k=2048, rank=128, rows=[0, 8], default 64 columns

It submits only when the locally built PlainProof verifies against the pool
share difficulty. Use ``--force-target-max`` for a no-submit canary that proves
the direct GPU host-signal/proof-building path is alive before spending time on
a real share loop.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]

for path in reversed(
    (
        THIS_DIR,
        REPO_ROOT / "py-pearl-mining",
        REPO_ROOT / "miner" / "miner-utils" / "src",
        REPO_ROOT / "miner" / "miner-base" / "src",
        REPO_ROOT / "miner" / "pearl-gateway" / "src",
        REPO_ROOT / "miner" / "pearl-gemm" / "src",
    )
):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


AKOYA_M = 8192
AKOYA_N = 32768
AKOYA_K = 2048
AKOYA_RANK = 128
FORCED_MAX_CANARY_NBITS = 0x207FFFFF
DEFAULT_A_NONCE_BYTES = 12
AKOYA_ROWS = [0, 8]
AKOYA_COLS = [
    0,
    1,
    8,
    9,
    16,
    17,
    24,
    25,
    32,
    33,
    40,
    41,
    48,
    49,
    56,
    57,
    64,
    65,
    72,
    73,
    80,
    81,
    88,
    89,
    96,
    97,
    104,
    105,
    112,
    113,
    120,
    121,
    128,
    129,
    136,
    137,
    144,
    145,
    152,
    153,
    160,
    161,
    168,
    169,
    176,
    177,
    184,
    185,
    192,
    193,
    200,
    201,
    208,
    209,
    216,
    217,
    224,
    225,
    232,
    233,
    240,
    241,
    248,
    249,
]


def build_boundary_hit_v1(
    *,
    job_uuid: str,
    incomplete_header_bytes: bytes,
    seed_hash: bytes | bytearray | None,
    share_verify_nbits: int,
    network_nbits: int,
    m: int,
    n: int,
    mining_config_bytes: bytes,
    a_row_indices: list[int],
    b_column_indices: list[int],
    a_refresh_mode: str,
    a_initial_random: bool,
    a_nonce_row: int,
    a_nonce_bytes: int,
    a_nonce_counter: int | None,
):
    from boundary_hit_v1 import A_MODE_CODES, BoundaryHitV1, SCHEMA_VERSION

    if seed_hash is None:
        raise RuntimeError("BoundaryHitV1 requires akoya_seed_hash on the live mining job")
    if not a_row_indices:
        raise RuntimeError("BoundaryHitV1 requires at least one A row index")
    if not b_column_indices:
        raise RuntimeError("BoundaryHitV1 requires at least one B column index")
    if a_refresh_mode not in A_MODE_CODES:
        raise RuntimeError(f"unsupported A refresh mode for BoundaryHitV1: {a_refresh_mode}")

    return BoundaryHitV1(
        schema_version=SCHEMA_VERSION,
        job_uuid=str(job_uuid),
        incomplete_header_bytes=bytes(incomplete_header_bytes),
        seed_hash=bytes(seed_hash),
        share_verify_nbits=int(share_verify_nbits),
        network_nbits=int(network_nbits),
        m=int(m),
        n=int(n),
        mining_config_bytes=bytes(mining_config_bytes),
        t_rows=min(int(value) for value in a_row_indices),
        t_cols=min(int(value) for value in b_column_indices),
        a_mode=A_MODE_CODES[a_refresh_mode],
        a_initial_random=bool(a_initial_random),
        a_nonce_row=int(a_nonce_row),
        a_nonce_bytes=int(a_nonce_bytes),
        a_nonce_counter=int(a_nonce_counter or 0),
    ).validate()


def emit_boundary_hit_v1_frame(boundary: Any, out_dir: Path, *, emission_index: int) -> dict[str, Any]:
    frame = boundary.to_frame()
    frame_sha256 = hashlib.sha256(frame).hexdigest()
    path = Path(out_dir) / f"boundary_hit_v1_{emission_index:06d}_{boundary.job_uuid}.frame"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(frame)
    return {
        "path": str(path),
        "sha256": frame_sha256,
        "frame_bytes": len(frame),
        "payload_bytes": len(frame) - 4,
        "emission_index": emission_index,
        "job_uuid": boundary.job_uuid,
        "t_rows": boundary.t_rows,
        "t_cols": boundary.t_cols,
    }


def emit_boundary_hit_v1_from_attempt_record(
    attempt_record: Mapping[str, Any],
    out_dir: Path,
    *,
    emission_index: int = 1,
) -> dict[str, Any]:
    from boundary_hit_v1 import BoundaryHitV1

    boundary = BoundaryHitV1.from_proof_artifact(attempt_record)
    return emit_boundary_hit_v1_frame(boundary, Path(out_dir), emission_index=emission_index)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if parsed != sorted(set(parsed)):
        raise argparse.ArgumentTypeError("indices must be sorted and unique")
    return parsed


def make_settings(args: argparse.Namespace):
    from miner_base.settings import MinerSettings

    overrides: dict[str, Any] = {
        "akoya_pool": True,
        "akoya_pool_host": args.host,
        "akoya_pool_port": args.port,
        "akoya_pool_worker": args.worker,
        "akoya_pool_gpu_name": args.gpu_name,
        "akoya_pool_version": args.version,
        "akoya_pool_git_sha": args.git_sha,
        "akoya_pool_timeout": args.timeout,
    }
    if args.wallet:
        overrides["akoya_pool_wallet"] = args.wallet
    return MinerSettings(**overrides)


def mining_config_for_args(args: argparse.Namespace):
    from pearl_gateway.comm.mining_configuration import PearlMiningConfigurationFactory

    return PearlMiningConfigurationFactory.create(
        common_dim=args.k,
        rank=args.rank,
        row_indices=args.rows,
        col_indices=args.cols,
    )


def p1k165_two_phase_num_consumers(args: argparse.Namespace) -> int:
    return (args.tile_size_m // 64) * 128


def target_words(value: int, torch: Any, device: str):
    from pearl_gemm import make_pow_target_tensor

    return make_pow_target_tensor(value, device=device)


def hash_and_commit(tensors: dict[str, Any], key_tensor: Any) -> None:
    from pearl_gemm import commitment_hash_from_merkle_roots, tensor_hash

    tensor_hash(tensors["A"].to(tensors["torch"].uint8), key_tensor, tensors["A_root"], tensors["scratchpad"])
    tensor_hash(tensors["B"].to(tensors["torch"].uint8), key_tensor, tensors["B_root"], tensors["scratchpad"])
    commitment_hash_from_merkle_roots(
        tensors["A_root"],
        tensors["B_root"],
        key_tensor,
        tensors["commitment_hash_A"],
        tensors["commitment_hash_B"],
    )


def ensure_akoya_bseed_b(args: argparse.Namespace, tensors: dict[str, Any], mining_job: Any, hash_key: bytes) -> dict[str, Any]:
    """Populate B from Akoya's deterministic BSeed expansion unless disabled."""

    torch = tensors["torch"]
    if args.random_b or mining_job.akoya_seed_hash is None:
        torch.randint(-63, 64, tensors["B"].shape, out=tensors["B"], dtype=torch.int8, device=tensors["B"].device)
        return {"mode": "random", "refreshed": True}

    import blake3

    from akoya_bseed import expand_bseed_matrix, padded_for_merkle

    seed_hash = bytes(mining_job.akoya_seed_hash)
    # Akoya can adjust share difficulty without rotating the header/seed job
    # payload. Bind the refill cache to the live share target as well so a
    # long-lived runner cannot accidentally reuse B across a stale difficulty
    # context when the session stays connected.
    cache_key = (
        seed_hash,
        bytes(hash_key),
        int(mining_job.akoya_share_difficulty or 0),
        args.n,
        args.k,
    )
    refreshed = tensors.get("_bseed_cache_key") != cache_key
    if refreshed:
        b_bytes = expand_bseed_matrix(seed_hash, args.n, args.k)
        expected_hash_b = blake3.blake3(padded_for_merkle(b_bytes), key=hash_key).digest()
        b_cpu = torch.frombuffer(bytearray(b_bytes), dtype=torch.int8).reshape(args.n, args.k)
        tensors["B"].copy_(b_cpu, non_blocking=True)
        tensors["_bseed_cache_key"] = cache_key
        tensors["_bseed_expected_hash_b"] = expected_hash_b
        tensors["_bseed_seed_hash"] = seed_hash
    else:
        expected_hash_b = tensors.get("_bseed_expected_hash_b")
        if expected_hash_b is None:
            # Defensive fallback for partially initialized state. This should
            # not happen on the normal path, but preserves correctness if a
            # caller injects a cache key without the paired hash.
            b_bytes = expand_bseed_matrix(seed_hash, args.n, args.k)
            expected_hash_b = blake3.blake3(padded_for_merkle(b_bytes), key=hash_key).digest()
            tensors["_bseed_expected_hash_b"] = expected_hash_b
    return {
        "mode": "akoya_bseed",
        "seed_hash": seed_hash.hex(),
        "expected_hash_b": expected_hash_b.hex(),
        "refreshed": bool(refreshed),
        "expected_hash_cached": not bool(refreshed),
    }


def refresh_a(args: argparse.Namespace, tensors: dict[str, Any]) -> dict[str, Any]:
    torch = tensors["torch"]
    mode = args.a_refresh_mode
    initialized = bool(tensors.get("_a_initialized"))
    info: dict[str, Any] = {"mode": mode, "initialized_before": initialized}

    if mode == "full-random":
        torch.randint(-63, 64, tensors["A"].shape, out=tensors["A"], dtype=torch.int8, device=tensors["A"].device)
        tensors["_a_initialized"] = True
        info["full_matrix_refreshed"] = True
        return info

    if not initialized:
        if args.a_initial_random:
            torch.randint(-63, 64, tensors["A"].shape, out=tensors["A"], dtype=torch.int8, device=tensors["A"].device)
            info["initial_fill"] = "random"
        else:
            tensors["A"].zero_()
            info["initial_fill"] = "zero"
        tensors["_a_initialized"] = True

    if mode == "fixed":
        info["full_matrix_refreshed"] = False
        return info

    if mode != "nonce-prefix":
        raise RuntimeError(f"unknown A refresh mode: {mode}")

    counter = int(tensors.get("_a_nonce_counter", 0))
    nonce_bytes = int(args.a_nonce_bytes)
    row = int(args.a_nonce_row)
    if nonce_bytes <= 0 or nonce_bytes > args.k:
        raise RuntimeError("--a-nonce-bytes must be in [1, K]")
    if row < 0 or row >= args.m:
        raise RuntimeError("--a-nonce-row must be in [0, M)")

    x = counter
    values: list[int] = []
    for _ in range(nonce_bytes):
        values.append((x % 127) - 63)
        x //= 127
    nonce = torch.tensor(values, dtype=torch.int8, device=tensors["A"].device)
    tensors["A"][row, :nonce_bytes].copy_(nonce)
    tensors["_a_nonce_counter"] = counter + 1
    info.update(
        {
            "full_matrix_refreshed": False,
            "nonce_counter": counter,
            "nonce_row": row,
            "nonce_bytes": nonce_bytes,
        }
    )
    return info


def allocate_tensors(args: argparse.Namespace, torch: Any) -> dict[str, Any]:
    from pearl_gemm import (
        get_host_signal_header_size,
        get_host_signal_sync_size,
        get_required_scratchpad_bytes,
    )

    device = args.device
    tensors: dict[str, Any] = {
        "torch": torch,
        "A": torch.empty((args.m, args.k), dtype=torch.int8, device=device),
        "B": torch.empty((args.n, args.k), dtype=torch.int8, device=device),
        "A_scales": torch.ones((args.m,), dtype=torch.float32, device=device),
        "B_scales": torch.ones((args.n,), dtype=torch.float32, device=device),
        "C": torch.empty((args.m, args.n), dtype=torch.bfloat16, device=device),
        "EAL": torch.empty((args.m, args.rank), dtype=torch.int8, device=device),
        "EBR": torch.empty((args.n, args.rank), dtype=torch.int8, device=device),
        "EAL_fp16": torch.empty((args.m, args.rank), dtype=torch.float16, device=device),
        "EBR_fp16": torch.empty((args.n, args.rank), dtype=torch.float16, device=device),
        "EAR_R_major": torch.empty((args.k, args.rank), dtype=torch.int8, device=device),
        "EBL_R_major": torch.empty((args.k, args.rank), dtype=torch.int8, device=device),
        "EAR_K_major": torch.empty((args.rank, args.k), dtype=torch.int8, device=device),
        "EBL_K_major": torch.empty((args.rank, args.k), dtype=torch.int8, device=device),
        "AxEBL_fp16": torch.empty((args.m, args.rank), dtype=torch.float16, device=device),
        "EARxBpEB_fp16": torch.empty((args.n, args.rank), dtype=torch.float16, device=device),
        "ApEA": torch.empty((args.m, args.k), dtype=torch.int8, device=device),
        "BpEB": torch.empty((args.n, args.k), dtype=torch.int8, device=device),
        "host_signal_sync": torch.zeros((get_host_signal_sync_size(),), dtype=torch.int8, device=device),
        "host_signal_header_pinned": torch.zeros((get_host_signal_header_size(),), dtype=torch.int8, pin_memory=True),
        "A_root": torch.empty(32, dtype=torch.uint8, device=device),
        "B_root": torch.empty(32, dtype=torch.uint8, device=device),
        "commitment_hash_A": torch.empty(32, dtype=torch.uint8, device=device),
        "commitment_hash_B": torch.empty(32, dtype=torch.uint8, device=device),
        "scratchpad": torch.empty(
            get_required_scratchpad_bytes(max(args.m * args.k, args.n * args.k)),
            dtype=torch.uint8,
            device=device,
        ),
    }
    if args.fast_sideband:
        num_blocks_m = (args.m + args.tile_size_m - 1) // args.tile_size_m
        num_blocks_n = (args.n + args.tile_size_n - 1) // args.tile_size_n
        num_tiles = num_blocks_m * num_blocks_n
        active_boundaries = (
            args.native_sideband_fill_boundaries
            if args.native_sideband_fill_boundaries >= 0
            else args.k // args.rank
        )
        num_consumers = p1k165_two_phase_num_consumers(args)
        tensors["global_sideband_journal"] = torch.zeros(
            (num_tiles * num_consumers * active_boundaries,),
            dtype=torch.int32,
            device=device,
        )
        tensors["p1k165_scalar16_two_phase_finalizer_transcripts"] = torch.empty(
            (0,),
            dtype=torch.int32,
            device=device,
        )
        tensors["p1k165_scalar16_two_phase_finalizer_metadata"] = torch.zeros(
            (num_tiles * num_consumers,),
            dtype=torch.int32,
            device=device,
        )
        # The proof-only kernel reads this side-channel control word when the
        # native global journal fill path is enabled.
        tensors["raw_global_sink"] = torch.zeros((1,), dtype=torch.uint64, device=device)
        tensors["raw_global_sink"][0] = int(active_boundaries)
    if args.int32_noising_a:
        tensors["AxEBL_int32"] = torch.empty((args.m, args.rank), dtype=torch.int32, device=device)
    if args.int32_noising_b:
        tensors["EARxBpEB_int32"] = torch.empty((args.n, args.rank), dtype=torch.int32, device=device)
    return tensors


def refill_inputs(args: argparse.Namespace, tensors: dict[str, Any], mining_job: Any, hash_key: bytes) -> dict[str, Any]:
    a_start = time.perf_counter()
    a_info = refresh_a(args, tensors)
    a_info["elapsed_s"] = time.perf_counter() - a_start

    b_start = time.perf_counter()
    b_info = ensure_akoya_bseed_b(args, tensors, mining_job, hash_key)
    b_info["elapsed_s"] = time.perf_counter() - b_start
    b_info["a_refresh"] = a_info
    return b_info


def materialize_noised_operands_torch(args: argparse.Namespace, tensors: dict[str, Any]) -> None:
    """Fallback noising path used when the fused CUDA noising kernels are unavailable.

    This mirrors the reference equations in miner/pearl-gemm/tests/test_pearl_gemm.py:
      ApEA = A + EAL @ EAR.T
      BpEB = B + EBR @ EBL.T
      AxEBL = A @ EBL
      EARxBpEB = BpEB @ EAR

    The proof kernel consumes int8 ApEA/BpEB. Denoise side tensors are still
    materialized so the same call surface works for proof-only and non-proof-only
    canaries.
    """
    torch = tensors["torch"]

    axebl_i32 = torch._int_mm(tensors["A"], tensors["EBL_R_major"])
    if "AxEBL_int32" in tensors:
        tensors["AxEBL_int32"].copy_(axebl_i32)
    tensors["AxEBL_fp16"].copy_((axebl_i32.to(torch.float32) * (2**-14)).to(torch.float16))
    del axebl_i32

    ea_i32 = torch._int_mm(tensors["EAL"], tensors["EAR_R_major"].t())
    ea_i32.add_(tensors["A"])
    tensors["ApEA"].copy_(ea_i32.to(torch.int8))
    del ea_i32

    eb_i32 = torch._int_mm(tensors["EBR"], tensors["EBL_R_major"].t())
    eb_i32.add_(tensors["B"])
    tensors["BpEB"].copy_(eb_i32.to(torch.int8))
    del eb_i32

    earxbpeb_i32 = torch._int_mm(tensors["BpEB"], tensors["EAR_R_major"])
    if "EARxBpEB_int32" in tensors:
        tensors["EARxBpEB_int32"].copy_(earxbpeb_i32)
    tensors["EARxBpEB_fp16"].copy_((earxbpeb_i32.to(torch.float32) * (2**-12)).to(torch.float16))
    del earxbpeb_i32
    torch.cuda.synchronize()


def run_noisy_once(args: argparse.Namespace, tensors: dict[str, Any], mining_job: Any) -> dict[str, Any]:
    import torch
    from miner_base.block_submission import create_proof
    from miner_base.commitment_hash import CommitmentHasher
    from pearl_gateway.comm.dataclasses import CommitmentHash, OpenedBlockInfo
    from pearl_gemm import (
        HostSignalStatus,
        extract_indices,
        get_host_signal_header,
        get_host_signal_header_size,
        noise_gen,
        noisy_gemm,
    )
    if args.fast_sideband:
        from pearl_gemm import proof_only_gemm, p1k165_scalar16_two_phase_finalizer
    from pearl_mining import IncompleteBlockHeader, verify_plain_proof_with_nbits

    mining_config = mining_config_for_args(args)
    hash_key = CommitmentHasher.get_key(mining_job.incomplete_header_bytes, mining_config)
    key_tensor = torch.frombuffer(bytearray(hash_key), dtype=torch.uint8).to(args.device)
    adjusted_target = mining_job.adjust_target(mining_config)
    pow_value = (1 << 256) - 1 if args.force_target_max else adjusted_target
    tensors["pow_target"] = target_words(pow_value, torch, args.device)

    tensors["host_signal_sync"].zero_()
    tensors["host_signal_header_pinned"].zero_()
    if args.fast_sideband:
        tensors["global_sideband_journal"].zero_()
        tensors["p1k165_scalar16_two_phase_finalizer_metadata"].zero_()
    if "AxEBL_int32" in tensors:
        tensors["AxEBL_int32"].zero_()
    if "EARxBpEB_int32" in tensors:
        tensors["EARxBpEB_int32"].zero_()

    bseed_info = refill_inputs(args, tensors, mining_job, hash_key)

    setup_start = time.perf_counter()
    hash_and_commit(tensors, key_tensor)
    noise_gen(
        R=args.rank,
        EAL=tensors["EAL"],
        EAL_fp16=tensors["EAL_fp16"],
        EAR_R_major=tensors["EAR_R_major"],
        EAR_K_major=tensors["EAR_K_major"],
        EBL_R_major=tensors["EBL_R_major"],
        EBL_K_major=tensors["EBL_K_major"],
        EBR=tensors["EBR"],
        EBR_fp16=tensors["EBR_fp16"],
        key_A=tensors["commitment_hash_A"],
        key_B=tensors["commitment_hash_B"],
    )
    if args.torch_noising:
        materialize_noised_operands_torch(args, tensors)
    torch.cuda.synchronize()
    setup_s = time.perf_counter() - setup_start
    b_root_hex = tensors["B_root"].cpu().numpy().tobytes().hex()
    bseed_info["computed_hash_b"] = b_root_hex
    if bseed_info.get("expected_hash_b"):
        bseed_info["hash_b_matches_expected"] = b_root_hex == bseed_info["expected_hash_b"]

    noisy_kwargs: dict[str, Any] = {
        "A": tensors["A"],
        "B": tensors["B"],
        "EAL": tensors["EAL"],
        "EAL_fp16": tensors["EAL_fp16"],
        "EBR": tensors["EBR"],
        "EBR_fp16": tensors["EBR_fp16"],
        "EAR_R_major": tensors["EAR_R_major"],
        "EBL_R_major": tensors["EBL_R_major"],
        "EAR_K_major": tensors["EAR_K_major"],
        "EBL_K_major": tensors["EBL_K_major"],
        "AxEBL_fp16": tensors["AxEBL_fp16"],
        "EARxBpEB_fp16": tensors["EARxBpEB_fp16"],
        "ApEA": tensors["ApEA"],
        "BpEB": tensors["BpEB"],
        "A_scales": tensors["A_scales"],
        "B_scales": tensors["B_scales"],
        "C": tensors["C"],
        "host_signal_header_pinned": tensors["host_signal_header_pinned"],
        "host_signal_sync": tensors["host_signal_sync"],
        "pow_target": tensors["pow_target"],
        "pow_key": tensors["commitment_hash_A"].view(torch.uint32),
        "AxEBL_int32": tensors.get("AxEBL_int32"),
        "EARxBpEB_int32": tensors.get("EARxBpEB_int32"),
        "tile_size_m": args.tile_size_m,
        "tile_size_n": args.tile_size_n,
        "tile_size_k": args.tile_size_k,
        "cluster_size_m": args.cluster_size_m,
        "cluster_size_n": args.cluster_size_n,
        "pipeline_stages": args.pipeline_stages,
        "swizzle": args.swizzle,
        "swizzle_n_maj": args.swizzle_n_maj,
        "run_noising_A": not args.torch_noising,
        "run_noising_B": not args.torch_noising,
        "skip_reduction": False,
        "skip_denoising": args.skip_denoising,
    }
    if "proof_only" in inspect.signature(noisy_gemm).parameters:
        noisy_kwargs["proof_only"] = args.proof_only
    elif args.proof_only:
        raise RuntimeError("pearl_gemm.noisy_gemm does not expose proof_only in this build")

    start_event = torch.cuda.Event(enable_timing=True)
    kernel_end_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    fast_sideband_profile: dict[str, Any] | None = None
    if args.fast_sideband:
        if not args.torch_noising:
            raise RuntimeError("--fast-sideband currently requires --torch-noising")
        if args.rank != 128:
            raise RuntimeError("--fast-sideband requires --rank 128")
        if (args.tile_size_m, args.tile_size_n, args.tile_size_k) not in {
            (128, 64, 128),
            (128, 256, 128),
        }:
            raise RuntimeError("--fast-sideband requires tile 128x64x128 or 128x256x128")
        if (args.cluster_size_m, args.cluster_size_n) != (1, 1):
            raise RuntimeError("--fast-sideband requires cluster 1x1")
        if args.k % args.rank != 0:
            raise RuntimeError("--fast-sideband requires K divisible by rank")
        active_boundaries = (
            args.native_sideband_fill_boundaries
            if args.native_sideband_fill_boundaries >= 0
            else args.k // args.rank
        )
        if active_boundaries <= 0 or active_boundaries > args.k // args.rank:
            raise RuntimeError("--native-sideband-fill-boundaries must be in [1, K/rank]")
        proof_only_gemm(
            tensors["ApEA"],
            tensors["BpEB"],
            tensors["host_signal_header_pinned"],
            tensors["host_signal_sync"],
            tensors["pow_target"],
            tensors["commitment_hash_A"].view(torch.uint32),
            rank=args.rank,
            tile_size_m=args.tile_size_m,
            tile_size_n=args.tile_size_n,
            tile_size_k=args.tile_size_k,
            cluster_size_m=args.cluster_size_m,
            cluster_size_n=args.cluster_size_n,
            pipeline_stages=args.pipeline_stages,
            swizzle=args.swizzle,
            swizzle_n_maj=args.swizzle_n_maj,
            inner_hash_counter=tensors.get("raw_global_sink"),
            coalesce_receipts=False,
            skip_reduction=False,
            skip_proof_check=True,
            enable_xq_journal=False,
            global_sideband_journal=tensors["global_sideband_journal"],
            enable_native_global_journal_fill=True,
            enable_p1k165_two_phase_pow_check=True,
        )
        kernel_end_event.record()
        num_blocks_m = (args.m + args.tile_size_m - 1) // args.tile_size_m
        num_blocks_n = (args.n + args.tile_size_n - 1) // args.tile_size_n
        num_tiles = num_blocks_m * num_blocks_n
        num_consumers = p1k165_two_phase_num_consumers(args)
        p1k165_scalar16_two_phase_finalizer(
            tensors["global_sideband_journal"],
            tensors["p1k165_scalar16_two_phase_finalizer_transcripts"],
            tensors["p1k165_scalar16_two_phase_finalizer_metadata"],
            tensors["host_signal_header_pinned"],
            tensors["host_signal_sync"],
            tensors["pow_target"],
            tensors["commitment_hash_A"].view(torch.uint32),
            args.m,
            args.n,
            args.k,
            args.rank,
            num_tiles,
            num_blocks_n,
            num_consumers,
            active_boundaries,
            True,
        )
        fast_sideband_profile = {
            "active_boundaries": active_boundaries,
            "num_blocks_m": num_blocks_m,
            "num_blocks_n": num_blocks_n,
            "num_tiles": num_tiles,
            "num_consumers": num_consumers,
        }
    else:
        noisy_gemm(**noisy_kwargs)
        kernel_end_event.record()
    end_event.record()
    end_event.synchronize()
    noisy_ms = float(start_event.elapsed_time(end_event))
    noisy_wall_s = time.perf_counter() - wall_start

    header = get_host_signal_header(tensors["host_signal_header_pinned"])
    header_status = str(getattr(header, "status", None))
    out: dict[str, Any] = {
        "status": "not_triggered",
        "setup_s": setup_s,
        "noisy_ms": noisy_ms,
        "noisy_wall_s": noisy_wall_s,
        "pow_target_mode": "forced_max" if args.force_target_max else "share_adjusted",
        "share_difficulty": mining_job.akoya_share_difficulty,
        "adjusted_target": adjusted_target,
        "host_signal_status": header_status,
        "host_signal_header_size_bytes": get_host_signal_header_size(),
        "bseed": bseed_info,
    }
    if fast_sideband_profile is not None:
        out["fast_sideband_profile"] = {
            **fast_sideband_profile,
            "kernel_ms": float(start_event.elapsed_time(kernel_end_event)),
            "finalizer_ms": float(kernel_end_event.elapsed_time(end_event)),
        }
    if header.status != HostSignalStatus.kSignalTriggered:
        return out

    idxs = extract_indices(header)
    if args.boundary_hit_v1_dir:
        emission_index = int(tensors.get("_boundary_hit_v1_emit_counter", 0)) + 1
        tensors["_boundary_hit_v1_emit_counter"] = emission_index
        boundary = build_boundary_hit_v1(
            job_uuid=mining_job.akoya_job_uuid,
            incomplete_header_bytes=mining_job.incomplete_header_bytes,
            seed_hash=mining_job.akoya_seed_hash,
            share_verify_nbits=mining_job.akoya_share_difficulty,
            network_nbits=mining_job.akoya_network_nbits,
            m=args.m,
            n=args.n,
            mining_config_bytes=mining_config.to_bytes(),
            a_row_indices=idxs.A_row_indices,
            b_column_indices=idxs.B_column_indices,
            a_refresh_mode=args.a_refresh_mode,
            a_initial_random=args.a_initial_random,
            a_nonce_row=args.a_nonce_row,
            a_nonce_bytes=args.a_nonce_bytes,
            a_nonce_counter=bseed_info.get("a_refresh", {}).get("nonce_counter"),
        )
        out["boundary_hit_v1"] = emit_boundary_hit_v1_frame(
            boundary,
            args.boundary_hit_v1_dir,
            emission_index=emission_index,
        )
    opened = OpenedBlockInfo(
        A_row_indices=idxs.A_row_indices,
        B_column_indices=idxs.B_column_indices,
        A=tensors["A"].cpu().detach(),
        B_t=tensors["B"].cpu().detach(),
        commitment_hash=CommitmentHash(
            noise_seed_A=tensors["commitment_hash_A"].cpu().numpy().tobytes(),
            noise_seed_B=tensors["commitment_hash_B"].cpu().numpy().tobytes(),
        ),
        noise_rank=args.rank,
    )
    proof_start = time.perf_counter()
    plain_proof = create_proof(opened, mining_job.incomplete_header_bytes)
    proof_s = time.perf_counter() - proof_start
    header_obj = IncompleteBlockHeader.from_bytes(mining_job.incomplete_header_bytes)
    share_ok, share_message = verify_plain_proof_with_nbits(
        header_obj,
        plain_proof,
        mining_job.akoya_share_difficulty,
    )
    verify_nbits = (
        FORCED_MAX_CANARY_NBITS
        if args.force_target_max and not args.submit
        else mining_job.akoya_share_difficulty
    )
    ok, message = verify_plain_proof_with_nbits(header_obj, plain_proof, verify_nbits)
    out.update(
        {
            "status": "verified" if ok else "invalid",
            "proof_build_s": proof_s,
            "plain_verify": {
                "valid": bool(ok),
                "message": message,
                "nbits": verify_nbits,
                "mode": "forced_max_canary" if verify_nbits == FORCED_MAX_CANARY_NBITS else "share_difficulty",
            },
            "share_verify": {
                "valid": bool(share_ok),
                "message": share_message,
                "nbits": mining_job.akoya_share_difficulty,
            },
            "indices": {
                "A_row_indices": idxs.A_row_indices,
                "B_column_indices": idxs.B_column_indices,
                "cartesian_cells": len(idxs.A_row_indices) * len(idxs.B_column_indices),
            },
            "plain_proof_base64_bytes": len(plain_proof.to_base64()),
            "_plain_proof": plain_proof,
            "_opened": opened,
        }
    )
    if args.save_proof_artifact:
        out["proof_artifact"] = {
            "schema": "akoya_split_boundary_proof_artifact.v1",
            "incomplete_header_bytes_hex": mining_job.incomplete_header_bytes.hex(),
            "plain_proof_base64": plain_proof.to_base64(),
            "mining_config_bytes_hex": opened.get_mining_config().to_bytes().hex(),
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "rank": args.rank,
            "indices": {
                "A_row_indices": idxs.A_row_indices,
                "B_column_indices": idxs.B_column_indices,
            },
            "a_generation": {
                "mode": args.a_refresh_mode,
                "initial_random": bool(args.a_initial_random),
                "nonce_row": args.a_nonce_row,
                "nonce_bytes": args.a_nonce_bytes,
                "nonce_counter": bseed_info.get("a_refresh", {}).get("nonce_counter"),
                "initial_fill": bseed_info.get("a_refresh", {}).get("initial_fill"),
            },
            "b_generation": {
                "mode": bseed_info.get("mode"),
                "seed_hash": bseed_info.get("seed_hash"),
                "expected_hash_b": bseed_info.get("expected_hash_b"),
            },
            "commitment_hash": {
                "noise_seed_A": tensors["commitment_hash_A"].cpu().numpy().tobytes().hex(),
                "noise_seed_B": tensors["commitment_hash_B"].cpu().numpy().tobytes().hex(),
            },
            "plain_verify_nbits": verify_nbits,
            "share_verify_nbits": mining_job.akoya_share_difficulty,
        }
    return out


def json_safe(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def write_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    import torch
    from miner_base.akoya_pool_client import AkoyaMiningClient

    if args.boundary_hit_v1_dir and args.random_b:
        raise RuntimeError("--boundary-hit-v1-dir requires deterministic Akoya BSeed-backed B; do not combine it with --random-b")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    torch.cuda.set_device(args.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.device)

    out_dir = args.out_dir or Path("akoya_direct_runs") / f"direct_akoya_{utc_stamp()}"
    settings = make_settings(args)
    tensors = allocate_tensors(args, torch)
    summary: dict[str, Any] = {
        "schema": "direct_gpu_akoya_submit.v1",
        "started_at": utc_stamp(),
        "repo_root": str(REPO_ROOT),
        "submit_enabled": bool(args.submit),
        "force_target_max": bool(args.force_target_max),
        "settings": {
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "rank": args.rank,
            "rows": args.rows,
            "cols_count": len(args.cols),
            "tile_size_m": args.tile_size_m,
            "tile_size_n": args.tile_size_n,
            "tile_size_k": args.tile_size_k,
            "cluster_size_m": args.cluster_size_m,
            "cluster_size_n": args.cluster_size_n,
            "pipeline_stages": args.pipeline_stages,
            "swizzle": args.swizzle,
            "swizzle_n_maj": args.swizzle_n_maj,
            "proof_only": args.proof_only,
            "fast_sideband": args.fast_sideband,
            "native_sideband_fill_boundaries": args.native_sideband_fill_boundaries,
            "a_refresh_mode": args.a_refresh_mode,
            "a_nonce_row": args.a_nonce_row,
            "a_nonce_bytes": args.a_nonce_bytes,
            "a_initial_random": args.a_initial_random,
            "skip_denoising": args.skip_denoising,
            "boundary_hit_v1_dir": str(args.boundary_hit_v1_dir) if args.boundary_hit_v1_dir else None,
            "worker": settings.akoya_pool_worker,
            "wallet": settings.akoya_pool_wallet,
            "host": settings.akoya_pool_host,
            "port": settings.akoya_pool_port,
        },
        "attempts": [],
    }
    deadline = time.monotonic() + args.duration_seconds if args.duration_seconds else None
    exit_code = 3
    try:
        with AkoyaMiningClient(settings) as client:
            for attempt in range(1, args.max_attempts + 1):
                if deadline is not None and time.monotonic() >= deadline:
                    summary["status"] = "timeout"
                    break
                job = client.get_mining_info()
                started = time.perf_counter()
                record = run_noisy_once(args, tensors, job)
                record["attempt"] = attempt
                record["elapsed_s"] = time.perf_counter() - started
                record["akoya_job_uuid"] = job.akoya_job_uuid
                record["akoya_height"] = job.akoya_height
                record["akoya_network_nbits"] = job.akoya_network_nbits
                record["akoya_share_difficulty"] = job.akoya_share_difficulty
                if record.get("status") == "verified" and args.submit:
                    if args.force_target_max and not args.allow_forced_submit:
                        record["submission"] = {
                            "skipped": True,
                            "reason": "forced target canary never submits",
                        }
                    else:
                        try:
                            result = client.submit_plain_proof(
                                record["_plain_proof"],
                                job,
                                record["_opened"],
                            )
                        except Exception as exc:
                            record["submission_exception"] = {
                                "type": type(exc).__name__,
                                "message": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                            summary["attempts"].append(json_safe(record))
                            summary["status"] = "rejected_or_submit_failed"
                            exit_code = 2
                            write_summary(out_dir, summary)
                            break
                        record["submission"] = result
                        summary["attempts"].append(json_safe(record))
                        summary["status"] = "accepted" if result.get("accepted") else "rejected"
                        exit_code = 0 if result.get("accepted") else 2
                        write_summary(out_dir, summary)
                        break
                summary["attempts"].append(json_safe(record))
                write_summary(out_dir, summary)
                if record.get("status") == "verified" and not args.submit:
                    summary["status"] = "verified_no_submit"
                    exit_code = 0
                    write_summary(out_dir, summary)
                    break
            else:
                summary["status"] = "no_accept_after_attempts"
    except Exception as exc:
        summary["status"] = "failed_exception"
        summary["exception"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        exit_code = 1
    finally:
        summary["finished_at"] = utc_stamp()
        try:
            summary["gpu"] = {
                "name": torch.cuda.get_device_name(args.device),
                "capability": list(torch.cuda.get_device_capability(args.device)),
                "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(args.device)),
            }
        except Exception:
            pass
        write_summary(out_dir, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=AKOYA_M)
    parser.add_argument("--n", type=int, default=AKOYA_N)
    parser.add_argument("--k", type=int, default=AKOYA_K)
    parser.add_argument("--rank", type=int, default=AKOYA_RANK)
    parser.add_argument("--rows", type=int_list, default=list(AKOYA_ROWS))
    parser.add_argument("--cols", type=int_list, default=list(AKOYA_COLS))
    parser.add_argument("--tile-size-m", type=int, default=128)
    parser.add_argument("--tile-size-n", type=int, default=256)
    parser.add_argument("--tile-size-k", type=int, default=128)
    parser.add_argument("--cluster-size-m", type=int, default=1)
    parser.add_argument("--cluster-size-n", type=int, default=1)
    parser.add_argument("--pipeline-stages", type=int, default=3)
    parser.add_argument("--swizzle", type=int)
    parser.add_argument("--swizzle-n-maj", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--int32-noising-a", action="store_true")
    parser.add_argument("--int32-noising-b", action="store_true")
    parser.add_argument(
        "--torch-noising",
        action="store_true",
        help="Materialize noised operands with torch._int_mm and skip fused CUDA noising kernels.",
    )
    parser.add_argument("--proof-only", action="store_true")
    parser.add_argument(
        "--fast-sideband",
        action="store_true",
        help=(
            "Use the P1K176/P1K165 native sideband + two-phase finalizer path "
            "instead of the current legal noisy_gemm proof path. Requires "
            "--torch-noising and currently remains local/experimental until "
            "the submit path produces an accepted share."
        ),
    )
    parser.add_argument(
        "--native-sideband-fill-boundaries",
        type=int,
        default=16,
        help="Number of rank boundaries for the native sideband journal; default matches Akoya K/rank.",
    )
    parser.add_argument("--skip-denoising", action="store_true")
    parser.add_argument("--force-target-max", action="store_true")
    parser.add_argument(
        "--a-refresh-mode",
        choices=("full-random", "fixed", "nonce-prefix"),
        default="full-random",
        help=(
            "How to vary A per attempt. full-random preserves the old behavior; "
            "nonce-prefix initializes A once and mutates a small prefix so the "
            "commitment changes without a full-matrix torch.randint per attempt; "
            "fixed is diagnostic-only and repeats the same A."
        ),
    )
    parser.add_argument("--a-nonce-row", type=int, default=0)
    parser.add_argument("--a-nonce-bytes", type=int, default=DEFAULT_A_NONCE_BYTES)
    parser.add_argument(
        "--a-initial-random",
        action="store_true",
        help="For fixed/nonce-prefix modes, initialize the full A matrix randomly once instead of zero-filling it.",
    )
    parser.add_argument(
        "--save-proof-artifact",
        action="store_true",
        help=(
            "On a verified hit, embed header, PlainProof base64, deterministic A/B "
            "generation metadata, and opened indices in summary.json so a CPU "
            "proof-node reconstruction can be verified offline."
        ),
    )
    parser.add_argument(
        "--boundary-hit-v1-dir",
        type=Path,
        help=(
            "On each triggered candidate, write a BoundaryHitV1 frame into this "
            "directory immediately after index extraction and before local "
            "create_proof(). Incompatible with --random-b."
        ),
    )
    parser.add_argument("--random-b", action="store_true", help="Use random B instead of Akoya BSeed expansion")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--allow-forced-submit", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="pool.akoyapool.com")
    parser.add_argument("--port", type=int, default=3333)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--wallet", default="")
    parser.add_argument("--worker", default=f"codex-direct-akoya-{utc_stamp()}")
    parser.add_argument("--gpu-name", default="Codex H200 Direct")
    parser.add_argument("--version", default="codex-p1k132")
    parser.add_argument("--git-sha", default="codex-p1k132")
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
