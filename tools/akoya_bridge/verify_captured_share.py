#!/usr/bin/env python3
"""Verify a captured Akoya accepted share with local Pearl proof APIs.

This is the second local gate for P1K-131. It proves the captured type-3
PlainProofShare can be reconstructed as a local pearl_mining.PlainProof and
accepted by verify_plain_proof_with_nbits when using the pool's share
difficulty field.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import struct

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))

from akoya_protocol import PlainProofShare, unpack_frame, unpack_payload  # noqa: E402


DEFAULT_FIXTURE = (
    REPO_ROOT
    / "codex_context"
    / "pearl-2026-05-15-current"
    / "akoya_recon"
    / "capture_20260515T1430Z"
    / "samples"
    / "c2s_first_plainproofshare.msgpack"
)


def _import_pearl_modules():
    try:
        import blake3  # type: ignore[import-not-found]
        import pearl_mining as pm  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Pearl Python dependencies. Run with the repo venv, for example:\n"
            "PYTHONPATH=/home/bereket/.local/lib/python3.12/site-packages "
            ".venv/bin/python tools/akoya_bridge/verify_captured_share.py"
        ) from exc
    return blake3, pm


def build_local_plain_proof(pm, share: PlainProofShare):
    config = share.mining_config
    rows = config.rows_pattern.indices_with_offset(share.t_rows)
    cols = config.cols_pattern.indices_with_offset(share.t_cols)
    m = share.a_proof.total_leaves * 1024 // config.common_dim
    n = share.b_proof.total_leaves * 1024 // config.common_dim

    a_merkle = pm.MerkleProof(
        list(share.a_proof.leaf_data),
        list(share.a_proof.leaf_indices),
        share.hash_a,
        list(share.a_proof.siblings),
        share.a_proof.total_leaves,
    )
    b_merkle = pm.MerkleProof(
        list(share.b_proof.leaf_data),
        list(share.b_proof.leaf_indices),
        share.hash_b,
        list(share.b_proof.siblings),
        share.b_proof.total_leaves,
    )
    return pm.PlainProof(
        m,
        n,
        config.common_dim,
        config.rank,
        pm.MatrixMerkleProof(a_merkle, rows),
        pm.MatrixMerkleProof(b_merkle, cols),
    )


def verify_fixture(path: Path) -> dict[str, str | int | bool]:
    blake3, pm = _import_pearl_modules()
    raw = path.read_bytes()
    if len(raw) >= 4 and int.from_bytes(raw[:4], "big") == len(raw) - 4:
        type_code, fields = unpack_frame(raw)
    else:
        type_code, fields = unpack_payload(raw)
    if type_code != 3:
        raise SystemExit(f"expected type 3 PlainProofShare, got type {type_code}")

    share = PlainProofShare.from_fields(fields)
    share.validate()

    header = pm.IncompleteBlockHeader.from_bytes(share.header_bytes)
    plain_proof = build_local_plain_proof(pm, share)
    ok, message = pm.verify_plain_proof_with_nbits(header, plain_proof, share.share_difficulty)
    if not ok:
        raise SystemExit(f"local proof rejected: {message}")

    diagnostic = pm.diagnostic_plain_proof_jackpot_controls(header, plain_proof)
    jackpot_words = [int(word) for word in diagnostic[0]]
    jackpot_bytes = b"".join(struct.pack("<I", word) for word in jackpot_words)
    jackpot_hash = bytes.fromhex(diagnostic[5])
    if jackpot_bytes != share.compact_result:
        raise SystemExit("field6 does not match canonical jackpot bytes")
    if jackpot_hash != share.claimed_hash:
        raise SystemExit("field7 does not match canonical jackpot hash")

    job_key = blake3.blake3(share.header_bytes + share.mining_config_bytes).digest()
    b_noise_seed = blake3.blake3(job_key + share.hash_b).digest()
    a_noise_seed = blake3.blake3(b_noise_seed + share.hash_a).digest()
    config = share.mining_config

    return {
        "ok": True,
        "message": message,
        "share_id": share.share_id,
        "m": share.a_proof.total_leaves * 1024 // config.common_dim,
        "n": share.b_proof.total_leaves * 1024 // config.common_dim,
        "k": config.common_dim,
        "rank": config.rank,
        "share_difficulty": share.share_difficulty,
        "field6_matches_canonical_jackpot": True,
        "field7_matches_canonical_hash": True,
        "hash_a": share.hash_a.hex(),
        "hash_b": share.hash_b.hex(),
        "job_key": job_key.hex(),
        "b_noise_seed": b_noise_seed.hex(),
        "a_noise_seed": a_noise_seed.hex(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    args = parser.parse_args()
    result = verify_fixture(args.fixture)
    print("local Akoya share proof verifies:", result["message"])
    for key in (
        "share_id",
        "m",
        "n",
        "k",
        "rank",
        "share_difficulty",
        "field6_matches_canonical_jackpot",
        "field7_matches_canonical_hash",
        "hash_a",
        "hash_b",
        "job_key",
        "b_noise_seed",
        "a_noise_seed",
    ):
        print(f"{key}: {result[key]}")


if __name__ == "__main__":
    main()
