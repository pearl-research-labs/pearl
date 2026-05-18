#!/usr/bin/env python3
"""Stage Akoya type-3 share bytes from a split-boundary proof artifact.

This is a no-GPU preflight for the accepted-share path.  It consumes a
``direct_gpu_akoya_submit.py --save-proof-artifact`` summary, builds the exact
Akoya MessagePack type-3 ``PlainProofShare`` frame shape, round-trips it through
the protocol parser, and saves byte hashes for later submit attempts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]

for path in reversed(
    (
        THIS_DIR,
        REPO_ROOT / "py-pearl-mining",
        REPO_ROOT / "miner" / "miner-utils" / "src",
        REPO_ROOT / "miner" / "miner-base" / "src",
        REPO_ROOT / "miner" / "pearl-gateway" / "src",
    )
):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def _find_artifact(summary: dict[str, Any]) -> dict[str, Any]:
    for attempt in summary.get("attempts", []):
        artifact = attempt.get("proof_artifact")
        if artifact:
            return artifact
    raise SystemExit("no proof_artifact found; rerun direct_gpu_akoya_submit.py with --save-proof-artifact")


def stage(args: argparse.Namespace) -> dict[str, Any]:
    import pearl_mining
    from akoya_protocol import PlainProofShare, load_frame
    from akoya_share_builder import build_share_from_decoded_plain_proof, canonical_jackpot_fields
    from plain_proof_bincode import decode_plain_proof_base64

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifact = _find_artifact(summary)

    out_dir = args.out_dir or summary_path.parent / "akoya_type3_stage"
    out_dir.mkdir(parents=True, exist_ok=True)

    header_bytes = bytes.fromhex(artifact["incomplete_header_bytes_hex"])
    seed_hash = bytes.fromhex(artifact["b_generation"]["seed_hash"])
    mining_config_bytes = bytes.fromhex(artifact["mining_config_bytes_hex"])
    share_difficulty = (
        int(args.share_difficulty, 0)
        if args.share_difficulty is not None
        else int(artifact["share_verify_nbits"])
    )
    plain_verify_nbits = int(artifact["plain_verify_nbits"])

    plain_proof = pearl_mining.PlainProof.from_base64(artifact["plain_proof_base64"])
    header = pearl_mining.IncompleteBlockHeader.from_bytes(header_bytes)

    plain_ok, plain_message = pearl_mining.verify_plain_proof_with_nbits(
        header,
        plain_proof,
        plain_verify_nbits,
    )
    share_ok, share_message = pearl_mining.verify_plain_proof_with_nbits(
        header,
        plain_proof,
        share_difficulty,
    )
    compact_result, claimed_hash = canonical_jackpot_fields(pearl_mining, header, plain_proof)
    decoded = decode_plain_proof_base64(artifact["plain_proof_base64"])
    share = build_share_from_decoded_plain_proof(
        plain_proof=decoded,
        header_bytes=header_bytes,
        seed_hash=seed_hash,
        share_difficulty=share_difficulty,
        mining_config_bytes=mining_config_bytes,
        compact_result=compact_result,
        claimed_hash=claimed_hash,
        share_id=args.share_id,
    )
    validation = share.validate()
    frame = share.frame()

    frame_path = out_dir / "type3_plain_proof_share.msgpack"
    frame_path.write_bytes(frame)
    parsed = load_frame(frame_path)
    if not isinstance(parsed, PlainProofShare):
        raise SystemExit(f"roundtrip parsed unexpected type: {type(parsed).__name__}")
    roundtrip_validation = parsed.validate()
    if parsed.to_fields() != share.to_fields():
        raise SystemExit("roundtrip fields changed")

    meta = {
        "schema": "akoya_type3_stage.v1",
        "summary": str(summary_path),
        "frame_path": str(frame_path),
        "frame_sha256": hashlib.sha256(frame).hexdigest(),
        "frame_bytes": len(frame),
        "share_id": share.share_id,
        "share_difficulty": share_difficulty,
        "plain_verify": {
            "nbits": plain_verify_nbits,
            "valid": bool(plain_ok),
            "message": plain_message,
        },
        "share_verify": {
            "nbits": share_difficulty,
            "valid": bool(share_ok),
            "message": share_message,
        },
        "header_bytes": len(header_bytes),
        "seed_hash": seed_hash.hex(),
        "mining_config_bytes": len(mining_config_bytes),
        "plain_proof_base64_bytes": len(artifact["plain_proof_base64"]),
        "compact_result_bytes": len(compact_result),
        "claimed_hash": claimed_hash.hex(),
        "validation": validation,
        "roundtrip_validation": roundtrip_validation,
        "note": (
            "This stages serializer/protocol bytes only. A public-pool accepted "
            "share still requires share_verify.valid=true for the live job "
            "difficulty and a type-4 accepted response."
        ),
    }
    (out_dir / "type3_plain_proof_share.metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2, sort_keys=True))
    return meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="summary.json containing proof_artifact")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--share-id", default="00000000-0000-4000-8000-000000000005")
    parser.add_argument(
        "--share-difficulty",
        help=(
            "Override the type-3 share difficulty nbits, e.g. 0x207fffff for "
            "a private soft-tester. Defaults to the artifact's live share nbits."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    stage(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
