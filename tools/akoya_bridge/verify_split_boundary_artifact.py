#!/usr/bin/env python3
"""Verify a split-boundary proof artifact from direct_gpu_akoya_submit.py.

The check intentionally rebuilds PlainProof from deterministic boundary
metadata, not from GPU accumulator/receipt/transcript state.
"""

from __future__ import annotations

import argparse
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


def _int7_counter_values(counter: int, width: int) -> list[int]:
    values: list[int] = []
    x = int(counter)
    for _ in range(width):
        values.append((x % 127) - 63)
        x //= 127
    return values


def _find_artifact(summary: dict[str, Any]) -> dict[str, Any]:
    for attempt in summary.get("attempts", []):
        artifact = attempt.get("proof_artifact")
        if artifact:
            return artifact
    raise SystemExit("no proof_artifact found; rerun direct_gpu_akoya_submit.py with --save-proof-artifact")


def reconstruct(args: argparse.Namespace) -> dict[str, Any]:
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    artifact = _find_artifact(summary)

    import torch
    from akoya_bseed import expand_bseed_matrix
    from miner_base.block_submission import create_proof
    from pearl_gateway.comm.dataclasses import CommitmentHash, OpenedBlockInfo
    import pearl_mining

    m = int(artifact["m"])
    n = int(artifact["n"])
    k = int(artifact["k"])
    rank = int(artifact["rank"])

    a_gen = artifact["a_generation"]
    if a_gen.get("initial_random"):
        raise SystemExit("cannot reconstruct A: artifact used a_initial_random")
    if a_gen.get("mode") not in {"fixed", "nonce-prefix"}:
        raise SystemExit(f"cannot reconstruct A mode: {a_gen.get('mode')}")

    A = torch.zeros((m, k), dtype=torch.int8)
    if a_gen.get("mode") == "nonce-prefix":
        counter = a_gen.get("nonce_counter")
        if counter is None:
            raise SystemExit("nonce-prefix artifact missing nonce_counter")
        row = int(a_gen["nonce_row"])
        width = int(a_gen["nonce_bytes"])
        A[row, :width] = torch.tensor(_int7_counter_values(int(counter), width), dtype=torch.int8)

    b_gen = artifact["b_generation"]
    if b_gen.get("mode") != "akoya_bseed":
        raise SystemExit(f"cannot reconstruct B mode: {b_gen.get('mode')}")
    seed_hash = bytes.fromhex(b_gen["seed_hash"])
    B_bytes = expand_bseed_matrix(seed_hash, n, k)
    B_t = torch.frombuffer(bytearray(B_bytes), dtype=torch.int8).reshape(n, k)

    opened = OpenedBlockInfo(
        A_row_indices=[int(x) for x in artifact["indices"]["A_row_indices"]],
        B_column_indices=[int(x) for x in artifact["indices"]["B_column_indices"]],
        A=A,
        B_t=B_t,
        commitment_hash=CommitmentHash(
            noise_seed_A=bytes.fromhex(artifact["commitment_hash"]["noise_seed_A"]),
            noise_seed_B=bytes.fromhex(artifact["commitment_hash"]["noise_seed_B"]),
        ),
        noise_rank=rank,
    )

    header_bytes = bytes.fromhex(artifact["incomplete_header_bytes_hex"])
    rebuilt = create_proof(opened, header_bytes)
    rebuilt_base64 = rebuilt.to_base64()
    expected_base64 = artifact["plain_proof_base64"]
    if rebuilt_base64 != expected_base64:
        raise SystemExit(
            "rebuilt PlainProof base64 mismatch: "
            f"expected {len(expected_base64)} bytes, got {len(rebuilt_base64)} bytes"
        )

    header = pearl_mining.IncompleteBlockHeader.from_bytes(header_bytes)
    nbits = int(artifact["plain_verify_nbits"])
    if hasattr(pearl_mining, "verify_plain_proof_with_nbits"):
        ok, message = pearl_mining.verify_plain_proof_with_nbits(header, rebuilt, nbits)
    else:
        ok, message = pearl_mining.verify_plain_proof(header, rebuilt)
    if not ok:
        raise SystemExit(f"rebuilt proof rejected: {message}")

    result = {
        "ok": True,
        "message": message,
        "summary": str(args.summary),
        "m": m,
        "n": n,
        "k": k,
        "rank": rank,
        "a_rows": len(opened.A_row_indices),
        "b_cols": len(opened.B_column_indices),
        "plain_proof_base64_bytes": len(rebuilt_base64),
        "nbits": nbits,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="summary.json containing proof_artifact")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    reconstruct(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
