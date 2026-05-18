#!/usr/bin/env python3
"""Minimal private Akoya-compatible soft pool for deterministic submit tests.

The server accepts one miner connection, sends a register ack and a low
difficulty job, verifies the submitted type-3 PlainProofShare locally, then
returns a type-4 ShareResult.  It is intended for private AKO-012 testing only;
it is not a public pool emulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
import tempfile
import time
import uuid


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

from akoya_protocol import (  # noqa: E402
    TYPE_JOB_ASSIGNMENT,
    TYPE_REGISTER,
    TYPE_REGISTER_ACK,
    TYPE_SHARE_RESULT,
    JobAssignment,
    PlainProofShare,
    pack_frame,
    parse_message,
    read_frame,
)
from verify_captured_share import verify_fixture  # noqa: E402


DEFAULT_SOFT_NBITS = 0x1E0FFFFF


def _artifact_job(summary: Path) -> tuple[bytes, bytes]:
    data = json.loads(summary.read_text(encoding="utf-8"))
    for attempt in data.get("attempts", []):
        artifact = attempt.get("proof_artifact")
        if artifact:
            return (
                bytes.fromhex(artifact["incomplete_header_bytes_hex"]),
                bytes.fromhex(artifact["b_generation"]["seed_hash"]),
            )
    raise SystemExit("summary has no proof_artifact")


def _verify_share_frame(frame: bytes) -> tuple[bool, str, dict]:
    with tempfile.NamedTemporaryFile(suffix=".msgpack") as tmp:
        tmp.write(frame)
        tmp.flush()
        try:
            result = verify_fixture(Path(tmp.name))
            return True, str(result["message"]), result
        except BaseException as exc:  # verify_fixture raises SystemExit on rejection.
            return False, str(exc), {}


def serve(args: argparse.Namespace) -> dict:
    if args.artifact_summary:
        header_bytes, seed_hash = _artifact_job(args.artifact_summary)
    else:
        header_bytes = bytes.fromhex(args.header_hex) if args.header_hex else bytes(76)
        seed_hash = bytes.fromhex(args.seed_hash) if args.seed_hash else bytes(32)
    if len(header_bytes) != 76:
        raise SystemExit("header must be 76 bytes")
    if len(seed_hash) != 32:
        raise SystemExit("seed hash must be 32 bytes")

    result = {
        "schema": "akoya_soft_pool_server.v1",
        "host": args.host,
        "port": args.port,
        "share_difficulty": args.share_difficulty,
        "network_nbits": args.network_nbits,
        "started_at": time.time(),
        "accepted": False,
    }

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        server.settimeout(args.timeout)
        conn, addr = server.accept()
        with conn:
            conn.settimeout(args.timeout)
            result["client_addr"] = list(addr)
            type_code, fields = read_frame(conn)
            if type_code != TYPE_REGISTER:
                raise SystemExit(f"expected register type 0, got {type_code}")
            register = parse_message(type_code, fields)
            result["register"] = {
                "worker": getattr(register, "worker", ""),
                "wallet": getattr(register, "wallet", ""),
                "gpu_name": getattr(register, "gpu_name", ""),
            }
            pool_uuid = str(uuid.uuid4())
            miner_id = str(uuid.uuid4())
            conn.sendall(pack_frame(TYPE_REGISTER_ACK, [True, pool_uuid, args.share_difficulty, miner_id]))
            job_uuid = str(uuid.uuid4())
            conn.sendall(
                pack_frame(
                    TYPE_JOB_ASSIGNMENT,
                    [
                        job_uuid,
                        header_bytes,
                        args.share_difficulty,
                        args.height,
                        seed_hash,
                        bytes(32),
                        args.network_nbits,
                    ],
                )
            )
            result["job"] = {
                "job_uuid": job_uuid,
                "height": args.height,
                "seed_hash": seed_hash.hex(),
            }
            while True:
                type_code, fields = read_frame(conn)
                if type_code == TYPE_JOB_ASSIGNMENT:
                    continue
                if type_code != 3:
                    raise SystemExit(f"expected type-3 share, got {type_code}")
                share = PlainProofShare.from_fields(fields)
                share.validate()
                rejection_reasons: list[str] = []
                if share.share_difficulty != args.share_difficulty:
                    rejection_reasons.append(
                        f"share difficulty {share.share_difficulty} does not match assigned difficulty {args.share_difficulty}"
                    )
                if share.header_bytes != header_bytes:
                    rejection_reasons.append("share header does not match assigned job header")
                if share.seed_hash != seed_hash:
                    rejection_reasons.append("share seed_hash does not match assigned job seed_hash")
                if rejection_reasons:
                    ok = False
                    message = "; ".join(rejection_reasons)
                    verification = {}
                    outcome = 1
                    conn.sendall(pack_frame(TYPE_SHARE_RESULT, [share.share_id, outcome, message]))
                    result.update(
                        {
                            "share_id": share.share_id,
                            "accepted": ok,
                            "message": message,
                            "verification": verification,
                            "finished_at": time.time(),
                        }
                    )
                    break
                frame = pack_frame(type_code, fields)
                ok, message, verification = _verify_share_frame(frame)
                outcome = 0 if ok else 1
                conn.sendall(pack_frame(TYPE_SHARE_RESULT, [share.share_id, outcome, message]))
                result.update(
                    {
                        "share_id": share.share_id,
                        "accepted": ok,
                        "message": message,
                        "verification": verification,
                        "finished_at": time.time(),
                    }
                )
                break

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3334)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--share-difficulty", type=lambda x: int(x, 0), default=DEFAULT_SOFT_NBITS)
    parser.add_argument("--network-nbits", type=lambda x: int(x, 0), default=0x207FFFFF)
    parser.add_argument("--height", type=int, default=1)
    parser.add_argument("--artifact-summary", type=Path)
    parser.add_argument("--header-hex")
    parser.add_argument("--seed-hash")
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = serve(parse_args(argv))
    return 0 if result.get("accepted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
