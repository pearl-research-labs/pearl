#!/usr/bin/env python3
"""Soft-accept a staged Akoya type-3 frame after local proof verification.

This is a deterministic protocol harness, not a public pool claim.  It verifies
the type-3 ``PlainProofShare`` with the share difficulty embedded in the frame,
then emits a type-4 ``ShareResult`` frame with outcome code 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


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

from akoya_protocol import TYPE_SHARE_RESULT, PlainProofShare, pack_frame, parse_frame  # noqa: E402
from verify_captured_share import verify_fixture  # noqa: E402


def soft_accept(args: argparse.Namespace) -> dict:
    frame_path = Path(args.frame)
    request = frame_path.read_bytes()
    parsed = parse_frame(request)
    if not isinstance(parsed, PlainProofShare):
        raise SystemExit(f"expected type-3 PlainProofShare frame, got {type(parsed).__name__}")

    verification = verify_fixture(frame_path)
    response = pack_frame(TYPE_SHARE_RESULT, [parsed.share_id, 0, args.message])
    response_path = args.out or frame_path.with_name("soft_share_result_accepted.msgpack")
    response_path.write_bytes(response)
    parsed_response = parse_frame(response)

    result = {
        "schema": "akoya_soft_accept_type3.v1",
        "request_frame": str(frame_path),
        "request_sha256": hashlib.sha256(request).hexdigest(),
        "request_bytes": len(request),
        "response_frame": str(response_path),
        "response_sha256": hashlib.sha256(response).hexdigest(),
        "response_bytes": len(response),
        "share_id": parsed.share_id,
        "share_difficulty": parsed.share_difficulty,
        "accepted": bool(getattr(parsed_response, "accepted", False)),
        "outcome_code": getattr(parsed_response, "outcome_code", None),
        "message": getattr(parsed_response, "message", ""),
        "verification": verification,
    }
    meta_path = response_path.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--message", default="accepted")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = soft_accept(parse_args(argv))
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
