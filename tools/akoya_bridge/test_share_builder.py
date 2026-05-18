#!/usr/bin/env python3
"""Akoya share-builder test that requires the Pearl Python extension.

Run from repo root:

    PYTHONPATH=/home/bereket/.local/lib/python3.12/site-packages \
      .venv/bin/python tools/akoya_bridge/test_share_builder.py
"""

from __future__ import annotations

from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))

from akoya_protocol import PlainProofShare, unpack_payload  # noqa: E402
from akoya_share_builder import build_share_from_pearl_plain_proof  # noqa: E402
from verify_captured_share import build_local_plain_proof  # noqa: E402

import pearl_mining as pm  # noqa: E402


FIXTURE = (
    REPO_ROOT
    / "codex_context"
    / "pearl-2026-05-15-current"
    / "akoya_recon"
    / "capture_20260515T1430Z"
    / "samples"
    / "c2s_first_plainproofshare.msgpack"
)


def main() -> None:
    type_code, fields = unpack_payload(FIXTURE.read_bytes())
    assert type_code == 3
    captured = PlainProofShare.from_fields(fields)
    captured.validate()

    header = pm.IncompleteBlockHeader.from_bytes(captured.header_bytes)
    local_plain_proof = build_local_plain_proof(pm, captured)
    rebuilt = build_share_from_pearl_plain_proof(
        pm=pm,
        plain_proof=local_plain_proof,
        header=header,
        header_bytes=captured.header_bytes,
        seed_hash=captured.seed_hash,
        share_difficulty=captured.share_difficulty,
        mining_config_bytes=captured.mining_config_bytes,
        share_id=captured.share_id,
    )

    assert rebuilt.to_fields() == captured.to_fields()
    assert rebuilt.validate()["share_id"] == captured.share_id
    print("akoya share builder fixture roundtrip: OK")


if __name__ == "__main__":
    main()

