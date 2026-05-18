#!/usr/bin/env python3
"""Akoya BSeed fixture regression.

Run from repo root:

    PYTHONPATH=/home/bereket/.local/lib/python3.12/site-packages \
      .venv/bin/python tools/akoya_bridge/test_bseed_expansion.py
"""

from __future__ import annotations

from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))

from akoya_bseed import hash_b_for_bseed, job_key_for_share, opened_b_from_leaf_indices  # noqa: E402
from akoya_protocol import PlainProofShare, unpack_payload  # noqa: E402


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
    share = PlainProofShare.from_fields(fields)
    share.validate()
    config = share.mining_config
    n = share.b_proof.total_leaves * 1024 // config.common_dim
    k = config.common_dim
    job_key = job_key_for_share(share.header_bytes, share.mining_config_bytes)

    hash_b = hash_b_for_bseed(share.seed_hash, n, k, job_key)
    opened_b = opened_b_from_leaf_indices(share.seed_hash, n, k, share.b_proof.leaf_indices)

    assert hash_b == share.hash_b, (hash_b.hex(), share.hash_b.hex())
    assert opened_b == share.opened_b
    print("akoya BSeed expansion fixture: OK")
    print(f"seed_hash: {share.seed_hash.hex()}")
    print(f"job_key: {job_key.hex()}")
    print(f"hash_b: {hash_b.hex()}")
    print(f"opened_b_len: {len(opened_b)}")


if __name__ == "__main__":
    main()
