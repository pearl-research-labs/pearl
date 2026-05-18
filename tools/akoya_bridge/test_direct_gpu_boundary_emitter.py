#!/usr/bin/env python3
"""No-GPU tests for the direct miner BoundaryHitV1 emitter helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from direct_gpu_akoya_submit import (  # noqa: E402
    build_boundary_hit_v1,
    emit_boundary_hit_v1_from_attempt_record,
)
from test_boundary_hit_v1 import (  # noqa: E402
    fixture_attempt_record,
    fixture_boundary,
)


def test_build_boundary_hit_v1_matches_fixture() -> None:
    attempt = fixture_attempt_record()
    artifact = attempt["proof_artifact"]
    boundary = build_boundary_hit_v1(
        job_uuid=str(attempt["akoya_job_uuid"]),
        incomplete_header_bytes=bytes.fromhex(str(artifact["incomplete_header_bytes_hex"])),
        seed_hash=bytes.fromhex(str(artifact["b_generation"]["seed_hash"])),
        share_verify_nbits=int(attempt["akoya_share_difficulty"]),
        network_nbits=int(attempt["akoya_network_nbits"]),
        m=int(artifact["m"]),
        n=int(artifact["n"]),
        mining_config_bytes=bytes.fromhex(str(artifact["mining_config_bytes_hex"])),
        a_row_indices=[int(value) for value in artifact["indices"]["A_row_indices"]],
        b_column_indices=[int(value) for value in artifact["indices"]["B_column_indices"]],
        a_refresh_mode=str(artifact["a_generation"]["mode"]),
        a_initial_random=bool(artifact["a_generation"]["initial_random"]),
        a_nonce_row=int(artifact["a_generation"]["nonce_row"]),
        a_nonce_bytes=int(artifact["a_generation"]["nonce_bytes"]),
        a_nonce_counter=int(artifact["a_generation"]["nonce_counter"]),
    )
    assert boundary == fixture_boundary()


def test_emit_boundary_hit_v1_from_attempt_record() -> None:
    expected = fixture_boundary().to_frame()
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = emit_boundary_hit_v1_from_attempt_record(
            fixture_attempt_record(),
            Path(tmp_dir),
            emission_index=7,
        )
        path = Path(result["path"])
        frame = path.read_bytes()
        assert path.name == "boundary_hit_v1_000007_12345678-1234-5678-9abc-def012345678.frame"
        assert frame == expected
        assert result["sha256"] == hashlib.sha256(frame).hexdigest()
        assert result["frame_bytes"] == len(expected)
        assert result["payload_bytes"] == len(expected) - 4
        assert result["emission_index"] == 7


def main() -> None:
    test_build_boundary_hit_v1_matches_fixture()
    test_emit_boundary_hit_v1_from_attempt_record()
    print("direct_gpu_boundary_emitter tests: OK")


if __name__ == "__main__":
    main()
