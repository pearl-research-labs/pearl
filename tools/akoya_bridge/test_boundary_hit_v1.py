#!/usr/bin/env python3
"""Direct tests for BoundaryHitV1 framing and mining-config expansion."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from boundary_hit_v1 import (  # noqa: E402
    A_MODE_NONCE_PREFIX,
    FRAME_SIZE,
    PAYLOAD_SIZE,
    BoundaryHitV1,
    BoundaryHitV1Error,
)


FIXTURE_JOB_UUID = "12345678-1234-5678-9abc-def012345678"
FIXTURE_HEADER = bytes(range(76))
FIXTURE_SEED_HASH = bytes(range(32))
FIXTURE_SHARE_NBITS = 0x1A300000
FIXTURE_NETWORK_NBITS = 0x1807E3C1
FIXTURE_M = 8192
FIXTURE_N = 32768
FIXTURE_MINING_CONFIG_HEX = (
    "00080000800000000701000000000001031f0000"
    "0000000000000000000000000000000000000000000000000000000000000000"
)
FIXTURE_T_ROWS = 1601
FIXTURE_T_COLS = 4864
FIXTURE_A_NONCE_ROW = 0
FIXTURE_A_NONCE_BYTES = 12
FIXTURE_A_NONCE_COUNTER = 123456789
FIXTURE_FRAME_SHA256 = "304f03dc25c4889aee9d664015b8861c6807f7191462990a0e3482a6a0b4525a"


def fixture_boundary() -> BoundaryHitV1:
    return BoundaryHitV1(
        schema_version=1,
        job_uuid=FIXTURE_JOB_UUID,
        incomplete_header_bytes=FIXTURE_HEADER,
        seed_hash=FIXTURE_SEED_HASH,
        share_verify_nbits=FIXTURE_SHARE_NBITS,
        network_nbits=FIXTURE_NETWORK_NBITS,
        m=FIXTURE_M,
        n=FIXTURE_N,
        mining_config_bytes=bytes.fromhex(FIXTURE_MINING_CONFIG_HEX),
        t_rows=FIXTURE_T_ROWS,
        t_cols=FIXTURE_T_COLS,
        a_mode=A_MODE_NONCE_PREFIX,
        a_initial_random=False,
        a_nonce_row=FIXTURE_A_NONCE_ROW,
        a_nonce_bytes=FIXTURE_A_NONCE_BYTES,
        a_nonce_counter=FIXTURE_A_NONCE_COUNTER,
    ).validate()


def fixture_attempt_record() -> dict[str, object]:
    return {
        "akoya_job_uuid": FIXTURE_JOB_UUID,
        "akoya_network_nbits": FIXTURE_NETWORK_NBITS,
        "akoya_share_difficulty": FIXTURE_SHARE_NBITS,
        "proof_artifact": {
            "schema": "akoya_split_boundary_proof_artifact.v1",
            "incomplete_header_bytes_hex": FIXTURE_HEADER.hex(),
            "mining_config_bytes_hex": FIXTURE_MINING_CONFIG_HEX,
            "m": FIXTURE_M,
            "n": FIXTURE_N,
            "k": 2048,
            "rank": 128,
            "indices": {
                "A_row_indices": [1601, 1609],
                "B_column_indices": [
                    4864,
                    4865,
                    4872,
                    4873,
                    4880,
                    4881,
                    4888,
                    4889,
                    4896,
                    4897,
                    4904,
                    4905,
                    4912,
                    4913,
                    4920,
                    4921,
                    4928,
                    4929,
                    4936,
                    4937,
                    4944,
                    4945,
                    4952,
                    4953,
                    4960,
                    4961,
                    4968,
                    4969,
                    4976,
                    4977,
                    4984,
                    4985,
                    4992,
                    4993,
                    5000,
                    5001,
                    5008,
                    5009,
                    5016,
                    5017,
                    5024,
                    5025,
                    5032,
                    5033,
                    5040,
                    5041,
                    5048,
                    5049,
                    5056,
                    5057,
                    5064,
                    5065,
                    5072,
                    5073,
                    5080,
                    5081,
                    5088,
                    5089,
                    5096,
                    5097,
                    5104,
                    5105,
                    5112,
                    5113,
                ],
            },
            "a_generation": {
                "mode": "nonce-prefix",
                "initial_random": False,
                "nonce_row": FIXTURE_A_NONCE_ROW,
                "nonce_bytes": FIXTURE_A_NONCE_BYTES,
                "nonce_counter": FIXTURE_A_NONCE_COUNTER,
            },
            "b_generation": {
                "mode": "akoya_bseed",
                "seed_hash": FIXTURE_SEED_HASH.hex(),
            },
            "share_verify_nbits": FIXTURE_SHARE_NBITS,
        },
    }


def test_roundtrip_and_field_sizes() -> None:
    boundary = fixture_boundary()
    payload = boundary.to_payload()
    frame = boundary.to_frame()

    assert len(payload) == PAYLOAD_SIZE
    assert len(frame) == FRAME_SIZE
    assert int.from_bytes(frame[:4], "big") == PAYLOAD_SIZE
    assert BoundaryHitV1.from_payload(payload) == boundary
    assert BoundaryHitV1.from_frame(frame) == boundary
    assert hashlib.sha256(frame).hexdigest() == FIXTURE_FRAME_SHA256


def test_mining_config_expansion() -> None:
    boundary = fixture_boundary()
    assert boundary.k == 2048
    assert boundary.rank == 128
    assert boundary.a_row_indices == [1601, 1609]
    assert boundary.b_column_indices[:8] == [4864, 4865, 4872, 4873, 4880, 4881, 4888, 4889]
    assert len(boundary.b_column_indices) == 64


def test_from_proof_artifact() -> None:
    boundary = fixture_boundary()
    rebuilt = BoundaryHitV1.from_proof_artifact(fixture_attempt_record())
    assert rebuilt == boundary
    assert rebuilt.to_frame() == boundary.to_frame()


def test_malformed_frame_rejection() -> None:
    frame = bytearray(fixture_boundary().to_frame())
    frame[3] -= 1
    try:
        BoundaryHitV1.from_frame(bytes(frame))
    except BoundaryHitV1Error:
        pass
    else:
        raise AssertionError("expected BoundaryHitV1Error for bad frame prefix")

    try:
        BoundaryHitV1.from_payload(b"\x00" * (PAYLOAD_SIZE - 1))
    except BoundaryHitV1Error:
        pass
    else:
        raise AssertionError("expected BoundaryHitV1Error for truncated payload")


def main() -> None:
    test_roundtrip_and_field_sizes()
    test_mining_config_expansion()
    test_from_proof_artifact()
    test_malformed_frame_rejection()
    print("boundary_hit_v1 tests: OK")


if __name__ == "__main__":
    main()
