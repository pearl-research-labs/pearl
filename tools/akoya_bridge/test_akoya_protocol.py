#!/usr/bin/env python3
"""Fixture tests for the Akoya pool protocol helpers.

Run from repo root:

    python3 tools/akoya_bridge/test_akoya_protocol.py
"""

from __future__ import annotations

from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))

from akoya_protocol import (  # noqa: E402
    HEADER_SIZE,
    MINING_CONFIG_SIZE,
    TYPE_JOB_ASSIGNMENT,
    TYPE_PLAIN_PROOF_SHARE,
    TYPE_REGISTER,
    TYPE_REGISTER_ACK,
    TYPE_SHARE_RESULT,
    JobAssignment,
    PlainProofShare,
    RegisterAck,
    RegisterMiner,
    ShareResult,
    parse_frame,
    parse_payload,
    pack_frame,
    unpack_frame,
    unpack_payload,
)


SAMPLES = (
    REPO_ROOT
    / "codex_context"
    / "pearl-2026-05-15-current"
    / "akoya_recon"
    / "capture_20260515T1430Z"
    / "samples"
)


def _load(name: str) -> bytes:
    return (SAMPLES / name).read_bytes()


def test_frame_prefixes() -> None:
    for frame_path in sorted(SAMPLES.glob("*.frame")):
        frame = frame_path.read_bytes()
        payload_len = int.from_bytes(frame[:4], "big")
        assert payload_len == len(frame) - 4, frame_path.name
        msg_from_frame = unpack_frame(frame)
        msg_from_payload = unpack_payload(frame[4:])
        assert msg_from_frame == msg_from_payload


def test_register_and_ack() -> None:
    register = parse_payload(_load("c2s_register.msgpack"))
    assert isinstance(register, RegisterMiner)
    assert register.wallet_address.startswith("prl1")
    assert register.common_dim == 2048
    frame = pack_frame(TYPE_REGISTER, register.to_fields())
    type_code, fields = unpack_frame(frame)
    assert type_code == TYPE_REGISTER
    assert RegisterMiner.from_fields(fields) == register

    ack = parse_payload(_load("s2c_register_ack.msgpack"))
    assert isinstance(ack, RegisterAck)
    assert ack.accepted is True
    assert ack.share_difficulty == 439353344


def test_job_assignment() -> None:
    job = parse_payload(_load("s2c_job_assignment.msgpack"))
    assert isinstance(job, JobAssignment)
    assert len(job.header_bytes) == HEADER_SIZE
    assert len(job.seed_hash) == 32
    assert job.share_difficulty == 439353344
    assert job.network_nbits == 403196433
    type_code, _ = unpack_payload(_load("s2c_job_assignment.msgpack"))
    assert type_code == TYPE_JOB_ASSIGNMENT


def test_share_result() -> None:
    result = parse_payload(_load("s2c_first_share_result.msgpack"))
    assert isinstance(result, ShareResult)
    assert result.share_id == "cf73bfaf-8cb7-4298-a1ef-86bd087ee9fa"
    assert result.accepted is True
    assert result.message == "Accepted"
    type_code, _ = unpack_payload(_load("s2c_first_share_result.msgpack"))
    assert type_code == TYPE_SHARE_RESULT


def test_plain_proof_share_schema_and_roundtrip() -> None:
    frame = _load("c2s_first_plainproofshare.frame")
    payload = _load("c2s_first_plainproofshare.msgpack")
    assert frame[4:] == payload

    type_code, fields = unpack_payload(payload)
    assert type_code == TYPE_PLAIN_PROOF_SHARE
    assert len(fields) == 15

    share = PlainProofShare.from_fields(fields)
    assert share.share_id == "cf73bfaf-8cb7-4298-a1ef-86bd087ee9fa"
    assert len(share.header_bytes) == HEADER_SIZE
    assert len(share.mining_config_bytes) == MINING_CONFIG_SIZE
    assert len(share.opened_a) == 4096
    assert len(share.opened_b) == 131072
    assert len(share.compact_result) == 64
    assert share.t_rows == 1601
    assert share.t_cols == 4864
    assert share.share_difficulty == 439353344

    config = share.mining_config
    assert config.common_dim == 2048
    assert config.rank == 128
    assert config.mma_type == 0
    assert config.to_bytes() == share.mining_config_bytes
    assert config.rows_pattern.to_list() == [0, 8]
    assert config.cols_pattern.to_list()[:8] == [0, 1, 8, 9, 16, 17, 24, 25]
    assert len(config.cols_pattern.to_list()) == 64

    summary = share.validate()
    assert summary["rows"] == [1601, 1609]
    assert summary["cols_count"] == 64
    assert summary["cols_head"] == [4864, 4865, 4872, 4873, 4880, 4881, 4888, 4889]
    assert summary["a_leaf_count"] == 4
    assert summary["b_leaf_count"] == 128
    assert summary["a_sibling_count"] == 15
    assert summary["b_sibling_count"] == 71

    semantic_frame = share.frame()
    rebuilt = parse_frame(semantic_frame)
    assert isinstance(rebuilt, PlainProofShare)
    assert rebuilt.to_fields() == share.to_fields()
    assert int.from_bytes(semantic_frame[:4], "big") == len(semantic_frame) - 4


def main() -> None:
    test_frame_prefixes()
    test_register_and_ack()
    test_job_assignment()
    test_share_result()
    test_plain_proof_share_schema_and_roundtrip()
    print("akoya_protocol fixture tests: OK")


if __name__ == "__main__":
    main()

