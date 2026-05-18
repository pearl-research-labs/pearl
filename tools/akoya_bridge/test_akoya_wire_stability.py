#!/usr/bin/env python3
"""Byte-stability tests for Akoya MessagePack framing."""

from __future__ import annotations

from pathlib import Path
import struct
import sys

import msgpack

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from akoya_protocol import (  # noqa: E402
    HEADER_SIZE,
    MINING_CONFIG_SIZE,
    TYPE_PLAIN_PROOF_SHARE,
    TYPE_REGISTER,
    MatrixProofWire,
    PlainProofShare,
    RegisterMiner,
    pack_frame,
    pack_payload,
    parse_frame,
)


PUBLIC_STAGING = Path("/srv/pearl/migration/current/public_staging")


def _capture_style_payload(type_code: int, fields: list[object]) -> bytes:
    return b"\x92\xd2" + struct.pack(">i", type_code) + msgpack.packb(fields, use_bin_type=True)


def _capture_style_frame(type_code: int, fields: list[object]) -> bytes:
    payload = _capture_style_payload(type_code, fields)
    return len(payload).to_bytes(4, "big") + payload


def _sample_share() -> PlainProofShare:
    a_leaf0 = b"A" * 1024
    a_leaf1 = b"B" * 1024
    b_leaf0 = b"C" * 1024
    return PlainProofShare(
        share_id="00000000-0000-4000-8000-000000000123",
        header_bytes=b"\x11" * HEADER_SIZE,
        opened_a=a_leaf0 + a_leaf1,
        seed_hash=b"\x22" * 32,
        t_rows=7,
        t_cols=9,
        compact_result=b"\x33" * 64,
        claimed_hash=b"\x44" * 32,
        share_difficulty=0x1A300000,
        hash_a=b"\x55" * 32,
        hash_b=b"\x66" * 32,
        opened_b=b_leaf0,
        mining_config_bytes=b"\x00" * MINING_CONFIG_SIZE,
        a_proof=MatrixProofWire(
            leaf_data=(a_leaf0, a_leaf1),
            leaf_indices=(0, 1),
            total_leaves=2,
            siblings=(b"\x77" * 32,),
        ),
        b_proof=MatrixProofWire(
            leaf_data=(b_leaf0,),
            leaf_indices=(0,),
            total_leaves=1,
            siblings=(b"\x88" * 32,),
        ),
    )


def test_pack_payload_uses_int32_type_code() -> None:
    payload = pack_payload(TYPE_REGISTER, ["worker"])
    assert payload == _capture_style_payload(TYPE_REGISTER, ["worker"])
    assert payload[:6] == b"\x92\xd2\x00\x00\x00\x00"


def test_register_parse_to_fields_pack_frame_is_byte_stable() -> None:
    register = RegisterMiner(
        session_uuid="00000000-0000-4000-8000-000000000001",
        wallet_address="prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqv3y4d",
        worker_name="worker-1",
        gpu_name="RTX 4090",
        common_dim=2048,
        miner_version="0.1.0",
        git_sha="deadbeef",
    )
    frame = _capture_style_frame(TYPE_REGISTER, register.to_fields())
    parsed = parse_frame(frame)
    assert isinstance(parsed, RegisterMiner)
    assert pack_frame(TYPE_REGISTER, parsed.to_fields()) == frame


def test_plainproofshare_parse_to_fields_pack_frame_is_byte_stable() -> None:
    share = _sample_share()
    frame = _capture_style_frame(TYPE_PLAIN_PROOF_SHARE, share.to_fields())
    parsed = parse_frame(frame)
    assert isinstance(parsed, PlainProofShare)
    assert pack_frame(TYPE_PLAIN_PROOF_SHARE, parsed.to_fields()) == frame


def test_public_staging_frames_roundtrip_if_present() -> None:
    register_frame = PUBLIC_STAGING / "c2s_register.frame"
    share_frame = PUBLIC_STAGING / "c2s_public_plainproofshare.frame"
    if not register_frame.exists() or not share_frame.exists():
        print("skip: /srv/pearl/migration/current/public_staging frames not present")
        return

    parsed_register = parse_frame(register_frame.read_bytes())
    assert isinstance(parsed_register, RegisterMiner)
    assert pack_frame(TYPE_REGISTER, parsed_register.to_fields()) == register_frame.read_bytes()

    parsed_share = parse_frame(share_frame.read_bytes())
    assert isinstance(parsed_share, PlainProofShare)
    assert pack_frame(TYPE_PLAIN_PROOF_SHARE, parsed_share.to_fields()) == share_frame.read_bytes()


def main() -> None:
    test_pack_payload_uses_int32_type_code()
    test_register_parse_to_fields_pack_frame_is_byte_stable()
    test_plainproofshare_parse_to_fields_pack_frame_is_byte_stable()
    test_public_staging_frames_roundtrip_if_present()
    print("akoya wire stability tests: OK")


if __name__ == "__main__":
    main()
