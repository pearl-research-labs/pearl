"""Unit tests for pearl_stratum.luckypool_client — object-param dialect.

Covers notify-parse ({job_id, header, target, height}), submit-format
({job_id, plain_proof, hs}), target hex parsing/endianness, and malformed
input rejection. The captured 76-byte header from the LuckyPool share dump is
used as a realistic notify `header`.
"""

from __future__ import annotations

import json

import pytest

from pearl_stratum.luckypool_client import (
    LuckyPoolJob,
    build_submit_params,
    parse_luckypool_notify,
    parse_target_hex,
    target_int_to_le_bytes,
)

# Captured 76-byte incomplete header (hex) from re_2026_05_30/sharedump/header.bin.
HEADER_HEX = (
    "01000000f9661239d86cd892e31455d6ca6c1a55747ab7d16a63c82143d271f417ca4999"
    "4f2738ce9c121c2285980708e168b1f4e1b8167b4e6f30fe911d555492a1afac6d3c22a6207fffff"
)
# (length is validated to 76 bytes by the parser; the exact bytes don't matter
#  for the parser tests beyond decoding cleanly.)
HEADER_HEX_76 = "01" * 76


def test_parse_target_hex_big_endian():
    # 0x...01 big-endian => 1
    assert parse_target_hex("0" * 63 + "1") == 1
    # with 0x prefix
    assert parse_target_hex("0x" + "0" * 63 + "ff") == 0xFF


def test_parse_target_hex_little_endian():
    assert parse_target_hex("01" + "00" * 31, big_endian=False) == 1


def test_target_le_is_32_bytes():
    le = target_int_to_le_bytes(1)
    assert len(le) == 32
    assert le[0] == 1 and le[1] == 0


def test_parse_notify_object_params():
    params = {
        "job_id": "abc-123",
        "header": HEADER_HEX_76,
        "target": "0" * 32 + "f" * 32,  # 256-bit, lower half set
        "height": 99,
    }
    job = parse_luckypool_notify(params)
    assert isinstance(job, LuckyPoolJob)
    assert job.job_id == "abc-123"
    assert len(job.header_bytes) == 76
    assert job.target == parse_target_hex(params["target"])
    assert len(job.target_le) == 32
    assert job.height == 99


def test_parse_notify_missing_height_ok():
    params = {"job_id": "j", "header": HEADER_HEX_76, "target": "0" * 63 + "1"}
    job = parse_luckypool_notify(params)
    assert job.height is None


def test_parse_notify_integer_target():
    params = {"job_id": "j", "header": HEADER_HEX_76, "target": 1234567}
    job = parse_luckypool_notify(params)
    assert job.target == 1234567


@pytest.mark.parametrize(
    "params",
    [
        ["positional", "not", "object"],          # positional (alphapool) shape rejected
        {"header": HEADER_HEX_76, "target": "1"},  # missing job_id
        {"job_id": "j", "target": "1"},            # missing header
        {"job_id": "j", "header": "zz", "target": "1"},  # bad hex
        {"job_id": "j", "header": "00" * 40, "target": "1"},  # wrong header length
        {"job_id": "j", "header": HEADER_HEX_76},  # missing target
        {"job_id": "j", "header": HEADER_HEX_76, "target": "0" * 64},  # zero target
    ],
)
def test_parse_notify_rejects_malformed(params):
    with pytest.raises(ValueError):
        parse_luckypool_notify(params)


def test_build_submit_params_object_shape():
    sp = build_submit_params("job-9", "QkFTRTY0UFJPT0Y=", hashrate=137.0e12)
    assert set(sp.keys()) == {"job_id", "plain_proof", "hs"}
    assert sp["job_id"] == "job-9"
    assert sp["plain_proof"] == "QkFTRTY0UFJPT0Y="
    assert isinstance(sp["hs"], float)
    # Real frame is JSON-serializable.
    json.dumps({"id": 1, "method": "mining.submit", "params": sp})
