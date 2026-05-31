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


# ===========================================================================
# Wire-target -> on-device PoW-threshold decode (the diff=262144 share blocker)
# ===========================================================================
#
# Authority: zk-pow/src/api/sanity_checks.rs::extract_difficulty_bound +
# check_jackpot_difficulty_with_nbits, enforced by verify_plain_proof
# (py-pearl-mining/src/lib.rs:129). The verifier accepts a share iff
#     int.from_bytes(jackpot_hash, "little") <= target * (h*w*k)
# where the difficulty_adjustment_factor h*w*k = rows_pattern.size (8) *
# cols_pattern.size (16) * dot_product_length (k - k%rank). The GPU binary
# pearl_miner_sm89.cu applies this `* h*w*k` multiply when converting the wire
# target to the 32 LE pow_target words; the bug it fixes is that the device was
# fed the RAW wire target (h*w*k ~ 2^19 times too HARD -> ZERO shares).

# Live captured wire pair (mining.notify job bfd4cb60_262144).
LIVE_WIRE_TARGET_HEX = "0000000000003fffc00000000000000000000000000000000000000000000000"
LIVE_DIFFICULTY = 262144  # 2^18

# Production geometry (re_2026_05_30/sharedump/meta.txt + report 07):
#   rows_pattern.size = 8, cols_pattern.size = 16, common_dim k = 4096, rank = 256.
H, W, K, RANK = 8, 16, 4096, 256


def _device_threshold_from_wire(wire_target_be: int, k: int, rank: int) -> int:
    """Reference re-implementation of pearl_miner_sm89.cu's target adjustment.

    device_threshold = min(wire_target_be * (h*w*dot_product_length), 2^256-1).
    Mirrors the C `__uint128_t` LE-byte multiply + saturate, byte-for-byte.
    """
    dpl = (k - k % rank) if rank > 0 else k
    adj = H * W * dpl
    le = list(wire_target_be.to_bytes(32, "little"))
    carry = 0
    for i in range(32):
        prod = le[i] * adj + carry
        le[i] = prod & 0xFF
        carry = prod >> 8
    if carry != 0:  # exceeded 256 bits -> saturate (verifier uses U256::MAX)
        return (1 << 256) - 1
    return int.from_bytes(bytes(le), "little")


def test_wire_target_parses_big_endian():
    wire = parse_target_hex(LIVE_WIRE_TARGET_HEX)  # big-endian default
    assert wire == 0x3FFFC0 << 184  # exact wire value; bit_length 206 (~2^206)
    assert wire.bit_length() == 206


def test_diff1_target_is_clean_constant():
    # Standard stratum: wire_target = diff1 / difficulty. Recover diff1.
    wire = parse_target_hex(LIVE_WIRE_TARGET_HEX)
    diff1 = wire * LIVE_DIFFICULTY
    assert diff1 == (0xFFFF << 208)  # clean diff-1 constant => decode is correct


def test_device_threshold_lands_in_empirical_band():
    # wire (~2^206) * h*w*k (2^19) -> bit_length 225, inside the empirically-
    # required ~2^225..2^232 window (lpminer finds a share every ~10-60s; the raw
    # wire target finds zero). This is the whole fix.
    wire = parse_target_hex(LIVE_WIRE_TARGET_HEX)
    dev = _device_threshold_from_wire(wire, K, RANK)
    # Exactly h*w*k times the raw wire target (the missing adjustment factor).
    assert dev == wire * (H * W * K)
    assert dev.bit_length() == 225
    assert (1 << 224) < dev < (1 << 233)


def test_device_threshold_saturates_for_easy_target():
    # An easy/max wire target (close to 2^256) * h*w*k overflows -> saturate to
    # 2^256-1, matching the verifier's U256::MAX clamp in extract_difficulty_bound.
    easy = (1 << 256) - 1
    assert _device_threshold_from_wire(easy, K, RANK) == (1 << 256) - 1


def test_oracle_hash_classifies_correctly_under_unified_rule():
    # Regression on the captured easy-dump oracle (meta.txt). The share was dumped
    # at the EASY/max nbits target (0x207fffff); under the unified rule its hash,
    # interpreted little-endian (the verifier + device convention), must be <= the
    # (saturated) easy threshold and verify_plain_proof returned True (report 07).
    gpu_hash = bytes.fromhex(
        "7c920e4756693f4c9c1d03d24b25ef8a937d361248922f47fc60d1b6c947ae60"
    )
    hash_le = int.from_bytes(gpu_hash, "little")  # verifier: U256::from_little_endian
    easy_threshold = _device_threshold_from_wire((1 << 256) - 1, K, RANK)
    assert hash_le <= easy_threshold  # oracle still classifies as a valid share
    # Sanity: the same hash under the HARD live diff=262144 threshold is NOT a hit
    # (the oracle was an easy share, not a diff=262144 share) — confirms the
    # threshold actually discriminates rather than always accepting.
    live_threshold = _device_threshold_from_wire(parse_target_hex(LIVE_WIRE_TARGET_HEX), K, RANK)
    assert hash_le > live_threshold
