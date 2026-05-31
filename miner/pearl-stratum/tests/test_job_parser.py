"""Unit tests for pearl_stratum.job — mining.notify parsing and target derivation.

Fixture data is the captured mining.notify from
C:/Source/pearl-investigation/messages_full.txt (first frame, conn[0]):

    job_id:                  "0000d446-3061"
    prevhash:                "46b849bae7551681283f02a20080cd3f0fd0dfad5e320b09b6af901291bfc554"
    incomplete_header_bytes: "0000402054c5bf911290afb6090b325eaddfd00f3fcd8000a2023f28811655e7ba49b846d262d62ab2f3dbbf2ddd73a2c00a9ccd9838264c4298998096ef5602b0bfec3b6130096a99a00618"
    ntime:                   54342
    nbits:                   "6a093061"
    version-or-jobtype:      "1a0ffff0"
    clean_jobs:              true
"""

from __future__ import annotations

import pytest

from pearl_stratum.job import (
    Job,
    _bits_to_target_int,
    _target_int_to_le_bytes,
    parse_notify,
)


SAMPLE_PARAMS = [
    "0000d446-3061",
    "46b849bae7551681283f02a20080cd3f0fd0dfad5e320b09b6af901291bfc554",
    "0000402054c5bf911290afb6090b325eaddfd00f3fcd8000a2023f28811655e7ba49b846d262d62ab2f3dbbf2ddd73a2c00a9ccd9838264c4298998096ef5602b0bfec3b6130096a99a00618",
    54342,
    "6a093061",   # ntime (Unix epoch)
    "1a0ffff0",   # nbits (compact target — testnet difficulty 1)
    True,
]


def test_parse_notify_happy_path() -> None:
    job = parse_notify(SAMPLE_PARAMS)
    assert isinstance(job, Job)
    assert job.job_id == "0000d446-3061"
    assert job.incomplete_header_bytes == bytes.fromhex(SAMPLE_PARAMS[2])
    assert job.nbits == 0x1A0FFFF0
    assert job.clean_jobs is True
    assert job.received_at > 0


def test_parse_notify_target_matches_bits_to_target() -> None:
    """Verify our target calc matches pearl_gateway's bits_to_target."""
    job = parse_notify(SAMPLE_PARAMS)
    # 0x1a0ffff0: exponent=0x1a, mantissa=0x0ffff0. target = 0x0ffff0 << (8*(0x1a-3)).
    exp = 0x1A
    mantissa = 0x0FFFF0
    expected = mantissa * (1 << (8 * (exp - 3)))
    assert job.target == expected


def test_target_le_is_32_bytes_little_endian() -> None:
    job = parse_notify(SAMPLE_PARAMS)
    assert len(job.target_le) == 32
    # Recover the integer from target_le and confirm round-trip.
    assert int.from_bytes(job.target_le, "little") == job.target


def test_target_le_endianness_distinct_from_be() -> None:
    """Sanity check: LE form should differ from BE form for nontrivial targets."""
    job = parse_notify(SAMPLE_PARAMS)
    be = job.target.to_bytes(32, "big")
    assert job.target_le != be  # nontrivial 256-bit values flip


def test_clean_jobs_default_true_when_missing() -> None:
    short = SAMPLE_PARAMS[:6]  # drop the trailing clean_jobs flag
    job = parse_notify(short)
    assert job.clean_jobs is True


def test_clean_jobs_false_propagates() -> None:
    params = list(SAMPLE_PARAMS)
    params[6] = False
    job = parse_notify(params)
    assert job.clean_jobs is False


def test_parse_notify_rejects_short_params() -> None:
    with pytest.raises(ValueError, match="list of >=6"):
        parse_notify(["only-one"])


def test_parse_notify_rejects_non_list() -> None:
    with pytest.raises(ValueError, match="list of >=6"):
        parse_notify("not a list")  # type: ignore[arg-type]


def test_parse_notify_rejects_bad_hex() -> None:
    bad = list(SAMPLE_PARAMS)
    bad[2] = "zz-not-hex"
    with pytest.raises(ValueError, match="not valid hex"):
        parse_notify(bad)


def test_parse_notify_accepts_int_nbits() -> None:
    params = list(SAMPLE_PARAMS)
    params[5] = 0x1A0FFFF0  # already int form
    job = parse_notify(params)
    assert job.nbits == 0x1A0FFFF0


def test_parse_notify_rejects_oversize_nbits() -> None:
    params = list(SAMPLE_PARAMS)
    params[5] = 0x1_0000_0000  # 33 bits
    with pytest.raises(ValueError, match="out of 32-bit"):
        parse_notify(params)


def test_parse_notify_rejects_zero_target() -> None:
    params = list(SAMPLE_PARAMS)
    params[5] = "00000000"  # mantissa=0 -> target=0
    with pytest.raises(ValueError, match="zero target"):
        parse_notify(params)


def test_raw_params_preserved() -> None:
    job = parse_notify(SAMPLE_PARAMS)
    assert job.raw_params == SAMPLE_PARAMS
    # And independence — mutating returned raw_params shouldn't affect input.
    job.raw_params.clear()
    assert SAMPLE_PARAMS[0] == "0000d446-3061"


def test_bits_to_target_small_exponent_branch() -> None:
    """Cover the exp < 3 branch (rare but legal)."""
    # exp=2, mantissa=0xffffff -> target = 0xffffff >> (8*(3-2)) = 0xffff
    assert _bits_to_target_int(0x02FFFFFF) == 0xFFFF


def test_target_le_saturates_at_256_bits() -> None:
    """Synthetic over-large target should be masked to 256 bits."""
    overflow_target = (1 << 256) + 7
    le = _target_int_to_le_bytes(overflow_target)
    assert len(le) == 32
    assert int.from_bytes(le, "little") == 7  # low bits survived
