"""JobRegistry: minting, lookup, eviction, latest."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from pearl_stratum_srv.job_registry import JobRegistry


@dataclass
class _FakeHeader:
    timestamp: int
    target_bits: int
    previous_block_hash: bytes

    def serialize_without_proof_commitment(self) -> bytes:
        return b"\x00" * 76


def _make_template(height: int, prev: bytes = b"\xab" * 32) -> MagicMock:
    t = MagicMock()
    t.height = height
    t.header = _FakeHeader(
        timestamp=0x6A093061,
        target_bits=0x1A0FFFF0,
        previous_block_hash=prev,
    )
    return t


def test_mint_assigns_unique_job_ids_with_height_prefix():
    reg = JobRegistry()
    e1 = reg.mint(_make_template(54321))
    e2 = reg.mint(_make_template(54321))
    assert e1.job_id != e2.job_id
    # height-prefix in hex
    assert e1.job_id.startswith("0000d431-")  # 54321 = 0xd431
    assert e2.job_id.startswith("0000d431-")


def test_get_returns_entry_by_id():
    reg = JobRegistry()
    entry = reg.mint(_make_template(100))
    assert reg.get(entry.job_id) is entry


def test_get_returns_none_for_unknown_job_id():
    reg = JobRegistry()
    reg.mint(_make_template(100))
    assert reg.get("deadbeef-9999") is None


def test_eviction_when_over_max_size():
    reg = JobRegistry(max_size=3)
    ids = [reg.mint(_make_template(h)).job_id for h in (1, 2, 3, 4, 5)]
    assert reg.get(ids[0]) is None  # evicted
    assert reg.get(ids[1]) is None  # evicted
    assert reg.get(ids[-1]) is not None  # newest survives
    assert len(reg) == 3


def test_latest_returns_most_recently_minted():
    reg = JobRegistry()
    reg.mint(_make_template(1))
    second = reg.mint(_make_template(2))
    assert reg.latest() is second


def test_latest_is_none_when_empty():
    reg = JobRegistry()
    assert reg.latest() is None


def test_entry_exposes_nbits_and_ntime_hex():
    reg = JobRegistry()
    entry = reg.mint(_make_template(0xD446))
    assert entry.nbits_hex == "1a0ffff0"
    assert entry.ntime_hex == "6a093061"
    assert entry.prev_hash_hex == "ab" * 32


def test_max_size_must_be_positive():
    with pytest.raises(ValueError):
        JobRegistry(max_size=0)
