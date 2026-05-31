"""Tests for the glibc `srand(0)`/`rand()` sequence reimplementation.

Reference sequence comes from `36_pearlhash_shim.md` §1.3, which lists the
first 16 outputs verified bit-exact against the actual pearl-miner v2 binary
running on CPU01. Our pure-Python implementation must match those values."""

from __future__ import annotations

import pytest

from pearl_stratum.pearlhash_rand import GlibcRand, glibc_rand_sequence


# 16 outputs of `srand(0); rand()...` per memo 36 §1.3.
EXPECTED_FIRST_16 = [
    0x6B8B4567, 0x327B23C6, 0x643C9869, 0x66334873,
    0x74B0DC51, 0x19495CFF, 0x2AE8944A, 0x625558EC,
    0x238E1F29, 0x46E87CCD, 0x3D1B58BA, 0x507ED7AB,
    0x2EB141F2, 0x41B71EFB, 0x79E2A9E3, 0x7545E146,
]


def test_first_16_outputs_match_memo() -> None:
    outs = glibc_rand_sequence(16)
    assert outs == EXPECTED_FIRST_16


def test_first_output_is_login_counter() -> None:
    """rand[0] == the counter prefix observed on every login frame."""
    g = GlibcRand(seed=0)
    assert g.next_u31() == 0x6B8B4567


def test_outputs_are_31_bit() -> None:
    """glibc rand() returns a value in [0, 2^31), high bit clear."""
    for v in glibc_rand_sequence(64):
        assert 0 <= v < (1 << 31)


def test_independent_instances_match() -> None:
    """Two GlibcRand(0) instances produce identical sequences."""
    a = [GlibcRand(seed=0).next_u31() for _ in range(4)]
    b = [GlibcRand(seed=0).next_u31() for _ in range(4)]
    # Each fresh instance starts from rand[0]; one-shot calls produce the same value.
    assert a == [EXPECTED_FIRST_16[0]] * 4
    assert b == [EXPECTED_FIRST_16[0]] * 4


def test_seed_zero_replaced_with_one() -> None:
    """glibc internally treats srand(0) as srand(1) — verify by comparison."""
    seq0 = glibc_rand_sequence(16, seed=0)
    seq1 = glibc_rand_sequence(16, seed=1)
    assert seq0 == seq1
    assert seq0 == EXPECTED_FIRST_16


def test_non_zero_seed_diverges() -> None:
    seq0 = glibc_rand_sequence(8, seed=0)
    seq2 = glibc_rand_sequence(8, seed=2)
    assert seq0 != seq2
