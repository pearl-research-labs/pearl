"""FP16 (A100) plaintext-certificate opener + assembler tests.

Mirrors the Rust oracle ``zk-pow/src/api/fp16/{commitment,verify}.rs``:

* the opener (:mod:`miner_base.fp16_commitment`) builds the per-operand keyed
  Merkle tree over raw FP16 ``u16`` rows and opens the tile's global rows;
* the assembler (:func:`miner_base.fp16_block_submission.create_fp16_proof`)
  wraps a winning tile into a ``pearl_mining.Fp16PlainProof``.

The honest operands are reproduced bit-for-bit from the Rust ``cert`` fixture's
``Gen`` xorshift (``honest_fixture`` in ``verify.rs``), so the assembled proof is
byte-identical to the committed ``node/zkpow/testdata/fp16_plain_proof_a100.bin``
and verifies.

Skipped unless the ``pearl_mining`` extension is importable (needs py3.12/3.13
here; the default miner interpreter cannot both run GPU kernels and import it).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

pearl_mining = pytest.importorskip("pearl_mining")

from miner_base.fp16_commitment import (  # noqa: E402
    DEFAULT_HASH_ID,
    commit_fp16_operand,
    commitment_keys,
    key_a,
    key_b,
)
from miner_base.fp16_block_submission import create_fp16_proof  # noqa: E402
from miner_base.layout import AxisPattern, DimType  # noqa: E402

# Honest-fixture shape (verify.rs::cert): m=8 > h=4 and n=128 > w=64, so unopened
# rows carry real siblings. k=256, rank 32.
_M, _N, _K, _R = 8, 128, 256, 32
_NBITS = 0x207FFFFF  # EASY_NBITS
_GEN_SEED = 0xDEADBEEF0BADF00D

# P_A = [(4, Blake)] -> h=4 ; P_B = [(4, Blake), (16, Fold)] -> w=64.
_P_A = AxisPattern(((4, DimType.BLAKE),))
_P_B = AxisPattern(((4, DimType.BLAKE), (16, DimType.FOLD)))

_FIXTURE = Path(__file__).resolve().parents[3] / "node" / "zkpow" / "testdata" / "fp16_plain_proof_a100.bin"
_U64 = (1 << 64) - 1


class _Gen:
    """Byte-identical to the Rust ``Gen`` xorshift64 operand generator."""

    def __init__(self, seed: int) -> None:
        self.s = seed & _U64

    def _next(self) -> int:
        self.s ^= (self.s << 13) & _U64
        self.s ^= self.s >> 7
        self.s ^= (self.s << 17) & _U64
        return self.s

    def operand(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.float16)
        for i in range(n):
            r = self._next()
            sign = 1.0 if (r & 1) == 0 else -1.0
            exp = ((r >> 1) % 9) - 3
            mant = 1.0 + ((r >> 8) % 1024) / 1024.0
            out[i] = np.float16(np.float32(sign * mant * (2.0 ** exp)))
        return out.view(np.uint16)


def _test_header() -> bytes:
    """The ``honest_fixture`` wire bytes: version 0, timestamp 0x66666666, nbits,
    with ASYMMETRIC prev_block/merkle_root (not palindromic under byte reversal, so
    the fixture exercises header byte-orientation across the FFI seam). The Rust
    generator sets the hash arrays to [0..31] / [0x40..0x5f]; IncompleteBlockHeader
    serializes them in reversed (wire) byte order, so the on-disk bytes are
    prev_block = [0x1f..0x00], merkle_root = [0x5f..0x40]."""
    return (
        struct.pack("<I", 0)
        + bytes(range(31, -1, -1))
        + bytes(range(0x5F, 0x3F, -1))
        + struct.pack("<II", 0x66666666, _NBITS)
    )


def _honest_operands() -> tuple[np.ndarray, np.ndarray]:
    g = _Gen(_GEN_SEED)
    a = g.operand(_M * _K)
    b = g.operand(_N * _K)
    return a, b


def test_patterns_are_contiguous() -> None:
    # The #7 resolution: the committed A100 patterns have contiguous tile
    # offsets, so a tile opens a dense contiguous run of global rows.
    assert _P_A.tile_offsets == list(range(4))
    assert _P_A.total == _P_A.tile_size == 4
    assert _P_B.tile_offsets == list(range(64))
    assert _P_B.total == _P_B.tile_size == 64


def test_opener_root_is_deterministic_and_opening_authenticates() -> None:
    a, _b = _honest_operands()
    header = _test_header()
    comm = commit_fp16_operand(a, _M, _K, key_a(header), DEFAULT_HASH_ID)
    # Re-committing identical rows reproduces the root (keyed, deterministic).
    comm2 = commit_fp16_operand(a, _M, _K, key_a(header), DEFAULT_HASH_ID)
    assert comm.root == comm2.root
    # Open the origin tile and check the native Merkle proof authenticates.
    opening = comm.open(_P_A.tile_offsets)
    assert opening.row_indices == [0, 1, 2, 3]
    native = pearl_mining.MerkleProof(
        [bytes(x) for x in opening.proof.leaf_data],
        list(opening.proof.leaf_indices),
        bytes(opening.proof.root),
        [bytes(s) for s in opening.proof.siblings],
        opening.proof.total_leaves,
    )
    assert native.verify(bytes(key_a(header)))


def test_keys_match_commitment_keys_helper() -> None:
    header = _test_header()
    ka, kb = commitment_keys(header, header)
    assert ka == key_a(header)
    assert kb == key_b(header)


def test_create_fp16_proof_origin_verifies() -> None:
    a, b = _honest_operands()
    header = _test_header()
    proof = create_fp16_proof(
        header, a, b, k=_K, m=_M, n=_N, rows_pattern=_P_A, cols_pattern=_P_B,
        tile_row=0, tile_col=0,
    )
    assert list(proof.values_a.row_indices) == [0, 1, 2, 3]
    assert list(proof.values_b.row_indices) == list(range(64))
    assert proof.a.num_rows == _M and proof.b.num_rows == _N
    assert proof.k == _K and proof.r == _R
    # Wire round-trip.
    assert pearl_mining.Fp16PlainProof.from_bytes(proof.to_bytes()).to_bytes() == proof.to_bytes()
    # Full plaintext verification accepts.
    ok, msg = pearl_mining.verify_fp16_plain_proof(
        pearl_mining.IncompleteBlockHeader.from_bytes(header), proof, None
    )
    assert ok, msg


@pytest.mark.skipif(not _FIXTURE.exists(), reason="committed FP16 Go fixture not present")
def test_create_fp16_proof_byte_matches_committed_fixture() -> None:
    a, b = _honest_operands()
    header = _test_header()
    blob = _FIXTURE.read_bytes()
    assert blob[:76] == header
    plen = struct.unpack_from("<I", blob, 76)[0]
    fixture_proof = blob[80 : 80 + plen]

    proof = create_fp16_proof(
        header, a, b, k=_K, m=_M, n=_N, rows_pattern=_P_A, cols_pattern=_P_B,
    )
    assert bytes(proof.to_bytes()) == fixture_proof


def test_create_fp16_proof_non_origin_tile() -> None:
    # A winner off the origin opens the contiguous global rows base+tile_offsets
    # (tr*h / tc*w). Requires the base-aware parse_proof (current zk-pow).
    a, b = _honest_operands()
    header = _test_header()
    proof = create_fp16_proof(
        header, a, b, k=_K, m=_M, n=_N, rows_pattern=_P_A, cols_pattern=_P_B,
        tile_row=1, tile_col=1,
    )
    assert list(proof.values_a.row_indices) == [4, 5, 6, 7]
    assert list(proof.values_b.row_indices) == list(range(64, 128))
    ok, msg = pearl_mining.verify_fp16_plain_proof(
        pearl_mining.IncompleteBlockHeader.from_bytes(header), proof, None
    )
    assert ok, msg
