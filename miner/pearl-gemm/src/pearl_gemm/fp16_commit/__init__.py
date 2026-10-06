"""Bit-exact A100 (``sm_80``) keyed-BLAKE3 Merkle commitment over FP16 operand
rows for the FP16 proof-of-useful-work scheme.

Reproduces ``zk-pow/src/api/fp16/commitment.rs::commit_operand`` -- the
``pearl_blake3::MerkleTree::with_chunk_len`` keyed Merkle root directly over the
FP16 rows (little-endian ``u16`` row-major, zero-padded to the ``HashId``
chunk length) -- bit-for-bit on real GA100 silicon. This is the operand
commitment root that goes into the FP16 certificate.
"""

from ._host import (
    ALLOWED_CHUNK_LENS,
    DEFAULT_CHUNK_LEN,
    commit_operand,
)

__all__ = [
    "ALLOWED_CHUNK_LENS",
    "DEFAULT_CHUNK_LEN",
    "commit_operand",
]
