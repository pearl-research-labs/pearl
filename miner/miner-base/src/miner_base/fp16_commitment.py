"""Keyed Merkle commitments and openings for the FP16 (A100) scheme.

The FP16 analogue of :mod:`miner_base.commitment`, bit-exact to the Rust
``zk-pow/src/api/fp16/commitment.rs``. Unlike FP8 -- which commits a separate
int8-values tree and a BF16-scales tree per operand -- FP16 is strictly simpler:
**one keyed-BLAKE3 Merkle tree per operand, directly over the raw FP16 rows**
(``u16`` bit patterns, little-endian, row-major). The committed leaves ARE the
FP16 values; there is no prequant layer.

The tree/key/hash-id discipline matches FP8 exactly (and the GPU
``pearl_gemm.fp16_commit`` kernel): leaves are fixed-size ``hash_id.chunk_len``
chunks of the zero-padded row bytes, built under a per-side opening key

    keyA = H_"key-A"(proposed_header)      (A-side tree, keys on sigma-hat)
    keyB = H_"key-B"(ancestor_header)      (B-side tree, keys on sigma-Delta)

(``zk-pow/src/api/fp16/noise.rs::commitment_keys``). An opening discloses exactly
the selected tile rows as the unique-minimal leaf/sibling set -- the same
``MerkleTree.compute_leaf_indices_from_rows`` + ``get_multileaf_proof`` the FP8
opener uses, so the Rust ``verify_and_open_rows`` rebuilds and accepts it.

Torch-free and importable under any interpreter: the Merkle primitives come from
``pearl_blake3`` (the standalone extension) or, when only the unified extension is
installed, from ``pearl_mining`` (which re-exports the identical types).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import blake3

try:  # the standalone blake3 extension (preferred; importable under py3.10/3.13)
    from pearl_blake3 import MerkleProof, MerkleTree
except ImportError:  # the unified pearl_mining extension re-exports the same types
    from pearl_mining import MerkleProof, MerkleTree

from .params import HashId

__all__ = [
    "LABEL_KEY_A",
    "LABEL_KEY_B",
    "DEFAULT_HASH_ID",
    "Fp16MatrixCommitment",
    "Fp16MatrixMerkleProof",
    "commit_fp16_operand",
    "key_a",
    "key_b",
    "commitment_keys",
    "rows_to_bytes",
]

# The protocol default Merkle leaf (the GPU ``fp16_commit`` kernel and the Rust
# both default to ``HashId::Blake3Chunk1024``).
DEFAULT_HASH_ID = HashId.BLAKE3_CHUNK_1024

# FP16 transcript labels (``zk-pow/src/api/fp16/noise.rs``): a distinct domain
# from the FP8 labels, so the two schemes never derive the same key from the same
# header.
LABEL_KEY_A = b"pearl/v4/FP16/key-A"
LABEL_KEY_B = b"pearl/v4/FP16/key-B"


def _subkey(label: bytes) -> bytes:
    """``subkey(label) = BLAKE3(label)`` -- the 32-byte role key (unkeyed)."""
    return blake3.blake3(label).digest(length=32)


def _hash_labelled(message: bytes, label: bytes) -> bytes:
    """``H_label(message) = keyed_hash(key=subkey(label), message)``
    (``noise.rs::hash_labelled``)."""
    return blake3.blake3(message, key=_subkey(label)).digest(length=32)


def key_a(proposed_header: bytes) -> bytes:
    """``keyA = H_"key-A"(proposed_header)`` -- the A-side tree/opening key."""
    return _hash_labelled(bytes(proposed_header), LABEL_KEY_A)


def key_b(ancestor_header: bytes) -> bytes:
    """``keyB = H_"key-B"(ancestor_header)`` -- the B-side tree/opening key."""
    return _hash_labelled(bytes(ancestor_header), LABEL_KEY_B)


def commitment_keys(proposed_header: bytes, ancestor_header: bytes) -> tuple[bytes, bytes]:
    """``(keyA, keyB)`` -- the per-side opening keys
    (``noise.rs::commitment_keys``). The miner proposes at depth 0, so a caller
    with ``ancestor_header == proposed_header`` keys both trees off one header."""
    return key_a(proposed_header), key_b(ancestor_header)


def rows_to_bytes(rows: object, num_rows: int, k: int) -> bytes:
    """The committed little-endian ``u16`` row-major byte image of an
    ``num_rows x k`` FP16 operand (``commitment.rs::rows_to_bytes``), before
    hash-id padding.

    ``rows`` may be raw ``bytes`` (already the ``u16`` LE image), a NumPy array
    of ``uint16`` / ``int16`` / ``float16`` bit patterns, a torch ``float16`` /
    ``int16`` tensor, or a flat sequence of ``u16`` integers. All are read as the
    operand's FP16 *bit patterns* -- never re-encoded.
    """
    want = num_rows * k * 2

    if isinstance(rows, (bytes, bytearray, memoryview)):
        raw = bytes(rows)
        if len(raw) != want:
            raise ValueError(f"operand is {len(raw)} bytes, expected {num_rows}*{k}*2 = {want}")
        return raw

    # torch tensor -> contiguous CPU, viewed as the underlying u16 bytes.
    if type(rows).__module__.split(".", 1)[0] == "torch":
        import torch  # local import: the opener stays torch-free when unused

        t = rows.detach().to("cpu").contiguous().reshape(num_rows, k)
        if t.dtype == torch.float16:
            t = t.view(torch.int16)
        elif t.dtype not in (torch.int16, torch.uint16):
            raise ValueError(f"expected float16 / int16 FP16 bit patterns, got {t.dtype}")
        raw = t.view(torch.uint8).numpy().tobytes()
        if len(raw) != want:
            raise ValueError(f"operand tensor is {len(raw)} bytes, expected {want}")
        return raw

    import numpy as np

    arr = np.ascontiguousarray(rows)
    if arr.dtype == np.float16:
        arr = arr.view(np.uint16)
    elif arr.dtype == np.int16:
        arr = arr.view(np.uint16)
    elif arr.dtype != np.uint16:
        # A flat int sequence (or a wider int array): take the low 16 bits.
        arr = (arr.astype(np.int64) & 0xFFFF).astype(np.uint16)
    raw = arr.astype("<u2", copy=False).reshape(-1).tobytes()
    if len(raw) != want:
        raise ValueError(f"operand has {len(raw)} bytes, expected {num_rows}*{k}*2 = {want}")
    return raw


@dataclass(frozen=True)
class Fp16MatrixMerkleProof:
    """A Merkle opening plus the GLOBAL operand row indices it authenticates.

    ``row_indices`` are the opened tile's committed (global) rows -- for the
    contiguous A100 layout, ``tile_row * h + P.tile_offsets()``. The Rust
    ``Fp16PlainProof.parse_proof`` validates them as ``base + tile_offsets()``.
    """

    proof: "MerkleProof"
    row_indices: list[int]

    @property
    def root(self) -> bytes:
        return bytes(self.proof.root)


@dataclass
class Fp16MatrixCommitment:
    """A committed FP16 operand: its keyed Merkle tree (root = digest) plus the
    shape/leaf metadata needed to open the unique-minimal row set."""

    tree: "MerkleTree"
    root: bytes
    num_rows: int
    k: int
    hash_id: HashId

    @property
    def row_nbytes(self) -> int:
        """Committed bytes per row: ``k`` values, 2 bytes each."""
        return self.k * 2

    def open(self, row_indices: Sequence[int]) -> Fp16MatrixMerkleProof:
        """Open exactly ``row_indices`` as the unique-minimal leaf/sibling set,
        bit-exact to ``commitment.rs::open_rows``."""
        indices = [int(i) for i in row_indices]
        if not indices:
            raise ValueError("must open at least one row")
        if any(i < 0 or i >= self.num_rows for i in indices):
            raise ValueError(f"opened row index out of range for {self.num_rows} rows")
        leaf_idx = MerkleTree.compute_leaf_indices_from_rows(
            indices, (self.num_rows, self.row_nbytes), self.hash_id.chunk_len
        )
        proof = self.tree.get_multileaf_proof(leaf_idx)
        return Fp16MatrixMerkleProof(proof, indices)


def commit_fp16_operand(
    rows: object,
    num_rows: int,
    k: int,
    key: bytes,
    hash_id: HashId = DEFAULT_HASH_ID,
) -> Fp16MatrixCommitment:
    """Build the keyed Merkle tree committing an ``num_rows x k`` FP16 operand
    (row-major ``u16`` bit patterns) under ``key``, bit-exact to
    ``commitment.rs::commit_operand`` and the GPU ``fp16_commit`` kernel:
    ``MerkleTree(hash_id.pad(rows_to_bytes(rows)), key, hash_id.chunk_len)``.
    """
    raw = rows_to_bytes(rows, num_rows, k)
    tree = MerkleTree(data=hash_id.pad(raw), key=bytes(key), chunk_len=hash_id.chunk_len)
    return Fp16MatrixCommitment(tree, bytes(tree.root), num_rows, k, hash_id)
