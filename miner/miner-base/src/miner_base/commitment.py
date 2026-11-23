"""Keyed BLAKE3 Merkle commitments and openings for FP8/v4 operands and MoE routing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from blake3 import blake3
from pearl_blake3 import MerkleProof, MerkleTree

from .protocol import HashId, encode_u32_le, hash_id_for_leaf


def tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().flatten().view(torch.uint8).numpy().tobytes()


@dataclass(frozen=True)
class MatrixMerkleProof:
    proof: MerkleProof
    row_indices: list[int]

    @property
    def root(self) -> bytes:
        return bytes(self.proof.root)


@dataclass
class MatrixCommitment:
    tree: MerkleTree
    digest: bytes
    rows: int
    row_nbytes: int
    hash_id: HashId

    def open(self, row_indices: list[int]) -> MatrixMerkleProof:
        leaf_idx = MerkleTree.compute_leaf_indices_from_rows(
            list(row_indices),
            (self.rows, self.row_nbytes),
            self.hash_id.chunk_len,
        )
        return MatrixMerkleProof(self.tree.get_multileaf_proof(leaf_idx), list(row_indices))


def commit_matrix(t: torch.Tensor, key: bytes, hash_id: HashId) -> MatrixCommitment:
    assert t.dim() == 2
    row_nbytes = t.shape[1] * t.element_size()
    tree = MerkleTree(
        data=hash_id.pad(tensor_bytes(t)),
        key=key,
        chunk_len=hash_id.chunk_len,
    )
    return MatrixCommitment(tree, bytes(tree.root), t.shape[0], row_nbytes, hash_id)


@dataclass
class PlanarCommitment:
    parts: list[MatrixCommitment]
    digest: bytes

    def open(self, row_indices: list[int]) -> list[MatrixMerkleProof]:
        return [part.open(row_indices) for part in self.parts]


def commit_planes(
    planes: Sequence[torch.Tensor],
    key: bytes,
    hash_id: HashId | int,
) -> PlanarCommitment:
    """Commit each 2D plane under ``key`` and combine their roots into ``HA``/``HB``."""
    resolved = hash_id if isinstance(hash_id, HashId) else hash_id_for_leaf(hash_id)
    assert len(planes) >= 1
    rows = planes[0].shape[0]
    assert all(p.shape[0] == rows for p in planes), "planes must share the row count"
    parts = [commit_matrix(p, key, resolved) for p in planes]
    digest = blake3(b"".join(part.digest for part in parts), key=key).digest()
    return PlanarCommitment(parts, digest)


def commit_routing(routing_tokens: list[int], key_a: bytes, hash_id: HashId) -> MerkleTree:
    raw = b"".join(encode_u32_le(token) for token in routing_tokens)
    return MerkleTree(data=hash_id.pad(raw), key=key_a, chunk_len=hash_id.chunk_len)


def hash_offsets(end_offsets: list[int], key_a: bytes, hash_id: HashId) -> bytes:
    raw = b"".join(encode_u32_le(offset) for offset in end_offsets)
    tree = MerkleTree(data=hash_id.pad(raw), key=key_a, chunk_len=hash_id.chunk_len)
    return bytes(tree.root)


def routing_leaf_indices(start: int, end: int, hash_id: HashId) -> list[int]:
    if not 0 <= start < end:
        raise ValueError(f"routing slice [{start}, {end}) must be non-empty")
    chunk = hash_id.chunk_len
    return list(range(4 * start // chunk, -(-4 * end // chunk)))
