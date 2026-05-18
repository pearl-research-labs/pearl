#!/usr/bin/env python3
"""Decode Pearl PlainProof bincode blobs produced by PlainProof.to_base64()."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import struct


CHUNK_SIZE = 1024
HASH_SIZE = 32


class PlainProofDecodeError(ValueError):
    pass


class _Cursor:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def read(self, nbytes: int) -> bytes:
        end = self.offset + nbytes
        if end > len(self.data):
            raise PlainProofDecodeError(f"truncated bincode at offset {self.offset}, need {nbytes} bytes")
        out = self.data[self.offset : end]
        self.offset = end
        return out

    def u64(self) -> int:
        out = struct.unpack_from("<Q", self.data, self.offset)[0]
        self.offset += 8
        return out

    def finish(self) -> None:
        if self.offset != len(self.data):
            raise PlainProofDecodeError(f"trailing bincode bytes: offset {self.offset}, len {len(self.data)}")


@dataclass(frozen=True)
class DecodedMerkleProof:
    leaf_data: tuple[bytes, ...]
    leaf_indices: tuple[int, ...]
    total_leaves: int
    root: bytes
    siblings: tuple[bytes, ...]


@dataclass(frozen=True)
class DecodedMatrixMerkleProof:
    proof: DecodedMerkleProof
    row_indices: tuple[int, ...]


@dataclass(frozen=True)
class DecodedPlainProof:
    m: int
    n: int
    k: int
    noise_rank: int
    a: DecodedMatrixMerkleProof
    bt: DecodedMatrixMerkleProof


def _read_u64_vec(cursor: _Cursor) -> tuple[int, ...]:
    length = cursor.u64()
    return tuple(cursor.u64() for _ in range(length))


def _read_fixed_bytes_vec(cursor: _Cursor, item_size: int) -> tuple[bytes, ...]:
    length = cursor.u64()
    return tuple(cursor.read(item_size) for _ in range(length))


def _read_leaf_data_vec(cursor: _Cursor) -> tuple[bytes, ...]:
    # MerkleProof.leaf_data uses serde_chunk_vec, which serializes Vec<[u8;1024]>
    # as Vec<&[u8]>. Bincode therefore stores one length prefix for the vector
    # and one length prefix for each byte slice.
    length = cursor.u64()
    out: list[bytes] = []
    for _ in range(length):
        item_len = cursor.u64()
        if item_len != CHUNK_SIZE:
            raise PlainProofDecodeError(f"leaf chunk must be {CHUNK_SIZE} bytes, got {item_len}")
        out.append(cursor.read(CHUNK_SIZE))
    return tuple(out)


def _read_merkle_proof(cursor: _Cursor) -> DecodedMerkleProof:
    leaf_data = _read_leaf_data_vec(cursor)
    leaf_indices = _read_u64_vec(cursor)
    total_leaves = cursor.u64()
    root = cursor.read(HASH_SIZE)
    siblings = _read_fixed_bytes_vec(cursor, HASH_SIZE)
    if len(leaf_data) != len(leaf_indices):
        raise PlainProofDecodeError("leaf_data and leaf_indices length mismatch")
    return DecodedMerkleProof(
        leaf_data=leaf_data,
        leaf_indices=leaf_indices,
        total_leaves=total_leaves,
        root=root,
        siblings=siblings,
    )


def _read_matrix_merkle_proof(cursor: _Cursor) -> DecodedMatrixMerkleProof:
    proof = _read_merkle_proof(cursor)
    row_indices = _read_u64_vec(cursor)
    return DecodedMatrixMerkleProof(proof=proof, row_indices=row_indices)


def decode_plain_proof_bytes(data: bytes) -> DecodedPlainProof:
    cursor = _Cursor(data)
    proof = DecodedPlainProof(
        m=cursor.u64(),
        n=cursor.u64(),
        k=cursor.u64(),
        noise_rank=cursor.u64(),
        a=_read_matrix_merkle_proof(cursor),
        bt=_read_matrix_merkle_proof(cursor),
    )
    cursor.finish()
    return proof


def decode_plain_proof_base64(data: str) -> DecodedPlainProof:
    return decode_plain_proof_bytes(base64.b64decode(data))
