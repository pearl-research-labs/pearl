#!/usr/bin/env python3
"""Akoya BSeed expansion helpers.

Akoya rejects shares whose B commitment does not match the deterministic
matrix expansion implied by the live job seed. The public Akoya mining CAPI
exports the same operation as ``pearl_capi_bseed_expand_and_merkle``; this
module is the Python oracle used before spending more H200 time.
"""

from __future__ import annotations

import blake3


CHUNK_SIZE = 1024
HASH_SIZE = 32


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def int7_from_xof_byte(value: int) -> int:
    """Map one BLAKE3 XOF byte into Akoya's signed int7 byte representation."""

    return ((value % 127) - 63) & 0xFF


def expand_bseed_matrix(seed_hash: bytes, n: int, k: int) -> bytes:
    """Return the deterministic ``B^T`` matrix bytes for a job seed.

    The returned bytes are uint8 two's-complement storage for int8 values in
    ``[-63, 63]`` and are laid out as ``n`` contiguous rows of width ``k``.
    """

    _require(len(seed_hash) == HASH_SIZE, f"seed_hash must be {HASH_SIZE} bytes")
    _require(n > 0 and k > 0, "n and k must be positive")
    raw = blake3.blake3(seed_hash).digest(length=n * k)
    return bytes(int7_from_xof_byte(value) for value in raw)


def padded_for_merkle(data: bytes) -> bytes:
    """Pad matrix bytes to the BLAKE3 chunk boundary used by Pearl Merkle roots."""

    remainder = len(data) % CHUNK_SIZE
    if remainder == 0:
        return data
    return data + b"\x00" * (CHUNK_SIZE - remainder)


def hash_b_for_bseed(seed_hash: bytes, n: int, k: int, job_key: bytes) -> bytes:
    """Compute Akoya/Pearl HashB for the deterministic BSeed expansion."""

    _require(len(job_key) == HASH_SIZE, f"job_key must be {HASH_SIZE} bytes")
    matrix = expand_bseed_matrix(seed_hash, n, k)
    return blake3.blake3(padded_for_merkle(matrix), key=job_key).digest()


def job_key_for_share(header_bytes: bytes, mining_config_bytes: bytes) -> bytes:
    """Return Pearl's commitment key for a header/config pair."""

    return blake3.blake3(header_bytes + mining_config_bytes).digest()


def opened_b_from_leaf_indices(seed_hash: bytes, n: int, k: int, leaf_indices: tuple[int, ...] | list[int]) -> bytes:
    """Extract opened B chunks from deterministic BSeed expansion."""

    matrix = expand_bseed_matrix(seed_hash, n, k)
    total_leaves = (len(matrix) + CHUNK_SIZE - 1) // CHUNK_SIZE
    chunks: list[bytes] = []
    for index in leaf_indices:
        _require(0 <= int(index) < total_leaves, f"leaf index out of range: {index}")
        start = int(index) * CHUNK_SIZE
        chunks.append(matrix[start : start + CHUNK_SIZE])
    return b"".join(chunks)
