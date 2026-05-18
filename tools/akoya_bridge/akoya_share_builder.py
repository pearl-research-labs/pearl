#!/usr/bin/env python3
"""Build Akoya type-3 PlainProofShare frames from Pearl PlainProof objects."""

from __future__ import annotations

from pathlib import Path
import struct
from typing import Any
import uuid

from akoya_protocol import (
    HASH_SIZE,
    HEADER_SIZE,
    MINING_CONFIG_SIZE,
    MatrixProofWire,
    PlainProofShare,
)
from plain_proof_bincode import DecodedMatrixMerkleProof, DecodedPlainProof, decode_plain_proof_base64


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _matrix_wire(matrix_proof: DecodedMatrixMerkleProof) -> MatrixProofWire:
    return MatrixProofWire(
        leaf_data=matrix_proof.proof.leaf_data,
        leaf_indices=matrix_proof.proof.leaf_indices,
        total_leaves=matrix_proof.proof.total_leaves,
        siblings=matrix_proof.proof.siblings,
    )


def _offset(indices: tuple[int, ...], name: str) -> int:
    _require(bool(indices), f"{name} row_indices must be non-empty")
    return min(indices)


def build_share_from_decoded_plain_proof(
    *,
    plain_proof: DecodedPlainProof,
    header_bytes: bytes,
    seed_hash: bytes,
    share_difficulty: int,
    mining_config_bytes: bytes,
    compact_result: bytes,
    claimed_hash: bytes,
    share_id: str | None = None,
) -> PlainProofShare:
    """Build an Akoya wire share from decoded local PlainProof internals."""

    _require(len(header_bytes) == HEADER_SIZE, f"header_bytes must be {HEADER_SIZE} bytes")
    _require(len(seed_hash) == HASH_SIZE, f"seed_hash must be {HASH_SIZE} bytes")
    _require(len(mining_config_bytes) == MINING_CONFIG_SIZE, f"mining_config_bytes must be {MINING_CONFIG_SIZE} bytes")
    _require(len(compact_result) == 64, "compact_result must be 64 bytes")
    _require(len(claimed_hash) == HASH_SIZE, f"claimed_hash must be {HASH_SIZE} bytes")

    share = PlainProofShare(
        share_id=share_id or str(uuid.uuid4()),
        header_bytes=header_bytes,
        opened_a=b"".join(plain_proof.a.proof.leaf_data),
        seed_hash=seed_hash,
        t_rows=_offset(plain_proof.a.row_indices, "A"),
        t_cols=_offset(plain_proof.bt.row_indices, "B"),
        compact_result=compact_result,
        claimed_hash=claimed_hash,
        share_difficulty=share_difficulty,
        hash_a=plain_proof.a.proof.root,
        hash_b=plain_proof.bt.proof.root,
        opened_b=b"".join(plain_proof.bt.proof.leaf_data),
        mining_config_bytes=mining_config_bytes,
        a_proof=_matrix_wire(plain_proof.a),
        b_proof=_matrix_wire(plain_proof.bt),
    )
    share.validate()
    return share


def canonical_jackpot_fields(pm: Any, header: Any, plain_proof: Any) -> tuple[bytes, bytes]:
    """Return Akoya fields 6 and 7 for a pearl_mining.PlainProof object."""

    diagnostic = pm.diagnostic_plain_proof_jackpot_controls(header, plain_proof)
    jackpot_words = [int(word) for word in diagnostic[0]]
    compact_result = b"".join(struct.pack("<I", word) for word in jackpot_words)
    claimed_hash = bytes.fromhex(diagnostic[5])
    return compact_result, claimed_hash


def build_share_from_pearl_plain_proof(
    *,
    pm: Any,
    plain_proof: Any,
    header: Any,
    header_bytes: bytes,
    seed_hash: bytes,
    share_difficulty: int,
    mining_config_bytes: bytes,
    share_id: str | None = None,
) -> PlainProofShare:
    """Build Akoya type-3 share fields from a live pearl_mining.PlainProof object."""

    compact_result, claimed_hash = canonical_jackpot_fields(pm, header, plain_proof)
    decoded = decode_plain_proof_base64(plain_proof.to_base64())
    return build_share_from_decoded_plain_proof(
        plain_proof=decoded,
        header_bytes=header_bytes,
        seed_hash=seed_hash,
        share_difficulty=share_difficulty,
        mining_config_bytes=mining_config_bytes,
        compact_result=compact_result,
        claimed_hash=claimed_hash,
        share_id=share_id,
    )

