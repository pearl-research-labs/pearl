"""Torch-free Pearl proof builder for rig deployment (Python 3.10 + numpy only).

Bit-exact replacement for `miner_base.block_submission.create_proof` on hosts
WITHOUT torch (the mining rigs run Python 3.10 with no torch). The canonical
miner_base path uses torch ONLY to turn the int8 A/B matrices into row-major
bytes for the merkle tree; everything else (MerkleTree, multileaf proofs,
PlainProof serialization, MiningConfiguration) already lives in the torch-free
Rust `pearl_mining` extension (built abi3-py310). This module reproduces the
SAME bytes the torch path produces, so `pearl_mining.verify_plain_proof`
accepts proofs built here — and the canary verifies locally before any submit,
so a divergence can never reach the pool.

Mirrors, line-for-line:
  - miner_base.matrix_merkle_tree.MatrixMerkleTree
  - miner_base.commitment_hash.CommitmentHasher.get_key
  - miner_base.block_submission.create_proof
  - pearl_gateway.comm.dataclasses.OpenedBlockInfo (+ PearlMiningConfigurationFactory.create)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from blake3 import blake3
from pearl_mining import (
    MERKLE_LEAF_SIZE,
    MMAType,
    MatrixMerkleProof,
    MerkleTree,
    MiningConfiguration,
    PeriodicPattern,
    PlainProof,
    pad_to_chunk_boundary,
)


class MatrixMerkleTree:
    """numpy port of miner_base.matrix_merkle_tree.MatrixMerkleTree (2D int8)."""

    LEAF_SIZE = MERKLE_LEAF_SIZE

    def __init__(self, arr: np.ndarray, key: bytes):
        if arr.size == 0:
            raise ValueError("tensor must be non-empty")
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {arr.ndim}D tensor")
        if arr.dtype != np.int8:
            raise ValueError(f"Expected int8 tensor, got {arr.dtype}")
        if len(key) != 32:
            raise ValueError(f"Expected 32-byte key, got {len(key)} bytes")
        self.tensor_shape = tuple(int(d) for d in arr.shape)
        self._tree = MerkleTree(data=self.pad_tensor(arr), key=key)

    @staticmethod
    def pad_tensor(arr: np.ndarray) -> bytes:
        # Row-major (C-order) int8 bytes — identical to the torch path's
        # `tensor.flatten().detach().cpu().numpy().tobytes()`.
        raw = np.ascontiguousarray(arr, dtype=np.int8).tobytes()
        return pad_to_chunk_boundary(raw)

    @property
    def root(self) -> bytes:
        return self._tree.root

    def leaf_indices_from_rows(self, row_indices: list[int]) -> list[int]:
        return MerkleTree.compute_leaf_indices_from_rows(row_indices, self.tensor_shape)

    def get_multileaf_proof(self, leaf_indices: list[int]) -> "MerkleProof":
        return self._tree.get_multileaf_proof(leaf_indices)


def get_key(incomplete_header_bytes: bytes, mining_config: MiningConfiguration) -> bytes:
    """Identical to miner_base.commitment_hash.CommitmentHasher.get_key."""
    return blake3(incomplete_header_bytes + mining_config.to_bytes()).digest()


@dataclass
class OpenedBlockInfo:
    """Torch-free port of pearl_gateway.comm.dataclasses.OpenedBlockInfo.

    A / B_t are numpy int8 arrays of shape (m, k) / (n, k). `commitment_hash`
    is unused by create_proof (kept for signature parity).
    """

    A_row_indices: list[int]
    B_column_indices: list[int]
    A: np.ndarray
    B_t: np.ndarray
    commitment_hash: Optional[object] = None
    noise_rank: int = 256

    def get_mining_config(self) -> MiningConfiguration:
        # Mirrors PearlMiningConfigurationFactory.create.
        row_offset = min(self.A_row_indices)
        col_offset = min(self.B_column_indices)
        rows_pattern = PeriodicPattern.from_list(
            [i - row_offset for i in self.A_row_indices])
        cols_pattern = PeriodicPattern.from_list(
            [i - col_offset for i in self.B_column_indices])
        return MiningConfiguration(
            common_dim=int(self.A.shape[1]),
            rank=self.noise_rank,
            mma_type=MMAType.Int7xInt7ToInt32,
            rows_pattern=rows_pattern,
            cols_pattern=cols_pattern,
            reserved=MiningConfiguration.RESERVED,
        )


def create_proof(opened_block_info: OpenedBlockInfo,
                 incomplete_header_bytes: bytes) -> PlainProof:
    """numpy port of miner_base.block_submission.create_proof (bit-exact)."""
    A = opened_block_info.A
    B_t = opened_block_info.B_t
    mining_config = opened_block_info.get_mining_config()

    hash_key = get_key(incomplete_header_bytes, mining_config)
    A_merkle_tree = MatrixMerkleTree(A, hash_key)
    B_merkle_tree = MatrixMerkleTree(B_t, hash_key)

    a_merkle_proof = MatrixMerkleProof(
        proof=A_merkle_tree.get_multileaf_proof(
            A_merkle_tree.leaf_indices_from_rows(opened_block_info.A_row_indices)),
        row_indices=opened_block_info.A_row_indices,
    )
    b_merkle_proof = MatrixMerkleProof(
        proof=B_merkle_tree.get_multileaf_proof(
            B_merkle_tree.leaf_indices_from_rows(opened_block_info.B_column_indices)),
        row_indices=opened_block_info.B_column_indices,
    )

    m, k = A.shape
    n, k2 = B_t.shape
    assert k == k2, f"Common dimension mismatch: {k} != {k2}"

    return PlainProof(
        m=int(m),
        n=int(n),
        k=int(k),
        noise_rank=opened_block_info.noise_rank,
        a_merkle_proof=a_merkle_proof,
        bt_merkle_proof=b_merkle_proof,
    )
