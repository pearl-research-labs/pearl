import torch
from blake3 import blake3
from pearl_gateway.comm.dataclasses import CommitmentHash
from pearl_mining import MiningConfiguration

from .matrix_merkle_tree import MatrixMerkleTree


class CommitmentHasher:
    """
    A namespace for commitment hash functions.
    """

    @staticmethod
    def get_key(incomplete_header_bytes: bytes, mining_config: MiningConfiguration) -> bytes:
        return blake3(incomplete_header_bytes + mining_config.to_bytes()).digest()

    @classmethod
    def commitment_hash(
        cls,
        A: torch.Tensor,
        B: torch.Tensor,
        incomplete_header_bytes: bytes,
        mining_config: MiningConfiguration,
    ) -> CommitmentHash:
        key = cls.get_key(incomplete_header_bytes, mining_config)
        merkle_tree_A = MatrixMerkleTree(A, key)

        # We hash B.T because we would like to expose a column strip of B
        merkle_tree_B = MatrixMerkleTree(B.T, key)

        return cls.commitment_hash_from_merkle_roots(merkle_tree_A.root, merkle_tree_B.root, key)

    @staticmethod
    def get_commitment_B_key(key: bytes, B_merkle_root: bytes) -> bytes:
        return blake3(key + B_merkle_root).digest()

    @staticmethod
    def get_commitment_A_key(commitment_B: bytes, A_merkle_root: bytes) -> bytes:
        return blake3(commitment_B + A_merkle_root).digest()

    @staticmethod
    def get_offsets_hash(routing_offsets: list[int], key: bytes) -> bytes:
        offsets = torch.tensor(routing_offsets, dtype=torch.int32)
        return MatrixMerkleTree.tensor_hash(offsets, key)

    @staticmethod
    def combine_routing_merkle_roots(
        hash_a: bytes, routing_root: bytes, offsets_root: bytes
    ) -> bytes:
        hash_routing = blake3(routing_root + offsets_root).digest()
        return blake3(hash_a + hash_routing).digest()

    @classmethod
    def commitment_hash_from_merkle_roots(
        cls,
        A_merkle_root: bytes,
        B_merkle_root: bytes,
        key: bytes,
        *,
        routing_root: bytes | None = None,
        offsets_root: bytes | None = None,
    ) -> CommitmentHash:
        moe_args = (routing_root, offsets_root)
        if any(arg is not None for arg in moe_args):
            if any(arg is None for arg in moe_args):
                raise ValueError("routing_root and offsets_root must be provided together")
            A_merkle_root = cls.combine_routing_merkle_roots(
                A_merkle_root, routing_root, offsets_root
            )

        commitment_B = cls.get_commitment_B_key(key, B_merkle_root)
        commitment_A = cls.get_commitment_A_key(commitment_B, A_merkle_root)
        return CommitmentHash(commitment_A, commitment_B)
