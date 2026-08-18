import torch
from blake3 import blake3
from pearl_gateway.comm.dataclasses import CommitmentHash
from pearl_mining import MiningConfiguration

from .matrix_merkle_tree import MatrixMerkleTree

# Domain-separation salts for the V3 (salted noise-seed) certificate derivation:
SEED_SALT_A = blake3(b"pearl/cert-v3/noise-seed/A").digest()
SEED_SALT_B = blake3(b"pearl/cert-v3/noise-seed/B").digest()

# root(32) || dim u32 LE(4) || zero padding(28) = one 64-byte BLAKE3 block.
_BIND_PAD = b"\x00" * 28


def _bind_root(merkle_root: bytes, dim: int, salt: bytes) -> bytes:
    return blake3(merkle_root + dim.to_bytes(4, "little") + _BIND_PAD, key=salt).digest()


def bind_root_a(a_merkle_root: bytes, m: int) -> bytes:
    """V3 salting of A's Merkle root; commits the row count ``m``."""
    return _bind_root(a_merkle_root, m, SEED_SALT_A)


def bind_root_b(b_merkle_root: bytes, n: int) -> bytes:
    """V3 salting of B's Merkle root; commits the column count ``n``."""
    return _bind_root(b_merkle_root, n, SEED_SALT_B)


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
        *,
        salted_dims: tuple[int, int] | None,
    ) -> CommitmentHash:
        key = cls.get_key(incomplete_header_bytes, mining_config)
        merkle_tree_A = MatrixMerkleTree(A, key)

        # We hash B.T because we would like to expose a column strip of B
        merkle_tree_B = MatrixMerkleTree(B.T, key)

        return cls.commitment_hash_from_merkle_roots(
            merkle_tree_A.root, merkle_tree_B.root, key, salted_dims=salted_dims
        )

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
        salted_dims: tuple[int, int] | None,
    ) -> CommitmentHash:
        # salted_dims has no default so callers must choose a derivation explicitly:
        # (m, n) for V3 salted seeds, None for the legacy (V1/V2) chain.
        # V3 (salted) derivation: salt each root before the MoE routing fold.
        if salted_dims is not None:
            m, n = salted_dims
            A_merkle_root = bind_root_a(A_merkle_root, m)
            B_merkle_root = bind_root_b(B_merkle_root, n)

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
