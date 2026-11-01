import torch
from miner_utils import get_logger
from pearl_mining import MERKLE_LEAF_SIZE, MerkleProof, MerkleTree, pad_to_chunk_boundary

_LOGGER = get_logger(__name__)


class MatrixMerkleTree:
    """
    A Merkle tree built from a 2D int8 tensor using BLAKE3 via pearl_mining.
    """

    LEAF_SIZE = MERKLE_LEAF_SIZE

    def __init__(self, tensor: torch.Tensor, key: bytes):
        if tensor.numel() == 0:
            raise ValueError("tensor must be non-empty")
        if tensor.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {tensor.dim()}D tensor")
        if tensor.dtype != torch.int8:
            raise ValueError(f"Expected int8 tensor, got {tensor.dtype}")
        if len(key) != 32:
            raise ValueError(f"Expected 32-byte key, got {len(key)} bytes")

        self.tensor_shape = tensor.shape
        self._tree = MerkleTree.from_buffer(tensor.detach().cpu().contiguous().numpy(), key)

    @classmethod
    def tensor_hash(cls, tensor: torch.Tensor, key: bytes) -> bytes:
        """Hash a tensor with blake3 keyed hash. Pads to LEAF_SIZE. CPU version."""
        from blake3 import blake3

        return blake3(cls.pad_tensor(tensor), key=key).digest()

    @property
    def root(self) -> bytes:
        return self._tree.root

    @property
    def leaf_hashes(self) -> list[bytes]:
        return self._tree.leaf_hashes

    @classmethod
    def pad_tensor(cls, tensor: torch.Tensor) -> bytes:
        """Convert tensor to bytes, padded to the BLAKE3 chunk boundary."""
        raw = tensor.flatten().detach().cpu().numpy().tobytes()
        return pad_to_chunk_boundary(raw)

    def leaf_indices_from_rows(self, row_indices: list[int]) -> list[int]:
        return MerkleTree.compute_leaf_indices_from_rows(row_indices, self.tensor_shape)

    def get_multileaf_proof(self, leaf_indices: list[int]) -> MerkleProof:
        """Generates a multiproof for an arbitrary set of leaves."""
        return self._tree.get_multileaf_proof(leaf_indices)
