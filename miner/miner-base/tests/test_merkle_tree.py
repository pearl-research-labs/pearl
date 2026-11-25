import math

import pytest
import torch
from blake3 import blake3
from miner_base.matrix_merkle_tree import MatrixMerkleTree


@pytest.fixture
def sample_key():
    """Sample 32-byte key for testing."""
    return b"0123456789abcdef" * 2  # 32 bytes


class TestMatrixMerkleTree:
    """Test suite for MatrixMerkleTree class."""

    @pytest.fixture
    def sample_tensor(self):
        """Sample tensor for testing - 128x128 int8 tensor."""
        torch.manual_seed(42)
        return torch.randint(-128, 127, (128, 128), dtype=torch.int8)

    @pytest.fixture
    def merkle_tree(self, sample_tensor, sample_key):
        """Merkle tree fixture."""
        return MatrixMerkleTree(sample_tensor, sample_key)

    def test_merkle_tree_initialization(self, merkle_tree, sample_tensor):
        """Test that MatrixMerkleTree initializes correctly."""
        assert merkle_tree.root is not None
        assert len(merkle_tree.leaf_hashes) > 0

        # Verify number of leaves is correct
        assert len(merkle_tree.leaf_hashes) == math.ceil(
            sample_tensor.numel() / MatrixMerkleTree.LEAF_SIZE
        )

    def test_tensor_tiling_correctness(self, sample_key):
        """Test that tensor tiling works correctly."""
        # Create a known tensor pattern
        tensor1 = torch.zeros((2, 1024), dtype=torch.int8)
        tensor2 = torch.zeros((2, 1024), dtype=torch.int8)
        # Fill first leaf (0, :) with 1s
        tensor1[0, :] = 1
        # Fill second leaf (1, 0) with 2s
        tensor2[0, :] = 2

        tree1 = MatrixMerkleTree(tensor1, sample_key)
        tree2 = MatrixMerkleTree(tensor2, sample_key)

        # Should have 2 leaves
        assert len(tree1.leaf_hashes) == 2
        assert len(tree2.leaf_hashes) == 2

        # First two leaves should have different hashes (different values)
        assert tree1.leaf_hashes[0] != tree2.leaf_hashes[0]

        # Second leaf should be the same (both zeros)
        assert tree1.leaf_hashes[1] == tree2.leaf_hashes[1]

    def test_tensor_hash_single_tile(self, sample_key):
        """Test with tensor smaller than tile size."""
        tensor = torch.randint(-128, 127, (1, MatrixMerkleTree.LEAF_SIZE), dtype=torch.int8)

        result = MatrixMerkleTree(tensor, sample_key)

        assert (
            result.root == blake3(tensor.detach().cpu().numpy().tobytes(), key=sample_key).digest()
        )

    def test_merkle_tree_empty_list(self, sample_key):
        """Test Merkle tree error handling for empty list."""
        with pytest.raises(ValueError, match="tensor must be non-empty"):
            MatrixMerkleTree(torch.empty((0, 1024), dtype=torch.int8), sample_key)

    def test_invalid_key_length(self, sample_tensor):
        """Test error handling for invalid key length in commitment_hash."""

        # Test with short key
        with pytest.raises(ValueError, match="Expected 32-byte key, got 8 bytes"):
            MatrixMerkleTree(sample_tensor, b"shortkey")

        # Test with long key
        long_key = b"a" * 40
        with pytest.raises(ValueError, match="Expected 32-byte key, got 40 bytes"):
            MatrixMerkleTree(sample_tensor, long_key)

    def test_different_tensors_same_structure(self, sample_key):
        """Test that different tensors with same structure produce different roots."""
        tensor1 = torch.zeros((32, 128), dtype=torch.int8)
        tensor2 = torch.ones((32, 128), dtype=torch.int8)

        tree1 = MatrixMerkleTree(tensor1, sample_key)
        tree2 = MatrixMerkleTree(tensor2, sample_key)

        # Roots should be different
        assert tree1.root != tree2.root

        # But structure should be the same
        assert len(tree1.leaf_hashes) == len(tree2.leaf_hashes)

    def test_different_keys_same_tensor(self):
        """Test that different keys produce different roots for same tensor."""
        tensor = torch.randint(-128, 127, (32, 128), dtype=torch.int8)
        key1 = b"0123456789abcdef" * 2
        key2 = b"fedcba9876543210" * 2

        tree1 = MatrixMerkleTree(tensor, key1)
        tree2 = MatrixMerkleTree(tensor, key2)

        # Roots should be different due to different keys
        assert tree1.root != tree2.root
        assert len(tree1.leaf_hashes) == len(tree2.leaf_hashes)
        # Leaf hashes should all be different due to different keys
        for hash1, hash2 in zip(tree1.leaf_hashes, tree2.leaf_hashes, strict=True):
            assert hash1 != hash2


class TestMatrixMerkleTreeVsBlake3:
    """Tests that verify MatrixMerkleTree root matches blake3 hash of padded tensor."""

    @pytest.mark.parametrize(
        "m, n",
        [
            (1, 1024),  # Single row, exactly one chunk
            (2, 512),  # Two rows, one chunk
            (3, 1024),  # Multiple rows
            (150, 512),  # Larger matrix
            (1, 1023),  # Not chunk aligned
            (2, 1025),  # Just over one chunk
            (1, 100),  # Small tensor needing padding
            (8192, 14336),  # very large tensor
        ],
    )
    def test_root_matches_blake3(self, sample_key, m, n):
        """Test that tree root matches direct blake3 hash of padded tensor."""
        tensor = torch.randint(-128, 127, (m, n), dtype=torch.int8)

        tree = MatrixMerkleTree(tensor, sample_key)

        # Get padded tensor bytes
        padded_tensor_bytes = tree.pad_tensor(tensor)

        # Direct blake3 hash
        expected_root = blake3(padded_tensor_bytes, key=sample_key).digest()

        assert tree.root == expected_root, f"Root mismatch for shape {m=}, {n=}"

    def test_root_of_non_contiguous_tensor(self, sample_key):
        """Test that a transposed view hashes the same as its contiguous copy."""
        tensor = torch.randint(-128, 127, (300, 700), dtype=torch.int8).T
        tree = MatrixMerkleTree(tensor, sample_key)
        expected_root = blake3(tree.pad_tensor(tensor), key=sample_key).digest()
        assert tree.root == expected_root
