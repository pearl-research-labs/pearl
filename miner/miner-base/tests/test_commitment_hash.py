import secrets

import pytest
import torch
from miner_base.commitment_hash import CommitmentHasher, bind_root_a, bind_root_b
from pearl_gateway.comm.dataclasses import CommitmentHash
from pearl_gateway.comm.mining_configuration import MiningConfiguration
from pearl_mining import IncompleteBlockHeader


@pytest.fixture
def incomplete_header_bytes() -> bytes:
    """Generate a random header bytes."""
    return secrets.token_bytes(IncompleteBlockHeader.SERIALIZED_SIZE)


@pytest.fixture
def mining_config(default_matmul_config) -> MiningConfiguration:
    """Generate a test mining configuration."""
    return default_matmul_config.mining_config


class TestCommitmentHasher:
    """Test suite for CommitmentHasher class."""

    def test_commitment_hash_basic(self, incomplete_header_bytes, mining_config):
        """Test basic commitment hash functionality."""
        A = torch.randint(-128, 127, (8, 8), dtype=torch.int8)
        B = torch.randint(-128, 127, (8, 8), dtype=torch.int8)

        result = CommitmentHasher.commitment_hash(
            A, B, incomplete_header_bytes, mining_config, salted_dims=None
        )

        assert isinstance(result, CommitmentHash)
        assert len(result.noise_seed_A) == 32
        assert len(result.noise_seed_B) == 32

    def test_commitment_hash_deterministic(self, incomplete_header_bytes, mining_config):
        """Test that commitment hash is deterministic."""
        A = torch.randint(-128, 127, (8, 8), dtype=torch.int8)
        B = torch.randint(-128, 127, (8, 8), dtype=torch.int8)

        result1 = CommitmentHasher.commitment_hash(
            A, B, incomplete_header_bytes, mining_config, salted_dims=None
        )
        result2 = CommitmentHasher.commitment_hash(
            A, B, incomplete_header_bytes, mining_config, salted_dims=None
        )

        assert result1 == result2

    def test_commitment_hash_different_order(self, incomplete_header_bytes, mining_config):
        """Test that different order produce different hashes."""

        A = torch.zeros((8, 8), dtype=torch.int8)
        B = torch.ones((8, 8), dtype=torch.int8)

        result1 = CommitmentHasher.commitment_hash(
            A, B, incomplete_header_bytes, mining_config, salted_dims=None
        )
        result2 = CommitmentHasher.commitment_hash(
            B, A, incomplete_header_bytes, mining_config, salted_dims=None
        )

        assert result1 != result2

    def test_commitment_hash_invalid_dimensions(self, incomplete_header_bytes, mining_config):
        """Test error handling for non-2D tensors."""

        # 1D tensor
        tensor_1d = torch.randint(-128, 127, (8,), dtype=torch.int8)
        with pytest.raises(ValueError, match="Expected 2D tensor"):
            CommitmentHasher.commitment_hash(
                tensor_1d, tensor_1d, incomplete_header_bytes, mining_config, salted_dims=None
            )

        # 3D tensor
        tensor_3d = torch.randint(-128, 127, (8, 8, 3), dtype=torch.int8)
        with pytest.raises(ValueError, match="Expected 2D tensor"):
            CommitmentHasher.commitment_hash(
                tensor_3d, tensor_3d, incomplete_header_bytes, mining_config, salted_dims=None
            )


@pytest.fixture
def job_key() -> bytes:
    """Return a fixed job key for reproducible hashes."""
    return bytes([0x11] * 32)


@pytest.fixture
def root_a() -> bytes:
    """Return a fixed A root for reproducible hashes."""
    return bytes([0xAA] * 32)


@pytest.fixture
def root_b() -> bytes:
    """Return a fixed B root for reproducible hashes."""
    return bytes([0xBB] * 32)


@pytest.fixture
def matrix_dimensions() -> tuple[int, int]:
    """Return fixed matrix dimensions for salted hashes."""
    return 192, 320


@pytest.fixture
def moe_roots() -> dict[str, bytes]:
    """Return fixed MoE roots for reproducible hashes."""
    return {
        "routing_root": bytes([0xCC] * 32),
        "offsets_root": bytes([0xDD] * 32),
    }


class TestSaltedSeedDerivation:
    """Verify vectors shared with the Rust seed tests."""

    def test_legacy_pinned_vector(self, root_a, root_b, job_key):
        """Test the legacy noise-seed chain."""
        result = CommitmentHasher.commitment_hash_from_merkle_roots(
            root_a,
            root_b,
            job_key,
            salted_dims=None,
        )
        assert (
            result.noise_seed_B.hex()
            == "add6f7ea5feebf89c8a77e2ebfa0d82442e7dbb0046dbd48971861d12fcb0177"
        )
        assert (
            result.noise_seed_A.hex()
            == "483b07b6f73105030b9482255f37723f3fed69ae916724ee8291848b8c28794b"
        )

    def test_salted_pinned_vector(self, root_a, root_b, job_key, matrix_dimensions):
        """Test the salted noise-seed chain."""
        result = CommitmentHasher.commitment_hash_from_merkle_roots(
            root_a,
            root_b,
            job_key,
            salted_dims=matrix_dimensions,
        )
        assert (
            result.noise_seed_B.hex()
            == "60ed9b73c5a9599b200b6cd563e7f0d5d9a67d2402d85fd4ef966c580080d0e5"
        )
        assert (
            result.noise_seed_A.hex()
            == "301784168005ec833ab0aa60006f7fe7faaa95307d8c1fc6819b2ffdd717eccf"
        )

    def test_salted_moe_pinned_vector(self, root_a, root_b, job_key, matrix_dimensions):
        """Pinned vector shared with the Rust test (`seed.rs::commitment_hash_pinned_vector_salted_moe`)."""
        result = CommitmentHasher.commitment_hash_from_merkle_roots(
            root_a,
            root_b,
            job_key,
            routing_root=bytes([0xCC] * 32),
            offsets_root=CommitmentHasher.get_offsets_hash([3, 5, 9, 12], job_key),
            salted_dims=matrix_dimensions,
        )
        assert (
            result.noise_seed_B.hex()
            == "60ed9b73c5a9599b200b6cd563e7f0d5d9a67d2402d85fd4ef966c580080d0e5"
        )
        assert (
            result.noise_seed_A.hex()
            == "268a2eb78b3b2d2fead262221b24c6346d0a6d201890a467751f623b7eacd5c2"
        )

    def test_salted_moe_fold_order(
        self,
        root_a,
        root_b,
        job_key,
        matrix_dimensions,
        moe_roots,
    ):
        """Test that MoE roots are salted before the routing fold."""
        matrix_rows, matrix_columns = matrix_dimensions
        prebound = CommitmentHasher.commitment_hash_from_merkle_roots(
            bind_root_a(root_a, matrix_rows),
            bind_root_b(root_b, matrix_columns),
            job_key,
            salted_dims=None,
            **moe_roots,
        )
        salted = CommitmentHasher.commitment_hash_from_merkle_roots(
            root_a,
            root_b,
            job_key,
            salted_dims=matrix_dimensions,
            **moe_roots,
        )
        assert prebound == salted
