import pytest
import torch
from blake3 import blake3
from miner_base.commitment_hash import CommitmentHasher
from miner_base.matrix_merkle_tree import MatrixMerkleTree
from pearl_gemm import commitment_hash_from_merkle_roots, get_required_scratchpad_bytes, tensor_hash


def hash_matrix(matrix: torch.Tensor, key: bytes) -> torch.Tensor:
    """Reference implementation for tensor hash."""
    hash_bytes = MatrixMerkleTree.tensor_hash(matrix, key)
    return torch.frombuffer(hash_bytes, dtype=torch.uint8).to("cuda")


@pytest.fixture(autouse=True)
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TestTensorHash:
    """Test tensor hash functionality on different matrix sizes."""

    @pytest.mark.parametrize(
        "shape",
        [
            (16, 16),  # 256 B: single sub-1024-byte chunk
            (32, 32),  # 1024 B: exactly one full chunk
            (64, 64),  # 4 KiB: 4 chunks, single block
            (128, 128),  # 16 chunks, single block
            (256, 256),  # 64 chunks, single block
            (362, 362),  # 131044 B: 128 chunks, largest single block @128 threads
            (363, 363),  # 129 chunks: first size needing two blocks
            (1, 1056),  # 2 chunks with a sub-64-byte (32 B) trailing remainder
            (1, 2080),  # 3 chunks with a sub-64-byte (32 B) trailing remainder
            (8192, 8192),
            (1337, 8192),
            (4096, 2048),
            (2048, 4096),
            (512, 512),
            (777, 1024),
            (2048, 3072),
            (
                2000,
                2048,
            ),
            (7952, 1024),
            (7984, 1024),
            (8192, 28672),
            (28672, 8192),
            (57344, 8192),
            (8192, 57344),
            (16384, 57344),
            (57344, 16384),
            (1245, 5136),
            (12451, 23141),
            (12345, 22141),
        ],
    )
    def test_tensor_hash_shapes(self, shape):
        """Test that CUDA implementation matches Python reference for various shapes."""
        # Create random tensor with the given shape
        matrix = torch.randint(0, 255, shape, dtype=torch.uint8, device="cuda")

        # Dynamically allocate scratchpad based on matrix size
        scratchpad_size = get_required_scratchpad_bytes(matrix.numel())
        scratchpad = torch.empty(scratchpad_size, dtype=torch.uint8, device="cuda")

        # Create random key (32 bytes for Blake3)
        cuda_result = torch.empty(blake3.digest_size, dtype=torch.uint8, device="cuda")
        key_tensor = torch.randint(0, 255, (blake3.digest_size,), dtype=torch.uint8, device="cuda")
        key_bytes = key_tensor.cpu().numpy().tobytes()

        # Compute hash using CUDA implementation
        tensor_hash(matrix, key_tensor, cuda_result, scratchpad)
        torch.cuda.synchronize()

        # Compute hash using Python reference
        python_result = hash_matrix(matrix.cpu(), key_bytes)

        # Compare results
        assert torch.equal(cuda_result, python_result), (
            f"Hash mismatch for shape {shape}: CUDA result doesn't match Python reference"
        )

    def test_commitment_hash_from_merkle_roots(self):
        """Test that commitment hash from merkle roots matches Python reference."""

        A_merkle_root = torch.randint(
            0, 255, (blake3.digest_size,), dtype=torch.uint8, device="cuda"
        )
        B_merkle_root = torch.randint(
            0, 255, (blake3.digest_size,), dtype=torch.uint8, device="cuda"
        )
        key = torch.randint(0, 255, (blake3.digest_size,), dtype=torch.uint8, device="cuda")
        cuda_result_A = torch.empty(blake3.digest_size, dtype=torch.uint8, device="cuda")
        cuda_result_B = torch.empty(blake3.digest_size, dtype=torch.uint8, device="cuda")
        commitment_hash_from_merkle_roots(
            A_merkle_root, B_merkle_root, key, cuda_result_A, cuda_result_B
        )
        torch.cuda.synchronize()

        python_result_B = blake3(
            key.cpu().numpy().tobytes() + B_merkle_root.cpu().numpy().tobytes()
        ).digest()
        python_result_A = blake3(python_result_B + A_merkle_root.cpu().numpy().tobytes()).digest()

        assert cuda_result_A.cpu().numpy().tobytes() == python_result_A, (
            "Commitment hash from merkle roots mismatch: CUDA result doesn't match Python reference"
        )
        assert cuda_result_B.cpu().numpy().tobytes() == python_result_B, (
            "Commitment hash from merkle roots mismatch: CUDA result doesn't match Python reference"
        )

    def test_commitment_hash_from_merkle_roots_moe(self):
        """MoE folding path matches the CommitmentHasher reference."""
        # Cumulative exclusive ends per expert; last == m * top_k.
        routing_offsets = [10, 21, 28, 40]

        def random_digest_tensor() -> torch.Tensor:
            return torch.randint(0, 255, (blake3.digest_size,), dtype=torch.uint8, device="cuda")

        A_merkle_root = random_digest_tensor()
        B_merkle_root = random_digest_tensor()
        key = random_digest_tensor()
        routing_root = random_digest_tensor()
        offsets_root = CommitmentHasher.get_offsets_hash(
            routing_offsets, key.cpu().numpy().tobytes()
        )
        offsets_hash = torch.frombuffer(bytearray(offsets_root), dtype=torch.uint8).cuda()

        cuda_result_A = torch.empty(blake3.digest_size, dtype=torch.uint8, device="cuda")
        cuda_result_B = torch.empty(blake3.digest_size, dtype=torch.uint8, device="cuda")
        commitment_hash_from_merkle_roots(
            A_merkle_root,
            B_merkle_root,
            key,
            cuda_result_A,
            cuda_result_B,
            routing_root=routing_root,
            offsets_hash=offsets_hash,
        )
        torch.cuda.synchronize()

        reference = CommitmentHasher.commitment_hash_from_merkle_roots(
            A_merkle_root.cpu().numpy().tobytes(),
            B_merkle_root.cpu().numpy().tobytes(),
            key.cpu().numpy().tobytes(),
            routing_root=routing_root.cpu().numpy().tobytes(),
            offsets_root=offsets_root,
        )

        assert cuda_result_A.cpu().numpy().tobytes() == reference.noise_seed_A, (
            "MoE commitment hash mismatch: CUDA result doesn't match Python reference"
        )
        assert cuda_result_B.cpu().numpy().tobytes() == reference.noise_seed_B, (
            "MoE commitment hash mismatch: CUDA result doesn't match Python reference"
        )

    @pytest.mark.parametrize(
        "shape",
        [
            (64, 64),  # 4 chunks, single block
            (512, 512),  # 256 chunks, single block @512 threads
            (724, 724),  # 524176 B: 512 chunks, largest single block @512 threads
            (725, 725),  # 513 chunks: first size needing two blocks @512 threads
            (8192, 8192),  # Large square matrix (8k x 8k)
            (1337, 8192),  # Irregular dimensions
            (4096, 2048),  # Rectangular matrix
            (2048, 4096),  # Rectangular matrix (transposed)
            (777, 1024),  # Another irregular shape
            (2048, 3072),  # Non-power-of-2 dimensions
            (
                2000,
                2048,
            ),  # Initial test case we encountered the bug with (31 complete blocks and remainder block with R=32)
            (7952, 1024),  # 62 complete blocks and remainder blocks with R=64
            (7984, 1024),  # 62 complete blocks and remainder blocks with R=48=32+16
            (8192, 28672),
            (28672, 8192),
            (57344, 8192),
            (8192, 57344),
            (16384, 57344),
            (57344, 16384),
            (1245, 5136),
            (12451, 23141),
            (12345, 22141),
        ],
    )
    def test_tensor_hash_512_threads(self, shape):
        """Test tensor hash with 512 threads per block."""

        # Create random tensor with the given shape
        matrix = torch.randint(0, 255, shape, dtype=torch.uint8, device="cuda")

        # Dynamically allocate scratchpad based on matrix size
        scratchpad_size = get_required_scratchpad_bytes(matrix.numel())
        scratchpad = torch.empty(scratchpad_size, dtype=torch.uint8, device="cuda")

        # Create random key (32 bytes for Blake3)
        cuda_result = torch.empty(blake3.digest_size, dtype=torch.uint8, device="cuda")
        key_tensor = torch.randint(0, 255, (blake3.digest_size,), dtype=torch.uint8, device="cuda")
        key_bytes = key_tensor.cpu().numpy().tobytes()

        # Compute hash using CUDA implementation with 512 threads per block
        tensor_hash(matrix, key_tensor, cuda_result, scratchpad, threads_per_block=512)
        torch.cuda.synchronize()

        # Compute hash using Python reference
        python_result = hash_matrix(matrix, key_bytes)

        # Compare results
        assert torch.equal(cuda_result, python_result), (
            f"Hash mismatch for shape {shape} with 512 threads: CUDA result doesn't match Python reference"
        )

    @pytest.mark.parametrize("threads_per_block", [128, 512])
    @pytest.mark.parametrize(
        "nbytes",
        [
            512,  # single sub-chunk input
            1056,  # one full chunk + 32-byte remainder
            128 * 1024 + 512,  # partial chunk alone in the second CTA @128 threads
        ],
    )
    def test_tensor_hash_unaligned_tail_of_allocation(self, nbytes, threads_per_block):
        backing_bytes = 2 * 1024 * 1024
        backing = torch.randint(0, 255, (backing_bytes,), dtype=torch.uint8, device="cuda")
        matrix = backing[-nbytes:].reshape(1, nbytes)
        assert matrix.is_contiguous()

        scratchpad_size = get_required_scratchpad_bytes(matrix.numel())
        scratchpad = torch.empty(scratchpad_size, dtype=torch.uint8, device="cuda")

        cuda_result = torch.empty(blake3.digest_size, dtype=torch.uint8, device="cuda")
        key_tensor = torch.randint(0, 255, (blake3.digest_size,), dtype=torch.uint8, device="cuda")
        key_bytes = key_tensor.cpu().numpy().tobytes()

        tensor_hash(
            matrix, key_tensor, cuda_result, scratchpad, threads_per_block=threads_per_block
        )
        torch.cuda.synchronize()

        python_result = hash_matrix(matrix.cpu(), key_bytes)

        assert torch.equal(cuda_result, python_result), (
            f"Hash mismatch for {nbytes}-byte tail input with {threads_per_block} threads: "
            "CUDA result doesn't match Python reference"
        )

    def test_tensor_hash_deterministic(self):
        """Test that tensor hash produces deterministic results."""
        shape = (8192, 8192)
        data = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
        key = torch.zeros(blake3.digest_size, dtype=torch.uint8, device="cuda")

        # Dynamically allocate scratchpad based on matrix size
        scratchpad_size = get_required_scratchpad_bytes(data.numel())
        scratchpad = torch.empty(scratchpad_size, dtype=torch.uint8, device="cuda")

        base_out = torch.zeros(blake3.digest_size, dtype=torch.uint8, device="cuda")
        tensor_hash(data, key, base_out, scratchpad)

        for _ in range(100000):
            new_out = torch.zeros(blake3.digest_size, dtype=torch.uint8, device="cuda")
            tensor_hash(data, key, new_out, scratchpad)

            assert torch.equal(base_out, new_out), "Hash mismatch"
