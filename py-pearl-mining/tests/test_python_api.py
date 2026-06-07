"""
Tests for the pearl_mining Python API (PyO3 bindings).
"""

import pearl_mining
import pytest

# --- Constants ---
DEFAULT_NBITS = 0x1D2FFFFF
DEFAULT_K = 1024
DEFAULT_RANK = 32

ROWS_PATTERN_LIST = [0, 8, 64, 72]
COLS_PATTERN_LIST = [0, 1, 8, 9, 32, 33, 40, 41]

# Correct signal range is [-64, 63]; 65 is out of range
OUT_OF_RANGE_SIGNAL_RANGE = (-64, 65)


# --- Helpers ---
def create_test_block_header(nbits: int = DEFAULT_NBITS) -> pearl_mining.IncompleteBlockHeader:
    return pearl_mining.IncompleteBlockHeader(
        version=0,
        prev_block=b"\x00" * 32,
        merkle_root=b"0123456789abcdef" * 2,
        timestamp=0x66666666,
        nbits=nbits,
    )


def create_default_mining_config(
    k: int, rank: int = DEFAULT_RANK
) -> pearl_mining.MiningConfiguration:
    rows_pattern = pearl_mining.PeriodicPattern.from_list(ROWS_PATTERN_LIST)
    cols_pattern = pearl_mining.PeriodicPattern.from_list(COLS_PATTERN_LIST)
    return pearl_mining.MiningConfiguration(
        common_dim=k,
        rank=rank,
        mma_type=pearl_mining.MMAType.Int7xInt7ToInt32,
        rows_pattern=rows_pattern,
        cols_pattern=cols_pattern,
        reserved=pearl_mining.MiningConfiguration.RESERVED,
    )


def generate_plain_proof(
    m: int,
    n: int,
    k: int,
    block_header: pearl_mining.IncompleteBlockHeader,
    rank: int = DEFAULT_RANK,
    signal_range: tuple[int, int] | None = None,
    wrong_jackpot_hash: bool = False,
) -> tuple[pearl_mining.PlainProof, pearl_mining.MiningConfiguration]:
    """Generate a PlainProof using mine(). Returns (plain_proof, mining_config)."""
    mining_config = create_default_mining_config(k, rank=rank)
    plain_proof = pearl_mining.mine(
        m,
        n,
        k,
        block_header,
        mining_config,
        signal_range=signal_range,
        wrong_jackpot_hash=wrong_jackpot_hash,
    )
    return plain_proof


def prove_and_verify(
    block_header: pearl_mining.IncompleteBlockHeader,
    plain_proof: pearl_mining.PlainProof,
    *,
    expect_valid: bool = True,
) -> tuple[pearl_mining.ZKProof, bool, str]:
    """Generate a ZK proof and verify it. Asserts the expected outcome."""
    proof = pearl_mining.generate_proof(block_header, plain_proof)
    is_valid, message = pearl_mining.verify_proof(block_header, proof)
    if expect_valid:
        assert is_valid, f"Verification unexpectedly failed: {message}"
    else:
        assert not is_valid, "Verification succeeded when it should have failed -- soundness issue!"
    return proof, is_valid, message


class TestTileConfiguration:
    """Test the 4x16 tile configuration (4 row indices, 8+8 col pattern)."""

    def test_4x16_tile(self):
        m, n, k = 256, 128, 1024
        block_header = create_test_block_header()
        plain_proof = generate_plain_proof(m, n, k, block_header)

        assert len(plain_proof.a.row_indices) > 0
        assert len(plain_proof.bt.row_indices) > 0

        prove_and_verify(block_header, plain_proof)


DIMENSION_CASES = [
    pytest.param(256, 128, 1088, 32, id="256x1088_1088x128_r32"),
    pytest.param(128, 256, 1152, 32, id="128x1152_1152x256_r32"),
    pytest.param(128, 256, 1024, 64, id="128x1024_1024x256_r64"),
    pytest.param(512, 384, 1920, 32, id="512x1920_1920x384_r32"),
]


class TestDifferentDimensions:
    """Parametrized tests over various matrix dimension / rank combos."""

    @pytest.mark.parametrize("m, n, k, rank", DIMENSION_CASES)
    def test_dimensions(self, m, n, k, rank):
        block_header = create_test_block_header()
        plain_proof = generate_plain_proof(m, n, k, block_header, rank=rank)
        prove_and_verify(block_header, plain_proof)


class TestVerifyPlainProof:
    """Tests for verify_plain_proof(), which checks the mining solution without ZK proving."""

    def test_valid_plain_proof(self):
        m, n, k = 256, 128, DEFAULT_K
        block_header = create_test_block_header()
        plain_proof = generate_plain_proof(m, n, k, block_header)

        is_valid, message = pearl_mining.verify_plain_proof(block_header, plain_proof)
        assert is_valid, f"verify_plain_proof failed on valid proof: {message}"

    def test_wrong_range_plain_proof(self):
        """Out-of-range signal values should fail plain proof verification too."""
        m, n, k = 256, 128, DEFAULT_K
        block_header = create_test_block_header()
        plain_proof = generate_plain_proof(
            m,
            n,
            k,
            block_header,
            signal_range=OUT_OF_RANGE_SIGNAL_RANGE,
        )

        is_valid, message = pearl_mining.verify_plain_proof(block_header, plain_proof)
        assert not is_valid, "verify_plain_proof accepted out-of-range matrices -- soundness issue!"


class TestSoundness:
    """Negative tests: proofs from invalid inputs must fail verification."""

    def test_wrong_range_matrices(self):
        """Out-of-range signal values (correct is [-64, 63]) should fail verification."""
        m, n, k = 256, 128, DEFAULT_K
        block_header = create_test_block_header()
        plain_proof = generate_plain_proof(
            m,
            n,
            k,
            block_header,
            signal_range=OUT_OF_RANGE_SIGNAL_RANGE,
        )

        prove_and_verify(block_header, plain_proof, expect_valid=False)

    def test_wrong_jackpot_hash(self):
        """Incorrect jackpot hash should fail verification."""
        m, n, k = 256, 128, 1024
        block_header = create_test_block_header(DEFAULT_NBITS)
        plain_proof = generate_plain_proof(
            m,
            n,
            k,
            block_header,
            wrong_jackpot_hash=True,
        )

        prove_and_verify(block_header, plain_proof, expect_valid=False)
