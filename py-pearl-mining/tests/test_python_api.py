"""
Tests for the pearl_mining Python API (PyO3 bindings).
"""

import base64

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
    k: int, rank: int = DEFAULT_RANK, e: int | None = None, top_k: int | None = None
) -> pearl_mining.MiningConfiguration:
    rows_pattern = pearl_mining.PeriodicPattern.from_list(ROWS_PATTERN_LIST)
    cols_pattern = pearl_mining.PeriodicPattern.from_list(COLS_PATTERN_LIST)
    moe = None if e is None else pearl_mining.MoEConfig(e=e, top_k=top_k)
    return pearl_mining.MiningConfiguration(
        common_dim=k,
        rank=rank,
        mma_type=pearl_mining.MMAType.Int7xInt7ToInt32,
        rows_pattern=rows_pattern,
        cols_pattern=cols_pattern,
        moe=moe,
    )


def generate_plain_proof(
    m: int,
    n: int,
    k: int,
    block_header: pearl_mining.IncompleteBlockHeader,
    rank: int = DEFAULT_RANK,
    signal_range: tuple[int, int] | None = None,
    wrong_jackpot_hash: bool = False,
) -> pearl_mining.PlainProof:
    """Generate a PlainProof using mine()."""
    mining_config = create_default_mining_config(k, rank=rank)
    return pearl_mining.mine(
        m,
        n,
        k,
        block_header,
        mining_config,
        signal_range=signal_range,
        wrong_jackpot_hash=wrong_jackpot_hash,
    )


def generate_moe_plain_proof(
    m: int,
    n: int,
    k: int,
    e: int,
    top_k: int,
    block_header: pearl_mining.IncompleteBlockHeader,
    rank: int = DEFAULT_RANK,
    signal_range: tuple[int, int] | None = None,
    wrong_jackpot_hash: bool = False,
) -> pearl_mining.PlainProof:
    """Generate a PlainProof using mine_moe()."""
    mining_config = create_default_mining_config(k, rank=rank, e=e, top_k=top_k)
    return pearl_mining.mine_moe(
        m,
        n,
        k,
        block_header,
        mining_config,
        signal_range=signal_range,
        wrong_jackpot_hash=wrong_jackpot_hash,
    )


def prove_and_verify(
    block_header: pearl_mining.IncompleteBlockHeader,
    plain_proof: pearl_mining.PlainProof,
    *,
    expect_valid: bool = True,
) -> tuple[pearl_mining.ZKProof, bool, str]:
    """Generate a ZK proof and verify it. Asserts the expected outcome."""
    proof = pearl_mining.generate_proof_v2(block_header, plain_proof)
    is_valid, message = pearl_mining.verify_proof_v2(block_header, proof)
    if expect_valid:
        assert is_valid, f"Verification unexpectedly failed: {message}"
    else:
        assert not is_valid, "Verification succeeded when it should have failed -- soundness issue!"
    return proof, is_valid, message


def moe_prove_and_verify(
    block_header: pearl_mining.IncompleteBlockHeader,
    moe_proof: pearl_mining.PlainProof,
    *,
    expect_valid: bool = True,
) -> tuple[pearl_mining.ZKProof, bool, str]:
    """Generate a ZK proof from an MoE mining solution and verify it."""
    proof = pearl_mining.generate_proof_v2(block_header, moe_proof)
    is_valid, message = pearl_mining.verify_proof_v2(block_header, proof)
    if expect_valid:
        assert is_valid, f"MoE verification unexpectedly failed: {message}"
    else:
        assert not is_valid, (
            "MoE verification succeeded when it should have failed -- soundness issue!"
        )
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
    """Tests for verify_plain_proof_v2(), which checks the mining solution without ZK proving."""

    def test_valid_plain_proof(self):
        m, n, k = 256, 128, DEFAULT_K
        block_header = create_test_block_header()
        plain_proof = generate_plain_proof(m, n, k, block_header)

        is_valid, message = pearl_mining.verify_plain_proof_v2(block_header, plain_proof)
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

        is_valid, message = pearl_mining.verify_plain_proof_v2(block_header, plain_proof)
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


# --- MoE constants ---
# MoE mining routes each token to `top_k` out of `e` experts, so each expert
# sees ~m*top_k/e tokens. m must be large enough that every expert gets at
# least one full row-pattern period worth of tokens, otherwise the miner will
# loop forever.
DEFAULT_E = 4
DEFAULT_TOP_K = 2
DEFAULT_MOE_M = 1024
DEFAULT_MOE_N = 256

MOE_DIMENSION_CASES = [
    pytest.param(1024, 256, 1024, 4, 2, 32, id="1024x1024_1024x256_e4_top2_r32"),
    pytest.param(1024, 256, 1088, 4, 2, 32, id="1024x1088_1088x256_e4_top2_r32"),
    pytest.param(1024, 384, 1024, 4, 2, 32, id="1024x1024_1024x384_e4_top2_r32"),
]


class TestMoEMineProveVerify:
    """End-to-end MoE: mine -> verify plain -> ZK prove -> ZK verify."""

    def test_basic_moe(self):
        block_header = create_test_block_header()
        moe_proof = generate_moe_plain_proof(
            DEFAULT_MOE_M,
            DEFAULT_MOE_N,
            DEFAULT_K,
            DEFAULT_E,
            DEFAULT_TOP_K,
            block_header,
        )

        is_valid, msg = pearl_mining.verify_plain_proof_v2(block_header, moe_proof)
        assert is_valid, f"MoE plain proof verification failed: {msg}"

        moe_prove_and_verify(block_header, moe_proof)

    @pytest.mark.parametrize("m, n, k, e, top_k, rank", MOE_DIMENSION_CASES)
    def test_moe_dimensions(self, m, n, k, e, top_k, rank):
        block_header = create_test_block_header()
        moe_proof = generate_moe_plain_proof(m, n, k, e, top_k, block_header, rank=rank)

        is_valid, msg = pearl_mining.verify_plain_proof_v2(block_header, moe_proof)
        assert is_valid, f"MoE plain proof verification failed: {msg}"

        moe_prove_and_verify(block_header, moe_proof)


class TestMoEVerifyPlainProof:
    """Tests for verify_plain_proof_v2() in the MoE context, which checks the MoE mining solution without ZK proving."""

    def test_valid_moe_plain_proof(self):
        block_header = create_test_block_header()
        moe_proof = generate_moe_plain_proof(
            DEFAULT_MOE_M,
            DEFAULT_MOE_N,
            DEFAULT_K,
            DEFAULT_E,
            DEFAULT_TOP_K,
            block_header,
        )

        is_valid, msg = pearl_mining.verify_plain_proof_v2(block_header, moe_proof)
        assert is_valid, f"verify_plain_proof failed on valid proof: {msg}"

    def test_wrong_range_moe_plain_proof(self):
        """Out-of-range signal values should fail MoE plain proof verification."""
        block_header = create_test_block_header()
        moe_proof = generate_moe_plain_proof(
            DEFAULT_MOE_M,
            DEFAULT_MOE_N,
            DEFAULT_K,
            DEFAULT_E,
            DEFAULT_TOP_K,
            block_header,
            signal_range=OUT_OF_RANGE_SIGNAL_RANGE,
        )

        is_valid, msg = pearl_mining.verify_plain_proof_v2(block_header, moe_proof)
        assert not is_valid, "verify_plain_proof accepted out-of-range matrices -- soundness issue!"


class TestMoESoundness:
    """Negative tests: MoE proofs from invalid inputs must fail verification."""

    def test_moe_wrong_range_matrices(self):
        """Out-of-range signal values should fail MoE ZK verification."""
        block_header = create_test_block_header()
        moe_proof = generate_moe_plain_proof(
            DEFAULT_MOE_M,
            DEFAULT_MOE_N,
            DEFAULT_K,
            DEFAULT_E,
            DEFAULT_TOP_K,
            block_header,
            signal_range=OUT_OF_RANGE_SIGNAL_RANGE,
        )

        moe_prove_and_verify(block_header, moe_proof, expect_valid=False)

    def test_moe_wrong_jackpot_hash(self):
        """Incorrect jackpot hash should fail MoE verification."""
        block_header = create_test_block_header()
        moe_proof = generate_moe_plain_proof(
            DEFAULT_MOE_M,
            DEFAULT_MOE_N,
            DEFAULT_K,
            DEFAULT_E,
            DEFAULT_TOP_K,
            block_header,
            wrong_jackpot_hash=True,
        )

        moe_prove_and_verify(block_header, moe_proof, expect_valid=False)


class TestMoESerialization:
    """Test MoE PlainProof base64 round-trip serialization."""

    def test_moe_proof_round_trip(self):
        block_header = create_test_block_header()
        moe_proof = generate_moe_plain_proof(
            DEFAULT_MOE_M,
            DEFAULT_MOE_N,
            DEFAULT_K,
            DEFAULT_E,
            DEFAULT_TOP_K,
            block_header,
        )

        encoded = moe_proof.to_base64()
        restored = pearl_mining.PlainProof.from_base64(encoded)

        is_valid_original, _ = pearl_mining.verify_plain_proof_v2(block_header, moe_proof)
        is_valid_restored, _ = pearl_mining.verify_plain_proof_v2(block_header, restored)
        assert is_valid_original and is_valid_restored, "Round-trip serialization broke the proof"


# --- Certificate-version dispatch (MoE fork crossover) ---

# bincode tag byte for Option::None; a legacy V1 PlainProof blob is the current
# dense serialization minus this trailing byte.
BINCODE_OPTION_NONE_TAG = 0x00

# Difficulty target far below what test proofs satisfy (DEFAULT_NBITS).
HARD_NBITS = 0x10FFFFFF

VALID_CERT_VERSIONS = [
    pearl_mining.CERT_VERSION_ZK_DENSE,
    pearl_mining.CERT_VERSION_ZK_MOE,
]

_DUMMY_LEAF_SIZE_BYTES = 1024
_DUMMY_HASH_SIZE_BYTES = 32
_DUMMY_DIMENSION = 4
_DUMMY_NOISE_RANK = 1
_DUMMY_EXPERT_COUNT = 2
_DUMMY_TOP_K = 1


def make_dummy_plain_proof(*, moe: bool) -> pearl_mining.PlainProof:
    """Lightweight (non-verifiable) PlainProof for eligibility checks only."""
    merkle_proof = pearl_mining.MerkleProof(
        total_leaves=1,
        leaf_data=[b"\x00" * _DUMMY_LEAF_SIZE_BYTES],
        leaf_indices=[0],
        root=b"\x00" * _DUMMY_HASH_SIZE_BYTES,
        siblings=[],
    )
    matrix_proof = pearl_mining.MatrixMerkleProof(proof=merkle_proof, row_indices=[0])
    moe_params = (
        pearl_mining.MoEProofParams(
            e=_DUMMY_EXPERT_COUNT,
            top_k=_DUMMY_TOP_K,
            expert_idx=0,
            routing_end_offsets=[_DUMMY_TOP_K, _DUMMY_EXPERT_COUNT],
            inner_a_rows=[0],
            routing_proof=merkle_proof,
        )
        if moe
        else None
    )
    return pearl_mining.PlainProof(
        m=_DUMMY_DIMENSION,
        n=_DUMMY_DIMENSION,
        k=_DUMMY_DIMENSION,
        noise_rank=_DUMMY_NOISE_RANK,
        a_merkle_proof=matrix_proof,
        bt_merkle_proof=matrix_proof,
        moe=moe_params,
    )


class TestLegacyV1Deserialization:
    """from_base64() must accept blobs from old miners (no trailing `moe` Option tag)."""

    def test_accepts_legacy_v1_blob(self):
        m, n, k = 256, 128, DEFAULT_K
        block_header = create_test_block_header()
        plain_proof = generate_plain_proof(m, n, k, block_header)

        raw = base64.b64decode(plain_proof.to_base64())
        assert raw[-1] == BINCODE_OPTION_NONE_TAG
        legacy_blob = base64.b64encode(raw[:-1]).decode()

        restored = pearl_mining.PlainProof.from_base64(legacy_blob)
        assert restored.min_cert_version == pearl_mining.CERT_VERSION_ZK_DENSE
        is_valid, msg = pearl_mining.verify_plain_proof_v2(block_header, restored)
        assert is_valid, f"restored legacy proof failed verification: {msg}"

    def test_rejects_garbage(self):
        garbage = base64.b64encode(b"\xab" * 7).decode()
        with pytest.raises(ValueError):
            pearl_mining.PlainProof.from_base64(garbage)


class TestCertVersionEligibility:
    """check_cert_version_eligible() enforces the crossover rule."""

    def test_min_cert_version_exposed(self):
        dense = make_dummy_plain_proof(moe=False)
        moe = make_dummy_plain_proof(moe=True)
        assert dense.min_cert_version == pearl_mining.CERT_VERSION_ZK_DENSE
        assert moe.min_cert_version == pearl_mining.CERT_VERSION_ZK_MOE

    def test_dense_proof_eligible_under_both_versions(self):
        dense = make_dummy_plain_proof(moe=False)
        for cert_version in VALID_CERT_VERSIONS:
            pearl_mining.check_cert_version_eligible(cert_version, dense)

    def test_moe_proof_rejected_before_crossover(self):
        moe = make_dummy_plain_proof(moe=True)
        pearl_mining.check_cert_version_eligible(pearl_mining.CERT_VERSION_ZK_MOE, moe)
        with pytest.raises(ValueError, match="crossover"):
            pearl_mining.check_cert_version_eligible(pearl_mining.CERT_VERSION_ZK_DENSE, moe)

    def test_unknown_versions_rejected(self):
        dense = make_dummy_plain_proof(moe=False)
        for version in (0, 3):
            with pytest.raises(ValueError, match="unknown certificate version"):
                pearl_mining.check_cert_version_eligible(version, dense)


class TestCertVersionDispatchers:
    """The *_for_cert_version entry points pick the circuit from the block's version."""

    @pytest.mark.parametrize("cert_version", VALID_CERT_VERSIONS)
    def test_dense_proof_round_trip_under_both_versions(self, cert_version):
        m, n, k = 256, 128, DEFAULT_K
        block_header = create_test_block_header()
        plain_proof = generate_plain_proof(m, n, k, block_header)

        is_valid, msg = pearl_mining.verify_plain_proof_for_cert_version(
            cert_version, block_header, plain_proof
        )
        assert is_valid, f"plain proof verification failed (cert v{cert_version}): {msg}"

        proof = pearl_mining.generate_proof_for_cert_version(
            cert_version, block_header, plain_proof
        )
        is_valid, msg = pearl_mining.verify_proof_for_cert_version(
            cert_version, block_header, proof
        )
        assert is_valid, f"ZK verification failed (cert v{cert_version}): {msg}"

    def test_moe_proof_rejected_for_v1_block(self):
        moe = make_dummy_plain_proof(moe=True)
        block_header = create_test_block_header()
        with pytest.raises(ValueError, match="crossover"):
            pearl_mining.generate_proof_for_cert_version(
                pearl_mining.CERT_VERSION_ZK_DENSE, block_header, moe
            )


class TestV1NbitsOverride:
    """verify_plain_proof_v1() supports pool share-difficulty overrides like v2."""

    def test_share_difficulty_override(self):
        m, n, k = 256, 128, DEFAULT_K
        block_header = create_test_block_header()
        plain_proof = generate_plain_proof(m, n, k, block_header)

        is_valid, msg = pearl_mining.verify_plain_proof_v1(
            block_header, plain_proof, nbits_override=DEFAULT_NBITS
        )
        assert is_valid, f"v1 verification failed at the mined difficulty: {msg}"

        is_valid, _ = pearl_mining.verify_plain_proof_v1(
            block_header, plain_proof, nbits_override=HARD_NBITS
        )
        assert not is_valid, "v1 verification accepted a proof above the override target"
