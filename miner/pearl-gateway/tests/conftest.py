import pytest
from pearl_mining import MatrixMerkleProof, MerkleProof, MoEProofParams, PlainProof

_LEAF_SIZE_BYTES = 1024
_HASH_SIZE_BYTES = 32
_DUMMY_DIMENSION = 4
_DUMMY_EXPERT_COUNT = 2
_DUMMY_EXPERT_INDEX = 0
_DUMMY_LEAF_INDEX = 0
_DUMMY_NOISE_RANK = 1
_DUMMY_TOP_K = 1
_DUMMY_TOTAL_LEAVES = 1
_DUMMY_ROUTING_END_OFFSETS = (_DUMMY_TOP_K, _DUMMY_EXPERT_COUNT)


def _dummy_merkle_proof() -> MerkleProof:
    return MerkleProof(
        total_leaves=_DUMMY_TOTAL_LEAVES,
        leaf_data=[b"\x00" * _LEAF_SIZE_BYTES],
        leaf_indices=[_DUMMY_LEAF_INDEX],
        root=b"\x00" * _HASH_SIZE_BYTES,
        siblings=[],
    )


def _build_plain_proof(*, moe: bool) -> PlainProof:
    merkle_proof = _dummy_merkle_proof()
    matrix_proof = MatrixMerkleProof(proof=merkle_proof, row_indices=[_DUMMY_LEAF_INDEX])
    moe_params = (
        MoEProofParams(
            e=_DUMMY_EXPERT_COUNT,
            top_k=_DUMMY_TOP_K,
            expert_idx=_DUMMY_EXPERT_INDEX,
            routing_end_offsets=list(_DUMMY_ROUTING_END_OFFSETS),
            inner_a_rows=[_DUMMY_LEAF_INDEX],
            routing_proof=merkle_proof,
        )
        if moe
        else None
    )
    return PlainProof(
        m=_DUMMY_DIMENSION,
        n=_DUMMY_DIMENSION,
        k=_DUMMY_DIMENSION,
        noise_rank=_DUMMY_NOISE_RANK,
        a_merkle_proof=matrix_proof,
        bt_merkle_proof=matrix_proof,
        moe=moe_params,
    )


@pytest.fixture
def make_dummy_plain_proof():
    """Factory for lightweight dense/MoE PlainProofs used in crossover tests.

    Unlike the realistic ``sample_plain_proof`` in the repo-wide conftest, these
    carry only enough structure to exercise certificate-version selection
    (``min_cert_version`` is derived from the presence of MoE params).
    """
    return _build_plain_proof


@pytest.fixture
def dummy_moe_proof(make_dummy_plain_proof):
    return make_dummy_plain_proof(moe=True)
