import torch
from miner_utils import get_logger
from pearl_gateway.comm.dataclasses import MoEBlockInfo, OpenedBlockInfo
from pearl_mining import MatrixMerkleProof, MoEProofParams, PlainProof

from .commitment_hash import CommitmentHasher
from .matrix_merkle_tree import MatrixMerkleTree

_LOGGER = get_logger(__name__)

_INT32_SIZE_BYTES = 4


def create_proof(
    opened_block_info: OpenedBlockInfo,
    incomplete_header_bytes: bytes,
) -> PlainProof:
    """Create a PlainProof from OpenedBlockInfo using non-noised A and B matrices."""
    _LOGGER.debug("Creating proof")
    A = opened_block_info.A
    B_t = opened_block_info.B_t
    mining_config = opened_block_info.get_mining_config()

    hash_key = CommitmentHasher.get_key(incomplete_header_bytes, mining_config)
    A_merkle_tree = MatrixMerkleTree(A, hash_key)
    B_merkle_tree = MatrixMerkleTree(B_t, hash_key)
    _LOGGER.debug("Generated merkle trees")

    a_merkle_proof = MatrixMerkleProof(
        proof=A_merkle_tree.get_multileaf_proof(
            A_merkle_tree.leaf_indices_from_rows(opened_block_info.A_row_indices)
        ),
        row_indices=opened_block_info.A_row_indices,
    )
    b_merkle_proof = MatrixMerkleProof(
        proof=B_merkle_tree.get_multileaf_proof(
            B_merkle_tree.leaf_indices_from_rows(opened_block_info.B_column_indices)
        ),
        row_indices=opened_block_info.B_column_indices,
    )

    m, k = A.shape
    n, k2 = B_t.shape
    assert k == k2, f"Common dimension mismatch: {k} != {k2}"

    moe_params = None
    if moe := opened_block_info.moe:
        moe_params = _build_moe_proof_params(moe, hash_key)
        # In the MoE case ``n`` is the per-expert intermediate dim; B_t holds all
        # experts stacked (E * n_per_expert rows), so override ``n`` accordingly.
        n = moe.n_per_expert
        _LOGGER.debug(f"Built MoE routing proof for expert {opened_block_info.moe.expert_index}")

    return PlainProof(
        m=m,
        n=n,
        k=k,
        noise_rank=opened_block_info.noise_rank,
        a_merkle_proof=a_merkle_proof,
        bt_merkle_proof=b_merkle_proof,
        moe=moe_params,
    )


def _build_moe_proof_params(moe: MoEBlockInfo, hash_key: bytes) -> MoEProofParams:
    routing = moe.routing_data

    # The routing data is the flat array of expert-sorted token indices (u32),
    # committed in a Merkle tree padded to the chunk boundary; each u32 entry is
    # one leaf row, so reshape to (-1, 4) int8 bytes.
    routing_rows = routing.contiguous().view(torch.int8).reshape(-1, _INT32_SIZE_BYTES)
    routing_tree = MatrixMerkleTree(routing_rows, hash_key)

    # expert_routing_offsets are per-expert exclusive ends, so this expert's start is
    # the previous expert's end (0 for expert 0), mirroring the Rust verifier.
    expert_start = 0 if moe.expert_index == 0 else moe.expert_routing_offsets[moe.expert_index - 1]
    global_routing_rows = [expert_start + inner_row for inner_row in moe.inner_a_rows]
    routing_proof = routing_tree.get_multileaf_proof(
        routing_tree.leaf_indices_from_rows(global_routing_rows)
    )
    return MoEProofParams(
        e=moe.num_experts,
        top_k=moe.top_k,
        expert_idx=moe.expert_index,
        routing_end_offsets=moe.expert_routing_offsets,
        inner_a_rows=list(moe.inner_a_rows),
        routing_proof=routing_proof,
    )
