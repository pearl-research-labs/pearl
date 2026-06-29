import pearl_gemm_cuda
import torch
from pearl_gemm_cuda import get_build_routing_data_scratchpad_bytes  # noqa: F401


def build_routing_data(
    topk_ids: torch.Tensor,
    routing_data: torch.Tensor,
    slot_indices: torch.Tensor,
    routing_offsets: torch.Tensor,
    scratchpad: torch.Tensor,
    num_experts: int,
) -> None:
    """Build canonical routing data for commitment hashing.

    Produces a deterministic per-expert stable ordering. Emits the sorted flat
    slot indices (``slot_indices``), their token indices (``routing_data ==
    slot_indices // top_k``), and per-expert exclusive-end offsets.

    Args:
        topk_ids: (m, K) int32 tensor of expert assignments per token
        routing_data: pre-allocated (m*K,) int32 output tensor (token indices)
        slot_indices: pre-allocated (m*K,) int32 output tensor (flat slot indices)
        routing_offsets: pre-allocated (num_experts,) int32 output tensor
        scratchpad: pre-allocated uint8 temp buffer (see
            get_build_routing_data_scratchpad_bytes for required size)
        num_experts: total number of experts E
    """
    pearl_gemm_cuda.build_routing_data(
        topk_ids, routing_data, slot_indices, routing_offsets, scratchpad, num_experts
    )


@torch.library.register_fake("pearl_gemm::build_routing_data")
def _abstract_build_routing_data(
    topk_ids: torch.Tensor,
    routing_data: torch.Tensor,
    slot_indices: torch.Tensor,
    routing_offsets: torch.Tensor,
    scratchpad: torch.Tensor,
    num_experts: int,
) -> None:
    return None
