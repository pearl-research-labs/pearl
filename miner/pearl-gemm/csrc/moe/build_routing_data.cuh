#pragma once

#include <ATen/ATen.h>

#include <cstddef>
#include <cstdint>

// Returns the minimum scratchpad size in bytes for a given numel = m * K.
// The caller should allocate a uint8 tensor of this size and pass it to
// build_routing_data. Supports up to 256 experts (8-bit key sort).
int64_t get_build_routing_data_scratchpad_bytes(int64_t numel);

// Stable counting sort for MoE routing data. Produces a deterministic ordering
// grouped by expert, suitable for commitment hashing: the sorted flat slot
// indices (slot_indices), their token indices
// (routing_data = slot_indices / top_k), and the per-expert exclusive-end
// offsets.
void build_routing_data(const at::Tensor& topk_ids,
                        const at::Tensor& routing_data,
                        const at::Tensor& slot_indices,
                        const at::Tensor& routing_offsets,
                        const at::Tensor& scratchpad, int64_t num_experts);
