#pragma once

#include "blake3/blake3.cuh"

namespace pearl {

class CommitmentHashFromBCommitmentKernel {
 public:
  using Element = uint8_t;
  static constexpr uint32_t MaxThreadsPerBlock = 1;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = 0;

  using RmemChainingValueLayout =
      Layout<Shape<Int<blake3::CHAINING_VALUE_SIZE_U32>>>;
  using RmemBlockLayout = Layout<Shape<Int<blake3::MSG_BLOCK_SIZE_U32>>>;

  struct Arguments {
    Element const* const ptr_A_merkle_root;
    Element const* const ptr_B_commitment_hash;
    Element* const ptr_A_commitment_hash;
    Element const* const ptr_routing_root;
    Element const* const ptr_offsets_hash;
  };

  struct Params {
    Element const* const ptr_A_merkle_root;
    Element const* const ptr_B_commitment_hash;
    Element* const ptr_A_commitment_hash;
    Element const* const ptr_routing_root;
    Element const* const ptr_offsets_hash;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return {args.ptr_A_merkle_root,
            args.ptr_B_commitment_hash,
            args.ptr_A_commitment_hash,
            args.ptr_routing_root,
            args.ptr_offsets_hash};
  }

  static dim3 get_grid_shape(Params const& params) { return dim3(1); }

  static dim3 get_block_shape() { return dim3(1); }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    static constexpr blake3::CompressParams single_block_params = {
        .counter = 0,
        .block_len = blake3::MSG_BLOCK_SIZE,
        .flags = blake3::CHUNK_START | blake3::CHUNK_END | blake3::ROOT,
    };

    Tensor rARoot = make_tensor<uint32_t>(RmemChainingValueLayout{});
    uint32_t const* A_merkle_root_u32 =
        reinterpret_cast<uint32_t const*>(params.ptr_A_merkle_root);

    if (params.ptr_routing_root != nullptr) {
      uint32_t const* routing_root_u32 =
          reinterpret_cast<uint32_t const*>(params.ptr_routing_root);
      uint32_t const* offsets_hash_u32 =
          reinterpret_cast<uint32_t const*>(params.ptr_offsets_hash);

      Tensor rBlockRouting = make_tensor<uint32_t>(RmemBlockLayout{});
      Tensor rHashRouting = make_tensor<uint32_t>(RmemChainingValueLayout{});
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
        rBlockRouting(i) = routing_root_u32[i];
        rBlockRouting(i + blake3::CHAINING_VALUE_SIZE_U32) =
            offsets_hash_u32[i];
        rHashRouting(i) = blake3::IV[i];
      }
      blake3::compress_msg_block_u32(rBlockRouting, rHashRouting,
                                     single_block_params);

      Tensor rBlockActivations = make_tensor<uint32_t>(RmemBlockLayout{});
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
        rBlockActivations(i) = A_merkle_root_u32[i];
        rBlockActivations(i + blake3::CHAINING_VALUE_SIZE_U32) =
            rHashRouting(i);
        rARoot(i) = blake3::IV[i];
      }
      blake3::compress_msg_block_u32(rBlockActivations, rARoot,
                                     single_block_params);
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
        rARoot(i) = A_merkle_root_u32[i];
      }
    }

    Tensor rBlockA = make_tensor<uint32_t>(RmemBlockLayout{});
    Tensor rChainingValueA = make_tensor<uint32_t>(RmemChainingValueLayout{});
    uint32_t const* B_commitment_hash_u32 =
        reinterpret_cast<uint32_t const*>(params.ptr_B_commitment_hash);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      rBlockA(i) = B_commitment_hash_u32[i];
      rBlockA(i + blake3::CHAINING_VALUE_SIZE_U32) = rARoot(i);
      rChainingValueA(i) = blake3::IV[i];
    }
    blake3::compress_msg_block_u32(rBlockA, rChainingValueA,
                                   single_block_params);

    uint32_t* A_commitment_hash_u32 =
        reinterpret_cast<uint32_t*>(params.ptr_A_commitment_hash);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      A_commitment_hash_u32[i] = rChainingValueA(i);
    }
  }
};

}  // namespace pearl
