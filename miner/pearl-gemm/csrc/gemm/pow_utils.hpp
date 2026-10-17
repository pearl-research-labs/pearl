#pragma once

#include <cutlass/cutlass.h>
#include <cute/atom/mma_traits_sm90_gmma.hpp>

#include "blake3/blake3.cuh"
#include "host_signal_header.hpp"
#include "utils.h"

namespace pearl {

using namespace cute;

// Rotation amount for hash accumulation mixing
static constexpr int HASH_ACCUMULATE_ROTATION = 13;

// 3-input XOR using PTX lop3 instruction for maximum efficiency
// LUT 0x96 = 0b10010110 implements d = a ^ b ^ c
CUTE_DEVICE
uint32_t xor3_lop3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// Rotate-XOR: computes rotl(x, shift) ^ y = ((x << shift) | (x >> (32-shift))) ^ y
template <int shift>
CUTE_DEVICE uint32_t rotl_xor(uint32_t x, uint32_t y) {
  static_assert(shift > 0 && shift < 32, "Shift must be in range (0, 32)");
  uint32_t rotated;
  // shf.l.wrap.b32 d, x, x, n  =>  d = (x << n) | (x >> (32-n)) = rotl(x, n)
  asm("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(rotated) : "r"(x), "n"(shift));
  return rotated ^ y;
}

// Process one layer of XOR tree reduction using lop3
template <class OutputLayerSize, class InputLayer>
CUTE_DEVICE auto process_xor_layer(InputLayer const& input_layer) {
  constexpr size_t input_size = InputLayer{}.size();
  constexpr size_t output_layer_size = OutputLayerSize{}.value;
  constexpr size_t triplets = input_size / 3;
  constexpr size_t remainder = input_size % 3;

  static_assert(output_layer_size == triplets + remainder,
                "Output layer size must match expected reduction");

  cute::array<uint32_t, output_layer_size> result;

  CUTLASS_PRAGMA_UNROLL
  for (size_t i = 0; i < triplets; ++i) {
    result[i] = xor3_lop3(input_layer[3 * i], input_layer[3 * i + 1],
                          input_layer[3 * i + 2]);
  }

  // Pass through remainder elements unchanged
  CUTLASS_PRAGMA_UNROLL
  for (size_t i = 0; i < remainder; ++i) {
    result[triplets + i] = input_layer[triplets * 3 + i];
  }

  return result;
}

// Compute XOR tree layer sizes at compile time
// Returns tuple of layer sizes for tree reduction (largest to smallest)
template <size_t N>
constexpr auto xor_tree_layer_sizes() {
  if constexpr (N <= 3) {
    return cute::make_tuple(cute::Int<N>{});
  } else {
    constexpr size_t next = (N / 3) + (N % 3);
    return cute::tuple_cat(cute::make_tuple(cute::Int<N>{}),
                           xor_tree_layer_sizes<next>());
  }
}

// XOR reduction of all uint32 elements in the input tensor
// Uses tree reduction with lop3
template <typename TensorType>
CUTE_DEVICE uint32_t xor_reduction(const TensorType& input_tensor) {
  constexpr size_t buffer_size =
      decltype(std::declval<TensorType>().size())::value;

  static_assert(buffer_size > 0, "Buffer size must be positive");

  // "cast" input tensor to array, compiler optimizes this away as everything is in registers
  cute::array<uint32_t, buffer_size> first_layer;
  CUTLASS_PRAGMA_UNROLL
  for (size_t i = 0; i < buffer_size; ++i) {
    first_layer[i] = input_tensor[i];
  }

  // Get layer size configuration (excluding first layer which we already have)
  constexpr auto all_layer_sizes = xor_tree_layer_sizes<buffer_size>();
  constexpr auto remaining_layers = cute::take<1, -1>(all_layer_sizes);

  // Tree reduction using fold
  auto final_layer = cute::fold(
      remaining_layers, first_layer, [](auto const& layer, auto target_size) {
        return process_xor_layer<decltype(target_size)>(layer);
      });

  // Final reduction based on remaining elements
  constexpr size_t final_size = cute::tuple_size_v<decltype(final_layer)>;
  static_assert(final_size >= 1 && final_size <= 3,
                "Final layer should have 1-3 elements");

  if constexpr (final_size == 1) {
    return final_layer[0];
  } else if constexpr (final_size == 2) {
    return final_layer[0] ^ final_layer[1];
  } else {
    return xor3_lop3(final_layer[0], final_layer[1], final_layer[2]);
  }
}

/// Tile-based hash accumulator for register-optimized transcript updates.
///
/// This struct preloads transcript elements into registers at tile start,
/// accumulates hashes in registers during the tile's k_block loop, then
/// writes back at tile end. This avoids memory accesses in the hot loop.
///
/// Template parameters:
///   KBlocksPerTile: Number of k_blocks per tile (bK / MMAAtom_K)
///   ReduceEveryK:   Reduction frequency (R / MMAAtom_K)
///   EnableDebug:    When true, atomicAdd to debug_counter on each reduction
///
template <int KBlocksPerTile, int ReduceEveryK, bool EnableDebug = false>
struct TileHashAccumulator {
  static constexpr int accums_per_tile =
      std::max<int>(1, KBlocksPerTile / ReduceEveryK);

  static_assert(blake3::MSG_BLOCK_SIZE_U32 % accums_per_tile == 0,
                "accums_per_tile must divide MSG_BLOCK_SIZE_U32");

 private:
  // Register array for accumulating hashes during tile
  uint32_t m_tile_transcript[accums_per_tile];

  // Position in transcript buffer (cycles through 0..MSG_BLOCK_SIZE_U32-1)
  uint32_t m_reduction_count = 0;

  // Running count of k_blocks processed (for reduction condition)
  uint32_t m_k_block_count = 0;

  // Per-instance constants
  uint32_t m_last_full_k_block;
  uint64_t* m_debug_counter;

 public:
  CUTLASS_DEVICE
  TileHashAccumulator(uint32_t last_full_k_block, uint64_t* debug_counter)
      : m_last_full_k_block(last_full_k_block),
        m_debug_counter(debug_counter) {}

  /// Preload transcript elements into registers at tile start
  template <typename TranscriptTensor>
  CUTLASS_DEVICE void preload(TranscriptTensor const& transcript) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < accums_per_tile; ++i) {
      m_tile_transcript[i] = transcript(m_reduction_count + i);
    }
  }

  /// Accumulate hash for this k_block (if reduction conditions are met).
  template <typename TensorType>
  CUTLASS_DEVICE void accumulate(TensorType& tensor, int k_block) {
    ++m_k_block_count;
    if ((m_k_block_count % ReduceEveryK == 0) &&
        (m_k_block_count <= m_last_full_k_block)) {
      // Hopper-only WGMMA sync; no-op on Blackwell consumer (sm_120/121)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
      warpgroup_wait<0>();
      warpgroup_fence_operand(tensor);
#endif
      if constexpr (EnableDebug) {
        atomicAdd((unsigned long long*)m_debug_counter, 1ULL);
      }

      uint32_t hash = xor_reduction(tensor);
      const int idx = k_block / ReduceEveryK;
      m_tile_transcript[idx] =
          rotl_xor<HASH_ACCUMULATE_ROTATION>(m_tile_transcript[idx], hash);
    }
  }

  /// Write back transcript elements after tile completes and advance position
  template <typename TranscriptTensor>
  CUTLASS_DEVICE void writeback(TranscriptTensor& transcript) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < accums_per_tile; ++i) {
      transcript(m_reduction_count + i) = m_tile_transcript[i];
    }

    // In case R > bK we might not need to advance the reduction count
    if ((KBlocksPerTile / ReduceEveryK > 0) ||
        (m_k_block_count % ReduceEveryK == 0)) {
      // Only need modulo at tile boundary
      m_reduction_count =
          (m_reduction_count + accums_per_tile) % blake3::MSG_BLOCK_SIZE_U32;
    }
  }
};

/// Compress transcript using BLAKE3 and check against PoW target.
/// Returns true if hash <= target (block found).
template <typename TranscriptTensor>
CUTLASS_DEVICE bool check_pow_target(const TranscriptTensor& transcript,
                                     const uint32_t* pow_target,
                                     const uint32_t* pow_key) {
  // Compress transcript using keyed BLAKE3 to get 32-byte hash
  Tensor hash = make_tensor<uint32_t>(Int<blake3::CHAINING_VALUE_SIZE_U32>{});
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
    hash(i) = pow_key[i];
  }
  blake3::compress_msg_block_u32(transcript, hash,
                                 blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);

  // uint256 comparison: hash <= target
  // Compare from MSW to LSW (index 7 = MSW, index 0 = LSW)
  bool block_found = true;  // Assume true, set false if hash > target
  CUTLASS_PRAGMA_UNROLL
  for (int i = blake3::CHAINING_VALUE_SIZE_U32 - 1; i >= 0; --i) {
    uint32_t target_i = pow_target[i];
    if (hash(i) > target_i) {
      block_found = false;  // hash > target
      break;
    }
    if (hash(i) < target_i) {
      break;  // hash < target, done
    }
    // hash(i) == target[i], continue to next word
  }

  return block_found;
}

/// Write host signal header with atomic locking.
/// TiledMma: The MMA type for computing thread coordinate partitions
/// TileShape: The tile shape (bM, bN, bK) for the MMA operation
/// ProblemShape: tuple of (M, N, K, R) or (M, N, K)
/// BlockCoord: tuple of (ix, iy, iz) tile coordinates
/// pow_target: uint32_t[8] PoW target for header
template <typename TiledMma, typename TileShape, typename ProblemShape,
          typename BlockCoord>
CUTLASS_DEVICE void write_host_signal_header(
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned,
    ProblemShape const& problem_shape, BlockCoord const& block_coord,
    int thread_idx, const uint32_t* pow_target) {
  auto ix = static_cast<uint32_t>(get<0>(block_coord));
  auto iy = static_cast<uint32_t>(get<1>(block_coord));
  auto iz = static_cast<uint32_t>(get<2>(block_coord));

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

  // Make the predicate tensors for thread coordinates
  Tensor cD = make_identity_tensor(select<0, 1>(TileShape{}));
  Tensor tCcD = thr_mma.partition_C(cD);

  cute::array<uint32_t, blake3::CHAINING_VALUE_SIZE_U32> target;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
    target[i] = pow_target[i];
  }

  // Acquire lock
  while (atomicCAS(&host_signal_sync->global_lock, 0, 1) != 0) {
    __threadfence();
  }

  if (host_signal_sync->status != HostSignalStatus::kSignalTriggered) {
    HostSignalHeader new_header = {
        .status = HostSignalStatus::kSignalTriggered,
        .gridDim = {gridDim.x, gridDim.y, gridDim.z},
        .blockDim = {blockDim.x, blockDim.y, blockDim.z},
        .blockIdx = {blockIdx.x, blockIdx.y, blockIdx.z},
        .tileCoord = {ix, iy, iz},
        .threadIdx = {threadIdx.x, threadIdx.y, threadIdx.z},
        .num_registers_per_thread = static_cast<uint16_t>(size(tCcD)),
        .mma_size = {get<0>(problem_shape), get<1>(problem_shape),
                     get<2>(problem_shape)},
        .mma_tile_size = {get<0>(TileShape{}), get<1>(TileShape{}),
                          get<2>(TileShape{})},
        .target = target,
    };

    static_assert(size(tCcD) <= new_header.thread_rows.size());
    for (int j = 0; j < size(tCcD); j++) {
      auto coord_m = get<0>(tCcD(j));
      auto coord_n = get<1>(tCcD(j));

      new_header.thread_rows[j] = coord_m;
      new_header.thread_cols[j] = coord_n;
    }

    // In case we found a block outside of matrix we dont want to trigger the signal.
    if (new_header.block_in_bounds()) {
      // We copy once to create one DMA transaction as host_signal_header is pinned memory
      *host_signal_header_pinned = new_header;
      host_signal_sync->status = HostSignalStatus::kSignalTriggered;
    }
  }

  // Release lock
  __threadfence();
  atomicExch(&host_signal_sync->global_lock, 0);
}

}  // namespace pearl
