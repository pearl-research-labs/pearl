#pragma once

#include <cutlass/cutlass.h>
#include <cute/atom/mma_traits_sm90_gmma.hpp>

#include "blake3/blake3.cuh"
#include "host_signal_header.hpp"
#include "utils.h"

namespace pearl {

using namespace cute;

#if defined(PEARL_P1K162_NOINLINE_POW_CHECK)
#define PEARL_POW_CHECK_DEVICE __device__ __noinline__
#else
#define PEARL_POW_CHECK_DEVICE CUTLASS_DEVICE
#endif

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

// Diagnostic companion to xor_reduction: preserve the same compile-time
// reduction tree shape without reading WGMMA accumulator registers.
template <typename TensorType>
CUTE_DEVICE uint32_t xor_reduction_dummy(uint32_t salt) {
  constexpr size_t buffer_size =
      decltype(std::declval<TensorType>().size())::value;

  static_assert(buffer_size > 0, "Buffer size must be positive");

  uint32_t seed = static_cast<uint32_t>(threadIdx.x) ^
                  (static_cast<uint32_t>(blockIdx.x) * 0x9e3779b9u) ^
                  (static_cast<uint32_t>(blockIdx.y) * 0x85ebca6bu) ^ salt;
  seed ^= static_cast<uint32_t>(buffer_size * 0x27d4eb2du);
  asm volatile("" : "+r"(seed));
  return seed;
}

/// XOR only accumulator entries whose logical C coordinates fall inside a
/// caller-provided rectangle.
///
/// This is the primitive needed for split-M panel partial records. The normal
/// verifier reducer consumes a whole native 2x64 lane fragment; split-M needs a
/// producer to keep only the selected row cells before transcript rotation
/// destroys row-level separability.
template <typename TensorType, typename CoordTensor>
CUTE_DEVICE uint32_t xor_reduction_selected_by_coord(
    const TensorType& input_tensor, const CoordTensor& coord_tensor,
    uint32_t row_start, uint32_t row_count, uint32_t col_start,
    uint32_t col_count) {
  constexpr size_t buffer_size =
      decltype(std::declval<TensorType>().size())::value;
  constexpr size_t coord_size =
      decltype(std::declval<CoordTensor>().size())::value;
  static_assert(buffer_size > 0, "Buffer size must be positive");
  static_assert(buffer_size == coord_size,
                "coordinate tensor must match accumulator tensor size");

  uint32_t result = 0;
  CUTLASS_PRAGMA_UNROLL
  for (size_t i = 0; i < buffer_size; ++i) {
    auto coord = coord_tensor(i);
    uint32_t const row = static_cast<uint32_t>(get<0>(coord));
    uint32_t const col = static_cast<uint32_t>(get<1>(coord));
    bool const in_row = (row >= row_start) && (row < row_start + row_count);
    bool const in_col = (col >= col_start) && (col < col_start + col_count);
    if (in_row && in_col) {
      result ^= static_cast<uint32_t>(input_tensor[i]);
    }
  }
  return result;
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
///   UseDummyReduction: Diagnostic-only non-accumulator register reduction
///
template <int KBlocksPerTile, int ReduceEveryK, bool EnableDebug = false,
          bool UseDummyReduction = false>
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

  /// Diagnostic-only state initializer for P1K-127. This keeps the real
  /// TileHashAccumulator update path live while removing transcript preload
  /// traffic from the measured branch.
  CUTLASS_DEVICE void init_zero_state() {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < accums_per_tile; ++i) {
      m_tile_transcript[i] = 0;
    }
  }

  /// Accumulate hash for this k_block (if reduction conditions are met).
  template <typename TensorType>
  CUTLASS_DEVICE void accumulate(TensorType& tensor, int k_block) {
    ++m_k_block_count;
    if ((m_k_block_count % ReduceEveryK == 0) &&
        (m_k_block_count <= m_last_full_k_block)) {
      warpgroup_wait<0>();
      warpgroup_fence_operand(tensor);
      accumulate_waited(tensor, k_block);
    }
  }

  /// Accumulate after the caller has already completed the required
  /// warpgroup_wait/fence. This is used for rank-boundary snapshots that are
  /// naturally at tile end, so the mainloop can share one wait with stage
  /// release instead of draining the WGMMA group inside the k_block body.
	  template <typename TensorType>
	  CUTLASS_DEVICE void accumulate_after_wait(TensorType& tensor, int k_block,
	                                            int consumed_k_blocks) {
	    m_k_block_count += consumed_k_blocks;
	    if ((m_k_block_count % ReduceEveryK == 0) &&
	        (m_k_block_count <= m_last_full_k_block)) {
	      accumulate_waited(tensor, k_block);
	    }
	  }

	  template <typename TensorType>
	  CUTLASS_DEVICE void accumulate_dummy_after_wait(int k_block,
	                                                  int consumed_k_blocks) {
	    static_assert(UseDummyReduction,
	                  "accumulate_dummy_after_wait is only for dummy reduction");
	    m_k_block_count += consumed_k_blocks;
	    if ((m_k_block_count % ReduceEveryK == 0) &&
	        (m_k_block_count <= m_last_full_k_block)) {
	      if constexpr (EnableDebug) {
	        atomicAdd((unsigned long long*)m_debug_counter, 1ULL);
	      }

	      uint32_t hash = xor_reduction_dummy<TensorType>(
	          static_cast<uint32_t>(k_block) ^ m_k_block_count);
	      const int idx = k_block / ReduceEveryK;
	      m_tile_transcript[idx] =
	          rotl_xor<HASH_ACCUMULATE_ROTATION>(m_tile_transcript[idx], hash);
	    }
	  }

	 private:
  template <typename TensorType>
  CUTLASS_DEVICE void accumulate_waited(TensorType& tensor, int k_block) {
    if constexpr (EnableDebug) {
      atomicAdd((unsigned long long*)m_debug_counter, 1ULL);
    }

    uint32_t hash;
    if constexpr (UseDummyReduction) {
      hash = xor_reduction_dummy<TensorType>(
          static_cast<uint32_t>(k_block) ^ m_k_block_count);
    } else {
      hash = xor_reduction(tensor);
    }
    const int idx = k_block / ReduceEveryK;
    m_tile_transcript[idx] =
        rotl_xor<HASH_ACCUMULATE_ROTATION>(m_tile_transcript[idx], hash);
  }

 public:

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

  /// Diagnostic-only sink for P1K-126: keep the real transcript update live in
  /// registers, but do not publish it to the verifier-visible transcript tensor.
  CUTLASS_DEVICE void sink_register_state() const {
    uint32_t folded = m_reduction_count ^ (m_k_block_count * 0x9e3779b9u) ^
                      (m_last_full_k_block * 0x85ebca6bu);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < accums_per_tile; ++i) {
      folded = rotl_xor<HASH_ACCUMULATE_ROTATION>(
          folded, m_tile_transcript[i] + uint32_t(i) * 0x27d4eb2du);
    }
    asm volatile("" : : "r"(folded) : "memory");
  }
};

/// Build a verifier transcript for a selected coordinate rectangle inside the
/// native accumulator fragment.
///
/// This does not replace the default fast full-fragment reducer. It exists for
/// split-M panel partial records, where each physical panel owns only a selected
/// row subset and must emit a prehash transcript before a separate finalizer
/// XOR-combines disjoint panels.
template <int KBlocksPerTile, int ReduceEveryK, bool EnableDebug = false>
struct TileSelectedTranscriptAccumulator {
 private:
  uint32_t m_transcript[blake3::MSG_BLOCK_SIZE_U32];
  uint32_t m_boundary_count = 0;
  uint32_t m_k_block_count = 0;
  uint32_t m_last_full_k_block;
  uint64_t* m_debug_counter;

 public:
  CUTLASS_DEVICE
  TileSelectedTranscriptAccumulator(uint32_t last_full_k_block,
                                    uint64_t* debug_counter)
      : m_last_full_k_block(last_full_k_block),
        m_debug_counter(debug_counter) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::MSG_BLOCK_SIZE_U32; ++i) {
      m_transcript[i] = 0;
    }
  }

  template <typename TensorType, typename CoordTensor>
  CUTLASS_DEVICE void accumulate(TensorType& tensor,
                                 CoordTensor const& coord_tensor,
                                 uint32_t row_start, uint32_t row_count,
                                 uint32_t col_start, uint32_t col_count) {
    ++m_k_block_count;
    if ((m_k_block_count % ReduceEveryK == 0) &&
        (m_k_block_count <= m_last_full_k_block)) {
      warpgroup_wait<0>();
      warpgroup_fence_operand(tensor);
      if constexpr (EnableDebug) {
        atomicAdd((unsigned long long*)m_debug_counter, 1ULL);
      }

      uint32_t const hash = xor_reduction_selected_by_coord(
          tensor, coord_tensor, row_start, row_count, col_start, col_count);
      int const slot = m_boundary_count % blake3::MSG_BLOCK_SIZE_U32;
      m_transcript[slot] =
          rotl_xor<HASH_ACCUMULATE_ROTATION>(m_transcript[slot], hash);
      ++m_boundary_count;
    }
  }

  CUTLASS_DEVICE void write_words(uint32_t* out) const {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::MSG_BLOCK_SIZE_U32; ++i) {
      out[i] = m_transcript[i];
    }
  }
};

struct NullTranscriptAccumulator {
  CUTLASS_DEVICE
  NullTranscriptAccumulator(uint32_t, uint64_t*) {}

  template <typename... Args>
  CUTLASS_DEVICE void preload(Args&&...) {}

  CUTLASS_DEVICE void init_zero_state() {}

  template <typename... Args>
  CUTLASS_DEVICE void accumulate(Args&&...) {}

  template <typename... Args>
  CUTLASS_DEVICE void accumulate_after_wait(Args&&...) {}

  template <typename... Args>
  CUTLASS_DEVICE void accumulate_dummy_after_wait(Args&&...) {}

  template <typename... Args>
  CUTLASS_DEVICE void writeback(Args&&...) {}

  CUTLASS_DEVICE void sink_register_state() const {}

  template <typename... Args>
  CUTLASS_DEVICE void write_words(Args&&...) const {}
};

template <typename KTraits>
constexpr bool native_global_journal_elides_consumer_transcript() {
#if defined(PEARL_P1K148_SCALAR16_DEFERRED_TRANSCRIPT) || \
    defined(PEARL_P1K150_SCALAR16_FINAL_GLOBAL_STORE) || \
    defined(PEARL_P1K154_SCALAR16_FINAL_SHARED_STORE) || \
    defined(PEARL_P1K165_TWO_PHASE_POW_CHECK)
  return false;
#else
  return KTraits::EnableNativeGlobalJournalFill && KTraits::SkipProofCheck &&
         !KTraits::EnableXqJournal && !KTraits::EnableCanonicalTranscript &&
         !KTraits::EnableDummyReduction;
#endif
}

/// Journal raw verifier rank-boundary X_q values into shared memory.
///
/// This preserves the mandatory WGMMA wait/fence and local accumulator XOR at
/// each proof boundary, but avoids carrying the 16-word transcript state through
/// the mainloop. The caller reconstructs the transcript after K completes.
template <int KBlocksPerTile, int ReduceEveryK, int MaxBoundaries,
          bool EnableDebug = false>
struct TileXqJournalAccumulator {
 private:
  uint32_t m_boundary_count = 0;
  uint32_t m_k_block_count = 0;
  uint32_t m_last_full_k_block;
  uint64_t* m_debug_counter;

 public:
  CUTLASS_DEVICE
  TileXqJournalAccumulator(uint32_t last_full_k_block, uint64_t* debug_counter)
      : m_last_full_k_block(last_full_k_block),
        m_debug_counter(debug_counter) {}

  template <typename TensorType>
  CUTLASS_DEVICE void accumulate(TensorType& tensor, int consumer_thread_idx,
                                 uint32_t* xq_journal) {
    ++m_k_block_count;
    if ((m_k_block_count % ReduceEveryK == 0) &&
        (m_k_block_count <= m_last_full_k_block)) {
      warpgroup_wait<0>();
      warpgroup_fence_operand(tensor);
      if constexpr (EnableDebug) {
        atomicAdd((unsigned long long*)m_debug_counter, 1ULL);
      }

      uint32_t hash = xor_reduction(tensor);
      if (m_boundary_count < MaxBoundaries) {
        xq_journal[consumer_thread_idx * MaxBoundaries + m_boundary_count] =
            hash;
      }
      ++m_boundary_count;
    }
  }
};

template <int MaxBoundaries, typename TranscriptTensor>
CUTLASS_DEVICE void reconstruct_transcript_from_xq_journal(
    TranscriptTensor& transcript, const uint32_t* xq_journal,
    int consumer_thread_idx, int boundary_count) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < blake3::MSG_BLOCK_SIZE_U32; ++i) {
    transcript(i) = 0;
  }

  CUTLASS_PRAGMA_NO_UNROLL
  for (int q = 0; q < boundary_count && q < MaxBoundaries; ++q) {
    int const slot = q % blake3::MSG_BLOCK_SIZE_U32;
    uint32_t hash = xq_journal[consumer_thread_idx * MaxBoundaries + q];
    transcript(slot) =
        rotl_xor<HASH_ACCUMULATE_ROTATION>(transcript(slot), hash);
  }
}

CUTE_DEVICE void write_panel_partial_transcript_record_v2(
    PanelPartialTranscriptRecordV2* records, int capacity, int record_index,
    uint64_t logical_receipt_id, uint16_t panel_slot, uint16_t panel_count,
    uint32_t row_start, uint32_t row_count, uint32_t col_start,
    uint32_t col_count, uint3 producer_block, uint3 producer_tile,
    uint32_t producer_thread, const uint32_t* transcript_words) {
  if (records == nullptr || record_index < 0 || record_index >= capacity) {
    return;
  }

  PanelPartialTranscriptRecordV2& record = records[record_index];
  // The panel producer may be a whole physical CTA, not one native lane. The
  // caller zeroes the record buffer before launch, then all contributing
  // consumer threads XOR their disjoint selected-cell transcript into this
  // record. Metadata writes are idempotent for one physical panel.
  record.magic = kPanelPartialTranscriptV2Magic;
  record.version = kPanelPartialTranscriptV2Version;
  record.flags = 1;
  record.logical_receipt_id = logical_receipt_id;
  record.panel_slot = panel_slot;
  record.panel_count = panel_count;
  record.row_start = row_start;
  record.row_count = row_count;
  record.col_start = col_start;
  record.col_count = col_count;
  record.producer_block[0] = producer_block.x;
  record.producer_block[1] = producer_block.y;
  record.producer_block[2] = producer_block.z;
  record.producer_tile[0] = producer_tile.x;
  record.producer_tile[1] = producer_tile.y;
  record.producer_tile[2] = producer_tile.z;
  record.producer_thread = producer_thread;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < blake3::MSG_BLOCK_SIZE_U32; ++i) {
    atomicXor(reinterpret_cast<unsigned int*>(&record.transcript_words[i]),
              static_cast<unsigned int>(transcript_words[i]));
  }
}

template <int KBlocksPerTile, int ReduceEveryK, bool EnableDebug = false>
struct CanonicalSlotStreamAccumulator {
 private:
  uint32_t m_boundary_count = 0;
  uint32_t m_k_block_count = 0;
  uint32_t m_last_full_k_block;
  uint64_t* m_debug_counter;

 public:
  CUTLASS_DEVICE
  CanonicalSlotStreamAccumulator(uint32_t last_full_k_block,
                                 uint64_t* debug_counter)
      : m_last_full_k_block(last_full_k_block),
        m_debug_counter(debug_counter) {}

  template <typename TensorType>
  CUTLASS_DEVICE void accumulate(TensorType& tensor, int consumer_thread_idx,
                                 uint32_t* transcript_slots) {
    ++m_k_block_count;
    if ((m_k_block_count % ReduceEveryK == 0) &&
        (m_k_block_count <= m_last_full_k_block)) {
      warpgroup_wait<0>();
      warpgroup_fence_operand(tensor);
      if constexpr (EnableDebug) {
        atomicAdd((unsigned long long*)m_debug_counter, 1ULL);
      }

      uint32_t hash = xor_reduction(tensor);
      int const slot = m_boundary_count % blake3::MSG_BLOCK_SIZE_U32;
      int const offset =
          consumer_thread_idx * blake3::MSG_BLOCK_SIZE_U32 + slot;
      transcript_slots[offset] =
          rotl_xor<HASH_ACCUMULATE_ROTATION>(transcript_slots[offset], hash);
      ++m_boundary_count;
    }
  }

  template <typename TensorType>
  CUTLASS_DEVICE void accumulate_after_wait(TensorType& tensor,
                                            int consumer_thread_idx,
                                            uint32_t* transcript_slots,
                                            int consumed_k_blocks) {
    m_k_block_count += consumed_k_blocks;
    if ((m_k_block_count % ReduceEveryK == 0) &&
        (m_k_block_count <= m_last_full_k_block)) {
      if constexpr (EnableDebug) {
        atomicAdd((unsigned long long*)m_debug_counter, 1ULL);
      }

      uint32_t hash = xor_reduction(tensor);
      int const slot = m_boundary_count % blake3::MSG_BLOCK_SIZE_U32;
      int const offset =
          consumer_thread_idx * blake3::MSG_BLOCK_SIZE_U32 + slot;
      transcript_slots[offset] =
          rotl_xor<HASH_ACCUMULATE_ROTATION>(transcript_slots[offset], hash);
      ++m_boundary_count;
    }
  }
};

CUTLASS_DEVICE void clear_canonical_transcript_slots(
    uint32_t* transcript_slots, int consumer_thread_idx) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < blake3::MSG_BLOCK_SIZE_U32; ++i) {
    transcript_slots[consumer_thread_idx * blake3::MSG_BLOCK_SIZE_U32 + i] = 0;
  }
}

template <typename TranscriptTensor>
CUTLASS_DEVICE void load_canonical_transcript_slots(
    TranscriptTensor& transcript, const uint32_t* transcript_slots,
    int consumer_thread_idx) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < blake3::MSG_BLOCK_SIZE_U32; ++i) {
    transcript(i) =
        transcript_slots[consumer_thread_idx * blake3::MSG_BLOCK_SIZE_U32 + i];
  }
}

#if defined(PEARL_P1K163_MANUAL_SCALAR16_POW_CHECK)
CUTLASS_DEVICE uint32_t p1k163_rotr32(uint32_t x, int n) {
  return (x << (32 - n)) | (x >> n);
}

#define PEARL_P1K163_G(a, b, c, d, mx, my) \
  do {                                      \
    a = a + b + mx;                         \
    d = p1k163_rotr32(d ^ a, 16);           \
    c = c + d;                              \
    b = p1k163_rotr32(b ^ c, 12);           \
    a = a + b + my;                         \
    d = p1k163_rotr32(d ^ a, 8);            \
    c = c + d;                              \
    b = p1k163_rotr32(b ^ c, 7);            \
  } while (0)

#define PEARL_P1K163_ROUND()              \
  do {                                    \
    PEARL_P1K163_G(v0, v4, v8, v12, b0, b1);     \
    PEARL_P1K163_G(v1, v5, v9, v13, b2, b3);     \
    PEARL_P1K163_G(v2, v6, v10, v14, b4, b5);    \
    PEARL_P1K163_G(v3, v7, v11, v15, b6, b7);    \
    PEARL_P1K163_G(v0, v5, v10, v15, b8, b9);    \
    PEARL_P1K163_G(v1, v6, v11, v12, b10, b11);  \
    PEARL_P1K163_G(v2, v7, v8, v13, b12, b13);   \
    PEARL_P1K163_G(v3, v4, v9, v14, b14, b15);   \
  } while (0)

#define PEARL_P1K163_PERMUTE()      \
  do {                              \
    uint32_t o0 = b0;               \
    uint32_t o1 = b1;               \
    uint32_t o2 = b2;               \
    uint32_t o3 = b3;               \
    uint32_t o4 = b4;               \
    uint32_t o5 = b5;               \
    uint32_t o6 = b6;               \
    uint32_t o7 = b7;               \
    uint32_t o8 = b8;               \
    uint32_t o9 = b9;               \
    uint32_t o10 = b10;             \
    uint32_t o11 = b11;             \
    uint32_t o12 = b12;             \
    uint32_t o13 = b13;             \
    uint32_t o14 = b14;             \
    uint32_t o15 = b15;             \
    b0 = o2;                        \
    b1 = o6;                        \
    b2 = o3;                        \
    b3 = o10;                       \
    b4 = o7;                        \
    b5 = o0;                        \
    b6 = o4;                        \
    b7 = o13;                       \
    b8 = o1;                        \
    b9 = o11;                       \
    b10 = o12;                      \
    b11 = o5;                       \
    b12 = o9;                       \
    b13 = o14;                      \
    b14 = o15;                      \
    b15 = o8;                       \
  } while (0)

template <typename TranscriptTensor>
CUTLASS_DEVICE bool check_pow_target_manual_scalar16(
    const TranscriptTensor& transcript, const uint32_t* pow_target,
    const uint32_t* pow_key) {
  uint32_t b0 = transcript(0);
  uint32_t b1 = transcript(1);
  uint32_t b2 = transcript(2);
  uint32_t b3 = transcript(3);
  uint32_t b4 = transcript(4);
  uint32_t b5 = transcript(5);
  uint32_t b6 = transcript(6);
  uint32_t b7 = transcript(7);
  uint32_t b8 = transcript(8);
  uint32_t b9 = transcript(9);
  uint32_t b10 = transcript(10);
  uint32_t b11 = transcript(11);
  uint32_t b12 = transcript(12);
  uint32_t b13 = transcript(13);
  uint32_t b14 = transcript(14);
  uint32_t b15 = transcript(15);

  uint32_t v0 = pow_key[0];
  uint32_t v1 = pow_key[1];
  uint32_t v2 = pow_key[2];
  uint32_t v3 = pow_key[3];
  uint32_t v4 = pow_key[4];
  uint32_t v5 = pow_key[5];
  uint32_t v6 = pow_key[6];
  uint32_t v7 = pow_key[7];
  uint32_t v8 = blake3::IV0;
  uint32_t v9 = blake3::IV1;
  uint32_t v10 = blake3::IV2;
  uint32_t v11 = blake3::IV3;
  uint32_t v12 = 0;
  uint32_t v13 = 0;
  uint32_t v14 = blake3::MSG_BLOCK_SIZE;
  uint32_t v15 =
      blake3::KEYED_HASH | blake3::CHUNK_START | blake3::CHUNK_END |
      blake3::ROOT;

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 6; ++i) {
    PEARL_P1K163_ROUND();
    PEARL_P1K163_PERMUTE();
  }
  PEARL_P1K163_ROUND();

  uint32_t h0 = v0 ^ v8;
  uint32_t h1 = v1 ^ v9;
  uint32_t h2 = v2 ^ v10;
  uint32_t h3 = v3 ^ v11;
  uint32_t h4 = v4 ^ v12;
  uint32_t h5 = v5 ^ v13;
  uint32_t h6 = v6 ^ v14;
  uint32_t h7 = v7 ^ v15;

  if (h7 > pow_target[7]) return false;
  if (h7 < pow_target[7]) return true;
  if (h6 > pow_target[6]) return false;
  if (h6 < pow_target[6]) return true;
  if (h5 > pow_target[5]) return false;
  if (h5 < pow_target[5]) return true;
  if (h4 > pow_target[4]) return false;
  if (h4 < pow_target[4]) return true;
  if (h3 > pow_target[3]) return false;
  if (h3 < pow_target[3]) return true;
  if (h2 > pow_target[2]) return false;
  if (h2 < pow_target[2]) return true;
  if (h1 > pow_target[1]) return false;
  if (h1 < pow_target[1]) return true;
  return h0 <= pow_target[0];
}
#endif

#if defined(PEARL_P1K164_SCHEDULED_SCALAR16_POW_CHECK)
template <int N>
CUTLASS_DEVICE uint32_t p1k164_rotr32(uint32_t x) {
  static_assert(N > 0 && N < 32, "rotate count must be in (0, 32)");
  return (x << (32 - N)) | (x >> N);
}

#define PEARL_P1K164_G(a, b, c, d, mx, my) \
  do {                                      \
    a = a + b + mx;                         \
    d = p1k164_rotr32<16>(d ^ a);           \
    c = c + d;                              \
    b = p1k164_rotr32<12>(b ^ c);           \
    a = a + b + my;                         \
    d = p1k164_rotr32<8>(d ^ a);            \
    c = c + d;                              \
    b = p1k164_rotr32<7>(b ^ c);            \
  } while (0)

#define PEARL_P1K164_ROUND(m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, \
                           m12, m13, m14, m15)                               \
  do {                                                                        \
    PEARL_P1K164_G(v0, v4, v8, v12, m0, m1);                                  \
    PEARL_P1K164_G(v1, v5, v9, v13, m2, m3);                                  \
    PEARL_P1K164_G(v2, v6, v10, v14, m4, m5);                                 \
    PEARL_P1K164_G(v3, v7, v11, v15, m6, m7);                                 \
    PEARL_P1K164_G(v0, v5, v10, v15, m8, m9);                                 \
    PEARL_P1K164_G(v1, v6, v11, v12, m10, m11);                               \
    PEARL_P1K164_G(v2, v7, v8, v13, m12, m13);                                \
    PEARL_P1K164_G(v3, v4, v9, v14, m14, m15);                                \
  } while (0)

template <typename TranscriptTensor>
CUTLASS_DEVICE bool check_pow_target_scheduled_scalar16(
    const TranscriptTensor& transcript, const uint32_t* pow_target,
    const uint32_t* pow_key) {
  uint32_t v0 = pow_key[0];
  uint32_t v1 = pow_key[1];
  uint32_t v2 = pow_key[2];
  uint32_t v3 = pow_key[3];
  uint32_t v4 = pow_key[4];
  uint32_t v5 = pow_key[5];
  uint32_t v6 = pow_key[6];
  uint32_t v7 = pow_key[7];
  uint32_t v8 = blake3::IV0;
  uint32_t v9 = blake3::IV1;
  uint32_t v10 = blake3::IV2;
  uint32_t v11 = blake3::IV3;
  uint32_t v12 = 0;
  uint32_t v13 = 0;
  uint32_t v14 = blake3::MSG_BLOCK_SIZE;
  uint32_t v15 =
      blake3::KEYED_HASH | blake3::CHUNK_START | blake3::CHUNK_END |
      blake3::ROOT;

  PEARL_P1K164_ROUND(transcript(0), transcript(1), transcript(2),
                     transcript(3), transcript(4), transcript(5),
                     transcript(6), transcript(7), transcript(8),
                     transcript(9), transcript(10), transcript(11),
                     transcript(12), transcript(13), transcript(14),
                     transcript(15));
  PEARL_P1K164_ROUND(transcript(2), transcript(6), transcript(3),
                     transcript(10), transcript(7), transcript(0),
                     transcript(4), transcript(13), transcript(1),
                     transcript(11), transcript(12), transcript(5),
                     transcript(9), transcript(14), transcript(15),
                     transcript(8));
  PEARL_P1K164_ROUND(transcript(3), transcript(4), transcript(10),
                     transcript(12), transcript(13), transcript(2),
                     transcript(7), transcript(14), transcript(6),
                     transcript(5), transcript(9), transcript(0),
                     transcript(11), transcript(15), transcript(8),
                     transcript(1));
  PEARL_P1K164_ROUND(transcript(10), transcript(7), transcript(12),
                     transcript(9), transcript(14), transcript(3),
                     transcript(13), transcript(15), transcript(4),
                     transcript(0), transcript(11), transcript(2),
                     transcript(5), transcript(8), transcript(1),
                     transcript(6));
  PEARL_P1K164_ROUND(transcript(12), transcript(13), transcript(9),
                     transcript(11), transcript(15), transcript(10),
                     transcript(14), transcript(8), transcript(7),
                     transcript(2), transcript(5), transcript(3),
                     transcript(0), transcript(1), transcript(6),
                     transcript(4));
  PEARL_P1K164_ROUND(transcript(9), transcript(14), transcript(11),
                     transcript(5), transcript(8), transcript(12),
                     transcript(15), transcript(1), transcript(13),
                     transcript(3), transcript(0), transcript(10),
                     transcript(2), transcript(6), transcript(4),
                     transcript(7));
  PEARL_P1K164_ROUND(transcript(11), transcript(15), transcript(5),
                     transcript(0), transcript(1), transcript(9),
                     transcript(8), transcript(6), transcript(14),
                     transcript(10), transcript(2), transcript(12),
                     transcript(3), transcript(4), transcript(7),
                     transcript(13));

  uint32_t h0 = v0 ^ v8;
  uint32_t h1 = v1 ^ v9;
  uint32_t h2 = v2 ^ v10;
  uint32_t h3 = v3 ^ v11;
  uint32_t h4 = v4 ^ v12;
  uint32_t h5 = v5 ^ v13;
  uint32_t h6 = v6 ^ v14;
  uint32_t h7 = v7 ^ v15;

  if (h7 > pow_target[7]) return false;
  if (h7 < pow_target[7]) return true;
  if (h6 > pow_target[6]) return false;
  if (h6 < pow_target[6]) return true;
  if (h5 > pow_target[5]) return false;
  if (h5 < pow_target[5]) return true;
  if (h4 > pow_target[4]) return false;
  if (h4 < pow_target[4]) return true;
  if (h3 > pow_target[3]) return false;
  if (h3 < pow_target[3]) return true;
  if (h2 > pow_target[2]) return false;
  if (h2 < pow_target[2]) return true;
  if (h1 > pow_target[1]) return false;
  if (h1 < pow_target[1]) return true;
  return h0 <= pow_target[0];
}
#endif

/// Compress transcript using BLAKE3 and check against PoW target.
/// Returns true if hash <= target (block found).
template <typename TranscriptTensor>
PEARL_POW_CHECK_DEVICE bool check_pow_target(const TranscriptTensor& transcript,
                                             const uint32_t* pow_target,
                                             const uint32_t* pow_key) {
#if defined(PEARL_P1K161_NOOP_POW_CHECK)
  (void)transcript;
  (void)pow_target;
  (void)pow_key;
  uint32_t nohit = 0;
  asm volatile("" : "+r"(nohit) :: "memory");
  return nohit != 0;
#elif defined(PEARL_P1K164_SCHEDULED_SCALAR16_POW_CHECK)
  return check_pow_target_scheduled_scalar16(transcript, pow_target, pow_key);
#elif defined(PEARL_P1K163_MANUAL_SCALAR16_POW_CHECK)
  return check_pow_target_manual_scalar16(transcript, pow_target, pow_key);
#else
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
#endif
}

/// XOR this lane's 16-word jackpot transcript with lane^1's transcript.
///
/// This is the cheapest first coalescing prototype: adjacent consumer lanes
/// already own disjoint WGMMA accumulator fragments. XOR-linear transcript
/// mixing means the pre-hash message for the union is the XOR of the two
/// zero-initialized child messages, as long as the emitted coordinate union is
/// verifier-encodable.
template <typename TranscriptTensor>
CUTLASS_DEVICE auto make_adjacent_lane_union_transcript(
    const TranscriptTensor& transcript) {
  Tensor result = make_tensor<uint32_t>(Int<blake3::MSG_BLOCK_SIZE_U32>{});
  unsigned mask = __activemask();
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < blake3::MSG_BLOCK_SIZE_U32; ++i) {
    uint32_t partner = __shfl_xor_sync(mask, transcript(i), 1);
    result(i) = transcript(i) ^ partner;
  }
  return result;
}

/// XOR the four transcripts in this lane's aligned 4-lane group.
///
/// The caller must only publish from the group leader. The matching header path
/// below validates that the four lanes emit exactly the verifier's 4x64
/// coordinate shape before signalling the host.
template <typename TranscriptTensor>
CUTLASS_DEVICE auto make_4x64_lane_union_transcript(
    const TranscriptTensor& transcript) {
  Tensor result = make_tensor<uint32_t>(Int<blake3::MSG_BLOCK_SIZE_U32>{});
  unsigned mask = __activemask();
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < blake3::MSG_BLOCK_SIZE_U32; ++i) {
    uint32_t v = transcript(i);
    v ^= __shfl_xor_sync(mask, transcript(i), 1);
    v ^= __shfl_xor_sync(mask, transcript(i), 2);
    v ^= __shfl_xor_sync(mask, transcript(i), 3);
    result(i) = v;
  }
  return result;
}

template <typename TranscriptTensor>
CUTLASS_DEVICE void write_panel_partial_transcript_record(
    PanelPartialTranscriptRecord* records, uint32_t record_idx,
    uint32_t panel_id, uint32_t partial_id, uint32_t tile_m, uint32_t tile_n,
    uint32_t tile_k, int thread_idx, const TranscriptTensor& transcript) {
  PanelPartialTranscriptRecord record;
  record.panel_id = panel_id;
  record.partial_id = partial_id;
  record.producer_block[0] = blockIdx.x;
  record.producer_block[1] = blockIdx.y;
  record.producer_block[2] = blockIdx.z;
  record.producer_tile[0] = tile_m;
  record.producer_tile[1] = tile_n;
  record.producer_tile[2] = tile_k;
  record.producer_thread = static_cast<uint32_t>(thread_idx);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < blake3::MSG_BLOCK_SIZE_U32; ++i) {
    record.transcript_words[i] = transcript(i);
  }
  records[record_idx] = record;
}

template <typename TranscriptTensor>
CUTLASS_DEVICE void combine_panel_partial_transcripts(
    TranscriptTensor& combined, const PanelPartialTranscriptRecord* records,
    int record_count) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < blake3::MSG_BLOCK_SIZE_U32; ++i) {
    combined(i) = 0;
  }
  CUTLASS_PRAGMA_NO_UNROLL
  for (int r = 0; r < record_count; ++r) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::MSG_BLOCK_SIZE_U32; ++i) {
      combined(i) ^= records[r].transcript_words[i];
    }
  }
}

template <auto MaxCount>
CUTLASS_DEVICE int unique_count(cute::array<uint16_t, MaxCount> const& values,
                                int value_count) {
  int unique = 0;
  CUTLASS_PRAGMA_NO_UNROLL
  for (int i = 0; i < value_count; ++i) {
    bool seen = false;
    CUTLASS_PRAGMA_NO_UNROLL
    for (int j = 0; j < i; ++j) {
      seen = seen || (values[j] == values[i]);
    }
    unique += seen ? 0 : 1;
  }
  return unique;
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
        .actual_receipt_h = 0,
        .actual_receipt_w = 0,
        .actual_receipt_cells = 0,
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

      new_header.thread_rows[j] = static_cast<uint16_t>(coord_m);
      new_header.thread_cols[j] = static_cast<uint16_t>(coord_n);
    }
    int const emitted_rows = unique_count(
        new_header.thread_rows, new_header.num_registers_per_thread);
    int const emitted_cols = unique_count(
        new_header.thread_cols, new_header.num_registers_per_thread);
    new_header.actual_receipt_h = static_cast<uint16_t>(emitted_rows);
    new_header.actual_receipt_w = static_cast<uint16_t>(emitted_cols);
    new_header.actual_receipt_cells =
        static_cast<uint32_t>(emitted_rows * emitted_cols);

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

/// Write a host signal for a coalesced adjacent-lane receipt.
///
/// The header stores the coordinate union from two WGMMA consumer threads. The
/// Python helper reduces these coordinates to sorted unique row/column sets,
/// and this P1K-021 gate only publishes the signal when the union is exactly
/// 2x128. A 4x64 fallback should use a separate explicitly named path.
template <typename TiledMma, typename TileShape, typename ProblemShape,
          typename BlockCoord>
CUTLASS_DEVICE void write_host_signal_header_pair(
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned,
    ProblemShape const& problem_shape, BlockCoord const& block_coord,
    int thread_idx_a, int thread_idx_b, const uint32_t* pow_target) {
  auto ix = static_cast<uint32_t>(get<0>(block_coord));
  auto iy = static_cast<uint32_t>(get<1>(block_coord));
  auto iz = static_cast<uint32_t>(get<2>(block_coord));

  TiledMma tiled_mma;
  auto thr_mma_a = tiled_mma.get_thread_slice(thread_idx_a);
  auto thr_mma_b = tiled_mma.get_thread_slice(thread_idx_b);

  Tensor cD = make_identity_tensor(select<0, 1>(TileShape{}));
  Tensor tCcD_a = thr_mma_a.partition_C(cD);
  Tensor tCcD_b = thr_mma_b.partition_C(cD);

  static_assert(size(tCcD_a) + size(tCcD_b) <=
                MAX_NUM_REGISTERS_PER_THREAD);

  cute::array<uint32_t, blake3::CHAINING_VALUE_SIZE_U32> target;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
    target[i] = pow_target[i];
  }

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
        .num_registers_per_thread =
            static_cast<uint16_t>(size(tCcD_a) + size(tCcD_b)),
        .actual_receipt_h = 0,
        .actual_receipt_w = 0,
        .actual_receipt_cells = 0,
        .mma_size = {get<0>(problem_shape), get<1>(problem_shape),
                     get<2>(problem_shape)},
        .mma_tile_size = {get<0>(TileShape{}), get<1>(TileShape{}),
                          get<2>(TileShape{})},
        .target = target,
    };

    int out = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < size(tCcD_a); j++) {
      new_header.thread_rows[out] = static_cast<uint16_t>(get<0>(tCcD_a(j)));
      new_header.thread_cols[out] = static_cast<uint16_t>(get<1>(tCcD_a(j)));
      ++out;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < size(tCcD_b); j++) {
      new_header.thread_rows[out] = static_cast<uint16_t>(get<0>(tCcD_b(j)));
      new_header.thread_cols[out] = static_cast<uint16_t>(get<1>(tCcD_b(j)));
      ++out;
    }

    int const emitted_rows = unique_count(
        new_header.thread_rows, new_header.num_registers_per_thread);
    int const emitted_cols = unique_count(
        new_header.thread_cols, new_header.num_registers_per_thread);
    bool const emits_2x128 = (emitted_rows == 2 && emitted_cols == 128);
    new_header.actual_receipt_h = static_cast<uint16_t>(emitted_rows);
    new_header.actual_receipt_w = static_cast<uint16_t>(emitted_cols);
    new_header.actual_receipt_cells =
        static_cast<uint32_t>(emitted_rows * emitted_cols);

    // P1K-032 adjacent-pair canary: publish the actual coordinate union so the
    // host gate can prove whether the native pair is the intended 2x128 shape.
    if (new_header.block_in_bounds()) {
      *host_signal_header_pinned = new_header;
      host_signal_sync->status = HostSignalStatus::kSignalTriggered;
    }
  }

  __threadfence();
  atomicExch(&host_signal_sync->global_lock, 0);
}

/// Write a host signal for an explicitly 4x64 coalesced receipt.
///
/// This is the P1K-022 wide-receipt path: the same four consumer lanes whose
/// transcripts were XORed for the PoW target check are emitted into the
/// HostSignalHeader, and publication is gated on the actual 4x64 shape.
template <typename TiledMma, typename TileShape, typename ProblemShape,
          typename BlockCoord>
CUTLASS_DEVICE void write_host_signal_header_4x64(
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned,
    ProblemShape const& problem_shape, BlockCoord const& block_coord,
    int thread_idx_base, const uint32_t* pow_target) {
  auto ix = static_cast<uint32_t>(get<0>(block_coord));
  auto iy = static_cast<uint32_t>(get<1>(block_coord));
  auto iz = static_cast<uint32_t>(get<2>(block_coord));

  TiledMma tiled_mma;
  auto thr_mma_0 = tiled_mma.get_thread_slice(thread_idx_base + 0);
  auto thr_mma_1 = tiled_mma.get_thread_slice(thread_idx_base + 1);
  auto thr_mma_2 = tiled_mma.get_thread_slice(thread_idx_base + 2);
  auto thr_mma_3 = tiled_mma.get_thread_slice(thread_idx_base + 3);

  Tensor cD = make_identity_tensor(select<0, 1>(TileShape{}));
  Tensor tCcD_0 = thr_mma_0.partition_C(cD);
  Tensor tCcD_1 = thr_mma_1.partition_C(cD);
  Tensor tCcD_2 = thr_mma_2.partition_C(cD);
  Tensor tCcD_3 = thr_mma_3.partition_C(cD);

  static_assert(size(tCcD_0) + size(tCcD_1) + size(tCcD_2) + size(tCcD_3) <=
                MAX_NUM_REGISTERS_PER_THREAD);

  cute::array<uint32_t, blake3::CHAINING_VALUE_SIZE_U32> target;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
    target[i] = pow_target[i];
  }

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
        .num_registers_per_thread = static_cast<uint16_t>(
            size(tCcD_0) + size(tCcD_1) + size(tCcD_2) + size(tCcD_3)),
        .actual_receipt_h = 0,
        .actual_receipt_w = 0,
        .actual_receipt_cells = 0,
        .mma_size = {get<0>(problem_shape), get<1>(problem_shape),
                     get<2>(problem_shape)},
        .mma_tile_size = {get<0>(TileShape{}), get<1>(TileShape{}),
                          get<2>(TileShape{})},
        .target = target,
    };

    int out = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < size(tCcD_0); j++) {
      new_header.thread_rows[out] = static_cast<uint16_t>(get<0>(tCcD_0(j)));
      new_header.thread_cols[out] = static_cast<uint16_t>(get<1>(tCcD_0(j)));
      ++out;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < size(tCcD_1); j++) {
      new_header.thread_rows[out] = static_cast<uint16_t>(get<0>(tCcD_1(j)));
      new_header.thread_cols[out] = static_cast<uint16_t>(get<1>(tCcD_1(j)));
      ++out;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < size(tCcD_2); j++) {
      new_header.thread_rows[out] = static_cast<uint16_t>(get<0>(tCcD_2(j)));
      new_header.thread_cols[out] = static_cast<uint16_t>(get<1>(tCcD_2(j)));
      ++out;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < size(tCcD_3); j++) {
      new_header.thread_rows[out] = static_cast<uint16_t>(get<0>(tCcD_3(j)));
      new_header.thread_cols[out] = static_cast<uint16_t>(get<1>(tCcD_3(j)));
      ++out;
    }

    int const emitted_rows = unique_count(
        new_header.thread_rows, new_header.num_registers_per_thread);
    int const emitted_cols = unique_count(
        new_header.thread_cols, new_header.num_registers_per_thread);
    bool const emits_4x64 = (emitted_rows == 4 && emitted_cols == 64);
    new_header.actual_receipt_h = static_cast<uint16_t>(emitted_rows);
    new_header.actual_receipt_w = static_cast<uint16_t>(emitted_cols);
    new_header.actual_receipt_cells =
        static_cast<uint32_t>(emitted_rows * emitted_cols);

    // P1K-032 canary: publish the actual four-lane coordinate union even when
    // it is not the expected 4x64 rectangle. The host-side gate rejects any
    // non-256-cell emission; publishing the mismatch is more useful than an
    // idle signal that hides the real device geometry.
    if (new_header.block_in_bounds()) {
      *host_signal_header_pinned = new_header;
      host_signal_sync->status = HostSignalStatus::kSignalTriggered;
    }
  }

  __threadfence();
  atomicExch(&host_signal_sync->global_lock, 0);
}

}  // namespace pearl
