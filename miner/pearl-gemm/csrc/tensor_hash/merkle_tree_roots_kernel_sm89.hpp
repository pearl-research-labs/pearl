// SPDX-License-Identifier: see LICENSE
//
// sm_89 port of merkle_tree_roots_kernel.hpp (Hopper).
//
// Substitution map (per SM89_PORT_SPEC.md §2 and the gemm sm_89 ports):
//   SM90_TMA_LOAD              -> SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>
//                                  (zfill predicate handles OOB chunks that
//                                  TMA's natural OOB padding gave us for free)
//   make_tma_copy(...)         -> stash the gmem pointer; partition by thread
//   prefetch_tma_descriptor    -> deleted (no TMA descriptors on sm_89)
//   PipelineTmaAsync<N>        -> cp.async.commit_group / wait_group<N>
//                                  (kept in-line: trivial pipe_read/pipe_write
//                                   indices, no PipelineAsync wrapper needed
//                                   because we have no producer/consumer split)
//   Producer warpgroup (128t)  -> deleted entirely; unified-warp model
//   warp 0 lane 0 TMA issue    -> every thread issues cp.async for its row
//   NamedBarrier::sync         -> __syncthreads()
//   Dual-pipeline mode         -> single pipeline only (cp.async has no 256-
//                                  thread descriptor limit; 512 consumers
//                                  load directly into a single buffer)
//
// Architectural choice: each consumer thread `tid` owns chunk
// `bid*kNumConsumerThreads + tid`. During a load stage the same thread issues
// cp.async to fill its own row `sA(tid, _, stage)` from `gA(tid, _, load_idx)`.
// Predicate `chunk_idx < num_chunks` controls zero-fill for OOB chunks (the
// last-block-doesn't-evenly-divide case that TMA handled implicitly).
//
// Smem layout, blake3 compute, and merkle-reduction code are byte-identical to
// the Hopper kernel; the only divergence is the load mechanism.

#pragma once

#include "blake3/blake3.cuh"
#include "cute/algorithm/copy.hpp"
#include "cute/atom/copy_traits_sm80.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "merkle_tree_utils.hpp"
#include "tensor_hash_constants.cuh"

#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/detail/layout.hpp>
#include <cutlass/gemm/collective/builders/sm90_common.inl>  // for Layout_K_SW*_Atom (ComposedLayout types; arch-agnostic)
#include <type_traits>

namespace pearl {

using namespace cute;

// Warpgroup-cooperative merkle tree kernel for sm_89.
//
// Template parameters (kept identical to the Hopper version for compat):
//   kNumConsumerThreads: Number of threads doing hash computation (128/256/512)
//   kNumStages         : Pipeline depth (2/3/4)
//   kThreadLoadSize    : Bytes loaded per "pipeline slice" (64/128/256/512)
template <int kNumConsumerThreads, int kNumStages, int kThreadLoadSize>
class MerkleTreeRootsKernelSm89 {
 public:
  using Element = uint8_t;
  using ArchTag = cutlass::arch::Sm89;

  static_assert(ArchTag::kMinComputeCapability >= 89);

  static constexpr int kNumWarpThreads = 32;
  static constexpr int kNumThreadsPerWarpGroup = 128;

  static_assert(kNumConsumerThreads >= kNumThreadsPerWarpGroup,
                "Need at least one warpgroup (128 threads)");
  static_assert(kNumConsumerThreads % kNumThreadsPerWarpGroup == 0,
                "Threads must be multiple of warpgroup size (128)");

  // Unified-warp model: kNumThreads == kNumConsumerThreads. No separate
  // producer warpgroup (sm_89 has no setmaxnreg to redistribute registers
  // between roles, so we collapse producer + consumer into one).
  static constexpr int kNumProducerThreads = 0;
  static constexpr int kNumThreads = kNumConsumerThreads;
  static constexpr int kNumWarps = kNumThreads / kNumWarpThreads;
  static constexpr int kNumConsumerWarps =
      kNumConsumerThreads / kNumWarpThreads;
  static constexpr int kNumConsumerWarpgroups =
      kNumConsumerThreads / kNumThreadsPerWarpGroup;

  static constexpr uint32_t MaxThreadsPerBlock = kNumThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  static constexpr int kLoadSize = 16;
  static constexpr int kChunkSize = 1024;
  static constexpr int kWordSize = 4;
  static constexpr int kPipelineStages = kNumStages;

  static_assert(kThreadLoadSize == 64 || kThreadLoadSize == 128 ||
                    kThreadLoadSize == 256 || kThreadLoadSize == 512,
                "kThreadLoadSize must be 64, 128, 256, or 512");
  static_assert(kChunkSize % kThreadLoadSize == 0,
                "kChunkSize must be divisible by kThreadLoadSize");

  static constexpr int kNumBlocksPerChunk =
      kChunkSize / blake3::MSG_BLOCK_SIZE;
  static constexpr int kNumWordsPerBlock =
      blake3::MSG_BLOCK_SIZE / sizeof(uint32_t);

  // Pipeline slice geometry (one cp.async wave per load_idx)
  static constexpr int kNumWordsPerLoad =
      kThreadLoadSize / sizeof(uint32_t);  // 16/32/64/128 uint32 per row
  static constexpr int kNumBlocksPerLoad =
      kThreadLoadSize / blake3::MSG_BLOCK_SIZE;  // 1/2/4/8 blocks per row
  static constexpr int kNumLoads = kChunkSize / kThreadLoadSize;  // 16/8/4/2

  // cp.async vector width: 16 bytes = 4 uint32 per issue.
  static constexpr int kCpAsyncBytes = 16;
  static constexpr int kWordsPerCpAsync = kCpAsyncBytes / sizeof(uint32_t);
  static_assert(kNumWordsPerLoad % kWordsPerCpAsync == 0,
                "kNumWordsPerLoad must be a multiple of 4 (16B cp.async)");
  static constexpr int kCpAsyncPerLoad =
      kNumWordsPerLoad / kWordsPerCpAsync;  // 4/8/16/32

  // Global memory layout for A: [num_chunks, chunk_size_in_words]
  using GmemLayoutTileA = Layout<Shape<int32_t, Int<kChunkSize / kWordSize>>,
                                 Stride<Int<kChunkSize / kWordSize>, Int<1>>>;

  // Shared memory layout: identical to Hopper version (these GMMA Layout_K
  // atoms are ComposedLayout<Swizzle, ...> types — arch-agnostic CuTe code,
  // not tied to any GMMA op).
  using SmemLayoutAtomA =
      std::conditional_t<kThreadLoadSize == 64,
                         GMMA::Layout_K_SW64_Atom<uint32_t>,
                         GMMA::Layout_K_SW128_Atom<uint32_t>>;

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(Int<kNumConsumerThreads>{}, Int<kNumWordsPerLoad>{},
                 Int<kPipelineStages>{})));

  using SmemLayoutAtomLeaves = GMMA::Layout_K_SW128_Atom<uint32_t>;
  using SmemLayoutLeaves = decltype(tile_to_shape(
      SmemLayoutAtomLeaves{},
      Shape<Int<blake3::CHAINING_VALUE_SIZE_U32>, Int<kNumConsumerThreads>>{}));

  static constexpr size_t AlignmentLeaves =
      cutlass::detail::alignment_for_swizzle(SmemLayoutLeaves{});
  static constexpr size_t AlignmentA =
      cutlass::detail::alignment_for_swizzle(SmemLayoutA{});
  static constexpr size_t Alignment = cute::max(AlignmentLeaves, AlignmentA);

  using RmemLayoutChainingValue =
      Layout<Shape<Int<blake3::CHAINING_VALUE_SIZE_U32>>>;
  using RmemLayoutBlock = Layout<Shape<Int<kNumWordsPerBlock>>>;
  using RmemLayoutChunk =
      Layout<Shape<Int<blake3::CHAINING_VALUE_SIZE_U32 * 2>>>;

  struct SharedStorage : cute::aligned_struct<Alignment> {
    cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutLeaves>,
                        AlignmentLeaves>
        smem_leaves;
    cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutA>, AlignmentA>
        smem_a;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  struct Arguments {
    const Element* ptr_data;
    const u32 data_len;
    Element* ptr_roots;
  };

  // Params no longer carries a TMA descriptor — just the gmem pointer.
  // `alignas(128)` retained to mirror the Hopper Params (kernel signature
  // alignment via CUTE_GRID_CONSTANT).
  struct alignas(128) Params {
    const Element* ptr_data;
    u32 data_len;
    Element* ptr_roots;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Params params;
    params.ptr_data = args.ptr_data;
    params.data_len = args.data_len;
    params.ptr_roots = args.ptr_roots;
    return params;
  }

  static dim3 get_grid_shape(Params const& params) {
    const size_t num_chunks =
        (params.data_len + blake3::CHUNK_SIZE - 1) / blake3::CHUNK_SIZE;
    return dim3((num_chunks + kNumConsumerThreads - 1) / kNumConsumerThreads);
  }

  static dim3 get_block_shape() { return dim3(kNumThreads); }

  // Hopper kernel exposes prefetch_tma_descriptors so the host can call it
  // before the launch; on sm_89 it is a no-op (no descriptors exist).
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const&) {}

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    const int tid = threadIdx.x;

    Tensor sLeaves = as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(shared_storage.smem_leaves.data()), SmemLayoutLeaves{}));

    // Position-independent swizzled smem view: lets us index by logical
    // (row, word, stage) coords; the swizzle resolves to the same physical
    // offset whether read by the loader (cp.async dst) or the consumer.
    Tensor sA = as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(shared_storage.smem_a.data()), SmemLayoutA{}));

    const size_t num_chunks =
        (params.data_len + blake3::CHUNK_SIZE - 1) / blake3::CHUNK_SIZE;
    const size_t num_grid_blocks =
        (num_chunks + kNumConsumerThreads - 1) / kNumConsumerThreads;

    Tensor mRoots = make_tensor(
        reinterpret_cast<uint32_t*>(params.ptr_roots),
        make_layout(
            make_shape(Int<blake3::CHAINING_VALUE_SIZE_U32>{}, num_grid_blocks),
            make_stride(Int<1>{}, Int<blake3::CHAINING_VALUE_SIZE_U32>{})));

    // Build a (num_chunks, kChunkSize/kWordSize) tensor over input bytes.
    Tensor mA = make_tensor(
        make_gmem_ptr(reinterpret_cast<uint32_t const*>(params.ptr_data)),
        make_shape(num_chunks, Int<kChunkSize / kWordSize>{}),
        make_stride(Int<kChunkSize / kWordSize>{}, Int<1>{}));

    // Per-thread chunk index. The last block can include OOB indices when
    // num_chunks is not a multiple of kNumConsumerThreads — those threads
    // get zero-filled loads (matching TMA's natural OOB padding semantics).
    const int bid = blockIdx.x;
    const size_t global_chunk_idx =
        static_cast<size_t>(bid) * kNumConsumerThreads + tid;
    const bool chunk_in_bounds = global_chunk_idx < num_chunks;

    // ============================================================
    //   Pipelined cp.async load + BLAKE3 compress
    // ============================================================
    //
    // Lifecycle:
    //   PROLOGUE       : issue first min(kPipelineStages-1, kNumLoads) stages
    //                    so steady-state has at most kPipelineStages-1 in flight
    //   STEADY STATE   : for each load_idx in [0..kNumLoads):
    //                      cp_async_wait<kPipelineStages-2>()
    //                      __syncthreads()
    //                      process kNumBlocksPerLoad blake3 blocks
    //                      issue cp.async for load_idx + (kPipelineStages-1)
    //                      cp_async_fence()
    //   EPILOGUE       : drain remaining stages
    //
    // Note: in this kernel each "pipeline stage" corresponds to a single
    // load_idx slice (kThreadLoadSize bytes per thread), NOT to a full
    // chunk. There are kNumLoads slices total to consume.

    // Register tensors for the chunk's chaining value and block buffer.
    Tensor rChainingValue = make_tensor<uint32_t>(RmemLayoutChainingValue{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      rChainingValue(i) = c_key[i];
    }

    // Partial-chunk handling: only relevant for the very last chunk in the
    // entire problem. `last_chunk_size` is in bytes.
    const u32 remainder = params.data_len % blake3::CHUNK_SIZE;
    const u32 last_chunk_size =
        (remainder == 0) ? blake3::CHUNK_SIZE : remainder;
    const bool is_last_chunk = chunk_in_bounds &&
                               (global_chunk_idx == num_chunks - 1) &&
                               (last_chunk_size < blake3::CHUNK_SIZE);

    // ---- Issue one cp.async stage for slice `load_idx` into smem stage. ----
    //
    // For my own row (= my thread's chunk), copy gA(tid, _, load_idx) into
    // sA(tid, _, stage) using kCpAsyncPerLoad 16-byte cp.async issues.
    // Predicate gates each 16B issue on:
    //   chunk_in_bounds AND (word_byte_offset < last_chunk_size when
    //                        global_chunk_idx == num_chunks-1)
    // OOB → zero-fill (cp.async.cg with zfill).
    //
    // Bit-exact note: we predicate on byte offsets within the chunk, mirroring
    // the Hopper zero_pad_partial_chunk_load logic but applied pre-load
    // instead of post-load. The result in sA is identical (zeros in the
    // OOB byte range, valid data elsewhere).
    using Vec128 = cute::uint128_t;
    auto issue_load = [&](int load_idx, int stage) {
      // Base byte offset within this chunk for slice `load_idx`.
      const u32 load_start_byte_in_chunk = load_idx * kThreadLoadSize;
      // Base gmem element offset (uint32) for this thread's chunk at this slice.
      // Use the gmem coordinate tensor mA to get the logical address.
      // mA(chunk, word) = ptr_data[chunk * (kChunkSize/4) + word].
      //
      // We write into sA(tid, w_in_load, stage) where w_in_load in [0..kNumWordsPerLoad).

      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < kCpAsyncPerLoad; ++v) {
        const int w_in_load = v * kWordsPerCpAsync;
        const u32 word_byte_in_chunk =
            load_start_byte_in_chunk + w_in_load * sizeof(uint32_t);

        // 16B granularity predicate. The cp.async block of 4 uint32s is
        // entirely "in-bounds" only if BOTH:
        //   1. chunk_idx < num_chunks      (OOB chunk → zero-fill all)
        //   2. word_byte_in_chunk + 16 <= last_chunk_size (when last chunk)
        //
        // For non-last chunks (always full 1024B), only condition 1 matters.
        // For the last (partial) chunk we additionally need the 16B block to
        // be entirely within last_chunk_size; partial-16B blocks are handled
        // post-load by zero_pad_partial_chunk_load_post (kept identical to
        // Hopper). To keep behavior bit-exact, we keep cp.async predicates
        // PER-16B and let the partial-16B masking happen after smem fill.
        bool pred = chunk_in_bounds;
        if (is_last_chunk) {
          pred = pred && (word_byte_in_chunk + kCpAsyncBytes <=
                          last_chunk_size);
        }

        // The position-independent swizzled smem tensor `sA` knows how to
        // resolve (tid, w_in_load, stage) → physical smem address that
        // respects the swizzle. We take the address of the first uint32 of
        // the 16B block; cp.async writes 4 contiguous uint32s, which the
        // swizzle's 16B-multiple stride preserves contiguity for (canonical
        // GMMA Layout_K_SW64/SW128 atoms have an inner contiguity of 16B
        // by construction).
        uint32_t* smem_dst_u32 = &sA(tid, w_in_load, stage);
        Vec128* smem_dst = reinterpret_cast<Vec128*>(smem_dst_u32);

        // gmem source address. Use mA's compact stride layout.
        // mA(chunk, word) address = ptr_data + (chunk*(kChunkSize/4) + word)
        // for an _in-bounds_ chunk. For OOB chunks the address doesn't matter
        // because pred==false → zfill.
        const size_t chunk_for_addr =
            chunk_in_bounds ? global_chunk_idx : 0;
        const uint32_t* gmem_src_u32 =
            reinterpret_cast<uint32_t const*>(params.ptr_data) +
            chunk_for_addr * (kChunkSize / sizeof(uint32_t)) +
            (load_start_byte_in_chunk / sizeof(uint32_t)) + w_in_load;
        Vec128 const* gmem_src = reinterpret_cast<Vec128 const*>(gmem_src_u32);

        cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<Vec128>::copy(
            *gmem_src, *smem_dst, pred);
      }
    };

    // ---- Prologue: prefetch up to kPipelineStages-1 stages ----
    constexpr int K_PIPE_MAX = kPipelineStages;
    const int num_prologue_loads =
        cute::min(K_PIPE_MAX - 1, kNumLoads);

    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_pipe = 0; k_pipe < num_prologue_loads; ++k_pipe) {
      issue_load(k_pipe, k_pipe);
      cp_async_fence();
    }
    // Issue remaining "empty" fences so the wait_group counter is consistent.
    // If kNumLoads < K_PIPE_MAX-1, top up with empty groups so the steady-
    // state wait<K_PIPE_MAX-2> doesn't underflow. (Hopper's PipelineTmaAsync
    // handles this internally; we replicate it.)
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_pipe = num_prologue_loads; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
      cp_async_fence();
    }

    // ---- Steady state ----
    int smem_pipe_read = 0;
    int smem_pipe_write = num_prologue_loads % K_PIPE_MAX;

    CUTLASS_PRAGMA_NO_UNROLL
    for (int load_idx = 0; load_idx < kNumLoads; ++load_idx) {
      // Wait until at most K_PIPE_MAX-2 prior cp.async groups are in flight,
      // i.e. the group for slice `load_idx` (smem_pipe_read stage) is done.
      cp_async_wait<K_PIPE_MAX - 2>();
      __syncthreads();

      // Optionally zero-pad partial-16B remnants inside the smem stage that
      // a 16B-granularity zfill predicate couldn't catch. For full chunks
      // (chunk_in_bounds && !is_last_chunk) this is a no-op.
      if (is_last_chunk) {
        zero_pad_partial_chunk_load_post(sA, tid, smem_pipe_read, load_idx,
                                          last_chunk_size);
      }

      // Process kNumBlocksPerLoad blake3 blocks from this slice.
      CUTLASS_PRAGMA_UNROLL
      for (int block_in_load = 0; block_in_load < kNumBlocksPerLoad;
           ++block_in_load) {
        int block_idx =
            load_idx * kNumBlocksPerLoad + block_in_load;
        compress_block(sA, rChainingValue, tid, smem_pipe_read,
                       block_in_load, block_idx);
      }

      // Issue cp.async for the next slice (load_idx + K_PIPE_MAX - 1).
      const int issue_idx = load_idx + (K_PIPE_MAX - 1);
      if (issue_idx < kNumLoads) {
        issue_load(issue_idx, smem_pipe_write);
      }
      cp_async_fence();

      smem_pipe_read = (smem_pipe_read + 1) % K_PIPE_MAX;
      smem_pipe_write = (smem_pipe_write + 1) % K_PIPE_MAX;
    }

    // Drain remaining cp.async groups before we overwrite smem with merkle
    // reduction state.
    cp_async_wait<0>();
    __syncthreads();

    // ---- Store final chunk hash into the sLeaves smem buffer ----
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      sLeaves(i, tid) = rChainingValue(i);
    }

    // Sync before the merkle reduction so all leaves are visible.
    __syncthreads();

    // ============================================================
    //   Merkle tree reduction (identical logic to Hopper kernel)
    // ============================================================
    const bool is_last_block = (bid == static_cast<int>(num_grid_blocks) - 1);

    const u32 num_leaves = [is_last_block, num_chunks,
                            &params]() -> u32 {
      if (!is_last_block) {
        return static_cast<u32>(kNumConsumerThreads);
      }
      const u32 chunks_in_this_block = num_chunks % kNumConsumerThreads;
      const u32 actual_chunks_in_block =
          (chunks_in_this_block == 0) ? static_cast<u32>(kNumConsumerThreads)
                                      : chunks_in_this_block;
      const u32 remainder_bytes = params.data_len % blake3::CHUNK_SIZE;
      const bool last_chunk_too_small =
          (remainder_bytes > 0) && (remainder_bytes < blake3::MSG_BLOCK_SIZE);
      return last_chunk_too_small
                 ? (actual_chunks_in_block > 0 ? actual_chunks_in_block - 1 : 0)
                 : actual_chunks_in_block;
    }();

    if (!is_last_block) {
      merkle_tree_utils::compute_perfect_mt<false>(sLeaves,
                                                    kNumConsumerThreads);
    } else {
      if ((num_leaves & (num_leaves - 1)) == 0) {
        merkle_tree_utils::compute_perfect_mt<false>(sLeaves, num_leaves);
      } else {
        merkle_tree_utils::compute_blake_mt<false>(sLeaves, num_leaves);
      }
    }

    if (tid < blake3::CHAINING_VALUE_SIZE_U32) {
      mRoots(tid, blockIdx.x) = sLeaves(tid, 0);
    }
  }

  // -------- Post-load partial-16B mask for the last (partial) chunk --------
  // The cp.async predicate is 16B granular; bytes inside a 16B block that
  // straddles last_chunk_size need to be cleared post-fill. This mirrors the
  // Hopper kernel's zero_pad_partial_chunk_load but only runs on the
  // straddle-block (the rest is zero-filled by the cp.async zfill predicate).
  template <class SmemTensorA>
  CUTLASS_DEVICE void zero_pad_partial_chunk_load_post(
      SmemTensorA& sA, int consumer_tid, int stage, int load_idx,
      u32 last_chunk_len) {
    const u32 load_start_byte = load_idx * kThreadLoadSize;
    if (load_start_byte >= last_chunk_len) {
      // Entire slice already zero-filled by predicate.
      return;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int w = 0; w < kNumWordsPerLoad; ++w) {
      const u32 word_start_byte = load_start_byte + w * sizeof(uint32_t);
      const u32 word_end_byte = word_start_byte + sizeof(uint32_t);

      if (word_start_byte >= last_chunk_len) {
        sA(consumer_tid, w, stage) = 0;
      } else if (word_end_byte > last_chunk_len) {
        const u32 valid_bytes = last_chunk_len - word_start_byte;
        const u32 mask = (1u << (valid_bytes * 8)) - 1;
        uint32_t val = sA(consumer_tid, w, stage);
        sA(consumer_tid, w, stage) = val & mask;
      }
    }
  }

  template <class SmemTensorA, class RmemTensorChainingValue>
  CUTLASS_DEVICE void compress_block(SmemTensorA const& sA,
                                     RmemTensorChainingValue& rChainingValue,
                                     int consumer_tid, int stage,
                                     int block_in_load, int block_idx) {
    Tensor rBlock = make_tensor<uint32_t>(RmemLayoutBlock{});
    int word_offset = block_in_load * kNumWordsPerBlock;

    // Issue 4-wide vector loads from swizzled smem into the register block.
    // The smem-load address must respect the swizzle; sA is the position-
    // independent swizzled view, so &sA(tid, word_offset + i*4, stage) is the
    // resolved physical address. Adjacent 4 uint32 are contiguous in the K-
    // major Layout_K_SW64/SW128 atom (inner stride 1, 16B chunk size), which
    // is why the original Hopper code does the same uint4 cast.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kNumWordsPerBlock / 4; ++i) {
      uint4 tmp = *reinterpret_cast<const uint4*>(
          &sA(consumer_tid, word_offset + i * 4, stage));
      rBlock(i * 4 + 0) = tmp.x;
      rBlock(i * 4 + 1) = tmp.y;
      rBlock(i * 4 + 2) = tmp.z;
      rBlock(i * 4 + 3) = tmp.w;
    }

    blake3::CompressParams params{
        .counter = blockIdx.x * kNumConsumerThreads + consumer_tid,
        .block_len = blake3::MSG_BLOCK_SIZE,
        .flags = blake3::KEYED_HASH};

    if (block_idx == 0) {
      params.flags |= blake3::CHUNK_START;
    }
    if (block_idx == kNumBlocksPerChunk - 1) {
      params.flags |= blake3::CHUNK_END;
    }

    blake3::compress_msg_block_u32(rBlock, rChainingValue, params);
  }
};

}  // namespace pearl
