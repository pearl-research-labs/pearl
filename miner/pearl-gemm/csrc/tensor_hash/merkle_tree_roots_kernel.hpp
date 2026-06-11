#include "blake3/blake3.cuh"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "merkle_tree_utils.hpp"
#include "tensor_hash_constants.cuh"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/memory.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/detail/layout.hpp>
#include <cutlass/gemm/collective/builders/sm90_common.inl>  // ss_smem_selector
#include <cutlass/pipeline/pipeline.hpp>
#include <type_traits>

namespace pearl {

using namespace cute;

// Named barriers for warpgroup synchronization (matching GEMM kernel pattern)
// Using barrier ID 0 which is available in single-CTA kernels
enum class TensorHashBarriers : uint32_t {
  PrimaryConsumers =
      0,  // Barrier for primary consumer group (or all consumers in single-pipeline mode)
  SecondaryConsumers =
      1  // Barrier for secondary consumer group in dual-pipeline mode
};

// Warpgroup-specialized merkle tree kernel (matching GEMM kernel pattern):
// - Warpgroup 0 (128 threads): Producer warpgroup
//   - Only warp 0 within warpgroup 0 issues TMA loads
//   - Warps 1-3 in warpgroup 0 are idle during load phase
// - Warpgroups 1+: Consumer warpgroups (do hash computation)
//
// SM90 specialization: Producer warpgroup (128 threads) + Consumer warpgroups
// Template parameters:
//   kNumConsumerThreads: Number of consumer threads (must be multiple of 128)
//   kNumStages: Number of pipeline stages (typically 2-4)
//   kThreadLoadSize: Bytes loaded per TMA operation (64, 128, 256, 512)
//   kApplyRoot: Whether the kernel applies BLAKE3's ROOT flag itself (set by the
//               host only when the whole input fits in a single CTA; otherwise
//               ROOT is applied downstream by ComputeBlakeMTKernel/ReduceRootsKernel).
//
// When kNumConsumerThreads > 256, the kernel uses dual pipelines:
//   - TMA descriptor dimensions are limited to 256, so we split into two groups
//   - Each group of 256 consumers has its own TMA region and pipeline
//   - Producer warpgroup issues TMA loads to both regions
//   - Consumer groups operate independently on their respective pipelines
template <int kNumConsumerThreads, int kNumStages, int kThreadLoadSize,
          bool kApplyRoot>
class MerkleTreeRootsKernel {
 public:
  using Element = uint8_t;
  using ArchTag = cutlass::arch::Sm90;

  static_assert(ArchTag::kMinComputeCapability >= 90);

  static constexpr int kNumWarpThreads = 32;
  static constexpr int kNumThreadsPerWarpGroup = 128;  // 4 warps per warpgroup

  // Consumer threads must be at least one warpgroup for proper synchronization
  static_assert(kNumConsumerThreads >= kNumThreadsPerWarpGroup,
                "Need at least one consumer warpgroup (128 threads)");
  static_assert(kNumConsumerThreads % kNumThreadsPerWarpGroup == 0,
                "Consumer threads must be multiple of warpgroup size (128)");

  // TMA dimension limit - each TMA descriptor dimension is limited to 256
  static constexpr int kMaxTmaThreads = 256;

  // Dual pipeline mode: when consumer threads > 256, we split into two groups
  static constexpr bool kUseDualPipelines =
      (kNumConsumerThreads > kMaxTmaThreads);
  static constexpr int kNumConsumerGroups = kUseDualPipelines ? 2 : 1;
  static constexpr int kTmaThreads =
      kUseDualPipelines ? kMaxTmaThreads : kNumConsumerThreads;
  static constexpr int kConsumersPerGroup =
      kNumConsumerThreads / kNumConsumerGroups;

  static_assert(!kUseDualPipelines || kNumConsumerThreads == 512,
                "Dual pipeline mode currently only supports exactly 512 "
                "consumer threads");

  // Producer warpgroup (128 threads) - only warp 0 does actual TMA work
  static constexpr int kNumProducerThreads = kNumThreadsPerWarpGroup;

  // Total threads = producer warpgroup + consumer warpgroups
  static constexpr int kNumThreads = kNumProducerThreads + kNumConsumerThreads;
  static constexpr int kNumWarps = kNumThreads / kNumWarpThreads;
  static constexpr int kNumConsumerWarps =
      kNumConsumerThreads / kNumWarpThreads;
  static constexpr int kNumConsumerWarpgroups =
      kNumConsumerThreads / kNumThreadsPerWarpGroup;

  static constexpr uint32_t MaxThreadsPerBlock = kNumThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  // Parameters exposed via template
  static constexpr int kLoadSize = 16;
  static constexpr int kChunkSize = 1024;
  static constexpr int kWordSize = 4;
  static constexpr int kPipelineStages = kNumStages;
  // kThreadLoadSize is already a template parameter

  // Validate thread load size
  static_assert(kThreadLoadSize == 64 || kThreadLoadSize == 128 ||
                    kThreadLoadSize == 256 || kThreadLoadSize == 512,
                "kThreadLoadSize must be 64, 128, 256, or 512");
  static_assert(kChunkSize % kThreadLoadSize == 0,
                "kChunkSize must be divisible by kThreadLoadSize");

  static constexpr int kNumBlocksPerChunk =
      kChunkSize / blake3::MSG_BLOCK_SIZE;  // 16 blocks of 64 bytes each
  static constexpr int kNumWordsPerBlock =
      blake3::MSG_BLOCK_SIZE / sizeof(uint32_t);  // 16 words per block

  // TMA load parameters
  static constexpr int kNumWordsPerLoad =
      kThreadLoadSize / sizeof(uint32_t);  // words per load
  static constexpr int kNumBlocksPerLoad =
      kThreadLoadSize / blake3::MSG_BLOCK_SIZE;  // blocks per load
  static constexpr int kNumLoads =
      kChunkSize / kThreadLoadSize;  // loads to complete a chunk

  // global memory layout for A: [num_chunks][chunk_size_in_words]
  // For dual pipeline: each TMA copy handles kTmaThreads (256) consecutive chunks
  using GmemLayoutTileA = Layout<Shape<int32_t, Int<kChunkSize / kWordSize>>,
                                 Stride<Int<kChunkSize / kWordSize>, Int<1>>>;

  // shared memory layout for A: [threads_per_group][words_per_load][pipeline_stages]
  // For dual pipelines, each group has kTmaThreads (256) threads
  // Use ss_smem_selector to automatically pick appropriate swizzle based on element type and dimensions
  // This reduces bank conflicts in shared memory for better throughput
  // Use SW64 swizzle for 64-byte loads, SW128 for larger loads (128, 256, 512)
  using SmemLayoutAtomA =
      std::conditional_t<kThreadLoadSize == 64,
                         GMMA::Layout_K_SW64_Atom<uint32_t>,
                         GMMA::Layout_K_SW128_Atom<uint32_t>>;

  // Per-group shared memory layout (used for TMA descriptor)
  using SmemLayoutA_PerGroup = decltype(tile_to_shape(
      SmemLayoutAtomA{}, make_shape(Int<kTmaThreads>{}, Int<kNumWordsPerLoad>{},
                                    Int<kPipelineStages>{})));

  // Full shared memory layout for all consumers (only used in single pipeline mode)
  using SmemLayoutA = std::conditional_t<
      kUseDualPipelines,
      SmemLayoutA_PerGroup,  // In dual mode, we have two separate arrays
      decltype(tile_to_shape(
          SmemLayoutAtomA{},
          make_shape(Int<kNumConsumerThreads>{}, Int<kNumWordsPerLoad>{},
                     Int<kPipelineStages>{})))>;

  // shared memory layout for leaves: [chaining_value_size][num_consumer_threads]
  // Use ss_smem_selector to automatically pick appropriate swizzle for bank conflict reduction
  using SmemLayoutAtomLeaves = GMMA::Layout_K_SW128_Atom<uint32_t>;
  using SmemLayoutLeaves = decltype(tile_to_shape(
      SmemLayoutAtomLeaves{},
      Shape<Int<blake3::CHAINING_VALUE_SIZE_U32>, Int<kNumConsumerThreads>>{}));

  // TMA descriptor - handles kTmaThreads (256) threads per copy
  using TMA_A = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<uint32_t const*>(nullptr)),
                  GmemLayoutTileA{}),
      take<0, 2>(SmemLayoutA_PerGroup{}),
      make_shape(Int<kTmaThreads>{}, Int<kNumWordsPerLoad>{}), _1{}));

  // TMA transaction bytes for pipeline barrier (per group)
  static constexpr uint32_t TmaTransactionBytesA = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutA_PerGroup{})) * sizeof(uint32_t));

  // Pipeline configuration
  using Pipeline = cutlass::PipelineTmaAsync<kPipelineStages>;
  using PipelineState = typename Pipeline::PipelineState;
  using PipelineBarrierType = typename Pipeline::ProducerBarrierType;

  static constexpr size_t AlignmentLeaves =
      cutlass::detail::alignment_for_swizzle(SmemLayoutLeaves{});
  static constexpr size_t AlignmentA =
      cutlass::detail::alignment_for_swizzle(SmemLayoutA_PerGroup{});
  static constexpr size_t Alignment = cute::max(AlignmentLeaves, AlignmentA);

  using RmemLayoutChainingValue =
      Layout<Shape<Int<blake3::CHAINING_VALUE_SIZE_U32>>>;
  using RmemLayoutBlock = Layout<Shape<Int<kNumWordsPerBlock>>>;
  using RmemLayoutChunk =
      Layout<Shape<Int<blake3::CHAINING_VALUE_SIZE_U32 * 2>>>;

  // SharedStorage for single pipeline mode
  struct SharedStorageSingle : cute::aligned_struct<Alignment> {
    cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutLeaves>,
                        AlignmentLeaves>
        smem_leaves;
    cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutA>, AlignmentA>
        smem_a;
    typename Pipeline::SharedStorage pipeline_storage;
  };

  // SharedStorage for dual pipeline mode
  struct SharedStorageDual : cute::aligned_struct<Alignment> {
    cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutLeaves>,
                        AlignmentLeaves>
        smem_leaves;
    // Two separate shared memory arrays for the two consumer groups
    cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutA_PerGroup>,
                        AlignmentA>
        smem_a_0;
    cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutA_PerGroup>,
                        AlignmentA>
        smem_a_1;
    // Two separate pipelines
    typename Pipeline::SharedStorage pipeline_storage_0;
    typename Pipeline::SharedStorage pipeline_storage_1;
  };

  using SharedStorage = std::conditional_t<kUseDualPipelines, SharedStorageDual,
                                           SharedStorageSingle>;

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  struct Arguments {
    const Element* ptr_data;
    const u32 data_len;  // Data length in bytes
    Element* ptr_roots;
  };

  struct alignas(128) Params {
    const Element* ptr_data;
    u32 data_len;  // Data length in bytes
    Element* ptr_roots;
    TMA_A tma_load_A;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Params params;
    params.ptr_data = args.ptr_data;
    params.data_len = args.data_len;
    params.ptr_roots = args.ptr_roots;

    // Use ceiling division to include partial chunks in TMA descriptor
    // TMA will read the full chunk size, and we'll zero-pad OOB data in the kernel
    const size_t num_chunks = (args.data_len + kChunkSize - 1) / kChunkSize;

    Tensor mA = make_tensor(
        make_gmem_ptr(reinterpret_cast<uint32_t const*>(args.ptr_data)),
        make_shape(num_chunks, Int<kChunkSize / kWordSize>{}),
        make_stride(Int<kChunkSize / kWordSize>{}, Int<1>{}));
    params.tma_load_A = make_tma_copy(
        cute::SM90_TMA_LOAD{}, mA, SmemLayoutA_PerGroup{}(_, _, _0{}),
        make_shape(Int<kTmaThreads>{}, Int<kNumWordsPerLoad>{}), _1{});

    return params;
  }

  static dim3 get_grid_shape(Params const& params) {
    // Each block processes kNumConsumerThreads chunks
    // Use ceiling division to include partial chunks
    const size_t num_chunks =
        (params.data_len + blake3::CHUNK_SIZE - 1) / blake3::CHUNK_SIZE;

    return dim3((num_chunks + kNumConsumerThreads - 1) / kNumConsumerThreads);
  }

  static dim3 get_block_shape() { return dim3(kNumThreads); }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_A.get_tma_descriptor());
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    const int tid = threadIdx.x;
    int const warp_idx = cutlass::canonical_warp_idx_sync();
    int const lane_predicate = cute::elect_one_sync();

    // Warpgroup-level separation (matching GEMM kernel pattern)
    int const warp_group_idx = cutlass::canonical_warp_group_idx();
    int const warp_group_thread_idx = tid % kNumThreadsPerWarpGroup;

    // Warpgroup 0 is producer, rest are consumers
    bool const is_producer_warpgroup = (warp_group_idx == 0);

    // Prefetch TMA descriptors from warp 0, lane 0
    if (warp_idx == 0 && lane_predicate) {
      prefetch_tma_descriptors(params);
    }

    // Create smem tensor for leaves (common to both modes)
    Tensor sLeaves = as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(shared_storage.smem_leaves.data()), SmemLayoutLeaves{}));

    // Calculate number of blocks for output tensor
    // Use ceiling division to include partial chunks
    const size_t num_chunks =
        (params.data_len + blake3::CHUNK_SIZE - 1) / blake3::CHUNK_SIZE;
    const size_t num_grid_blocks =
        (num_chunks + kNumConsumerThreads - 1) / kNumConsumerThreads;

    // kApplyRoot (compile-time) is set by the host only when this is the single
    // CTA of the whole input, so the kernel must apply BLAKE3's ROOT
    // finalization (otherwise ROOT is applied later by ComputeBlakeMTKernel /
    // ReduceRootsKernel). When the entire message is a single chunk, BLAKE3 sets
    // ROOT on that chunk's last block (no Merkle parent compression happens).
    const bool is_single_chunk = (num_chunks == 1);

    // Output tensor - use stride (1, CHAINING_VALUE_SIZE_U32) for coalesced writes
    // Each block writes 8 consecutive uint32_t values, blocks are spaced 8 words apart
    Tensor mRoots = make_tensor(
        reinterpret_cast<uint32_t*>(params.ptr_roots),
        make_layout(
            make_shape(Int<blake3::CHAINING_VALUE_SIZE_U32>{}, num_grid_blocks),
            make_stride(Int<1>{}, Int<blake3::CHAINING_VALUE_SIZE_U32>{})));

    if constexpr (kUseDualPipelines) {
      // ============ DUAL PIPELINE MODE (512 consumers = 2 groups of 256) ============
      // Consumer thread index (0 to kNumConsumerThreads-1)
      const int consumer_tid = tid - kNumProducerThreads;
      // Which consumer group does this thread belong to? (0 or 1)
      const int consumer_group =
          is_producer_warpgroup ? 0 : (consumer_tid / kConsumersPerGroup);
      // Thread index within the consumer group (0 to 255)
      const int tid_in_group = consumer_tid % kConsumersPerGroup;

      // Create smem tensors for both groups
      Tensor sA_0 = as_position_independent_swizzle_tensor(
          make_tensor(make_smem_ptr(shared_storage.smem_a_0.data()),
                      SmemLayoutA_PerGroup{}));
      Tensor sA_1 = as_position_independent_swizzle_tensor(
          make_tensor(make_smem_ptr(shared_storage.smem_a_1.data()),
                      SmemLayoutA_PerGroup{}));

      // Initialize pipelines
      // For producers: participate in both pipelines
      // For consumers: participate only in their group's pipeline
      typename Pipeline::Params pipeline_params_0;
      pipeline_params_0.transaction_bytes = TmaTransactionBytesA;
      pipeline_params_0.role =
          is_producer_warpgroup
              ? Pipeline::ThreadCategory::Producer
              : (consumer_group == 0
                     ? Pipeline::ThreadCategory::Consumer
                     : Pipeline::ThreadCategory::NonParticipant);
      pipeline_params_0.is_leader =
          is_producer_warpgroup
              ? (warp_group_thread_idx == 0)  // Producer leader
              : (consumer_group == 0 &&
                 tid_in_group == 0);  // Consumer group 0 leader
      pipeline_params_0.num_consumers = kConsumersPerGroup;

      typename Pipeline::Params pipeline_params_1;
      pipeline_params_1.transaction_bytes = TmaTransactionBytesA;
      pipeline_params_1.role =
          is_producer_warpgroup
              ? Pipeline::ThreadCategory::Producer
              : (consumer_group == 1
                     ? Pipeline::ThreadCategory::Consumer
                     : Pipeline::ThreadCategory::NonParticipant);
      pipeline_params_1.is_leader =
          is_producer_warpgroup
              ? (warp_group_thread_idx == 0)  // Producer leader
              : (consumer_group == 1 &&
                 tid_in_group == 0);  // Consumer group 1 leader
      pipeline_params_1.num_consumers = kConsumersPerGroup;

      Pipeline pipeline_0(shared_storage.pipeline_storage_0, pipeline_params_0,
                          Shape<_1, _1, _1>{});
      Pipeline pipeline_1(shared_storage.pipeline_storage_1, pipeline_params_1,
                          Shape<_1, _1, _1>{});

      // Ensure pipeline initialization is visible to all threads
      __syncthreads();

      if (is_producer_warpgroup) {
        // ============ PRODUCER WARPGROUP (128 threads, warps 0-3) ============
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (tid / 32) % 4, 0);

        if (warp_idx_in_warpgroup == 0) {
          PipelineState smem_pipe_write_0 =
              cutlass::make_producer_start_state<Pipeline>();
          PipelineState smem_pipe_write_1 =
              cutlass::make_producer_start_state<Pipeline>();

          // Producer warp does TMA loading to both groups
          producer_loop_dual(params, pipeline_0, pipeline_1, sA_0, sA_1,
                             shared_storage, smem_pipe_write_0,
                             smem_pipe_write_1);

          // Producer warp participates in load_tail for both pipelines
          load_tail_dual(pipeline_0, pipeline_1, smem_pipe_write_0,
                         smem_pipe_write_1);
        }
      } else if (consumer_group == 0) {
        if (is_single_chunk) {
          consumer_loop_dual<kApplyRoot>(params, pipeline_0, sA_0, sLeaves,
                                         tid_in_group, consumer_tid, 0);
        } else {
          consumer_loop_dual<false>(params, pipeline_0, sA_0, sLeaves,
                                    tid_in_group, consumer_tid, 0);
        }
      } else {
        if (is_single_chunk) {
          consumer_loop_dual<kApplyRoot>(params, pipeline_1, sA_1, sLeaves,
                                         tid_in_group, consumer_tid, 1);
        } else {
          consumer_loop_dual<false>(params, pipeline_1, sA_1, sLeaves,
                                    tid_in_group, consumer_tid, 1);
        }
      }
    } else /* !kUseDualPipelines */ {
      // ============ SINGLE PIPELINE MODE (up to 256 consumers) ============
      // Initialize pipeline with warpgroup-specialized roles
      // ONE leader per warpgroup (thread 0 of each warpgroup)
      typename Pipeline::Params pipeline_params;
      pipeline_params.transaction_bytes = TmaTransactionBytesA;
      pipeline_params.role = is_producer_warpgroup
                                 ? Pipeline::ThreadCategory::Producer
                                 : Pipeline::ThreadCategory::Consumer;
      pipeline_params.is_leader =
          (warp_group_thread_idx == 0);  // First thread of each warpgroup
      pipeline_params.num_consumers = kNumConsumerThreads;

      Pipeline pipeline(shared_storage.pipeline_storage, pipeline_params,
                        Shape<_1, _1, _1>{});

      // Ensure pipeline initialization is visible to all threads
      __syncthreads();

      // Create smem tensor for A
      Tensor sA = as_position_independent_swizzle_tensor(make_tensor(
          make_smem_ptr(shared_storage.smem_a.data()), SmemLayoutA{}));

      if (is_producer_warpgroup) {
        // ============ PRODUCER WARPGROUP (128 threads, warps 0-3) ============
        // Only warp 0 within producer warpgroup does actual TMA loading
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (tid / 32) % 4, 0);

        if (warp_idx_in_warpgroup == 0) {
          PipelineState smem_pipe_write =
              cutlass::make_producer_start_state<Pipeline>();

          // Producer warp does TMA loading
          producer_loop(params, pipeline, sA, shared_storage, smem_pipe_write);

          // Producer warp participates in load_tail
          load_tail(pipeline, smem_pipe_write);
        }
        // Warps 1-3 in producer warpgroup are idle during TMA phase
      } else {
        // ============ CONSUMER WARPGROUPS (128 threads each) ============
        // Consumer thread index (0 to kNumConsumerThreads-1)
        const int consumer_tid = tid - kNumProducerThreads;
        if (is_single_chunk) {
          consumer_loop<kApplyRoot>(params, pipeline, sA, sLeaves,
                                    consumer_tid);
        } else {
          consumer_loop<false>(params, pipeline, sA, sLeaves, consumer_tid);
        }
        // Note: consumer_loop syncs all consumer warpgroups after each pipeline stage
      }
    }

    // Sync all threads (producer + consumer) before merkle tree reduction
    __syncthreads();

    // Calculate number of leaves (chunks) in this block
    const size_t bid = blockIdx.x;
    const bool is_last_block = (bid == num_grid_blocks - 1);

    // Determine actual number of leaves in this block.
    // The Python reference zero-pads any non-empty trailing data up to a full
    // 1024-byte chunk, so every chunk counted by the ceiling division (including
    // a partial last chunk of any size, even < 64 bytes) contributes one leaf.
    const u32 num_leaves = [is_last_block, num_chunks]() -> u32 {
      if (!is_last_block) {
        return static_cast<u32>(kNumConsumerThreads);
      }
      // num_chunks already includes partial chunks (ceiling division)
      const u32 chunks_in_this_block = num_chunks % kNumConsumerThreads;
      return (chunks_in_this_block == 0) ? static_cast<u32>(kNumConsumerThreads)
                                         : chunks_in_this_block;
    }();

    // Reduce into a Merkle Tree (all threads participate for __syncthreads)
    // Choose algorithm based on whether num_leaves is a power of 2
    if (!is_last_block) {
      // Non-last blocks always have power-of-2 leaves (kNumConsumerThreads is power of 2)
      merkle_tree_utils::compute_perfect_mt<false>(sLeaves,
                                                   kNumConsumerThreads);
    } else if ((num_leaves & (num_leaves - 1)) == 0) {
      // Last block, power of 2: use perfect merkle tree.
      // If this is the only block, the final parent compression must apply ROOT.
      merkle_tree_utils::compute_perfect_mt<kApplyRoot>(sLeaves, num_leaves);
    } else {
      // Last block, not a power of 2: use BLAKE3's merkle tree structure.
      merkle_tree_utils::compute_blake_mt<kApplyRoot>(sLeaves, num_leaves);
    }

    // Copy the root to the output (use first 8 threads)
    if (tid < blake3::CHAINING_VALUE_SIZE_U32) {
      mRoots(tid, blockIdx.x) = sLeaves(tid, 0);
    }
  }

  template <class SmemTensorA>
  CUTLASS_DEVICE void producer_loop(Params const& params, Pipeline& pipeline,
                                    SmemTensorA& sA,
                                    SharedStorage& shared_storage,
                                    PipelineState& smem_pipe_write) {
    const int bid = blockIdx.x;
    // Use ceiling division to match TMA descriptor (includes partial chunks)
    const int num_chunks =
        (params.data_len + blake3::CHUNK_SIZE - 1) / blake3::CHUNK_SIZE;

    // View of our CTA's tile of A
    Tensor mA = params.tma_load_A.get_tma_tensor(
        make_shape(num_chunks, Int<kChunkSize / kWordSize>{}));

    // Get a view of our current tile, partitioned by load (kThreadLoadSize bytes each)
    // Note: kTmaThreads == kNumConsumerThreads in single pipeline mode
    Tensor gA =
        local_tile(mA, make_shape(Int<kTmaThreads>{}, Int<kNumWordsPerLoad>{}),
                   make_coord(bid, _));

    // Partition for TMA (done outside lane_predicate, like in collective_mainloop.hpp)
    auto [tAgA, tAsA] =
        tma_partition(params.tma_load_A, Int<0>{}, Layout<_1>{},
                      group_modes<0, 2>(sA), group_modes<0, 2>(gA));

    // DO TMA LOAD from a single thread (matching collective_mainloop.hpp pattern)
    int lane_predicate = cute::elect_one_sync();
    if (lane_predicate) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (int load_idx = 0; load_idx < kNumLoads; ++load_idx) {
        pipeline.producer_acquire(smem_pipe_write);

        PipelineBarrierType* tma_barrier =
            pipeline.producer_get_barrier(smem_pipe_write);
        auto stage = smem_pipe_write.index();

        // Issue TMA copy with barrier
        copy(params.tma_load_A.with(*tma_barrier, 0 /*mcast_mask*/),
             tAgA(_, load_idx), tAsA(_, stage));

        pipeline.producer_commit(smem_pipe_write, TmaTransactionBytesA);
        ++smem_pipe_write;
      }
    }
  }

  CUTLASS_DEVICE
  void load_tail(Pipeline& pipeline, PipelineState& smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();
    // Issue the epilogue waits from the producer warp leader (matching GEMM pattern)
    if (lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
       * then would just be acquired since the phase was still inverted from make_producer_start_state
       */
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  // ==================== ZERO-PADDING HELPER ====================

  /// Zero-pad partial chunks in shared memory (per-load iteration)
  /// Each TMA load brings in kThreadLoadSize bytes into one stage
  /// We need to zero the out-of-bounds parts within that loaded data
  template <class SmemTensorA>
  CUTLASS_DEVICE void zero_pad_partial_chunk_load(SmemTensorA& sA,
                                                  int consumer_tid, int stage,
                                                  int load_idx,
                                                  u32 last_chunk_len) {
    // This load brought in bytes [load_start_byte, load_start_byte + kThreadLoadSize)
    // from the chunk into sA(consumer_tid, 0..kNumWordsPerLoad-1, stage)
    const u32 load_start_byte = load_idx * kThreadLoadSize;

    // If this entire load is beyond valid data, zero everything
    if (load_start_byte >= last_chunk_len) {
      CUTLASS_PRAGMA_UNROLL
      for (int w = 0; w < kNumWordsPerLoad; ++w) {
        sA(consumer_tid, w, stage) = 0;
      }
      return;
    }

    // Some or all of this load contains valid data
    // Process each word in the load
    CUTLASS_PRAGMA_UNROLL
    for (int w = 0; w < kNumWordsPerLoad; ++w) {
      const u32 word_start_byte = load_start_byte + w * sizeof(uint32_t);
      const u32 word_end_byte = word_start_byte + sizeof(uint32_t);

      if (word_start_byte >= last_chunk_len) {
        // Entire word is OOB - zero it
        sA(consumer_tid, w, stage) = 0;
      } else if (word_end_byte > last_chunk_len) {
        // Partial word - some bytes valid, some OOB
        const u32 valid_bytes = last_chunk_len - word_start_byte;
        // Mask: valid_bytes=1 -> 0xFF, =2 -> 0xFFFF, =3 -> 0xFFFFFF
        const u32 mask = (1u << (valid_bytes * 8)) - 1;

        // Read, mask, write back
        uint32_t val = sA(consumer_tid, w, stage);
        sA(consumer_tid, w, stage) = val & mask;
      }
      // else: word is fully valid, no action needed
    }
  }

  // ==================== DUAL PIPELINE MODE FUNCTIONS ====================

  template <class SmemTensorA_0, class SmemTensorA_1>
  CUTLASS_DEVICE void producer_loop_dual(
      Params const& params, Pipeline& pipeline_0, Pipeline& pipeline_1,
      SmemTensorA_0& sA_0, SmemTensorA_1& sA_1, SharedStorage& shared_storage,
      PipelineState& smem_pipe_write_0, PipelineState& smem_pipe_write_1) {
    const int bid = blockIdx.x;
    // Use ceiling division to match TMA descriptor (includes partial chunks)
    const int num_chunks =
        (params.data_len + blake3::CHUNK_SIZE - 1) / blake3::CHUNK_SIZE;

    // View of our CTA's tile of A (global memory)
    Tensor mA = params.tma_load_A.get_tma_tensor(
        make_shape(num_chunks, Int<kChunkSize / kWordSize>{}));

    // For dual pipeline, we load kTmaThreads (256) chunks per TMA copy
    // Group 0: chunks [bid * kNumConsumerThreads, bid * kNumConsumerThreads + kTmaThreads)
    // Group 1: chunks [bid * kNumConsumerThreads + kTmaThreads, bid * kNumConsumerThreads + 2*kTmaThreads)

    // Get views for each group
    Tensor gA_0 =
        local_tile(mA, make_shape(Int<kTmaThreads>{}, Int<kNumWordsPerLoad>{}),
                   make_coord(bid * kNumConsumerGroups + 0, _));
    Tensor gA_1 =
        local_tile(mA, make_shape(Int<kTmaThreads>{}, Int<kNumWordsPerLoad>{}),
                   make_coord(bid * kNumConsumerGroups + 1, _));

    // Partition for TMA (done outside lane_predicate, like in collective_mainloop.hpp)
    auto [tAgA_0, tAsA_0] =
        tma_partition(params.tma_load_A, Int<0>{}, Layout<_1>{},
                      group_modes<0, 2>(sA_0), group_modes<0, 2>(gA_0));
    auto [tAgA_1, tAsA_1] =
        tma_partition(params.tma_load_A, Int<0>{}, Layout<_1>{},
                      group_modes<0, 2>(sA_1), group_modes<0, 2>(gA_1));

    // DO TMA LOAD from a single thread (matching collective_mainloop.hpp pattern)
    int lane_predicate = cute::elect_one_sync();
    if (lane_predicate) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (int load_idx = 0; load_idx < kNumLoads; ++load_idx) {
        // Acquire both pipelines
        pipeline_0.producer_acquire(smem_pipe_write_0);
        pipeline_1.producer_acquire(smem_pipe_write_1);

        // Get barriers for both pipelines
        PipelineBarrierType* tma_barrier_0 =
            pipeline_0.producer_get_barrier(smem_pipe_write_0);
        PipelineBarrierType* tma_barrier_1 =
            pipeline_1.producer_get_barrier(smem_pipe_write_1);
        auto stage_0 = smem_pipe_write_0.index();
        auto stage_1 = smem_pipe_write_1.index();

        // Issue TMA copy to group 0
        copy(params.tma_load_A.with(*tma_barrier_0, 0 /*mcast_mask*/),
             tAgA_0(_, load_idx), tAsA_0(_, stage_0));

        // Issue TMA copy to group 1
        copy(params.tma_load_A.with(*tma_barrier_1, 0 /*mcast_mask*/),
             tAgA_1(_, load_idx), tAsA_1(_, stage_1));

        // Commit both pipelines
        pipeline_0.producer_commit(smem_pipe_write_0, TmaTransactionBytesA);
        pipeline_1.producer_commit(smem_pipe_write_1, TmaTransactionBytesA);
        ++smem_pipe_write_0;
        ++smem_pipe_write_1;
      }
    }
  }

  CUTLASS_DEVICE
  void load_tail_dual(Pipeline& pipeline_0, Pipeline& pipeline_1,
                      PipelineState& smem_pipe_write_0,
                      PipelineState& smem_pipe_write_1) {
    int lane_predicate = cute::elect_one_sync();
    if (lane_predicate) {
      pipeline_0.producer_tail(smem_pipe_write_0);
      pipeline_1.producer_tail(smem_pipe_write_1);
    }
  }

  template <bool IsSingleChunk, class SmemTensorA, class SmemTensorLeaves>
  CUTLASS_DEVICE void consumer_loop_dual(Params const& params,
                                         Pipeline& pipeline, SmemTensorA& sA,
                                         SmemTensorLeaves const& sLeaves,
                                         int tid_in_group, int consumer_tid,
                                         int group_idx) {

    PipelineState smem_pipe_read;

    // Register tensor (chaining value) of our hash's state
    Tensor rChainingValue = make_tensor<uint32_t>(RmemLayoutChainingValue{});

    // Initialize chaining value with c_key
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      rChainingValue(i) = c_key[i];
    }

    // Calculate if this thread is processing the last (potentially partial) chunk
    // Use ceiling division to include partial chunks in the count
    const size_t num_chunks =
        (params.data_len + blake3::CHUNK_SIZE - 1) / blake3::CHUNK_SIZE;
    const u32 remainder = params.data_len % blake3::CHUNK_SIZE;
    const u32 last_chunk_size =
        (remainder == 0) ? blake3::CHUNK_SIZE : remainder;
    const u32 global_chunk_idx =
        blockIdx.x * kNumConsumerThreads + consumer_tid;
    const bool is_last_chunk = (global_chunk_idx == num_chunks - 1) &&
                               (last_chunk_size < blake3::CHUNK_SIZE);

    CUTLASS_PRAGMA_NO_UNROLL
    for (int load_idx = 0; load_idx < kNumLoads; ++load_idx) {
      // Wait for TMA load to complete
      pipeline.consumer_wait(smem_pipe_read);
      auto stage = smem_pipe_read.index();

      // Zero-pad OOB data in this load if processing the last (partial) chunk
      if (is_last_chunk) {
        zero_pad_partial_chunk_load(sA, tid_in_group, stage, load_idx,
                                    last_chunk_size);
      }

      // Process kNumBlocksPerLoad blocks from this load
      CUTLASS_PRAGMA_UNROLL
      for (int block_in_load = 0; block_in_load < kNumBlocksPerLoad;
           ++block_in_load) {
        int block_idx = load_idx * kNumBlocksPerLoad +
                        block_in_load;  // Global block index (0-15)
        // Use tid_in_group for shared memory access (0-255 within the group's smem region)
        // Use consumer_tid for counter calculation (global index 0-511)
        compress_block_dual<IsSingleChunk>(sA, rChainingValue, tid_in_group,
                                           consumer_tid, stage, block_in_load,
                                           block_idx);
      }

      // Sync all consumers within this group after processing this stage
      // Use different barrier IDs for each group
      cutlass::arch::NamedBarrier::sync(
          kConsumersPerGroup,
          static_cast<uint32_t>(group_idx == 0
                                    ? TensorHashBarriers::PrimaryConsumers
                                    : TensorHashBarriers::SecondaryConsumers));

      // Release the pipeline stage
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
    }

    // Store final hash result to sLeaves using consumer_tid (global index 0-511)
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      sLeaves(i, consumer_tid) = rChainingValue(i);
    }
  }

  // ==================== SINGLE PIPELINE MODE FUNCTIONS ====================

  template <bool IsSingleChunk, class SmemTensorA, class SmemTensorLeaves>
  CUTLASS_DEVICE void consumer_loop(Params const& params, Pipeline& pipeline,
                                    SmemTensorA& sA,
                                    SmemTensorLeaves const& sLeaves,
                                    int consumer_tid) {

    PipelineState smem_pipe_read;

    // Register tensor (chaining value) of our hash's state
    Tensor rChainingValue = make_tensor<uint32_t>(RmemLayoutChainingValue{});

    // Initialize chaining value with c_key
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      rChainingValue(i) = c_key[i];
    }

    // Calculate if this thread is processing the last (potentially partial) chunk
    // Use ceiling division to include partial chunks in the count
    const size_t num_chunks =
        (params.data_len + blake3::CHUNK_SIZE - 1) / blake3::CHUNK_SIZE;
    const u32 remainder = params.data_len % blake3::CHUNK_SIZE;
    const u32 last_chunk_size =
        (remainder == 0) ? blake3::CHUNK_SIZE : remainder;
    const u32 global_chunk_idx =
        blockIdx.x * kNumConsumerThreads + consumer_tid;
    const bool is_last_chunk = (global_chunk_idx == num_chunks - 1) &&
                               (last_chunk_size < blake3::CHUNK_SIZE);

    CUTLASS_PRAGMA_NO_UNROLL
    for (int load_idx = 0; load_idx < kNumLoads; ++load_idx) {
      // Wait for TMA load to complete
      pipeline.consumer_wait(smem_pipe_read);
      auto stage = smem_pipe_read.index();

      // Zero-pad OOB data in this load if processing the last (partial) chunk
      if (is_last_chunk) {
        zero_pad_partial_chunk_load(sA, consumer_tid, stage, load_idx,
                                    last_chunk_size);
      }

      // Process kNumBlocksPerLoad blocks from this load
      CUTLASS_PRAGMA_UNROLL
      for (int block_in_load = 0; block_in_load < kNumBlocksPerLoad;
           ++block_in_load) {
        int block_idx = load_idx * kNumBlocksPerLoad +
                        block_in_load;  // Global block index (0-15)
        compress_block<IsSingleChunk>(sA, rChainingValue, consumer_tid, stage,
                                      block_in_load, block_idx);
      }

      // Sync all consumer warpgroups after processing this stage
      // This ensures all consumers finish before any release the stage
      cutlass::arch::NamedBarrier::sync(
          kNumConsumerThreads,
          static_cast<uint32_t>(TensorHashBarriers::PrimaryConsumers));

      // Release the pipeline stage
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
    }

    // Store final hash result to sLeaves
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      sLeaves(i, consumer_tid) = rChainingValue(i);
    }
  }

  template <bool IsSingleChunk, class SmemTensorA,
            class RmemTensorChainingValue>
  CUTLASS_DEVICE void compress_block(SmemTensorA const& sA,
                                     RmemTensorChainingValue& rChainingValue,
                                     int consumer_tid, int stage,
                                     int block_in_load, int block_idx) {
    // Create register tensor for block data
    Tensor rBlock = make_tensor<uint32_t>(RmemLayoutBlock{});

    // Calculate word offset within the load for this block
    int word_offset = block_in_load * kNumWordsPerBlock;

    // Copy words from shared memory using 128-bit loads [consumer_thread][word_in_load][stage]
    // Load 4 words (128 bits) at a time instead of individual 32-bit loads
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kNumWordsPerBlock / 4; ++i) {
      uint4 tmp = *reinterpret_cast<const uint4*>(
          &sA(consumer_tid, word_offset + i * 4, stage));
      rBlock(i * 4 + 0) = tmp.x;
      rBlock(i * 4 + 1) = tmp.y;
      rBlock(i * 4 + 2) = tmp.z;
      rBlock(i * 4 + 3) = tmp.w;
    }

    // Compress the block - all chunks are 1024 bytes, all blocks are 64 bytes
    blake3::CompressParams params{
        .counter = blockIdx.x * kNumConsumerThreads + consumer_tid,
        .block_len = blake3::MSG_BLOCK_SIZE,
        .flags = blake3::KEYED_HASH};

    // Set CHUNK_START on the first block (block_idx == 0)
    if (block_idx == 0) {
      params.flags |= blake3::CHUNK_START;
    }

    // Set CHUNK_END on the last block (block_idx == kNumBlocksPerChunk - 1)
    if (block_idx == kNumBlocksPerChunk - 1) {
      params.flags |= blake3::CHUNK_END;
      // If the entire message is this single chunk, this is also the final
      // compression that produces the digest, so it must be flagged ROOT.
      if constexpr (IsSingleChunk) {
        params.flags |= blake3::ROOT;
      }
    }

    blake3::compress_msg_block_u32(rBlock, rChainingValue, params);
  }

  // Dual pipeline version of compress_block
  // Uses tid_in_group (0-255) for shared memory access
  // Uses global_consumer_tid (0-511) for counter calculation
  template <bool IsSingleChunk, class SmemTensorA,
            class RmemTensorChainingValue>
  CUTLASS_DEVICE void compress_block_dual(
      SmemTensorA const& sA, RmemTensorChainingValue& rChainingValue,
      int tid_in_group, int global_consumer_tid, int stage, int block_in_load,
      int block_idx) {
    // Create register tensor for block data
    Tensor rBlock = make_tensor<uint32_t>(RmemLayoutBlock{});

    // Calculate word offset within the load for this block
    int word_offset = block_in_load * kNumWordsPerBlock;

    // Copy words from shared memory using 128-bit loads [tid_in_group][word_in_load][stage]
    // Use tid_in_group for smem access (0-255 within group's smem region)
    // Load 4 words (128 bits) at a time instead of individual 32-bit loads
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kNumWordsPerBlock / 4; ++i) {
      uint4 tmp = *reinterpret_cast<const uint4*>(
          &sA(tid_in_group, word_offset + i * 4, stage));
      rBlock(i * 4 + 0) = tmp.x;
      rBlock(i * 4 + 1) = tmp.y;
      rBlock(i * 4 + 2) = tmp.z;
      rBlock(i * 4 + 3) = tmp.w;
    }

    // Compress the block - all chunks are 1024 bytes, all blocks are 64 bytes
    // Use global_consumer_tid for counter (0-511 for proper chunk identification)
    blake3::CompressParams params{
        .counter = blockIdx.x * kNumConsumerThreads + global_consumer_tid,
        .block_len = blake3::MSG_BLOCK_SIZE,
        .flags = blake3::KEYED_HASH};

    // Set CHUNK_START on the first block (block_idx == 0)
    if (block_idx == 0) {
      params.flags |= blake3::CHUNK_START;
    }

    // Set CHUNK_END on the last block (block_idx == kNumBlocksPerChunk - 1)
    if (block_idx == kNumBlocksPerChunk - 1) {
      params.flags |= blake3::CHUNK_END;
      // If the entire message is this single chunk, this is also the final
      // compression that produces the digest, so it must be flagged ROOT.
      if constexpr (IsSingleChunk) {
        params.flags |= blake3::ROOT;
      }
    }

    blake3::compress_msg_block_u32(rBlock, rChainingValue, params);
  }
};
}  // namespace pearl