#pragma once

#include "cute/tensor.hpp"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "collective_epilogue.hpp"
#include "collective_mainloop.hpp"

#include "named_barrier.hpp"
#include "tile_scheduler.hpp"

#include "blake3/blake3.cuh"
#include "blake3/blake3_constants.hpp"
#include "host_signal_header.hpp"
#include "pow_utils.hpp"
#include "utils.h"

namespace pearl {

using namespace cute;

#ifndef PEARL_P1K158_CHECK_WARP_STRIDE
#define PEARL_P1K158_CHECK_WARP_STRIDE 1
#endif

#ifndef PEARL_P1K165_FORCE_PUBLISH_CONSUMER
#define PEARL_P1K165_FORCE_PUBLISH_CONSUMER 0
#endif

#ifndef PEARL_P1K165_FORCE_PUBLISH_TILE_M
#define PEARL_P1K165_FORCE_PUBLISH_TILE_M 0
#endif

#ifndef PEARL_P1K165_FORCE_PUBLISH_TILE_N
#define PEARL_P1K165_FORCE_PUBLISH_TILE_N 0
#endif

#ifndef PEARL_P1K165_FORCE_PUBLISH_TILE_B
#define PEARL_P1K165_FORCE_PUBLISH_TILE_B 0
#endif

static_assert(PEARL_P1K158_CHECK_WARP_STRIDE >= 1,
              "PEARL_P1K158_CHECK_WARP_STRIDE must be >= 1");
static_assert(PEARL_P1K165_FORCE_PUBLISH_CONSUMER >= 0,
              "PEARL_P1K165_FORCE_PUBLISH_CONSUMER must be >= 0");

CUTLASS_DEVICE bool p1k158_should_run_pow_check(int consumer_tix) {
#if PEARL_P1K158_CHECK_WARP_STRIDE == 1
  (void)consumer_tix;
  return true;
#else
  int const consumer_warp = consumer_tix / cutlass::NumThreadsPerWarp;
  return (consumer_warp % PEARL_P1K158_CHECK_WARP_STRIDE) == 0;
#endif
}

template <typename BlockCoord>
CUTLASS_DEVICE bool p1k165_force_publish_matches(BlockCoord const& block_coord,
                                                 int consumer_tix) {
#if defined(PEARL_P1K165_FORCE_PUBLISH_FIXED_CONSUMER)
  return consumer_tix == PEARL_P1K165_FORCE_PUBLISH_CONSUMER &&
         int(get<0>(block_coord)) == PEARL_P1K165_FORCE_PUBLISH_TILE_M &&
         int(get<1>(block_coord)) == PEARL_P1K165_FORCE_PUBLISH_TILE_N &&
         int(get<2>(block_coord)) == PEARL_P1K165_FORCE_PUBLISH_TILE_B;
#else
  (void)block_coord;
  (void)consumer_tix;
  return false;
#endif
}

template <typename KTraits, typename TileShape_MNK, typename MainloopParams,
          typename SharedStorage, typename BlockCoord>
CUTLASS_DEVICE void finalize_native_2x64_ring_from_producer(
    MainloopParams const& mainloop_params, SharedStorage& shared_storage,
    BlockCoord const& block_coord, int xq_boundary_count) {
  int const lane_idx = threadIdx.x % cutlass::NumThreadsPerWarp;

#if defined(PEARL_P1K111_NATIVE_SIDEBAND_ORACLE) || \
    defined(PEARL_P1K112_NATIVE_SIDEBAND_FILL_ONLY)
  if constexpr (KTraits::EnableNative2x64Ring) {
    int const journal_words =
        KTraits::kNumMmaThreads * KTraits::kXqJournalMaxBoundaries;
    CUTLASS_PRAGMA_NO_UNROLL
    for (int idx = lane_idx; idx < journal_words;
         idx += cutlass::NumThreadsPerWarp) {
      uint32_t const consumer_tix =
          uint32_t(idx / KTraits::kXqJournalMaxBoundaries);
      uint32_t const boundary =
          uint32_t(idx % KTraits::kXqJournalMaxBoundaries);
      uint32_t const tile_mix =
          (uint32_t(get<0>(block_coord)) * 0x9e3779b9u) ^
          (uint32_t(get<1>(block_coord)) * 0x85ebca6bu) ^
          (uint32_t(get<2>(block_coord)) * 0xc2b2ae35u);
      shared_storage.xq_journal.data()[idx] =
          tile_mix ^ (consumer_tix * 0x27d4eb2du) ^
          (boundary * 0x165667b1u);
    }
    __syncwarp();
  }
#if defined(PEARL_P1K112_NATIVE_SIDEBAND_FILL_ONLY)
  return;
#endif
#endif

  CUTLASS_PRAGMA_NO_UNROLL
  for (int consumer_tix = lane_idx; consumer_tix < KTraits::kNumMmaThreads;
       consumer_tix += cutlass::NumThreadsPerWarp) {
    auto transcript_extraction_tensor =
        make_tensor<uint32_t>(Int<blake3::MSG_BLOCK_SIZE_U32>{});
    reconstruct_transcript_from_xq_journal<KTraits::kXqJournalMaxBoundaries>(
        transcript_extraction_tensor, shared_storage.xq_journal.data(),
        consumer_tix, xq_boundary_count);

    if (mainloop_params.coalesce_receipts) {
      auto union_transcript =
          make_adjacent_lane_union_transcript(transcript_extraction_tensor);
      if ((consumer_tix & 1) == 0 &&
          p1k158_should_run_pow_check(consumer_tix)) {
        bool const block_found =
            check_pow_target(union_transcript, mainloop_params.ptr_pow_target,
                             mainloop_params.ptr_pow_key);
        if (block_found) {
          write_host_signal_header_pair<typename KTraits::TiledMma,
                                        TileShape_MNK>(
              mainloop_params.host_signal_sync,
              mainloop_params.host_signal_header_pinned,
              mainloop_params.problem_shape, block_coord, consumer_tix,
              consumer_tix ^ 1, mainloop_params.ptr_pow_target);
        }
      }
    } else {
      if (p1k158_should_run_pow_check(consumer_tix)) {
        bool const block_found =
            check_pow_target(transcript_extraction_tensor,
                             mainloop_params.ptr_pow_target,
                             mainloop_params.ptr_pow_key);
        if (block_found) {
          write_host_signal_header<typename KTraits::TiledMma, TileShape_MNK>(
              mainloop_params.host_signal_sync,
              mainloop_params.host_signal_header_pinned,
              mainloop_params.problem_shape, block_coord, consumer_tix,
              mainloop_params.ptr_pow_target);
        }
      }
    }
  }
}

template <typename KTraits, typename MainloopParams, typename BlockCoord>
CUTLASS_DEVICE void fill_native_global_journal_from_producer(
    MainloopParams const& mainloop_params, BlockCoord const& block_coord,
    int xq_boundary_count) {
  if constexpr (KTraits::EnableNativeGlobalJournalFill) {
    auto* journal = mainloop_params.ptr_global_sideband_journal;
    if (journal == nullptr) {
      return;
    }

    auto [M, N, K, R] = mainloop_params.problem_shape;
    (void)M;
    (void)K;
    (void)R;
    int const num_blocks_n = (N + KTraits::bN - 1) / KTraits::bN;
    int const tile_id = int(get<0>(block_coord)) * num_blocks_n +
                        int(get<1>(block_coord));
    int const num_tiles = mainloop_params.global_sideband_tiles;
    int const num_consumers = mainloop_params.global_sideband_consumers;
    int const journal_boundaries = mainloop_params.global_sideband_boundaries;
    int const active_boundaries =
        xq_boundary_count < journal_boundaries ? xq_boundary_count
                                               : journal_boundaries;
    if (tile_id < 0 || tile_id >= num_tiles || num_consumers <= 0 ||
        journal_boundaries <= 0 || active_boundaries <= 0) {
      return;
    }

    int const lane_idx = threadIdx.x % cutlass::NumThreadsPerWarp;
    uint32_t const tile_mix =
        (uint32_t(get<0>(block_coord)) * 0x9e3779b9u) ^
        (uint32_t(get<1>(block_coord)) * 0x85ebca6bu) ^
        (uint32_t(get<2>(block_coord)) * 0xc2b2ae35u);
    int const total = num_consumers * active_boundaries;
    CUTLASS_PRAGMA_NO_UNROLL
    for (int idx = lane_idx; idx < total; idx += cutlass::NumThreadsPerWarp) {
      uint32_t const consumer_tix = uint32_t(idx / active_boundaries);
      uint32_t const boundary = uint32_t(idx % active_boundaries);
      int const offset =
          (tile_id * num_consumers + int(consumer_tix)) * journal_boundaries +
          int(boundary);
      journal[offset] = tile_mix ^ (consumer_tix * 0x27d4eb2du) ^
                        (boundary * 0x165667b1u);
    }
    __syncwarp();
  }
}

template <typename KTraits, typename BlockCoord>
CUTLASS_DEVICE void fill_native_2x64_synthetic_sideband_for_consumer(
    uint32_t* xq_journal, BlockCoord const& block_coord, int consumer_tix,
    int xq_boundary_count) {
  uint32_t const tile_mix =
      (uint32_t(get<0>(block_coord)) * 0x9e3779b9u) ^
      (uint32_t(get<1>(block_coord)) * 0x85ebca6bu) ^
      (uint32_t(get<2>(block_coord)) * 0xc2b2ae35u);
  CUTLASS_PRAGMA_NO_UNROLL
  for (int q = 0; q < xq_boundary_count &&
                  q < KTraits::kXqJournalMaxBoundaries;
       ++q) {
    xq_journal[consumer_tix * KTraits::kXqJournalMaxBoundaries + q] =
        tile_mix ^ (uint32_t(consumer_tix) * 0x27d4eb2du) ^
        (uint32_t(q) * 0x165667b1u);
  }
}

template <typename KTraits, typename TileShape_MNK, typename MainloopParams,
          typename TranscriptTensor, typename BlockCoord>
CUTLASS_DEVICE void check_native_2x64_synthetic_sideband_for_consumer(
    MainloopParams const& mainloop_params,
    TranscriptTensor& transcript_extraction_tensor,
    BlockCoord const& block_coord, int consumer_tix) {
  if (mainloop_params.coalesce_receipts) {
    auto union_transcript =
        make_adjacent_lane_union_transcript(transcript_extraction_tensor);
    if ((consumer_tix & 1) == 0 &&
        p1k158_should_run_pow_check(consumer_tix)) {
      bool const block_found =
          check_pow_target(union_transcript, mainloop_params.ptr_pow_target,
                           mainloop_params.ptr_pow_key);
      if (block_found) {
        write_host_signal_header_pair<typename KTraits::TiledMma,
                                      TileShape_MNK>(
            mainloop_params.host_signal_sync,
            mainloop_params.host_signal_header_pinned,
            mainloop_params.problem_shape, block_coord, consumer_tix,
            consumer_tix ^ 1, mainloop_params.ptr_pow_target);
      }
    }
  } else {
    if (p1k158_should_run_pow_check(consumer_tix)) {
      bool const block_found =
          check_pow_target(transcript_extraction_tensor,
                           mainloop_params.ptr_pow_target,
                           mainloop_params.ptr_pow_key);
      if (block_found) {
        write_host_signal_header<typename KTraits::TiledMma, TileShape_MNK>(
            mainloop_params.host_signal_sync,
            mainloop_params.host_signal_header_pinned,
            mainloop_params.problem_shape, block_coord, consumer_tix,
            mainloop_params.ptr_pow_target);
      }
    }
  }
}

template <typename KTraits, typename TileScheduler>
__global__ void __launch_bounds__(
    KTraits::kNumWarps* cutlass::NumThreadsPerWarp, 1)
    hopper_proof_gemm_ws(
        CUTE_GRID_CONSTANT
        typename ::pearl::CollectiveMainloop<KTraits>::Params const
            mainloop_params,
        CUTE_GRID_CONSTANT
        typename TileScheduler::Params const scheduler_params) {

  using TileShape_MNK = typename KTraits::TileShape_MNK;
  using ClusterShape = typename KTraits::ClusterShape_MNK;

  static constexpr int NumMmaThreads = size(typename KTraits::TiledMma{});
  static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;
  static constexpr int srcLane = KTraits::srcLane;

  using CollectiveMainloop = ::pearl::CollectiveMainloop<KTraits>;
  using MainloopPipeline = typename KTraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using WorkTileInfo = typename TileScheduler::WorkTileInfo;
  static constexpr bool SkipReduction = KTraits::SkipReduction;
  static constexpr bool SkipProofCheck = KTraits::SkipProofCheck;

  static_assert(KTraits::ProofOnly,
                "hopper_proof_gemm_ws must only be used for ProofOnly kernels");

  extern __shared__ char shared_memory[];
  auto& shared_storage =
      *reinterpret_cast<typename KTraits::SharedStorage*>(shared_memory);

  int const lane_predicate = cute::elect_one_sync();
  int const warp_idx = cutlass::canonical_warp_idx_sync();

  if (warp_idx == 0 && lane_predicate) {
    CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
  }

  PipelineParams pipeline_params;
  pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytes;
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  pipeline_params.role = warp_group_idx == 0
                             ? MainloopPipeline::ThreadCategory::Producer
                             : MainloopPipeline::ThreadCategory::Consumer;
  pipeline_params.is_leader = lane_predicate;
  pipeline_params.num_consumers = NumMmaThreads;

  MainloopPipeline pipeline(shared_storage.pipeline, pipeline_params,
                            ClusterShape{});

  CollectiveMainloop collective_mainloop;

  const int k_tile_count =
      cutlass::ceil_div(shape<1>(mainloop_params.layout_A), KTraits::bK);

  if constexpr (size(ClusterShape{}) > 1) {
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
  } else {
    __syncthreads();
  }

  static_assert(KTraits::kNumWarps == 8 || KTraits::kNumWarps == 12 ||
                KTraits::kNumWarps == 16 || KTraits::kNumWarps == 20);
  if (warp_group_idx == 0) {
    cutlass::arch::warpgroup_reg_dealloc<KTraits::kNumWarps == 16 ? 32 : 24>();

    int warp_idx_in_warpgroup =
        __shfl_sync(0xffffffff,
                    (threadIdx.x / cutlass::NumThreadsPerWarp) %
                        cutlass::NumWarpsPerWarpGroup,
                    srcLane);
    if (warp_idx_in_warpgroup == 0) {
      PipelineState smem_pipe_write =
          cutlass::make_producer_start_state<MainloopPipeline>();

      uint16_t const tma_mcast_mask_a = create_tma_multicast_mask<1>(
          Layout<ClusterShape>{}, block_id_in_cluster());
      uint16_t const tma_mcast_mask_b = create_tma_multicast_mask<0>(
          Layout<ClusterShape>{}, block_id_in_cluster());
      TileScheduler scheduler{};

      WorkTileInfo work_tile_info =
          scheduler.get_initial_work(scheduler_params);
      CUTLASS_PRAGMA_NO_UNROLL
      while (work_tile_info.is_valid(scheduler_params)) {

        cute::tuple<int32_t, int32_t, int32_t> block_coord =
            work_tile_info.template get_block_coord<ClusterShape>(
                scheduler_params);

        collective_mainloop.load(mainloop_params, pipeline, smem_pipe_write,
                                 shared_storage, block_coord, k_tile_count,
                                 tma_mcast_mask_a, tma_mcast_mask_b);
	        collective_mainloop.load_tail(pipeline, smem_pipe_write);

#if !defined(PEARL_P1K150_SCALAR16_FINAL_GLOBAL_STORE) && \
    !defined(PEARL_P1K165_TWO_PHASE_POW_CHECK)
        if constexpr (KTraits::EnableNativeGlobalJournalFill) {
          int requested_boundary_count = shape<1>(mainloop_params.layout_A) / KTraits::R;
          if (mainloop_params.global_sideband_boundaries > 0 &&
              mainloop_params.global_sideband_boundaries < requested_boundary_count) {
            requested_boundary_count = mainloop_params.global_sideband_boundaries;
          }
          fill_native_global_journal_from_producer<KTraits>(
              mainloop_params, block_coord, requested_boundary_count);
        }
#endif

#if defined(PEARL_P1K113_NATIVE_SIDEBAND_CONSUMER_SYNTH_CHECK) || \
    defined(PEARL_P1K114_NATIVE_SIDEBAND_CONSUMER_FILL_ONLY) || \
    defined(PEARL_P1K115_NATIVE_SIDEBAND_FILL_SWEEP) || \
    defined(PEARL_P1K116_NATIVE_SIDEBAND_CHECK_SWEEP)
        constexpr bool kRunNativeProducerFinalizer = false;
#else
        constexpr bool kRunNativeProducerFinalizer = true;
#endif
        if constexpr (kRunNativeProducerFinalizer && !SkipReduction &&
                      !SkipProofCheck && KTraits::EnableNative2x64Ring) {
          int const xq_boundary_count =
              shape<1>(mainloop_params.layout_A) / KTraits::R;
          finalize_native_2x64_ring_from_producer<KTraits, TileShape_MNK>(
              mainloop_params, shared_storage, block_coord, xq_boundary_count);
        }

        work_tile_info = scheduler.template get_next_work</*IsProducer=*/true>(
            scheduler_params, work_tile_info);
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<
        KTraits::kConsumerRegisterTarget>();

    TileScheduler scheduler{};

    typename KTraits::TiledMma tiled_mma;
    PipelineState smem_pipe_read;

    int consumer_tix = static_cast<int>(threadIdx.x) - NumCopyThreads;

    bool local_block_found = 0;
    int block_found_k_tile = 0;

    collective_mainloop.mma_init();

    WorkTileInfo work_tile_info = scheduler.get_initial_work(scheduler_params);
    CUTLASS_PRAGMA_NO_UNROLL
    while (work_tile_info.is_valid(scheduler_params)) {
      Tensor tCrC = partition_fragment_C(
          tiled_mma, select<0, 1>(TileShape_MNK{}));
      clear(tCrC);

      using TranscriptWordCount = Int<
          native_global_journal_elides_consumer_transcript<KTraits>()
              ? 1
              : blake3::MSG_BLOCK_SIZE_U32>;
      auto transcript_extraction_tensor =
          make_tensor<uint32_t>(TranscriptWordCount{});
      if constexpr (!SkipReduction) {
        clear(transcript_extraction_tensor);
      }

      cute::tuple<int32_t, int32_t, int32_t> block_coord =
          work_tile_info.template get_block_coord<ClusterShape>(
              scheduler_params);

		      collective_mainloop.mma(mainloop_params, pipeline, smem_pipe_read, tCrC,
		                              transcript_extraction_tensor, local_block_found,
		                              block_found_k_tile, consumer_tix, shared_storage,
		                              block_coord, k_tile_count);

#if defined(PEARL_P1K113_NATIVE_SIDEBAND_CONSUMER_SYNTH_CHECK) || \
    defined(PEARL_P1K114_NATIVE_SIDEBAND_CONSUMER_FILL_ONLY) || \
    defined(PEARL_P1K115_NATIVE_SIDEBAND_FILL_SWEEP) || \
    defined(PEARL_P1K116_NATIVE_SIDEBAND_CHECK_SWEEP)
      if constexpr (!SkipReduction && !SkipProofCheck &&
                    KTraits::EnableNative2x64Ring) {
        int const xq_boundary_count =
            shape<1>(mainloop_params.layout_A) / KTraits::R;
        int sideband_boundary_count = xq_boundary_count;
#if defined(PEARL_P1K115_NATIVE_SIDEBAND_FILL_SWEEP) || \
    defined(PEARL_P1K116_NATIVE_SIDEBAND_CHECK_SWEEP)
        if (mainloop_params.inner_hash_counter != nullptr) {
          uint64_t const requested = mainloop_params.inner_hash_counter[0];
          sideband_boundary_count = requested > uint64_t(xq_boundary_count)
                                        ? xq_boundary_count
                                        : int(requested);
        }
#endif
        fill_native_2x64_synthetic_sideband_for_consumer<KTraits>(
            shared_storage.xq_journal.data(), block_coord, consumer_tix,
            sideband_boundary_count);
#if defined(PEARL_P1K113_NATIVE_SIDEBAND_CONSUMER_SYNTH_CHECK) || \
    defined(PEARL_P1K116_NATIVE_SIDEBAND_CHECK_SWEEP)
        reconstruct_transcript_from_xq_journal<
            KTraits::kXqJournalMaxBoundaries>(
            transcript_extraction_tensor, shared_storage.xq_journal.data(),
            consumer_tix, sideband_boundary_count);
        check_native_2x64_synthetic_sideband_for_consumer<KTraits,
                                                          TileShape_MNK>(
            mainloop_params, transcript_extraction_tensor, block_coord,
            consumer_tix);
#endif
      }
#endif

#if defined(PEARL_P1K165_FORCE_PUBLISH_FIXED_CONSUMER)
      if constexpr (!SkipReduction && !SkipProofCheck) {
        if (p1k165_force_publish_matches(block_coord, consumer_tix)) {
          write_host_signal_header<typename KTraits::TiledMma, TileShape_MNK>(
              mainloop_params.host_signal_sync,
              mainloop_params.host_signal_header_pinned,
              mainloop_params.problem_shape, block_coord, consumer_tix,
              mainloop_params.ptr_pow_target);
        }
      }
#else
			      if constexpr (!SkipReduction && !SkipProofCheck &&
			                    KTraits::EnableXqJournal) {
	        int const xq_boundary_count =
	            shape<1>(mainloop_params.layout_A) / KTraits::R;
	        reconstruct_transcript_from_xq_journal<
	            KTraits::kXqJournalMaxBoundaries>(
	            transcript_extraction_tensor, shared_storage.xq_journal.data(),
	            consumer_tix, xq_boundary_count);
	      }

		      if constexpr (!SkipReduction && !SkipProofCheck &&
		                    !KTraits::EnableNative2x64Ring) {
        if (mainloop_params.coalesce_receipts) {
          auto union_transcript =
              make_adjacent_lane_union_transcript(transcript_extraction_tensor);
          if ((consumer_tix & 1) == 0 &&
              p1k158_should_run_pow_check(consumer_tix)) {
            local_block_found =
                check_pow_target(union_transcript, mainloop_params.ptr_pow_target,
                                 mainloop_params.ptr_pow_key);

            if (local_block_found) {
              write_host_signal_header_pair<typename KTraits::TiledMma,
                                            TileShape_MNK>(
                  mainloop_params.host_signal_sync,
                  mainloop_params.host_signal_header_pinned,
                  mainloop_params.problem_shape, block_coord, consumer_tix,
                  consumer_tix ^ 1,
                  mainloop_params.ptr_pow_target);
            }
          }
        } else {
          if (p1k158_should_run_pow_check(consumer_tix)) {
            local_block_found = check_pow_target(transcript_extraction_tensor,
                                                 mainloop_params.ptr_pow_target,
                                                 mainloop_params.ptr_pow_key);

            if (local_block_found) {
              write_host_signal_header<typename KTraits::TiledMma, TileShape_MNK>(
                  mainloop_params.host_signal_sync,
                  mainloop_params.host_signal_header_pinned,
                  mainloop_params.problem_shape, block_coord, consumer_tix,
                  mainloop_params.ptr_pow_target);
            }
          }
        }
      }
#endif

      work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(
          scheduler_params, work_tile_info);
    }
  }
}

template <typename KTraits, typename TileScheduler>
__global__ void __launch_bounds__(
    KTraits::kNumWarps* cutlass::NumThreadsPerWarp, 1)
    hopper_gemm_ws(CUTE_GRID_CONSTANT
                   typename ::pearl::CollectiveMainloop<KTraits>::Params const
                       mainloop_params,
                   CUTE_GRID_CONSTANT
                   typename ::pearl::CollectiveEpilogue<KTraits>::Params const
                       epilogue_params,
                   CUTE_GRID_CONSTANT
                   typename TileScheduler::Params const scheduler_params) {

  using TileShape_MNK = typename KTraits::TileShape_MNK;
  using ClusterShape = typename KTraits::ClusterShape_MNK;

  static constexpr int NumMmaThreads = size(typename KTraits::TiledMma{});
  static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;
  static constexpr int srcLane = KTraits::srcLane;

  using CollectiveMainloop = ::pearl::CollectiveMainloop<KTraits>;
  using CollectiveEpilogue = ::pearl::CollectiveEpilogue<KTraits>;

  using MainloopPipeline = typename KTraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  using DenoisePipeline = typename KTraits::DenoisePipeline;
  using DenoisePipelineParams = typename DenoisePipeline::Params;
  using DenoisePipelineState = typename DenoisePipeline::PipelineState;

  using WorkTileInfo = typename TileScheduler::WorkTileInfo;
  static constexpr bool SkipDenoising = KTraits::SkipDenoising;
  static constexpr bool SkipReduction = KTraits::SkipReduction;
  static constexpr bool ProofOnly = KTraits::ProofOnly;
  static constexpr bool SkipProofCheck = KTraits::SkipProofCheck;

  extern __shared__ char shared_memory[];
  auto& shared_storage =
      *reinterpret_cast<typename KTraits::SharedStorage*>(shared_memory);

  int const lane_predicate = cute::elect_one_sync();
  int const warp_idx = cutlass::canonical_warp_idx_sync();

  // Issue Tma Descriptor Prefetch from a single thread
  if (warp_idx == 0 && lane_predicate) {
    CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
    if constexpr (!ProofOnly) {
      CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
    }
  }

  // Obtain warp index
  int const warp_group_thread_idx =
      threadIdx.x % cutlass::NumThreadsPerWarpGroup;

  // TMA load pipeline: 1 thread in producer WG is producer, MMA threads are consumers
  PipelineParams pipeline_params;
  pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytes;
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  pipeline_params.role = warp_group_idx == 0
                             ? MainloopPipeline::ThreadCategory::Producer
                             : MainloopPipeline::ThreadCategory::Consumer;
  pipeline_params.is_leader = lane_predicate;
  pipeline_params.num_consumers = NumMmaThreads;

  // Denoise load pipelines: 1 thread in producer WG is producer, MMA threads are consumers
  // Two pipelines since transaction bytes differ and we want to wait on loads separately
  DenoisePipelineParams AxEB_pipeline_params;
  AxEB_pipeline_params.transaction_bytes =
      CollectiveEpilogue::TmaTransactionBytesAxEB;
  AxEB_pipeline_params.role = warp_group_idx == 0
                                  ? DenoisePipeline::ThreadCategory::Producer
                                  : DenoisePipeline::ThreadCategory::Consumer;
  AxEB_pipeline_params.is_leader = warp_group_thread_idx == 0;
  AxEB_pipeline_params.num_consumers = NumMmaThreads;

  DenoisePipelineParams EAxBpEB_pipeline_params;
  EAxBpEB_pipeline_params.transaction_bytes =
      CollectiveEpilogue::TmaTransactionBytesEAxBpEB;
  EAxBpEB_pipeline_params.role =
      warp_group_idx == 0 ? DenoisePipeline::ThreadCategory::Producer
                          : DenoisePipeline::ThreadCategory::Consumer;
  EAxBpEB_pipeline_params.is_leader = warp_group_thread_idx == 0;
  EAxBpEB_pipeline_params.num_consumers = NumMmaThreads;

  // We're counting on pipeline constructor to call cutlass::arch::fence_barrier_init()
  //  and also to initialize barriers
  MainloopPipeline pipeline(shared_storage.pipeline, pipeline_params,
                            ClusterShape{});
  DenoisePipeline AxEB_pipeline(shared_storage.AxEB_pipeline,
                                AxEB_pipeline_params, ClusterShape{});
  DenoisePipeline EAxBpEB_pipeline(shared_storage.EAxBpEB_pipeline,
                                   EAxBpEB_pipeline_params, ClusterShape{});

  CollectiveMainloop collective_mainloop;
  CollectiveEpilogue collective_epilogue;

  const int k_tile_count =
      cutlass::ceil_div(shape<1>(mainloop_params.layout_A), KTraits::bK);

  // We need this to guarantee that the Pipeline init is visible to all producers and consumer blocks in the Cluster
  if constexpr (size(ClusterShape{}) > 1) {
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
  } else {
    __syncthreads();
  }

  static_assert(KTraits::kNumWarps == 8 || KTraits::kNumWarps == 12 ||
                KTraits::kNumWarps == 16 || KTraits::kNumWarps == 20);
  if (warp_group_idx == 0) {  // Producer
    // cutlass::arch::warpgroup_reg_dealloc<24>();
    cutlass::arch::warpgroup_reg_dealloc<KTraits::kNumWarps == 16 ? 32 : 24>();

    int warp_idx_in_warpgroup =
        __shfl_sync(0xffffffff,
                    (threadIdx.x / cutlass::NumThreadsPerWarp) %
                        cutlass::NumWarpsPerWarpGroup,
                    srcLane);
    if (warp_idx_in_warpgroup == 0) {  // Load A, B in producer warp 0
      PipelineState smem_pipe_write =
          cutlass::make_producer_start_state<MainloopPipeline>();

      DenoisePipelineState AxEB_pipe_write =
          cutlass::make_producer_start_state<DenoisePipeline>();
      DenoisePipelineState EAxBpEB_pipe_write =
          cutlass::make_producer_start_state<DenoisePipeline>();
      // tma masks are used to determine what data this CTA receives when participating in multicast
      uint16_t const tma_mcast_mask_a = create_tma_multicast_mask<1>(
          Layout<ClusterShape>{}, block_id_in_cluster());
      uint16_t const tma_mcast_mask_b = create_tma_multicast_mask<0>(
          Layout<ClusterShape>{}, block_id_in_cluster());
      TileScheduler scheduler{};

      WorkTileInfo work_tile_info =
          scheduler.get_initial_work(scheduler_params);
      CUTLASS_PRAGMA_NO_UNROLL
      while (work_tile_info.is_valid(scheduler_params)) {

        cute::tuple<int32_t, int32_t, int32_t> block_coord =
            work_tile_info.template get_block_coord<ClusterShape>(
                scheduler_params);

        collective_mainloop.load(mainloop_params, pipeline, smem_pipe_write,
                                 shared_storage, block_coord, k_tile_count,
                                 tma_mcast_mask_a, tma_mcast_mask_b);

        if constexpr (!SkipDenoising) {
          // we move mainloop load_tail inside the denoise for cluster-wide sync purposes
          collective_epilogue.load_denoise(
              pipeline, smem_pipe_write, epilogue_params, AxEB_pipeline,
              EAxBpEB_pipeline, AxEB_pipe_write, EAxBpEB_pipe_write,
              shared_storage, block_coord, tma_mcast_mask_a, tma_mcast_mask_b);

          collective_epilogue.load_denoise_tail(AxEB_pipeline, EAxBpEB_pipeline,
                                                AxEB_pipe_write,
                                                EAxBpEB_pipe_write);
        } else {
          collective_mainloop.load_tail(pipeline, smem_pipe_write);
        }

        work_tile_info = scheduler.template get_next_work</*IsProducer=*/true>(
            scheduler_params, work_tile_info);
      }
    }
  } else {  // Consumer
    cutlass::arch::warpgroup_reg_alloc<
        KTraits::kConsumerRegisterTarget>();

    TileScheduler scheduler{};

    // Initialize matmul objects.
    typename KTraits::TiledMma tiled_mma;
    typename KTraits::TiledMmaDenoise tiled_mma_denoise;

    PipelineState smem_pipe_read;

    DenoisePipelineState AxEB_pipe_read;
    DenoisePipelineState EAxBpEB_pipe_read;

    int consumer_tix = static_cast<int>(threadIdx.x) - NumCopyThreads;

    // Reduction parameters
    bool local_block_found = 0;
    int block_found_k_tile = 0;

    collective_mainloop.mma_init();

    WorkTileInfo work_tile_info = scheduler.get_initial_work(scheduler_params);
    CUTLASS_PRAGMA_NO_UNROLL
    while (work_tile_info.is_valid(scheduler_params)) {
      // GEMM accumulator.
      Tensor tCrC = partition_fragment_C(
          tiled_mma, select<0, 1>(TileShape_MNK{}));  // (M, N)
      clear(tCrC);

      // Transcript for accumulating intermediate hashes
      auto transcript_extraction_tensor =
          make_tensor<uint32_t>(Int<blake3::MSG_BLOCK_SIZE_U32>{});
      if constexpr (!SkipReduction) {
        clear(transcript_extraction_tensor);
      }

      cute::tuple<int32_t, int32_t, int32_t> block_coord =
          work_tile_info.template get_block_coord<ClusterShape>(
              scheduler_params);

      collective_mainloop.mma(mainloop_params, pipeline, smem_pipe_read, tCrC,
                              transcript_extraction_tensor, local_block_found,
                              block_found_k_tile, consumer_tix, shared_storage,
                              block_coord, k_tile_count);

      if constexpr (!ProofOnly) {
#ifdef PEARL_P1K141_SKIP_NON_PROOF_EPILOGUE
        (void)epilogue_params;
        (void)tiled_mma;
        (void)consumer_tix;
        (void)block_coord;
#else
        // Convert to float to accumulate denoising
        Tensor tCrD_fp32 = make_tensor_like<float>(tCrC);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrD_fp32); ++i) {
          tCrD_fp32(i) = static_cast<float>(tCrC(i));
        }

        if constexpr (!SkipDenoising) {
          warpgroup_wait<0>();
          collective_epilogue.denoise(tCrD_fp32, shared_storage, AxEB_pipeline,
                                      EAxBpEB_pipeline, AxEB_pipe_read,
                                      EAxBpEB_pipe_read, consumer_tix);
        }

        collective_epilogue.scale(epilogue_params, tCrD_fp32, shared_storage,
                                  tiled_mma, consumer_tix, block_coord);

#ifndef PEARL_P1K143_SKIP_EPILOGUE_STORE_CALL
        collective_epilogue.store(epilogue_params, shared_storage, consumer_tix,
                                  block_coord);
#endif
#endif
      }

      if constexpr (!SkipReduction && !SkipProofCheck) {
        if (mainloop_params.coalesce_receipts) {
          auto union_transcript =
              make_adjacent_lane_union_transcript(transcript_extraction_tensor);
          if ((consumer_tix & 1) == 0 &&
              p1k158_should_run_pow_check(consumer_tix)) {
            local_block_found =
                check_pow_target(union_transcript, mainloop_params.ptr_pow_target,
                                 mainloop_params.ptr_pow_key);

            if (local_block_found) {
              write_host_signal_header_pair<typename KTraits::TiledMma,
                                            TileShape_MNK>(
                  mainloop_params.host_signal_sync,
                  mainloop_params.host_signal_header_pinned,
                  mainloop_params.problem_shape, block_coord, consumer_tix,
                  consumer_tix ^ 1,
                  mainloop_params.ptr_pow_target);
            }
          }
        } else {
          if (p1k158_should_run_pow_check(consumer_tix)) {
            local_block_found = check_pow_target(transcript_extraction_tensor,
                                                 mainloop_params.ptr_pow_target,
                                                 mainloop_params.ptr_pow_key);

            if (local_block_found) {
              write_host_signal_header<typename KTraits::TiledMma, TileShape_MNK>(
                  mainloop_params.host_signal_sync,
                  mainloop_params.host_signal_header_pinned,
                  mainloop_params.problem_shape, block_coord, consumer_tix,
                  mainloop_params.ptr_pow_target);
            }
          }
        }
      }

      if constexpr (!ProofOnly) {
#if !defined(PEARL_P1K141_SKIP_NON_PROOF_EPILOGUE) && \
    !defined(PEARL_P1K143_SKIP_EPILOGUE_STORE_CALL)
        collective_epilogue.store_tail();
#endif
      }
      work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(
          scheduler_params, work_tile_info);
    }
  }
}

}  // namespace pearl
