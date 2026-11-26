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

  extern __shared__ char shared_memory[];
  auto& shared_storage =
      *reinterpret_cast<typename KTraits::SharedStorage*>(shared_memory);

  int const lane_predicate = cute::elect_one_sync();
  int const warp_idx = cutlass::canonical_warp_idx_sync();

  // Issue Tma Descriptor Prefetch from a single thread
  if (warp_idx == 0 && lane_predicate) {
    CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
    CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
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
    // cutlass::arch::warpgroup_reg_alloc<KTraits::kNumWarps == 12 ? 240 : 160>();
    cutlass::arch::warpgroup_reg_alloc<KTraits::kNumWarps == 8    ? 256
                                       : KTraits::kNumWarps == 12 ? 240
                                       : KTraits::kNumWarps == 16 ? 160
                                                                  : 112>();

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
                              k_tile_count);

      // Convert to float to accumulate denoising
      Tensor tCrD_fp32 = make_tensor_like<float>(tCrC);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tCrD_fp32); ++i) {
        tCrD_fp32(i) = static_cast<float>(tCrC(i));
      }

      if constexpr (!SkipDenoising) {
        // Hopper-only WGMMA sync; no-op on Blackwell consumer (sm_120/121)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        warpgroup_wait<0>();
#endif
        collective_epilogue.denoise(tCrD_fp32, shared_storage, AxEB_pipeline,
                                    EAxBpEB_pipeline, AxEB_pipe_read,
                                    EAxBpEB_pipe_read, consumer_tix);
      }

      collective_epilogue.scale(epilogue_params, tCrD_fp32, shared_storage,
                                tiled_mma, consumer_tix, block_coord);

      collective_epilogue.store(epilogue_params, shared_storage, consumer_tix,
                                block_coord);

      if constexpr (!SkipReduction) {
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

      collective_epilogue.store_tail();
      work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(
          scheduler_params, work_tile_info);
    }
  }
}

}  // namespace pearl
