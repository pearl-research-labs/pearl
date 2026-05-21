// SPDX-License-Identifier: see LICENSE
//
// sm_89 entry-point kernel for pearl-gemm. Companion to pearl_gemm_kernel.h
// (the sm_90a Hopper kernel `hopper_gemm_ws`). The two coexist; runtime
// dispatch from pearl_gemm_launch_template.h selects based on
// cudaDeviceProp.major.
//
// Architectural choice (per SM89_PORT_SPEC.md §0 and agent C9):
//   Unified-warp model. Every warp does cp.async loads AND mma.sync compute.
//   Drops the Hopper producer/consumer warpgroup split because sm_89 has no
//   setmaxnreg to redistribute the register file between specialized warps,
//   and cp.async doesn't have TMA's bulk-async behavior that justified
//   warp specialization.
//
// STATUS: skeleton. Fills in once the mainloop/epilogue collectives' load
// and mma bodies are wired. Tile scheduler is the existing
// StaticPersistentTileScheduler from tile_scheduler.hpp (cluster-aware code
// path no-ops when ClusterShape=(1,1,1) per kernel_traits_sm89.hpp).

#pragma once

#include <concepts>

#include "cute/tensor.hpp"

#include <cutlass/arch/barrier.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "collective_epilogue_sm89.hpp"
#include "collective_mainloop_sm89.hpp"

// NB: do NOT include tile_scheduler.hpp here — its SingleTileScheduler /
// StaticPersistentTileScheduler use sm_90 cluster intrinsics. The sm_89 host
// launcher (pearl_gemm_sm89_host.h) supplies its own SimpleTileScheduler.
#include "named_barrier.hpp"

#include "blake3/blake3.cuh"
#include "blake3/blake3_constants.hpp"
#include "host_signal_header.hpp"
#include "pow_utils.hpp"
#include "utils.h"

#if defined(PEARL_GEMM_BUILD_SM89) || defined(PEARL_GEMM_BUILD_SM120)
// Pulled in for the optional MultiNonceTileScheduler path. The header only
// defines a NonceContext type + a templated scheduler struct, so unconditional
// inclusion is safe — nothing is instantiated unless a multi-nonce scheduler
// type is bound at kernel template instantiation time.
#include "pearl_gemm_sm89_multinonce_scheduler.hpp"
#endif

namespace pearl {

using namespace cute;

// ---- Detection trait: does the scheduler's Params carry a NonceContext
//      pointer field (i.e., is this a multi-nonce scheduler)? Used to gate the
//      per-iteration Mainloop/Epilogue param override below. PersistentSwizzled
//      and Simple schedulers have no such field; MultiNonceTileScheduler does.
//
//      The detection is a compile-time `requires` check on the Params type,
//      so the false branch never instantiates the per-nonce override block
//      (which would otherwise fail to compile against schedulers that don't
//      define a `NonceContext` field).
template <typename SchedulerParams>
concept HasNonceContextsField = requires(SchedulerParams const& sp) {
  sp.ptr_nonce_contexts;
};

// Detection trait: does the WorkTileInfo carry a `first_nonce_in_cohort` flag?
// Used to gate the 10th mainloop argument so single-nonce schedulers
// (PersistentSwizzled, Simple) which don't have the field continue to compile
// against the same kernel template.
template <typename WorkTileInfo_>
concept HasFirstNonceFlag = requires(WorkTileInfo_ const& w) {
  { w.first_nonce_in_cohort } -> std::convertible_to<bool>;
};

template <typename KTraits, typename TileScheduler>
__global__ void __launch_bounds__(KTraits::kNumThreads, /*minBlocksPerSM=*/1)
ada_gemm(CUTE_GRID_CONSTANT
         typename ::pearl::CollectiveMainloopSm89<KTraits>::Params const
             mainloop_params,
         CUTE_GRID_CONSTANT
         typename ::pearl::CollectiveEpilogueSm89<KTraits>::Params const
             epilogue_params,
         CUTE_GRID_CONSTANT
         typename TileScheduler::Params const scheduler_params) {

  using TileShape_MNK  = typename KTraits::TileShape_MNK;
  using ClusterShape   = typename KTraits::ClusterShape_MNK;  // = (1,1,1)

  using CollectiveMainloop = ::pearl::CollectiveMainloopSm89<KTraits>;
  using CollectiveEpilogue = ::pearl::CollectiveEpilogueSm89<KTraits>;
  using WorkTileInfo       = typename TileScheduler::WorkTileInfo;

  static constexpr bool SkipDenoising = KTraits::SkipDenoising;
  static constexpr bool SkipReduction = KTraits::SkipReduction;

  extern __shared__ char shared_memory[];
  auto& shared_storage =
      *reinterpret_cast<typename KTraits::SharedStorage*>(shared_memory);

  // Sm_89: the mainloop body uses raw cp.async.commit_group / wait_group + named
  // barriers directly. No cutlass::PipelineAsync wrapper instance is needed —
  // its Params API was designed for the producer/consumer warp-specialized
  // pattern we deliberately dropped. We pass dummy refs only to keep the
  // mainloop signature stable with the Hopper version.

  CollectiveMainloop collective_mainloop;
  CollectiveEpilogue collective_epilogue;

  int const k_tile_count =
      cutlass::ceil_div(get<2>(mainloop_params.problem_shape), KTraits::bK);

  // Single-CTA sync — no cluster on sm_89.
  __syncthreads();

  // ===========================================================================
  // Unified warp body (no producer/consumer split).
  // ===========================================================================
  TileScheduler scheduler{};
  typename KTraits::TiledMma        tiled_mma;
  typename KTraits::TiledMmaDenoise tiled_mma_denoise;

  int const thread_idx = threadIdx.x;

  bool local_block_found = false;
  int  block_found_k_tile = 0;

  collective_mainloop.mma_init();

  auto work_tile_info = scheduler.get_initial_work(scheduler_params);
  CUTLASS_PRAGMA_NO_UNROLL
  while (work_tile_info.is_valid(scheduler_params)) {
    // GEMM accumulator: int32 partition_fragment_C from the TiledMma.
    auto tCrC = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
    clear(tCrC);

    auto transcript_extraction_tensor =
        make_tensor<uint32_t>(Int<blake3::MSG_BLOCK_SIZE_U32>{});
    if constexpr (!SkipReduction) {
      clear(transcript_extraction_tensor);
    }

    cute::tuple<int32_t, int32_t, int32_t> block_coord =
        work_tile_info.template get_block_coord<ClusterShape>(scheduler_params);

    // -----------------------------------------------------------------
    // Per-iteration param override for multi-nonce scheduler.
    //
    // When the bound TileScheduler is MultiNonceTileScheduler, its Params
    // carries a device array of NonceContext entries. The third slot of
    // block_coord is the nonce_idx; we copy mainloop_params/epilogue_params
    // for this iteration and patch in the per-nonce pointers (ptr_A,
    // ptr_A_scales, ptr_C). All other fields — including ptr_B, the layouts,
    // and any PoW signal pointers — stay the same since B is held constant
    // across the nonce sweep in the mining inner loop.
    //
    // When the scheduler has no `ptr_nonce_contexts` field (PersistentSwizzled,
    // Simple), this `if constexpr` evaluates to false and the override is a
    // no-op — the existing single-nonce path is bit-exact.
    auto mainloop_params_local = mainloop_params;
    auto epilogue_params_local = epilogue_params;
    if constexpr (HasNonceContextsField<typename TileScheduler::Params>) {
      auto const* ctxs = scheduler_params.ptr_nonce_contexts;
      int const nonce_idx = cute::get<2>(block_coord);
      if (ctxs != nullptr) {
        auto const& ctx = ctxs[nonce_idx];
        mainloop_params_local.ptr_A           = ctx.ptr_A;
        epilogue_params_local.ptr_A_scales    = ctx.ptr_A_scales;
        epilogue_params_local.ptr_C           =
            static_cast<typename CollectiveEpilogue::ElementOut*>(ctx.ptr_C);
        // Wave-13: per-nonce PoW signal slots � each nonce writes to its own
        // HostSignalHeader (pinned) + HostSignalSync (device) so 256 concurrent
        // nonces dont all collide on one global_lock CAS. ptr_pow_target and
        // ptr_pow_key are also per-nonce since each nonce derives a unique
        // commitment_hash and adjusted target.
        mainloop_params_local.host_signal_header_pinned =
            reinterpret_cast<HostSignalHeader*>(ctx.host_signal_header_pinned);
        mainloop_params_local.host_signal_sync =
            reinterpret_cast<HostSignalSync*>(ctx.host_signal_sync);
        mainloop_params_local.ptr_pow_target  = ctx.ptr_pow_target;
        mainloop_params_local.ptr_pow_key     = ctx.ptr_pow_key;
      }
    }

    // -----------------------------------------------------------------
    // Mainloop: cp.async multistage. See collective_mainloop_sm89.hpp.
    //
    // first_nonce_in_cohort: only meaningful when the bound TileScheduler is
    // MultiNonceTileScheduler — pulled off the WorkTileInfo to tell the
    // mainloop whether the same CTA already loaded B for this (m, n) tile in
    // the previous nonce iteration (`false` → skip B cp.async fetches if the
    // mainloop is instantiated with kPersistB=true). For single-nonce
    // schedulers the flag defaults to `true` and the mainloop fetches B
    // every call as before.
    bool first_nonce_in_cohort = true;
    if constexpr (HasFirstNonceFlag<WorkTileInfo>) {
      first_nonce_in_cohort = work_tile_info.first_nonce_in_cohort;
    }
    collective_mainloop.mainloop(
        mainloop_params_local, shared_storage, block_coord, k_tile_count,
        tCrC, transcript_extraction_tensor,
        local_block_found, block_found_k_tile, thread_idx,
        first_nonce_in_cohort);

    // int32 -> float32 for scale (and optionally denoise).
    auto tCrD_fp32 = make_tensor_like<float>(tCrC);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tCrD_fp32); ++i) {
      tCrD_fp32(i) = static_cast<float>(tCrC(i));
    }

    if constexpr (!SkipDenoising) {
      __syncthreads();  // mainloop smem free -> denoise loads can land
      collective_epilogue.load_denoise(epilogue_params_local, shared_storage,
                                       block_coord, thread_idx);
      collective_epilogue.denoise(epilogue_params_local, tCrD_fp32,
                                  shared_storage, tiled_mma, block_coord,
                                  thread_idx);
    }

    // NB: scale takes the MAIN int8 TiledMma (defines per-thread accumulator
    // layout), not the denoise TiledMma.
    collective_epilogue.scale(epilogue_params_local, tCrD_fp32, shared_storage,
                              tiled_mma, thread_idx, block_coord);
    collective_epilogue.store(epilogue_params_local, shared_storage, thread_idx,
                              block_coord);

    if constexpr (!SkipReduction) {
      local_block_found = check_pow_target(
          transcript_extraction_tensor,
          mainloop_params_local.ptr_pow_target,
          mainloop_params_local.ptr_pow_key);

      if (local_block_found) {
        write_host_signal_header<typename KTraits::TiledMma, TileShape_MNK>(
            mainloop_params_local.host_signal_sync,
            mainloop_params_local.host_signal_header_pinned,
            mainloop_params_local.problem_shape, block_coord, thread_idx,
            mainloop_params_local.ptr_pow_target);
      }
    }

    collective_epilogue.store_tail();
    work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(
        scheduler_params, work_tile_info);
  }
}

}  // namespace pearl
