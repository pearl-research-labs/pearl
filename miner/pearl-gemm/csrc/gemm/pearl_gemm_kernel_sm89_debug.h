// SPDX-License-Identifier: see LICENSE
//
// Debug-only sm_89 kernel: runs the SAME mainloop as ada_gemm() but skips
// the entire epilogue (no scale, no cast, no smem stage). Writes raw int32
// tCrC accumulator directly to a flat int32 gmem buffer via thr_mma.partition_C.
//
// Purpose: isolate mainloop+MMA correctness from scale/store. Bit-exact compare
// against torch._int_mm (int8 x int8 -> int32 is exact, no rounding).
//
// If this kernel produces correct output, the bug is somewhere in
// CollectiveEpilogueSm89::scale() / store() / R2S / S2G. If this kernel
// produces wrong output, the bug is in CollectiveMainloopSm89::mainloop()
// or in the TiledMma / smem layout / G2S / S2R wiring.

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "collective_mainloop_sm89.hpp"
#include "kernel_traits_sm89.hpp"

#include "pow_utils.hpp"  // pulled in transitively but make dependency explicit
#include "utils.h"

namespace pearl {

using namespace cute;

// Per-CTA debug kernel. Same lifecycle as ada_gemm() but the epilogue is
// replaced by a direct int32 partition_C store.
template <typename KTraits, typename TileScheduler>
__global__ void __launch_bounds__(KTraits::kNumThreads, /*minBlocksPerSM=*/1)
ada_gemm_debug_int32(
    CUTE_GRID_CONSTANT
    typename ::pearl::CollectiveMainloopSm89<KTraits>::Params const
        mainloop_params,
    int32_t* __restrict__ ptr_C_i32,
    int64_t ldc,
    CUTE_GRID_CONSTANT
    typename TileScheduler::Params const scheduler_params) {

  using TileShape_MNK = typename KTraits::TileShape_MNK;
  using ClusterShape  = typename KTraits::ClusterShape_MNK;  // (1,1,1)

  using CollectiveMainloop = ::pearl::CollectiveMainloopSm89<KTraits>;

  static_assert(KTraits::SkipReduction,
                "debug kernel disables hash accumulator path");
  static_assert(KTraits::SkipDenoising,
                "debug kernel disables denoise path");

  extern __shared__ char shared_memory[];
  auto& shared_storage =
      *reinterpret_cast<typename KTraits::SharedStorage*>(shared_memory);

  CollectiveMainloop collective_mainloop;

  int const k_tile_count =
      cutlass::ceil_div(get<2>(mainloop_params.problem_shape), KTraits::bK);

  __syncthreads();

  TileScheduler scheduler{};
  typename KTraits::TiledMma tiled_mma;

  int const thread_idx = threadIdx.x;
  bool local_block_found = false;
  int  block_found_k_tile = 0;

  collective_mainloop.mma_init();

  auto work_tile_info = scheduler.get_initial_work(scheduler_params);
  CUTLASS_PRAGMA_NO_UNROLL
  while (work_tile_info.is_valid(scheduler_params)) {
    // Accumulator fragment — partitioned by the SAME tiled_mma the mainloop
    // uses. Layout is (V, MMA_M, MMA_N) per thread.
    auto tCrC = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
    clear(tCrC);

    // Dummy transcript tensor — SkipReduction=true makes hash_accumulator
    // calls no-ops, but the mainloop signature still requires the symbol.
    auto transcript_extraction_tensor =
        make_tensor<uint32_t>(Int<blake3::MSG_BLOCK_SIZE_U32>{});

    cute::tuple<int32_t, int32_t, int32_t> block_coord =
        work_tile_info.template get_block_coord<ClusterShape>(scheduler_params);

    // -----------------------------------------------------------------
    // Mainloop: runs exactly as production. Fills tCrC with int32 sums.
    // -----------------------------------------------------------------
    collective_mainloop.mainloop(
        mainloop_params, shared_storage, block_coord, k_tile_count,
        tCrC, transcript_extraction_tensor,
        local_block_found, block_found_k_tile, thread_idx);

    // -----------------------------------------------------------------
    // Debug epilogue: dump tCrC int32 directly to gmem via partition_C.
    // partition_C on a gmem (bM, bN) tile gives a per-thread (V, MMA_M,
    // MMA_N) view that exactly mirrors tCrC's layout, so the store
    // collapses to one st.global per fragment element.
    // -----------------------------------------------------------------
    auto m_block = cute::get<0>(block_coord);
    auto n_block = cute::get<1>(block_coord);
    int const M = cute::get<0>(mainloop_params.problem_shape);
    int const N = cute::get<1>(mainloop_params.problem_shape);

    auto mC = make_tensor(
        make_gmem_ptr(ptr_C_i32),
        cute::make_layout(cute::make_shape(M, N),
                          cute::make_stride(ldc, cute::_1{})));
    auto gC = local_tile(mC, select<0, 1>(TileShape_MNK{}),
                          cute::make_coord(m_block, n_block));

    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    auto tCgC = thr_mma.partition_C(gC);   // (V, MMA_M, MMA_N)

    // Predicated copy: out-of-bounds tail elements skipped when M or N
    // aren't multiples of (bM, bN). Use identity coords to gate.
    auto cC = make_identity_tensor(select<0, 1>(TileShape_MNK{}));
    auto tCcC = thr_mma.partition_C(cC);

    int const residual_M = M - m_block * int(KTraits::bM);
    int const residual_N = N - n_block * int(KTraits::bN);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(tCrC); ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<2>(tCrC); ++j) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < size<0>(tCrC); ++v) {
          int const mi = cute::get<0>(tCcC(v, i, j));
          int const ni = cute::get<1>(tCcC(v, i, j));
          if (mi < residual_M && ni < residual_N) {
            tCgC(v, i, j) = tCrC(v, i, j);
          }
        }
      }
    }

    work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(
        scheduler_params, work_tile_info);
  }
}

}  // namespace pearl
