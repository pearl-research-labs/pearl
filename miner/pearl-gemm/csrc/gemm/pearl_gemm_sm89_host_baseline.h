// SPDX-License-Identifier: see LICENSE
//
// Baseline sm_89 host launcher — uses the original SimpleTileScheduler.
// Kept side-by-side with pearl_gemm_sm89_host.h (which now uses the new
// PersistentSwizzledTileScheduler) so we can A/B build the two configurations
// on the same hardware. Use by compiling with `-DUSE_BASELINE_SCHEDULER` and
// including this header in place of pearl_gemm_sm89_host.h.

#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

#include "kernel_traits_sm89.hpp"
#include "collective_mainloop_sm89.hpp"
#include "collective_epilogue_sm89.hpp"
#include "pearl_gemm_kernel_sm89.h"
#include "pearl_gemm_sm89_host.h"  // for SimpleTileScheduler (also defines pearl_gemm_sm89_run)

namespace pearl {
namespace sm89 {

// Baseline launcher: uses SimpleTileScheduler (naive row-major, 1 CTA per tile).
template <typename KTraits>
void pearl_gemm_sm89_run_baseline(
    typename pearl::CollectiveMainloopSm89<KTraits>::Arguments const& mainloop_args,
    typename pearl::CollectiveEpilogueSm89<KTraits>::Arguments const& epilogue_args,
    int M, int N, int /*K*/, cudaStream_t stream) {
  using Mainloop  = pearl::CollectiveMainloopSm89<KTraits>;
  using Epilogue  = pearl::CollectiveEpilogueSm89<KTraits>;
  using Scheduler = SimpleTileScheduler;

  int const num_blocks_m = (M + KTraits::bM - 1) / KTraits::bM;
  int const num_blocks_n = (N + KTraits::bN - 1) / KTraits::bN;

  typename Scheduler::Arguments sched_args{num_blocks_m, num_blocks_n};
  auto sched_params    = Scheduler::to_underlying_arguments(sched_args);
  auto mainloop_params = Mainloop::to_underlying_arguments(mainloop_args);
  auto epilogue_params = Epilogue::to_underlying_arguments(epilogue_args);

  dim3 grid  = Scheduler::get_grid_dim(sched_args, /*num_sm=*/0);
  dim3 block(KTraits::kNumThreads, 1, 1);

  size_t smem_size = sizeof(typename KTraits::SharedStorage);
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(
        (void const*)&pearl::ada_gemm<KTraits, Scheduler>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_size));
    if (e != cudaSuccess) {
      std::fprintf(stderr,
                   "pearl_gemm_sm89_run_baseline: cudaFuncSetAttribute(MaxDynamicSmem=%zu) "
                   "failed: %s\n", smem_size, cudaGetErrorString(e));
      return;
    }
    attr_set = true;
  }
  pearl::ada_gemm<KTraits, Scheduler>
      <<<grid, block, smem_size, stream>>>(mainloop_params, epilogue_params,
                                           sched_params);
}

}  // namespace sm89
}  // namespace pearl
