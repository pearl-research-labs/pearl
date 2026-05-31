// SPDX-License-Identifier: see LICENSE
//
// Debug host launcher: invokes ada_gemm_debug_int32 — same mainloop as
// production but writes raw int32 tCrC accumulator to gmem, no epilogue.
//
// Mirrors pearl_gemm_sm89_host.h's launch path (smem opt-in attribute, grid
// dims from SimpleTileScheduler) but with the debug kernel symbol.

#pragma once

#include <cstdio>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

#include "kernel_traits_sm89.hpp"
#include "collective_mainloop_sm89.hpp"
#include "pearl_gemm_kernel_sm89_debug.h"
#include "pearl_gemm_sm89_host.h"  // for SimpleTileScheduler

namespace pearl {
namespace sm89 {

template <typename KTraits>
void pearl_gemm_sm89_debug_run(
    typename pearl::CollectiveMainloopSm89<KTraits>::Arguments const& mainloop_args,
    int32_t* ptr_C_i32, int64_t ldc,
    int M, int N, int /*K*/, cudaStream_t stream) {
  using Mainloop  = pearl::CollectiveMainloopSm89<KTraits>;
  using Scheduler = SimpleTileScheduler;

  int const num_blocks_m = (M + KTraits::bM - 1) / KTraits::bM;
  int const num_blocks_n = (N + KTraits::bN - 1) / KTraits::bN;

  Scheduler::Arguments sched_args{num_blocks_m, num_blocks_n};
  auto sched_params    = Scheduler::to_underlying_arguments(sched_args);
  auto mainloop_params = Mainloop::to_underlying_arguments(mainloop_args);

  dim3 grid  = Scheduler::get_grid_dim(sched_args, /*num_sm=*/0);
  dim3 block(KTraits::kNumThreads, 1, 1);

  size_t smem_size = sizeof(typename KTraits::SharedStorage);
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(
        (void const*)&pearl::ada_gemm_debug_int32<KTraits, Scheduler>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_size));
    if (e != cudaSuccess) {
      std::fprintf(stderr,
                   "pearl_gemm_sm89_debug_run: cudaFuncSetAttribute "
                   "(MaxDynamicSmem=%zu) failed: %s\n",
                   smem_size, cudaGetErrorString(e));
      return;
    }
    attr_set = true;
  }

  pearl::ada_gemm_debug_int32<KTraits, Scheduler>
      <<<grid, block, smem_size, stream>>>(mainloop_params, ptr_C_i32, ldc,
                                           sched_params);
}

}  // namespace sm89
}  // namespace pearl
