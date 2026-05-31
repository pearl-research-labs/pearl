// SPDX-License-Identifier: see LICENSE
//
// sm_89 host launcher for pearl_noising_B. Companion to pearl_noisingB_host.h.

#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include "cute/tensor.hpp"

#include "pearl_noisingB_kernel_sm89.h"

namespace pearl {
namespace sm89 {

// Top-level device kernel that drives a single NoisingKernelBSm89 instance.
// Matches the `cutlass::device_kernel` indirection used by the Hopper host.
template <typename Kernel>
__global__ void __launch_bounds__(Kernel::MaxThreadsPerBlock,
                                  Kernel::MinBlocksPerMultiprocessor)
device_kernel_noisingB_sm89(typename Kernel::Params const params) {
  extern __shared__ char smem[];
  Kernel kernel;
  kernel(params, smem);
}

template <typename ElementDenoise_EARxBpEB, class TileShape_NRK, int kStages,
          bool IsEvenK = false>
void run_pearl_noising_B_sm89(int8_t const* ptr_B, int8_t const* ptr_EBR,
                              int8_t const* ptr_EBL, int8_t const* ptr_EAR,
                              int8_t* ptr_BpEB,
                              ElementDenoise_EARxBpEB* ptr_EARxBpEB,
                              int n, int k, int /*k_blocks_per_split*/,
                              cudaStream_t stream = 0) {
  using namespace cute;
  using Element = int8_t;
  using ElementDenoise = ElementDenoise_EARxBpEB;
  // For sm_89 we always use the NoReduction path (int32 case): one CTA per
  // n_block processes the full k dimension. The k_blocks_per_split parameter
  // is accepted for API compatibility with the Hopper launcher but unused.
  using Kernel = NoisingKernelBSm89<TileShape_NRK, /*kNumThreads=*/128, Element,
                                    ElementDenoise, kStages, IsEvenK,
                                    /*NoReduction=*/true>;

  int total_k_blocks = cutlass::ceil_div(k, get<2>(TileShape_NRK{}));

  typename Kernel::Arguments args{
      .ptr_B = ptr_B,
      .ptr_EBR = ptr_EBR,
      .ptr_EAR = ptr_EAR,
      .ptr_EBL = ptr_EBL,
      .ptr_BpEB = ptr_BpEB,
      .ptr_EARxBpEB = ptr_EARxBpEB,
      .n = n,
      .k = k,
      .num_k_blocks = total_k_blocks,
      .total_k_blocks = total_k_blocks};

  typename Kernel::Params kernel_params = Kernel::to_underlying_arguments(args);

  dim3 grid = Kernel::get_grid_shape(kernel_params);
  dim3 block = Kernel::get_block_shape();
  constexpr int smem_size = Kernel::SharedStorageSize;

  auto kernel_fn = device_kernel_noisingB_sm89<Kernel>;
  if (smem_size >= 48 * 1024) {
    cudaError_t e = cudaFuncSetAttribute(
        (void const*)&device_kernel_noisingB_sm89<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (e != cudaSuccess) {
      std::fprintf(
          stderr,
          "run_pearl_noising_B_sm89: cudaFuncSetAttribute(MaxDynamicSmem=%d) "
          "failed: %s\n",
          smem_size, cudaGetErrorString(e));
      return;
    }
  }
  kernel_fn<<<grid, block, smem_size, stream>>>(kernel_params);
}

}  // namespace sm89
}  // namespace pearl
