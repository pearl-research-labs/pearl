// SPDX-License-Identifier: see LICENSE
//
// sm_89 host launcher for the unified-warp pearl-noisingA kernel.
// Companion to pearl_noisingA_host.h (Hopper). Uses plain `<<<>>>` launch
// (no `cutlass::device_kernel`, no TMA descriptors, no cluster attrs).

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

// Avoid pulling in c10/torch (error_check.hpp). Standalone tests can include
// this header without a PyTorch build.
#include "pearl_api_params.h"
#include "pearl_noisingA_kernel_sm89.h"

#define PEARL_SM89_CUDA_CHECK(x)                                          \
  do {                                                                    \
    cudaError_t _e = (x);                                                 \
    if (_e != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__,         \
                   cudaGetErrorString(_e));                               \
      std::abort();                                                       \
    }                                                                     \
  } while (0)

namespace pearl {

// __global__ trampoline: __launch_bounds__ provides the per-CTA register cap
// in lieu of warpgroup_reg_alloc/dealloc on sm_90.
template <typename Kernel>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock,
                             Kernel::MinBlocksPerMultiprocessor)
void pearl_noisingA_sm89_global(
    CUTE_GRID_CONSTANT typename Kernel::Params const params) {
  extern __shared__ char smem_buf[];
  Kernel kernel;
  kernel(params, smem_buf);
}

}  // namespace pearl

template <class ElementDenoise_AxEBL, class TileShape_MRK, int kStages,
          bool IsEvenK = false>
void run_pearl_noising_A_sm89(PearlAPIParams const& params,
                              cudaStream_t stream = 0) {
  using namespace cute;
  using Element        = int8_t;
  using ElementDenoise = ElementDenoise_AxEBL;
  // sm_89 always uses NoReduction=true: one CTA per m_block processes the full
  // K dimension. The kernel's static_assert (denoise_dtype_bits == 32 ||
  // NoReduction) requires NoReduction=true for fp16; for int32 the sm_89 port
  // does not implement the cross-CTA atomic-add reduction path either, so the
  // grid is also 1D-over-M with full-K accumulation in registers.
  static constexpr bool NoReduction = true;

  using NoisingKernel = pearl::NoisingKernelASm89<
      TileShape_MRK, /*kNumThreads=*/128, Element, ElementDenoise, kStages,
      IsEvenK, NoReduction>;

  int total_k_blocks = ceil_div(params.k, get<2>(TileShape_MRK{}));
  bool no_reduce = NoReduction || params.k_blocks_per_split_noising_A <= 0;
  typename NoisingKernel::Arguments args{
      .ptr_A     = static_cast<Element const*>(params.ptr_A),
      .ptr_EAL   = static_cast<Element const*>(params.ptr_EAL),
      .ptr_EAR   = static_cast<Element const*>(params.ptr_EAR_R_major),
      .ptr_EBL   = static_cast<Element const*>(params.ptr_EBL_K_major),
      .ptr_A_out = static_cast<Element*>(params.ptr_ApEA),
      .ptr_AxEBL = static_cast<ElementDenoise*>(params.ptr_AxEBL),
      .m         = params.m,
      .k         = params.k,
      .num_k_blocks =
          no_reduce ? total_k_blocks : params.k_blocks_per_split_noising_A,
      .total_k_blocks = total_k_blocks};

  typename NoisingKernel::Params kernel_params =
      NoisingKernel::to_underlying_arguments(args);

  dim3 grid  = NoisingKernel::get_grid_shape(kernel_params);
  dim3 block = NoisingKernel::get_block_shape();
  constexpr int smem_size = NoisingKernel::SharedStorageSize;

  if (smem_size >= 48 * 1024) {
    PEARL_SM89_CUDA_CHECK(cudaFuncSetAttribute(
        (void const*)&pearl::pearl_noisingA_sm89_global<NoisingKernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }

  pearl::pearl_noisingA_sm89_global<NoisingKernel>
      <<<grid, block, smem_size, stream>>>(kernel_params);
  PEARL_SM89_CUDA_CHECK(cudaGetLastError());
}
