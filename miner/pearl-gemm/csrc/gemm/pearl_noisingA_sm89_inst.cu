// SPDX-License-Identifier: see LICENSE
//
// sm_89 instantiations for noisingA.
//   R=64:  (bM,R,bK) = (64,64,64), kStages=2, {int32, fp16}, IsEvenK=true.
//   R=128: (bM,R,bK) = (64,128,64), kStages=2, {int32, fp16}, IsEvenK=true.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "cute/tensor.hpp"
#include <cutlass/numeric_types.h>

#include "pearl_noisingA_kernel_sm89.h"
#include "pearl_noisingA_sm89_host.h"

namespace pearl {

using NoisingKernelA64x64x64_R64_int32 = NoisingKernelASm89<
    /*TileShape_MRK=*/ cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>,
    /*kNumThreads=*/   128,
    /*Element=*/       int8_t,
    /*ElementDenoise=*/int32_t,
    /*kStages=*/       2,
    /*IsEvenK=*/       true,
    /*NoReduction=*/   true>;

using NoisingKernelA64x128x64_R128_int32 = NoisingKernelASm89<
    /*TileShape_MRK=*/ cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>,
    /*kNumThreads=*/   128,
    /*Element=*/       int8_t,
    /*ElementDenoise=*/int32_t,
    /*kStages=*/       2,
    /*IsEvenK=*/       true,
    /*NoReduction=*/   true>;

// fp16 variants — same shape, scaling-cast epilogue (line 597-612 of
// pearl_noisingA_kernel_sm89.h divides accumulator by kAxEBLScaleFactor and
// stores as ElementDenoise). NoReduction=true required by the kernel's
// static_assert for the 16-bit denoise dtype.
using NoisingKernelA64x64x64_R64_fp16 = NoisingKernelASm89<
    /*TileShape_MRK=*/ cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>,
    /*kNumThreads=*/   128,
    /*Element=*/       int8_t,
    /*ElementDenoise=*/cutlass::half_t,
    /*kStages=*/       2,
    /*IsEvenK=*/       true,
    /*NoReduction=*/   true>;

using NoisingKernelA64x128x64_R128_fp16 = NoisingKernelASm89<
    /*TileShape_MRK=*/ cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>,
    /*kNumThreads=*/   128,
    /*Element=*/       int8_t,
    /*ElementDenoise=*/cutlass::half_t,
    /*kStages=*/       2,
    /*IsEvenK=*/       true,
    /*NoReduction=*/   true>;

}  // namespace pearl

// Explicit instantiations of the dispatch wrapper (global namespace, matches
// pearl_noisingA_sm89_host.h:48). Forces nvcc to emit the device kernel and
// register it with the CUDA runtime so cudaFuncSetAttribute can find it.
template void run_pearl_noising_A_sm89<int32_t,
                                       cute::Shape<cute::Int<64>, cute::Int<128>,
                                                   cute::Int<64>>,
                                       /*kStages=*/2,
                                       /*IsEvenK=*/false>(
    PearlAPIParams const&, cudaStream_t);
template void run_pearl_noising_A_sm89<int32_t,
                                       cute::Shape<cute::Int<64>, cute::Int<128>,
                                                   cute::Int<64>>,
                                       /*kStages=*/2,
                                       /*IsEvenK=*/true>(
    PearlAPIParams const&, cudaStream_t);
// fp16 variants (R=64 and R=128, both IsEvenK polarities) so cudaFuncSetAttribute
// finds the device kernel symbol regardless of which (R, k_mod) path the runtime
// dispatch picks.
template void run_pearl_noising_A_sm89<cutlass::half_t,
                                       cute::Shape<cute::Int<64>, cute::Int<64>,
                                                   cute::Int<64>>,
                                       /*kStages=*/2,
                                       /*IsEvenK=*/false>(
    PearlAPIParams const&, cudaStream_t);
template void run_pearl_noising_A_sm89<cutlass::half_t,
                                       cute::Shape<cute::Int<64>, cute::Int<64>,
                                                   cute::Int<64>>,
                                       /*kStages=*/2,
                                       /*IsEvenK=*/true>(
    PearlAPIParams const&, cudaStream_t);
template void run_pearl_noising_A_sm89<cutlass::half_t,
                                       cute::Shape<cute::Int<64>, cute::Int<128>,
                                                   cute::Int<64>>,
                                       /*kStages=*/2,
                                       /*IsEvenK=*/false>(
    PearlAPIParams const&, cudaStream_t);
template void run_pearl_noising_A_sm89<cutlass::half_t,
                                       cute::Shape<cute::Int<64>, cute::Int<128>,
                                                   cute::Int<64>>,
                                       /*kStages=*/2,
                                       /*IsEvenK=*/true>(
    PearlAPIParams const&, cudaStream_t);

namespace {
template <typename Kernel>
void launch_noisingA_R64_or_R128(
    int8_t const* A, int8_t const* EAL, int8_t const* EAR, int8_t const* EBL,
    int8_t* ApEA, int32_t* AxEBL,
    int M, int K, cudaStream_t stream) {

  typename Kernel::Arguments args{
      .ptr_A          = A,
      .ptr_EAL        = EAL,
      .ptr_EAR        = EAR,
      .ptr_EBL        = EBL,
      .ptr_A_out      = ApEA,
      .ptr_AxEBL      = AxEBL,
      .m              = M,
      .k              = K,
      .num_k_blocks   = (K + Kernel::kBlockK - 1) / Kernel::kBlockK,
      .total_k_blocks = (K + Kernel::kBlockK - 1) / Kernel::kBlockK};

  typename Kernel::Params kernel_params = Kernel::to_underlying_arguments(args);

  dim3 grid  = Kernel::get_grid_shape(kernel_params);
  dim3 block = Kernel::get_block_shape();
  constexpr int smem_size = Kernel::SharedStorageSize;

  if (smem_size >= 48 * 1024) {
    cudaError_t e = cudaFuncSetAttribute(
        (void const*)&pearl::pearl_noisingA_sm89_global<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (e != cudaSuccess) {
      std::fprintf(stderr, "cudaFuncSetAttribute(MaxDynamicSmem=%d) failed: %s\n",
                   smem_size, cudaGetErrorString(e));
      std::abort();
    }
  }

  pearl::pearl_noisingA_sm89_global<Kernel>
      <<<grid, block, smem_size, stream>>>(kernel_params);
}
}  // namespace

extern "C" void pearl_noisingA_sm89_64x64x64_R64_int32(
    int8_t const* A, int8_t const* EAL, int8_t const* EAR, int8_t const* EBL,
    int8_t* ApEA, int32_t* AxEBL,
    int M, int K, cudaStream_t stream) {
  launch_noisingA_R64_or_R128<pearl::NoisingKernelA64x64x64_R64_int32>(
      A, EAL, EAR, EBL, ApEA, AxEBL, M, K, stream);
}

extern "C" void pearl_noisingA_sm89_64x128x64_R128_int32(
    int8_t const* A, int8_t const* EAL, int8_t const* EAR, int8_t const* EBL,
    int8_t* ApEA, int32_t* AxEBL,
    int M, int K, cudaStream_t stream) {
  launch_noisingA_R64_or_R128<pearl::NoisingKernelA64x128x64_R128_int32>(
      A, EAL, EAR, EBL, ApEA, AxEBL, M, K, stream);
}
