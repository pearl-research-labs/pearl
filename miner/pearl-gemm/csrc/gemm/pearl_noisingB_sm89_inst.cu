// SPDX-License-Identifier: see LICENSE
//
// sm_89 instantiations of noisingB:
//   R=64:  TileShape_NRK = (64, 64, 64), kStages = 2, {int32, fp16}, IsEvenK = true.
//   R=128: TileShape_NRK = (64, 128, 64), kStages = 2, {int32, fp16}, IsEvenK = true.

#include <cuda_runtime.h>
#include "cute/tensor.hpp"
#include <cutlass/numeric_types.h>

#include "pearl_noisingB_kernel_sm89.h"
#include "pearl_noisingB_sm89_host.h"

namespace pearl {
namespace sm89 {

// R=64 int32 explicit instantiation.
template void run_pearl_noising_B_sm89<int32_t,
                                       cute::Shape<cute::Int<64>, cute::Int<64>,
                                                   cute::Int<64>>,
                                       /*kStages=*/2,
                                       /*IsEvenK=*/true>(
    int8_t const* ptr_B, int8_t const* ptr_EBR, int8_t const* ptr_EBL,
    int8_t const* ptr_EAR, int8_t* ptr_BpEB, int32_t* ptr_EARxBpEB, int n,
    int k, int k_blocks_per_split, cudaStream_t stream);

// R=128 int32 explicit instantiation — required for the pybind dispatch path
// so the device kernel + cudaFuncSetAttribute registration get wired in. The
// instantiations/noisingB_R128_int32_64x64_2stages.cu file only instantiates
// the outer run_pearl_noising_B_ wrapper which doesn't transitively register
// the device kernel with the CUDA runtime; we have to do it here.
template void run_pearl_noising_B_sm89<int32_t,
                                       cute::Shape<cute::Int<64>, cute::Int<128>,
                                                   cute::Int<64>>,
                                       /*kStages=*/2,
                                       /*IsEvenK=*/true>(
    int8_t const* ptr_B, int8_t const* ptr_EBR, int8_t const* ptr_EBL,
    int8_t const* ptr_EAR, int8_t* ptr_BpEB, int32_t* ptr_EARxBpEB, int n,
    int k, int k_blocks_per_split, cudaStream_t stream);

// fp16 instantiations — same shape as int32 variants, scaling-cast epilogue
// (lines 575-589 of pearl_noisingB_kernel_sm89.h divides accumulator by
// kEARxBpEBScaleFactor and stores as ElementDenoise). NoReduction is
// hardcoded true inside the host (see pearl_noisingB_sm89_host.h), which is
// required by the kernel's `denoise_dtype_bits == 32 || NoReduction`
// static_assert for the 16-bit denoise dtype. Both IsEvenK polarities so
// runtime dispatch on (k % bK == 0) lands on a compiled instance.
template void run_pearl_noising_B_sm89<cutlass::half_t,
                                       cute::Shape<cute::Int<64>, cute::Int<64>,
                                                   cute::Int<64>>,
                                       /*kStages=*/2,
                                       /*IsEvenK=*/false>(
    int8_t const* ptr_B, int8_t const* ptr_EBR, int8_t const* ptr_EBL,
    int8_t const* ptr_EAR, int8_t* ptr_BpEB, cutlass::half_t* ptr_EARxBpEB,
    int n, int k, int k_blocks_per_split, cudaStream_t stream);
template void run_pearl_noising_B_sm89<cutlass::half_t,
                                       cute::Shape<cute::Int<64>, cute::Int<64>,
                                                   cute::Int<64>>,
                                       /*kStages=*/2,
                                       /*IsEvenK=*/true>(
    int8_t const* ptr_B, int8_t const* ptr_EBR, int8_t const* ptr_EBL,
    int8_t const* ptr_EAR, int8_t* ptr_BpEB, cutlass::half_t* ptr_EARxBpEB,
    int n, int k, int k_blocks_per_split, cudaStream_t stream);
template void run_pearl_noising_B_sm89<cutlass::half_t,
                                       cute::Shape<cute::Int<64>, cute::Int<128>,
                                                   cute::Int<64>>,
                                       /*kStages=*/2,
                                       /*IsEvenK=*/false>(
    int8_t const* ptr_B, int8_t const* ptr_EBR, int8_t const* ptr_EBL,
    int8_t const* ptr_EAR, int8_t* ptr_BpEB, cutlass::half_t* ptr_EARxBpEB,
    int n, int k, int k_blocks_per_split, cudaStream_t stream);
template void run_pearl_noising_B_sm89<cutlass::half_t,
                                       cute::Shape<cute::Int<64>, cute::Int<128>,
                                                   cute::Int<64>>,
                                       /*kStages=*/2,
                                       /*IsEvenK=*/true>(
    int8_t const* ptr_B, int8_t const* ptr_EBR, int8_t const* ptr_EBL,
    int8_t const* ptr_EAR, int8_t* ptr_BpEB, cutlass::half_t* ptr_EARxBpEB,
    int n, int k, int k_blocks_per_split, cudaStream_t stream);

// C-callable trampoline (R=64, kept for standalone test).
extern "C" void pearl_noisingB_sm89_64x64x64_R64_int32(
    int8_t const* B, int8_t const* EBR, int8_t const* EBL, int8_t const* EAR,
    int8_t* BpEB, int32_t* EARxBpEB, int N, int K, cudaStream_t stream) {
  using TileShape_NRK = cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>;
  run_pearl_noising_B_sm89<int32_t, TileShape_NRK, /*kStages=*/2,
                           /*IsEvenK=*/true>(B, EBR, EBL, EAR, BpEB, EARxBpEB,
                                              N, K, /*k_blocks_per_split=*/-1,
                                              stream);
}

// C-callable trampoline (R=128).
extern "C" void pearl_noisingB_sm89_64x128x64_R128_int32(
    int8_t const* B, int8_t const* EBR, int8_t const* EBL, int8_t const* EAR,
    int8_t* BpEB, int32_t* EARxBpEB, int N, int K, cudaStream_t stream) {
  using TileShape_NRK = cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>;
  run_pearl_noising_B_sm89<int32_t, TileShape_NRK, /*kStages=*/2,
                           /*IsEvenK=*/true>(B, EBR, EBL, EAR, BpEB, EARxBpEB,
                                              N, K, /*k_blocks_per_split=*/-1,
                                              stream);
}

}  // namespace sm89
}  // namespace pearl
