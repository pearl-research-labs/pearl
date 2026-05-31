// SPDX-License-Identifier: see LICENSE
//
// Compile- and link-time smem size probe for sm_89 R=128 variants.
// Prints sizeof(SharedStorage) for each candidate KTraits instantiation.
// This is a HOST-only program; it does not launch a kernel.
//
// Build:
//   nvcc -gencode arch=compute_89,code=sm_89 -std=c++20 -O3 \
//        -I . -I .. -I ../../third_party/cutlass/include ... \
//        _probe_smem_sizes.cu -o probe_smem_sizes
// Run on any host (no GPU needed).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "cute/tensor.hpp"

#include "kernel_traits_sm89.hpp"

#include <cstdio>

namespace pearl {
namespace sm89 {

using TraitsR128_bM64_bN64_Denoise = KernelTraitsSm89<
    int8_t, cutlass::bfloat16_t, cutlass::half_t, float,
    cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>, cute::Int<128>>,
    true, true, 1, 1, true, false, 2, false>;
using TraitsR128_bM64_bN128_Denoise = KernelTraitsSm89<
    int8_t, cutlass::bfloat16_t, cutlass::half_t, float,
    cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>, cute::Int<128>>,
    true, true, 1, 1, true, false, 2, false>;
using TraitsR128_bM64_bN64_NoDenoise = KernelTraitsSm89<
    int8_t, cutlass::bfloat16_t, cutlass::half_t, float,
    cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>, cute::Int<128>>,
    true, true, 1, 1, true, true, 2, false>;
using TraitsR128_bM64_bN128_NoDenoise = KernelTraitsSm89<
    int8_t, cutlass::bfloat16_t, cutlass::half_t, float,
    cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>, cute::Int<128>>,
    true, true, 1, 1, true, true, 2, false>;
using TraitsR64_bM128_bN128_Denoise = KernelTraitsSm89<
    int8_t, cutlass::bfloat16_t, cutlass::half_t, float,
    cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<64>, cute::Int<64>>,
    true, true, 1, 1, true, false, 3, false>;
using TraitsR128_bM64_bN128_Denoise_Pow = KernelTraitsSm89<
    int8_t, cutlass::bfloat16_t, cutlass::half_t, float,
    cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>, cute::Int<128>>,
    true, true, 1, 1, false /*SkipReduction=false: PoW*/, false, 2, false>;

// Register-resident denoise (this session) — drops the 4 (bM/bN × R)
// fp16 denoise smem buffers. The denoise epilogue streams factor rows
// from gmem (L1-cached) and computes per-thread outer products in
// registers. Unlocks R=128 at bM=bN=128.
using TraitsR128_bM128_bN128_Denoise_RegRes = KernelTraitsSm89<
    int8_t, cutlass::bfloat16_t, cutlass::half_t, float,
    cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<64>, cute::Int<128>>,
    true, true, 1, 1, true /*SkipReduction*/, false /*SkipDenoising*/, 2, false,
    true /*kRegisterResidentDenoise*/>;
using TraitsR128_bM128_bN128_Denoise_RegRes_Pow = KernelTraitsSm89<
    int8_t, cutlass::bfloat16_t, cutlass::half_t, float,
    cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<64>, cute::Int<128>>,
    true, true, 1, 1, false /*PoW*/, false, 2, false,
    true /*kRegisterResidentDenoise*/>;

}  // namespace sm89
}  // namespace pearl

int main() {
  using namespace pearl::sm89;
  printf("=== sm_89 SharedStorage sizes ===\n");
  printf("%-50s %8zu bytes  (%6.2f KB)\n",
         "R=128 bM=64  bN=64   Denoise (existing)",
         sizeof(typename TraitsR128_bM64_bN64_Denoise::SharedStorage),
         sizeof(typename TraitsR128_bM64_bN64_Denoise::SharedStorage) / 1024.0);
  printf("%-50s %8zu bytes  (%6.2f KB)\n",
         "R=128 bM=64  bN=128  Denoise (target)",
         sizeof(typename TraitsR128_bM64_bN128_Denoise::SharedStorage),
         sizeof(typename TraitsR128_bM64_bN128_Denoise::SharedStorage) / 1024.0);
  printf("%-50s %8zu bytes  (%6.2f KB)\n",
         "R=128 bM=64  bN=64   NoDenoise",
         sizeof(typename TraitsR128_bM64_bN64_NoDenoise::SharedStorage),
         sizeof(typename TraitsR128_bM64_bN64_NoDenoise::SharedStorage) / 1024.0);
  printf("%-50s %8zu bytes  (%6.2f KB)\n",
         "R=128 bM=64  bN=128  NoDenoise",
         sizeof(typename TraitsR128_bM64_bN128_NoDenoise::SharedStorage),
         sizeof(typename TraitsR128_bM64_bN128_NoDenoise::SharedStorage) / 1024.0);
  printf("%-50s %8zu bytes  (%6.2f KB)\n",
         "R=64  bM=128 bN=128  Denoise (production R=64)",
         sizeof(typename TraitsR64_bM128_bN128_Denoise::SharedStorage),
         sizeof(typename TraitsR64_bM128_bN128_Denoise::SharedStorage) / 1024.0);
  printf("%-50s %8zu bytes  (%6.2f KB)\n",
         "R=128 bM=64  bN=128  Denoise+PoW",
         sizeof(typename TraitsR128_bM64_bN128_Denoise_Pow::SharedStorage),
         sizeof(typename TraitsR128_bM64_bN128_Denoise_Pow::SharedStorage) / 1024.0);
  printf("%-50s %8zu bytes  (%6.2f KB)\n",
         "R=128 bM=128 bN=128 Denoise RegRes (new)",
         sizeof(typename TraitsR128_bM128_bN128_Denoise_RegRes::SharedStorage),
         sizeof(typename TraitsR128_bM128_bN128_Denoise_RegRes::SharedStorage) / 1024.0);
  printf("%-50s %8zu bytes  (%6.2f KB)\n",
         "R=128 bM=128 bN=128 Denoise+PoW RegRes",
         sizeof(typename TraitsR128_bM128_bN128_Denoise_RegRes_Pow::SharedStorage),
         sizeof(typename TraitsR128_bM128_bN128_Denoise_RegRes_Pow::SharedStorage) / 1024.0);
  printf("\nsm_89 opt-in cap = 99 KB (101376 bytes).\n");
  return 0;
}
