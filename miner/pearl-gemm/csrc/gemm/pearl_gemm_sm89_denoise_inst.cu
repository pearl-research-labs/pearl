// SPDX-License-Identifier: see LICENSE
//
// sm_89 single-config instantiation: int8 GEMM with fp16 denoise epilogue,
// (bM,bN,bK,R) = (128,128,64,64), kStages=3, SkipDenoising=false.
// Companion to pearl_gemm_sm89_inst.cu (noiseless path).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "cute/tensor.hpp"

#include "kernel_traits_sm89.hpp"
#include "collective_mainloop_sm89.hpp"
#include "collective_epilogue_sm89.hpp"
#include "pearl_gemm_kernel_sm89.h"
#include "pearl_gemm_sm89_host.h"

namespace pearl {
namespace sm89 {

// Same tile shape as the noiseless instantiation but SkipDenoising=false.
// SharedStorageDenoise overlaps the (A, B) tile memory with the four denoise
// factor smem buffers, so the marginal smem cost is just smem_C + scales +
// barriers (which fit under sm_89's 100 KB opt-in cap).
using DenoiseTraits128x128x64_R64 = KernelTraitsSm89<
    /*ElementIn=*/      int8_t,
    /*ElementOut=*/     cutlass::bfloat16_t,
    /*ElementDenoise=*/ cutlass::half_t,
    /*ElementScale=*/   float,
    /*TileShape_MNKR=*/ cute::Shape<cute::Int<128>, cute::Int<128>,
                                    cute::Int<64>,  cute::Int<64>>,
    /*Is_Even_M=*/      true,
    /*Is_Even_N=*/      true,
    /*cM=*/             1,
    /*cN=*/             1,
    /*SkipReduction=*/  true,
    /*SkipDenoising=*/  false,
    /*kStages=*/        3,
    /*EnableDebug=*/    false>;

template void pearl_gemm_sm89_run<DenoiseTraits128x128x64_R64>(
    typename pearl::CollectiveMainloopSm89<
        DenoiseTraits128x128x64_R64>::Arguments const& mainloop_args,
    typename pearl::CollectiveEpilogueSm89<
        DenoiseTraits128x128x64_R64>::Arguments const& epilogue_args,
    int M, int N, int K, cudaStream_t stream,
    NonceContext const* ptr_nonce_contexts,
    int nonce_batch_size);

extern "C" void pearl_gemm_sm89_denoise_128x128x64_R64(
    int8_t const* A, int64_t lda,
    int8_t const* B, int64_t ldb,
    cutlass::bfloat16_t* C, int64_t ldc,
    float const* A_scales,
    float const* B_scales,
    cutlass::half_t const* EAL,
    cutlass::half_t const* EBR,
    cutlass::half_t const* AxEBL,
    cutlass::half_t const* EARxBpEB,
    int M, int N, int K,
    cudaStream_t stream) {
  using KTraits = DenoiseTraits128x128x64_R64;
  using Mainloop = pearl::CollectiveMainloopSm89<KTraits>;
  using Epilogue = pearl::CollectiveEpilogueSm89<KTraits>;

  typename Mainloop::Arguments mainloop_args{};
  mainloop_args.ptr_A    = A;
  mainloop_args.layout_A =
      cute::make_layout(cute::make_shape(M, K), cute::make_stride(lda, cute::_1{}));
  mainloop_args.ptr_B    = B;
  mainloop_args.layout_B =
      cute::make_layout(cute::make_shape(N, K), cute::make_stride(ldb, cute::_1{}));
  mainloop_args.problem_shape = cute::make_tuple(M, N, K, /*R=*/64);
  mainloop_args.ptr_pow_target            = nullptr;
  mainloop_args.ptr_pow_key               = nullptr;
  mainloop_args.host_signal_sync          = nullptr;
  mainloop_args.host_signal_header_pinned = nullptr;
  mainloop_args.inner_hash_counter        = nullptr;

  typename Epilogue::Arguments epilogue_args{};
  epilogue_args.ptr_C   = C;
  epilogue_args.layout_C =
      cute::make_layout(cute::make_shape(M, N), cute::make_stride(ldc, cute::_1{}));
  epilogue_args.ptr_A_scales = A_scales;
  epilogue_args.ptr_B_scales = B_scales;
  epilogue_args.ptr_EAL      = EAL;
  epilogue_args.ptr_EARxBpEB = EARxBpEB;
  epilogue_args.ptr_AxEBL    = AxEBL;
  epilogue_args.ptr_EBR      = EBR;
  epilogue_args.problem_shape = cute::make_tuple(M, N, K, /*R=*/64);

  pearl_gemm_sm89_run<KTraits>(mainloop_args, epilogue_args, M, N, K, stream);
}

}  // namespace sm89
}  // namespace pearl

// ============================================================================
// R=128 variant — bM=bN=64, bK=64, kStages=2. Used by alphapool/R=128 driver.
// SkipReduction=true (no PoW accumulator), SkipDenoising=false (denoise epilogue ON).
// ============================================================================

namespace pearl {
namespace sm89 {

using DenoiseTraits64x64x64_R128 = KernelTraitsSm89<
    /*ElementIn=*/      int8_t,
    /*ElementOut=*/     cutlass::bfloat16_t,
    /*ElementDenoise=*/ cutlass::half_t,
    /*ElementScale=*/   float,
    /*TileShape_MNKR=*/ cute::Shape<cute::Int<64>, cute::Int<64>,
                                    cute::Int<64>, cute::Int<128>>,
    /*Is_Even_M=*/      true,
    /*Is_Even_N=*/      true,
    /*cM=*/             1,
    /*cN=*/             1,
    /*SkipReduction=*/  true,
    /*SkipDenoising=*/  false,
    /*kStages=*/        2,
    /*EnableDebug=*/    false>;

template void pearl_gemm_sm89_run<DenoiseTraits64x64x64_R128>(
    typename pearl::CollectiveMainloopSm89<
        DenoiseTraits64x64x64_R128>::Arguments const&,
    typename pearl::CollectiveEpilogueSm89<
        DenoiseTraits64x64x64_R128>::Arguments const&,
    int, int, int, cudaStream_t,
    NonceContext const*, int);

}  // namespace sm89
}  // namespace pearl
