// SPDX-License-Identifier: see LICENSE
//
// sm_89 PoW instantiation at the production mining tile:
//   (bM,bN,bK,R) = (128,256,128,256), kStages=2,
//   SkipReduction=false (PoW transcript accumulator ON),
//   SkipDenoising=false (denoise epilogue ON),
//   kRegisterResidentDenoise=false (R-strip-tiled smem-resident denoise — the
//     fix for the 30x denoise slowdown). Staging all four FULL R=256 factor
//     tiles would need 4*(bM+bN)*R*sizeof(fp16) = 768 KB, far over the 99 KB
//     cap; the register-resident path avoided that but re-read every factor row
//     from gmem PER accumulator entry (~170x redundant traffic → memory-bound,
//     ~5 tmac_s). The smem-resident path now tiles R into strips of kRTile (64):
//     per strip only 4*(bM+bN)*kRTile*2 bytes are staged (unions with the
//     mainloop A/B smem under the cap) and each factor row is read from gmem
//     exactly once, with the corrections done as fp16 tensor-core MMAs.
//
// The PoW transcript accumulator (TileHashAccumulator in pow_utils.hpp) operates
// entirely on the register fragment tCrC + a small per-thread transcript tensor;
// it adds ZERO shared-memory cost vs the noiseless tile.
//
// This is the kernel benched by bench_sm89_pouw_re2.cu for the full-PoUW
// tmac_s comparison against lpminer's ~134.

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

using PowTraits128x256x128_R256 = KernelTraitsSm89<
    /*ElementIn=*/      int8_t,
    /*ElementOut=*/     cutlass::half_t,   // half_t (matches lpminer; 2B like bf16)
    /*ElementDenoise=*/ cutlass::half_t,
    /*ElementScale=*/   float,
    /*TileShape_MNKR=*/ cute::Shape<cute::Int<128>, cute::Int<256>,
                                    cute::Int<128>, cute::Int<256>>,
    /*Is_Even_M=*/      true,
    /*Is_Even_N=*/      true,
    /*cM=*/             1,
    /*cN=*/             1,
    /*SkipReduction=*/  false,   // PoW transcript accumulator ON
    /*SkipDenoising=*/  false,   // denoise epilogue ON
    /*kStages=*/        2,
    /*EnableDebug=*/    false,
    /*kRegisterResidentDenoise=*/ false>;  // R-strip-tiled smem-resident denoise
                                           // (the 30x-slowdown fix): stages each
                                           // R-strip of the four factors in smem
                                           // once and does the corrections as
                                           // fp16 tensor-core MMAs, instead of
                                           // re-reading every factor row from
                                           // gmem per accumulator entry.

template void pearl_gemm_sm89_run<PowTraits128x256x128_R256>(
    typename pearl::CollectiveMainloopSm89<
        PowTraits128x256x128_R256>::Arguments const&,
    typename pearl::CollectiveEpilogueSm89<
        PowTraits128x256x128_R256>::Arguments const&,
    int, int, int, cudaStream_t,
    NonceContext const*, int);

extern "C" void pearl_gemm_sm89_pow_128x256x128_R256(
    int8_t const* A, int64_t lda,
    int8_t const* B, int64_t ldb,
    cutlass::half_t* C, int64_t ldc,
    float const* A_scales,
    float const* B_scales,
    cutlass::half_t const* EAL,
    cutlass::half_t const* EBR,
    cutlass::half_t const* AxEBL,
    cutlass::half_t const* EARxBpEB,
    uint32_t const* pow_target,
    uint32_t const* pow_key,
    void* host_signal_sync,
    void* host_signal_header_pinned,
    uint64_t* inner_hash_counter,
    int M, int N, int K,
    cudaStream_t stream) {
  using KTraits = PowTraits128x256x128_R256;
  using Mainloop = pearl::CollectiveMainloopSm89<KTraits>;
  using Epilogue = pearl::CollectiveEpilogueSm89<KTraits>;

  typename Mainloop::Arguments mainloop_args{};
  mainloop_args.ptr_A    = A;
  mainloop_args.layout_A =
      cute::make_layout(cute::make_shape(M, K), cute::make_stride(lda, cute::_1{}));
  mainloop_args.ptr_B    = B;
  mainloop_args.layout_B =
      cute::make_layout(cute::make_shape(N, K), cute::make_stride(ldb, cute::_1{}));
  mainloop_args.problem_shape = cute::make_tuple(M, N, K, /*R=*/256);
  mainloop_args.ptr_pow_target            = pow_target;
  mainloop_args.ptr_pow_key               = pow_key;
  mainloop_args.host_signal_sync          = host_signal_sync;
  mainloop_args.host_signal_header_pinned = host_signal_header_pinned;
  mainloop_args.inner_hash_counter        = inner_hash_counter;

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
  epilogue_args.problem_shape = cute::make_tuple(M, N, K, /*R=*/256);

  pearl_gemm_sm89_run<KTraits>(mainloop_args, epilogue_args, M, N, K, stream);
}

}  // namespace sm89
}  // namespace pearl
