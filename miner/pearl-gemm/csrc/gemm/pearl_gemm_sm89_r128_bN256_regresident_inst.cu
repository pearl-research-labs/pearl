// SPDX-License-Identifier: see LICENSE
//
// sm_89 R=128 bM=128 bN=256 instantiation using the REGISTER-RESIDENT denoise
// epilogue (kRegisterResidentDenoise=true).
//
// Wave-3 wider-N variant of pearl_gemm_sm89_r128_regresident_inst.cu (which
// covers bM=128 bN=128 R=128). The earlier wave-2 work shipped:
//   - bM=64  bN=128 R=128 (pearl_gemm_sm89_r128_bM64_inst.cu)
//   - bM=128 bN=128 R=128 register-resident (pearl_gemm_sm89_r128_regresident_inst.cu)
// The prior bN=256 attempts (pearl_gemm_sm89_{denoise_,}inst_128x256.cu) used
// R=64 and the SMEM-RESIDENT denoise; bN=256 + R=128 + smem-resident denoise
// would have overflowed the 99 KB sm_89 opt-in cap. Register-resident denoise
// streams the four (bM/bN × R) fp16 factor tiles per-thread from gmem (L1
// cached), so the SharedStorage shape becomes the same union { A,B | C } as
// the noiseless path — freeing ~64 KB and unlocking bN=256.
//
// Smem budget (sm_89 99 KB opt-in cap):
//   bM=128 bN=256 bK=64 kStages=2 R=128 regresident
//     smem_A  = 128 * 64 * 1 B * 2 stages = 16384 B
//     smem_B  = 256 * 64 * 1 B * 2 stages = 32768 B
//     A+B     = 49152 B
//     smem_C  = 128 * 256 * 2 B            = 65536 B
//     union   = max(49152, 65536)          = 65536 B  ~64 KB
//     scales  = 128*4 + 256*4              = 1536 B
//     barriers (PipelineAsync+DenoisePipeline)        ~ 1 KB
//     TOTAL                                ≈ 66-67 KB    <-- fits
//
// Warp grid: kNumWarps = bM/16 = 8 at bM=128 (kWarpRows=2, kWarpCols=4). Per-
// warp output footprint = 64M × 64N at bN=256. The MMA atom is SM80 16x8x32
// int8, so per-warp atoms = (64/16) × (64/8) = 4 × 8 = 32 atoms per K-tile.
// Per-thread accumulator = (V=4 int32) × (MMA_M=4) × (MMA_N=8) = 128 int32
// regs in the accumulator alone, plus operand A/B fragments and ptr arithmetic
// for the per-thread register-resident denoise. Total estimate 200-230 regs at
// kStages=2; ptxas --register-usage-level=10 will spill if it exceeds 255 (we
// require 0 spills).
//
// Exposes (extern "C" trampoline for benches / pybind dispatch):
//   pearl_gemm_sm89_denoise_regresident_128x256x64_R128

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

// ============================================================================
// Denoise variant — register-resident epilogue, bM=128 bN=256 bK=64 R=128.
// SkipReduction=true (no PoW), SkipDenoising=false, kRegisterResidentDenoise=true.
// ============================================================================

using DenoiseTraits128x256x64_R128_RegRes = KernelTraitsSm89<
    /*ElementIn=*/      int8_t,
    /*ElementOut=*/     cutlass::bfloat16_t,
    /*ElementDenoise=*/ cutlass::half_t,
    /*ElementScale=*/   float,
    /*TileShape_MNKR=*/ cute::Shape<cute::Int<128>, cute::Int<256>,
                                    cute::Int<64>, cute::Int<128>>,
    /*Is_Even_M=*/      true,
    /*Is_Even_N=*/      true,
    /*cM=*/             1,
    /*cN=*/             1,
    /*SkipReduction=*/  true,
    /*SkipDenoising=*/  false,
    /*kStages=*/        2,
    /*EnableDebug=*/    false,
    /*kRegisterResidentDenoise=*/ true>;

template void pearl_gemm_sm89_run<DenoiseTraits128x256x64_R128_RegRes>(
    typename pearl::CollectiveMainloopSm89<
        DenoiseTraits128x256x64_R128_RegRes>::Arguments const&,
    typename pearl::CollectiveEpilogueSm89<
        DenoiseTraits128x256x64_R128_RegRes>::Arguments const&,
    int, int, int, cudaStream_t,
    NonceContext const*, int);

}  // namespace sm89
}  // namespace pearl

extern "C" void pearl_gemm_sm89_denoise_regresident_128x256x64_R128(
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
  using KTraits = pearl::sm89::DenoiseTraits128x256x64_R128_RegRes;
  using Mainloop = pearl::CollectiveMainloopSm89<KTraits>;
  using Epilogue = pearl::CollectiveEpilogueSm89<KTraits>;

  typename Mainloop::Arguments mainloop_args{};
  mainloop_args.ptr_A    = A;
  mainloop_args.layout_A =
      cute::make_layout(cute::make_shape(M, K), cute::make_stride(lda, cute::_1{}));
  mainloop_args.ptr_B    = B;
  mainloop_args.layout_B =
      cute::make_layout(cute::make_shape(N, K), cute::make_stride(ldb, cute::_1{}));
  mainloop_args.problem_shape = cute::make_tuple(M, N, K, /*R=*/128);
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
  epilogue_args.problem_shape = cute::make_tuple(M, N, K, /*R=*/128);

  pearl::sm89::pearl_gemm_sm89_run<KTraits>(mainloop_args, epilogue_args,
                                            M, N, K, stream);
}
