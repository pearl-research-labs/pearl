// SPDX-License-Identifier: see LICENSE
//
// sm_89 R=128 bM=128 bN=128 instantiation using the REGISTER-RESIDENT denoise
// epilogue (kRegisterResidentDenoise=true).
//
// This is the path that mirrors alpha-miner's R=128 fit on sm_89 (see
// C:/Source/pearl-investigation/alpha_r128_denoise_re_2026_05_17.md). The
// four (bM/bN × R) fp16 denoise tiles are NOT staged in shared memory —
// they are streamed per-thread directly from gmem (cached in L1) inside the
// denoise epilogue. This frees ~128 KB at (bM=128, bN=128, R=128) that the
// smem-resident path would otherwise overflow with.
//
// Smem budget (sm_89 99 KB opt-in cap), target rows of `_probe_smem_sizes`:
//   variant                                          SharedStorage
//   ────────────────────────────────────────────     ─────────────
//   R=128 bM=128 bN=128 Denoise smem-resident        ~131 KB (over cap, can't build)
//   R=128 bM=128 bN=128 Denoise register-resident    ~33 KB (this file)
//   R=128 bM=128 bN=128 NoDenoise (ref)              ~33 KB
//
// Warp grid: kernel_traits_sm89.hpp derives kNumWarps = bM/16 = 8 for bM=128
// (kWarpRows=2, kWarpCols=4; per-warp 64M × 32N footprint).
//
// Exposes (extern "C" trampoline for benches / pybind dispatch):
//   pearl_gemm_sm89_denoise_regresident_128x128x64_R128

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
// Denoise variant — register-resident epilogue, bM=128 bN=128 bK=64 R=128.
// SkipReduction=true (no PoW), SkipDenoising=false, kRegisterResidentDenoise=true.
// ============================================================================

using DenoiseTraits128x128x64_R128_RegRes = KernelTraitsSm89<
    /*ElementIn=*/      int8_t,
    /*ElementOut=*/     cutlass::bfloat16_t,
    /*ElementDenoise=*/ cutlass::half_t,
    /*ElementScale=*/   float,
    /*TileShape_MNKR=*/ cute::Shape<cute::Int<128>, cute::Int<128>,
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

template void pearl_gemm_sm89_run<DenoiseTraits128x128x64_R128_RegRes>(
    typename pearl::CollectiveMainloopSm89<
        DenoiseTraits128x128x64_R128_RegRes>::Arguments const&,
    typename pearl::CollectiveEpilogueSm89<
        DenoiseTraits128x128x64_R128_RegRes>::Arguments const&,
    int, int, int, cudaStream_t,
    NonceContext const*, int);

}  // namespace sm89
}  // namespace pearl

extern "C" void pearl_gemm_sm89_denoise_regresident_128x128x64_R128(
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
  using KTraits = pearl::sm89::DenoiseTraits128x128x64_R128_RegRes;
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
