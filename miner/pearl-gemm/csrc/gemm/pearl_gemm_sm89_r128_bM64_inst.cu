// SPDX-License-Identifier: see LICENSE
//
// sm_89 R=128 bM=64 instantiations — alphapool/R=128 production path.
//
// Adds the high-throughput bM=64,bN=128 variant alongside the existing
// bM=bN=64 baseline (which lives in pearl_gemm_sm89_inst.cu /
// _denoise_inst.cu / _pow_inst.cu).
//
// Smem budget (sm_89 99 KB opt-in cap, validated via _probe_smem_sizes.cu):
//
//   variant                                   SharedStorage
//   ────────────────────────────────────────  ─────────────
//   R=128 bM=64 bN=64  Denoise (existing)     65.0 KB
//   R=128 bM=64 bN=128 Denoise (this file)    97.0 KB  <- 2 KB headroom
//   R=128 bM=64 bN=128 NoDenoise              25.0 KB
//   R=128 bM=64 bN=128 Denoise+PoW            97.0 KB
//
// Warp grid: kernel_traits_sm89.hpp derives kNumWarps = bM/16 = 4 for bM=64
// (kWarpRows=2, kWarpCols=2; atom group footprint 32M × 16N replicated to
// cover 64M × 128N for this file's bN=128 variants). S2GValueLayoutC at
// bN=128 = (16, 4) elements/thread; G2S thread layout = (32, 4) for bK=64.
//
// Exposes (extern "C" trampolines used by the standalone benches):
//   pearl_gemm_sm89_noiseless_64x128x64_R128
//   pearl_gemm_sm89_denoise_64x128x64_R128
//   pearl_gemm_sm89_pow_64x128x64_R128

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
// Noiseless variant: SkipReduction=true, SkipDenoising=true. No PoW accum,
// no denoise epilogue. Used by bench_sm89_standalone-style throughput probes.
// ============================================================================

using NoiselessTraits64x128x64_R128 = KernelTraitsSm89<
    /*ElementIn=*/      int8_t,
    /*ElementOut=*/     cutlass::bfloat16_t,
    /*ElementDenoise=*/ cutlass::half_t,
    /*ElementScale=*/   float,
    /*TileShape_MNKR=*/ cute::Shape<cute::Int<64>, cute::Int<128>,
                                    cute::Int<64>, cute::Int<128>>,
    /*Is_Even_M=*/      true,
    /*Is_Even_N=*/      true,
    /*cM=*/             1,
    /*cN=*/             1,
    /*SkipReduction=*/  true,
    /*SkipDenoising=*/  true,
    /*kStages=*/        2,
    /*EnableDebug=*/    false>;

template void pearl_gemm_sm89_run<NoiselessTraits64x128x64_R128>(
    typename pearl::CollectiveMainloopSm89<
        NoiselessTraits64x128x64_R128>::Arguments const&,
    typename pearl::CollectiveEpilogueSm89<
        NoiselessTraits64x128x64_R128>::Arguments const&,
    int, int, int, cudaStream_t,
    NonceContext const*, int);

extern "C" void pearl_gemm_sm89_noiseless_64x128x64_R128(
    int8_t const* A, int64_t lda,
    int8_t const* B, int64_t ldb,
    cutlass::bfloat16_t* C, int64_t ldc,
    float const* A_scales,
    float const* B_scales,
    int M, int N, int K,
    cudaStream_t stream) {
  using KTraits = NoiselessTraits64x128x64_R128;
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
  epilogue_args.problem_shape = cute::make_tuple(M, N, K, /*R=*/128);

  pearl_gemm_sm89_run<KTraits>(mainloop_args, epilogue_args, M, N, K, stream);
}

// ============================================================================
// Denoise variant: SkipReduction=true, SkipDenoising=false. Denoise epilogue
// ON (consumes EAL/EBR/AxEBL/EARxBpEB fp16 tiles).
// ============================================================================

using DenoiseTraits64x128x64_R128 = KernelTraitsSm89<
    /*ElementIn=*/      int8_t,
    /*ElementOut=*/     cutlass::bfloat16_t,
    /*ElementDenoise=*/ cutlass::half_t,
    /*ElementScale=*/   float,
    /*TileShape_MNKR=*/ cute::Shape<cute::Int<64>, cute::Int<128>,
                                    cute::Int<64>, cute::Int<128>>,
    /*Is_Even_M=*/      true,
    /*Is_Even_N=*/      true,
    /*cM=*/             1,
    /*cN=*/             1,
    /*SkipReduction=*/  true,
    /*SkipDenoising=*/  false,
    /*kStages=*/        2,
    /*EnableDebug=*/    false>;

template void pearl_gemm_sm89_run<DenoiseTraits64x128x64_R128>(
    typename pearl::CollectiveMainloopSm89<
        DenoiseTraits64x128x64_R128>::Arguments const&,
    typename pearl::CollectiveEpilogueSm89<
        DenoiseTraits64x128x64_R128>::Arguments const&,
    int, int, int, cudaStream_t,
    NonceContext const*, int);

extern "C" void pearl_gemm_sm89_denoise_64x128x64_R128(
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
  using KTraits = DenoiseTraits64x128x64_R128;
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

  pearl_gemm_sm89_run<KTraits>(mainloop_args, epilogue_args, M, N, K, stream);
}

// ============================================================================
// PoW variant: SkipReduction=false, SkipDenoising=false. PoW accumulator ON,
// denoise epilogue ON. Production mining target.
// ============================================================================

using PowTraits64x128x64_R128 = KernelTraitsSm89<
    /*ElementIn=*/      int8_t,
    /*ElementOut=*/     cutlass::bfloat16_t,
    /*ElementDenoise=*/ cutlass::half_t,
    /*ElementScale=*/   float,
    /*TileShape_MNKR=*/ cute::Shape<cute::Int<64>, cute::Int<128>,
                                    cute::Int<64>, cute::Int<128>>,
    /*Is_Even_M=*/      true,
    /*Is_Even_N=*/      true,
    /*cM=*/             1,
    /*cN=*/             1,
    /*SkipReduction=*/  false,
    /*SkipDenoising=*/  false,
    /*kStages=*/        2,
    /*EnableDebug=*/    false>;

template void pearl_gemm_sm89_run<PowTraits64x128x64_R128>(
    typename pearl::CollectiveMainloopSm89<
        PowTraits64x128x64_R128>::Arguments const&,
    typename pearl::CollectiveEpilogueSm89<
        PowTraits64x128x64_R128>::Arguments const&,
    int, int, int, cudaStream_t,
    NonceContext const*, int);

}  // namespace sm89
}  // namespace pearl

extern "C" void pearl_gemm_sm89_pow_64x128x64_R128(
    int8_t const* A, int64_t lda,
    int8_t const* B, int64_t ldb,
    cutlass::bfloat16_t* C, int64_t ldc,
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
  using KTraits = pearl::sm89::PowTraits64x128x64_R128;
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
  epilogue_args.problem_shape = cute::make_tuple(M, N, K, /*R=*/128);

  pearl::sm89::pearl_gemm_sm89_run<KTraits>(mainloop_args, epilogue_args, M, N, K, stream);
}
