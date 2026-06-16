// SPDX-License-Identifier: see LICENSE
//
// sm_89 PoW instantiations for mining search A/B: same consensus-visible
// 128x256 R=256 tile as production, but bK=64 with kStages={2,3}. These are
// gated from pearl_miner_sm89.cu by PEARL_SM89_POW_BK64_STAGE and default-off.

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

template <int Stages>
using PowTraits128x256x64_R256_NoDenoise_NoStore = KernelTraitsSm89<
    /*ElementIn=*/      int8_t,
    /*ElementOut=*/     cutlass::half_t,
    /*ElementDenoise=*/ cutlass::half_t,
    /*ElementScale=*/   float,
    /*TileShape_MNKR=*/ cute::Shape<cute::Int<128>, cute::Int<256>,
                                    cute::Int<64>,  cute::Int<256>>,
    /*Is_Even_M=*/      true,
    /*Is_Even_N=*/      true,
    /*cM=*/             1,
    /*cN=*/             1,
    /*SkipReduction=*/  false,
    /*SkipDenoising=*/  true,
    /*kStages=*/        Stages,
    /*EnableDebug=*/    false,
    /*kRegisterResidentDenoise=*/ false,
    /*kMiningNoStore=*/ true>;

template <int Stages>
static void run_pow_bk64_nodenoise_nostore(
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
  using KTraits = PowTraits128x256x64_R256_NoDenoise_NoStore<Stages>;
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

using PowTraits128x256x64_R256_NoDenoise_NoStore_S2 =
    PowTraits128x256x64_R256_NoDenoise_NoStore<2>;
using PowTraits128x256x64_R256_NoDenoise_NoStore_S3 =
    PowTraits128x256x64_R256_NoDenoise_NoStore<3>;

template void pearl_gemm_sm89_run<PowTraits128x256x64_R256_NoDenoise_NoStore_S2>(
    typename pearl::CollectiveMainloopSm89<
        PowTraits128x256x64_R256_NoDenoise_NoStore_S2>::Arguments const&,
    typename pearl::CollectiveEpilogueSm89<
        PowTraits128x256x64_R256_NoDenoise_NoStore_S2>::Arguments const&,
    int, int, int, cudaStream_t,
    NonceContext const*, int);

template void pearl_gemm_sm89_run<PowTraits128x256x64_R256_NoDenoise_NoStore_S3>(
    typename pearl::CollectiveMainloopSm89<
        PowTraits128x256x64_R256_NoDenoise_NoStore_S3>::Arguments const&,
    typename pearl::CollectiveEpilogueSm89<
        PowTraits128x256x64_R256_NoDenoise_NoStore_S3>::Arguments const&,
    int, int, int, cudaStream_t,
    NonceContext const*, int);

extern "C" void pearl_gemm_sm89_pow_128x256x64_R256_nodenoise_nostore_s2(
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
  run_pow_bk64_nodenoise_nostore<2>(
      A, lda, B, ldb, C, ldc, A_scales, B_scales, EAL, EBR, AxEBL, EARxBpEB,
      pow_target, pow_key, host_signal_sync, host_signal_header_pinned,
      inner_hash_counter, M, N, K, stream);
}

extern "C" void pearl_gemm_sm89_pow_128x256x64_R256_nodenoise_nostore_s3(
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
  run_pow_bk64_nodenoise_nostore<3>(
      A, lda, B, ldb, C, ldc, A_scales, B_scales, EAL, EBR, AxEBL, EARxBpEB,
      pow_target, pow_key, host_signal_sync, host_signal_header_pinned,
      inner_hash_counter, M, N, K, stream);
}

}  // namespace sm89
}  // namespace pearl
