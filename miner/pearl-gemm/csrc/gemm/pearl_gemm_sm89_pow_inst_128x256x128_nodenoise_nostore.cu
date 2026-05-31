// SPDX-License-Identifier: see LICENSE
//
// sm_89 PoW instantiation: MINING no-C-store (Lever A) + DENOISE-OFF search
// variant. Identical to pearl_gemm_sm89_pow_inst_128x256x128_nostore.cu except
// SkipDenoising=true: the fp16 denoise correction epilogue is elided.
//
// Correctness: the BLAKE3 transcript + PoW digest are folded from the int32
// GEMM accumulator (tCrC) inside the mainloop and written back BEFORE denoise()
// runs; denoise() only produces the useful-work C output (tCrD_fp32) which does
// NOT enter the transcript or check_pow_target. So this kernel yields a
// BIT-IDENTICAL transcript + gpu_hash to the denoise-ON kernel — it just skips
// the fp16 denoise MMAs. Used for the mining SEARCH loop; on a HIT the one
// winning nonce is re-run through the denoise-ON kernel to materialize the
// share proof's useful-work C.
//
// kMiningNoStore stays true (full mining shape C = 32 GB would OOM 16 GB).
// static_assert(!kMiningNoStore || !SkipReduction): satisfied (SkipReduction=false).

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

using PowTraits128x256x128_R256_NoDenoise_NoStore = KernelTraitsSm89<
    /*ElementIn=*/      int8_t,
    /*ElementOut=*/     cutlass::half_t,
    /*ElementDenoise=*/ cutlass::half_t,
    /*ElementScale=*/   float,
    /*TileShape_MNKR=*/ cute::Shape<cute::Int<128>, cute::Int<256>,
                                    cute::Int<128>, cute::Int<256>>,
    /*Is_Even_M=*/      true,
    /*Is_Even_N=*/      true,
    /*cM=*/             1,
    /*cN=*/             1,
    /*SkipReduction=*/  false,   // PoW transcript accumulator ON
    /*SkipDenoising=*/  true,    // <-- denoise epilogue OFF (search variant)
    /*kStages=*/        2,
    /*EnableDebug=*/    false,
    /*kRegisterResidentDenoise=*/ false,
    /*kMiningNoStore=*/ true>;   // Lever A: skip the C store entirely

template void pearl_gemm_sm89_run<PowTraits128x256x128_R256_NoDenoise_NoStore>(
    typename pearl::CollectiveMainloopSm89<
        PowTraits128x256x128_R256_NoDenoise_NoStore>::Arguments const&,
    typename pearl::CollectiveEpilogueSm89<
        PowTraits128x256x128_R256_NoDenoise_NoStore>::Arguments const&,
    int, int, int, cudaStream_t,
    NonceContext const*, int);

extern "C" void pearl_gemm_sm89_pow_128x256x128_R256_nodenoise_nostore(
    int8_t const* A, int64_t lda,
    int8_t const* B, int64_t ldb,
    cutlass::half_t* C, int64_t ldc,   // C ignored (kept for ABI symmetry)
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
  using KTraits = PowTraits128x256x128_R256_NoDenoise_NoStore;
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
  epilogue_args.ptr_C   = C;   // unused in no-store mode
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
