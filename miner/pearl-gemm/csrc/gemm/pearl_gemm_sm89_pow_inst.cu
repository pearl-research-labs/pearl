// SPDX-License-Identifier: see LICENSE
//
// sm_89 instantiation with the PoW transcript accumulator ENABLED.
// Same tile (bM=128, bN=128, bK=64, R=64) and kStages=3 as the production
// `pearl_gemm_sm89_denoise_*` variant — the only difference is
// `SkipReduction=false`, which flips on the `TileHashAccumulator` calls inside
// `collective_mainloop_sm89.hpp` (already wired) and the post-mainloop
// `check_pow_target` + `write_host_signal_header` calls in
// `pearl_gemm_kernel_sm89.h:148-161`.
//
// Exposes:
//   pearl_gemm_sm89_pow_128x128x64_R64
//     ↳ Same as the denoise variant but takes pow_target/pow_key + signal
//       structures. Signal structures are device pointers — caller (host
//       orchestrator) allocates pinned + maps them.
//
// Use this for:
//   - Benching the overhead of the hash accumulator on the hot path.
//   - Validating bit-exact transcript output vs CPU reference.
//   - Wiring up the real miner once the host orchestration is in place.

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

using PowTraits128x128x64_R64 = KernelTraitsSm89<
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
    /*SkipReduction=*/  false,   // ← PoW accumulator ON
    /*SkipDenoising=*/  false,   // denoise epilogue ON (matches production)
    /*kStages=*/        3,
    /*EnableDebug=*/    false>;

template void pearl_gemm_sm89_run<PowTraits128x128x64_R64>(
    typename pearl::CollectiveMainloopSm89<
        PowTraits128x128x64_R64>::Arguments const&,
    typename pearl::CollectiveEpilogueSm89<
        PowTraits128x128x64_R64>::Arguments const&,
    int, int, int, cudaStream_t,
    NonceContext const*, int);

// C trampoline with the PoW parameters wired through.
//   ptr_pow_target: device pointer to uint32_t[8] (256-bit target, hash ≤ target = block found)
//   ptr_pow_key:    device pointer to uint32_t[8] (BLAKE3 keying material)
//   host_signal_sync: device pointer to HostSignalSync (8-byte aligned, contains
//                     global_lock + status enum). Allocate on host as pinned and
//                     pass the device-mapped pointer.
//   host_signal_header_pinned: device pointer to HostSignalHeader (128-byte aligned).
//                              Caller polls host-side after each launch; status flips
//                              to kSignalTriggered when a CTA finds a hash ≤ target.
//   inner_hash_counter: optional device pointer to uint64_t for debug counts (or nullptr).
extern "C" void pearl_gemm_sm89_pow_128x128x64_R64(
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
    void* host_signal_sync,             // HostSignalSync*
    void* host_signal_header_pinned,    // HostSignalHeader*
    uint64_t* inner_hash_counter,
    int M, int N, int K,
    cudaStream_t stream) {
  using KTraits = PowTraits128x128x64_R64;
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
  epilogue_args.problem_shape = cute::make_tuple(M, N, K, /*R=*/64);

  pearl_gemm_sm89_run<KTraits>(mainloop_args, epilogue_args, M, N, K, stream);
}

}  // namespace sm89
}  // namespace pearl

// ============================================================================
// R=128 variant — bM=bN=64, bK=64, kStages=2. Used by alphapool/R=128 driver.
// ============================================================================

namespace pearl {
namespace sm89 {

using PowTraits64x64x64_R128 = KernelTraitsSm89<
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
    /*SkipReduction=*/  false,
    /*SkipDenoising=*/  false,
    /*kStages=*/        2,
    /*EnableDebug=*/    false>;

template void pearl_gemm_sm89_run<PowTraits64x64x64_R128>(
    typename pearl::CollectiveMainloopSm89<
        PowTraits64x64x64_R128>::Arguments const&,
    typename pearl::CollectiveEpilogueSm89<
        PowTraits64x64x64_R128>::Arguments const&,
    int, int, int, cudaStream_t,
    NonceContext const*, int);

// Also instantiate the dispatch wrapper for both IsEvenN values (BOOL_SWITCH).
// We need these on disk so the device kernel is registered for the pybind path.

}  // namespace sm89
}  // namespace pearl

extern "C" void pearl_gemm_sm89_pow_64x64x64_R128(
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
  using KTraits = pearl::sm89::PowTraits64x64x64_R128;
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
