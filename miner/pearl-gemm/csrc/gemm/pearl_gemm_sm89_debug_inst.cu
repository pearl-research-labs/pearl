// SPDX-License-Identifier: see LICENSE
//
// Debug-only instantiation: same (bM,bN,bK,R) = (128,128,64,64), kStages=3,
// noiseless config as pearl_gemm_sm89_inst.cu, but writes raw int32
// accumulator to a flat int32 buffer (no scale, no cast, no smem stage).
//
// Compare the int32 output against torch._int_mm (bit-exact, no rounding) to
// isolate whether the mainloop + MMA + smem/G2S/S2R wiring is correct.

#include <cuda_runtime.h>
#include "cute/tensor.hpp"

#include "kernel_traits_sm89.hpp"
#include "collective_mainloop_sm89.hpp"
#include "pearl_gemm_kernel_sm89_debug.h"
#include "pearl_gemm_sm89_debug_host.h"

namespace pearl {
namespace sm89 {

// Match the production traits exactly (same tile, stages, swizzle, etc.) so
// the debug path exercises the same mainloop. ElementOut is unused here
// (no epilogue) but the traits template requires it.
using DebugTraits128x128x64_R64 = KernelTraitsSm89<
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
    /*SkipDenoising=*/  true,
    /*kStages=*/        3,
    /*EnableDebug=*/    false>;

template void pearl_gemm_sm89_debug_run<DebugTraits128x128x64_R64>(
    typename pearl::CollectiveMainloopSm89<
        DebugTraits128x128x64_R64>::Arguments const& mainloop_args,
    int32_t* ptr_C_i32, int64_t ldc,
    int M, int N, int K, cudaStream_t stream);

// C-callable trampoline.
extern "C" void pearl_gemm_sm89_debug_int32_dump(
    int8_t const* A, int64_t lda,
    int8_t const* B, int64_t ldb,
    int32_t* C, int64_t ldc,
    int M, int N, int K,
    cudaStream_t stream) {
  using KTraits  = DebugTraits128x128x64_R64;
  using Mainloop = pearl::CollectiveMainloopSm89<KTraits>;

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

  pearl_gemm_sm89_debug_run<KTraits>(mainloop_args, C, ldc, M, N, K, stream);
}

}  // namespace sm89
}  // namespace pearl
