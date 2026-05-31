// Baseline twin of pearl_gemm_sm89_inst.cu: same NoiselessTraits instantiation,
// but the C trampoline calls pearl_gemm_sm89_run_baseline (SimpleTileScheduler)
// instead of pearl_gemm_sm89_run (PersistentSwizzledTileScheduler).
//
// Exposes `pearl_gemm_sm89_noiseless_128x128x64_R64_baseline`.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "cute/tensor.hpp"

#include "kernel_traits_sm89.hpp"
#include "collective_mainloop_sm89.hpp"
#include "collective_epilogue_sm89.hpp"
#include "pearl_gemm_kernel_sm89.h"
#include "pearl_gemm_sm89_host_baseline.h"

namespace pearl {
namespace sm89 {

using NoiselessTraits128x128x64_R64_BL = KernelTraitsSm89<
    int8_t, cutlass::bfloat16_t, cutlass::half_t, float,
    cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<64>, cute::Int<64>>,
    true, true, 1, 1, true, true, 3, false>;

template void pearl_gemm_sm89_run_baseline<NoiselessTraits128x128x64_R64_BL>(
    typename pearl::CollectiveMainloopSm89<
        NoiselessTraits128x128x64_R64_BL>::Arguments const&,
    typename pearl::CollectiveEpilogueSm89<
        NoiselessTraits128x128x64_R64_BL>::Arguments const&,
    int, int, int, cudaStream_t);

extern "C" void pearl_gemm_sm89_noiseless_128x128x64_R64_baseline(
    int8_t const* A, int64_t lda,
    int8_t const* B, int64_t ldb,
    cutlass::bfloat16_t* C, int64_t ldc,
    float const* A_scales,
    float const* B_scales,
    int M, int N, int K,
    cudaStream_t stream) {
  using KTraits = NoiselessTraits128x128x64_R64_BL;
  using Mainloop = pearl::CollectiveMainloopSm89<KTraits>;
  using Epilogue = pearl::CollectiveEpilogueSm89<KTraits>;

  typename Mainloop::Arguments mainloop_args{};
  mainloop_args.ptr_A    = A;
  mainloop_args.layout_A =
      cute::make_layout(cute::make_shape(M, K), cute::make_stride(lda, cute::_1{}));
  mainloop_args.ptr_B    = B;
  mainloop_args.layout_B =
      cute::make_layout(cute::make_shape(N, K), cute::make_stride(ldb, cute::_1{}));
  mainloop_args.problem_shape = cute::make_tuple(M, N, K, 64);
  mainloop_args.ptr_pow_target            = nullptr;
  mainloop_args.ptr_pow_key               = nullptr;
  mainloop_args.host_signal_sync          = nullptr;
  mainloop_args.host_signal_header_pinned = nullptr;
  mainloop_args.inner_hash_counter        = nullptr;

  typename Epilogue::Arguments epilogue_args{};
  epilogue_args.ptr_C   = C;
  epilogue_args.layout_C =
      cute::make_layout(cute::make_shape(M, N), cute::make_stride(ldc, cute::_1{}));
  epilogue_args.ptr_A_scales  = A_scales;
  epilogue_args.ptr_B_scales  = B_scales;
  epilogue_args.problem_shape = cute::make_tuple(M, N, K, 64);

  pearl_gemm_sm89_run_baseline<KTraits>(mainloop_args, epilogue_args, M, N, K, stream);
}

}  // namespace sm89
}  // namespace pearl
