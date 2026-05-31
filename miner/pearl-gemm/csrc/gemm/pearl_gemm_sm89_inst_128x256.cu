// SPDX-License-Identifier: see LICENSE
//
// sm_89 single-config instantiation: noiseless int8 GEMM, (bM,bN,bK,R) =
// (128,256,64,64), kStages=2, SkipDenoising=true, SkipReduction=true.
// Wider-N variant of pearl_gemm_sm89_inst.cu — exercises the union'd
// SharedStorageNoDenoise that overlaps (smem_A+smem_B) with smem_C.
//
// Tile-size choice for sm_89 noiseless at bN=256:
//   union(smem_A+smem_B, smem_C) = max(2*(128*64 + 256*64)*kStages * 1B,
//                                      128*256 * 2B)
//                                = max(2*24576*kStages, 65536)
// kStages=2: max(98304, 65536) = 98304 B → too tight with scales+barriers.
// Actually at kStages=2: smem_A = 128*64*2 = 16384, smem_B = 256*64*2 = 32768
//   sum = 49152; union with smem_C = max(49152, 65536) = 65536.
//   + scales (1536) + barriers (~1024) ≈ ~68 KB — fits 99 KB optin.
// kStages=3 would also fit (~76 KB) but spec says start at kStages=2 to mirror
// alpha-miner's register-pressure-aware production setting at wider N.

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

using NoiselessTraits128x256x64_R64 = KernelTraitsSm89<
    /*ElementIn=*/      int8_t,
    /*ElementOut=*/     cutlass::bfloat16_t,
    /*ElementDenoise=*/ cutlass::half_t,
    /*ElementScale=*/   float,
    /*TileShape_MNKR=*/ cute::Shape<cute::Int<128>, cute::Int<256>,
                                    cute::Int<64>,  cute::Int<64>>,
    /*Is_Even_M=*/      true,
    /*Is_Even_N=*/      true,
    /*cM=*/             1,
    /*cN=*/             1,
    /*SkipReduction=*/  true,
    /*SkipDenoising=*/  true,
    /*kStages=*/        2,
    /*EnableDebug=*/    false>;

template void pearl_gemm_sm89_run<NoiselessTraits128x256x64_R64>(
    typename pearl::CollectiveMainloopSm89<
        NoiselessTraits128x256x64_R64>::Arguments const& mainloop_args,
    typename pearl::CollectiveEpilogueSm89<
        NoiselessTraits128x256x64_R64>::Arguments const& epilogue_args,
    int M, int N, int K, cudaStream_t stream,
    NonceContext const* ptr_nonce_contexts,
    int nonce_batch_size);

extern "C" void pearl_gemm_sm89_noiseless_128x256x64_R64(
    int8_t const* A, int64_t lda,
    int8_t const* B, int64_t ldb,
    cutlass::bfloat16_t* C, int64_t ldc,
    float const* A_scales,
    float const* B_scales,
    int M, int N, int K,
    cudaStream_t stream) {
  using KTraits = NoiselessTraits128x256x64_R64;
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
  epilogue_args.ptr_A_scales  = A_scales;
  epilogue_args.ptr_B_scales  = B_scales;
  epilogue_args.problem_shape = cute::make_tuple(M, N, K, /*R=*/64);

  pearl_gemm_sm89_run<KTraits>(mainloop_args, epilogue_args, M, N, K, stream);
}

}  // namespace sm89
}  // namespace pearl
