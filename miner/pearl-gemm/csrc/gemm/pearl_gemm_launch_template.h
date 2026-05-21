#pragma once

#include "cute/tensor.hpp"

#include <cutlass/kernel_hardware_info.h>
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"

#include "pearl_api_params.h"
#include "static_switch.h"

// sm_89 (Ada) and sm_120 (consumer Blackwell) share the same code path: both
// use sm_80-class MMA atoms (SM80_16x8x32_S32S8S8S32_TN) and sm_80 cp.async
// loads, with no TMA / no clusters / no mbarrier fence. nvcc emits per-arch
// SASS from the gencode list (see setup.py:_ARCH_GENCODES).
#if defined(PEARL_GEMM_BUILD_SM89) || defined(PEARL_GEMM_BUILD_SM120)
  #define PEARL_GEMM_USE_SM89_PATH 1
  #include "kernel_traits_sm89.hpp"
  #include "collective_mainloop_sm89.hpp"
  #include "collective_epilogue_sm89.hpp"
  #include "pearl_gemm_sm89_host.h"
  #include "pearl_noisingA_sm89_host.h"
  #include "pearl_noisingB_sm89_host.h"
#else
  // sm_90a (Hopper) path — uses cluster launch + TMA descriptors.
  #include "cutlass/cluster_launch.hpp"
  #include "pearl_gemm_host.h"
  #include "pearl_noisingA_host.h"
  #include "pearl_noisingB_host.h"
#endif

template <class ElementDenoise_AxEBL, int R, int bM_noising, int bK_noising,
          int kStages>
void run_pearl_noising_A_(PearlAPIParams& params, cudaStream_t stream = 0) {
  using namespace cute;
  using TileShape_MRK = Shape<Int<bM_noising>, Int<R>, Int<bK_noising>>;

#if defined(PEARL_GEMM_USE_SM89_PATH)
  BOOL_SWITCH(params.k % get<2>(TileShape_MRK{}) == 0, IsEvenKNoising,
              run_pearl_noising_A_sm89<ElementDenoise_AxEBL, TileShape_MRK,
                                       kStages, IsEvenKNoising>(params,
                                                                stream););
#else
  BOOL_SWITCH(params.k % get<2>(TileShape_MRK{}) == 0, IsEvenKNoising,
              run_pearl_noising_A<ElementDenoise_AxEBL, TileShape_MRK, kStages,
                                  IsEvenKNoising>(params, stream););
#endif
}

template <class ElementDenoise_EARxBpEB, int R, int bN_noising, int bK_noising,
          int kStages>
void run_pearl_noising_B_(PearlAPIParams& params, cudaStream_t stream = 0) {
  using namespace cute;
  using TileShape_NRK = Shape<Int<bN_noising>, Int<R>, Int<bK_noising>>;

#if defined(PEARL_GEMM_USE_SM89_PATH)
  // sm_89 noisingB takes raw pointers, not PearlAPIParams. The fields
  // used here are the same ones the Hopper variant reads from `params`
  // inside `pearl_noisingB_host.h:34-35`.
  BOOL_SWITCH(
      params.k % get<2>(TileShape_NRK{}) == 0, IsEvenKNoising,
      pearl::sm89::run_pearl_noising_B_sm89<
          ElementDenoise_EARxBpEB, TileShape_NRK, kStages, IsEvenKNoising>(
          static_cast<int8_t const*>(params.ptr_B),
          static_cast<int8_t const*>(params.ptr_EBR),
          static_cast<int8_t const*>(params.ptr_EBL_R_major),
          static_cast<int8_t const*>(params.ptr_EAR_K_major),
          static_cast<int8_t*>(params.ptr_BpEB),
          static_cast<ElementDenoise_EARxBpEB*>(params.ptr_EARxBpEB),
          params.n, params.k, params.k_blocks_per_split_noising_B, stream););
#else
  BOOL_SWITCH(params.k % get<2>(TileShape_NRK{}) == 0, IsEvenKNoising,
              run_pearl_noising_B<ElementDenoise_EARxBpEB, TileShape_NRK,
                                  kStages, IsEvenKNoising>(params, stream););
#endif
}

template <class ElementOut, int R, int bM, int bN, int bK, int kStages,
          int cM = 1, int cN = 1, bool SkipReduction = true,
          bool SkipDenoising = false, bool EnableDebug = false>
void run_pearl_gemm_(PearlAPIParams& params, cudaStream_t stream = 0) {
  using namespace cute;
  using TileShape_MNKR = Shape<Int<bM>, Int<bN>, Int<bK>, Int<R>>;
  bool is_even_m = params.m % get<0>(TileShape_MNKR{}) == 0;
  bool is_even_n = params.n % get<1>(TileShape_MNKR{}) == 0;

#if defined(PEARL_GEMM_USE_SM89_PATH)
  // Build sm_89 KTraits + Mainloop/Epilogue Arguments from PearlAPIParams,
  // then call into pearl_gemm_sm89_run. ElementDenoise is hard-coded to
  // half_t to match the Hopper path (pearl_gemm_host.h:25) and the existing
  // sm_89 .cu instantiations.
  BOOL_SWITCH(
      is_even_m, IsEvenM,
      BOOL_SWITCH(
          is_even_n, IsEvenN,

          using ElementIn       = int8_t;
          using ElementDenoise_ = cutlass::half_t;
          using ElementScale    = float;
          using KTraits         = pearl::KernelTraitsSm89<
              ElementIn, ElementOut, ElementDenoise_, ElementScale,
              TileShape_MNKR, IsEvenM, IsEvenN, cM, cN, SkipReduction,
              SkipDenoising, kStages, EnableDebug>;
          using Mainloop = pearl::CollectiveMainloopSm89<KTraits>;
          using Epilogue = pearl::CollectiveEpilogueSm89<KTraits>;

          int64_t const lda = static_cast<int64_t>(params.k);
          int64_t const ldb = static_cast<int64_t>(params.k);
          int64_t const ldc = static_cast<int64_t>(params.n);
          auto problem_shape = cute::make_tuple(params.m, params.n,
                                                params.k, R);

          typename Mainloop::Arguments mainloop_args{};
          mainloop_args.ptr_A    =
              static_cast<int8_t const*>(params.ptr_ApEA);
          mainloop_args.layout_A = cute::make_layout(
              cute::make_shape(params.m, params.k),
              cute::make_stride(lda, cute::_1{}));
          mainloop_args.ptr_B    =
              static_cast<int8_t const*>(params.ptr_BpEB);
          mainloop_args.layout_B = cute::make_layout(
              cute::make_shape(params.n, params.k),
              cute::make_stride(ldb, cute::_1{}));
          mainloop_args.problem_shape = problem_shape;
          mainloop_args.ptr_pow_target =
              static_cast<uint32_t const*>(params.ptr_pow_target);
          mainloop_args.ptr_pow_key =
              static_cast<uint32_t const*>(params.ptr_pow_key);
          mainloop_args.host_signal_sync = params.host_signal_sync;
          mainloop_args.host_signal_header_pinned =
              params.host_signal_header_pinned;
          mainloop_args.inner_hash_counter = params.inner_hash_counter;

          typename Epilogue::Arguments epilogue_args{};
          epilogue_args.ptr_C    = static_cast<ElementOut*>(params.ptr_C);
          epilogue_args.layout_C = cute::make_layout(
              cute::make_shape(params.m, params.n),
              cute::make_stride(ldc, cute::_1{}));
          epilogue_args.ptr_A_scales =
              static_cast<float const*>(params.ptr_A_scales);
          epilogue_args.ptr_B_scales =
              static_cast<float const*>(params.ptr_B_scales);
          epilogue_args.ptr_EAL =
              static_cast<cutlass::half_t const*>(params.ptr_EAL_mma);
          epilogue_args.ptr_EARxBpEB =
              static_cast<cutlass::half_t const*>(params.ptr_EARxBpEB_mma);
          epilogue_args.ptr_AxEBL =
              static_cast<cutlass::half_t const*>(params.ptr_AxEBL_mma);
          epilogue_args.ptr_EBR =
              static_cast<cutlass::half_t const*>(params.ptr_EBR_mma);
          epilogue_args.problem_shape = problem_shape;

          // Pass the (optional) NonceContext array + batch size through to the
          // host launcher. The launcher consults the env var
          // PEARL_SM89_PERSISTENT_NONCE to decide whether to dispatch the
          // multi-nonce kernel template. When the env var is unset OR the
          // context pointer is null, the launcher takes the single-nonce path.
          auto const* nonce_ctxs =
              static_cast<pearl::sm89::NonceContext const*>(
                  params.ptr_nonce_contexts);
          pearl::sm89::pearl_gemm_sm89_run<KTraits>(
              mainloop_args, epilogue_args, params.m, params.n, params.k,
              stream, nonce_ctxs, params.nonce_batch_size);
      ););
#else
  BOOL_SWITCH(
      is_even_m, IsEvenM,
      BOOL_SWITCH(
          is_even_n, IsEvenN,

          run_pearl_gemm<ElementOut, TileShape_MNKR, kStages, cM, cN, IsEvenM,
                         IsEvenN, SkipReduction, SkipDenoising, EnableDebug>(
              params, stream);

      ););
#endif
}
