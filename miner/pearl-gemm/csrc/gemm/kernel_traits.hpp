#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include <cutlass/arch/arch.h>
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

namespace pearl {
using namespace cute;

template <typename ElementIn_, typename ElementOut_, typename ElementDenoise_,
          typename ElementScale_, typename TileShape_MNKR_, bool Is_Even_M_,
          bool Is_Even_N_, int cM_, int cN_, bool SkipReduction_,
          bool SkipDenoising_, int kStages_, bool EnableDebug_>
struct KernelTraits {

  using ElementIn = ElementIn_;
  using ElementScale = ElementScale_;
  using ElementAccum = int32_t;  // accum dtype for main gemm
  using ElementOut = ElementOut_;
  using ElementDenoise =
      ElementDenoise_;                // dtype of denoise matrices before gemm
  using ElementDenoiseAccum = float;  // accum dtype for denoise gemm
  using index_t = int64_t;

  using TileShape_MNKR = TileShape_MNKR_;
  static constexpr bool Is_Even_M = Is_Even_M_;
  static constexpr bool Is_Even_N = Is_Even_N_;
  static constexpr bool SkipReduction = SkipReduction_;
  static constexpr bool SkipDenoising = SkipDenoising_;
  static constexpr int kStages = kStages_;
  static constexpr bool EnableDebug = EnableDebug_;
  static constexpr int srcLane = 0;

  using ProblemShape = Shape<int, int, int, int>;
  static_assert(is_same_v<ElementDenoise, half_t> ||
                is_same_v<ElementDenoise, int32_t>);
  static_assert(is_same_v<ElementDenoise, half_t> ||
                is_same_v<ElementDenoise, int32_t>);

  static constexpr int bM = get<0>(TileShape_MNKR{});
  static constexpr int bN = get<1>(TileShape_MNKR{});
  static constexpr int bK = get<2>(TileShape_MNKR{});
  static constexpr int R = get<3>(TileShape_MNKR{});

  // Use a 64 x bN tile per warpgroup; so thread count controlled by tile_size_m parameter
  static constexpr int kNumMmaWarpgroups = bM / 64;
  static constexpr int kNumMmaThreads = kNumMmaWarpgroups * 128;
  // Use one warp in producer warpgroup for TMA
  static constexpr int kNumProducerThreads = cutlass::NumThreadsPerWarp;
  static constexpr int kNumThreads = kNumMmaThreads + 128;
  static constexpr int kNumWarps = kNumThreads / cutlass::NumThreadsPerWarp;

  // Warp tiling constants. Used both for the SM80 mma.sync atom layout
  // (Blackwell port) and for the reduce_buffer permutation. 4 warps per
  // MMA warpgroup, each warp owning 16 rows of the per-WG accumulator.
  static constexpr int kWarpRows = 4 * kNumMmaWarpgroups;
  static constexpr int kWarpCols = 1;
  using MMAWarpLayout = Layout<Shape<Int<kWarpRows>, Int<kWarpCols>, _1>>;
  using MMAWarpTileShape = Shape<_16, _8, _32>;


  using TileShape_MNK = Shape<Int<bM>, Int<bN>, Int<bK>>;
  // used for denoising
  using TileShape_MNR = Shape<Int<bM>, Int<bN>, Int<R>>;

  static constexpr int kClusterSizeM = cM_;
  static constexpr int kClusterSizeN = cN_;
  using ClusterShape_MNK = Shape<Int<kClusterSizeM>, Int<kClusterSizeN>, _1>;
  // Multicasting to a single CTA has been known to be worse perf than non-multicast
  using TMAOpA =
      std::conditional_t<(kClusterSizeN > 1), cute::SM90_TMA_LOAD_MULTICAST,
                         cute::SM90_TMA_LOAD>;
  using TMAOpB =
      std::conditional_t<(kClusterSizeM > 1), cute::SM90_TMA_LOAD_MULTICAST,
                         cute::SM90_TMA_LOAD>;
  // GEMM traits
  //
  // Blackwell port (sm_120a / sm_121a — GB10, RTX 50-series): WGMMA is not
  // available on consumer Blackwell. Replace the Hopper SM90 WGMMA atom with a
  // 4-warp-per-warpgroup tile of SM80 mma.sync 16x8x32 int8 atoms. The
  // per-thread accumulator-fragment layout is bit-identical to Hopper because
  // Hopper CLayout_64xN was constructed exactly this way (4 warps in M × N/8
  // atoms across × interleaved (v1,v0) value packing). This preserves
  // PoUW-relevant xor_reduction(tCrC) outputs bit-for-bit.
  using AtomLayoutMNK = Layout<Shape<Int<kWarpRows>, _1, _1>>;
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::MMA_Atom<cute::SM80_16x8x32_S32S8S8S32_TN>{},
      AtomLayoutMNK{},
      cute::Tile<cute::Underscore, cute::Int<bN>, cute::Underscore>{}));
  using MMATraits = typename TiledMma::Atom::Traits;
  using MMAAtomShape_MNK = typename TiledMma::AtomShape_MNK;
  using MMAAtom_K = decltype(get<2>(MMAAtomShape_MNK{}));

  using SmemLayoutAtomA =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementIn,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}),
                 Int<kStages>{})));

  using SmemLayoutAtomB =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementIn,
               decltype(cute::get<1>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}),
                 Int<kStages>{})));

  using SmemLayoutAtomC =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementOut,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<1>(TileShape_MNK{}))>());
  using SmemLayoutC =
      decltype(tile_to_shape(SmemLayoutAtomC{}, select<0, 1>(TileShape_MNK{})));

  using SmemCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, ElementOut>;

  using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;

  // Probably don't need more than 1 stage here because denoise load latency
  // is well-hidden under the rest of the matmul
  static constexpr int kDenoiseStages = 1;
  using DenoisePipeline = typename cutlass::PipelineTmaAsync<kDenoiseStages>;

  // Scales traits
  using G2SScales_copy_op = SM80_CP_ASYNC_CACHEALWAYS<ElementScale>;
  using G2SScales_copy_traits = Copy_Traits<G2SScales_copy_op>;
  using G2SScales_copy_atom = Copy_Atom<G2SScales_copy_traits, ElementScale>;
  using SmemLayoutScaleA = Layout<Shape<Int<bM>>, Stride<_1>>;
  using SmemLayoutScaleB = Layout<Shape<Int<bN>>, Stride<_1>>;

  using G2SScalesCopyA = decltype(make_tiled_copy(
      G2SScales_copy_atom{}, Layout<Shape<Int<bM>>, Stride<_1>>{},
      Layout<Shape<_1>, Stride<_1>>{}));
  using G2SScalesCopyB = decltype(make_tiled_copy(
      G2SScales_copy_atom{}, Layout<Shape<Int<bN>>, Stride<_1>>{},
      Layout<Shape<_1>, Stride<_1>>{}));

  // Denoising
  // MMA — Blackwell SM80 mma.sync 16x8x16 fp16 atoms tiled the same way as
  // the main GEMM (kWarpRows warps in M, bN expansion via permutation). See
  // the main-GEMM commentary above for the bit-identicality argument.
  using AtomLayoutMNR = Layout<Shape<Int<kWarpRows>, _1, _1>>;
  using TiledMmaDenoise = decltype(cute::make_tiled_mma(
      cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>{},
      AtomLayoutMNR{},
      cute::Tile<cute::Underscore, cute::Int<bN>, cute::Underscore>{}));
  // SMEM layouts
  using SmemLayoutEAL_Atom =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementDenoise, Int<bM>, Int<R>>());
  using SmemLayoutEAL = decltype(tile_to_shape(
      SmemLayoutEAL_Atom{}, Shape<Int<bM>, Int<R>, Int<kDenoiseStages>>{}));

  using SmemLayoutEBR_Atom =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementDenoise, Int<bN>, Int<R>>());
  using SmemLayoutEBR = decltype(tile_to_shape(
      SmemLayoutEBR_Atom{}, Shape<Int<bN>, Int<R>, Int<kDenoiseStages>>{}));

  // Other factors have 1 layout if fp16, or 2 layouts if int32
  using SmemLayoutAxEBL_Atom =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementDenoise, Int<bM>, Int<R>>());
  using SmemLayoutAxEBL = decltype(tile_to_shape(
      SmemLayoutAxEBL_Atom{}, Shape<Int<bM>, Int<R>, Int<kDenoiseStages>>{}));

  using SmemLayoutEARxBpEB_Atom =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementDenoise, Int<bN>, Int<R>>());
  using SmemLayoutEARxBpEB =
      decltype(tile_to_shape(SmemLayoutEARxBpEB_Atom{},
                             Shape<Int<bN>, Int<R>, Int<kDenoiseStages>>{}));

  static_assert(R % 16 == 0);  // needed for this MMA op

  // NOTE if you change these, also change the pipeline stages heuristic in
  // heuristics.hpp!
  struct SharedStorageDenoise : cute::aligned_struct<128> {
    // Overlapping to allow larger tile sizes.
    // Denoise factors are all fp16 and used as inputs to SS WGMMA. Currently
    //  all denoise factors' smem storage are disjoint with each other while
    //  overlapped with mainloop smem, so we wait for mainloop gemms to finish
    //  before starting loads for denoise factors.
    union {
      struct {
        cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutA>,
                            cutlass::detail::alignment_for_swizzle(
                                SmemLayoutA{})>
            smem_A;
        cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutB>,
                            cutlass::detail::alignment_for_swizzle(
                                SmemLayoutB{})>
            smem_B;
      };

      // Denoise phase 1 (EAL x EARxBpEB) — held in SMEM only during the
      // first denoise WGMMA. After consumer_release of EAxBpEB_pipeline +
      // arrival on NamedBarriers::DenoisePhase1Consumed, the producer is
      // allowed to reload this region for phase 2.
      struct {
        cute::array_aligned<ElementDenoise, cute::cosize_v<SmemLayoutEAL>,
                            cutlass::detail::alignment_for_swizzle(
                                SmemLayoutEAL{})>
            smem_EAL;
        cute::array_aligned<ElementDenoise, cute::cosize_v<SmemLayoutEARxBpEB>,
                            cutlass::detail::alignment_for_swizzle(
                                SmemLayoutEARxBpEB{})>
            smem_EARxBpEB;
      };

      // Denoise phase 2 (AxEBL x EBR) — alias of phase 1's SMEM. Producer
      // waits for DenoisePhase1Consumed before loading.
      struct {
        cute::array_aligned<ElementDenoise, cute::cosize_v<SmemLayoutAxEBL>,
                            cutlass::detail::alignment_for_swizzle(
                                SmemLayoutAxEBL{})>
            smem_AxEBL;
        cute::array_aligned<ElementDenoise, cute::cosize_v<SmemLayoutEBR>,
                            cutlass::detail::alignment_for_swizzle(
                                SmemLayoutEBR{})>
            smem_EBR;
      };

      // Path 3: smem_C removed; epilogue now writes registers directly to gmem.
    };

    cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScaleA>,
                        cutlass::detail::alignment_for_swizzle(
                            SmemLayoutScaleA{})>
        smem_scale_a;
    cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScaleB>,
                        cutlass::detail::alignment_for_swizzle(
                            SmemLayoutScaleB{})>
        smem_scale_b;

    struct {
      typename MainloopPipeline::SharedStorage pipeline;
      typename DenoisePipeline::SharedStorage AxEB_pipeline;
      typename DenoisePipeline::SharedStorage EAxBpEB_pipeline;
    };
  };

  struct SharedStorageNoDenoise : cute::aligned_struct<128> {
    struct {
      cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutA>,
                          cutlass::detail::alignment_for_swizzle(SmemLayoutA{})>
          smem_A;
      cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutB>,
                          cutlass::detail::alignment_for_swizzle(SmemLayoutB{})>
          smem_B;
    };
    // Path 3: smem_C removed; epilogue does direct register->gmem stores.

    cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScaleA>,
                        cutlass::detail::alignment_for_swizzle(
                            SmemLayoutScaleA{})>
        smem_scale_a;
    cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScaleB>,
                        cutlass::detail::alignment_for_swizzle(
                            SmemLayoutScaleB{})>
        smem_scale_b;

    struct {
      typename MainloopPipeline::SharedStorage pipeline;
      typename DenoisePipeline::SharedStorage AxEB_pipeline;
      typename DenoisePipeline::SharedStorage EAxBpEB_pipeline;
    };
  };

  using SharedStorage =
      cute::conditional_t<SkipDenoising, SharedStorageNoDenoise,
                          SharedStorageDenoise>;
};

}  // namespace pearl
