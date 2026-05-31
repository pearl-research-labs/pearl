// SPDX-License-Identifier: see LICENSE
//
// pearl-gemm KernelTraits for sm_89 (Ada Lovelace) — int8 GEMM, bf16 output.
//
// Companion to kernel_traits.hpp (sm_90a). API surface is kept compatible with
// the Hopper KernelTraits so that downstream collective files can opt into the
// sm_89 substrate by including this header instead.
//
// Substitutions vs. sm_90a path (see SM89_PORT_SPEC.md §2 for derivations):
//
//   sm_90a (kernel_traits.hpp)           sm_89 (this file)
//   ---------------------------------    ----------------------------------------------
//   SM90_TMA_LOAD[_MULTICAST]            SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>
//   SM90_TMA_STORE (epilogue C)          AutoVectorizingCopyWithAssumedAlignment<128>
//   GMMA::ss_op_selector<int8,..>        SM80_16x8x32_S32S8S8S32_TN
//   GMMA::ss_op_selector<fp16,..>        SM80_16x8x16_F32F16F16F32_TN  (denoise)
//   PipelineTmaAsync<kStages>            PipelineAsync<kStages>
//   ss_smem_selector<Major::K,int8>      composition(Swizzle<3,4,3>, K-major layout)
//   ss_smem_selector<Major::K,fp16>      composition(Swizzle<3,3,3>, K-major layout)
//   ClusterShape (cM,cN)                 forced to (1,1)
//   setmaxnreg / warpgroup_reg_*         removed (use __launch_bounds__)
//   warpgroup_arrive / commit / wait     removed (mma.sync is synchronous)
//
// Smem budget on sm_89: 100 KB/CTA dynamic shared (101376 B), vs. H100's 228 KB.
//   Recommended: bM=128, bN=128, bK=128, R=64, kStages=3 → ~98 KB (1 CTA/SM).
//   See SM89_PORT_SPEC.md §3 for the full sweep.

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_sm75.hpp"
#include "cute/atom/copy_traits_sm80.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm80.hpp"
#include "cute/swizzle.hpp"

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/layout.hpp"  // alignment_for_swizzle
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

namespace pearl {
using namespace cute;

template <typename ElementIn_, typename ElementOut_, typename ElementDenoise_,
          typename ElementScale_, typename TileShape_MNKR_, bool Is_Even_M_,
          bool Is_Even_N_, int cM_, int cN_, bool SkipReduction_,
          bool SkipDenoising_, int kStages_, bool EnableDebug_,
          bool kRegisterResidentDenoise_ = false>
struct KernelTraitsSm89 {

  // ---------- element types (unchanged from sm_90a) ----------
  using ElementIn           = ElementIn_;
  using ElementScale        = ElementScale_;
  using ElementAccum        = int32_t;
  using ElementOut          = ElementOut_;
  using ElementDenoise      = ElementDenoise_;
  using ElementDenoiseAccum = float;
  using index_t             = int64_t;

  using TileShape_MNKR = TileShape_MNKR_;
  static constexpr bool Is_Even_M     = Is_Even_M_;
  static constexpr bool Is_Even_N     = Is_Even_N_;
  static constexpr bool SkipReduction = SkipReduction_;
  static constexpr bool SkipDenoising = SkipDenoising_;
  static constexpr int  kStages       = kStages_;
  static constexpr bool EnableDebug   = EnableDebug_;
  static constexpr int  srcLane       = 0;
  // When true, the denoise epilogue does NOT stage EAL/EBR/AxEBL/EARxBpEB
  // tiles in shared memory. Instead, each thread streams its own (M, R) and
  // (N, R) factor rows directly from gmem (cached in L1) and computes the
  // denoise outer-product inline, per the per-thread tCrD partition. This
  // mirrors alpha-miner's register-resident epilogue (see
  // C:/Source/pearl-investigation/alpha_r128_denoise_re_2026_05_17.md). It
  // unlocks R=128 with larger bN tiles (e.g., bM=128 bN=128) that would
  // otherwise overflow the 99 KB opt-in smem cap with the smem-resident
  // (4 × bM × R × sizeof(fp16)) buffer.
  // Wave-10: auto-enable register-resident denoise at (bM=128, bN=256, R=128)
  // so the noisy_gemm dispatch path (which goes through MATMUL_CONFIG_SWITCH ->
  // run_pearl_gemm_<...> -> KernelTraitsSm89<...,kRegisterResidentDenoise_=false>
  // default) fits in the 99 KB opt-in smem cap.  Without this, the smem-resident
  // arm at (128, 256, 128) is 4*(128+256)*128*sizeof(fp16) = 384 KB, far over
  // the cap, and the dispatch-path instantiation fails to compile.  The wave-3
  // standalone inst file (pearl_gemm_sm89_r128_bN256_regresident_inst.cu) passes
  // kRegisterResidentDenoise_=true explicitly; this OR makes both paths work.
  static constexpr bool kRegisterResidentDenoise =
      kRegisterResidentDenoise_ ||
      (get<0>(TileShape_MNKR{}) == 128 && get<1>(TileShape_MNKR{}) == 256 &&
       get<3>(TileShape_MNKR{}) == 128);

  static_assert(kStages >= 2 && kStages <= 4,
                "sm_89 dynamic smem cap (100 KB) supports 2-4 stages at "
                "128x128x128 int8 tiles. Larger kStages overflows.");

  using ProblemShape = Shape<int, int, int, int>;
  static_assert(is_same_v<ElementDenoise, half_t> ||
                is_same_v<ElementDenoise, int32_t>);

  static constexpr int bM = get<0>(TileShape_MNKR{});
  static constexpr int bN = get<1>(TileShape_MNKR{});
  static constexpr int bK = get<2>(TileShape_MNKR{});
  static constexpr int R  = get<3>(TileShape_MNKR{});

  using TileShape_MNK = Shape<Int<bM>, Int<bN>, Int<bK>>;
  using TileShape_MNR = Shape<Int<bM>, Int<bN>, Int<R>>;

  // ---------- thread layout ----------
  // sm_89: unified-warp model. Every warp does cp.async loads AND mma.sync
  // compute. No producer/consumer split because sm_89 lacks setmaxnreg to
  // redistribute regs between warpgroups; splitting would waste half the
  // register file on the "load-only" warp.
  //
  // Warp grid is derived from bM so we can support both bM=128 (8 warps in a
  // 2x4 grid; per-warp output 64x32 int32) and bM=64 (4 warps in a 2x2 grid;
  // per-warp output 32x32 int32). bM=64 R=128 is required for alphapool's
  // rank=128 mining params — at bM=128 R=128 the SharedStorageDenoise arm
  // would exceed sm_89's 99 KB opt-in smem cap.
  //   bM=128 -> kNumWarps=8 -> kWarpRows=2, kWarpCols=4 (current)
  //   bM=64  -> kNumWarps=4 -> kWarpRows=2, kWarpCols=2 (new path)
  static_assert(bM == 64 || bM == 128,
                "sm_89 KTraits warp grid only validated for bM in {64,128}");
  static constexpr int kNumWarps           = bM / 16;       // 4 (bM=64) or 8 (bM=128)
  static constexpr int kNumThreads         = kNumWarps * cutlass::NumThreadsPerWarp;
  // Micro-opt #3: parameterize __launch_bounds__ minBlocksPerSM.
  // bM=128 (R=64): SharedStorage ~98 KB/CTA → only 1 CTA/SM fits (smem-bound).
  // bM=64  (R=128): SharedStorage ~50 KB/CTA → 2 CTAs/SM may fit if registers
  //   allow. Per Agent-survey memo, this can recover 3-8% latency hiding.
  // ptxas may force a register reduction at 2 CTAs/SM that introduces spills;
  // override at build time by setting PEARL_SM89_MIN_BLOCKS_PER_SM if needed.
  static constexpr int kMinBlocksPerSM     = (bM == 64) ? 2 : 1;
  // Compat aliases (consumers in shared headers reference these names):
  static constexpr int kNumMmaWarpgroups   = 1;        // no warpgroup MMA on sm_89
  static constexpr int kNumMmaThreads      = kNumThreads;
  static constexpr int kNumProducerThreads = 0;

  // ---------- cluster stubs (sm_89 has no clusters) ----------
  static_assert(cM_ == 1 && cN_ == 1,
                "sm_89 has no thread-block clusters. Set cM=cN=1.");
  static constexpr int kClusterSizeM = 1;
  static constexpr int kClusterSizeN = 1;
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  // Inert TMA op aliases — these names appear in shared headers; we tag them
  // with a sentinel that static_assert can catch if the sm_89 collective
  // accidentally instantiates make_tma_copy(TMAOpA{}, ...).
  struct DisabledTmaOpSm89 {};
  using TMAOpA = DisabledTmaOpSm89;
  using TMAOpB = DisabledTmaOpSm89;

  // ---------- Main int8 TiledMMA ----------
  // SM80 atom: 1 warp = m16 x n8 x k32, int8 x int8 -> int32.
  // Warp grid: kWarpRows (M) x kWarpCols (N) = (bM/16, 1).  Every warp owns
  // exactly ONE atom-row in M and SWEEPS all atoms in N.  This produces a
  // per-thread output footprint with M-cardinality = 2 (matching the Hopper
  // GMMA footprint and `MinerSettings.rows_pattern=[0, 8]` the pool advertises
  // via `pearl.set_mining_params`).  See `pearl-investigation/wave8_proof_diff_2026_05_18.md`
  // for the static analysis that pinned the prior <2,2,1>/<2,4,1> layout as
  // the silent-async-drop cause: per-thread thread_rows = {r,r+8,r+16,r+24}
  // (size 4) drove `PearlMiningConfigurationFactory.create()` to a rows_pattern
  // length-4 byte sequence at offset 9 of mining_config, which no longer
  // matched the pool's `[0,8]` expectation → job_key mismatch → merkle root
  // verification failed → pool silently dropped every share.
  //   bM=128 -> 8 warps in 8x1 layout, per-warp footprint 16M x bN int32
  //   bM=64  -> 4 warps in 4x1 layout, per-warp footprint 16M x bN int32
  static constexpr int kWarpRows = bM / 16;   // one atom per warp in M
  static constexpr int kWarpCols = 1;         // all atoms in single N-column
  static_assert(kWarpRows * kWarpCols == kNumWarps,
                "kWarpRows * kWarpCols must equal kNumWarps");
  using AtomLayoutMNK = Layout<Shape<Int<kWarpRows>, Int<kWarpCols>, _1>>;
  using TiledMma = decltype(make_tiled_mma(
      MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>{},
      AtomLayoutMNK{},
      Tile<Int<bM>, Int<bN>, _32>{}));

  using MMAAtomShape_MNK = Shape<_16, _8, _32>;
  using MMAAtom_K        = _32;

  // Reduce-buffer / scale warp grid layout (consumed by epilogue).
  using MMAWarpLayout    = Layout<Shape<Int<kWarpRows>, Int<kWarpCols>, _1>>;
  using MMAWarpTileShape = Shape<_16, _8, _32>;

  // ---------- Smem layouts ----------
  // int8 K-major swizzle atom — match CUTLASS canonical sm_80 default int8
  // K-major config exactly (`test/unit/gemm/device/default_gemm_configuration.hpp`
  // L343+, `test/unit/cute/hopper/cooperative_gemm.cu` L60). The Hopper SW128
  // atom (Swizzle<3,4,3> over Shape<_8,_128>) is GMMA-only — it does NOT match
  // the SM75_U32x4_LDSM_N lane→address contract on sm_80/89.
  //
  // Canonical: Swizzle<2,4,3> over (rows=16, K=64), stride (K, 1).
  // bM must be a multiple of 16, bK must be a multiple of 64.
  static_assert(bM % 16 == 0,
                "sm_89 int8 SmemLayoutAtom rows=16; bM must be multiple of 16");
  static_assert(bK % 64 == 0,
                "sm_89 int8 SmemLayoutAtom K=64; bK must be multiple of 64");
  using SmemLayoutAtomK_int8 = decltype(composition(
      Swizzle<2, 4, 3>{},
      Layout<Shape<_16, _64>, Stride<_64, _1>>{}));

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomK_int8{},
      make_shape(Int<bM>{}, Int<bK>{}, Int<kStages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomK_int8{},
      make_shape(Int<bN>{}, Int<bK>{}, Int<kStages>{})));

  // bf16 / fp16 K-major: 8-byte (= 4 element) chunk size, Swizzle<3,3,3>.
  using SmemLayoutAtomK_fp16 = decltype(composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_8, _64>, Stride<_64, _1>>{}));

  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomK_fp16{},
      make_shape(Int<bM>{}, Int<bN>{})));

  // ---------- Copy atoms ----------
  // G→S: 16-byte cp.async, vectorized. 16 int8 elements per thread.
  using G2SCopyAtomAB = Copy_Atom<
      Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>>,
      ElementIn>;
  // Thread layout must cover the (bM, bK) tile cleanly with kNumThreads
  // threads, each writing 16 contiguous int8 along K.
  //   K-threads = bK / 16,  M-threads = kNumThreads / K-threads.
  //   Total per issue = (M_thr, K_thr * 16).
  static constexpr int kG2S_K_threads = bK / 16;        // 4 when bK=64, 8 when bK=128
  static_assert(kG2S_K_threads > 0 && bK % 16 == 0,
                "bK must be a multiple of 16 (cp.async vector width)");
  static constexpr int kG2S_M_threads = kNumThreads / kG2S_K_threads;
  static_assert(kG2S_M_threads * kG2S_K_threads == kNumThreads,
                "kNumThreads must divide evenly into (M, K) thread grid");
  using G2SThreadLayoutAB = Layout<Shape<Int<kG2S_M_threads>, Int<kG2S_K_threads>>,
                                   Stride<Int<kG2S_K_threads>, _1>>;
  using G2SValueLayoutAB  = Layout<Shape<_1, _16>>;
  using G2SCopyA = decltype(make_tiled_copy(
      G2SCopyAtomAB{}, G2SThreadLayoutAB{}, G2SValueLayoutAB{}));
  using G2SCopyB = decltype(make_tiled_copy(
      G2SCopyAtomAB{}, G2SThreadLayoutAB{}, G2SValueLayoutAB{}));

  // S→R: ldmatrix non-transposed (.N) — int8 K-major operands feed TN MMA
  // directly. SM75_U32x4_LDSM_N loads 4× uint32 per thread = 16 bytes = 16 int8
  // elements per thread along K. ldmatrix.x4.N's lane mapping matches the
  // SM80_16x8x32 MMA's per-lane A/B operand expectation for K-inner data.
  using S2RCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementIn>;
  using S2RCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, ElementIn>;

  // R→S epilogue: CUTLASS's `Copy_Traits<SM90_U32x4_STSM_N>` is gated on
  // `CUTE_ARCH_STSM_SM90_ENABLED` (sm_90 only) even though the underlying
  // `stmatrix.m8n8.x4` PTX is sm_75+. To avoid that gate, use plain 32-bit
  // shared stores via UniversalCopy. Costs ~1 extra smem instruction per
  // bf16 pair but keeps us off the sm_90 path.
  using SmemCopyAtomC = Copy_Atom<UniversalCopy<uint32_t>, ElementOut>;

  // S→G epilogue: sm_89 has no TMA store; use plain vectorized st.global.v4.
  using S2GCopyAtomC = Copy_Atom<
      AutoVectorizingCopyWithAssumedAlignment<128>,
      ElementOut>;
  // 256 threads write the (bM,bN) bf16 tile via vectorized stores.
  //   Thread layout: (bM/16) × 32 threads = 8 × 32 = 256 at bM=128.
  //   Value layout : 16 rows × (bN/32) cols per thread.
  //   Coverage     : (bM/16)*16 rows × 32*(bN/32) cols = bM × bN. ✓
  //   At bN=128: value (16,4) → 64 elem/thread (= 128 B = 8× vec4 of bf16).
  //   At bN=256: value (16,8) → 128 elem/thread.
  static_assert(bM % 16 == 0, "S2GThreadLayoutC: bM must be a multiple of 16");
  static_assert(bN % 32 == 0, "S2GValueLayoutC: bN must be a multiple of 32");
  // With kNumWarps = bM/16, this assertion is self-satisfying for any
  // supported bM; kept as a sanity check on the (bM, kNumThreads) coupling.
  static_assert((bM / 16) * 32 == kNumThreads,
                "S2GThreadLayoutC: (bM/16)*32 must equal kNumThreads");
  using S2GThreadLayoutC = Layout<Shape<Int<bM/16>, _32>, Stride<_32, _1>>;
  using S2GValueLayoutC  = Layout<Shape<_16, Int<bN/32>>, Stride<Int<bN/32>, _1>>;
  using S2GCopyC = decltype(make_tiled_copy(
      S2GCopyAtomC{}, S2GThreadLayoutC{}, S2GValueLayoutC{}));

  // ---------- Scales (already sm_80-compatible on the sm_90a path) ----------
  using G2SScales_copy_op     = SM80_CP_ASYNC_CACHEALWAYS<ElementScale>;
  using G2SScales_copy_traits = Copy_Traits<G2SScales_copy_op>;
  using G2SScales_copy_atom   = Copy_Atom<G2SScales_copy_traits, ElementScale>;
  using SmemLayoutScaleA      = Layout<Shape<Int<bM>>, Stride<_1>>;
  using SmemLayoutScaleB      = Layout<Shape<Int<bN>>, Stride<_1>>;
  using G2SScalesCopyA        = decltype(make_tiled_copy(
      G2SScales_copy_atom{}, Layout<Shape<Int<bM>>, Stride<_1>>{},
      Layout<Shape<_1>, Stride<_1>>{}));
  using G2SScalesCopyB        = decltype(make_tiled_copy(
      G2SScales_copy_atom{}, Layout<Shape<Int<bN>>, Stride<_1>>{},
      Layout<Shape<_1>, Stride<_1>>{}));

  // ---------- Pipelines ----------
  // PipelineAsync uses cp.async.commit_group / wait_group<N> under the hood,
  // not mbarrier.expect_tx — so no transaction_bytes parameter on producer_commit().
  using MainloopPipeline = cutlass::PipelineAsync<kStages>;

  static constexpr int kDenoiseStages = 1;
  using DenoisePipeline  = cutlass::PipelineAsync<kDenoiseStages>;

  // ---------- Denoise MMA (fp16 → fp32) ----------
  using TiledMmaDenoise = decltype(make_tiled_mma(
      MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
      AtomLayoutMNK{}));
  static_assert(R % 16 == 0, "Denoise MMA atom k-dim is 16; R must be a multiple.");

  // Denoise smem (fp16 K-major). For R=128 → K-row = 256 bytes; Swizzle<3,3,3>
  // is the canonical fp16 atom.
  using SmemLayoutAtomDenoise = decltype(composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_8, Int<R>>, Stride<Int<R>, _1>>{}));

  using SmemLayoutEAL = decltype(tile_to_shape(
      SmemLayoutAtomDenoise{},
      Shape<Int<bM>, Int<R>, Int<kDenoiseStages>>{}));
  using SmemLayoutEBR = decltype(tile_to_shape(
      SmemLayoutAtomDenoise{},
      Shape<Int<bN>, Int<R>, Int<kDenoiseStages>>{}));
  using SmemLayoutAxEBL = decltype(tile_to_shape(
      SmemLayoutAtomDenoise{},
      Shape<Int<bM>, Int<R>, Int<kDenoiseStages>>{}));
  using SmemLayoutEARxBpEB = decltype(tile_to_shape(
      SmemLayoutAtomDenoise{},
      Shape<Int<bN>, Int<R>, Int<kDenoiseStages>>{}));

  // ---------- Shared storage ----------
  // Structurally identical to the sm_90a SharedStorage; only the pipeline
  // subtypes differ (PipelineAsync vs PipelineTmaAsync).
  struct SharedStorageDenoise : cute::aligned_struct<128> {
    union {
      struct {
        cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutA>,
                            cutlass::detail::alignment_for_swizzle(SmemLayoutA{})>
            smem_A;
        cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutB>,
                            cutlass::detail::alignment_for_swizzle(SmemLayoutB{})>
            smem_B;
      };
      struct {
        cute::array_aligned<ElementDenoise, cute::cosize_v<SmemLayoutAxEBL>,
                            cutlass::detail::alignment_for_swizzle(SmemLayoutAxEBL{})>
            smem_AxEBL;
        cute::array_aligned<ElementDenoise, cute::cosize_v<SmemLayoutEBR>,
                            cutlass::detail::alignment_for_swizzle(SmemLayoutEBR{})>
            smem_EBR;
        cute::array_aligned<ElementDenoise, cute::cosize_v<SmemLayoutEAL>,
                            cutlass::detail::alignment_for_swizzle(SmemLayoutEAL{})>
            smem_EAL;
        cute::array_aligned<ElementDenoise, cute::cosize_v<SmemLayoutEARxBpEB>,
                            cutlass::detail::alignment_for_swizzle(SmemLayoutEARxBpEB{})>
            smem_EARxBpEB;
      };
      cute::array_aligned<ElementOut, cute::cosize_v<SmemLayoutC>,
                          cutlass::detail::alignment_for_swizzle(SmemLayoutC{})>
          smem_C;
    };
    cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScaleA>,
                        cutlass::detail::alignment_for_swizzle(SmemLayoutScaleA{})>
        smem_scale_a;
    cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScaleB>,
                        cutlass::detail::alignment_for_swizzle(SmemLayoutScaleB{})>
        smem_scale_b;
    struct {
      typename MainloopPipeline::SharedStorage pipeline;
      typename DenoisePipeline::SharedStorage  AxEB_pipeline;
      typename DenoisePipeline::SharedStorage  EAxBpEB_pipeline;
    };
  };

  // Union layout: the mainloop A/B tiles overlap smem_C because the mainloop
  // ends with `cp_async_wait<0>() + __syncthreads()` (collective_mainloop_sm89.hpp
  // line ~278) before the epilogue's R2S stage writes smem_C in scale(). The
  // scale() body also opens with cp.async + __syncthreads() (after staging the
  // scale vectors). Both fences are block-wide and dominate the smem-union
  // transition, so no additional sync is required for the union to be safe.
  // This saves ~32 KB at (bM=128, bN=128) and ~64 KB at (bM=128, bN=256), which
  // is the difference between "fits in 99 KB optin" and "doesn't" at bN=256.
  struct SharedStorageNoDenoise : cute::aligned_struct<128> {
    union {
      struct {
        cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutA>,
                            cutlass::detail::alignment_for_swizzle(SmemLayoutA{})>
            smem_A;
        cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutB>,
                            cutlass::detail::alignment_for_swizzle(SmemLayoutB{})>
            smem_B;
      };
      cute::array_aligned<ElementOut, cute::cosize_v<SmemLayoutC>,
                          cutlass::detail::alignment_for_swizzle(SmemLayoutC{})>
          smem_C;
    };
    cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScaleA>,
                        cutlass::detail::alignment_for_swizzle(SmemLayoutScaleA{})>
        smem_scale_a;
    cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScaleB>,
                        cutlass::detail::alignment_for_swizzle(SmemLayoutScaleB{})>
        smem_scale_b;
    struct {
      typename MainloopPipeline::SharedStorage pipeline;
      typename DenoisePipeline::SharedStorage  AxEB_pipeline;
      typename DenoisePipeline::SharedStorage  EAxBpEB_pipeline;
    };
  };

  // ---------- Register-resident denoise SharedStorage ----------
  // When kRegisterResidentDenoise=true, the four (bM/bN × R) fp16 denoise
  // tiles are NOT staged in smem. They are streamed per-thread directly from
  // gmem (L1-cached) inside the denoise epilogue. The shared storage shape
  // becomes identical to the NoDenoise arm: union { A,B | C }. This is what
  // makes R=128 at bM=128 bN=128 fit on sm_89 (saves ~128 KB vs the resident
  // path that would otherwise overflow the 99 KB opt-in cap).
  struct SharedStorageDenoiseRegResident : cute::aligned_struct<128> {
    union {
      struct {
        cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutA>,
                            cutlass::detail::alignment_for_swizzle(SmemLayoutA{})>
            smem_A;
        cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutB>,
                            cutlass::detail::alignment_for_swizzle(SmemLayoutB{})>
            smem_B;
      };
      cute::array_aligned<ElementOut, cute::cosize_v<SmemLayoutC>,
                          cutlass::detail::alignment_for_swizzle(SmemLayoutC{})>
          smem_C;
    };
    cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScaleA>,
                        cutlass::detail::alignment_for_swizzle(SmemLayoutScaleA{})>
        smem_scale_a;
    cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScaleB>,
                        cutlass::detail::alignment_for_swizzle(SmemLayoutScaleB{})>
        smem_scale_b;
    struct {
      typename MainloopPipeline::SharedStorage pipeline;
      typename DenoisePipeline::SharedStorage  AxEB_pipeline;
      typename DenoisePipeline::SharedStorage  EAxBpEB_pipeline;
    };
  };

  using SharedStorage = cute::conditional_t<
      SkipDenoising, SharedStorageNoDenoise,
      cute::conditional_t<kRegisterResidentDenoise,
                          SharedStorageDenoiseRegResident,
                          SharedStorageDenoise>>;

  // ---------- Smem budget guard ----------
  static_assert(
      2 * cute::cosize_v<SmemLayoutA> * sizeof(ElementIn) <= 100 * 1024,
      "sm_89 dynamic smem cap is ~100 KB; A+B tile per stage exceeds it. "
      "Reduce kStages, bM, bN, or bK.");
};

}  // namespace pearl
