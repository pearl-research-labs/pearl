// SPDX-License-Identifier: see LICENSE
//
// sm_89 port of pearl_noisingB_kernel.h (Hopper sm_90a).
//
// Differences vs. Hopper:
//   - Unified-warp model: one warpgroup (128 threads) does loads + both GEMMs
//     sequentially per k_block. Hopper splits into producer + 2 consumer WGs;
//     sm_89 has no setmaxnreg to redistribute regs, so we collapse to 1 WG.
//   - TMA loads replaced with `cp.async` + `cp_async_fence` + `cp_async_wait<>`.
//   - TMA stores replaced with vectorized st.global via
//     AutoVectorizingCopyWithAssumedAlignment<128>.
//   - WGMMA replaced with SM80_16x8x32_S32S8S8S32_TN; gemm is synchronous,
//     no warpgroup_arrive/commit_batch/wait<0>.
//   - smem swizzle: Swizzle<2,4,3> over Shape<_16,_64> (canonical sm_80 int8
//     K-major, matches the noiseless kernel_traits_sm89.hpp atom).
//   - Hopper's "set accum to A then MMA" trick (half-MMA + STSM) is replaced
//     with a simpler smem-roundtrip: write the int8 EBR@EBL result to
//     smem_BpEB, then block-wide add smem_B to smem_BpEB, then S2G.
//
// Data flow per k_block (k_iter in [0..total_iters)):
//   1. Wait for B[stage_in], EAR[stage_in], EBL[stage_in] in smem.
//   2. tCrBpEB (int32) := EBR @ EBL[stage_in] via TiledMmaNKR.
//   3. Convert int32 -> int8 with WRAPPING semantics (matching torch.int8
//      cast). CUTLASS's NumericArrayConverter SATURATES, which diverges from
//      the Python reference whenever a fragment sum overflows int8 range.
//   4. R2S: write tCrBpEB_int8 -> smem_BpEB[stage_out] using
//      make_tiled_copy_C(UniversalCopy<int8>, mma_nkr). 1 byte per issue to
//      sidestep the non-unit-stride C-fragment layout under K-major swizzle.
//   5. __syncthreads.
//   6. Block-wide elementwise: smem_BpEB[i] += smem_B[i] for the (bN,bK) tile.
//      Both buffers use the same swizzle, so equal physical offsets map to
//      equal logical (n,k) coords. Vec16 chunks per thread.
//   7. __syncthreads.
//   8. S2G: smem_BpEB[stage_out] -> gmem BpEB[k_block].
//   9. ldmatrix loads from smem_BpEB[stage_out] (A op of NRK MMA) and
//      smem_EAR[stage_in] (B op) -> tCrEARxBpEB accum.
//  10. Issue next cp.async for next stage's B, EAR, EBL.
//
// After loop:
//   - R2S tCrEARxBpEB -> smem_EARxBpEB (scale + cast for fp16).
//   - S2G smem_EARxBpEB -> gmem EARxBpEB.

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_sm75.hpp"
#include "cute/atom/copy_traits_sm80.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm80.hpp"
#include "cute/swizzle.hpp"
#include "cute/tensor.hpp"

#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/detail/layout.hpp>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "pearl_gemm_constants.hpp"
#include "utils.h"

namespace pearl {
namespace sm89 {

using namespace cute;

template <class TileShape_NRK_, int kNumThreads_, class Element,
          class ElementDenoise, int kStages_, bool IsEvenK, bool NoReduction>
class NoisingKernelBSm89 {

 public:
  using ElementScale = float;
  using ElementAccum = int32_t;
  static_assert(cute::is_same_v<Element, int8_t>);
  static constexpr int denoise_dtype_bits = cute::sizeof_bits_v<ElementDenoise>;
  static_assert(denoise_dtype_bits == 16 || denoise_dtype_bits == 32,
                "Denoise dtype size must be 16 or 32 bits");
  static_assert(denoise_dtype_bits == 32 || NoReduction,
                "Don't support reduction with fp16");

  using TileShape_NRK = TileShape_NRK_;
  using ArchTag = cutlass::arch::Sm89;
  static constexpr int kBlockN = get<0>(TileShape_NRK{});  // bN
  static constexpr int R       = get<1>(TileShape_NRK{});  // R
  static constexpr int kBlockK = get<2>(TileShape_NRK{});  // bK
  using TileShape_NKR = Shape<Int<kBlockN>, Int<kBlockK>, Int<R>>;

  static constexpr int kNumThreads = kNumThreads_;
  static_assert(kNumThreads == 128, "sm_89 noisingB uses one warpgroup");
  static constexpr uint32_t MaxThreadsPerBlock = kNumThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int kStages = kStages_;
  static constexpr int kStagesOut = kStages;

  static_assert(kBlockN == 64);
  static_assert(R == 64 || R == 128);
  static_assert(kBlockK == 64);

  // Cluster stubs (sm_89: no clusters)
  static constexpr int kClusterM = 1;
  static constexpr int kClusterN = 1;
  using ClusterShape = Shape<_1, _1, _1>;

  // ---------- MMA atoms ----------
  // TiledMmaNRK: produces (bN, R) from (bN, bK_inner) x (R, bK_inner).
  //   Atom 16x8x32. AtomLayout<2,2,1>, Tile<32,32,32> -> per-Tile=32x32x32.
  //   tile_to_shape covers (bN=64, R=64, bK=64) in 2x2x2 atom repeats.
  using AtomLayoutMNK = Layout<Shape<_2, _2, _1>>;
  using TiledMmaNRK = decltype(make_tiled_mma(
      MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>{},
      AtomLayoutMNK{},
      Tile<_32, _32, _32>{}));

  // TiledMmaNKR: produces (bN, bK) from (bN, R_inner) x (bK, R_inner).
  using TiledMmaNKR = decltype(make_tiled_mma(
      MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>{},
      AtomLayoutMNK{},
      Tile<_32, _32, _32>{}));

  // ---------- Smem layouts (K-major int8, canonical sm_80 swizzle) ----------
  using SmemLayoutAtomK = decltype(composition(
      Swizzle<2, 4, 3>{},
      Layout<Shape<_16, _64>, Stride<_64, _1>>{}));

  // B: bN x bK (K-major), pipelined
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomK{}, Shape<Int<kBlockN>, Int<kBlockK>, Int<kStages>>{}));
  // EAR: R x bK (K-major), pipelined
  using SmemLayoutEAR = decltype(tile_to_shape(
      SmemLayoutAtomK{}, Shape<Int<R>, Int<kBlockK>, Int<kStages>>{}));
  // EBR: bN x R (R-major, single-buffer)
  using SmemLayoutEBR = decltype(tile_to_shape(
      SmemLayoutAtomK{}, Shape<Int<kBlockN>, Int<R>>{}));
  // EBL: bK x R (R-major), pipelined
  using SmemLayoutEBL = decltype(tile_to_shape(
      SmemLayoutAtomK{}, Shape<Int<kBlockK>, Int<R>, Int<kStages>>{}));
  // BpEB: bN x bK (K-major), pipelined for double-buffered output staging.
  using SmemLayoutBpEB = decltype(tile_to_shape(
      SmemLayoutAtomK{},
      Shape<Int<kBlockN>, Int<kBlockK>, Int<kStagesOut>>{}));

  // EARxBpEB: bN x R, R-major, single-buffer (written once at the end).
  using SmemLayoutEARxBpEB = Layout<
      Shape<Int<kBlockN>, Int<R>>, Stride<Int<R>, _1>>;

  // ---------- Copy atoms ----------
  // G->S 16B cp.async vectorized over K (16 int8 per thread per issue).
  using G2SCopyAtomAB = Copy_Atom<
      Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>>, Element>;

  static constexpr int kG2S_K_threads = kBlockK / 16;  // 4 for bK=64
  static constexpr int kG2S_M_threads = kNumThreads / kG2S_K_threads;  // 32
  using G2SThreadLayoutAB = Layout<
      Shape<Int<kG2S_M_threads>, Int<kG2S_K_threads>>,
      Stride<Int<kG2S_K_threads>, _1>>;
  using G2SValueLayoutAB = Layout<Shape<_1, _16>>;
  using G2SCopyAB = decltype(make_tiled_copy(
      G2SCopyAtomAB{}, G2SThreadLayoutAB{}, G2SValueLayoutAB{}));

  static constexpr int kG2S_R_threads = R / 16;  // 4 for R=64, 8 for R=128
  static constexpr int kG2S_N_threads = kNumThreads / kG2S_R_threads;
  using G2SThreadLayoutRMaj = Layout<
      Shape<Int<kG2S_N_threads>, Int<kG2S_R_threads>>,
      Stride<Int<kG2S_R_threads>, _1>>;
  using G2SCopyRMaj = decltype(make_tiled_copy(
      G2SCopyAtomAB{}, G2SThreadLayoutRMaj{}, G2SValueLayoutAB{}));

  // S->R ldmatrix for int8 K-major operands of MMA.
  using S2RCopyAtomAB = Copy_Atom<SM75_U32x4_LDSM_N, Element>;

  // R->S for BpEB result: the C-fragment layout per thread per SM80_16x8x32
  // atom has each thread holding 4 int8 (post int32->int8 convert) at output
  // positions with non-unit stride. UniversalCopy<int8_t> is safe — 1 byte
  // per issue, fully respects the layout's non-contiguous strides.
  using R2SCopyAtomBpEB = Copy_Atom<UniversalCopy<int8_t>, Element>;

  // R->S for EARxBpEB (final write to smem): generic vectorized.
  using CopyOpR2S = AutoVectorizingCopyWithAssumedAlignment<128>;
  using SmemCopyAtomEARxBpEB = Copy_Atom<CopyOpR2S, ElementDenoise>;

  // S->G for BpEB (per k_block): vectorized st.global.v4 (16B per thread).
  using S2GCopyAtomBpEB = Copy_Atom<
      AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
  using S2GThreadLayoutBpEB = G2SThreadLayoutAB;
  using S2GValueLayoutBpEB  = G2SValueLayoutAB;
  using S2GCopyBpEB = decltype(make_tiled_copy(
      S2GCopyAtomBpEB{}, S2GThreadLayoutBpEB{}, S2GValueLayoutBpEB{}));

  // S->G for EARxBpEB: vectorized 16B/thread st.global.v4.
  using S2GCopyAtomEARxBpEB = Copy_Atom<
      AutoVectorizingCopyWithAssumedAlignment<128>, ElementDenoise>;
  static constexpr int kS2G_R_threads_out = R / 16;
  static constexpr int kS2G_N_threads_out = kNumThreads / kS2G_R_threads_out;
  using S2GThreadLayoutEARxBpEB = Layout<
      Shape<Int<kS2G_N_threads_out>, Int<kS2G_R_threads_out>>,
      Stride<Int<kS2G_R_threads_out>, _1>>;
  static constexpr int kS2G_elt_per_issue =
      16 / sizeof(ElementDenoise);
  using S2GValueLayoutEARxBpEB = Layout<
      Shape<_1, Int<kS2G_elt_per_issue>>>;
  using S2GCopyEARxBpEB = decltype(make_tiled_copy(
      S2GCopyAtomEARxBpEB{}, S2GThreadLayoutEARxBpEB{},
      S2GValueLayoutEARxBpEB{}));

  // ---------- Shared storage ----------
  static constexpr size_t Alignment = 128;

  struct SharedStorage : cute::aligned_struct<Alignment> {
    union {
      // mainloop: B, EAR, EBR, EBL, BpEB
      struct {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutB>, Alignment>
            smem_B;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutEAR>, Alignment>
            smem_EAR;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutEBR>, Alignment>
            smem_EBR;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutEBL>, Alignment>
            smem_EBL;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutBpEB>, Alignment>
            smem_BpEB;
      };
      // epilogue: EARxBpEB
      cute::array_aligned<ElementDenoise, cute::cosize_v<SmemLayoutEARxBpEB>,
                          Alignment>
          smem_EARxBpEB;
    };
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // ---------- Args / Params ----------
  using ShapeT = cute::Shape<int32_t, int32_t>;
  using StrideT = cute::Shape<int32_t, _1>;
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  struct Arguments {
    Element const* const ptr_B;
    Element const* const ptr_EBR;
    Element const* const ptr_EAR;
    Element const* const ptr_EBL;
    Element* const ptr_BpEB;
    ElementDenoise* const ptr_EARxBpEB;
    int n;
    int k;
    int num_k_blocks;
    int total_k_blocks;
  };

  struct Params {
    Element const* const ptr_B;
    Element const* const ptr_EBR;
    Element const* const ptr_EBL;
    Element const* const ptr_EAR;
    Element* const ptr_BpEB;
    ElementDenoise* const ptr_EARxBpEB;
    int n;
    int k;
    int num_k_blocks;
    int total_k_blocks;
    LayoutT layout_B;
    LayoutT layout_EAR;
    LayoutT layout_EBR;
    LayoutT layout_EBL;
    LayoutT layout_EARxBpEB;
    LayoutT layout_BpEB;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    LayoutT layout_B =
        make_layout(make_shape(args.n, args.k), make_stride(args.k, _1{}));
    LayoutT layout_EAR =
        make_layout(make_shape(R, args.k), make_stride(args.k, _1{}));
    LayoutT layout_EBR =
        make_layout(make_shape(args.n, R), make_stride(R, _1{}));
    LayoutT layout_EBL =
        make_layout(make_shape(args.k, R), make_stride(R, _1{}));
    LayoutT layout_BpEB =
        make_layout(make_shape(args.n, args.k), make_stride(args.k, _1{}));
    LayoutT layout_EARxBpEB =
        make_layout(make_shape(args.n, R), make_stride(R, _1{}));

    return {.ptr_B = args.ptr_B,
            .ptr_EBR = args.ptr_EBR,
            .ptr_EBL = args.ptr_EBL,
            .ptr_EAR = args.ptr_EAR,
            .ptr_BpEB = args.ptr_BpEB,
            .ptr_EARxBpEB = args.ptr_EARxBpEB,
            .n = args.n,
            .k = args.k,
            .num_k_blocks = args.num_k_blocks,
            .total_k_blocks = args.total_k_blocks,
            .layout_B = layout_B,
            .layout_EAR = layout_EAR,
            .layout_EBR = layout_EBR,
            .layout_EBL = layout_EBL,
            .layout_EARxBpEB = layout_EARxBpEB,
            .layout_BpEB = layout_BpEB};
  }

  static dim3 get_grid_shape(Params const& params) {
    if constexpr (NoReduction) {
      return dim3(cutlass::ceil_div(params.n, kBlockN), 1, 1);
    } else {
      return dim3(cutlass::ceil_div(params.n, kBlockN),
                  cutlass::ceil_div(params.k,
                                    params.num_k_blocks * kBlockK),
                  1);
    }
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  // ===========================================================================
  // Device entry point.
  // ===========================================================================
  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    auto& shared_storage =
        *reinterpret_cast<SharedStorage*>(smem_buf);
    int const tid = threadIdx.x;

    int const n_block     = blockIdx.x;
    int const k_block_min = blockIdx.y * params.num_k_blocks;
    int const num_k_blocks_cta =
        cute::min(params.num_k_blocks, params.total_k_blocks - k_block_min);

    // GMEM tensors
    Tensor mB    = make_tensor(make_gmem_ptr(params.ptr_B), params.layout_B);
    Tensor mEAR  = make_tensor(make_gmem_ptr(params.ptr_EAR), params.layout_EAR);
    Tensor mEBR  = make_tensor(make_gmem_ptr(params.ptr_EBR), params.layout_EBR);
    Tensor mEBL  = make_tensor(make_gmem_ptr(params.ptr_EBL), params.layout_EBL);
    Tensor mBpEB = make_tensor(make_gmem_ptr(params.ptr_BpEB),
                               params.layout_BpEB);

    // CTA-local tiles
    Tensor gB =
        local_tile(mB, select<0, 2>(TileShape_NRK{}), make_coord(n_block, _));
    Tensor gEAR =
        local_tile(mEAR, select<1, 2>(TileShape_NRK{}), make_coord(_0{}, _));
    Tensor gEBR =
        local_tile(mEBR, select<0, 1>(TileShape_NRK{}),
                   make_coord(n_block, _0{}));
    Tensor gEBL =
        local_tile(mEBL, select<2, 1>(TileShape_NRK{}), make_coord(_, _0{}));
    Tensor gBpEB =
        local_tile(mBpEB, select<0, 2>(TileShape_NRK{}),
                   make_coord(n_block, _));

    // SMEM tensors
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_B.data()),
                            SmemLayoutB{});
    Tensor sEAR = make_tensor(make_smem_ptr(shared_storage.smem_EAR.data()),
                              SmemLayoutEAR{});
    Tensor sEBR = make_tensor(make_smem_ptr(shared_storage.smem_EBR.data()),
                              SmemLayoutEBR{});
    Tensor sEBL = make_tensor(make_smem_ptr(shared_storage.smem_EBL.data()),
                              SmemLayoutEBL{});
    Tensor sBpEB = make_tensor(make_smem_ptr(shared_storage.smem_BpEB.data()),
                               SmemLayoutBpEB{});

    // ---------- G->S copy partitions ----------
    G2SCopyAB copy_KMaj;
    auto thr_copy_KMaj = copy_KMaj.get_slice(tid);
    auto tBgB    = thr_copy_KMaj.partition_S(gB);
    auto tBsB    = thr_copy_KMaj.partition_D(sB);
    auto tEARgEAR = thr_copy_KMaj.partition_S(gEAR);
    auto tEARsEAR = thr_copy_KMaj.partition_D(sEAR);

    G2SCopyRMaj copy_RMaj;
    auto thr_copy_RMaj = copy_RMaj.get_slice(tid);
    auto tEBRgEBR = thr_copy_RMaj.partition_S(gEBR);
    auto tEBRsEBR = thr_copy_RMaj.partition_D(sEBR);
    auto tEBLgEBL = thr_copy_RMaj.partition_S(gEBL);
    auto tEBLsEBL = thr_copy_RMaj.partition_D(sEBL);

    // ---------- MMA setups ----------
    TiledMmaNRK mma_nrk;
    auto thr_mma_nrk = mma_nrk.get_thread_slice(tid);
    Tensor tCsBpEB_nrk = thr_mma_nrk.partition_A(sBpEB);   // (MMA,N,K,P)
    Tensor tCsEAR_nrk  = thr_mma_nrk.partition_B(sEAR);    // (MMA,R,K,P)

    TiledMmaNKR mma_nkr;
    auto thr_mma_nkr = mma_nkr.get_thread_slice(tid);
    Tensor tCsEBR_nkr = thr_mma_nkr.partition_A(sEBR);     // (MMA,N,R)
    Tensor tCsEBL_nkr = thr_mma_nkr.partition_B(sEBL);     // (MMA,K,R,P)

    // Register fragments (single-stage, sized to one MMA tile)
    Tensor tCrBpEB_nrk = thr_mma_nrk.make_fragment_A(tCsBpEB_nrk(_, _, _, 0));
    Tensor tCrEAR_nrk  = thr_mma_nrk.make_fragment_B(tCsEAR_nrk(_, _, _, 0));
    Tensor tCrEBR_nkr  = thr_mma_nkr.make_fragment_A(tCsEBR_nkr);
    Tensor tCrEBL_nkr  = thr_mma_nkr.make_fragment_B(tCsEBL_nkr(_, _, _, 0));

    // Accumulators
    Tensor tCrEARxBpEB =
        partition_fragment_C(mma_nrk, select<0, 1>(TileShape_NRK{}));
    Tensor tCrBpEB_acc =
        partition_fragment_C(mma_nkr, select<0, 1>(TileShape_NKR{}));
    Tensor tCrBpEB_int8 = make_fragment_like<Element>(tCrBpEB_acc);
    clear(tCrEARxBpEB);

    // ---------- S->R retile (ldmatrix) ----------
    auto s2r_BpEB = make_tiled_copy_A(S2RCopyAtomAB{}, mma_nrk);
    auto s2r_EAR  = make_tiled_copy_B(S2RCopyAtomAB{}, mma_nrk);
    auto s2r_EBR  = make_tiled_copy_A(S2RCopyAtomAB{}, mma_nkr);
    auto s2r_EBL  = make_tiled_copy_B(S2RCopyAtomAB{}, mma_nkr);

    auto s2r_thr_BpEB = s2r_BpEB.get_slice(tid);
    auto s2r_thr_EAR  = s2r_EAR.get_slice(tid);
    auto s2r_thr_EBR  = s2r_EBR.get_slice(tid);
    auto s2r_thr_EBL  = s2r_EBL.get_slice(tid);

    auto tXsBpEB = s2r_thr_BpEB.partition_S(sBpEB);
    auto tXrBpEB = s2r_thr_BpEB.retile_D(tCrBpEB_nrk);
    auto tXsEAR  = s2r_thr_EAR.partition_S(sEAR);
    auto tXrEAR  = s2r_thr_EAR.retile_D(tCrEAR_nrk);
    auto tXsEBR  = s2r_thr_EBR.partition_S(sEBR);
    auto tXrEBR  = s2r_thr_EBR.retile_D(tCrEBR_nkr);
    auto tXsEBL  = s2r_thr_EBL.partition_S(sEBL);
    auto tXrEBL  = s2r_thr_EBL.retile_D(tCrEBL_nkr);

    // ---------- R->S of tCrBpEB_int8 to smem_BpEB ----------
    auto r2s_BpEB_copy = make_tiled_copy_C(R2SCopyAtomBpEB{}, mma_nkr);
    auto r2s_thr_BpEB = r2s_BpEB_copy.get_slice(tid);
    auto tR2SsBpEB = r2s_thr_BpEB.partition_D(sBpEB);   // (CPY,CPY_N,CPY_K,P)

    // ---------- S->G of BpEB (per k_block) ----------
    S2GCopyBpEB s2g_BpEB;
    auto s2g_thr_BpEB = s2g_BpEB.get_slice(tid);
    auto tSGsBpEB = s2g_thr_BpEB.partition_S(sBpEB);
    auto tSGgBpEB = s2g_thr_BpEB.partition_D(gBpEB);

    // ---------- Number of k_blocks to iterate ----------
    int const num_k_iters = num_k_blocks_cta;
    int const total_iters = num_k_iters;

    // ---------- PROLOGUE: EBR (once) + first (kStages-1) stages ----------
    copy(copy_RMaj, tEBRgEBR, tEBRsEBR);
    int k_block_load = k_block_min;
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < kStages - 1; ++k_pipe) {
      if (k_pipe < num_k_iters) {
        copy(copy_KMaj, tBgB(_, _, _, k_block_load),
             tBsB(_, _, _, k_pipe));
        copy(copy_KMaj, tEARgEAR(_, _, _, k_block_load),
             tEARsEAR(_, _, _, k_pipe));
        copy(copy_RMaj, tEBLgEBL(_, _, _, k_block_load),
             tEBLsEBL(_, _, _, k_pipe));
        ++k_block_load;
      }
      cp_async_fence();
    }

    // Wait for stage 0 to land.
    cp_async_wait<kStages - 2>();
    __syncthreads();

    // Load EBR -> regs once (it doesn't change across k_blocks).
    copy(s2r_EBR, tXsEBR, tXrEBR);

    int smem_pipe_read  = 0;
    int smem_pipe_write = kStages - 1;
    // ---------- MAIN LOOP ----------
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_iter = 0; k_iter < total_iters; ++k_iter) {
      int const k_block = k_block_min + k_iter;
      int const stage_in  = smem_pipe_read;
      int const stage_out = k_iter % kStagesOut;

      // Step 1: compute_BpEB = EBR @ EBL[stage_in].
      clear(tCrBpEB_acc);
      copy(s2r_EBL, tXsEBL(_, _, _, stage_in), tXrEBL);
      cute::gemm(mma_nkr, tCrEBR_nkr, tCrEBL_nkr, tCrBpEB_acc);

      // Step 2: convert int32 -> int8 with WRAPPING semantics (matching
      // torch.int8 cast). CUTLASS NumericArrayConverter saturates; we need
      // modular wrap so that (e.g.) int32(133) -> int8(-123).
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tCrBpEB_acc); ++i) {
        tCrBpEB_int8(i) = static_cast<int8_t>(
            static_cast<int32_t>(tCrBpEB_acc(i)) & 0xff);
      }

      // Step 3: R2S write tCrBpEB_int8 -> smem_BpEB[stage_out].
      auto tR2SrBpEB = r2s_thr_BpEB.retile_S(tCrBpEB_int8);
      copy(r2s_BpEB_copy, tR2SrBpEB, tR2SsBpEB(_, _, _, stage_out));
      __syncthreads();

      // Step 4: block-wide elementwise add: smem_BpEB[stage_out][i] +=
      // smem_B[stage_in][i] over the (bN,bK) tile.
      // Both smem buffers use the SAME SmemLayoutAtomK swizzle, so the
      // physical byte at offset i in smem_BpEB[stage_out] corresponds to the
      // same logical (n,k) coord as offset i in smem_B[stage_in]. We can
      // iterate over the flat physical storage directly using vec16 chunks.
      {
        constexpr int kStageStrideElts =
            cute::cosize_v<SmemLayoutB> / kStages;
        static_assert(cute::cosize_v<SmemLayoutB> % kStages == 0,
                      "Per-stage smem size must divide evenly");
        constexpr int kStageStrideEltsBpEB =
            cute::cosize_v<SmemLayoutBpEB> / kStagesOut;
        static_assert(cute::cosize_v<SmemLayoutBpEB> % kStagesOut == 0,
                      "Per-stage smem size must divide evenly (BpEB)");
        int8_t* p_BpEB = shared_storage.smem_BpEB.data() +
                         stage_out * kStageStrideEltsBpEB;
        int8_t const* p_B = shared_storage.smem_B.data() +
                            stage_in * kStageStrideElts;
        constexpr int kTotalElts = kBlockN * kBlockK;
        constexpr int kVecElts = 16;
        constexpr int kIters = kTotalElts / (kNumThreads * kVecElts);
        CUTLASS_PRAGMA_UNROLL
        for (int it = 0; it < kIters; ++it) {
          int const off = (it * kNumThreads + tid) * kVecElts;
          int4 const vB = *reinterpret_cast<int4 const*>(p_B + off);
          int4 vBpEB    = *reinterpret_cast<int4 const*>(p_BpEB + off);
          int8_t* a = reinterpret_cast<int8_t*>(&vBpEB);
          int8_t const* b = reinterpret_cast<int8_t const*>(&vB);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < kVecElts; ++j) {
            a[j] = int8_t(int(a[j]) + int(b[j]));
          }
          *reinterpret_cast<int4*>(p_BpEB + off) = vBpEB;
        }
      }
      __syncthreads();

      // Step 5: S2G BpEB[stage_out] -> gmem.
      copy(s2g_BpEB, tSGsBpEB(_, _, _, stage_out),
           tSGgBpEB(_, _, _, k_block));

      // Step 6: EARxBpEB += BpEB[stage_out] @ EAR[stage_in].
      copy(s2r_BpEB, tXsBpEB(_, _, _, stage_out), tXrBpEB);
      copy(s2r_EAR, tXsEAR(_, _, _, stage_in), tXrEAR);
      cute::gemm(mma_nrk, tCrBpEB_nrk, tCrEAR_nrk, tCrEARxBpEB);

      // Step 7: issue next cp.async stage.
      if (k_iter + (kStages - 1) < num_k_iters) {
        int const k_block_to_load = k_block_min + k_iter + (kStages - 1);
        copy(copy_KMaj, tBgB(_, _, _, k_block_to_load),
             tBsB(_, _, _, smem_pipe_write));
        copy(copy_KMaj, tEARgEAR(_, _, _, k_block_to_load),
             tEARsEAR(_, _, _, smem_pipe_write));
        copy(copy_RMaj, tEBLgEBL(_, _, _, k_block_to_load),
             tEBLsEBL(_, _, _, smem_pipe_write));
      }
      cp_async_fence();
      smem_pipe_write = (smem_pipe_write + 1) % kStages;
      smem_pipe_read  = (smem_pipe_read  + 1) % kStages;

      // Wait for next stage.
      if (k_iter + 1 < total_iters) {
        cp_async_wait<kStages - 2>();
        __syncthreads();
      }
    }  // end main loop

    // Drain
    cp_async_wait<0>();
    __syncthreads();

    // ---------- EARxBpEB epilogue ----------
    Tensor sEARxBpEB =
        make_tensor(make_smem_ptr(shared_storage.smem_EARxBpEB.data()),
                    SmemLayoutEARxBpEB{});

    auto r2s_EARxBpEB =
        make_tiled_copy_C(SmemCopyAtomEARxBpEB{}, mma_nrk);
    auto r2s_thr_EARxBpEB = r2s_EARxBpEB.get_slice(tid);
    auto tR2SsEARxBpEB = r2s_thr_EARxBpEB.partition_D(sEARxBpEB);

    if constexpr (!cute::is_same_v<ElementDenoise, int32_t>) {
      Tensor tCrEARxBpEB_out =
          make_fragment_like<ElementDenoise>(tCrEARxBpEB);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tCrEARxBpEB); ++i) {
        tCrEARxBpEB_out(i) = static_cast<ElementDenoise>(
            static_cast<ElementScale>(tCrEARxBpEB(i)) /
            static_cast<ElementScale>(pearl::kEARxBpEBScaleFactor));
      }
      auto tR2SrEARxBpEB = r2s_thr_EARxBpEB.retile_S(tCrEARxBpEB_out);
      copy(r2s_EARxBpEB, tR2SrEARxBpEB, tR2SsEARxBpEB);
    } else {
      auto tR2SrEARxBpEB = r2s_thr_EARxBpEB.retile_S(tCrEARxBpEB);
      copy(r2s_EARxBpEB, tR2SrEARxBpEB, tR2SsEARxBpEB);
    }

    __syncthreads();

    // S->G EARxBpEB
    Tensor mEARxBpEB = make_tensor(make_gmem_ptr(params.ptr_EARxBpEB),
                                   params.layout_EARxBpEB);
    Tensor gEARxBpEB = local_tile(mEARxBpEB, select<0, 1>(TileShape_NRK{}),
                                  make_coord(n_block, _0{}));

    S2GCopyEARxBpEB s2g_out;
    auto s2g_thr_out = s2g_out.get_slice(tid);
    auto tSGsOut = s2g_thr_out.partition_S(sEARxBpEB);
    auto tSGgOut = s2g_thr_out.partition_D(gEARxBpEB);
    copy(s2g_out, tSGsOut, tSGgOut);
  }
};

}  // namespace sm89
}  // namespace pearl
