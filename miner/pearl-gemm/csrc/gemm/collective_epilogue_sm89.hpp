// SPDX-License-Identifier: see LICENSE
//
// sm_89 epilogue collective for pearl-gemm.
// Path: int32 accumulator (in regs) -> apply per-row A_scales + per-col B_scales
//       -> cast to bf16 -> stmatrix into smem -> vectorized st.global.v4 to gmem.
//
// STATUS: noiseless path (scale + store) wired and bit-exact on sm_89. Denoise
// hooks (load_denoise + denoise) ported from the Hopper version per
// SM89_PORT_SPEC.md §4 — guarded by `if constexpr (!KTraits::SkipDenoising)`
// so the noiseless instantiation remains a no-op.
// Reference: collective_epilogue.hpp scale() (Hopper version) lines 460-550,
// load_denoise() lines 233-337, denoise() lines 361-457.
//
// Sm_89 substitutions vs Hopper:
//   SM90_TMA_STORE         -> S2GCopyC (UniversalCopy<uint128_t>, vectorized st.global)
//   tma_store_arrive       -> __syncthreads()  (block-wide commit on sm_89)
//   NamedBarrier(Epilogue) -> __syncthreads()
//   TMA load + mbarrier    -> cp.async (SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>) + fence + wait
//   make_fragment_A (WGMMA desc) -> partition_fragment_A + LDSM smem->regs
//   warpgroup_arrive/commit/wait -> dropped (sm_89 mma.sync is synchronous)
//   DenoisePipeline (consumer_wait/release) -> cp_async_wait<0> + __syncthreads
//
// The scale() body, the convert_type cast, and the R2S stmatrix step are
// arch-agnostic and ported verbatim from the Hopper version.

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_sm75.hpp"
#include "cute/atom/copy_traits_sm80.hpp"
#include "cute/tensor.hpp"

#include <cutlass/arch/barrier.h>
#include "cutlass/cutlass.h"

#include "convert_util.h"  // convert_type<>
#include "named_barrier.hpp"
#include "pearl_gemm_constants.hpp"  // kIntToFp16ScaleFactor

namespace pearl {
using namespace cute;

template <typename KTraits>
struct CollectiveEpilogueSm89 {
  using ElementIn       = typename KTraits::ElementIn;
  using ElementOut      = typename KTraits::ElementOut;
  using ElementOutput   = ElementOut;   // alias used in Hopper API
  using ElementDenoise  = typename KTraits::ElementDenoise;
  using ElementAccum    = float;  // fp32 (matches scale() input dtype)
  using ElementScale    = typename KTraits::ElementScale;
  using TileShape_MNK   = typename KTraits::TileShape_MNK;
  using TileShape_MNR   = typename KTraits::TileShape_MNR;
  using SmemLayoutC     = typename KTraits::SmemLayoutC;
  using SmemLayoutScaleA = typename KTraits::SmemLayoutScaleA;
  using SmemLayoutScaleB = typename KTraits::SmemLayoutScaleB;
  using SmemLayoutEAL       = typename KTraits::SmemLayoutEAL;
  using SmemLayoutEBR       = typename KTraits::SmemLayoutEBR;
  using SmemLayoutAxEBL     = typename KTraits::SmemLayoutAxEBL;
  using SmemLayoutEARxBpEB  = typename KTraits::SmemLayoutEARxBpEB;
  using SmemCopyAtomC   = typename KTraits::SmemCopyAtomC;
  using S2GCopyC        = typename KTraits::S2GCopyC;
  using G2SScalesCopyA  = typename KTraits::G2SScalesCopyA;
  using G2SScalesCopyB  = typename KTraits::G2SScalesCopyB;
  using ProblemShape    = typename KTraits::ProblemShape;
  using TiledMmaDenoise = typename KTraits::TiledMmaDenoise;

  static constexpr int bM             = KTraits::bM;
  static constexpr int bN             = KTraits::bN;
  static constexpr int R              = KTraits::R;
  static constexpr int kRTile         = KTraits::kRTile;
  static constexpr int kNumRStrips    = KTraits::kNumRStrips;
  static constexpr int kNumMmaThreads = KTraits::kNumMmaThreads;
  static constexpr int kNumThreads    = KTraits::kNumThreads;

  // ---------- Denoise G2S cp.async copy ----------
  // Each denoise tensor is (bM or bN, R) fp16, R-major in gmem. We stage ONE
  // R-strip of width kRTile at a time. Vectorize at 16B (= 8 fp16) per thread
  // along R. Thread layout has kThrR threads in the kRTile strip, kThrM threads
  // in M, with kThrR * 8 = kRTile and kThrM * kThrR = kNumThreads.
  static constexpr int kG2SDenoise_ElementsPerThread = 8;  // 16B / fp16
  static_assert(kRTile % kG2SDenoise_ElementsPerThread == 0,
                "kRTile must be a multiple of 8 (16B vectorized cp.async).");
  static constexpr int kG2SDenoise_R_threads =
      kRTile / kG2SDenoise_ElementsPerThread;
  static_assert(kNumThreads % kG2SDenoise_R_threads == 0,
                "kNumThreads must be a multiple of kG2SDenoise_R_threads.");
  static constexpr int kG2SDenoise_M_threads =
      kNumThreads / kG2SDenoise_R_threads;

  using G2SCopyAtomDenoise = cute::Copy_Atom<
      cute::Copy_Traits<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>>,
      ElementDenoise>;
  using G2SThreadLayoutDenoise = cute::Layout<
      cute::Shape<cute::Int<kG2SDenoise_M_threads>,
                  cute::Int<kG2SDenoise_R_threads>>,
      cute::Stride<cute::Int<kG2SDenoise_R_threads>, cute::_1>>;
  using G2SValueLayoutDenoise = cute::Layout<
      cute::Shape<cute::_1, cute::Int<kG2SDenoise_ElementsPerThread>>>;
  using G2SCopyDenoise = decltype(cute::make_tiled_copy(
      G2SCopyAtomDenoise{}, G2SThreadLayoutDenoise{}, G2SValueLayoutDenoise{}));

  // ---------- Denoise S2R LDSM copies (sm_75+) ----------
  // A operand: 16M × 16K fp16 per atom → 32 lanes × 8 fp16/lane = 16B/lane.
  // B operand:  8N × 16K fp16 per atom → 32 lanes × 4 fp16/lane =  8B/lane.
  // Use x4 for A (16B/thread) and x2 for B (8B/thread). The TiledCopy
  // make_tiled_copy_{A,B} infers value count from the MMA atom, so the
  // LDSM atom width must match exactly.
  using S2RCopyAtomDenoiseA =
      cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElementDenoise>;
  using S2RCopyAtomDenoiseB =
      cute::Copy_Atom<cute::SM75_U32x2_LDSM_N, ElementDenoise>;

  // C layout: row-major (M, N) with stride (N, 1).
  using LayoutC = cute::Layout<cute::Shape<int, int>, cute::Stride<int64_t, _1>>;

  struct Arguments {
    ElementOut*         ptr_C;
    LayoutC             layout_C;
    ElementScale const* ptr_A_scales;
    ElementScale const* ptr_B_scales;
    // Denoise inputs (ignored when SkipDenoising):
    ElementDenoise const* ptr_EAL        = nullptr;
    ElementDenoise const* ptr_EARxBpEB   = nullptr;
    ElementDenoise const* ptr_AxEBL      = nullptr;
    ElementDenoise const* ptr_EBR        = nullptr;
    ProblemShape          problem_shape{};
  };
  using Params = Arguments;

  static Params to_underlying_arguments(Arguments const& a) { return a; }
  CUTE_HOST_DEVICE static void prefetch_tma_descriptors(Params const&) {}

  // ===========================================================================
  // load_denoise: top-level entry from the kernel. For the register-resident
  // path it is a no-op (factors streamed inside denoise()). For the smem-
  // resident path the staging is now done PER R-STRIP inside denoise() (see
  // stage_denoise_strip below), so this top-level hook is also a no-op — it is
  // kept for API compatibility with the kernel call site.
  // ===========================================================================
  template <typename SharedStorage>
  CUTLASS_DEVICE void load_denoise(
      Params const& params, SharedStorage& smem,
      cute::tuple<int, int, int> const& block_coord, int thread_idx) {
    (void)params;
    (void)smem;
    (void)block_coord;
    (void)thread_idx;
  }

  // ---------------------------------------------------------------------------
  // stage_denoise_strip: cp.async-load the [r_off, r_off+kRTile) R-strip of the
  // four fp16 factors (EAL, EARxBpEB, AxEBL, EBR) from gmem into smem. Each
  // factor is read from gmem EXACTLY ONCE across the full R sweep (one strip per
  // call, kNumRStrips calls total) — vs the register-resident path's 170×
  // per-accumulator re-read. Issues a single cp.async batch + fence + wait +
  // block-wide __syncthreads() (kDenoiseStages=1, no load/compute overlap).
  // ---------------------------------------------------------------------------
  template <typename SharedStorage>
  CUTLASS_DEVICE void stage_denoise_strip(
      Params const& params, SharedStorage& smem,
      cute::tuple<int, int, int> const& block_coord, int thread_idx,
      int r_off) {
    auto m_block = cute::get<0>(block_coord);
    auto n_block = cute::get<1>(block_coord);
    int const M = cute::get<0>(params.problem_shape);
    int const N = cute::get<1>(params.problem_shape);

    // GMEM tensors: (M or N, R) fp16, R-major (stride <R, 1>).
    auto layout_AxEBL = cute::make_layout(
        cute::make_shape(M, cute::Int<R>{}),
        cute::make_stride(cute::Int<R>{}, cute::_1{}));
    auto layout_EAL = cute::make_layout(
        cute::make_shape(M, cute::Int<R>{}),
        cute::make_stride(cute::Int<R>{}, cute::_1{}));
    auto layout_EBR = cute::make_layout(
        cute::make_shape(N, cute::Int<R>{}),
        cute::make_stride(cute::Int<R>{}, cute::_1{}));
    auto layout_EARxBpEB = cute::make_layout(
        cute::make_shape(N, cute::Int<R>{}),
        cute::make_stride(cute::Int<R>{}, cute::_1{}));

    auto mAxEBL    = make_tensor(make_gmem_ptr(params.ptr_AxEBL),    layout_AxEBL);
    auto mEAL      = make_tensor(make_gmem_ptr(params.ptr_EAL),      layout_EAL);
    auto mEBR      = make_tensor(make_gmem_ptr(params.ptr_EBR),      layout_EBR);
    auto mEARxBpEB = make_tensor(make_gmem_ptr(params.ptr_EARxBpEB), layout_EARxBpEB);

    // Per-CTA + per-R-strip tiles: (bMN, kRTile). The R-coord index is
    // r_off / kRTile (local_tile tiles the (M,R) tensor into (bMN, kRTile)
    // sub-tiles; the second coord selects the R-strip).
    int const r_idx = r_off / kRTile;
    auto gAxEBL = local_tile(mAxEBL,
        cute::Shape<cute::Int<bM>, cute::Int<kRTile>>{},
        cute::make_coord(m_block, r_idx));
    auto gEAL = local_tile(mEAL,
        cute::Shape<cute::Int<bM>, cute::Int<kRTile>>{},
        cute::make_coord(m_block, r_idx));
    auto gEBR = local_tile(mEBR,
        cute::Shape<cute::Int<bN>, cute::Int<kRTile>>{},
        cute::make_coord(n_block, r_idx));
    auto gEARxBpEB = local_tile(mEARxBpEB,
        cute::Shape<cute::Int<bN>, cute::Int<kRTile>>{},
        cute::make_coord(n_block, r_idx));

    // SMEM tensors: (bMN, kRTile, kDenoiseStages=1) — slice the stage dim.
    auto sAxEBL = make_tensor(make_smem_ptr(smem.smem_AxEBL.data()),
                              SmemLayoutAxEBL{})(cute::_, cute::_, cute::_0{});
    auto sEBR = make_tensor(make_smem_ptr(smem.smem_EBR.data()),
                            SmemLayoutEBR{})(cute::_, cute::_, cute::_0{});
    auto sEAL = make_tensor(make_smem_ptr(smem.smem_EAL.data()),
                            SmemLayoutEAL{})(cute::_, cute::_, cute::_0{});
    auto sEARxBpEB =
        make_tensor(make_smem_ptr(smem.smem_EARxBpEB.data()),
                    SmemLayoutEARxBpEB{})(cute::_, cute::_, cute::_0{});

    // G2S cp.async copy.
    G2SCopyDenoise g2s;
    auto thr_g2s = g2s.get_slice(thread_idx);
    auto tAxEBLg = thr_g2s.partition_S(gAxEBL);
    auto tAxEBLs = thr_g2s.partition_D(sAxEBL);
    auto tEALg   = thr_g2s.partition_S(gEAL);
    auto tEALs   = thr_g2s.partition_D(sEAL);
    auto tEBRg   = thr_g2s.partition_S(gEBR);
    auto tEBRs   = thr_g2s.partition_D(sEBR);
    auto tEARxBpEBg = thr_g2s.partition_S(gEARxBpEB);
    auto tEARxBpEBs = thr_g2s.partition_D(sEARxBpEB);

    cute::copy(g2s, tEALg,      tEALs);
    cute::copy(g2s, tEARxBpEBg, tEARxBpEBs);
    cute::copy(g2s, tAxEBLg,    tAxEBLs);
    cute::copy(g2s, tEBRg,      tEBRs);

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();
  }

  // Tail is a no-op on sm_89: we don't pipeline denoise loads.
  template <typename SharedStorage>
  CUTLASS_DEVICE void load_denoise_tail(SharedStorage&, int) {}

  // ===========================================================================
  // denoise_register_resident: per-thread denoise correction, no smem.
  //
  // For each accumulator entry tCrD(v, i, j) that this thread owns, the entry
  // sits at coord (m_idx, n_idx) inside the per-CTA (bM, bN) tile. We compute
  //
  //   dot1 = sum_{r=0..R-1} EAL[gm + m_idx][r] * EARxBpEB[gn + n_idx][r]
  //   dot2 = sum_{r=0..R-1} AxEBL[gm + m_idx][r] * EBR[gn + n_idx][r]
  //   tCrD = (tCrD * inv_scale + dot1 + dot2) * scale
  //
  // where gm = m_block * bM, gn = n_block * bN. All factor reads come straight
  // from gmem; the L1 cache amortizes the per-row reuse across threads (32
  // threads/atom share 16 m-rows; many atoms reuse the same rows).
  //
  // Row pointers are precomputed once per accumulator entry. The inner R
  // loop is unrolled by 8 (matches the 16-B cp.async stride of the smem
  // path) — full unroll explodes register pressure (4×4×4 outer × R=128
  // exceeds 255 regs and triggers heavy local-mem spills).
  // ===========================================================================
  template <typename FrgD, typename MainTiledMma>
  CUTLASS_DEVICE void denoise_register_resident(
      Params const& params, FrgD& tCrD,
      MainTiledMma const& main_tiled_mma,
      cute::tuple<int, int, int> const& block_coord, int thread_idx) {
    static_assert(R % 8 == 0,
                  "Register-resident denoise expects R to be a multiple of 8 "
                  "(uint4 load width for fp16).");

    auto m_block = cute::get<0>(block_coord);
    auto n_block = cute::get<1>(block_coord);
    int const gm_offset = m_block * bM;
    int const gn_offset = n_block * bN;

    // Strides: factor tensors are R-major in gmem, stride <R, 1>.
    auto const* eal_ptr        = params.ptr_EAL;
    auto const* eaxbpeb_ptr    = params.ptr_EARxBpEB;
    auto const* axebl_ptr      = params.ptr_AxEBL;
    auto const* ebr_ptr        = params.ptr_EBR;

    // Per-thread accumulator coord lookup via the MAIN tiled_mma.
    auto thr_mma = main_tiled_mma.get_slice(thread_idx);
    auto cD = make_identity_tensor(cute::select<0, 1>(TileShape_MNK{}));
    auto tCcD = thr_mma.partition_C(cD);

    float const inv_scale =
        1.f / static_cast<float>(pearl::kIntToFp16ScaleFactor);
    float const scale =
        static_cast<float>(pearl::kIntToFp16ScaleFactor);

    // Walk the per-thread accumulator. tCrD shape is (V, MMA_M, MMA_N); the
    // identity tensor tCcD reports each entry's (m, n) coord inside (bM, bN).
    // WAVE-11 PATCH: outer j loop (MMA_N) is the dominant axis at bN=256 with
    // kWarpCols=1 (MMA_N = 32 atom-repeats). Full unroll over 32 j-iters * 4
    // v-iters = 128 inner-body copies blows the live-set past 255 regs and
    // forces ~7 KB local-memory spill. Switch j loop to no-unroll (#pragma
    // unroll 1) and inner R loop from full unroll to unroll 8.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(tCrD); ++i) {
      #pragma unroll 1
      for (int j = 0; j < size<2>(tCrD); ++j) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < size<0>(tCrD); ++v) {
          int const m_idx = gm_offset + cute::get<0>(tCcD(v, i, j));
          int const n_idx = gn_offset + cute::get<1>(tCcD(v, i, j));

          auto const* eal_row     = eal_ptr     + static_cast<size_t>(m_idx) * R;
          auto const* eaxbpeb_row = eaxbpeb_ptr + static_cast<size_t>(n_idx) * R;
          auto const* axebl_row   = axebl_ptr   + static_cast<size_t>(m_idx) * R;
          auto const* ebr_row     = ebr_ptr     + static_cast<size_t>(n_idx) * R;

          // Keep the inner R loop NOT fully unrolled: at R=128 and a 4x4x4
          // outer footprint per thread, full unroll would balloon register
          // pressure past 255 and force spills to local memory. Letting the
          // R loop stay rolled (or only unrolled by a small factor) keeps
          // the live-set bounded. The two dots are kept separate so the
          // compiler can issue independent ldg.f16 streams.
          float dot1 = 0.f;
          float dot2 = 0.f;
          #pragma unroll 8
          for (int r = 0; r < R; ++r) {
            float ea = static_cast<float>(eal_row[r]);
            float eb = static_cast<float>(eaxbpeb_row[r]);
            float ax = static_cast<float>(axebl_row[r]);
            float br = static_cast<float>(ebr_row[r]);
            dot1 += ea * eb;
            dot2 += ax * br;
          }
          tCrD(v, i, j) = (tCrD(v, i, j) * inv_scale + dot1 + dot2) * scale;
        }
      }
    }
  }

  // ===========================================================================
  // denoise: applies the two fp16 GEMM corrections to tCrD (fp32 accumulator)
  // around a 2^-12 / 2^12 scale window. See pearl_gemm_constants.hpp for the
  // factor and the Hopper denoise() body for the math.
  //
  //   tCrD = tCrD * (1 / kIntToFp16ScaleFactor)
  //   tCrD += EAL    @ EARxBpEB     (fp16 MMA, fp32 accum)
  //   tCrD += AxEBL  @ EBR          (fp16 MMA, fp32 accum)
  //   tCrD = tCrD * kIntToFp16ScaleFactor
  //
  // The Hopper comments say `-=`, but the actual upstream `gemm()` call
  // accumulates `+=`. The sign flip lives in the noise-generation kernels
  // (see pearl_gemm_constants.hpp `kEBRScaleFactorDenoise = -1 * …`). We mirror
  // the Hopper math exactly — no sign flip here.
  //
  // Register-resident mode (KTraits::kRegisterResidentDenoise = true): the two
  // corrections are computed per-thread over the (V, MMA_M, MMA_N) layout of
  // tCrD. For each accumulator entry the thread owns, we load its EAL[m, :]
  // and EARxBpEB[n, :] rows from gmem (cached in L1) and compute the R-length
  // dot in fp32. Same for AxEBL/EBR. No smem footprint, no LDSM, no MMA atom
  // for the denoise.
  // ===========================================================================
  template <typename SharedStorage, typename FrgD, typename MainTiledMma>
  CUTLASS_DEVICE void denoise(Params const& params, FrgD& tCrD,
                              SharedStorage& smem,
                              MainTiledMma const& main_tiled_mma,
                              cute::tuple<int, int, int> const& block_coord,
                              int thread_idx) {
    if constexpr (KTraits::SkipDenoising) {
      (void)params;
      (void)smem;
      (void)main_tiled_mma;
      (void)block_coord;
      (void)thread_idx;
      return;
    } else if constexpr (KTraits::kRegisterResidentDenoise) {
      denoise_register_resident(params, tCrD, main_tiled_mma, block_coord,
                                thread_idx);
      return;
    } else {
      // R-strip-tiled smem-resident denoise. The two corrections
      //   tCrD += EAL @ EARxBpEB^T  and  tCrD += AxEBL @ EBR^T
      // are R-length dot products realized as fp16 tensor-core MMAs. We tile
      // the R dimension into kNumRStrips strips of width kRTile: per strip we
      // cp.async-stage only that (bMN × kRTile) slice of each factor into smem
      // (so the staged footprint fits the 99 KB cap) and run the partial MMA,
      // accumulating into the fp32 tCrD across strips. Summing partial MMAs
      // over R-strips is a reassociation of the same R-length sum — bit-exact
      // up to fp32 reduction-order, which the reference oracle confirms is ~0.
      //
      // Each factor row is read from gmem EXACTLY ONCE over the full sweep
      // (kNumRStrips staged loads), eliminating the 170× per-accumulator
      // re-read that made the register-resident path memory-bandwidth-bound.

      // Pre-scale: tCrD *= 1 / kIntToFp16ScaleFactor.
      float const inv_scale =
          1.f / static_cast<float>(pearl::kIntToFp16ScaleFactor);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < cute::size(tCrD); ++i) {
        tCrD(i) *= inv_scale;
      }

      TiledMmaDenoise tiled_mma_denoise;

      // S2R LDSM tiled copies (lane mapping matched to the MMA A/B operands).
      auto s2r_a = make_tiled_copy_A(S2RCopyAtomDenoiseA{}, tiled_mma_denoise);
      auto s2r_b = make_tiled_copy_B(S2RCopyAtomDenoiseB{}, tiled_mma_denoise);
      auto s2r_thr_a = s2r_a.get_slice(thread_idx);
      auto s2r_thr_b = s2r_b.get_slice(thread_idx);

      CUTLASS_PRAGMA_NO_UNROLL
      for (int rs = 0; rs < kNumRStrips; ++rs) {
        int const r_off = rs * kRTile;

        // Stage this R-strip of all four factors into smem (cp.async + sync).
        stage_denoise_strip(params, smem, block_coord, thread_idx, r_off);

        // SMEM tensors for this strip (sliced at stage 0).
        auto sAxEBL = make_tensor(make_smem_ptr(smem.smem_AxEBL.data()),
                                  SmemLayoutAxEBL{})(cute::_, cute::_, cute::_0{});
        auto sEBR = make_tensor(make_smem_ptr(smem.smem_EBR.data()),
                                SmemLayoutEBR{})(cute::_, cute::_, cute::_0{});
        auto sEAL = make_tensor(make_smem_ptr(smem.smem_EAL.data()),
                                SmemLayoutEAL{})(cute::_, cute::_, cute::_0{});
        auto sEARxBpEB =
            make_tensor(make_smem_ptr(smem.smem_EARxBpEB.data()),
                        SmemLayoutEARxBpEB{})(cute::_, cute::_, cute::_0{});

        auto thr_mma = tiled_mma_denoise.get_slice(thread_idx);
        // Fragments over the kRTile strip: (MMA, MMA_M/N, MMA_R=kRTile/16).
        auto tCrEAL      = thr_mma.partition_fragment_A(sEAL);
        auto tCrEARxBpEB = thr_mma.partition_fragment_B(sEARxBpEB);
        auto tCrAxEBL    = thr_mma.partition_fragment_A(sAxEBL);
        auto tCrEBR      = thr_mma.partition_fragment_B(sEBR);

        auto tXsEAL      = s2r_thr_a.partition_S(sEAL);
        auto tXsEARxBpEB = s2r_thr_b.partition_S(sEARxBpEB);
        auto tXsAxEBL    = s2r_thr_a.partition_S(sAxEBL);
        auto tXsEBR      = s2r_thr_b.partition_S(sEBR);

        auto tXrEAL      = s2r_thr_a.retile_D(tCrEAL);
        auto tXrEARxBpEB = s2r_thr_b.retile_D(tCrEARxBpEB);
        auto tXrAxEBL    = s2r_thr_a.retile_D(tCrAxEBL);
        auto tXrEBR      = s2r_thr_b.retile_D(tCrEBR);

        // Load this strip's operands smem -> regs.
        cute::copy(s2r_a, tXsEAL,      tXrEAL);
        cute::copy(s2r_b, tXsEARxBpEB, tXrEARxBpEB);
        cute::copy(s2r_a, tXsAxEBL,    tXrAxEBL);
        cute::copy(s2r_b, tXsEBR,      tXrEBR);

        // Two partial fp16 MMAs into the fp32 tCrD accumulator (+= over strips).
        cute::gemm(tiled_mma_denoise, tCrEAL,   tCrEARxBpEB, tCrD);
        cute::gemm(tiled_mma_denoise, tCrAxEBL, tCrEBR,      tCrD);

        // Strip's smem must be fully consumed before the next strip's
        // stage_denoise_strip() overwrites it.
        __syncthreads();
      }

      // Post-scale: tCrD *= kIntToFp16ScaleFactor (== 1<<12).
      float const scale = static_cast<float>(pearl::kIntToFp16ScaleFactor);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < cute::size(tCrD); ++i) {
        tCrD(i) *= scale;
      }
    }
  }

  // ===========================================================================
  // scale(): apply per-row A scale × per-col B scale to the fp32 accumulator,
  // then cast to bf16 and R2S-stage into smem_C.
  // ===========================================================================
  template <typename SharedStorage, typename FrgTensor, typename TiledMma>
  CUTLASS_DEVICE void scale(Params const& params, FrgTensor& tCrD,
                            SharedStorage& smem, TiledMma tiled_mma,
                            int thread_idx,
                            cute::tuple<int, int, int> const& block_coord) {
    // Use named coords to avoid `_` shadowing cute::_.
    auto m_block = cute::get<0>(block_coord);
    auto n_block = cute::get<1>(block_coord);
    int const M = cute::get<0>(params.problem_shape);
    int const N = cute::get<1>(params.problem_shape);
    int const residual_M = M - m_block * bM;
    int const residual_N = N - n_block * bN;

    // ----- cp.async A_scales + B_scales gmem -> smem -----
    auto AScales = make_tensor(make_gmem_ptr(params.ptr_A_scales),
                               cute::make_layout(cute::make_shape(M)));
    auto BScales = make_tensor(make_gmem_ptr(params.ptr_B_scales),
                               cute::make_layout(cute::make_shape(N)));
    auto gAscales = local_tile(AScales, cute::select<0>(TileShape_MNK{}),
                               cute::make_coord(m_block));
    auto gBscales = local_tile(BScales, cute::select<1>(TileShape_MNK{}),
                               cute::make_coord(n_block));
    auto sAscales = make_tensor(make_smem_ptr(smem.smem_scale_a.data()),
                                SmemLayoutScaleA{});
    auto sBscales = make_tensor(make_smem_ptr(smem.smem_scale_b.data()),
                                SmemLayoutScaleB{});

    G2SScalesCopyA g2s_a;
    G2SScalesCopyB g2s_b;
    auto thr_a = g2s_a.get_slice(thread_idx);
    auto thr_b = g2s_b.get_slice(thread_idx);
    auto tAg = thr_a.partition_S(gAscales);
    auto tAs = thr_a.partition_D(sAscales);
    auto tBg = thr_b.partition_S(gBscales);
    auto tBs = thr_b.partition_D(sBscales);

    if (thread_idx < bM) {
      if constexpr (KTraits::Is_Even_M) {
        cute::copy(g2s_a, tAg, tAs);
      } else if (thread_idx < residual_M) {
        cute::copy(g2s_a, tAg, tAs);
      }
    }
    if (thread_idx < bN) {
      if constexpr (KTraits::Is_Even_N) {
        cute::copy(g2s_b, tBg, tBs);
      } else if (thread_idx < residual_N) {
        cute::copy(g2s_b, tBg, tBs);
      }
    }
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();  // block-wide sync; LoadScales NamedBarrier is sm_90-only

    // ----- Apply scales to per-thread accumulator fragment -----
    auto thr_mma = tiled_mma.get_slice(thread_idx);
    auto cD = make_identity_tensor(cute::select<0, 1>(TileShape_MNK{}));
    auto tCcD = thr_mma.partition_C(cD);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(tCrD); ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<2>(tCrD); ++j) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < size<0>(tCrD); ++v) {
          int m_idx = cute::get<0>(tCcD(v, i, j));
          int n_idx = cute::get<1>(tCcD(v, i, j));
          tCrD(v, i, j) *= (sAscales(m_idx) * sBscales(n_idx));
        }
      }
    }

    // ----- fp32 -> bf16, R2S stage into smem_C -----
    // Mining no-C-store mode (Lever A): the transcript already consumed the
    // accumulator in the mainloop; the materialized C is never read, so skip
    // the R2S stage entirely (and store() below is a no-op). The scale
    // arithmetic above still runs so the denoise/scale work is timed and not
    // dead-code-eliminated.
    if constexpr (!KTraits::kMiningNoStore) {
      auto tCrC_out = convert_type<ElementOut>(tCrD);
      auto sC = make_tensor(make_smem_ptr(smem.smem_C.data()), SmemLayoutC{});
      auto smem_tiled_copy_C = make_tiled_copy_C(SmemCopyAtomC{}, tiled_mma);
      auto smem_thr_copy_C = smem_tiled_copy_C.get_thread_slice(thread_idx);
      auto taccCrC = smem_thr_copy_C.retile_S(tCrC_out);
      auto taccCsC = smem_thr_copy_C.partition_D(sC);
      cute::copy(smem_tiled_copy_C, taccCrC, taccCsC);
    }
  }

  // ===========================================================================
  // store(): smem_C (bf16) -> gmem C via vectorized 16-byte st.global.v4.
  // Block-wide __syncthreads() replaces the Hopper asymmetric write-warp +
  // NamedBarrier::Epilogue pattern since every warp participates in the S2G
  // copy on sm_89.
  // ===========================================================================
  template <typename SharedStorage>
  CUTLASS_DEVICE void store(Params const& params, SharedStorage& smem,
                            int thread_idx,
                            cute::tuple<int, int, int> const& block_coord) {
    // Mining no-C-store mode (Lever A): no smem_C was staged and no gmem C is
    // written. Skip the whole S2G path (and its preceding __syncthreads()).
    if constexpr (KTraits::kMiningNoStore) {
      (void)params; (void)smem; (void)thread_idx; (void)block_coord;
      return;
    }
    auto m_block = cute::get<0>(block_coord);
    auto n_block = cute::get<1>(block_coord);

    // R2S stage (in scale()) must be visible block-wide before S2G.
    __syncthreads();

    auto sC = make_tensor(make_smem_ptr(smem.smem_C.data()), SmemLayoutC{});
    auto mC = make_tensor(make_gmem_ptr(params.ptr_C), params.layout_C);
    auto gC = local_tile(mC, cute::select<0, 1>(TileShape_MNK{}),
                         cute::make_coord(m_block, n_block));

    S2GCopyC gmem_tiled_copy_C;
    auto gmem_thr_copy = gmem_tiled_copy_C.get_thread_slice(thread_idx);
    auto tSsC = gmem_thr_copy.partition_S(sC);
    auto tDgC = gmem_thr_copy.partition_D(gC);

    // Predicate residual rows/cols (Is_Even_M/N gates the fast path).
    int const M = cute::get<0>(params.problem_shape);
    int const N = cute::get<1>(params.problem_shape);
    int const residual_M = M - m_block * bM;
    int const residual_N = N - n_block * bN;

    if constexpr (KTraits::Is_Even_M && KTraits::Is_Even_N) {
      cute::copy(gmem_tiled_copy_C, tSsC, tDgC);
    } else {
      auto cC = make_identity_tensor(cute::select<0, 1>(TileShape_MNK{}));
      auto tCcC = gmem_thr_copy.partition_S(cC);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<1>(tSsC); ++i) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size<2>(tSsC); ++j) {
          bool in = (cute::get<0>(tCcC(Int<0>{}, i, j)) < residual_M) &&
                    (cute::get<1>(tCcC(Int<0>{}, i, j)) < residual_N);
          if (in) cute::copy(gmem_tiled_copy_C, tSsC(_, i, j), tDgC(_, i, j));
        }
      }
    }
  }

  CUTLASS_DEVICE void store_tail() { __syncthreads(); }
};

}  // namespace pearl
