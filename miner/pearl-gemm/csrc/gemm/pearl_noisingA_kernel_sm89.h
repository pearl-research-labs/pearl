// SPDX-License-Identifier: see LICENSE
//
// sm_89 port of pearl_noisingA_kernel.h (Hopper). Unified-warp model: a single
// warpgroup-sized CTA (128 threads = 4 warps) sequentially performs all
// loads + both MMAs + the smem-stage + S2G ApEA writeback, per k_block.
//
// Substitution map vs the Hopper version (see SM89_PORT_SPEC.md §2):
//   SM90_TMA_LOAD              -> SM80_CP_ASYNC_CACHEGLOBAL<uint128_t> + cp.async loop
//   SM90_TMA_STORE             -> AutoVectorizingCopyWithAssumedAlignment<128>
//   GMMA::ss_op_selector<int8> -> SM80_16x8x32_S32S8S8S32_TN MMA atom
//                                  + AtomLayout<2,2,1>, Tile<32,32,32> (sm_80 canonical)
//   PipelineTmaAsync           -> raw cp.async + cp_async_fence + cp_async_wait<>
//   ss_smem_selector<K,int8>   -> composition(Swizzle<2,4,3>, Layout<Shape<_16,_64>,Stride<_64,_1>>)
//   3 warpgroups (prod/2 cons) -> 1 warpgroup (unified-warp): same warp does
//                                  cp.async + both MMAs + add + R2S + S2G
//   warpgroup_arrive/commit    -> deleted (sm_80 mma.sync is synchronous)
//   warpgroup_reg_alloc/dealloc -> deleted (use __launch_bounds__)
//   NamedBarrier::sync/arrive  -> __syncthreads()
//   tma_store_arrive/wait      -> __syncthreads()
//   Hopper LDSM/STSM A-add     -> in-smem add via partition_C of MKR MMA on
//                                  swizzled smem_ApEA + smem_A (no LDSM/STSM
//                                  needed; permute_Aregs_fp8 also unneeded
//                                  because we never look at register pairs
//                                  across lanes — the MMA C partition gives
//                                  per-thread (V, M, K) → (m_idx, n_idx)
//                                  positions directly).
//
// Architectural choice: every k_block runs sequentially as
//   wait stage  -> S2R load A, EBL, EAL (once), EAR
//                -> MRK MMA accumulating into tCrAxEBL
//                -> MKR MMA into tCrApEA (cleared per iter)
//                -> convert int32->int8 (saturates; valid for int7 inputs)
//                -> R2S into smem_ApEA[pipe_out] via partition_C(MKR)
//                -> in-smem add: smem_ApEA[m,n] += smem_A[m,n] (same partition)
//                -> S2G smem_ApEA[pipe_out] -> gmem ApEA[k_block]
//                -> issue next cp.async stage
//
// At end: write tCrAxEBL int32 accumulator to gmem AxEBL via partition_C(MRK).
//
// Smem budget at bM=64, R=64, bK=64, kStages=2: ~36 KB (fits 100 KB cap).

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "cute/algorithm/copy.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_sm75.hpp"
#include "cute/atom/copy_traits_sm80.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm80.hpp"
#include "cute/swizzle.hpp"

#include "pearl_gemm_constants.hpp"
#include "utils.h"

namespace pearl {

using namespace cute;

template <class TileShape_MRK_, int kNumThreads_, class Element_,
          class ElementDenoise_, int kStages_, bool IsEvenK, bool NoReduction>
class NoisingKernelASm89 {

 public:
  using TileShape_MRK = TileShape_MRK_;
  using Element       = Element_;
  using ElementDenoise = ElementDenoise_;
  using ElementScale  = float;
  using ElementIndex  = uint8_t;
  using ElementAccum  = int32_t;

  static_assert(cute::is_same_v<Element, int8_t>);
  static constexpr int denoise_dtype_bits = cute::sizeof_bits_v<ElementDenoise>;
  static_assert(denoise_dtype_bits == 16 || denoise_dtype_bits == 32,
                "Denoise dtype size must be 16 or 32 bits");
  static_assert(denoise_dtype_bits == 32 || NoReduction,
                "Don't support reduction with fp16");

  using ArchTag = cutlass::arch::Sm89;
  static constexpr int kBlockM = get<0>(TileShape_MRK{});   // bM
  static constexpr int R       = get<1>(TileShape_MRK{});   // R
  static constexpr int kBlockK = get<2>(TileShape_MRK{});   // bK
  using TileShape_MKR = Shape<Int<kBlockM>, Int<kBlockK>, Int<R>>;
  using TileShape_MR  = Shape<Int<kBlockM>, Int<R>>;

  static_assert(kBlockM == 64, "Hopper version is hard-coded to bM==64");
  static_assert(R == 64 || R == 128);
  static_assert(kBlockK == 64);

  static constexpr int kStages    = kStages_;
  static constexpr int kStagesOut = kStages_;
  static_assert(kStages >= 2 && kStages <= 4,
                "sm_89 dynamic smem cap (100 KB) restricts kStages here.");

  // Unified-warp model: 128 threads (4 warps) — one warpgroup-sized CTA.
  static constexpr uint32_t kNumThreads = 128;
  static_assert(kNumThreads_ == 128, "sm_89 noisingA uses 128 threads");
  static constexpr uint32_t MaxThreadsPerBlock = kNumThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = R == 64 ? 2 : 1;

  static constexpr int kClusterM = 1;
  static constexpr int kClusterN = 1;
  using ClusterShape = Shape<Int<kClusterM>, Int<kClusterN>, _1>;

  // ---------------- TiledMMA ----------------
  // Canonical sm_80 int8 default config (test/unit/.../default_gemm_configuration.hpp:343).
  // 4 warps in a 2x2x1 layout, atom replicated via Tile<32,32,32> permutation.
  //   AtomLayout<2,2,1> + Tile<32,32,32> gives a 32x32x32 tile per atom group;
  //   for bM=R=bK=64, the tiled_mma stamps 2x2 atom-groups in M,N and 2 along K.
  using AtomLayoutMma = Layout<Shape<_2, _2, _1>>;

  // MMA for A * EBL: (bM, R, bK) = (64, 64, 64), output (bM, R) int32.
  using TiledMmaMRK = decltype(make_tiled_mma(
      MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>{},
      AtomLayoutMma{},
      Tile<_32, _32, _32>{}));

  // MMA for EAL * EAR^T: (bM, bK, R) — same SM80 atom, same atom layout, same tile.
  using TiledMmaMKR = decltype(make_tiled_mma(
      MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>{},
      AtomLayoutMma{},
      Tile<_32, _32, _32>{}));

  // ---------------- Smem layouts ----------------
  // int8 K-major canonical sm_80 atom: Swizzle<2,4,3> over (16, 64) stride (64,1).
  // Requires bM % 16 == 0 and bK % 64 == 0; (bM=R=bK=64) all satisfy.
  using SmemLayoutAtomK_int8 = decltype(composition(
      Swizzle<2, 4, 3>{},
      Layout<Shape<_16, _64>, Stride<_64, _1>>{}));

  // A: (bM, bK, kStages) K-major.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomK_int8{},
      make_shape(Int<kBlockM>{}, Int<kBlockK>{}, Int<kStages>{})));
  // EBL: (R, bK, kStages) K-major.
  using SmemLayoutEBL = decltype(tile_to_shape(
      SmemLayoutAtomK_int8{},
      make_shape(Int<R>{}, Int<kBlockK>{}, Int<kStages>{})));
  // EAL: (bM, R) R-major (loaded once per CTA, K = R here).
  using SmemLayoutEAL = decltype(tile_to_shape(
      SmemLayoutAtomK_int8{},
      make_shape(Int<kBlockM>{}, Int<R>{})));
  // EAR: (bK, R, kStages) R-major (the "K" of the MMA-K is R, hence K-major
  // atom over (bK, R)).
  using SmemLayoutEAR = decltype(tile_to_shape(
      SmemLayoutAtomK_int8{},
      make_shape(Int<kBlockK>{}, Int<R>{}, Int<kStages>{})));
  // ApEA: (bM, bK, kStagesOut) K-major (output staging).
  using SmemLayoutApEA = decltype(tile_to_shape(
      SmemLayoutAtomK_int8{},
      make_shape(Int<kBlockM>{}, Int<kBlockK>{}, Int<kStagesOut>{})));

  // ---------------- Copy atoms ----------------
  // G->S: 16-byte cp.async, vectorized.
  using G2SCopyAtom = Copy_Atom<
      Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>>,
      Element>;

  // 128 threads cover (M-tile, K-tile=64) with each thread writing 16 int8
  // along K. K-threads = 64 / 16 = 4. M-threads = 128 / 4 = 32. So a single
  // cp.async issue covers 32 M rows by 64 K elements per call.
  static constexpr int kG2S_K_threads = kBlockK / 16;   // 4
  static_assert(kG2S_K_threads > 0 && kBlockK % 16 == 0);
  static constexpr int kG2S_M_threads = kNumThreads / kG2S_K_threads;  // 32
  static_assert(kG2S_M_threads * kG2S_K_threads == kNumThreads);
  using G2SThreadLayout = Layout<Shape<Int<kG2S_M_threads>, Int<kG2S_K_threads>>,
                                 Stride<Int<kG2S_K_threads>, _1>>;
  using G2SValueLayout  = Layout<Shape<_1, _16>>;

  // A: covers (bM=64, bK=64) -> need to repeat M direction 2x with 32 threads.
  // tile_to_shape on the TiledCopy via thread layout: we'll let partition_S
  // handle iteration over (bM, bK) by giving partition_S the full sA tensor.
  using G2SCopyA = decltype(make_tiled_copy(
      G2SCopyAtom{}, G2SThreadLayout{}, G2SValueLayout{}));
  using G2SCopyEBL = decltype(make_tiled_copy(
      G2SCopyAtom{}, G2SThreadLayout{}, G2SValueLayout{}));
  using G2SCopyEAR = decltype(make_tiled_copy(
      G2SCopyAtom{}, G2SThreadLayout{}, G2SValueLayout{}));

  // EAL: load once, (bM, R) — same K-major thread layout (R is the "K" of this
  // tensor since it's R-major, R=64 fits the 4-thread vector width).
  static constexpr int kG2S_R_threads = R / 16;          // 4
  static_assert(R % 16 == 0);
  static constexpr int kG2S_M_threads_EAL = kNumThreads / kG2S_R_threads;
  using G2SThreadLayoutEAL = Layout<Shape<Int<kG2S_M_threads_EAL>, Int<kG2S_R_threads>>,
                                    Stride<Int<kG2S_R_threads>, _1>>;
  using G2SCopyEAL = decltype(make_tiled_copy(
      G2SCopyAtom{}, G2SThreadLayoutEAL{}, G2SValueLayout{}));

  // S->R via LDSM (sm_75+). Loading int8 K-major data for the MMA operands.
  using S2RCopyAtomA   = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using S2RCopyAtomB   = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using S2RCopyAtomEAL = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using S2RCopyAtomEAR = Copy_Atom<SM75_U32x4_LDSM_N, Element>;

  // R->S for the ApEA accumulator is implemented inline (per-element stores
  // via partition_C of the MKR MMA on swizzled smem_ApEA). The MMA C
  // fragment's V=4 int8 values land at non-contiguous smem offsets due to
  // the int8 swizzle + MMA layout, so a single uint32 STSM would not
  // vectorize cleanly. The STSM CUTE atom (SM90_U32x4_STSM_N) is also
  // gated on sm_90 even though the underlying PTX is sm_75+. Element-wise
  // stores via the CUTE partition_C view are simpler and let the compiler
  // pack the int8 stores naturally.

  // S->G for ApEA: plain vectorized 16-byte st.global.v4.
  using S2GCopyAtomApEA = Copy_Atom<
      AutoVectorizingCopyWithAssumedAlignment<128>,
      Element>;

  // ---------------- ApEA store layout (S->G) ----------------
  // For the smem -> gmem ApEA copy, we use 128 threads in a (M, K) thread
  // grid that vectorizes by 16. Each thread writes 16 int8s along K.
  using S2GThreadLayoutApEA = G2SThreadLayout;  // same shape, same 16-byte width
  using S2GValueLayoutApEA  = Layout<Shape<_1, _16>>;
  using S2GCopyApEA = decltype(make_tiled_copy(
      S2GCopyAtomApEA{}, S2GThreadLayoutApEA{}, S2GValueLayoutApEA{}));

  // ---------------- Layouts for GMEM Tensors ----------------
  using ShapeT  = cute::Shape<int32_t, int32_t>;
  using StrideT = cute::Shape<int32_t, _1>;
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  // TMA requires 128B alignment; we keep 128B alignment for smem regardless.
  static constexpr size_t Alignment = 128;

  struct SharedStorage : cute::aligned_struct<Alignment> {
    // Mainloop tile buffers. union with smem_AxEBL is dropped on sm_89 -
    // simpler to have AxEBL stay in registers (R=64 fits).
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutA>, Alignment>
        smem_A;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutEBL>, Alignment>
        smem_EBL;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutEAL>, Alignment>
        smem_EAL;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutEAR>, Alignment>
        smem_EAR;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutApEA>, Alignment>
        smem_ApEA;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // ---------------- Arguments / Params ----------------
  struct Arguments {
    Element const* const ptr_A;
    Element const* const ptr_EAL;
    Element const* const ptr_EAR;
    Element const* const ptr_EBL;
    Element* const ptr_A_out;
    ElementDenoise* const ptr_AxEBL;
    int m;
    int k;
    int num_k_blocks;  // unused in NoReduction; kept for API parity
    int total_k_blocks;
  };

  struct Params {
    Element const* const ptr_A;
    Element const* const ptr_EAL;
    Element const* const ptr_EAR;
    Element const* const ptr_EBL;
    Element* const ptr_A_out;
    ElementDenoise* const ptr_AxEBL;
    int m;
    int k;
    int num_k_blocks;
    int total_k_blocks;
    LayoutT layout_A;
    LayoutT layout_EBL;
    LayoutT layout_EAL;
    LayoutT layout_EAR;
    LayoutT layout_AxEBL;
    LayoutT layout_ApEA;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    LayoutT layout_A =
        make_layout(make_shape(args.m, args.k), make_stride(args.k, _1{}));
    LayoutT layout_EBL =
        make_layout(make_shape(R, args.k), make_stride(args.k, _1{}));
    LayoutT layout_EAL =
        make_layout(make_shape(args.m, R), make_stride(R, _1{}));
    LayoutT layout_EAR =
        make_layout(make_shape(args.k, R), make_stride(R, _1{}));
    LayoutT layout_AxEBL =
        make_layout(make_shape(args.m, R), make_stride(R, _1{}));
    LayoutT layout_ApEA =
        make_layout(make_shape(args.m, args.k), make_stride(args.k, _1{}));
    return {.ptr_A          = args.ptr_A,
            .ptr_EAL        = args.ptr_EAL,
            .ptr_EAR        = args.ptr_EAR,
            .ptr_EBL        = args.ptr_EBL,
            .ptr_A_out      = args.ptr_A_out,
            .ptr_AxEBL      = args.ptr_AxEBL,
            .m              = args.m,
            .k              = args.k,
            .num_k_blocks   = args.num_k_blocks,
            .total_k_blocks = args.total_k_blocks,
            .layout_A       = layout_A,
            .layout_EBL     = layout_EBL,
            .layout_EAL     = layout_EAL,
            .layout_EAR     = layout_EAR,
            .layout_AxEBL   = layout_AxEBL,
            .layout_ApEA    = layout_ApEA};
  }

  static dim3 get_grid_shape(Params const& params) {
    if constexpr (NoReduction) {
      return dim3(ceil_div(params.m, kBlockM), 1, 1);
    } else {
      return dim3(ceil_div(params.m, kBlockM),
                  ceil_div(params.k, params.num_k_blocks * kBlockK), 1);
    }
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  // ---------------- Kernel entry ----------------
  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int const tid = threadIdx.x;
    int const m_block = blockIdx.x;
    int const k_block_min = blockIdx.y * params.num_k_blocks;
    int const num_k_blocks_cta =
        cute::min(params.num_k_blocks, params.total_k_blocks - k_block_min);
    int const k_block_max = k_block_min + num_k_blocks_cta;

    // Build GMEM tensors and CTA-local tile views.
    Tensor mA   = make_tensor(make_gmem_ptr(params.ptr_A),   params.layout_A);
    Tensor mEBL = make_tensor(make_gmem_ptr(params.ptr_EBL), params.layout_EBL);
    Tensor mEAL = make_tensor(make_gmem_ptr(params.ptr_EAL), params.layout_EAL);
    Tensor mEAR = make_tensor(make_gmem_ptr(params.ptr_EAR), params.layout_EAR);
    Tensor mApEA = make_tensor(make_gmem_ptr(params.ptr_A_out),
                                params.layout_ApEA);

    Tensor gA   = local_tile(mA, select<0, 2>(TileShape_MRK{}),
                              make_coord(m_block, _));     // (bM, bK, k_tiles)
    Tensor gEBL = local_tile(mEBL, select<1, 2>(TileShape_MRK{}),
                              make_coord(_0{}, _));         // (R, bK, k_tiles)
    Tensor gEAR = local_tile(mEAR, select<2, 1>(TileShape_MRK{}),
                              make_coord(_, _0{}));         // (bK, R, k_tiles)
    Tensor gEAL = local_tile(mEAL, select<0, 1>(TileShape_MRK{}),
                              make_coord(m_block, _0{}));   // (bM, R)
    Tensor gApEA = local_tile(mApEA, select<0, 2>(TileShape_MRK{}),
                               make_coord(m_block, _));     // (bM, bK, k_tiles)

    // Build SMEM tensors.
    Tensor sA   = make_tensor(make_smem_ptr(shared_storage.smem_A.data()),
                               SmemLayoutA{});      // (bM, bK, P)
    Tensor sEBL = make_tensor(make_smem_ptr(shared_storage.smem_EBL.data()),
                               SmemLayoutEBL{});    // (R, bK, P)
    Tensor sEAL = make_tensor(make_smem_ptr(shared_storage.smem_EAL.data()),
                               SmemLayoutEAL{});    // (bM, R)
    Tensor sEAR = make_tensor(make_smem_ptr(shared_storage.smem_EAR.data()),
                               SmemLayoutEAR{});    // (bK, R, P)
    Tensor sApEA = make_tensor(make_smem_ptr(shared_storage.smem_ApEA.data()),
                                SmemLayoutApEA{});  // (bM, bK, OP)
    Tensor sApEA_pi = as_position_independent_swizzle_tensor(sApEA);
    Tensor sA_pi = as_position_independent_swizzle_tensor(sA);

    // ---------------- G->S copy partitioning ----------------
    G2SCopyA   copy_a;
    G2SCopyEBL copy_ebl;
    G2SCopyEAR copy_ear;
    G2SCopyEAL copy_eal;

    auto thr_copy_a   = copy_a.get_slice(tid);
    auto thr_copy_ebl = copy_ebl.get_slice(tid);
    auto thr_copy_ear = copy_ear.get_slice(tid);
    auto thr_copy_eal = copy_eal.get_slice(tid);

    auto tAgA   = thr_copy_a.partition_S(gA);    // (CPY, CPY_M, CPY_K, k_tiles)
    auto tAsA   = thr_copy_a.partition_D(sA);    // (CPY, CPY_M, CPY_K, PIPE)
    auto tEBLgEBL = thr_copy_ebl.partition_S(gEBL);
    auto tEBLsEBL = thr_copy_ebl.partition_D(sEBL);
    auto tEARgEAR = thr_copy_ear.partition_S(gEAR);
    auto tEARsEAR = thr_copy_ear.partition_D(sEAR);
    auto tEALgEAL = thr_copy_eal.partition_S(gEAL);
    auto tEALsEAL = thr_copy_eal.partition_D(sEAL);

    // ---------------- MMA partitioning ----------------
    TiledMmaMRK tiled_mma_mrk;
    TiledMmaMKR tiled_mma_mkr;
    auto thr_mma_mrk = tiled_mma_mrk.get_thread_slice(tid);
    auto thr_mma_mkr = tiled_mma_mkr.get_thread_slice(tid);

    // For AxEBL = A @ EBL^T: A is operand A (bM, bK), EBL is operand B (R, bK).
    auto tCsA_mrk   = thr_mma_mrk.partition_A(sA);    // (MMA, MMA_M, MMA_K, P)
    auto tCsEBL     = thr_mma_mrk.partition_B(sEBL);  // (MMA, MMA_R, MMA_K, P)
    // For ApEA = EAL @ EAR^T: EAL is operand A (bM, R), EAR is operand B (bK, R).
    auto tCsEAL     = thr_mma_mkr.partition_A(sEAL);  // (MMA, MMA_M, MMA_R)
    auto tCsEAR     = thr_mma_mkr.partition_B(sEAR);  // (MMA, MMA_K, MMA_R, P)

    // Accumulator fragments.
    auto tCrAxEBL = partition_fragment_C(tiled_mma_mrk,
                                          select<0, 1>(TileShape_MRK{}));  // (V, MMA_M, MMA_R)
    auto tCrApEA  = partition_fragment_C(tiled_mma_mkr,
                                          select<0, 1>(TileShape_MKR{}));  // (V, MMA_M, MMA_K)
    auto tCrApEA_int8 = make_fragment_like<Element>(tCrApEA);
    clear(tCrAxEBL);

    // Register fragments for operands (per-k_block sized).
    auto tCrA_mrk  = thr_mma_mrk.make_fragment_A(tCsA_mrk(_, _, _, Int<0>{}));   // (MMA, MMA_M, MMA_K)
    auto tCrEBL    = thr_mma_mrk.make_fragment_B(tCsEBL(_, _, _, Int<0>{}));     // (MMA, MMA_R, MMA_K)
    auto tCrEAL    = thr_mma_mkr.make_fragment_A(tCsEAL);                         // (MMA, MMA_M, MMA_R)
    auto tCrEAR    = thr_mma_mkr.make_fragment_B(tCsEAR(_, _, _, Int<0>{}));     // (MMA, MMA_K, MMA_R)

    // S->R copy partitioning (ldmatrix).
    auto s2r_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma_mrk);
    auto s2r_thr_a = s2r_a.get_slice(tid);
    auto tXsA_mrk = s2r_thr_a.partition_S(sA);
    auto tXrA_mrk = s2r_thr_a.retile_D(tCrA_mrk);

    auto s2r_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma_mrk);
    auto s2r_thr_b = s2r_b.get_slice(tid);
    auto tXsEBL = s2r_thr_b.partition_S(sEBL);
    auto tXrEBL = s2r_thr_b.retile_D(tCrEBL);

    auto s2r_eal = make_tiled_copy_A(S2RCopyAtomEAL{}, tiled_mma_mkr);
    auto s2r_thr_eal = s2r_eal.get_slice(tid);
    auto tXsEAL = s2r_thr_eal.partition_S(sEAL);
    auto tXrEAL = s2r_thr_eal.retile_D(tCrEAL);

    auto s2r_ear = make_tiled_copy_B(S2RCopyAtomEAR{}, tiled_mma_mkr);
    auto s2r_thr_ear = s2r_ear.get_slice(tid);
    auto tXsEAR = s2r_thr_ear.partition_S(sEAR);
    auto tXrEAR = s2r_thr_ear.retile_D(tCrEAR);

    // ---------------- S->G for ApEA ----------------
    S2GCopyApEA s2g_apea;
    auto s2g_thr_apea = s2g_apea.get_slice(tid);
    auto tOsApEA = s2g_thr_apea.partition_S(sApEA);   // (CPY, CPY_M, CPY_K, OP)
    auto tOgApEA = s2g_thr_apea.partition_D(gApEA);   // (CPY, CPY_M, CPY_K, k_tiles)

    // ============================================================
    //  PROLOGUE
    // ============================================================
    // Load EAL once (R=64 fits one cp.async block). EAL has no stage dim.
    copy(copy_eal, tEALgEAL, tEALsEAL);

    // Prefetch first (kStages - 1) pipeline stages of A, EBL, EAR.
    constexpr int K_PIPE_MAX = kStages;
    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
      int kb = k_block_min + k_pipe;
      if (kb < k_block_max) {
        copy(copy_a,   tAgA(_, _, _, kb),   tAsA(_, _, _, k_pipe));
        copy(copy_ebl, tEBLgEBL(_, _, _, kb), tEBLsEBL(_, _, _, k_pipe));
        copy(copy_ear, tEARgEAR(_, _, _, kb), tEARsEAR(_, _, _, k_pipe));
      }
      cp_async_fence();
    }

    int smem_pipe_read  = 0;
    int smem_pipe_write = K_PIPE_MAX - 1;
    int smem_pipe_out   = 0;  // ApEA staging slot

    // ============================================================
    //  STEADY STATE: one iter per k_block, sequential MMAs.
    // ============================================================
    // Capture loop bound ONCE before the body (see SM89_PORT_SPEC.md note
    // about the halved-K bug in collective_mainloop_sm89). The inner index
    // mutates pipe_read/write, but the upper bound `k_block_max` is fixed.
    CUTE_NO_UNROLL
    for (int k_block = k_block_min; k_block < k_block_max; ++k_block) {

      // Wait for the *current* stage (smem_pipe_read) to be filled.
      cp_async_wait<K_PIPE_MAX - 2>();
      __syncthreads();

      // --- 1. Load operands from smem to regs for both MMAs ---
      copy(s2r_a, tXsA_mrk(_, _, _, smem_pipe_read), tXrA_mrk);
      copy(s2r_b, tXsEBL(_, _, _, smem_pipe_read), tXrEBL);
      copy(s2r_eal, tXsEAL, tXrEAL);  // EAL loaded once (no pipeline)
      copy(s2r_ear, tXsEAR(_, _, _, smem_pipe_read), tXrEAR);

      // --- 2. MMA: AxEBL += A @ EBL^T (accumulator, NOT cleared) ---
      cute::gemm(tiled_mma_mrk, tCrA_mrk, tCrEBL, tCrAxEBL);

      // --- 3. MMA: ApEA = EAL @ EAR^T (cleared per iter) ---
      clear(tCrApEA);
      cute::gemm(tiled_mma_mkr, tCrEAL, tCrEAR, tCrApEA);

      // --- 4. int32 -> int8 with WRAPPING semantics (match torch.int8 cast).
      // CUTLASS NumericArrayConverter saturates; we need modular wrap so
      // that (e.g.) int32(133) -> int8(-123). Mirrors noisingB sm_89's fix
      // (line 479-486 of pearl_noisingB_kernel_sm89.h). The Hopper path can
      // saturate because its half-MMA+STSM trick bypasses arithmetic cast.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tCrApEA); ++i) {
        tCrApEA_int8(i) = static_cast<int8_t>(
            static_cast<int32_t>(tCrApEA(i)) & 0xff);
      }

      // --- 5. R2S: stage tCrApEA_int8 into smem_ApEA[pipe_out] via
      //            partition_C(tiled_mma_mkr). Plain element-wise stores
      //            since the MMA's V=4 values land at non-contiguous smem
      //            offsets — no STSM, no LDSM, no permute trickery. ---
      {
        auto thr_mma_mkr_for_r2s = tiled_mma_mkr.get_thread_slice(tid);
        auto thr_C_apea = thr_mma_mkr_for_r2s.partition_C(sApEA_pi); // (V, M, K, OP)
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(tCrApEA_int8); ++i) {
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<2>(tCrApEA_int8); ++j) {
            CUTLASS_PRAGMA_UNROLL
            for (int v = 0; v < size<0>(tCrApEA_int8); ++v) {
              thr_C_apea(v, i, j, smem_pipe_out) = tCrApEA_int8(v, i, j);
            }
          }
        }
      }
      __syncthreads();  // R2S visible block-wide

      // --- 6. In-smem add: smem_ApEA[i,k] += smem_A[i,k]  ---
      {
        auto thr_mma_mkr_for_add = tiled_mma_mkr.get_thread_slice(tid);
        auto tCsA_for_add  = thr_mma_mkr_for_add.partition_C(sA_pi);   // (V, M, K, P)
        auto thr_C_apea    = thr_mma_mkr_for_add.partition_C(sApEA_pi); // (V, M, K, OP)
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(tCsA_for_add); ++i) {
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<2>(tCsA_for_add); ++j) {
            CUTLASS_PRAGMA_UNROLL
            for (int v = 0; v < size<0>(tCsA_for_add); ++v) {
              Element a_v = tCsA_for_add(v, i, j, smem_pipe_read);
              Element e_v = thr_C_apea(v, i, j, smem_pipe_out);
              thr_C_apea(v, i, j, smem_pipe_out) = Element(int(a_v) + int(e_v));
            }
          }
        }
      }
      __syncthreads();  // smem_ApEA add results visible

      // --- 7. S2G: smem_ApEA[pipe_out] -> gApEA[k_block] ---
      copy(s2g_apea, tOsApEA(_, _, _, smem_pipe_out),
           tOgApEA(_, _, _, k_block));

      // --- 8. Issue cp.async for the next pipeline stage ---
      int kb_issue = k_block + (K_PIPE_MAX - 1);
      if (kb_issue < k_block_max) {
        copy(copy_a,   tAgA(_, _, _, kb_issue),
                       tAsA(_, _, _, smem_pipe_write));
        copy(copy_ebl, tEBLgEBL(_, _, _, kb_issue),
                       tEBLsEBL(_, _, _, smem_pipe_write));
        copy(copy_ear, tEARgEAR(_, _, _, kb_issue),
                       tEARsEAR(_, _, _, smem_pipe_write));
      }
      cp_async_fence();

      // Advance pipeline indices.
      smem_pipe_read  = (smem_pipe_read + 1) % K_PIPE_MAX;
      smem_pipe_write = (smem_pipe_write + 1) % K_PIPE_MAX;
      smem_pipe_out   = (smem_pipe_out + 1) % kStagesOut;
    }  // end k_block loop

    // Drain any in-flight cp.async groups before we exit.
    cp_async_wait<0>();
    __syncthreads();

    // ============================================================
    //  EPILOGUE: write tCrAxEBL to gmem (M, R) at offset m_block * bM.
    // ============================================================
    // tCrAxEBL has (V, MMA_M, MMA_R) layout. partition_C of a gmem (bM, R)
    // tile by the same tiled_mma_mrk gives an identical per-thread shape.
    Tensor mAxEBL = make_tensor(make_gmem_ptr(params.ptr_AxEBL),
                                 params.layout_AxEBL);
    Tensor gAxEBL = local_tile(mAxEBL, select<0, 1>(TileShape_MRK{}),
                                make_coord(m_block, _0{}));

    auto tCgAxEBL = thr_mma_mrk.partition_C(gAxEBL);

    if constexpr (cute::is_same_v<ElementDenoise, int32_t>) {
      // Bit-exact int32 path: just store accumulators verbatim.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<1>(tCrAxEBL); ++i) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size<2>(tCrAxEBL); ++j) {
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < size<0>(tCrAxEBL); ++v) {
            tCgAxEBL(v, i, j) = tCrAxEBL(v, i, j);
          }
        }
      }
    } else {
      // fp16 path: divide by kAxEBLScaleFactor (= 2^14) and downcast.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<1>(tCrAxEBL); ++i) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size<2>(tCrAxEBL); ++j) {
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < size<0>(tCrAxEBL); ++v) {
            ElementScale scaled =
                static_cast<ElementScale>(tCrAxEBL(v, i, j)) /
                static_cast<ElementScale>(pearl::kAxEBLScaleFactor);
            tCgAxEBL(v, i, j) = static_cast<ElementDenoise>(scaled);
          }
        }
      }
    }
  }
};

}  // namespace pearl
