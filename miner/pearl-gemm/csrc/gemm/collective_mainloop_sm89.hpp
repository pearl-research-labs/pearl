// SPDX-License-Identifier: see LICENSE
//
// sm_89 mainloop collective for pearl-gemm. Int8 x int8 -> int32 GEMM with
// optional inner-hash transcript accumulation. Drops the Hopper
// producer/consumer warp-specialized pipeline for a uniform multistage
// cp.async pipeline (the canonical sm_80/sm_89 CUTLASS pattern).
//
// Cadence reference: CUTLASS examples/cute/tutorial/sgemm_sm80.cu
//   ::gemm_device. We follow its prefetch/wait/issue interleave but:
//     - use int8 atoms (SM80_16x8x32_S32S8S8S32_TN) instead of TF32
//     - add per-k_block hash accumulation (TileHashAccumulator from pow_utils.hpp)
//     - track local_block_found for PoW early-exit signaling
//
// Lifecycle per CTA:
//   PROLOGUE   : issue (kStages - 1) cp.async stages of A, B; fence each
//   REG PREFETCH: wait for stage 0; ldmatrix the k_block=0 frags for A, B
//   STEADY STATE: for each k_tile:
//                  for each k_block:
//                    if last k_block: advance pipe_read, cp.async_wait
//                    ldmatrix k_block_next from smem into regs
//                    if first k_block of tile: issue cp.async for next gmem tile
//                    gemm(tiled_mma, tCrA[k_block], tCrB[k_block], tCrC)
//                    hash_accumulator.accumulate(tCrC, k_block)
//                  hash_accumulator.writeback(transcript)
//   EPILOGUE   : NamedBarrier signal MmaComplete (consumed by epilogue collective)

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor.hpp"

#include <cutlass/arch/barrier.h>
#include "cutlass/cutlass.h"
#include "cutlass/pipeline/pipeline.hpp"

#include "host_signal_header.hpp"
#include "named_barrier.hpp"
#include "pow_utils.hpp"
#include "utils.h"

namespace pearl {
using namespace cute;

// kPersistB: when true, the mainloop exposes a `first_nonce_in_cohort` runtime
// flag (defaulted to true for back-compat). On the first call, both A and B
// are issued through the cp.async prologue and steady-state. On subsequent
// calls within the same CTA's lifetime (same B operand, different A),
// B-fetches are skipped and the kStages B-slots in smem are reused from the
// previous nonce.
//
// NB (correctness scope): B-skip is only safe when num_k_tiles ≤ kStages, i.e.
// every K-tile of B can live in smem simultaneously across the persisted
// cohort. Production K (≥ 1024) violates this — the persisted-B path is wired
// today as a structural hook so the MultiNonceTileScheduler (agent 5/7 owns)
// and the host nonce-batcher can plug into a stable template/runtime API. The
// fall-back path (kPersistB=true + force_b_fetch override) is preserved so
// production K continues to work; the env-flag-gated launcher activates the
// skip only on cohort sizes that fit smem.
//
// See the report at the bottom of csrc/gemm/_test_persist_b.cu for the smem-
// accounting derivation that bounds when B-skip is safe.
template <typename KTraits, bool kPersistB_ = false>
struct CollectiveMainloopSm89 {
  using ElementIn      = typename KTraits::ElementIn;
  using TileShape_MNK  = typename KTraits::TileShape_MNK;
  using TileShape_MNR  = typename KTraits::TileShape_MNR;
  using SmemLayoutA    = typename KTraits::SmemLayoutA;
  using SmemLayoutB    = typename KTraits::SmemLayoutB;
  using TiledMma       = typename KTraits::TiledMma;
  using G2SCopyA       = typename KTraits::G2SCopyA;
  using G2SCopyB       = typename KTraits::G2SCopyB;
  using S2RCopyAtomA   = typename KTraits::S2RCopyAtomA;
  using S2RCopyAtomB   = typename KTraits::S2RCopyAtomB;
  using MMAAtom_K      = typename KTraits::MMAAtom_K;

  using MainloopPipeline = typename KTraits::MainloopPipeline;
  using PipelineState    = typename MainloopPipeline::PipelineState;
  using PipelineParams   = typename MainloopPipeline::Params;

  static constexpr int  kStages        = KTraits::kStages;
  static constexpr int  kNumMmaThreads = KTraits::kNumMmaThreads;
  static constexpr bool kPersistB      = kPersistB_;

  // Layout of A: (M, K) row-major (K-inner). B: (N, K) likewise.
  using LayoutA = cute::Layout<cute::Shape<int, int>, cute::Stride<int64_t, _1>>;
  using LayoutB = cute::Layout<cute::Shape<int, int>, cute::Stride<int64_t, _1>>;

  struct Arguments {
    ElementIn const* ptr_A;
    LayoutA          layout_A;
    ElementIn const* ptr_B;
    LayoutB          layout_B;
    typename KTraits::ProblemShape problem_shape;
    uint32_t const*  ptr_pow_target;
    uint32_t const*  ptr_pow_key;
    void*            host_signal_sync;             // → HostSignalSync*
    void*            host_signal_header_pinned;    // → HostSignalHeader*
    uint64_t*        inner_hash_counter;
  };
  struct Params {
    ElementIn const* ptr_A;
    LayoutA          layout_A;
    ElementIn const* ptr_B;
    LayoutB          layout_B;
    typename KTraits::ProblemShape problem_shape;
    uint32_t const*    ptr_pow_target;
    uint32_t const*    ptr_pow_key;
    HostSignalSync*    host_signal_sync;
    HostSignalHeader*  host_signal_header_pinned;
    uint64_t*          inner_hash_counter;
  };

  static Params to_underlying_arguments(Arguments const& a) {
    return Params{
      a.ptr_A, a.layout_A, a.ptr_B, a.layout_B, a.problem_shape,
      a.ptr_pow_target, a.ptr_pow_key,
      reinterpret_cast<HostSignalSync*>(a.host_signal_sync),
      reinterpret_cast<HostSignalHeader*>(a.host_signal_header_pinned),
      a.inner_hash_counter,
    };
  }
  CUTE_HOST_DEVICE static void prefetch_tma_descriptors(Params const&) {}

  // Block-wide signal: clear DenoiseComplete on entry so the (potential)
  // epilogue's load_denoise gating can re-arrive each iteration.
  CUTLASS_DEVICE void mma_init() const {
    // Sm_89 has no warp-specialization so block-wide __syncthreads()
    // dominates any NamedBarrier-arrive. CUTLASS's NamedBarrier helpers
    // require sm_90 (see cutlass/arch/barrier.h:43 `CUDA_BARRIER_ENABLED`),
    // so we deliberately avoid them.
    if constexpr (!KTraits::SkipDenoising) {
      __syncthreads();
    }
  }

  // first_nonce_in_cohort: must be `true` on the first invocation of this CTA
  //   (default), and `false` on subsequent invocations that share the same
  //   B operand and intend to skip the B cp.async fetches.
  // b_issue_counter: optional device-side uint32 counter; when non-null, each
  //   thread that issues a B cp.async tile-load increments this atomically.
  //   Test harness reads this back to verify that subsequent-nonce invocations
  //   with kPersistB=true actually skip the B fetch.
  template <typename SharedStorage, typename FrgTensorC,
            typename TranscriptTensor>
  CUTLASS_DEVICE void mainloop(
      Params const& params,
      SharedStorage& smem,
      cute::tuple<int, int, int> block_coord,
      int k_tile_count,
      FrgTensorC& tCrC,
      TranscriptTensor& transcript_extraction_tensor,
      bool& local_block_found,
      int& block_found_k_tile,
      int thread_idx,
      bool first_nonce_in_cohort = true,
      unsigned int* b_issue_counter = nullptr) {
    using namespace cute;

    // NB: destructure to named coords; do NOT bind to `_` because that
    //     shadows cute::_ (the slicing sentinel) for the rest of the scope.
    auto m_block = cute::get<0>(block_coord);
    auto n_block = cute::get<1>(block_coord);

    // --------- Build global tensors and tile them for this CTA ----------
    auto mA = make_tensor(make_gmem_ptr(params.ptr_A), params.layout_A);
    auto mB = make_tensor(make_gmem_ptr(params.ptr_B), params.layout_B);

    auto gA = local_tile(mA, select<0, 2>(TileShape_MNK{}),
                           make_coord(m_block, _));   // (bM, bK, k)
    auto gB = local_tile(mB, select<1, 2>(TileShape_MNK{}),
                           make_coord(n_block, _));   // (bN, bK, k)

    // --------- Shared memory tensors (swizzled, kStages-buffered) ------
    auto sA = make_tensor(make_smem_ptr(smem.smem_A.data()),
                            SmemLayoutA{});  // (bM, bK, kStages)
    auto sB = make_tensor(make_smem_ptr(smem.smem_B.data()),
                            SmemLayoutB{});  // (bN, bK, kStages)

    // --------- gmem -> smem copy partition (cp.async) ------------------
    G2SCopyA copy_a;
    G2SCopyB copy_b;
    auto thr_copy_a = copy_a.get_slice(thread_idx);
    auto thr_copy_b = copy_b.get_slice(thread_idx);
    auto tAgA = thr_copy_a.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)
    auto tAsA = thr_copy_a.partition_D(sA);  // (CPY, CPY_M, CPY_K, PIPE)
    auto tBgB = thr_copy_b.partition_S(gB);
    auto tBsB = thr_copy_b.partition_D(sB);

    // --------- MMA partitioning ----------------------------------------
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

    // Register fragments (per-k_block sized — we hold one k_block in regs
    // at a time, double-buffered via tCrA/tCrA_next pattern). Use Int<0>{}
    // (compile-time literal) for the pipeline-dim slice so partition_fragment_A
    // sees a 2D Tensor view, not an element access.
    auto tCrA = thr_mma.partition_fragment_A(sA(_, _, Int<0>{}));
    auto tCrB = thr_mma.partition_fragment_B(sB(_, _, Int<0>{}));

    // --------- smem -> register copy retile (ldmatrix) -----------------
    auto s2r_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_a = s2r_a.get_slice(thread_idx);
    auto s2r_thr_b = s2r_b.get_slice(thread_idx);
    auto tXsA = s2r_thr_a.partition_S(sA);          // (CPY, MMA_M, MMA_K, PIPE)
    auto tXsB = s2r_thr_b.partition_S(sB);
    auto tXrA = s2r_thr_a.retile_D(tCrA);           // (CPY, MMA_M, MMA_K)
    auto tXrB = s2r_thr_b.retile_D(tCrB);

    constexpr int K_PIPE_MAX  = kStages;                  // = stages in flight
    constexpr int K_BLOCK_MAX = size<2>(tCrA);            // k_blocks per tile

    // --------- PROLOGUE: prefetch first kStages-1 stages ---------------
    // When kPersistB and !first_nonce_in_cohort: SKIP B prologue fetches.
    // smem.smem_B retains the kStages-1 B-tiles loaded by the prior nonce's
    // own prologue (the steady-state loop overwrites no more than (kStages-1)
    // B-stages by its end if the K-tile count is bounded; see report).
    bool const fetch_b = first_nonce_in_cohort || !kPersistB;
    int k_tile_next = 0;
    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
      copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
      if (fetch_b) {
        copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
        if (b_issue_counter != nullptr && thread_idx == 0) {
          atomicAdd(b_issue_counter, 1u);
        }
      }
      cp_async_fence();
      --k_tile_count;
      if (k_tile_count > 0) ++k_tile_next;
    }

    int smem_pipe_read  = 0;
    int smem_pipe_write = K_PIPE_MAX - 1;

    // --------- Wait for the first smem stage to be ready ----------------
    // (No register prefetch here: operands are loaded just-in-time per
    // k_block inside the loop. See the single-buffered-B note below.)
    if (K_BLOCK_MAX > 1) {
      cp_async_wait<K_PIPE_MAX - 2>();
      __syncthreads();
    }

    // --------- Hash accumulator -----------------------------------------
    const uint32_t last_full_k_block = get<2>(params.problem_shape) / MMAAtom_K{};
    constexpr int reduce_every_k = KTraits::R / MMAAtom_K{};
    using HashAccumulator = TileHashAccumulator<K_BLOCK_MAX, reduce_every_k,
                                                KTraits::EnableDebug>;
    HashAccumulator hash_accumulator(last_full_k_block, params.inner_hash_counter);

    // --------- STEADY STATE: pipelined gmem->smem and smem->reg --------
    // NB: the loop bound MUST be captured before the loop body, because the
    // inner loop decrements k_tile_count at every k_block==0. If the bound
    // were re-evaluated each iter (e.g. `k_tile < k_tile_count + K_PIPE_MAX-1`),
    // the loop would terminate after one outer iteration on a 2-tile problem
    // (k_tile_count goes 0 → -1 inside iter 0 → bound becomes 1 → iter 1
    // fails the check). The canonical sm_80 multistage uses a while loop on
    // `k_tile_count > -(K_PIPE_MAX-1)`; we capture the equivalent bound here.
    int const k_tile_total = k_tile_count + (K_PIPE_MAX - 1);
    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < k_tile_total; ++k_tile) {
      if constexpr (!KTraits::SkipReduction) {
        hash_accumulator.preload(transcript_extraction_tensor);
      }

      CUTE_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        // Single-buffered operand load: ldmatrix THIS k_block's A/B fragments
        // from the CURRENT smem stage into the SINGLE register fragment
        // immediately before the MMA that consumes them. The prior design
        // prefetched k_block+1 into a separate register slot, keeping TWO sets
        // of B operand fragments (bN=256 → 32 atoms each) co-resident with the
        // 128-int32 accumulator and forcing ptxas to spill the top accumulators
        // (STACK:272). Loading just-in-time halves the live B-operand footprint;
        // the kStages smem cp.async double-buffer still hides global latency.
        copy(s2r_a, tXsA(_, _, k_block, smem_pipe_read), tXrA(_, _, 0));
        copy(s2r_b, tXsB(_, _, k_block, smem_pipe_read), tXrB(_, _, 0));

        // First k_block of tile: issue cp.async for next tile
        if (k_block == 0) {
          if (k_tile_count > 0) {
            copy(copy_a, tAgA(_, _, _, k_tile_next),
                 tAsA(_, _, _, smem_pipe_write));
            if (fetch_b) {
              copy(copy_b, tBgB(_, _, _, k_tile_next),
                   tBsB(_, _, _, smem_pipe_write));
              if (b_issue_counter != nullptr && thread_idx == 0) {
                atomicAdd(b_issue_counter, 1u);
              }
            }
          }
          cp_async_fence();
          --k_tile_count;
          if (k_tile_count > 0) ++k_tile_next;
          smem_pipe_write = (smem_pipe_write + 1) % K_PIPE_MAX;
        }

        // Compute mma for this k_block (sm_80 mma.sync is synchronous;
        // no warpgroup_arrive/commit_batch needed).
        cute::gemm(tiled_mma, tCrA(_, _, 0), tCrB(_, _, 0), tCrC);

        if constexpr (!KTraits::SkipReduction) {
          hash_accumulator.accumulate(tCrC, k_block);
        }

        // Last k_block: advance read pipe to the next smem stage and wait
        // for it (the cp.async issued above / in prior tiles). Done AFTER the
        // MMA so the current k_block's load read the correct (current) stage.
        if (k_block == K_BLOCK_MAX - 1) {
          smem_pipe_read = (smem_pipe_read + 1) % K_PIPE_MAX;
          cp_async_wait<K_PIPE_MAX - 2>();
          __syncthreads();
        }
      }  // end k_block loop

      if constexpr (!KTraits::SkipReduction) {
        hash_accumulator.writeback(transcript_extraction_tensor);
      }
    }  // end k_tile loop

    // Drain any remaining cp.async groups before exit.
    cp_async_wait<0>();
    __syncthreads();
    // No NamedBarrier::arrive(MmaComplete) on sm_89 — block-wide sync above
    // is sufficient since every warp participates in both mainloop and epilogue.

    // PoW handling lives in the kernel driver after this returns.
    (void)local_block_found;
    (void)block_found_k_tile;
  }
};

}  // namespace pearl
