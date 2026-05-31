// SPDX-License-Identifier: see LICENSE
//
// MultiNonceTileScheduler — device-side persistent CTA work queue that fans a
// single kernel launch across NonceCount nonces and (num_blocks_m × num_blocks_n)
// output tiles per nonce.
//
// Companion to PersistentSwizzledTileScheduler in pearl_gemm_sm89_host.h, which
// only iterates the tiles of ONE nonce per launch. This scheduler is the
// foundation for closing the ~13× gap to alpha-miner: a single launch covers
// 256 nonces, persistent CTAs amortize launch overhead, and B (held constant
// across nonces in the mining inner loop) is hot in L2 across the entire wave.
//
// Atomic-counter claim model:
//   * Host allocates a single uint32_t in device memory (`global_work_idx`),
//     memset to 0 before launch.
//   * Each CTA loops: warp-elected thread does `atomicAdd(global_work_idx, 1)`;
//     warp-broadcasts the claimed index; tile + nonce coords decode from it.
//   * When the claimed index >= total_work, the CTA exits.
//
// This is the canonical Hopper PersistentTileSchedulerWS model adapted to sm_89
// (no clusters, no setmaxnreg). It eliminates the row-major bias of an
// implicit blockIdx.x-based assignment and lets fast CTAs steal work from
// slow ones automatically.
//
// Linear (chunked) → (m_block, n_block, nonce_idx) decomposition.
//
// Two granularities for atomicAdd:
//
//   (a) Per work item (1-claim, linear):
//         raw_idx = atomicAdd(counter, 1)         // 0..total_work-1
//         m_block, n_block, nonce_idx = decode(raw_idx)
//       Maximally fair, but consecutive raw_idx claims by ONE CTA are not
//       guaranteed to land on the same (m, n) — interleaving CTAs can split a
//       cohort across CTAs and break smem-reuse correctness for kPersistB.
//
//   (b) Per cohort (NonceCount-claim, chunked) — THIS IMPLEMENTATION:
//         cohort_idx = atomicAdd(counter, 1)      // 0..tiles_per_nonce-1
//         (m_block, n_block) = decode(cohort_idx)
//         nonce_idx = local 0..NonceCount-1 (CTA-local loop)
//       Each CTA atomically claims one (m, n) tile AND all NonceCount nonces
//       for it. The CTA iterates the cohort locally, exposing
//       first_nonce_in_cohort=(local==0) to the kernel body. This is the
//       cleanest setup for the kPersistB hook in collective_mainloop_sm89.hpp
//       (`first_nonce_in_cohort=true` -> full A+B fetch; `=false` -> B-reuse).
//
// Rasterization within the (m, n) plane: major-N (n_block varies fastest)
// matches the existing PersistentSwizzledTileScheduler's default and keeps
// B-tiles hot in L2 across the cohort wave. A later L2-aware swizzle can be
// slotted in alongside the PersistentSwizzledTileScheduler machinery in
// pearl_gemm_sm89_host.h without disturbing the per-cohort claim contract.
//
// TODO(agent-4 cross-nonce smem reuse, kPersistB integration):
//   The mainloop already exposes a `kPersistB` template flag + `first_nonce_in_
//   cohort` runtime flag (collective_mainloop_sm89.hpp:63,156). To activate
//   smem-B reuse end-to-end this scheduler must:
//     1. Surface `first_nonce_in_cohort` on the WorkTileInfo (DONE here).
//     2. Have the kernel body forward it as the 10th arg of mainloop().
//     3. Have the host launcher instantiate the kernel with a mainloop bound
//        as `CollectiveMainloopSm89<KTraits, /*kPersistB=*/true>` when the
//        problem K-tile count fits the smem-B accounting bound described in
//        _test_persist_b.cu.
//   Step 1 is the only piece this scheduler owns; 2-3 are downstream wiring.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cute/tensor.hpp"

namespace pearl {
namespace sm89 {

// ===========================================================================
// NonceContext — per-nonce work item description.
//
// Pointers MUST be device pointers. The scheduler reads
// `nonce_contexts[nonce_idx].ptr_A` etc. inside the mainloop. Layout/strides
// are inherited from Mainloop::Params (single layout reused across nonces —
// all nonces share the same M, N, K).
//
// Wave-13: extended with per-nonce PoW signal slots so each nonce in a 256-
// nonce launch can write its own HostSignalHeader without colliding on a
// single global_lock CAS. The kernel's per-iter param override patches all 7
// pointers below from `ptr_nonce_contexts[nonce_idx]` (see
// pearl_gemm_kernel_sm89.h, HasNonceContextsField branch).
// ===========================================================================
struct NonceContext {
  int8_t const*       ptr_A;                     // per-nonce A matrix (m, k) row-major
  float  const*       ptr_A_scales;              // per-nonce A row scales (m,)
  // Per-nonce output offset, expressed as a raw bfloat16_t* (caller decides
  // whether it's an offset into a contiguous output buffer or a separate
  // allocation). Mainloop adds nothing — epilogue writes here directly.
  void*               ptr_C;                     // bfloat16_t* (kept as void* to avoid
                                                  // pulling cutlass numeric_types here)
  // -- Wave-13 per-nonce PoW signal slots --------------------------------------
  void*               host_signal_header_pinned; // HostSignalHeader* (pinned host mem)
  void*               host_signal_sync;          // HostSignalSync*    (device mem)
  uint32_t const*     ptr_pow_target;            // uint32_t[8]        (device mem)
  uint32_t const*     ptr_pow_key;               // uint32_t[8]        (device mem)
};

// ===========================================================================
// MultiNonceTileScheduler<NonceCount>
//
// Interface contract (matches PersistentSwizzledTileScheduler):
//   * to_underlying_arguments(Arguments) -> Params
//   * get_grid_dim(Arguments, num_sm)    -> dim3
//   * WorkTileInfo::is_valid(Params)     -> bool
//   * WorkTileInfo::get_block_coord<ClusterShape>(Params)
//         -> cute::tuple<int32_t, int32_t, int32_t>  // (m_block, n_block, nonce_idx)
//   * get_initial_work(Params)           -> WorkTileInfo (first claim)
//   * get_next_work<IsProducer=false>(Params, WorkTileInfo) -> WorkTileInfo
//   * init_consumer(), prefetch_next_work(), broadcast_next_work()  — no-op stubs
//
// The `nonce_idx` slot of get_block_coord is read by the mainloop to redirect
// `ptr_A`/`ptr_A_scales` via `params.ptr_nonce_contexts[nonce_idx]`. The
// PersistentSwizzledTileScheduler returns `0` there (single-nonce mode), and
// the mainloop falls back to `params.ptr_A` when `ptr_nonce_contexts == nullptr`.
// ===========================================================================
template <int NonceCount_>
struct MultiNonceTileScheduler {
  static constexpr int NonceCount = NonceCount_;
  static_assert(NonceCount > 0, "NonceCount must be positive");

  struct Arguments {
    int num_blocks_m;
    int num_blocks_n;
    // Device pointer to a 4-byte atomic counter; host sets to 0 pre-launch.
    uint32_t* global_work_idx;
    // Device array of per-nonce contexts, length >= NonceCount. May be nullptr
    // for testing the scheduler in degenerate "fan only" mode (every nonce
    // uses the unchanged Mainloop/Epilogue Params).
    NonceContext const* ptr_nonce_contexts;
  };

  struct Params {
    int num_blocks_m;
    int num_blocks_n;
    int tiles_per_nonce;        // = num_blocks_m * num_blocks_n  (= cohort count)
    cutlass::FastDivmod num_blocks_n_divmod;  // for (m, n) decode of cohort_idx
    uint32_t* global_work_idx;
    NonceContext const* ptr_nonce_contexts;
  };

  static Params to_underlying_arguments(Arguments const& a) {
    int const tpn = a.num_blocks_m * a.num_blocks_n;
    return Params{
        a.num_blocks_m,
        a.num_blocks_n,
        tpn,
        cutlass::FastDivmod(a.num_blocks_n > 0 ? a.num_blocks_n : 1),
        a.global_work_idx,
        a.ptr_nonce_contexts,
    };
  }

  // Persistent grid: one CTA per SM, capped by cohort count. Each CTA claims
  // one cohort_idx via atomicAdd, iterates NonceCount nonces locally, exits
  // when the counter exceeds tiles_per_nonce.
  //
  // NB: grid dim is bounded by cohort count (tiles_per_nonce), NOT total work
  // (= tiles_per_nonce × NonceCount). One CTA per cohort lets the kernel body
  // loop NonceCount work items locally before claiming the next cohort.
  static dim3 get_grid_dim(Arguments const& a, int num_sm) {
    int const cohorts = a.num_blocks_m * a.num_blocks_n;
    int grid = (num_sm < cohorts) ? num_sm : cohorts;
    if (grid < 1) grid = 1;
    return dim3(static_cast<uint32_t>(grid), 1u, 1u);
  }

  struct WorkTileInfo {
    int  cohort_idx;             // claimed (m, n) tile index, 0..tiles_per_nonce-1
    int  m_block;
    int  n_block;
    int  nonce_idx;              // local 0..NonceCount-1 within the cohort
    bool first_nonce_in_cohort;  // == (nonce_idx == 0)
    bool valid;

    CUTLASS_DEVICE bool is_valid(Params const&) const { return valid; }

    template <typename ClusterShape>
    CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t>
    get_block_coord(Params const&) const {
      return cute::make_tuple(m_block, n_block, nonce_idx);
    }
  };

  CUTLASS_DEVICE MultiNonceTileScheduler() {}

  // ---- Claim a fresh cohort (one (m, n) tile, all NonceCount nonces). ----
  // Caller MUST be inside a __syncthreads/warp-uniform region before calling.
  // Thread 0 does the atomicAdd; result is broadcast via smem. nonce_idx is
  // set to 0 — the kernel body then ticks through NonceCount-1 via
  // get_next_work() without re-incrementing the global counter (see below).
  CUTLASS_DEVICE WorkTileInfo claim_next_cohort(Params const& p) const {
    __shared__ uint32_t s_cohort_idx;

    if (threadIdx.x == 0) {
      s_cohort_idx = atomicAdd(p.global_work_idx, 1u);
    }
    __syncthreads();
    uint32_t const cohort = s_cohort_idx;

    WorkTileInfo info{};
    info.cohort_idx            = static_cast<int>(cohort);
    info.nonce_idx             = 0;
    info.first_nonce_in_cohort = true;
    info.valid                 = (static_cast<int>(cohort) < p.tiles_per_nonce);
    if (!info.valid) return info;

    // major-N rasterization: cohort_idx = m * num_blocks_n + n.
    int n_b;
    int m_b = p.num_blocks_n_divmod.divmod(n_b, static_cast<int>(cohort));
    info.m_block = m_b;
    info.n_block = n_b;
    return info;
  }

  CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& p) const {
    return claim_next_cohort(p);
  }

  CUTLASS_DEVICE void init_consumer() const {}
  CUTLASS_DEVICE void prefetch_next_work(Params const&, WorkTileInfo&) const {}
  CUTLASS_DEVICE void broadcast_next_work(WorkTileInfo&) const {}

  // ---- Get next work item ---------------------------------------------------
  // If the CTA still has nonces left in its current cohort (nonce_idx <
  // NonceCount - 1), advance LOCALLY — no atomic. Otherwise claim a new
  // cohort via atomicAdd.
  template <bool IsProducer = false>
  CUTLASS_DEVICE WorkTileInfo
  get_next_work(Params const& p, WorkTileInfo const& cur) const {
    if (cur.valid && cur.nonce_idx + 1 < NonceCount) {
      WorkTileInfo next               = cur;
      next.nonce_idx                  = cur.nonce_idx + 1;
      next.first_nonce_in_cohort      = false;
      return next;
    }
    return claim_next_cohort(p);
  }
};

}  // namespace sm89
}  // namespace pearl
