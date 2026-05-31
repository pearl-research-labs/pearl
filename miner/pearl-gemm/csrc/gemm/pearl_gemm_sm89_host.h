// SPDX-License-Identifier: see LICENSE
//
// sm_89 host launcher for pearl-gemm. Companion to the Hopper pearl_gemm_host.h.
// Calls the unified-warp ada_gemm kernel from pearl_gemm_kernel_sm89.h.
//
// Entry point: `void pearl_gemm_sm89_run<KTraits>(args, stream)`.
//
// Four schedulers are reachable from this header:
//
//   PersistentSwizzledTileScheduler  ← used by pearl_gemm_sm89_run (production)
//       Persistent grid sized to num_sm. Grid-stride traversal. L2-aware tile
//       swizzle derived from device L2 cache size and the K dimension. Fixes
//       the 4096³+ L2 cliff that the row-major SimpleTileScheduler walks into
//       (B working-set > 48 MB L2 → DRAM thrash).
//
//   SimpleTileScheduler              ← used by debug + early-port test paths
//       One CTA per output tile, naive row-major. Kept so int32 debug paths
//       continue to behave deterministically while tuning the production
//       scheduler.
//
//   MultiNonceTileScheduler<256>     ← gated by PEARL_SM89_PERSISTENT_NONCE=1
//       Device-side persistent CTA work queue (defined in
//       pearl_gemm_sm89_multinonce_scheduler.hpp). Each launch fans across 256
//       nonces × (num_blocks_m × num_blocks_n) tiles via atomicAdd on a
//       device-side global counter. Foundation for the alpha-miner-style
//       persistent inner-loop; closes ~3-5× of the 13× gap by amortizing launch
//       overhead and keeping B hot in L2 across the full nonce sweep.
//
//   SkinnyShapeTileScheduler         ← gated by PEARL_SM89_STREAMK=1 + skinny shape
//       Aspect-aware DP-only scheduler used when shape is heavily rectangular
//       (M/N or N/M ≥ 4). Sizes grid = total_tiles exactly (no persistent
//       over-subscription) and rasterizes along the short axis first so
//       consecutive CTAs share the long-axis operand tile in L2. Same kernel
//       template, same hash-transcript semantics — only the (tile_idx → m,n)
//       mapping and grid extent differ from PersistentSwizzled.
//
//       Why "STREAMK" naming when this isn't real CUTLASS StreamK:
//         A faithful port of cutlass::gemm::threadblock::ThreadblockSwizzleStreamK
//         requires splitting the K reduction of a single output tile across
//         multiple CTAs and fixing up partial accumulators via either L2
//         atomics or a separate reduction wave. Pearl mining's epilogue derives
//         PoW state and the inner-hash transcript from the FULL-K accumulator
//         of a single CTA; the transcript is consumed bit-exact by share
//         validation. Splitting K across CTAs would require a per-tile
//         cross-CTA hash-combine step that doesn't exist in the protocol — so
//         partial-K StreamK is not a refactor, it's a protocol change.
//         Documentation and bench artifacts keep the STREAMK name because the
//         heuristic + skinny-shape rasterization it enables is what
//         CUTLASS StreamK degenerates to once the data-parallel-efficiency
//         threshold (kDpEfficiencyThreshold=0.92) eliminates partial-K work
//         from its dispatch plan — which empirically holds for every pearl
//         production shape (all ≥85% wave efficiency, see heuristics.hpp
//         is_skinny_aspect / is_partial_wave classification).

#pragma once

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cute/tensor.hpp"

#include "kernel_traits_sm89.hpp"
#include "collective_mainloop_sm89.hpp"
#include "collective_epilogue_sm89.hpp"
#include "pearl_gemm_kernel_sm89.h"
#include "pearl_gemm_sm89_multinonce_scheduler.hpp"

namespace pearl {
namespace sm89 {

// ===========================================================================
// SimpleTileScheduler — naive row-major, 1 CTA per tile. Kept for debug paths
// that want a deterministic launch geometry (test_sm89_debug_int32 etc.).
// ===========================================================================
struct SimpleTileScheduler {
  struct Arguments {
    int num_blocks_m;
    int num_blocks_n;
  };
  struct Params {
    int num_blocks_m;
    int num_blocks_n;
  };
  static Params to_underlying_arguments(Arguments const& a) {
    return Params{a.num_blocks_m, a.num_blocks_n};
  }
  static dim3 get_grid_dim(Arguments const& a, int /*num_sm*/) {
    return dim3(static_cast<uint32_t>(a.num_blocks_m),
                static_cast<uint32_t>(a.num_blocks_n), 1u);
  }

  struct WorkTileInfo {
    int m_block;
    int n_block;
    bool valid;
    CUTLASS_DEVICE bool is_valid(Params const&) const { return valid; }
    template <typename ClusterShape>
    CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t>
    get_block_coord(Params const&) const {
      return cute::make_tuple(m_block, n_block, 0);
    }
  };

  CUTLASS_DEVICE SimpleTileScheduler() {}
  CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const&) const {
    return WorkTileInfo{int(blockIdx.x), int(blockIdx.y), true};
  }
  CUTLASS_DEVICE void init_consumer() const {}
  CUTLASS_DEVICE void prefetch_next_work(Params const&, WorkTileInfo&) const {}
  CUTLASS_DEVICE void broadcast_next_work(WorkTileInfo&) const {}
  template <bool IsProducer = false>
  CUTLASS_DEVICE WorkTileInfo get_next_work(Params const&,
                                            WorkTileInfo const&) const {
    return WorkTileInfo{0, 0, false};  // single-tile per block: done after one
  }
};

// ===========================================================================
// PersistentSwizzledTileScheduler — production sm_89 scheduler.
//
// Persistent grid (gridDim.x = min(num_sm, total_tiles)). Each CTA processes
// tile_idx = blockIdx.x, blockIdx.x + gridDim.x, blockIdx.x + 2*gridDim.x, …
// until is_valid returns false. No atomics.
//
// The linear → (m_block, n_block) mapping applies a CUTLASS-style L2 swizzle
// (cluster-grouped rasterization): within a swizzle group, the major axis
// (longer of M, N) cycles fast for `swizzle` consecutive tile indices before
// the non-major axis advances; that pattern keeps `swizzle` major-axis tiles
// hot in L2 across the wave instead of streaming every tile of the row.
//
// Group size `swizzle` is chosen on the host to fit ~2/3 of the device's L2
// (typically 48 MB on AD103 / 4070 Ti SUPER). See pearl_gemm_sm89_run below.
// ===========================================================================
struct PersistentSwizzledTileScheduler {
  struct Arguments {
    int num_blocks_m;
    int num_blocks_n;
    int swizzle;            // 1 = no swizzle (degenerates to major-fastest scan)
    bool swizzle_n_maj;     // true: N is the major (slow-cycling) axis
  };
  struct Params {
    int total_blocks;
    cutlass::FastDivmod l2_minor_divmod;
    cutlass::FastDivmod l2_major_divmod;
    cutlass::FastDivmod l2_minor_residual_divmod;
    int num_maj_swizzle_groups;
    bool swizzle_n_maj;
  };

  static Params to_underlying_arguments(Arguments const& a) {
    int const num_blocks_nonmaj =
        a.swizzle_n_maj ? a.num_blocks_m : a.num_blocks_n;
    int const num_blocks_maj =
        a.swizzle_n_maj ? a.num_blocks_n : a.num_blocks_m;
    int const swizzle = (a.swizzle > 0) ? a.swizzle : 1;
    int const num_maj_swizzle_groups = num_blocks_maj / swizzle;
    int const num_maj_remainder      = num_blocks_maj % swizzle;
    return Params{
      a.num_blocks_m * a.num_blocks_n,
      cutlass::FastDivmod(swizzle),
      cutlass::FastDivmod(swizzle * num_blocks_nonmaj),
      cutlass::FastDivmod(num_maj_remainder > 0 ? num_maj_remainder : 1),
      num_maj_swizzle_groups,
      a.swizzle_n_maj,
    };
  }

  static dim3 get_grid_dim(Arguments const& a, int num_sm) {
    int const total = a.num_blocks_m * a.num_blocks_n;
    int grid = (num_sm < total) ? num_sm : total;
    if (grid < 1) grid = 1;
    return dim3(static_cast<uint32_t>(grid), 1u, 1u);
  }

  struct WorkTileInfo {
    int tile_idx;
    CUTLASS_DEVICE bool is_valid(Params const& p) const {
      return tile_idx < p.total_blocks;
    }
    template <typename ClusterShape>
    CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t>
    get_block_coord(Params const& p) const {
      int l2_mod, l2_quotient, nonmaj_block, maj_block, l2_maj_block;
      l2_quotient = p.l2_major_divmod.divmod(l2_mod, tile_idx);
      if (l2_quotient < p.num_maj_swizzle_groups) {
        nonmaj_block = p.l2_minor_divmod.divmod(l2_maj_block, l2_mod);
      } else {
        nonmaj_block =
            p.l2_minor_residual_divmod.divmod(l2_maj_block, l2_mod);
      }
      maj_block = l2_maj_block + l2_quotient * p.l2_minor_divmod.divisor;
      int m_block = p.swizzle_n_maj ? nonmaj_block : maj_block;
      int n_block = p.swizzle_n_maj ? maj_block    : nonmaj_block;
      return cute::make_tuple(m_block, n_block, 0);
    }
  };

  CUTLASS_DEVICE PersistentSwizzledTileScheduler() {}
  CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const&) const {
    return WorkTileInfo{int(blockIdx.x)};
  }
  CUTLASS_DEVICE void init_consumer() const {}
  CUTLASS_DEVICE void prefetch_next_work(Params const&, WorkTileInfo&) const {}
  CUTLASS_DEVICE void broadcast_next_work(WorkTileInfo&) const {}
  template <bool IsProducer = false>
  CUTLASS_DEVICE WorkTileInfo
  get_next_work(Params const&, WorkTileInfo const& cur) const {
    return WorkTileInfo{cur.tile_idx + int(gridDim.x)};
  }
};

// ===========================================================================
// SkinnyShapeTileScheduler — DP-only, short-axis-fastest rasterization.
//
// Grid extent = total_tiles exactly (no persistent over-subscription, no
// grid-stride loop). Within the grid, tile_idx maps to (m_block, n_block) by
// iterating the SHORT axis fastest:
//
//   - If num_blocks_n < num_blocks_m (M-skinny, e.g. 16384×4096 → 128×32):
//       n_block = tile_idx % num_blocks_n
//       m_block = tile_idx / num_blocks_n
//     Consecutive CTAs hit consecutive N tiles for the same M row, sharing the
//     A row panel in L2 across `num_blocks_n` CTAs.
//
//   - If num_blocks_m < num_blocks_n (N-skinny, e.g. 4096×16384 → 32×128):
//       m_block = tile_idx % num_blocks_m
//       n_block = tile_idx / num_blocks_m
//     Consecutive CTAs hit consecutive M tiles for the same N column, sharing
//     the B column panel in L2 across `num_blocks_m` CTAs.
//
// This is what CUTLASS' ThreadblockSwizzleStreamK reduces to once K-splitting
// is disabled (its `get_tile_offset` flips between row-major and column-major
// raster based on `tiled_shape().m() < tiled_shape().n()`). The persistent
// loop is omitted because at total_tiles ≥ num_sm (which is true for every
// shape this scheduler activates on), there's exactly one CTA per tile and
// no grid-stride is needed.
//
// Compatibility: same Params/WorkTileInfo interface as PersistentSwizzled,
// safe to bind to the same `ada_gemm` kernel template.
// ===========================================================================
struct SkinnyShapeTileScheduler {
  struct Arguments {
    int num_blocks_m;
    int num_blocks_n;
    bool short_axis_is_n;  // true: N is the short axis (M-skinny)
  };
  struct Params {
    int total_blocks;
    cutlass::FastDivmod short_axis_divmod;
    bool short_axis_is_n;
  };

  static Params to_underlying_arguments(Arguments const& a) {
    int const short_axis_blocks =
        a.short_axis_is_n ? a.num_blocks_n : a.num_blocks_m;
    int const divisor = (short_axis_blocks > 0) ? short_axis_blocks : 1;
    return Params{
      a.num_blocks_m * a.num_blocks_n,
      cutlass::FastDivmod(divisor),
      a.short_axis_is_n,
    };
  }

  static dim3 get_grid_dim(Arguments const& a, int /*num_sm*/) {
    int const total = a.num_blocks_m * a.num_blocks_n;
    int const grid  = (total > 0) ? total : 1;
    return dim3(static_cast<uint32_t>(grid), 1u, 1u);
  }

  struct WorkTileInfo {
    int tile_idx;
    CUTLASS_DEVICE bool is_valid(Params const& p) const {
      return tile_idx < p.total_blocks;
    }
    template <typename ClusterShape>
    CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t>
    get_block_coord(Params const& p) const {
      int long_block, short_block;
      long_block = p.short_axis_divmod.divmod(short_block, tile_idx);
      int m_block = p.short_axis_is_n ? long_block  : short_block;
      int n_block = p.short_axis_is_n ? short_block : long_block;
      return cute::make_tuple(m_block, n_block, 0);
    }
  };

  CUTLASS_DEVICE SkinnyShapeTileScheduler() {}
  CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const&) const {
    return WorkTileInfo{int(blockIdx.x)};
  }
  CUTLASS_DEVICE void init_consumer() const {}
  CUTLASS_DEVICE void prefetch_next_work(Params const&, WorkTileInfo&) const {}
  CUTLASS_DEVICE void broadcast_next_work(WorkTileInfo&) const {}
  template <bool IsProducer = false>
  CUTLASS_DEVICE WorkTileInfo
  get_next_work(Params const&, WorkTileInfo const&) const {
    // Single-tile per block: grid extent = total_tiles, no grid-stride loop.
    return WorkTileInfo{INT_MAX};  // is_valid returns false (INT_MAX > total)
  }
};

// ===========================================================================
// Production launcher. Picks PersistentSwizzledTileScheduler with a swizzle
// width chosen from device L2 cache size. PEARL_SM89_STREAMK=1 + skinny shape
// (aspect ratio ≥ 4) selects SkinnyShapeTileScheduler instead.
// ===========================================================================
template <typename KTraits>
void pearl_gemm_sm89_run(
    typename pearl::CollectiveMainloopSm89<KTraits>::Arguments const& mainloop_args,
    typename pearl::CollectiveEpilogueSm89<KTraits>::Arguments const& epilogue_args,
    int M, int N, int K, cudaStream_t stream,
    NonceContext const* ptr_nonce_contexts = nullptr,
    int nonce_batch_size = 0) {
  using Mainloop  = pearl::CollectiveMainloopSm89<KTraits>;
  using Epilogue  = pearl::CollectiveEpilogueSm89<KTraits>;
  using Scheduler = PersistentSwizzledTileScheduler;

  int const num_blocks_m = (M + KTraits::bM - 1) / KTraits::bM;
  int const num_blocks_n = (N + KTraits::bN - 1) / KTraits::bN;

  // Cache device props per-thread. cudaGetDeviceProperties is ~ms-slow and
  // these values don't change during a process lifetime.
  static thread_local int cached_dev = -1;
  static thread_local int num_sm     = 60;                // AD103 default
  static thread_local int l2_bytes   = 48 * 1024 * 1024;  // AD103 default
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev != cached_dev) {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    num_sm     = p.multiProcessorCount;
    l2_bytes   = p.l2CacheSize;
    cached_dev = dev;
  }

  // Pick longer tile axis as "major". Within a swizzle group the MAJOR axis
  // cycles fast (consecutive CTAs in the group hit consecutive major-axis
  // tiles, reusing the operand tile that lives on the NON-major axis across
  // the group). The group-of-16 override (below) may force this to N-maj
  // regardless of shape, so we compute the effective `num_blocks_maj` after
  // resolving the override.
  bool const swizzle_n_maj = (num_blocks_n >= num_blocks_m);

  // Pick swizzle group width adaptively from shape skewness.
  //   - "Skinny" (one axis ≥ 4× the other): default to small group (S=4) to
  //     break CUDA's default M-fastest dispatch cliff (16384×4096×4096 went
  //     from 8 → 58 TOPS denoise main = 6.94× at S=4).
  //   - "Balanced" (axes within 2× of each other): larger group works better.
  //     S=32 on AD103 keeps 8192³ neutral and adds +7% at 4096³ denoise.
  //   Empirically swept S ∈ {4, 8, 16, 32, 64} on rig04 GPU 0 with mfarm-agent
  //   paused (clean L2). S=4 vs S=32 sit at opposite ends; this two-mode
  //   heuristic picks the better one per shape.
  // Override at runtime via PEARL_SM89_SWIZZLE env var (for ablation).
  int const num_blocks_min = std::max(1, std::min(num_blocks_m, num_blocks_n));
  int const num_blocks_max = std::max(num_blocks_m, num_blocks_n);
  int const shape_ratio = num_blocks_max / num_blocks_min;
  int swizzle = (shape_ratio >= 4) ? 4 : 32;
  bool swizzle_n_maj_eff = swizzle_n_maj;
  // Cache PEARL_SM89_SWIZZLE parse across launches. getenv+atoi is ~1-2µs/call
  // and the env-var value is fixed for the process lifetime. Sentinel 0 means
  // "no override"; >0 means "use this value". (Micro-opt #2)
  static int const cached_swizzle_override = []() {
    char const* envp = std::getenv("PEARL_SM89_SWIZZLE");
    if (envp == nullptr) return 0;
    int v = std::atoi(envp);
    return (v >= 1 && v <= 256) ? v : 0;
  }();
  // Alpha-miner pattern: fixed group-of-16 with N as the major (fast-cycling)
  // axis. Traced from alpha-miner v1.4.0 sm_89 cubin (`block_id ÷ 16` prologue
  // at 0x0090-0x0300). Within a group of 16 consecutive CTAs, M is held
  // constant and N cycles 0..15 → the A row panel for the M tile is reused 16×
  // across the group. Mathematically equivalent to CUTLASS
  // `GemmIdentityThreadblockSwizzle<4>` (group_size = 1<<4 = 16) on the M axis.
  //
  // EMPIRICAL RESULTS (CPU02 4070 Ti SUPER):
  //
  // Wave-3 (2026-05-17, unlocked clocks, mfarm-agent off): reported group-of-16
  // LOSING by 13-22% vs adaptive S=4/32 at 4096³ through 16384×4096. Wave-4
  // re-bench on the same hardware with `nvidia-smi -lgc 2340 -lmc 10501` and
  // 5-trial stability checks shows the wave-3 deltas were an artifact of cold
  // / contended GPU clock state — they did not reproduce.
  //
  // Wave-4 (2026-05-18, locked clocks, 5-trial stability):
  //     2048³:               tied within 0.1% (variants all ~99.9 TOPS)
  //     4096³:               g16 +0.6% (168.27 → 169.27 TOPS, neutral)
  //     8192×8192×4096:      4xM +1.0% (best at this shape)
  //     16384×16384×4096:    tied within 0.3% across all variants
  //     32768×32768×4096:    g16 +0.4% (221.79 → 222.53)
  //     65536×16384×4096:    g16 +1.4% (218.21 → 221.30)
  //     16384×65536×4096:    16xM +1.3% (218.54 → 221.28)
  //     131072×4096×4096:    tied within 0.2% (skinny path, all variants ~205)
  //     4096×131072×4096:    tied within 0.2% (skinny path, all variants ~205)
  //  CSV: pearl-investigation/bench_group16_plus_persistent_2026_05_17.csv.
  //
  // Wave-4 takeaway: there is NO L2 cliff at 65536²+ that any combination of
  // (swizzle width × major axis) unlocks. The PersistentSwizzledTileScheduler
  // already amortizes B-tile reuse across the full output-tile sweep via the
  // grid-stride loop (grid sized to num_sm). Adding group-of-16 on top is
  // +0.4 to +1.4% at large balanced shapes — real but too small to ship as
  // a default change. Production stays on the adaptive S=4/32 picker.
  //
  // Env override `PEARL_SM89_GROUP16_SWIZZLE=1` is kept for ablation +
  // toolchain regression testing. Takes precedence over the adaptive picker
  // but is itself overridden by PEARL_SM89_SWIZZLE / PEARL_SM89_SWIZZLE_NMAJ
  // if those are set.
  static bool const cached_group16_swizzle = []() {
    char const* envp = std::getenv("PEARL_SM89_GROUP16_SWIZZLE");
    return envp != nullptr && envp[0] == '1';
  }();
  if (cached_group16_swizzle) {
    swizzle           = 16;
    swizzle_n_maj_eff = true;  // alpha-miner makes N the fast-cycling axis
  }
  if (cached_swizzle_override > 0) swizzle = cached_swizzle_override;
  // PEARL_SM89_SWIZZLE_NMAJ: ablation override for the major-axis choice.
  // 0 → force M-major; 1 → force N-major; unset → keep adaptive (longer axis).
  // Lets us A/B "group-of-K + N-major (alpha-style)" vs "group-of-K + adaptive"
  // vs "group-of-K + M-major" without touching the swizzle width — used by the
  // group16+persistent combo bench at 65536²+ where the L2 cliff lives.
  static int const cached_nmaj_override = []() {
    char const* envp = std::getenv("PEARL_SM89_SWIZZLE_NMAJ");
    if (envp == nullptr) return -1;
    if (envp[0] == '0') return 0;
    if (envp[0] == '1') return 1;
    return -1;
  }();
  if (cached_nmaj_override == 0) swizzle_n_maj_eff = false;
  if (cached_nmaj_override == 1) swizzle_n_maj_eff = true;
  // Cap by major axis: a group can't be wider than the axis it sits on.
  // Re-resolve num_blocks_maj using the (possibly toggled) swizzle_n_maj_eff
  // so cap stays correct when group16 mode forces N-major.
  int const num_blocks_maj_eff =
      swizzle_n_maj_eff ? num_blocks_n : num_blocks_m;
  if (swizzle > num_blocks_maj_eff) swizzle = num_blocks_maj_eff;
  if (swizzle < 1) swizzle = 1;

  // Env-gated dispatch toggle to the MultiNonceTileScheduler. When
  // PEARL_SM89_PERSISTENT_NONCE=1 AND the caller has supplied a non-null
  // NonceContext array, we instantiate the kernel with
  // MultiNonceTileScheduler<256> and each launch fans across all 256 nonces of
  // (num_blocks_m × num_blocks_n) tiles. Each cohort iterates 256 nonces
  // locally on a single CTA — the inner loop's `first_nonce_in_cohort` flag
  // tells the mainloop whether to skip the B prologue (only safe with the
  // kPersistB=true mainloop instantiation AND num_k_tiles ≤ kStages; both are
  // verified before activation below).
  // Cache PEARL_SM89_PERSISTENT_NONCE parse (micro-opt #2: per-launch getenv).
  static bool const cached_persistent_nonce = []() {
    char const* envp = std::getenv("PEARL_SM89_PERSISTENT_NONCE");
    return envp != nullptr && envp[0] == '1';
  }();
  bool use_multinonce = cached_persistent_nonce;
  if (use_multinonce && ptr_nonce_contexts == nullptr) {
    // Caller asked for the path but didn't supply context pointers. Fall back
    // to the single-nonce path rather than running 256 no-op iterations.
    use_multinonce = false;
  }

  // ---------------------------------------------------------------------------
  // PEARL_SM89_STREAMK: env-gated SkinnyShapeTileScheduler dispatch.
  //
  // Heuristic (see also pearl::sm89::is_skinny_aspect in heuristics.hpp):
  //   activate when shape_ratio = max(num_blocks_m, num_blocks_n) /
  //                                min(num_blocks_m, num_blocks_n) >= 4
  //   AND PEARL_SM89_STREAMK=1.
  //
  // Multi-nonce takes precedence: when persistent-nonce is active we keep the
  // multi-nonce scheduler since it has its own L2 reuse story (B held hot
  // across 256 nonces) that beats any single-launch rasterization order.
  //
  // The heuristic skips activation on shapes where total_tiles < num_sm
  // (under-occupied grid) — that case is handled by the existing persistent
  // scheduler's grid = min(num_sm, total_tiles) policy. Real CUTLASS StreamK
  // would partial-K split here; pearl's transcript semantics prevent that
  // (see file header).
  //
  // EMPIRICAL RESULT (2026-05-17 bench on CPU02 4070 Ti SUPER pearl-ab
  // container, clean GPU, two back-to-back runs):
  //   aspect=4 (4096x16384x4096, 16384x4096x4096): persistent 274-313 TOPS,
  //     streamk 275-316 TOPS → −0.2% to +1.1%, within noise.
  //   aspect=8 (2048x16384x4096, 16384x2048x4096): persistent 269-307 TOPS,
  //     streamk 272-310 TOPS → +0.8% to +1.0%.
  //   balanced (2048/4096/8192/16384²): identical (heuristic skips).
  // The persistent scheduler's adaptive S=4 swizzle on skinny shapes was
  // already capturing most of the headroom that short-axis-fastest
  // rasterization can offer. StreamK ships as a non-regressing alternative
  // with a marginal aspect>=8 edge, gated behind the env-var so production
  // can default to the persistent path while bench/A/B tooling can toggle.
  // Bench CSV at pearl-investigation/bench_streamk_2026_05_17.csv.
  // ---------------------------------------------------------------------------
  static bool const cached_streamk = []() {
    char const* envp = std::getenv("PEARL_SM89_STREAMK");
    return envp != nullptr && envp[0] == '1';
  }();
  int const total_tiles = num_blocks_m * num_blocks_n;
  bool const skinny_aspect = (shape_ratio >= 4);
  bool const enough_tiles  = (total_tiles >= num_sm);
  bool use_streamk = cached_streamk && skinny_aspect && enough_tiles &&
                     !use_multinonce;

  typename Scheduler::Arguments sched_args{
      num_blocks_m, num_blocks_n, swizzle, swizzle_n_maj_eff};
  auto sched_params    = Scheduler::to_underlying_arguments(sched_args);
  auto mainloop_params = Mainloop::to_underlying_arguments(mainloop_args);
  auto epilogue_params = Epilogue::to_underlying_arguments(epilogue_args);

  dim3 grid  = Scheduler::get_grid_dim(sched_args, num_sm);
  dim3 block(KTraits::kNumThreads, 1, 1);

  size_t smem_size = sizeof(typename KTraits::SharedStorage);
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(
        (void const*)&pearl::ada_gemm<KTraits, Scheduler>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_size));
    if (e != cudaSuccess) {
      std::fprintf(stderr,
                   "pearl_gemm_sm89_run: cudaFuncSetAttribute(MaxDynamicSmem=%zu) "
                   "failed: %s — your sm_89 device supports %d KB optin smem; "
                   "shrink kStages or bK in the KTraits instantiation.\n",
                   smem_size, cudaGetErrorString(e),
                   [](){
                     int dev2; cudaGetDevice(&dev2);
                     cudaDeviceProp p2; cudaGetDeviceProperties(&p2, dev2);
                     return int(p2.sharedMemPerBlockOptin / 1024);
                   }());
      return;
    }
    attr_set = true;
  }

  // SkinnyShape kernel's smem requirement is identical to the single-nonce
  // path (same KTraits::SharedStorage), but it's a different __global__ symbol,
  // so cudaFuncSetAttribute must be invoked again for the new template binding.
  using SkinnySched = SkinnyShapeTileScheduler;
  static bool sk_attr_set = false;
  if (use_streamk && !sk_attr_set) {
    cudaError_t e = cudaFuncSetAttribute(
        (void const*)&pearl::ada_gemm<KTraits, SkinnySched>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_size));
    if (e != cudaSuccess) {
      std::fprintf(stderr,
                   "pearl_gemm_sm89_run: streamk cudaFuncSetAttribute"
                   "(MaxDynamicSmem=%zu) failed: %s — falling back to "
                   "PersistentSwizzled.\n",
                   smem_size, cudaGetErrorString(e));
      use_streamk = false;
    } else {
      sk_attr_set = true;
    }
  }

  // Multi-nonce kernel's smem requirement is identical to the single-nonce
  // path (same KTraits::SharedStorage), but it's a different __global__ symbol,
  // so cudaFuncSetAttribute must be invoked again for the new template binding.
  using MultiNonceSched = MultiNonceTileScheduler</*NonceCount=*/256>;
  static bool mn_attr_set = false;
  if (use_multinonce && !mn_attr_set) {
    cudaError_t e = cudaFuncSetAttribute(
        (void const*)&pearl::ada_gemm<KTraits, MultiNonceSched>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_size));
    if (e != cudaSuccess) {
      std::fprintf(stderr,
                   "pearl_gemm_sm89_run: multinonce cudaFuncSetAttribute"
                   "(MaxDynamicSmem=%zu) failed: %s\n",
                   smem_size, cudaGetErrorString(e));
      use_multinonce = false;  // fall back to single-nonce path
    } else {
      mn_attr_set = true;
    }
  }

  // Tier 2b: L2 access policy window — pin operand B as "persisting" in L2 up
  // to (min(48 MB, sizeof B)). Pearl mining holds B (the weight matrix)
  // constant across many nonces while A varies; persisting B keeps it hot.
  // For a one-shot bench A/B this is a small win, but production with persistent
  // CTA over 256 nonces (alpha-miner style) will see a bigger effect.
  //
  // Tunables (read once per launch — env-var override only, no flag wiring):
  //   PEARL_SM89_NO_L2_POLICY    — set to ablate the window entirely.
  //   PEARL_SM89_L2_HIT_RATIO    — float in (0, 1]. Fraction of accesses inside
  //                                the window that get `hitProp`; the rest get
  //                                `missProp`. Default 1.0. Lowering it leaves
  //                                some way-set capacity for non-B traffic
  //                                (epilogue C writes, A streaming reads).
  //   PEARL_SM89_L2_WINDOW_BYTES — soft cap on window size, in bytes. Default
  //                                = l2CacheSize (48 MB on AD103). The actual
  //                                window is min(sizeof B, l2_bytes, env-cap).
  //                                Useful at 8192³+ where sizeof(B)=64 MB so
  //                                the window's first 48 MB get tagged
  //                                persisting and the rest stream — but
  //                                shrinking the window can leave more capacity
  //                                for A reads.
  // Cache the L2 policy env vars (micro-opt #2: per-launch getenv).
  //   l2_no_policy: nonzero if PEARL_SM89_NO_L2_POLICY was set.
  //   l2_window_cap_bytes: 0 means "no cap"; >0 means cap window to that many bytes.
  //   l2_hit_ratio_override: <0 means "no override"; in (0,1] means use this value.
  static bool const l2_no_policy = (std::getenv("PEARL_SM89_NO_L2_POLICY") != nullptr);
  static size_t const l2_window_cap_bytes = []() -> size_t {
    char const* envb = std::getenv("PEARL_SM89_L2_WINDOW_BYTES");
    if (envb == nullptr) return 0;
    return static_cast<size_t>(std::strtoull(envb, nullptr, 10));
  }();
  static float const l2_hit_ratio_override = []() {
    char const* envh = std::getenv("PEARL_SM89_L2_HIT_RATIO");
    if (envh == nullptr) return -1.0f;
    float v = static_cast<float>(std::atof(envh));
    return (v > 0.0f && v <= 1.0f) ? v : -1.0f;
  }();

  cudaStreamAttrValue prev_attr{};
  bool restore_attr = false;
  if (mainloop_args.ptr_B != nullptr && !l2_no_policy) {
    size_t const B_bytes = static_cast<size_t>(N) * static_cast<size_t>(K);
    size_t window        = std::min<size_t>(B_bytes, l2_bytes);
    if (l2_window_cap_bytes > 0 && l2_window_cap_bytes < window) {
      window = l2_window_cap_bytes;
    }
    float hit_ratio = (l2_hit_ratio_override > 0.0f) ? l2_hit_ratio_override : 1.0f;
    // Save existing attribute so we can restore it (caller may have set one).
    (void)cudaStreamGetAttribute(
        stream, cudaStreamAttributeAccessPolicyWindow, &prev_attr);
    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr   = const_cast<int8_t*>(mainloop_args.ptr_B);
    attr.accessPolicyWindow.num_bytes  = window;
    attr.accessPolicyWindow.hitRatio   = hit_ratio;
    attr.accessPolicyWindow.hitProp    = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp   = cudaAccessPropertyStreaming;
    if (cudaStreamSetAttribute(
            stream, cudaStreamAttributeAccessPolicyWindow, &attr) == cudaSuccess) {
      restore_attr = true;
    }
  }

  if (use_multinonce) {
    // ---- Multi-nonce launch path ----
    // Allocate (or reuse) the device-side global work counter. We hold one
    // counter per process inside a per-stream cudaMalloc; the cost is ~4 B
    // and amortizes across all launches. Memset to 0 on every launch.
    static thread_local uint32_t* d_work_idx = nullptr;
    if (d_work_idx == nullptr) {
      if (cudaMalloc(&d_work_idx, sizeof(uint32_t)) != cudaSuccess) {
        std::fprintf(stderr,
                     "pearl_gemm_sm89_run: cudaMalloc(d_work_idx) failed; "
                     "falling back to single-nonce path.\n");
        d_work_idx = nullptr;
        use_multinonce = false;
      }
    }
    if (use_multinonce) {
      cudaMemsetAsync(d_work_idx, 0, sizeof(uint32_t), stream);

      typename MultiNonceSched::Arguments mn_args{
          num_blocks_m, num_blocks_n, d_work_idx,
          ptr_nonce_contexts};
      auto mn_params = MultiNonceSched::to_underlying_arguments(mn_args);
      dim3 mn_grid   = MultiNonceSched::get_grid_dim(mn_args, num_sm);

      pearl::ada_gemm<KTraits, MultiNonceSched>
          <<<mn_grid, block, smem_size, stream>>>(
              mainloop_params, epilogue_params, mn_params);

      if (restore_attr) {
        cudaStreamSetAttribute(
            stream, cudaStreamAttributeAccessPolicyWindow, &prev_attr);
      }
      return;
    }
  }

  if (use_streamk) {
    // ---- StreamK skinny-shape launch path ----
    // Short axis becomes the fast-cycling axis: when N is shorter, consecutive
    // CTAs hit consecutive N tiles (sharing the A row in L2); when M is
    // shorter, consecutive CTAs hit consecutive M tiles (sharing the B
    // column). Grid = total_tiles exactly so no CTA is wasted.
    bool const short_axis_is_n = (num_blocks_n <= num_blocks_m);
    typename SkinnySched::Arguments sk_args{
        num_blocks_m, num_blocks_n, short_axis_is_n};
    auto sk_params = SkinnySched::to_underlying_arguments(sk_args);
    dim3 sk_grid   = SkinnySched::get_grid_dim(sk_args, num_sm);

    pearl::ada_gemm<KTraits, SkinnySched>
        <<<sk_grid, block, smem_size, stream>>>(
            mainloop_params, epilogue_params, sk_params);

    if (restore_attr) {
      cudaStreamSetAttribute(
          stream, cudaStreamAttributeAccessPolicyWindow, &prev_attr);
    }
    return;
  }

  pearl::ada_gemm<KTraits, Scheduler>
      <<<grid, block, smem_size, stream>>>(mainloop_params, epilogue_params,
                                           sched_params);

  if (restore_attr) {
    // Restore previous attribute (or zero it if none was set).
    cudaStreamSetAttribute(
        stream, cudaStreamAttributeAccessPolicyWindow, &prev_attr);
  }
}

}  // namespace sm89
}  // namespace pearl
