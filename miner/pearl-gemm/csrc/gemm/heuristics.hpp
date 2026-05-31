#pragma once

#include <cuda_runtime_api.h>
#include <cutlass/numeric_types.h>
#include <algorithm>  // std::min, std::max
#include <cmath>      // std::ceil

static inline int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

static constexpr int64_t kDefaultNoisingTileSizeMN = 64;
static constexpr int64_t kDefaultNoisingTileSizeK = 64;

// =============================================================================
// Shape classification helpers for the sm_89 tile scheduler.
//
// Used by pearl_gemm_sm89_host.h's `pearl_gemm_sm89_run` and by bench harnesses
// that want to mirror the launcher's scheduler selection. Centralized here so
// downstream tooling (Python heuristic doctors, future arch.hpp dispatch)
// doesn't reimplement the classification.
//
// Threshold chosen at 4 because empirically:
//   - aspect 1..2:  PersistentSwizzled S=32 wins (balanced, big L2 group)
//   - aspect 3:     PersistentSwizzled S=4  marginally wins (skew but enough
//                   tiles for the persistent stride to share B effectively)
//   - aspect >=4:   short-axis-fastest rasterization (SkinnyShape) can win at
//                   the cost of giving up the persistent stride's per-CTA
//                   L2 amortization. The PEARL_SM89_STREAMK gate makes this
//                   opt-in until A/B'd at every production shape.
// =============================================================================
namespace pearl { namespace sm89 {

// Aspect ratio of the tile grid: max(num_blocks_m, num_blocks_n) /
// min(num_blocks_m, num_blocks_n). Returns 1 for degenerate (single-tile) grids.
inline int tile_grid_aspect_ratio(int num_blocks_m, int num_blocks_n) {
  int const lo = std::max(1, std::min(num_blocks_m, num_blocks_n));
  int const hi = std::max(num_blocks_m, num_blocks_n);
  return hi / lo;
}

// "Skinny" = aspect ratio of the tile grid >= 4. SkinnyShapeTileScheduler is
// designed for this regime.
inline bool is_skinny_aspect(int num_blocks_m, int num_blocks_n) {
  return tile_grid_aspect_ratio(num_blocks_m, num_blocks_n) >= 4;
}

// "Partial wave" = grid total tiles less than num_sm, i.e. less than one
// full GPU-wide wave. This is where CUTLASS StreamK would partial-K-split to
// fill the missing SMs; pearl's hash transcript prevents that, so callers
// can use this signal to log a warning or pick a smaller bM/bN tile.
inline bool is_partial_wave(int num_blocks_m, int num_blocks_n, int num_sm) {
  return num_blocks_m * num_blocks_n < num_sm;
}

}}  // namespace pearl::sm89

static inline int get_swizzle_size(int K, int tile_size_n,
                                   cudaDeviceProp const* const dprops) {
  // heuristic: allocate approximately 2/3 of L2 cache for tiles of B
  int B_size_bytes = tile_size_n * K;
  int L2_size_bytes = dprops->l2CacheSize;
  int swizzle = (2 * L2_size_bytes / 3) / B_size_bytes;
  // Round down to a multiple of 4
  swizzle = 4 * (swizzle / 4);
  return std::min(128, swizzle);
}

static inline int get_pipeline_stages(int tile_size_m, int tile_size_n,
                                      int tile_size_k, int R,
                                      bool skip_denoising,
                                      cudaDeviceProp const* const dprops) {
  int const smem_size = dprops->sharedMemPerBlockOptin;
  // A, B (int8) and their pipeline (2 int64 mbarriers per stage)
  int const AB_one_stage_size = (tile_size_m * tile_size_k) +
                                (tile_size_n * tile_size_k) +
                                (2 * sizeof(int64_t));
  // C (bf16)
  int const C_size = tile_size_m * tile_size_n * sizeof(cutlass::bfloat16_t);
  // AxEBL, EBR overlap with C for load.
  int const AxEB_size =
      skip_denoising
          ? 0
          : (sizeof(cutlass::half_t) * tile_size_m + tile_size_n) * R;
  int const C_union_size = std::max(C_size, AxEB_size);
  // A_scales, B_scales (fp32)
  int const scale_size = (tile_size_m + tile_size_n) * sizeof(float);
  int const rest_size = 128;

  int const pipeline_stages =
      (smem_size - (C_union_size + scale_size + rest_size)) / AB_one_stage_size;
  return std::max(1, pipeline_stages);
}

static inline int get_num_k_blocks(int MN, int tile_size_mn, int K,
                                   int tile_size_k,
                                   cudaDeviceProp const* const dprops) {
  /* Heuristic: for Split-K noising kernels, pick a split size that gets
     good GPU utilization.

     Without splitting, the noising kernels divide the work into worktiles of
     size tile_size_mn. All worktiles contain approximately the same amount of
     work, so they tend to be processed in "waves". In a single wave, each SM
     will be fully occupied by some number of CTAs, and all those CTAs will
     process a single worktile. The exception is the tail wave, in which some
     SMs may not be fully occupied, but still have to wait for the kernel to
     finish. This leads to GPU under-utilization.

     A single wave processes
         num_ctas = ctas_per_sm * num_sms
     worktiles, so the (non-integral) number of waves is
         num_waves = num_worktiles / num_ctas.
     We define
         wave_efficiency = num_waves / ceil(num_waves)
     and, if the non-split kernel would underutilize the GPU, choose a split
     size that maximizes wave efficiency.

     For example, suppose that tile_size_m = 64, tile_size_k = 128, and
     M = K = 8192, and we run on an H200 with 128 SMs.
     1 wave = 2 CTAs/SM * 128 SMs = 256 worktiles.
     Without splitting, we have only 8192/64 = 128 worktiles, or 50% wave
     efficiency.
     By splitting into 2 splits (so num_k_blocks_per_split = 32), we get 100%
     wave effiency.
  */
  int k_blocks_per_tile = ceil_div(K, tile_size_k);
  int total_num_blocks = ceil_div(MN, tile_size_mn) * k_blocks_per_tile;
  int num_sms = dprops->multiProcessorCount;
  // CTAs per SM determined by max occupancy we can get from the kernel
  int desired_CTAs_per_SM = 2;
  int num_ctas = desired_CTAs_per_SM * num_sms;

  auto get_num_waves = [&](int num_k_blocks_per_split) {
    int num_work_items = ceil_div(total_num_blocks, num_k_blocks_per_split);
    return static_cast<float>(num_work_items) / static_cast<float>(num_ctas);
  };

  auto get_wave_efficiency = [&](int num_k_blocks_per_split) {
    float waves = get_num_waves(num_k_blocks_per_split);
    return waves / std::ceil(waves);
  };

  // If we can get almost 1 full wave without splitting, do so
  if (get_num_waves(k_blocks_per_tile) >= 0.8f) {
    // don't split
    return 0;
  }
  // Otherwise, pick the number of blocks that gets the best wave efficiency
  //  However, if num_splits is too large then the kernel also becomes inefficient
  //  due to increased GMEM reads.  So we find the best wave efficiency and then choose
  //  the smallest number of splits that gets at least 85% of the best efficiency.
  float best_wave_efficiency = 0.f;
  std::vector<float> wave_efficiencies;
  wave_efficiencies.reserve(k_blocks_per_tile);
  for (int num_splits = 2; num_splits < k_blocks_per_tile; ++num_splits) {
    int num_k_blocks_per_split = ceil_div(k_blocks_per_tile, num_splits);
    float wave_efficiency = get_wave_efficiency(num_k_blocks_per_split);
    if (wave_efficiency > best_wave_efficiency) {
      best_wave_efficiency = wave_efficiency;
    }
    wave_efficiencies.push_back(wave_efficiency);
  }
  for (int num_splits = 2; num_splits < k_blocks_per_tile; ++num_splits) {
    if (wave_efficiencies[num_splits - 2] >= 0.85 * best_wave_efficiency) {
      int best_num_k_blocks = ceil_div(k_blocks_per_tile, num_splits);
      return best_num_k_blocks;
    }
  }
  return 0;
}
