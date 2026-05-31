// SPDX-License-Identifier: see LICENSE
//
// Torch-free on-device noise-factor generator for the standalone sm_89 miner.
//
// Wraps pearl::NoiseGenerationKernel<R, NumThreads> (the SAME kernel the
// production pipeline + the e2e tests use) WITHOUT pulling in c10/torch. The
// stock launcher run_noise_generation() (noise_generation_host.h) includes
// error_check.hpp -> c10/util/Exception.h, which we cannot link in the
// dependency-free miner. This file re-implements the launch with a plain
// cudaGetLastError check and instantiates the kernel at R=256 (the production
// mining noise_rank — the stock noise_generation.cu only instantiates R<=128).
//
// Correctness: the kernel's per-thread blake3 message index is the LINEAR
// 32-byte-chunk index over the flattened (rows, R) / (k, R) tensors
// (thread_coord = bid_in_category*NumThreads + tid + 1). This indexing is
// independent of R, so the R=256 output is bit-exact with
// miner-base/noise_generation.py and pearl_miner_host.hpp's noise_dense /
// noise_sparse (verified by the verify-mode host gate, which uses the same
// reference). seed_A="A_tensor", seed_B="B_tensor", keys = a/b_noise_seed.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "cute/tensor.hpp"
#include "noise_generation_kernel.h"

#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>

// Device-side bit-exact replica of the host fill_AB() splitmix64 contract
// (pearl_miner_sm89.cu): dst[i] = (int8)((splitmix64_i(seed) % 127) - 63).
// Each thread regenerates its own splitmix64 stream by fast-forwarding the
// additive constant (splitmix64 state after i steps = seed + (i+1)*GAMMA).
__device__ __forceinline__ uint64_t sm64_mix(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
__global__ void fill_AB_kernel(int8_t* dst, size_t n, uint64_t seed) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // host: s starts at `seed`; iteration i does s += GAMMA then mixes s.
  // So the state fed to the mixer at step i is seed + (i+1)*GAMMA.
  uint64_t s = seed + (i + 1) * 0x9E3779B97F4A7C15ULL;
  uint64_t r = sm64_mix(s);
  dst[i] = (int8_t)((int)(r % 127) - 63);
}

extern "C" void pearl_miner_fill_AB_sm89(int8_t* dst, size_t n, uint64_t seed,
                                         cudaStream_t stream) {
  int block = 256;
  size_t grid = (n + block - 1) / block;
  fill_AB_kernel<<<grid, block, 0, stream>>>(dst, n, seed);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "fill_AB launch: %s\n", cudaGetErrorString(e));
    std::abort();
  }
}

// Split an R-major (rows, R=256) int8 matrix into two contiguous (rows, 128)
// halves: out_lo gets columns [0,128), out_hi columns [128,256). The chained
// two-pass noisingA/B reads each R-half as a STANDALONE contiguous (rows,128)
// matrix (the R=128 kernel hard-codes ld=128), so a strided slice of the
// (rows,256) buffer is NOT what it expects — we must physically repack.
__global__ void split_rmajor_256_kernel(const int8_t* in, int8_t* out_lo,
                                        int8_t* out_hi, size_t rows) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = rows * 128;
  if (i >= total) return;
  size_t r = i / 128, c = i % 128;     // dest (row, col<128)
  out_lo[i] = in[r * 256 + c];
  out_hi[i] = in[r * 256 + 128 + c];
}

extern "C" void pearl_miner_split_rmajor_256_sm89(
    const int8_t* in, int8_t* out_lo, int8_t* out_hi, size_t rows,
    cudaStream_t stream) {
  size_t total = rows * 128;
  int block = 256;
  size_t grid = (total + block - 1) / block;
  split_rmajor_256_kernel<<<grid, block, 0, stream>>>(in, out_lo, out_hi, rows);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "split_rmajor launch: %s\n", cudaGetErrorString(e));
    std::abort();
  }
}

// Plain C-callable launcher (no torch). Pointers that are null are skipped by
// the kernel (their block count is set to 0). EAL/EBR are (m/n, R) int8 dense;
// EAR/EBL are produced in BOTH R-major (K,R) and K-major (R,K) layouts (the
// noising kernels want EAR_R_major + EBL_K_major for A, EBL_R_major +
// EAR_K_major for B). fp16 outputs are not needed (the miner zeroes the denoise
// factors), so we pass nullptr for them.
extern "C" void pearl_miner_noisegen_sm89_R256(
    int8_t* EAL, int8_t* EBR,
    int8_t* EAR_R_major, int8_t* EAR_K_major,
    int8_t* EBL_R_major, int8_t* EBL_K_major,
    const uint8_t* key_A, const uint8_t* key_B,
    int M, int N, int K, cudaStream_t stream) {
  using namespace cute;
  constexpr int R = 256;
  constexpr int NumThreads = 128;
  using Kernel = pearl::NoiseGenerationKernel<R, NumThreads>;

  bool const gen_EAR = EAR_R_major || EAR_K_major;
  bool const gen_EBL = EBL_R_major || EBL_K_major;

  typename Kernel::Arguments args{
      .ptr_EAL = EAL,
      .ptr_EAL_fp16 = nullptr,
      .ptr_EAR_R_major = EAR_R_major,
      .ptr_EAR_K_major = EAR_K_major,
      .ptr_EBL_R_major = EBL_R_major,
      .ptr_EBL_K_major = EBL_K_major,
      .ptr_EBR = EBR,
      .ptr_EBR_fp16 = nullptr,
      .num_rows_EAL = EAL ? M : 0,
      .length_EAR = gen_EAR ? K : 0,
      .length_EBL = gen_EBL ? K : 0,
      .num_rows_EBR = EBR ? N : 0,
      .ptr_key_A = key_A,
      .ptr_key_B = key_B,
      .ptr_aux_buffer = nullptr,
      .aux_buffer_size = 0};

  typename Kernel::Params params = Kernel::to_underlying_arguments(args);
  dim3 grid = Kernel::get_grid_shape(params);
  dim3 block = Kernel::get_block_shape();
  constexpr int smem_size = Kernel::SharedStorageSize;

  auto kernel = cutlass::device_kernel<Kernel>;
  if (smem_size >= 48 * 1024) {
    cudaError_t e = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (e != cudaSuccess) {
      std::fprintf(stderr, "noisegen cudaFuncSetAttribute(%d): %s\n", smem_size,
                   cudaGetErrorString(e));
      std::abort();
    }
  }
  kernel<<<grid, block, smem_size, stream>>>(params);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "noisegen launch: %s\n", cudaGetErrorString(e));
    std::abort();
  }
}
