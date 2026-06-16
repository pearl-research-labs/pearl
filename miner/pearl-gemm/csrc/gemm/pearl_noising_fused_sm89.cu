// SPDX-License-Identifier: see LICENSE
//
// pearl_noising_fused_sm89 — single-pass R=256 noising (2026-06-12).
//
//   out[m*K + k] = (int8)( X[m*K + k] + Σ_{r<256} D[m*256 + r] · S[k*256 + r] )
//
// Replaces, per operand, the legacy chain of:
//   2× pearl_noisingA/B 64x128x64_R128 passes (int32 SIMT, wrap after each
//   half — exact because mod-256 addition distributes over the halves),
//   4× pearl_miner_split_rmajor_256 repacks (the R128 kernels hard-code
//   ld=128 and could not consume the (rows,256) noisegen output directly),
//   and the dead AxEBL/EARxBpEB int32 "denoise scratch" outputs that mining
//   never reads.
// This kernel consumes the noisegen R-major outputs (D = EAL/EBR (rows,256),
// S = EAR_R/EBL_R (K,256)) DIRECTLY, computes the full 256-dot in int32 via
// dp4a from shared memory, adds the raw operand, and truncates once
// ((int8_t) of the int32 sum == mod-256 wrap == pearl_miner::i8wrap; C++20
// makes signed narrowing modular, and a single wrap of the full sum equals
// the chain's wrap(wrap(lo)+hi) mod 256).
//
// Shapes are the fixed mining shape only: rows ∈ {131072}, K=4096, R=256 —
// the launcher asserts rows%64==0 && K%64==0.
//
// Tile: 64 rows × 64 k per CTA, 256 threads, 16 outputs/thread (4×4).
// smem: D-tile 64×256 + S-tile 64×256 int8 = 32 KB. Inner loop: 64 iterations
// of 8 smem int loads + 16 dp4a. ~34.4 G-dp4a per (131072,4096) operand.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static __global__ void pearl_noising_fused_r256_kernel(
    const int8_t* __restrict__ X, const int8_t* __restrict__ D,
    const int8_t* __restrict__ S, int8_t* __restrict__ out, int K) {
  // +16B row pad: rows stay int4-aligned (272 % 16 == 0) while consecutive
  // rows shift 4 banks, breaking the 16-way conflict the unpadded 256B stride
  // produces on the s-row loads (tk strides of 4 rows landed on one bank).
  __shared__ int8_t sD[64][272];   // 17 KB: D rows  [m0 .. m0+64)
  __shared__ int8_t sS[64][272];   // 17 KB: S rows  [k0 .. k0+64)
  const int m0 = blockIdx.y * 64;
  const int k0 = blockIdx.x * 64;

  // Cooperative tile load: 64 rows × 16 int4 segments per tile, 256 threads
  // -> 4 (row,segment) pairs per thread per tile. Global rows are 256B apart
  // (unpadded); smem rows are 272B apart, both 16B-aligned.
  {
    const int4* gD = reinterpret_cast<const int4*>(D + (size_t)m0 * 256);
    const int4* gS = reinterpret_cast<const int4*>(S + (size_t)k0 * 256);
    #pragma unroll
    for (int i = threadIdx.x; i < 64 * 16; i += 256) {
      const int row = i >> 4, seg = i & 15;
      reinterpret_cast<int4*>(&sD[row][0])[seg] = gD[row * 16 + seg];
      reinterpret_cast<int4*>(&sS[row][0])[seg] = gS[row * 16 + seg];
    }
  }
  __syncthreads();

  // 4×4 micro-tile per thread: rows tm..tm+3, cols tk..tk+3 (within the tile).
  const int tm = (threadIdx.x / 16) * 4;
  const int tk = (threadIdx.x % 16) * 4;
  const int* dRow[4]; const int* sRow[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    dRow[i] = reinterpret_cast<const int*>(&sD[tm + i][0]);  // 64 ints = 256 int8
    sRow[i] = reinterpret_cast<const int*>(&sS[tk + i][0]);
  }
  int acc[4][4] = {};
  #pragma unroll 4
  for (int r4 = 0; r4 < 64; ++r4) {                 // 4 r-lanes per dp4a
    int d[4], s[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) { d[i] = dRow[i][r4]; s[i] = sRow[i][r4]; }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
      #pragma unroll
      for (int j = 0; j < 4; ++j)
        acc[i][j] = __dp4a(d[i], s[j], acc[i][j]);
  }

  // Add the raw operand and truncate once (mod-256 wrap), 4 bytes per store.
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const size_t row = (size_t)(m0 + tm + i) * K + (k0 + tk);
    uint32_t xin;  // 4 raw int8 (k .. k+3), 4B-aligned (k0+tk is a multiple of 4)
    xin = *reinterpret_cast<const uint32_t*>(X + row);
    uint32_t packed = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      int32_t v = acc[i][j] + (int8_t)((xin >> (8 * j)) & 0xFF);
      packed |= ((uint32_t)v & 0xFF) << (8 * j);
    }
    *reinterpret_cast<uint32_t*>(out + row) = packed;
  }
}

extern "C" void pearl_noising_fused_r256(
    const int8_t* X, const int8_t* D, const int8_t* S, int8_t* out,
    int rows, int K, cudaStream_t stream) {
  if (rows % 64 != 0 || K % 64 != 0) {
    fprintf(stderr, "pearl_noising_fused_r256: rows/K must be multiples of 64 (got %d,%d)\n",
            rows, K);
    std::exit(2);
  }
  dim3 grid(K / 64, rows / 64);
  pearl_noising_fused_r256_kernel<<<grid, 256, 0, stream>>>(X, D, S, out, K);
}
