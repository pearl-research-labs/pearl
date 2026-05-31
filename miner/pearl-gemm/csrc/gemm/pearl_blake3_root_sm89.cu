// On-GPU keyed-BLAKE3 chunk chaining values for the matrix merkle root.
// Each thread hashes one 1024-byte chunk (16 blocks chained, counter=chunk index)
// and writes its 32-byte output CV. The cheap tree reduce of the CVs runs host-side
// (blake3_root_from_chunk_cvs) on only the 16 MB CV array — so the 512 MB matrix
// never leaves VRAM. Device compress is a byte-for-byte port of the validated
// blake3_tree_host compress (== pearl_mining keyed BLAKE3).
#include <cstdint>
#include <cuda_runtime.h>

namespace {
__constant__ uint32_t d_IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                                 0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
__constant__ uint8_t d_MSG[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13}};
enum { CHUNK_START = 1, CHUNK_END = 2, KEYED_HASH = 16 };

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ void g(uint32_t* s, int a, int b, int c, int d, uint32_t x, uint32_t y) {
  s[a] = s[a] + s[b] + x; s[d] = rotr(s[d] ^ s[a], 16);
  s[c] = s[c] + s[d];     s[b] = rotr(s[b] ^ s[c], 12);
  s[a] = s[a] + s[b] + y; s[d] = rotr(s[d] ^ s[a], 8);
  s[c] = s[c] + s[d];     s[b] = rotr(s[b] ^ s[c], 7);
}
__device__ void compress(const uint32_t cv[8], const uint32_t m[16], uint64_t counter,
                         uint32_t block_len, uint32_t flags, uint32_t out[8]) {
  uint32_t s[16] = {cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
                    d_IV[0], d_IV[1], d_IV[2], d_IV[3],
                    (uint32_t)counter, (uint32_t)(counter >> 32), block_len, flags};
  uint32_t mm[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) mm[i] = m[i];
#pragma unroll
  for (int r = 0; r < 7; ++r) {
    const uint8_t* z = d_MSG[r];
    g(s, 0, 4, 8, 12, mm[z[0]], mm[z[1]]);  g(s, 1, 5, 9, 13, mm[z[2]], mm[z[3]]);
    g(s, 2, 6, 10, 14, mm[z[4]], mm[z[5]]); g(s, 3, 7, 11, 15, mm[z[6]], mm[z[7]]);
    g(s, 0, 5, 10, 15, mm[z[8]], mm[z[9]]); g(s, 1, 6, 11, 12, mm[z[10]], mm[z[11]]);
    g(s, 2, 7, 8, 13, mm[z[12]], mm[z[13]]); g(s, 3, 4, 9, 14, mm[z[14]], mm[z[15]]);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = s[i] ^ s[i + 8];
}

// One thread per 1024-byte chunk. data is row-major int8 matrix bytes (n_chunks*1024).
__global__ void chunk_cvs_kernel(const uint8_t* __restrict__ data, uint32_t n_chunks,
                                 uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3,
                                 uint32_t k4, uint32_t k5, uint32_t k6, uint32_t k7,
                                 uint32_t* __restrict__ cvs) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_chunks) return;
  uint32_t cv[8] = {k0, k1, k2, k3, k4, k5, k6, k7};
  const uint8_t* p = data + (size_t)c * 1024;
#pragma unroll 1
  for (int b = 0; b < 16; ++b) {
    uint32_t m[16];
#pragma unroll
    for (int w = 0; w < 16; ++w) {
      const uint8_t* q = p + b * 64 + w * 4;
      m[w] = (uint32_t)q[0] | ((uint32_t)q[1] << 8) | ((uint32_t)q[2] << 16) | ((uint32_t)q[3] << 24);
    }
    uint32_t flags = KEYED_HASH | (b == 0 ? CHUNK_START : 0) | (b == 15 ? CHUNK_END : 0);
    uint32_t out[8];
    compress(cv, m, (uint64_t)c, 64, flags, out);
#pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = out[i];
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) cvs[(size_t)c * 8 + i] = cv[i];
}
}  // namespace

// Compute per-chunk output CVs of `d_data` (n_bytes, multiple of 1024) keyed by
// `key` (host 32B), into d_cvs (device, n_chunks*32 bytes). Caller reduces to root.
extern "C" void pearl_blake3_chunk_cvs_sm89(const uint8_t* d_data, size_t n_bytes,
                                            const uint8_t key[32], uint32_t* d_cvs,
                                            cudaStream_t stream) {
  uint32_t kw[8];
  for (int i = 0; i < 8; ++i)
    kw[i] = (uint32_t)key[4 * i] | ((uint32_t)key[4 * i + 1] << 8) |
            ((uint32_t)key[4 * i + 2] << 16) | ((uint32_t)key[4 * i + 3] << 24);
  uint32_t n_chunks = (uint32_t)(n_bytes / 1024);
  int threads = 256;
  int blocks = (int)((n_chunks + threads - 1) / threads);
  chunk_cvs_kernel<<<blocks, threads, 0, stream>>>(
      d_data, n_chunks, kw[0], kw[1], kw[2], kw[3], kw[4], kw[5], kw[6], kw[7], d_cvs);
}
