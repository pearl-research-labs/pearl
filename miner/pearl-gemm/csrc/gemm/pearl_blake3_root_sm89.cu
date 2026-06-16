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
enum { CHUNK_START = 1, CHUNK_END = 2, PARENT = 4, ROOT = 8, KEYED_HASH = 16 };

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
  // 1024B chunk is 1024-aligned (cudaMalloc) so read message words as aligned
  // little-endian uint32 via the read-only cache (4x fewer loads than byte-wise;
  // GPU is LE so a uint32 load == the byte-assembled word).
  const uint32_t* pw = reinterpret_cast<const uint32_t*>(data + (size_t)c * 1024);
#pragma unroll 1
  for (int b = 0; b < 16; ++b) {
    uint32_t m[16];
#pragma unroll
    for (int w = 0; w < 16; ++w) m[w] = __ldg(pw + b * 16 + w);
    uint32_t flags = KEYED_HASH | (b == 0 ? CHUNK_START : 0) | (b == 15 ? CHUNK_END : 0);
    uint32_t out[8];
    compress(cv, m, (uint64_t)c, 64, flags, out);
#pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = out[i];
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) cvs[(size_t)c * 8 + i] = cv[i];
}
// One thread per parent node: out[i] = parent of (in[2i], in[2i+1]). Internal nodes
// use KEYED_HASH|PARENT and keep the 8-word CV; the single top node (is_top) also sets
// ROOT so out[0..7] (LE) IS the 32-byte merkle root. Bit-identical to the host fold
// blake3_root_from_chunk_cvs (same pairwise order, same compress, ROOT only on top).
__global__ void reduce_level_kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out,
                                    uint32_t half, uint32_t k0, uint32_t k1, uint32_t k2,
                                    uint32_t k3, uint32_t k4, uint32_t k5, uint32_t k6,
                                    uint32_t k7, int is_top) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= half) return;
  uint32_t key[8] = {k0, k1, k2, k3, k4, k5, k6, k7};
  uint32_t m[16];
#pragma unroll
  for (int j = 0; j < 8; ++j) m[j]     = in[(size_t)(2 * i) * 8 + j];      // left child CV
#pragma unroll
  for (int j = 0; j < 8; ++j) m[8 + j] = in[(size_t)(2 * i + 1) * 8 + j];  // right child CV
  uint32_t flags = KEYED_HASH | PARENT | (is_top ? ROOT : 0u);
  uint32_t o[8];
  compress(key, m, 0, 64, flags, o);
#pragma unroll
  for (int j = 0; j < 8; ++j) out[(size_t)i * 8 + j] = o[j];
}
}  // namespace

// On-GPU tree reduce of the per-chunk CVs in `d_cvs` (n_chunks*8 u32, n_chunks a power
// of 2 >= 2) to the 32-byte keyed-BLAKE3 root, written to host `out_root`. Ping-pongs
// between d_cvs (CLOBBERED — caller's per-attempt scratch) and d_scratch (>= n_chunks*4
// u32). Replaces the 16MB D2H + single-threaded host fold: only 32 bytes leave the GPU.
extern "C" void pearl_blake3_root_sm89(uint32_t* d_cvs, uint32_t n_chunks,
                                       const uint8_t key[32], uint32_t* d_scratch,
                                       uint8_t out_root[32], cudaStream_t stream) {
  uint32_t kw[8];
  for (int i = 0; i < 8; ++i)
    kw[i] = (uint32_t)key[4 * i] | ((uint32_t)key[4 * i + 1] << 8) |
            ((uint32_t)key[4 * i + 2] << 16) | ((uint32_t)key[4 * i + 3] << 24);
  uint32_t* in = d_cvs;
  uint32_t* out = d_scratch;
  uint32_t n = n_chunks;
  int t = 256;
  while (n > 2) {
    uint32_t half = n / 2;
    int b = (int)((half + t - 1) / t);
    reduce_level_kernel<<<b, t, 0, stream>>>(in, out, half, kw[0], kw[1], kw[2], kw[3],
                                             kw[4], kw[5], kw[6], kw[7], 0);
    uint32_t* tmp = in; in = out; out = tmp;
    n = half;
  }
  // n == 2: single top node carries the ROOT flag -> out[0..7] = root (LE = 32 bytes).
  reduce_level_kernel<<<1, 1, 0, stream>>>(in, out, 1, kw[0], kw[1], kw[2], kw[3],
                                           kw[4], kw[5], kw[6], kw[7], 1);
  cudaMemcpyAsync(out_root, out, 32, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
}

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
