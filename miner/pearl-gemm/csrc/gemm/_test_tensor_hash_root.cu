// Standalone bit-exactness + compile test for the on-device merkle-root path.
// Validates (1) tensor_hash_host.hpp compiles WITHOUT torch, and (2) the device
// keyed merkle root == pearl_mining.MerkleTree(pad(A), key).root, so it can drive
// the real-commitment mining seeds. Build sm_120 (local 5090) or sm_89 (rig).
//   nvcc -gencode arch=compute_120,code=sm_120 -std=c++20 -O3 -I . -I .. \
//     -I ../../third_party/cutlass/include -I ../../third_party/cutlass/tools/util/include \
//     -I ../../third_party/cutlass/examples/common --expt-relaxed-constexpr \
//     --expt-extended-lambda -DNDEBUG -DPEARL_GEMM_BUILD_SM120 _test_tensor_hash_root.cu -o /tmp/tt
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

// tensor_hash_host.hpp uses TORCH_CHECK only on its error path; shim it so the
// standalone (no-torch) build compiles. Valid params never trigger it.
#ifndef TORCH_CHECK
#define TORCH_CHECK(cond, ...)                                                  \
  do {                                                                          \
    if (!(cond)) {                                                              \
      fprintf(stderr, "TORCH_CHECK failed %s:%d\n", __FILE__, __LINE__);        \
      std::exit(3);                                                             \
    }                                                                           \
  } while (0)
#endif

#include "tensor_hash/tensor_hash_host.hpp"

// splitmix64 int8 fill, bit-exact with _splitmix64_fill (run_canary.py) and
// fill_AB (pearl_miner_sm89.cu): map (z % 127) - 63 into [-63, 63].
static void fill(std::vector<int8_t>& v, uint64_t seed) {
  uint64_t s = seed;
  for (auto& x : v) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    x = (int8_t)((int)(z % 127) - 63);
  }
}

int main() {
  const int M = 256, K = 4096;          // 1 MiB -> 1024 chunks -> exercises the
  const size_t n = (size_t)M * K;       // multi-block ReduceRootsKernel path.
  std::vector<int8_t> hA(n);
  fill(hA, 12345);
  { FILE* f = fopen("/tmp/ha.bin", "wb"); fwrite(hA.data(), 1, n, f); fclose(f); }

  uint8_t key[32];
  for (int i = 0; i < 32; ++i) key[i] = (uint8_t)(i * 7 + 1);

  uint8_t *dA = 0, *dKey = 0, *dRoot = 0, *dScratch = 0;
  if (cudaMalloc(&dA, n) != cudaSuccess) { printf("malloc fail\n"); return 2; }
  cudaMemcpy(dA, hA.data(), n, cudaMemcpyHostToDevice);
  cudaMalloc(&dKey, 32);
  cudaMemcpy(dKey, key, 32, cudaMemcpyHostToDevice);
  cudaMalloc(&dRoot, 32);

  const size_t tpb = 128, bytes_per_block = tpb * 1024;
  const size_t req_blocks = (n + bytes_per_block - 1) / bytes_per_block;
  cudaMalloc(&dScratch, req_blocks * 32);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const uint32_t num_blocks = (uint32_t)(n / 1024);

  tensor_hash(dA, (uint32_t)n, dRoot, dKey, num_blocks,
              /*threads_per_block=*/128, /*num_stages=*/2,
              /*leaves_per_mt_block=*/512, dScratch, prop, 0);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    printf("kernel fail: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 2;
  }
  uint8_t root[32];
  cudaMemcpy(root, dRoot, 32, cudaMemcpyDeviceToHost);
  printf("ROOT ");
  for (int i = 0; i < 32; ++i) printf("%02x", root[i]);
  printf("\nKEY  ");
  for (int i = 0; i < 32; ++i) printf("%02x", key[i]);
  printf("\nSEED 12345 M %d K %d\n", M, K);
  return 0;
}
