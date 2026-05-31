// Bit-exact validation: R=128 bM=64 bN=128 vs R=128 bM=64 bN=64.
// Tests across 10 random seeds at multiple shapes.
//
// For a non-PoW kernel, the GEMM result C[i][j] = round(sum_k A[i][k]*B[j][k]) * Ascale[i] * Bscale[j]
// is independent of tile size — output should be bit-exact under same inputs.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cutlass/numeric_types.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>

namespace pearl {
namespace sm89 {
extern "C" void pearl_gemm_sm89_pow_64x64x64_R128(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    cutlass::half_t const*, cutlass::half_t const*,
    cutlass::half_t const*, cutlass::half_t const*,
    uint32_t const*, uint32_t const*, void*, void*, uint64_t*,
    int, int, int, cudaStream_t);
extern "C" void pearl_gemm_sm89_pow_64x128x64_R128(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    cutlass::half_t const*, cutlass::half_t const*,
    cutlass::half_t const*, cutlass::half_t const*,
    uint32_t const*, uint32_t const*, void*, void*, uint64_t*,
    int, int, int, cudaStream_t);
extern "C" void pearl_gemm_sm89_denoise_64x128x64_R128(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    cutlass::half_t const*, cutlass::half_t const*,
    cutlass::half_t const*, cutlass::half_t const*,
    int, int, int, cudaStream_t);
}
}

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

static bool validate_pair(int M, int N, int K, unsigned seed) {
  srand(seed);
  int const R = 128;
  std::vector<int8_t> hA(size_t(M)*K), hB(size_t(N)*K);
  std::vector<float>  hAs(M), hBs(N);
  std::vector<cutlass::half_t> hEAL(size_t(M)*R), hEBR(size_t(N)*R);
  std::vector<cutlass::half_t> hAxEBL(size_t(M)*R), hEARxBpEB(size_t(N)*R);
  for (auto& v : hA) v = int8_t((rand() % 255) - 127);
  for (auto& v : hB) v = int8_t((rand() % 255) - 127);
  for (auto& v : hAs) v = 0.01f + 0.0001f * (rand() % 100);
  for (auto& v : hBs) v = 0.01f + 0.0001f * (rand() % 100);
  // Half scales — keep small to avoid overflow in BF16 output.
  for (auto& v : hEAL) v = cutlass::half_t(float(rand() % 100) * 0.001f);
  for (auto& v : hEBR) v = cutlass::half_t(float(rand() % 100) * 0.001f);
  for (auto& v : hAxEBL) v = cutlass::half_t(float(rand() % 100) * 0.001f);
  for (auto& v : hEARxBpEB) v = cutlass::half_t(float(rand() % 100) * 0.001f);

  int8_t  *dA, *dB;
  float   *dAs, *dBs;
  cutlass::bfloat16_t *dC_ref, *dC_new;
  cutlass::half_t *dEAL, *dEBR, *dAxEBL, *dEARxBpEB;
  CUCHK(cudaMalloc(&dA,  hA.size()));
  CUCHK(cudaMalloc(&dB,  hB.size()));
  CUCHK(cudaMalloc(&dAs, hAs.size()*4));
  CUCHK(cudaMalloc(&dBs, hBs.size()*4));
  CUCHK(cudaMalloc(&dC_ref,  size_t(M)*N*2));
  CUCHK(cudaMalloc(&dC_new,  size_t(M)*N*2));
  CUCHK(cudaMalloc(&dEAL,      hEAL.size()*2));
  CUCHK(cudaMalloc(&dEBR,      hEBR.size()*2));
  CUCHK(cudaMalloc(&dAxEBL,    hAxEBL.size()*2));
  CUCHK(cudaMalloc(&dEARxBpEB, hEARxBpEB.size()*2));
  CUCHK(cudaMemcpy(dA,  hA.data(),  hA.size(),  cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dB,  hB.data(),  hB.size(),  cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dAs, hAs.data(), hAs.size()*4, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dBs, hBs.data(), hBs.size()*4, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dEAL,      hEAL.data(),      hEAL.size()*2, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dEBR,      hEBR.data(),      hEBR.size()*2, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dAxEBL,    hAxEBL.data(),    hAxEBL.size()*2, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dEARxBpEB, hEARxBpEB.data(), hEARxBpEB.size()*2, cudaMemcpyHostToDevice));
  CUCHK(cudaMemset(dC_ref, 0, size_t(M)*N*2));
  CUCHK(cudaMemset(dC_new, 0, size_t(M)*N*2));

  // PoW state (impossible target — never passes; isolates the GEMM compute).
  uint32_t *dPowTarget, *dPowKey;
  uint64_t *dInnerHashCounter;
  void *dHostSignalSync, *hostSignalHeaderPinned;
  std::vector<uint32_t> hPowTarget(8, 0u);
  std::vector<uint32_t> hPowKey(8, 0u);
  CUCHK(cudaMalloc(&dPowTarget, 32));
  CUCHK(cudaMalloc(&dPowKey,    32));
  CUCHK(cudaMemcpy(dPowTarget, hPowTarget.data(), 32, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dPowKey,    hPowKey.data(),    32, cudaMemcpyHostToDevice));
  CUCHK(cudaMalloc(&dInnerHashCounter, 8));
  CUCHK(cudaMemset(dInnerHashCounter, 0, 8));
  CUCHK(cudaMalloc(&dHostSignalSync, 64));
  CUCHK(cudaMemset(dHostSignalSync, 0, 64));
  CUCHK(cudaMallocHost(&hostSignalHeaderPinned, 256));
  memset(hostSignalHeaderPinned, 0, 256);

  // Run reference (R=128 bM=64 bN=64)
  pearl::sm89::pearl_gemm_sm89_pow_64x64x64_R128(
      dA, K, dB, K, dC_ref, N, dAs, dBs,
      dEAL, dEBR, dAxEBL, dEARxBpEB,
      dPowTarget, dPowKey, dHostSignalSync, hostSignalHeaderPinned,
      dInnerHashCounter,
      M, N, K, 0);
  CUCHK(cudaDeviceSynchronize());

  CUCHK(cudaMemset(dInnerHashCounter, 0, 8));
  CUCHK(cudaMemset(dHostSignalSync, 0, 64));

  // Run new (R=128 bM=64 bN=128)
  pearl::sm89::pearl_gemm_sm89_pow_64x128x64_R128(
      dA, K, dB, K, dC_new, N, dAs, dBs,
      dEAL, dEBR, dAxEBL, dEARxBpEB,
      dPowTarget, dPowKey, dHostSignalSync, hostSignalHeaderPinned,
      dInnerHashCounter,
      M, N, K, 0);
  CUCHK(cudaDeviceSynchronize());

  std::vector<cutlass::bfloat16_t> hC_ref(size_t(M)*N), hC_new(size_t(M)*N);
  CUCHK(cudaMemcpy(hC_ref.data(), dC_ref, hC_ref.size()*2, cudaMemcpyDeviceToHost));
  CUCHK(cudaMemcpy(hC_new.data(), dC_new, hC_new.size()*2, cudaMemcpyDeviceToHost));

  // Compare bit-exact.
  int diffs = 0;
  float max_abs_diff = 0.0f;
  int first_diff_idx = -1;
  for (size_t i = 0; i < hC_ref.size(); ++i) {
    if (memcmp(&hC_ref[i], &hC_new[i], 2) != 0) {
      diffs++;
      if (first_diff_idx < 0) first_diff_idx = int(i);
      float ref = float(hC_ref[i]);
      float new_ = float(hC_new[i]);
      float d = std::abs(ref - new_);
      if (d > max_abs_diff) max_abs_diff = d;
    }
  }

  cudaFree(dA); cudaFree(dB); cudaFree(dAs); cudaFree(dBs);
  cudaFree(dC_ref); cudaFree(dC_new);
  cudaFree(dEAL); cudaFree(dEBR); cudaFree(dAxEBL); cudaFree(dEARxBpEB);
  cudaFree(dPowTarget); cudaFree(dPowKey); cudaFree(dInnerHashCounter); cudaFree(dHostSignalSync);
  cudaFreeHost(hostSignalHeaderPinned);

  if (diffs > 0) {
    float ref0 = float(hC_ref[first_diff_idx]);
    float new0 = float(hC_new[first_diff_idx]);
    printf("  %4dx%4dx%4d seed=%u: DIFFS=%d/%zu (%.4f%%) max_abs=%.4f first@%d ref=%.4f new=%.4f\n",
           M, N, K, seed, diffs, hC_ref.size(),
           100.0 * diffs / hC_ref.size(), max_abs_diff, first_diff_idx, ref0, new0);
    return false;
  } else {
    printf("  %4dx%4dx%4d seed=%u: OK bit-exact\n", M, N, K, seed);
    return true;
  }
}

int main(int argc, char** argv) {
  int dev = (argc >= 2) ? std::atoi(argv[1]) : 0;
  CUCHK(cudaSetDevice(dev));
  cudaDeviceProp p;
  CUCHK(cudaGetDeviceProperties(&p, dev));
  printf("device %d: %s sm_%d%d\n", dev, p.name, p.major, p.minor);
  printf("Validation: R=128 bM=64 bN=128 PoW vs R=128 bM=64 bN=64 PoW (10 seeds)\n\n");

  int const shapes[][3] = {
    {512, 512, 512},
    {1024, 1024, 1024},
    {2048, 2048, 2048},
  };

  int total = 0, pass = 0;
  for (auto const& s : shapes) {
    for (int seed = 1; seed <= 10; ++seed) {
      total++;
      if (validate_pair(s[0], s[1], s[2], unsigned(seed * 31337))) pass++;
    }
    printf("\n");
  }
  printf("Total: %d/%d bit-exact\n", pass, total);
  return (pass == total) ? 0 : 2;
}
