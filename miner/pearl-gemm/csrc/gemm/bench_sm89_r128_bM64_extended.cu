// Extended bench for R=128 bM=64 bN=128 vs baselines, including:
//   - R=64 bM=128 bN=128 Denoise
//   - R=128 bM=64 bN=64 Denoise (existing R=128 baseline, via PoW trampoline w/ hard target)
//   - R=128 bM=64 bN=128 Noiseless (new)
//   - R=128 bM=64 bN=128 Denoise (new)
// Multiple shapes: 1024^2, 2048^3, 4096^3, 8192^2
// Median + p99 over N iterations.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace pearl {
namespace sm89 {
// Existing baselines (built into pearl_gemm_sm89_denoise_inst.cu + _pow_inst.cu)
extern "C" void pearl_gemm_sm89_denoise_128x128x64_R64(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    cutlass::half_t const*, cutlass::half_t const*,
    cutlass::half_t const*, cutlass::half_t const*,
    int, int, int, cudaStream_t);
extern "C" void pearl_gemm_sm89_pow_64x64x64_R128(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    cutlass::half_t const*, cutlass::half_t const*,
    cutlass::half_t const*, cutlass::half_t const*,
    uint32_t const*, uint32_t const*, void*, void*, uint64_t*,
    int, int, int, cudaStream_t);
// New (this session)
extern "C" void pearl_gemm_sm89_noiseless_64x128x64_R128(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    int, int, int, cudaStream_t);
extern "C" void pearl_gemm_sm89_denoise_64x128x64_R128(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    cutlass::half_t const*, cutlass::half_t const*,
    cutlass::half_t const*, cutlass::half_t const*,
    int, int, int, cudaStream_t);
extern "C" void pearl_gemm_sm89_pow_64x128x64_R128(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    cutlass::half_t const*, cutlass::half_t const*,
    cutlass::half_t const*, cutlass::half_t const*,
    uint32_t const*, uint32_t const*, void*, void*, uint64_t*,
    int, int, int, cudaStream_t);
}
}

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

enum class Variant {
  R64_BN128_DENOISE,
  R128_BN64_POW,      // existing R=128 baseline (with PoW, but hard target = no contention)
  R128_BN128_NOISELESS,
  R128_BN128_DENOISE,
  R128_BN128_POW,
};

static char const* variant_name(Variant v) {
  switch (v) {
    case Variant::R64_BN128_DENOISE:    return "R=64  bM=128 bN=128 Denoise  (prod R=64 baseline)";
    case Variant::R128_BN64_POW:        return "R=128 bM=64  bN=64  PoW(hard) (existing R=128 baseline)";
    case Variant::R128_BN128_NOISELESS: return "R=128 bM=64  bN=128 Noiseless (new)";
    case Variant::R128_BN128_DENOISE:   return "R=128 bM=64  bN=128 Denoise   (new)";
    case Variant::R128_BN128_POW:       return "R=128 bM=64  bN=128 PoW(hard) (new, prod target)";
  }
  return "?";
}

struct PerShapeResults {
  int M, N, K;
  Variant v;
  double median_us;
  double p99_us;
  double tops;
};

// Per-iter timing: run kernel + record event AFTER each call individually.
static double bench(Variant v, int M, int N, int K, int iters,
                    std::vector<double>& per_iter_us) {
  int const R = (v == Variant::R64_BN128_DENOISE) ? 64 : 128;
  std::vector<int8_t> hA(size_t(M)*K), hB(size_t(N)*K);
  std::vector<float>  hAs(M, 0.01f), hBs(N, 0.01f);
  for (auto& v_ : hA) v_ = int8_t((rand() % 255) - 127);
  for (auto& v_ : hB) v_ = int8_t((rand() % 255) - 127);

  int8_t  *dA, *dB;
  float   *dAs, *dBs;
  cutlass::bfloat16_t *dC;
  cutlass::half_t *dEAL, *dEBR, *dAxEBL, *dEARxBpEB;
  CUCHK(cudaMalloc(&dA,  hA.size()));
  CUCHK(cudaMalloc(&dB,  hB.size()));
  CUCHK(cudaMalloc(&dAs, hAs.size()*4));
  CUCHK(cudaMalloc(&dBs, hBs.size()*4));
  CUCHK(cudaMalloc(&dC,  size_t(M)*N*2));
  CUCHK(cudaMalloc(&dEAL,      size_t(M)*R*2));
  CUCHK(cudaMalloc(&dEBR,      size_t(N)*R*2));
  CUCHK(cudaMalloc(&dAxEBL,    size_t(M)*R*2));
  CUCHK(cudaMalloc(&dEARxBpEB, size_t(N)*R*2));
  CUCHK(cudaMemset(dEAL,      0, size_t(M)*R*2));
  CUCHK(cudaMemset(dEBR,      0, size_t(N)*R*2));
  CUCHK(cudaMemset(dAxEBL,    0, size_t(M)*R*2));
  CUCHK(cudaMemset(dEARxBpEB, 0, size_t(N)*R*2));
  CUCHK(cudaMemcpy(dA,  hA.data(),  hA.size(),  cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dB,  hB.data(),  hB.size(),  cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dAs, hAs.data(), hAs.size()*4, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dBs, hBs.data(), hBs.size()*4, cudaMemcpyHostToDevice));

  // PoW state (only used by variants that need it).
  uint32_t *dPowTarget = nullptr, *dPowKey = nullptr;
  uint64_t *dInnerHashCounter = nullptr;
  void *dHostSignalSync = nullptr, *hostSignalHeaderPinned = nullptr;
  std::vector<uint32_t> hPowTarget(8, 0u);  // all-zero target = nothing ever passes (hard target)
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

  auto run_once = [&]() {
    switch (v) {
      case Variant::R64_BN128_DENOISE:
        pearl::sm89::pearl_gemm_sm89_denoise_128x128x64_R64(
            dA, K, dB, K, dC, N, dAs, dBs,
            dEAL, dEBR, dAxEBL, dEARxBpEB, M, N, K, 0);
        break;
      case Variant::R128_BN64_POW:
        pearl::sm89::pearl_gemm_sm89_pow_64x64x64_R128(
            dA, K, dB, K, dC, N, dAs, dBs,
            dEAL, dEBR, dAxEBL, dEARxBpEB,
            dPowTarget, dPowKey, dHostSignalSync, hostSignalHeaderPinned,
            dInnerHashCounter,
            M, N, K, 0);
        break;
      case Variant::R128_BN128_NOISELESS:
        pearl::sm89::pearl_gemm_sm89_noiseless_64x128x64_R128(
            dA, K, dB, K, dC, N, dAs, dBs, M, N, K, 0);
        break;
      case Variant::R128_BN128_DENOISE:
        pearl::sm89::pearl_gemm_sm89_denoise_64x128x64_R128(
            dA, K, dB, K, dC, N, dAs, dBs,
            dEAL, dEBR, dAxEBL, dEARxBpEB, M, N, K, 0);
        break;
      case Variant::R128_BN128_POW:
        pearl::sm89::pearl_gemm_sm89_pow_64x128x64_R128(
            dA, K, dB, K, dC, N, dAs, dBs,
            dEAL, dEBR, dAxEBL, dEARxBpEB,
            dPowTarget, dPowKey, dHostSignalSync, hostSignalHeaderPinned,
            dInnerHashCounter,
            M, N, K, 0);
        break;
    }
  };
  // Warmup
  for (int i = 0; i < 5; ++i) run_once();
  CUCHK(cudaDeviceSynchronize());

  // Per-iter timing.
  per_iter_us.clear();
  per_iter_us.reserve(iters);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int i = 0; i < iters; ++i) {
    cudaEventRecord(e0);
    run_once();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    per_iter_us.push_back(double(ms) * 1000.0);
  }

  cudaFree(dA); cudaFree(dB); cudaFree(dAs); cudaFree(dBs); cudaFree(dC);
  cudaFree(dEAL); cudaFree(dEBR); cudaFree(dAxEBL); cudaFree(dEARxBpEB);
  cudaFree(dPowTarget); cudaFree(dPowKey); cudaFree(dInnerHashCounter); cudaFree(dHostSignalSync);
  cudaFreeHost(hostSignalHeaderPinned);

  std::vector<double> sorted = per_iter_us;
  std::sort(sorted.begin(), sorted.end());
  double median_us = sorted[sorted.size() / 2];
  double p99_us = sorted[(int)(sorted.size() * 0.99)];
  double tops = 2.0 * double(M) * double(N) * double(K) / (median_us * 1e-6) * 1e-12;
  printf("  %-55s  M=%5d N=%5d K=%5d  med=%8.2f us  p99=%8.2f us  %6.2f TOPS\n",
         variant_name(v), M, N, K, median_us, p99_us, tops);

  return tops;
}

int main(int argc, char** argv) {
  int dev = (argc >= 2) ? std::atoi(argv[1]) : 0;
  int iters = (argc >= 3) ? std::atoi(argv[2]) : 50;
  CUCHK(cudaSetDevice(dev));
  cudaDeviceProp p;
  CUCHK(cudaGetDeviceProperties(&p, dev));
  printf("device %d: %s sm_%d%d smem_optin=%zu KB iters/run=%d\n",
         dev, p.name, p.major, p.minor,
         p.sharedMemPerBlockOptin / 1024, iters);
  printf("Reference: 4070 Ti SUPER peak INT8 dense = 353 TOPS\n\n");

  // CSV header to stderr so we can capture cleanly.
  fprintf(stderr, "shape,variant,median_us,p99_us,tops\n");

  // Square shapes + one rectangular at 4096x4096x8192 K-heavy.
  struct Shape { int M, N, K; };
  std::vector<Shape> shapes = {
    {1024, 1024, 1024},
    {2048, 2048, 2048},
    {4096, 4096, 4096},
    {8192, 8192, 8192},
  };

  std::vector<Variant> variants = {
    Variant::R64_BN128_DENOISE,
    Variant::R128_BN64_POW,
    Variant::R128_BN128_NOISELESS,
    Variant::R128_BN128_DENOISE,
    Variant::R128_BN128_POW,
  };

  for (auto const& s : shapes) {
    printf("=== %dx%dx%d ===\n", s.M, s.N, s.K);
    for (auto v : variants) {
      std::vector<double> per_iter_us;
      // Reduce iters for the largest shape to keep runtime reasonable.
      int eff_iters = (s.M >= 8192) ? std::max(10, iters/2) : iters;
      double tops = bench(v, s.M, s.N, s.K, eff_iters, per_iter_us);
      std::sort(per_iter_us.begin(), per_iter_us.end());
      double med = per_iter_us[per_iter_us.size()/2];
      double p99 = per_iter_us[(int)(per_iter_us.size()*0.99)];
      // CSV row
      char shape_str[32];
      snprintf(shape_str, sizeof(shape_str), "%dx%dx%d", s.M, s.N, s.K);
      char variant_str[64];
      snprintf(variant_str, sizeof(variant_str), "%s", variant_name(v));
      // Strip spaces/commas for CSV
      for (char* c = variant_str; *c; ++c) if (*c==',') *c=';';
      fprintf(stderr, "%s,\"%s\",%.2f,%.2f,%.2f\n", shape_str, variant_str, med, p99, tops);
    }
    printf("\n");
  }
  return 0;
}
