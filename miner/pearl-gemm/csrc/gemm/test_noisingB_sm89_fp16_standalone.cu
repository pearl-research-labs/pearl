// SPDX-License-Identifier: see LICENSE
//
// Standalone correctness smoke test for the sm_89 fp16 noisingB path
// (EARxBpEB_type = cutlass::half_t, R = 64). Exercises the
// run_pearl_noising_B_sm89<cutlass::half_t, ..., true> instantiation added in
// pearl_noisingB_sm89_inst.cu. The fp16 path reuses the same kernel as int32
// except the final epilogue divides by kEARxBpEBScaleFactor (= 2^12) and
// downcasts to fp16 via R2S + S2G of a fp16-typed smem buffer (the union's
// epilogue arm).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cute/tensor.hpp"
#include "pearl_noisingB_sm89_host.h"
#include "pearl_gemm_constants.hpp"
#include <cutlass/numeric_types.h>

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

#ifndef R_DIM_OVERRIDE
#define R_DIM_OVERRIDE 64
#endif
static constexpr int R_DIM = R_DIM_OVERRIDE;
static constexpr int kBN_TILE = 64;
static constexpr int kBK_TILE = 64;

// CPU reference, fp16 epilogue version:
//   BpEB[n,k] = int8(B[n,k] + sum_r EBR[n,r] * EBL[k,r])
//   EARxBpEB_int32 = sum_k BpEB[n,k] * EAR[r*K + k]
//   EARxBpEB_fp16  = float(EARxBpEB_int32) / 2^12 downcast to fp16
static void ref_BpEB(int N, int K, int8_t const* B, int8_t const* EBR,
                     int8_t const* EBL, int8_t* BpEB) {
  for (int n = 0; n < N; ++n) {
    for (int kk = 0; kk < K; ++kk) {
      int32_t eb = 0;
      for (int r = 0; r < R_DIM; ++r) {
        eb += int32_t(EBR[n * R_DIM + r]) * int32_t(EBL[kk * R_DIM + r]);
      }
      int32_t sum = int32_t(B[n * K + kk]) + eb;
      BpEB[n * K + kk] = int8_t(sum & 0xff);
    }
  }
}

static void ref_EARxBpEB_fp16(int N, int K, int8_t const* BpEB,
                              int8_t const* EAR, __half* EARxBpEB_fp16) {
  constexpr float kScale = float(pearl::kEARxBpEBScaleFactor);  // 4096
  for (int n = 0; n < N; ++n) {
    for (int r = 0; r < R_DIM; ++r) {
      int32_t acc = 0;
      for (int kk = 0; kk < K; ++kk) {
        acc += int32_t(BpEB[n * K + kk]) * int32_t(EAR[r * K + kk]);
      }
      EARxBpEB_fp16[n * R_DIM + r] = __float2half(float(acc) / kScale);
    }
  }
}

static int run_case(int N, int K, unsigned seed) {
  using TileShape_NRK = cute::Shape<cute::Int<kBN_TILE>, cute::Int<R_DIM>,
                                    cute::Int<kBK_TILE>>;
  if (N % kBN_TILE != 0 || K % kBK_TILE != 0) {
    fprintf(stderr, "N=%d K=%d not multiples of tile=64\n", N, K);
    return 1;
  }
  std::srand(seed);

  std::vector<int8_t> hB(size_t(N) * K), hEBR(size_t(N) * R_DIM),
      hEBL(size_t(K) * R_DIM), hEAR(size_t(R_DIM) * K);
  std::vector<int8_t> hBpEB(size_t(N) * K), hBpEB_ref(size_t(N) * K);
  std::vector<__half> hEARxBpEB(size_t(N) * R_DIM),
      hEARxBpEB_ref(size_t(N) * R_DIM);

  auto rand_int7  = [](){ return int8_t((std::rand() % 127) - 64); };
  auto rand_int6  = [](){ return int8_t((std::rand() % 63)  - 32); };
  for (auto& v : hB)   v = rand_int7();
  for (auto& v : hEBR) v = rand_int6();
  std::fill(hEBL.begin(), hEBL.end(), int8_t(0));
  for (int k = 0; k < K; ++k) {
    int p0 = std::rand() % R_DIM;
    int p1 = std::rand() % R_DIM;
    hEBL[k * R_DIM + p0] = int8_t(hEBL[k * R_DIM + p0] + 1);
    hEBL[k * R_DIM + p1] = int8_t(hEBL[k * R_DIM + p1] - 1);
  }
  std::fill(hEAR.begin(), hEAR.end(), int8_t(0));
  // EAR shape (R, K) K-major; one ±1 per "row" of length K won't help here;
  // we'll fill EAR randomly with int4 to get a non-trivial matmul.
  for (auto& v : hEAR) v = int8_t((std::rand() % 7) - 3);

  int8_t  *dB, *dEBR, *dEBL, *dEAR, *dBpEB;
  __half  *dEARxBpEB_fp16;
  CUCHK(cudaMalloc(&dB,    hB.size()));
  CUCHK(cudaMalloc(&dEBR,  hEBR.size()));
  CUCHK(cudaMalloc(&dEBL,  hEBL.size()));
  CUCHK(cudaMalloc(&dEAR,  hEAR.size()));
  CUCHK(cudaMalloc(&dBpEB, hBpEB.size()));
  CUCHK(cudaMalloc(&dEARxBpEB_fp16, hEARxBpEB.size() * sizeof(__half)));

  CUCHK(cudaMemcpy(dB,   hB.data(),   hB.size(),   cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dEBR, hEBR.data(), hEBR.size(), cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dEBL, hEBL.data(), hEBL.size(), cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dEAR, hEAR.data(), hEAR.size(), cudaMemcpyHostToDevice));
  CUCHK(cudaMemset(dBpEB, 0xAA, hBpEB.size()));
  CUCHK(cudaMemset(dEARxBpEB_fp16, 0xAA, hEARxBpEB.size() * sizeof(__half)));

  // Direct template invocation (the auto-gen path uses run_pearl_noising_B_
  // which is in pearl_gemm_launch_template.h; here we skip the launcher
  // layer and call the sm_89-specific function in pearl::sm89 namespace).
  pearl::sm89::run_pearl_noising_B_sm89<cutlass::half_t, TileShape_NRK,
                                        /*kStages=*/2, /*IsEvenK=*/true>(
      dB, dEBR, dEBL, dEAR, dBpEB,
      reinterpret_cast<cutlass::half_t*>(dEARxBpEB_fp16),
      N, K, /*k_blocks_per_split=*/-1, /*stream=*/0);
  CUCHK(cudaDeviceSynchronize());
  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    fprintf(stderr, "kernel launch error: %s\n", cudaGetErrorString(launch_err));
    return 1;
  }

  CUCHK(cudaMemcpy(hBpEB.data(),     dBpEB,
                   hBpEB.size(), cudaMemcpyDeviceToHost));
  CUCHK(cudaMemcpy(hEARxBpEB.data(), dEARxBpEB_fp16,
                   hEARxBpEB.size() * sizeof(__half),
                   cudaMemcpyDeviceToHost));

  ref_BpEB(N, K, hB.data(), hEBR.data(), hEBL.data(), hBpEB_ref.data());
  ref_EARxBpEB_fp16(N, K, hBpEB_ref.data(), hEAR.data(), hEARxBpEB_ref.data());

  long bad_bpeb = 0;
  for (size_t i = 0; i < hBpEB.size(); ++i)
    if (hBpEB[i] != hBpEB_ref[i]) ++bad_bpeb;

  long bad_earxbpeb = 0;
  float worst_diff = 0.0f;
  for (size_t i = 0; i < hEARxBpEB.size(); ++i) {
    float g = __half2float(hEARxBpEB[i]);
    float r = __half2float(hEARxBpEB_ref[i]);
    float d = std::fabs(g - r);
    if (d > worst_diff) worst_diff = d;
    // Allow 4 ULPs relative tolerance (fp16 rounding-mode mismatch tolerance).
    float tol = std::max(2e-3f, 4e-3f * std::fabs(r));
    if (d > tol) ++bad_earxbpeb;
  }

  long bad = bad_bpeb + bad_earxbpeb;
  printf("N=%d K=%d R=%d seed=%u  bad_bpeb=%ld/%zu  bad_earxbpeb=%ld/%zu  worst_diff=%.6f",
         N, K, R_DIM, seed, bad_bpeb, hBpEB.size(), bad_earxbpeb,
         hEARxBpEB.size(), worst_diff);
  if (bad == 0) printf("   PASS\n");
  else          printf("   FAIL\n");

  CUCHK(cudaFree(dB));
  CUCHK(cudaFree(dEBR));
  CUCHK(cudaFree(dEBL));
  CUCHK(cudaFree(dEAR));
  CUCHK(cudaFree(dBpEB));
  CUCHK(cudaFree(dEARxBpEB_fp16));
  return bad == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  int dev = (argc >= 2) ? std::atoi(argv[1]) : 0;
  CUCHK(cudaSetDevice(dev));
  cudaDeviceProp p;
  CUCHK(cudaGetDeviceProperties(&p, dev));
  int cc = p.major * 10 + p.minor;
  printf("device %d: %s sm_%d (smem/SM optin: %d KB)\n",
         dev, p.name, cc, int(p.sharedMemPerBlockOptin / 1024));

  int rc = 0;
  rc |= run_case(128, 512, 0);
  rc |= run_case( 64,  64, 1);
  rc |= run_case(256, 1024, 2);
  rc |= run_case(512, 512, 3);
  rc |= run_case(1024, 1024, 4);
  printf("\n%s\n", rc == 0 ? "ALL PASS" : "FAIL");
  return rc;
}
