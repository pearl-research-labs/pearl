// SPDX-License-Identifier: see LICENSE
//
// Self-contained correctness test for pearl_noisingB_sm89_64x64x64_R64_int32.
// Compares kernel output against CPU int32 references for both:
//   - BpEB     = B + (EBR @ EBL.t())                    (cast to int8)
//   - EARxBpEB = BpEB @ EAR                             (int32)
//
// Build (in WSL):
//   nvcc -gencode arch=compute_89,code=sm_89 -std=c++20 -O3
//        -I . -I .. -I ../../third_party/cutlass/include
//        -I ../../third_party/cutlass/tools/util/include
//        -I ../../third_party/cutlass/examples/common
//        --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG
//        pearl_noisingB_sm89_inst.cu test_noisingB_sm89_standalone.cu
//        -lcudart -o /tmp/test_noisingB_sm89

#include <cuda_runtime.h>

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace pearl {
namespace sm89 {
extern "C" void pearl_noisingB_sm89_64x64x64_R64_int32(
    int8_t const* B, int8_t const* EBR, int8_t const* EBL, int8_t const* EAR,
    int8_t* BpEB, int32_t* EARxBpEB, int N, int K, cudaStream_t stream);
}
}

#define CUCHK(x)                                                              \
  do {                                                                        \
    auto _e = (x);                                                            \
    if (_e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__,                  \
              cudaGetErrorString(_e));                                        \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

static constexpr int R = 64;

// Reference: EB[n,k] = sum_r EBR[n,r] * EBL[k,r] (both R-major).
//            BpEB[n,k] = B[n,k] + EB[n,k] (int8 cast).
static void ref_BpEB(int N, int K, int8_t const* B, int8_t const* EBR,
                     int8_t const* EBL, int8_t* BpEB) {
  for (int n = 0; n < N; ++n) {
    for (int kk = 0; kk < K; ++kk) {
      int32_t eb = 0;
      for (int r = 0; r < R; ++r) {
        eb += int32_t(EBR[n * R + r]) * int32_t(EBL[kk * R + r]);
      }
      int32_t sum = int32_t(B[n * K + kk]) + eb;
      // int8 cast = wrap to int8 (matches torch.int8 cast semantics: low 8 bits).
      BpEB[n * K + kk] = int8_t(sum & 0xff);
    }
  }
}

// Reference: EARxBpEB[n,r] = sum_k BpEB[n,k] * EAR[k,r] (EAR is K-major,
//            so EAR[k,r] = stride r*K + k? No — the API states EAR is K-major
//            with shape (R, k) and stride (K, 1), but compute_ref_noise_B
//            uses `torch._int_mm(BpEB_ref_for_matmul, EAR_ref)` where EAR_ref
//            is `tensor_generator.EAR_R_major`. torch._int_mm computes
//            (M,K) @ (K,N): BpEB (n,k) @ EAR_R_major (k, r) -> (n, r). So
//            EAR is laid out (K, R) in R-major order when accessed as (k,r).
//            But Hopper code uses tma_load_EAR with select<1,2>(TileShape_NRK)
//            = (R, bK), and `make_layout(make_shape(R, args.k),
//                                        make_stride(args.k, _1{}))` =
//            (R, K) K-major. That's the same as our (R,K) buffer where
//            element (r,k) is at offset r*K+k. So EAR_R_major used in the
//            Python ref must be transposed when passed in — but the kernel
//            takes "EAR" pointer expecting (R, K) K-major.
//
// So EARxBpEB[n,r] = sum_k BpEB[n,k] * EAR[r*K + k].
static void ref_EARxBpEB(int N, int K, int8_t const* BpEB, int8_t const* EAR,
                         int32_t* EARxBpEB) {
  for (int n = 0; n < N; ++n) {
    for (int r = 0; r < R; ++r) {
      int32_t acc = 0;
      for (int kk = 0; kk < K; ++kk) {
        acc += int32_t(BpEB[n * K + kk]) * int32_t(EAR[r * K + kk]);
      }
      EARxBpEB[n * R + r] = acc;
    }
  }
}

static int run_case(int N, int K, unsigned seed, bool verbose = true) {
  std::srand(seed);

  std::vector<int8_t> hB(size_t(N) * K), hEBR(size_t(N) * R),
      hEBL(size_t(K) * R), hEAR(size_t(R) * K);
  std::vector<int8_t> hBpEB(size_t(N) * K), hBpEB_ref(size_t(N) * K);
  std::vector<int32_t> hEARxBpEB(size_t(N) * R),
      hEARxBpEB_ref(size_t(N) * R);

  // Small values so the int32 EARxBpEB result stays bounded.
  for (auto& v : hB)   v = int8_t((std::rand() % 21) - 10);     // [-10..10]
  for (auto& v : hEBR) v = int8_t((std::rand() % 21) - 10);
  for (auto& v : hEBL) v = int8_t((std::rand() % 21) - 10);
  for (auto& v : hEAR) v = int8_t((std::rand() % 21) - 10);

  // ---- DEBUG knobs (set via env vars) ----
  if (std::getenv("PEARL_DEBUG_EBR_ZERO") != nullptr) {
    std::fill(hEBR.begin(), hEBR.end(), int8_t(0));
  }
  if (std::getenv("PEARL_DEBUG_EBL_ZERO") != nullptr) {
    std::fill(hEBL.begin(), hEBL.end(), int8_t(0));
  }
  if (std::getenv("PEARL_DEBUG_B_ZERO") != nullptr) {
    std::fill(hB.begin(), hB.end(), int8_t(0));
  }
  if (std::getenv("PEARL_DEBUG_EAR_IDENTITY") != nullptr) {
    // EAR is (R, K) K-major; element (r, k) at offset r*K+k.
    // Set EAR to "select column r=k": EAR[r,k] = (r==k) ? 1 : 0 (only for k<R).
    std::fill(hEAR.begin(), hEAR.end(), int8_t(0));
    for (int r = 0; r < R && r < K; ++r) hEAR[r * K + r] = 1;
  }
  if (std::getenv("PEARL_DEBUG_EBR_ONE_EBL_E00") != nullptr) {
    // EBR all 1s, EBL[k=0, r=0] = 1, else 0.
    std::fill(hEBR.begin(), hEBR.end(), int8_t(1));
    std::fill(hEBL.begin(), hEBL.end(), int8_t(0));
    hEBL[0 * R + 0] = 1;
  }
  if (std::getenv("PEARL_DEBUG_EBL_ONES") != nullptr) {
    std::fill(hEBL.begin(), hEBL.end(), int8_t(1));
  }
  if (std::getenv("PEARL_DEBUG_EBR_ONES") != nullptr) {
    std::fill(hEBR.begin(), hEBR.end(), int8_t(1));
  }
  if (std::getenv("PEARL_DEBUG_EBR_EBL_ONES") != nullptr) {
    // EBR all 1s, EBL all 1s. EB[n,k] = R = 64 for all n,k.
    std::fill(hEBR.begin(), hEBR.end(), int8_t(1));
    std::fill(hEBL.begin(), hEBL.end(), int8_t(1));
  }
  if (std::getenv("PEARL_DEBUG_EBL_K0_ONES") != nullptr) {
    // EBR all 1s, EBL[0,:] all 1s, EBL[k>0,:] all 0s.
    // EB[n,0] = sum_r 1*1 = R. EB[n,k>0] = 0.
    std::fill(hEBR.begin(), hEBR.end(), int8_t(1));
    std::fill(hEBL.begin(), hEBL.end(), int8_t(0));
    for (int r = 0; r < R; ++r) hEBL[0 * R + r] = 1;
  }
  if (std::getenv("PEARL_DEBUG_EBL_K0K1_ONES") != nullptr) {
    // EBR all 1s, EBL[0,:] = 1, EBL[1,:] = 1, EBL[k>1,:] = 0.
    // EB[n,0] = R, EB[n,1] = R, else 0.
    std::fill(hEBR.begin(), hEBR.end(), int8_t(1));
    std::fill(hEBL.begin(), hEBL.end(), int8_t(0));
    for (int r = 0; r < R; ++r) {
      hEBL[0 * R + r] = 1;
      hEBL[1 * R + r] = 1;
    }
  }
  if (std::getenv("PEARL_DEBUG_EBL_IDENTITY") != nullptr) {
    // EBL[k,r] = (k%R == r) ? 1 : 0. Then EB[n,k] = EBR[n, k%R].
    std::fill(hEBL.begin(), hEBL.end(), int8_t(0));
    for (int kk = 0; kk < K; ++kk) hEBL[kk * R + (kk % R)] = 1;
  }
  if (std::getenv("PEARL_DEBUG_EBR_IDENTITY") != nullptr) {
    // EBR[n,r] = (n%R == r) ? 1 : 0.
    std::fill(hEBR.begin(), hEBR.end(), int8_t(0));
    for (int n = 0; n < N; ++n) hEBR[n * R + (n % R)] = 1;
  }
  if (std::getenv("PEARL_DEBUG_EBR_R0") != nullptr) {
    // EBR[n,0] = 1, else 0. EBL = identity (EBL[k,r] = (k==r?1:0)).
    // EB[n,k] = sum_r EBR[n,r]*EBL[k,r] = EBR[n,k] if k<R else 0
    //        = (k==0 ? 1 : 0) since EBR[n,r] is only nonzero at r=0.
    std::fill(hEBR.begin(), hEBR.end(), int8_t(0));
    for (int n = 0; n < N; ++n) hEBR[n * R + 0] = 1;
    std::fill(hEBL.begin(), hEBL.end(), int8_t(0));
    for (int kk = 0; kk < K && kk < R; ++kk) hEBL[kk * R + kk] = 1;
  }
  if (std::getenv("PEARL_DEBUG_PRINT_INPUT") != nullptr) {
    printf("hB[0..7]   = "); for (int i = 0; i < 8; ++i) printf("%d ", hB[i]); printf("\n");
    printf("hEBR[0..7] = "); for (int i = 0; i < 8; ++i) printf("%d ", hEBR[i]); printf("\n");
    printf("hEBL[0..7] = "); for (int i = 0; i < 8; ++i) printf("%d ", hEBL[i]); printf("\n");
    printf("hEAR[0..7] = "); for (int i = 0; i < 8; ++i) printf("%d ", hEAR[i]); printf("\n");
  }

  // CPU references
  ref_BpEB(N, K, hB.data(), hEBR.data(), hEBL.data(), hBpEB_ref.data());
  ref_EARxBpEB(N, K, hBpEB_ref.data(), hEAR.data(), hEARxBpEB_ref.data());

  // Device buffers
  int8_t *dB, *dEBR, *dEBL, *dEAR, *dBpEB;
  int32_t* dEARxBpEB;
  CUCHK(cudaMalloc(&dB, hB.size()));
  CUCHK(cudaMalloc(&dEBR, hEBR.size()));
  CUCHK(cudaMalloc(&dEBL, hEBL.size()));
  CUCHK(cudaMalloc(&dEAR, hEAR.size()));
  CUCHK(cudaMalloc(&dBpEB, hBpEB.size()));
  CUCHK(cudaMalloc(&dEARxBpEB, hEARxBpEB.size() * sizeof(int32_t)));

  CUCHK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dEBR, hEBR.data(), hEBR.size(), cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dEBL, hEBL.data(), hEBL.size(), cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dEAR, hEAR.data(), hEAR.size(), cudaMemcpyHostToDevice));
  CUCHK(cudaMemset(dBpEB, 0xAA, hBpEB.size()));
  CUCHK(cudaMemset(dEARxBpEB, 0xAA, hEARxBpEB.size() * sizeof(int32_t)));

  pearl::sm89::pearl_noisingB_sm89_64x64x64_R64_int32(
      dB, dEBR, dEBL, dEAR, dBpEB, dEARxBpEB, N, K, /*stream=*/0);
  CUCHK(cudaDeviceSynchronize());
  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    fprintf(stderr, "kernel launch error: %s\n",
            cudaGetErrorString(launch_err));
    return 1;
  }

  CUCHK(cudaMemcpy(hBpEB.data(), dBpEB, hBpEB.size(),
                   cudaMemcpyDeviceToHost));
  CUCHK(cudaMemcpy(hEARxBpEB.data(), dEARxBpEB,
                   hEARxBpEB.size() * sizeof(int32_t),
                   cudaMemcpyDeviceToHost));

  long bad_BpEB = 0, bad_EAR = 0;
  long worst_idx_BpEB = -1, worst_idx_EAR = -1;
  int32_t worst_BpEB_got = 0, worst_BpEB_ref = 0;
  int32_t worst_EAR_got = 0, worst_EAR_ref = 0;
  int32_t worst_EAR_diff = 0;

  for (size_t i = 0; i < hBpEB.size(); ++i) {
    if (hBpEB[i] != hBpEB_ref[i]) {
      ++bad_BpEB;
      if (worst_idx_BpEB < 0) {
        worst_idx_BpEB = long(i);
        worst_BpEB_got = hBpEB[i];
        worst_BpEB_ref = hBpEB_ref[i];
      }
    }
  }
  for (size_t i = 0; i < hEARxBpEB.size(); ++i) {
    if (hEARxBpEB[i] != hEARxBpEB_ref[i]) {
      ++bad_EAR;
      int32_t d = std::abs(hEARxBpEB[i] - hEARxBpEB_ref[i]);
      if (d > std::abs(worst_EAR_diff)) {
        worst_EAR_diff = hEARxBpEB[i] - hEARxBpEB_ref[i];
        worst_idx_EAR = long(i);
        worst_EAR_got = hEARxBpEB[i];
        worst_EAR_ref = hEARxBpEB_ref[i];
      }
    }
  }

  printf("N=%d K=%d seed=%u\n", N, K, seed);
  printf("  BpEB     bad=%ld/%zu", bad_BpEB, hBpEB.size());
  if (bad_BpEB == 0) {
    printf("  PASS\n");
  } else {
    long row = worst_idx_BpEB / K;
    long col = worst_idx_BpEB % K;
    printf("  FAIL  worst@[%ld,%ld]  ref=%d got=%d\n", row, col, worst_BpEB_ref,
           worst_BpEB_got);
  }
  if (std::getenv("PEARL_DEBUG_DUMP_BPEB") != nullptr) {
    printf("  BpEB sample (rows 0..7 and around mismatch):\n");
    for (int n = 0; n < N; ++n) {
      bool any_diff = false;
      for (int k = 0; k < K && !any_diff; ++k) {
        if (hBpEB[n*K+k] != hBpEB_ref[n*K+k]) any_diff = true;
      }
      if (n < 4 || any_diff) {
        printf("    ref n=%3d k=0..7: ", n);
        for (int k = 0; k < 8 && k < K; ++k) printf("%4d ", int(hBpEB_ref[n*K+k]));
        printf("\n    got n=%3d k=0..7: ", n);
        for (int k = 0; k < 8 && k < K; ++k) printf("%4d ", int(hBpEB[n*K+k]));
        printf("  %s\n", any_diff ? "DIFF" : "ok");
      }
    }
  }
  printf("  EARxBpEB bad=%ld/%zu", bad_EAR, hEARxBpEB.size());
  if (bad_EAR == 0) {
    printf("  PASS\n");
  } else {
    long row = worst_idx_EAR / R;
    long col = worst_idx_EAR % R;
    printf("  FAIL  worst@[%ld,%ld]  ref=%d got=%d diff=%d\n", row, col,
           worst_EAR_ref, worst_EAR_got, worst_EAR_diff);
    if (verbose) {
      printf("  sample EARxBpEB mismatches:\n");
      int shown = 0;
      for (size_t i = 0; i < hEARxBpEB.size() && shown < 12; ++i) {
        if (hEARxBpEB[i] != hEARxBpEB_ref[i]) {
          long r = long(i) / R, c = long(i) % R;
          printf("    [%4ld,%4ld] ref=%9d got=%9d\n", r, c,
                 hEARxBpEB_ref[i], hEARxBpEB[i]);
          ++shown;
        }
      }
    }
  }

  cudaFree(dB);
  cudaFree(dEBR);
  cudaFree(dEBL);
  cudaFree(dEAR);
  cudaFree(dBpEB);
  cudaFree(dEARxBpEB);
  return (bad_BpEB == 0 && bad_EAR == 0) ? 0 : 1;
}

int main(int argc, char** argv) {
  int dev = (argc >= 2) ? std::atoi(argv[1]) : 0;
  CUCHK(cudaSetDevice(dev));
  cudaDeviceProp p;
  CUCHK(cudaGetDeviceProperties(&p, dev));
  int cc = p.major * 10 + p.minor;
  printf("device %d: %s sm_%d (smem/SM optin: %d KB)\n", dev, p.name, cc,
         int(p.sharedMemPerBlockOptin / 1024));
  if (cc != 89) {
    fprintf(stderr, "WARN: built for sm_89; current is sm_%d\n", cc);
  }

  int rc = 0;
  // Acceptance case from the task spec: N=128, K=512, R=64.
  rc |= run_case(128, 512, 0);
  if (rc != 0) {
    printf("\n(stopping after first failure)\n");
    return rc;
  }
  rc |= run_case(64, 64, 1);
  rc |= run_case(256, 1024, 2);
  if (rc == 0)
    printf("\nALL PASS\n");
  else
    printf("\nFAIL\n");
  return rc;
}
