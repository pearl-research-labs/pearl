// SPDX-License-Identifier: see LICENSE
//
// Standalone correctness smoke test for the sm_89 fp16 noisingA path
// (AxEBL_type = cutlass::half_t, R = 64).
//
// This exercises the run_pearl_noising_A_sm89<cutlass::half_t, ..., true>
// instantiation that was added in pearl_noisingA_sm89_inst.cu. The fp16 path
// reuses the same kernel as int32 except for the final epilogue which divides
// the int32 accumulator by kAxEBLScaleFactor (= 2^14) and downcasts to fp16.
//
// Build (in WSL with CUDA 12.x):
//   nvcc -gencode arch=compute_89,code=compute_89 -std=c++20 -O3
//        -I . -I .. -I ../../third_party/cutlass/include
//        -I ../../third_party/cutlass/tools/util/include
//        -I ../../third_party/cutlass/examples/common
//        --expt-relaxed-constexpr --expt-extended-lambda
//        -DNDEBUG -DPEARL_GEMM_BUILD_SM89
//        pearl_noisingA_sm89_inst.cu test_noisingA_sm89_fp16_standalone.cu
//        -lcudart -o /tmp/test_noisingA_sm89_fp16

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Forward declarations: pull in the sm_89 host template + the PearlAPIParams
// type so we can invoke the template directly.
#include "cute/tensor.hpp"
#include "pearl_api_params.h"
#include "pearl_noisingA_sm89_host.h"
#include "pearl_gemm_constants.hpp"
#include <cutlass/numeric_types.h>

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

#ifndef R_DIM_OVERRIDE
#define R_DIM_OVERRIDE 64
#endif
static constexpr int R_DIM = R_DIM_OVERRIDE;
static constexpr int kBM_TILE = 64;
static constexpr int kBK_TILE = 64;

// CPU reference, fp16 epilogue version:
//   ApEA = int8(A + sum_r EAL * EAR)
//   AxEBL_int32 = sum_k A * EBL  (int32 acc)
//   AxEBL_fp16  = float(AxEBL_int32) / 2^14, downcast to fp16
static void ref_noisingA_fp16(int M, int K,
                              int8_t const* A,
                              int8_t const* EAL,
                              int8_t const* EAR,
                              int8_t const* EBL_user,
                              int8_t* ApEA,
                              __half* AxEBL_fp16) {
    constexpr float kScale = float(pearl::kAxEBLScaleFactor);  // 16384
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            int32_t ea = 0;
            for (int r = 0; r < R_DIM; ++r) {
                ea += int32_t(EAL[i*R_DIM + r]) * int32_t(EAR[k*R_DIM + r]);
            }
            int32_t s = int32_t(A[i*K + k]) + ea;
            ApEA[i*K + k] = int8_t(s);
        }
    }
    for (int i = 0; i < M; ++i) {
        for (int r = 0; r < R_DIM; ++r) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += int32_t(A[i*K + k]) * int32_t(EBL_user[k*R_DIM + r]);
            }
            AxEBL_fp16[i*R_DIM + r] = __float2half(float(acc) / kScale);
        }
    }
}

static int run_case(int M, int K, unsigned seed) {
    using TileShape_MRK = cute::Shape<cute::Int<kBM_TILE>, cute::Int<R_DIM>,
                                      cute::Int<kBK_TILE>>;
    if (M % kBM_TILE != 0 || K % kBK_TILE != 0) {
        fprintf(stderr, "M=%d K=%d not multiples of tile=64\n", M, K);
        return 1;
    }
    std::srand(seed);
    std::vector<int8_t>  hA(size_t(M) * K);
    std::vector<int8_t>  hEAL(size_t(M) * R_DIM);
    std::vector<int8_t>  hEAR(size_t(K) * R_DIM);
    std::vector<int8_t>  hEBL_user(size_t(K) * R_DIM);
    std::vector<int8_t>  hEBL_kernel(size_t(R_DIM) * K);
    std::vector<int8_t>  hApEA(size_t(M) * K), hApEA_ref(size_t(M) * K);
    std::vector<__half>  hAxEBL_fp16(size_t(M) * R_DIM), hAxEBL_ref_fp16(size_t(M) * R_DIM);

    auto rand_int7_a = [](){ return int8_t((std::rand() % 127) - 64); };
    auto rand_int6   = [](){ return int8_t((std::rand() % 63)  - 32); };
    for (auto& v : hA)   v = rand_int7_a();
    for (auto& v : hEAL) v = rand_int6();
    std::fill(hEAR.begin(), hEAR.end(), int8_t(0));
    for (int k = 0; k < K; ++k) {
        int p0 = std::rand() % R_DIM;
        int p1 = std::rand() % R_DIM;
        hEAR[k * R_DIM + p0] = int8_t(hEAR[k * R_DIM + p0] + 1);
        hEAR[k * R_DIM + p1] = int8_t(hEAR[k * R_DIM + p1] - 1);
    }
    std::fill(hEBL_user.begin(), hEBL_user.end(), int8_t(0));
    for (int k = 0; k < K; ++k) {
        int p0 = std::rand() % R_DIM;
        int p1 = std::rand() % R_DIM;
        hEBL_user[k * R_DIM + p0] = int8_t(hEBL_user[k * R_DIM + p0] + 1);
        hEBL_user[k * R_DIM + p1] = int8_t(hEBL_user[k * R_DIM + p1] - 1);
    }
    // Transpose EBL_user (K,R) -> EBL_kernel (R,K)
    for (int k = 0; k < K; ++k)
        for (int r = 0; r < R_DIM; ++r)
            hEBL_kernel[r * K + k] = hEBL_user[k * R_DIM + r];

    int8_t  *dA, *dEAL, *dEAR, *dEBL, *dApEA;
    __half  *dAxEBL_fp16;
    CUCHK(cudaMalloc(&dA,    hA.size()));
    CUCHK(cudaMalloc(&dEAL,  hEAL.size()));
    CUCHK(cudaMalloc(&dEAR,  hEAR.size()));
    CUCHK(cudaMalloc(&dEBL,  hEBL_kernel.size()));
    CUCHK(cudaMalloc(&dApEA, hApEA.size()));
    CUCHK(cudaMalloc(&dAxEBL_fp16, hAxEBL_fp16.size() * sizeof(__half)));

    CUCHK(cudaMemcpy(dA,   hA.data(),   hA.size(),   cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEAL, hEAL.data(), hEAL.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEAR, hEAR.data(), hEAR.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEBL, hEBL_kernel.data(), hEBL_kernel.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemset(dApEA, 0xAA, hApEA.size()));
    CUCHK(cudaMemset(dAxEBL_fp16, 0xAA, hAxEBL_fp16.size() * sizeof(__half)));

    // Build PearlAPIParams for the sm_89 fp16 launcher.
    PearlAPIParams params{};
    params.ptr_A            = dA;
    params.ptr_EAL          = dEAL;
    params.ptr_EAR_R_major  = dEAR;
    params.ptr_EBL_K_major  = dEBL;
    params.ptr_ApEA         = dApEA;
    params.ptr_AxEBL        = dAxEBL_fp16;
    params.m                = M;
    params.k                = K;
    params.k_blocks_per_split_noising_A = 0;

    run_pearl_noising_A_sm89<cutlass::half_t, TileShape_MRK,
                             /*kStages=*/2, /*IsEvenK=*/true>(params, 0);
    CUCHK(cudaDeviceSynchronize());
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "kernel launch error: %s\n", cudaGetErrorString(launch_err));
        return 1;
    }

    CUCHK(cudaMemcpy(hApEA.data(),       dApEA,       hApEA.size(),       cudaMemcpyDeviceToHost));
    CUCHK(cudaMemcpy(hAxEBL_fp16.data(), dAxEBL_fp16, hAxEBL_fp16.size() * sizeof(__half),
                     cudaMemcpyDeviceToHost));
    ref_noisingA_fp16(M, K, hA.data(), hEAL.data(), hEAR.data(), hEBL_user.data(),
                      hApEA_ref.data(), hAxEBL_ref_fp16.data());

    long bad_apea = 0;
    for (size_t i = 0; i < hApEA.size(); ++i)
        if (hApEA[i] != hApEA_ref[i]) ++bad_apea;

    // fp16 compare: bit-exact match (both reference and kernel run the same
    // int32 accumulation then divide by 2^14 and downcast). Allow a 1-ulp
    // tolerance for the float->half rounding mode difference (host
    // __float2half uses round-to-nearest-even; CUTLASS may use the same but
    // PTX-level half_t casts use saturating-to-finite).
    long bad_axebl = 0;
    float worst_axebl_diff = 0.0f;
    for (size_t i = 0; i < hAxEBL_fp16.size(); ++i) {
        float g = __half2float(hAxEBL_fp16[i]);
        float r = __half2float(hAxEBL_ref_fp16[i]);
        float d = std::fabs(g - r);
        if (d > worst_axebl_diff) worst_axebl_diff = d;
        // Accept up to 4 ULPs of fp16 difference (relative to the dominant
        // magnitude). For most values that's ~0.001-0.05 absolute. Bit-exact
        // is the goal but we tolerate small rounding-mode mismatches.
        float tol = std::max(2e-3f, 4e-3f * std::fabs(r));
        if (d > tol) ++bad_axebl;
    }

    long bad = bad_apea + bad_axebl;
    printf("M=%d K=%d R=%d seed=%u  bad_apea=%ld/%zu  bad_axebl=%ld/%zu  worst_axebl_diff=%.6f",
           M, K, R_DIM, seed, bad_apea, hApEA.size(), bad_axebl, hAxEBL_fp16.size(),
           worst_axebl_diff);
    if (bad == 0) printf("   PASS\n");
    else          printf("   FAIL\n");

    CUCHK(cudaFree(dA));
    CUCHK(cudaFree(dEAL));
    CUCHK(cudaFree(dEAR));
    CUCHK(cudaFree(dEBL));
    CUCHK(cudaFree(dApEA));
    CUCHK(cudaFree(dAxEBL_fp16));
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
    rc |= run_case(256, 256, 2);
    rc |= run_case(512, 512, 3);
    rc |= run_case(1024, 1024, 4);
    rc |= run_case(2048,  256, 5);
    rc |= run_case( 256, 2048, 6);
    printf("\n%s\n", rc == 0 ? "ALL PASS" : "FAIL");
    return rc;
}
