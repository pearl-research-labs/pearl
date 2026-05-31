// SPDX-License-Identifier: see LICENSE
//
// Standalone correctness test for pearl_noisingA_sm89_64x64x64_R64_int32.
//
// Reference:
//   EA[i,k]    = sum_r EAL[i,r] * EAR[k,r]   (int32)
//   ApEA[i,k]  = (A[i,k] + EA[i,k]) truncated to int8
//   AxEBL[i,r] = sum_k A[i,k] * EBL[k,r]      (int32)
//
// Tile size: bM=64, R=64, bK=64.
//
// Build (in WSL):
//   nvcc -gencode arch=compute_89,code=sm_89 -std=c++20 -O3
//        -I . -I .. -I ../../third_party/cutlass/include
//        -I ../../third_party/cutlass/tools/util/include
//        -I ../../third_party/cutlass/examples/common
//        --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG
//        pearl_noisingA_sm89_inst.cu test_noisingA_sm89_standalone.cu
//        -lcudart -o /tmp/test_noisingA_sm89

#include <cuda_runtime.h>

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void pearl_noisingA_sm89_64x64x64_R64_int32(
    int8_t const* A, int8_t const* EAL, int8_t const* EAR, int8_t const* EBL,
    int8_t* ApEA, int32_t* AxEBL,
    int M, int K, cudaStream_t stream);

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

static constexpr int R_DIM = 64;

// CPU reference. Each input matrix is int8 K- or R-major as labeled.
// A: (M, K), K-major (i.e. row-major with stride K).
// EAL: (M, R), R-major.
// EAR: (K, R), R-major.   (used as (R, K) via transpose semantically — i.e.
//                          EA[i,k] = sum_r EAL[i,r] * EAR[k,r].)
// EBL: (K, R), K-major... wait, K-major means stride (1, K). For a (K, R)
//      shape stored K-major, row-index of element (k, r) is k*R+r OR k+r*K?
//      Per the python ref `torch._int_mm(A_ref, EBL_ref)`, EBL is (K, R)
//      indexed [k, r] → element at offset k*R + r. That's R-INNER (R-major
//      stride) for a (K, R) tensor. The Hopper kernel calls it `EBL_K_major`
//      because it's contiguous along K when transposed for the GEMM.
//
//      But the kernel uses TMA layout for EBL with shape (R, k) stride (k, 1).
//      That's saying EBL is shape (R, K) with stride (K, 1), i.e. row-major
//      with rows = R. Equivalent to a (K, R) tensor with stride (R, 1)... NO.
//      A (R, K) row-major tensor with stride (K, 1) is the TRANSPOSE of a
//      (K, R) row-major tensor with stride (R, 1).
//
//      So the kernel sees EBL as logically (R, K) with each row being a
//      contiguous K-vector — meaning a single column of the (K, R) "user-view"
//      maps to a row of the "kernel-view". For the matmul AxEBL = A @ EBL
//      we want sum_k A[i,k]*EBL[k,r]; if kernel reads EBL_kernel[r,k] for
//      the (r-th row, k-th col), then EBL_kernel[r,k] = EBL_user[k,r].
//
//      So the user-view EBL is (K, R) row-major (stride R, 1) at the test
//      level; but the kernel was given a pointer treating it as (R, K) row-
//      major (stride K, 1). These are the SAME memory if (K, R) row-major
//      stores element (k, r) at offset k*R + r, AND (R, K) row-major stores
//      element (r, k) at offset r*K + k — which is a DIFFERENT layout.
//
//      Resolution: the test builds two buffers — one for the user view
//      ("torch._int_mm(A_ref, EBL_ref)" reads (K,R) shape with R-stride),
//      one transposed for the kernel ("EBL_K_major"). The Hopper test
//      does this via tensor_generator EBL_K_major attr. For our standalone
//      we'll generate raw bytes in *both* views and pass the kernel view to
//      the kernel.
static void ref_noisingA(int M, int K,
                         int8_t const* A,      // (M, K) row-major
                         int8_t const* EAL,    // (M, R) row-major
                         int8_t const* EAR,    // (K, R) row-major
                         int8_t const* EBL_user, // (K, R) row-major
                         int8_t* ApEA,         // (M, K) row-major
                         int32_t* AxEBL) {     // (M, R) row-major
    // EA[i,k] = sum_r EAL[i,r] * EAR[k,r]   ; ApEA = (A + EA) cast to int8
    // Inputs are constrained to "int7" range (A in [-64,62], EAL in [-32,31],
    // EAR has at most two ±1 per row) so the sum fits in int8 and no saturation.
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
    // AxEBL[i,r] = sum_k A[i,k] * EBL[k,r]
    for (int i = 0; i < M; ++i) {
        for (int r = 0; r < R_DIM; ++r) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += int32_t(A[i*K + k]) * int32_t(EBL_user[k*R_DIM + r]);
            }
            AxEBL[i*R_DIM + r] = acc;
        }
    }
}

static int run_case(int M, int K, unsigned seed,
                    bool dump_mismatches = true) {
    if (M % 64 != 0 || K % 64 != 0) {
        fprintf(stderr, "M=%d K=%d not multiples of 64\n", M, K);
        return 1;
    }
    std::srand(seed);
    std::vector<int8_t>  hA(size_t(M) * K);
    std::vector<int8_t>  hEAL(size_t(M) * R_DIM);
    std::vector<int8_t>  hEAR(size_t(K) * R_DIM);
    std::vector<int8_t>  hEBL_user(size_t(K) * R_DIM);   // (K, R) row-major
    std::vector<int8_t>  hEBL_kernel(size_t(R_DIM) * K); // (R, K) row-major (transpose of user view)
    std::vector<int8_t>  hApEA(size_t(M) * K), hApEA_ref(size_t(M) * K);
    std::vector<int32_t> hAxEBL(size_t(M) * R_DIM), hAxEBL_ref(size_t(M) * R_DIM);

    // Build int7-style inputs (A in [-64,62], EAL in [-32,31], EAR / EBL
    // are sparse with exactly one +1 and one -1 per row). Matches the
    // GemmTensorGenerator pattern (compute_EAR_and_EBL) so the int32 sum
    // A + EA stays in int8 range and `int8_t(sum)` truncation matches the
    // kernel's int32→int8 saturating cast.
    auto rand_int7_a = [](){ return int8_t((std::rand() % 127) - 64); };  // [-64,62]
    auto rand_int6   = [](){ return int8_t((std::rand() % 63)  - 32); };  // [-32,30]
    for (auto& v : hA)   v = rand_int7_a();
    for (auto& v : hEAL) v = rand_int6();
    // EAR: shape (K, R), one +1 and one -1 per row at random R-positions.
    std::fill(hEAR.begin(), hEAR.end(), int8_t(0));
    for (int k = 0; k < K; ++k) {
        int p0 = std::rand() % R_DIM;
        int p1 = std::rand() % R_DIM;
        // if same, picking 0 net effect — that's fine, still in range
        hEAR[k * R_DIM + p0] = int8_t(hEAR[k * R_DIM + p0] + 1);
        hEAR[k * R_DIM + p1] = int8_t(hEAR[k * R_DIM + p1] - 1);
    }
    // EBL_user: shape (K, R), same sparse pattern.
    std::fill(hEBL_user.begin(), hEBL_user.end(), int8_t(0));
    for (int k = 0; k < K; ++k) {
        int p0 = std::rand() % R_DIM;
        int p1 = std::rand() % R_DIM;
        hEBL_user[k * R_DIM + p0] = int8_t(hEBL_user[k * R_DIM + p0] + 1);
        hEBL_user[k * R_DIM + p1] = int8_t(hEBL_user[k * R_DIM + p1] - 1);
    }
    // Transpose EBL_user (K,R) -> EBL_kernel (R,K).
    for (int k = 0; k < K; ++k)
        for (int r = 0; r < R_DIM; ++r)
            hEBL_kernel[r * K + k] = hEBL_user[k * R_DIM + r];

    int8_t  *dA, *dEAL, *dEAR, *dEBL, *dApEA;
    int32_t *dAxEBL;
    CUCHK(cudaMalloc(&dA,    hA.size()));
    CUCHK(cudaMalloc(&dEAL,  hEAL.size()));
    CUCHK(cudaMalloc(&dEAR,  hEAR.size()));
    CUCHK(cudaMalloc(&dEBL,  hEBL_kernel.size()));
    CUCHK(cudaMalloc(&dApEA, hApEA.size()));
    CUCHK(cudaMalloc(&dAxEBL, hAxEBL.size() * sizeof(int32_t)));

    CUCHK(cudaMemcpy(dA,   hA.data(),   hA.size(),   cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEAL, hEAL.data(), hEAL.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEAR, hEAR.data(), hEAR.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEBL, hEBL_kernel.data(), hEBL_kernel.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemset(dApEA, 0xAA, hApEA.size()));
    CUCHK(cudaMemset(dAxEBL, 0xAA, hAxEBL.size() * sizeof(int32_t)));

    pearl_noisingA_sm89_64x64x64_R64_int32(
        dA, dEAL, dEAR, dEBL, dApEA, dAxEBL, M, K, /*stream=*/0);
    CUCHK(cudaDeviceSynchronize());
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "kernel launch error: %s\n", cudaGetErrorString(launch_err));
        return 1;
    }

    CUCHK(cudaMemcpy(hApEA.data(),  dApEA,  hApEA.size(),  cudaMemcpyDeviceToHost));
    CUCHK(cudaMemcpy(hAxEBL.data(), dAxEBL, hAxEBL.size() * sizeof(int32_t),
                     cudaMemcpyDeviceToHost));
    ref_noisingA(M, K, hA.data(), hEAL.data(), hEAR.data(), hEBL_user.data(),
                 hApEA_ref.data(), hAxEBL_ref.data());

    // Compare ApEA (int8, (M, K))
    long bad_apea = 0;
    long worst_apea_idx = 0;
    int worst_apea_got = 0, worst_apea_ref = 0;
    int worst_apea_diff = 0;
    for (size_t i = 0; i < hApEA.size(); ++i) {
        if (hApEA[i] != hApEA_ref[i]) {
            ++bad_apea;
            int d = std::abs(int(hApEA[i]) - int(hApEA_ref[i]));
            if (d > std::abs(worst_apea_diff)) {
                worst_apea_diff = int(hApEA[i]) - int(hApEA_ref[i]);
                worst_apea_idx  = long(i);
                worst_apea_got  = int(hApEA[i]);
                worst_apea_ref  = int(hApEA_ref[i]);
            }
        }
    }
    // Compare AxEBL (int32, (M, R))
    long bad_axebl = 0;
    long worst_axebl_idx = 0;
    int32_t worst_axebl_got = 0, worst_axebl_ref = 0;
    int32_t worst_axebl_diff = 0;
    for (size_t i = 0; i < hAxEBL.size(); ++i) {
        if (hAxEBL[i] != hAxEBL_ref[i]) {
            ++bad_axebl;
            int32_t d = std::abs(hAxEBL[i] - hAxEBL_ref[i]);
            if (d > std::abs(worst_axebl_diff)) {
                worst_axebl_diff = hAxEBL[i] - hAxEBL_ref[i];
                worst_axebl_idx  = long(i);
                worst_axebl_got  = hAxEBL[i];
                worst_axebl_ref  = hAxEBL_ref[i];
            }
        }
    }

    long bad = bad_apea + bad_axebl;
    printf("M=%d K=%d R=%d seed=%u  bad_apea=%ld/%zu  bad_axebl=%ld/%zu",
           M, K, R_DIM, seed, bad_apea, hApEA.size(), bad_axebl, hAxEBL.size());
    if (bad == 0) {
        printf("   PASS\n");
    } else {
        printf("   FAIL\n");
        if (bad_apea && dump_mismatches) {
            long r = worst_apea_idx / K, c = worst_apea_idx % K;
            printf("  worst ApEA@[%ld,%ld] ref=%d got=%d diff=%d\n",
                   r, c, worst_apea_ref, worst_apea_got, worst_apea_diff);
            printf("  sample ApEA mismatches:\n");
            int shown = 0;
            for (size_t i = 0; i < hApEA.size() && shown < 16; ++i) {
                if (hApEA[i] != hApEA_ref[i]) {
                    long rr = long(i) / K, cc = long(i) % K;
                    printf("    [%4ld,%4ld] ref=%4d got=%4d diff=%4d\n",
                           rr, cc, int(hApEA_ref[i]), int(hApEA[i]),
                           int(hApEA[i]) - int(hApEA_ref[i]));
                    ++shown;
                }
            }
        }
        if (bad_axebl && dump_mismatches) {
            long r = worst_axebl_idx / R_DIM, c = worst_axebl_idx % R_DIM;
            printf("  worst AxEBL@[%ld,%ld] ref=%d got=%d diff=%d\n",
                   r, c, worst_axebl_ref, worst_axebl_got, worst_axebl_diff);
            printf("  sample AxEBL mismatches:\n");
            int shown = 0;
            for (size_t i = 0; i < hAxEBL.size() && shown < 16; ++i) {
                if (hAxEBL[i] != hAxEBL_ref[i]) {
                    long rr = long(i) / R_DIM, cc = long(i) % R_DIM;
                    printf("    [%4ld,%4ld] ref=%9d got=%9d diff=%9d\n",
                           rr, cc, hAxEBL_ref[i], hAxEBL[i],
                           hAxEBL[i] - hAxEBL_ref[i]);
                    ++shown;
                }
            }
        }
    }

    CUCHK(cudaFree(dA));
    CUCHK(cudaFree(dEAL));
    CUCHK(cudaFree(dEAR));
    CUCHK(cudaFree(dEBL));
    CUCHK(cudaFree(dApEA));
    CUCHK(cudaFree(dAxEBL));
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
    if (cc != 89) {
        fprintf(stderr, "WARN: built for sm_89; current is sm_%d\n", cc);
    }

    int rc = 0;
    rc |= run_case(128, 512, 0);
    if (rc != 0) {
        printf("\n(stopping after first failure for clarity)\n");
        return rc;
    }
    rc |= run_case(64, 64, 1);
    rc |= run_case(256, 256, 2);
    rc |= run_case(512, 512, 3);
    rc |= run_case(1024, 1024, 4);
    rc |= run_case(2048, 256, 5);
    rc |= run_case(256, 2048, 6);
    if (rc == 0) printf("\nALL PASS\n");
    else         printf("\nFAIL\n");
    return rc;
}
