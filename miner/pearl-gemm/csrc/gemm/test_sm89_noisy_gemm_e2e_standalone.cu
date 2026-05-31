// SPDX-License-Identifier: see LICENSE
//
// End-to-end standalone composition test for the sm_89 noisy-GEMM pipeline.
//
// Pipeline:
//   1. noisingA:  (A, EAL, EAR_R_major, EBL_K_major) -> (ApEA, AxEBL_int32)
//   2. noisingB:  (B, EBR, EBL_R_major, EAR_K_major) -> (BpEB, EARxBpEB_int32)
//   3. denoise_converter:
//        AxEBL_int32   * 2^-14 -> AxEBL_fp16
//        EARxBpEB_int32 * 2^-12 -> EARxBpEB_fp16
//      (CPU-side cast; the generic GPU kernel lives in denoise_converter.cu but
//       its host wrapper takes a `PearlAPIParams` struct — for a self-contained
//       test the CPU cast is simpler and bit-equivalent within tolerance.)
//   4. pearl_gemm_sm89_denoise_128x128x64_R64:
//        (ApEA, BpEB, EAL_fp16=-1*EAL, EBR_fp16=-4*EBR, AxEBL_fp16, EARxBpEB_fp16,
//         A_scales, B_scales) -> C (bf16)
//
// Reference (mirrors test_pearl_gemm.py::TestNoisyGEMM.test_int7_noisy_gemm):
//   C_ref = bf16( (A @ B^T) * A_scales[:,None] * B_scales[None,:] )
//
// The denoise epilogue cancels the noise contributions:
//   ApEA @ BpEB^T = (A+EA)(B+EB)^T = AB^T + EA B^T + A EB^T + EA EB^T
//   - EAL @ (EAR^T B^T)       cancels EA B^T  (since EA = EAL @ EAR^T)
//   - (A @ EBL) @ EBR^T       cancels A EB^T  (since EB = EBR @ EBL^T)
//   - EAL @ (EAR^T EB^T) = ? — actually the denoise intermediates carry the
//     full BpEB contribution, so EAxBpEB = EAL @ (BpEB @ EAR_R_major)
//     captures BOTH EA B^T and EA EB^T parts. Net result: AB^T survives.
//
// Build (in WSL):
//   /usr/local/cuda-12.8/bin/nvcc -gencode arch=compute_89,code=sm_89 -std=c++20
//        -O3 -I . -I .. -I ../../third_party/cutlass/include
//        -I ../../third_party/cutlass/tools/util/include
//        -I ../../third_party/cutlass/examples/common
//        --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG
//        pearl_noisingA_sm89_inst.cu pearl_noisingB_sm89_inst.cu
//        pearl_gemm_sm89_denoise_inst.cu denoise_converter.cu
//        test_sm89_noisy_gemm_e2e_standalone.cu
//        -lcudart -o /tmp/test_sm89_e2e

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---- Extern C entry points of the three GPU kernels we compose ------------
extern "C" void pearl_noisingA_sm89_64x64x64_R64_int32(
    int8_t const* A, int8_t const* EAL, int8_t const* EAR, int8_t const* EBL,
    int8_t* ApEA, int32_t* AxEBL,
    int M, int K, cudaStream_t stream);

namespace pearl {
namespace sm89 {
extern "C" void pearl_noisingB_sm89_64x64x64_R64_int32(
    int8_t const* B, int8_t const* EBR, int8_t const* EBL, int8_t const* EAR,
    int8_t* BpEB, int32_t* EARxBpEB, int N, int K, cudaStream_t stream);

extern "C" void pearl_gemm_sm89_denoise_128x128x64_R64(
    int8_t const* A, int64_t lda,
    int8_t const* B, int64_t ldb,
    cutlass::bfloat16_t* C, int64_t ldc,
    float const* A_scales,
    float const* B_scales,
    cutlass::half_t const* EAL,
    cutlass::half_t const* EBR,
    cutlass::half_t const* AxEBL,
    cutlass::half_t const* EARxBpEB,
    int M, int N, int K,
    cudaStream_t stream);
}  // namespace sm89
}  // namespace pearl

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

static constexpr int R_DIM             = 64;
static constexpr int kAxEBLScale       = 1 << 14;  // 2^14
static constexpr int kEARxBpEBScale    = 1 << 12;  // 2^12
static constexpr int kEAL_DenoiseScale = -1;       // see pearl_gemm_constants.hpp
static constexpr int kEBR_DenoiseScale = -4;       // -1 * (2^14)/(2^12)

// ---- bf16/fp16 helpers (bit-level conversions) ----------------------------
static float bf16_to_float(uint16_t bf) {
    uint32_t b = uint32_t(bf) << 16;
    float v;
    std::memcpy(&v, &b, 4);
    return v;
}

static uint16_t float_to_half(float v) {
    __half hv = __float2half(v);
    uint16_t bits;
    std::memcpy(&bits, &hv, 2);
    return bits;
}

static uint16_t float_to_bf16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t rounding_bias = 0x7fffu + lsb;
    return uint16_t((bits + rounding_bias) >> 16);
}

// ---- CPU reference --------------------------------------------------------
// Mirrors test_pearl_gemm.py::compute_ref_tensor:
//   C_ref = bf16( (A @ B^T) * A_scales[:,None] * B_scales[None,:] )
// All arithmetic in fp64 for max precision, then cast to bf16 at the end.
static void ref_gemm(int M, int N, int K,
                     int8_t const* A, int8_t const* B,
                     float const* A_scales, float const* B_scales,
                     uint16_t* C_bf16) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // (A @ B^T)[i,j] = sum_k A[i,k] * B[j,k]
            int64_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += int64_t(A[i * K + k]) * int64_t(B[j * K + k]);
            }
            double v = double(acc) * double(A_scales[i]) * double(B_scales[j]);
            float vf = float(v);
            C_bf16[i * N + j] = float_to_bf16(vf);
        }
    }
}

// Generate sparse EAR/EBL: each row of an (K, R) tensor has exactly one +1
// and one -1, placed at random positions. Matches compute_EAR_and_EBL in the
// Python tensor generator. If p0==p1 the row is all zeros (matches Python).
static void make_sparse_KR(std::vector<int8_t>& M_KR, int K, int R) {
    M_KR.assign(size_t(K) * R, int8_t(0));
    for (int k = 0; k < K; ++k) {
        int p0 = std::rand() % R;
        int p1 = std::rand() % R;
        M_KR[k * R + p0] = int8_t(M_KR[k * R + p0] + 1);
        M_KR[k * R + p1] = int8_t(M_KR[k * R + p1] - 1);
    }
}

// Transpose (K, R) row-major -> (R, K) row-major.
static void transpose_KR_to_RK(int8_t const* src, std::vector<int8_t>& dst,
                                int K, int R) {
    dst.assign(size_t(R) * K, int8_t(0));
    for (int k = 0; k < K; ++k)
        for (int r = 0; r < R; ++r)
            dst[r * K + k] = src[k * R + r];
}

// ---- Case runner ----------------------------------------------------------
struct CaseSpec {
    int M, N, K;
    unsigned seed;
    float atol;
    float rtol;
};

static int run_case(CaseSpec const& spec) {
    int M = spec.M, N = spec.N, K = spec.K, R = R_DIM;
    if (M % 64 != 0 || N % 64 != 0 || K % 64 != 0) {
        fprintf(stderr, "M=%d N=%d K=%d not multiples of 64\n", M, N, K);
        return 1;
    }
    std::srand(spec.seed);

    // ---- Build host tensors (int7-style ranges, matching Python generator) -
    // A in [-64, 62], EAL in [-32, 30], EBR in [-32, 30].
    std::vector<int8_t> hA(size_t(M) * K), hB(size_t(N) * K);
    std::vector<int8_t> hEAL(size_t(M) * R), hEBR(size_t(N) * R);
    std::vector<int8_t> hEAR_KR;   // (K, R) row-major  -- noisingA's EAR arg
    std::vector<int8_t> hEAR_RK;   // (R, K) row-major  -- noisingB's EAR arg
    std::vector<int8_t> hEBL_KR;   // (K, R) row-major  -- noisingB's EBL arg
    std::vector<int8_t> hEBL_RK;   // (R, K) row-major  -- noisingA's EBL arg
    std::vector<float>  hAs(M), hBs(N);

    auto rand_a   = []() { return int8_t((std::rand() % 127) - 64); }; // [-64,62]
    auto rand_eal = []() { return int8_t((std::rand() % 63)  - 32); }; // [-32,30]
    for (auto& v : hA)   v = rand_a();
    for (auto& v : hB)   v = rand_a();
    for (auto& v : hEAL) v = rand_eal();
    for (auto& v : hEBR) v = rand_eal();

    make_sparse_KR(hEAR_KR, K, R);
    transpose_KR_to_RK(hEAR_KR.data(), hEAR_RK, K, R);
    make_sparse_KR(hEBL_KR, K, R);
    transpose_KR_to_RK(hEBL_KR.data(), hEBL_RK, K, R);

    // Small positive scales matching test_pearl_gemm.py defaults.
    for (auto& v : hAs) v = 0.005f + (std::rand() / float(RAND_MAX)) * 0.02f;
    for (auto& v : hBs) v = 0.005f + (std::rand() / float(RAND_MAX)) * 0.02f;

    // ---- Device allocation ------------------------------------------------
    int8_t  *dA, *dB, *dEAL, *dEBR;
    int8_t  *dEAR_KR, *dEAR_RK, *dEBL_KR, *dEBL_RK;
    int8_t  *dApEA, *dBpEB;
    int32_t *dAxEBL_i32, *dEARxBpEB_i32;
    cutlass::half_t *dAxEBL_fp16, *dEARxBpEB_fp16;
    cutlass::half_t *dEAL_fp16, *dEBR_fp16;
    float   *dAs, *dBs;
    cutlass::bfloat16_t *dC;

    CUCHK(cudaMalloc(&dA,            hA.size()));
    CUCHK(cudaMalloc(&dB,            hB.size()));
    CUCHK(cudaMalloc(&dEAL,          hEAL.size()));
    CUCHK(cudaMalloc(&dEBR,          hEBR.size()));
    CUCHK(cudaMalloc(&dEAR_KR,       hEAR_KR.size()));
    CUCHK(cudaMalloc(&dEAR_RK,       hEAR_RK.size()));
    CUCHK(cudaMalloc(&dEBL_KR,       hEBL_KR.size()));
    CUCHK(cudaMalloc(&dEBL_RK,       hEBL_RK.size()));
    CUCHK(cudaMalloc(&dApEA,         size_t(M) * K));
    CUCHK(cudaMalloc(&dBpEB,         size_t(N) * K));
    CUCHK(cudaMalloc(&dAxEBL_i32,    size_t(M) * R * sizeof(int32_t)));
    CUCHK(cudaMalloc(&dEARxBpEB_i32, size_t(N) * R * sizeof(int32_t)));
    CUCHK(cudaMalloc(&dAxEBL_fp16,   size_t(M) * R * 2));
    CUCHK(cudaMalloc(&dEARxBpEB_fp16, size_t(N) * R * 2));
    CUCHK(cudaMalloc(&dEAL_fp16,     size_t(M) * R * 2));
    CUCHK(cudaMalloc(&dEBR_fp16,     size_t(N) * R * 2));
    CUCHK(cudaMalloc(&dAs,           hAs.size() * 4));
    CUCHK(cudaMalloc(&dBs,           hBs.size() * 4));
    CUCHK(cudaMalloc(&dC,            size_t(M) * N * 2));

    // ---- H2D copies -------------------------------------------------------
    CUCHK(cudaMemcpy(dA,      hA.data(),      hA.size(),      cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dB,      hB.data(),      hB.size(),      cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEAL,    hEAL.data(),    hEAL.size(),    cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEBR,    hEBR.data(),    hEBR.size(),    cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEAR_KR, hEAR_KR.data(), hEAR_KR.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEAR_RK, hEAR_RK.data(), hEAR_RK.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEBL_KR, hEBL_KR.data(), hEBL_KR.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEBL_RK, hEBL_RK.data(), hEBL_RK.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dAs,     hAs.data(),     hAs.size() * 4, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dBs,     hBs.data(),     hBs.size() * 4, cudaMemcpyHostToDevice));
    CUCHK(cudaMemset(dApEA, 0xAA, size_t(M) * K));
    CUCHK(cudaMemset(dBpEB, 0xAA, size_t(N) * K));
    CUCHK(cudaMemset(dAxEBL_i32, 0xAA, size_t(M) * R * sizeof(int32_t)));
    CUCHK(cudaMemset(dEARxBpEB_i32, 0xAA, size_t(N) * R * sizeof(int32_t)));
    CUCHK(cudaMemset(dC, 0, size_t(M) * N * 2));

    // ---- 1. noisingA: produces ApEA, AxEBL_int32 --------------------------
    pearl_noisingA_sm89_64x64x64_R64_int32(
        dA, dEAL, dEAR_KR, dEBL_RK, dApEA, dAxEBL_i32, M, K, /*stream=*/0);
    CUCHK(cudaDeviceSynchronize());
    if (auto e = cudaGetLastError(); e != cudaSuccess) {
        fprintf(stderr, "noisingA launch error: %s\n", cudaGetErrorString(e));
        return 1;
    }

    // ---- 2. noisingB: produces BpEB, EARxBpEB_int32 -----------------------
    pearl::sm89::pearl_noisingB_sm89_64x64x64_R64_int32(
        dB, dEBR, dEBL_KR, dEAR_RK, dBpEB, dEARxBpEB_i32, N, K, /*stream=*/0);
    CUCHK(cudaDeviceSynchronize());
    if (auto e = cudaGetLastError(); e != cudaSuccess) {
        fprintf(stderr, "noisingB launch error: %s\n", cudaGetErrorString(e));
        return 1;
    }

    // ---- 3. Build denoise tensors -----------------------------------------
    // (a) AxEBL_fp16  = AxEBL_int32  * 2^-14   (CPU cast)
    // (b) EARxBpEB_fp16 = EARxBpEB_int32 * 2^-12   (CPU cast)
    // (c) EAL_fp16 = -1 * EAL   (CPU cast)
    // (d) EBR_fp16 = -4 * EBR   (CPU cast)
    std::vector<int32_t> hAxEBL_i32(size_t(M) * R);
    std::vector<int32_t> hEARxBpEB_i32(size_t(N) * R);
    CUCHK(cudaMemcpy(hAxEBL_i32.data(),    dAxEBL_i32,
                     hAxEBL_i32.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUCHK(cudaMemcpy(hEARxBpEB_i32.data(), dEARxBpEB_i32,
                     hEARxBpEB_i32.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));

    std::vector<uint16_t> hAxEBL_fp16(size_t(M) * R);
    std::vector<uint16_t> hEARxBpEB_fp16(size_t(N) * R);
    for (size_t i = 0; i < hAxEBL_i32.size(); ++i) {
        float v = float(hAxEBL_i32[i]) / float(kAxEBLScale);
        hAxEBL_fp16[i] = float_to_half(v);
    }
    for (size_t i = 0; i < hEARxBpEB_i32.size(); ++i) {
        float v = float(hEARxBpEB_i32[i]) / float(kEARxBpEBScale);
        hEARxBpEB_fp16[i] = float_to_half(v);
    }

    std::vector<uint16_t> hEAL_fp16(size_t(M) * R);
    std::vector<uint16_t> hEBR_fp16(size_t(N) * R);
    for (size_t i = 0; i < hEAL.size(); ++i) {
        hEAL_fp16[i] = float_to_half(float(kEAL_DenoiseScale) * float(hEAL[i]));
    }
    for (size_t i = 0; i < hEBR.size(); ++i) {
        hEBR_fp16[i] = float_to_half(float(kEBR_DenoiseScale) * float(hEBR[i]));
    }

    CUCHK(cudaMemcpy(dAxEBL_fp16,    hAxEBL_fp16.data(),    hAxEBL_fp16.size() * 2,
                     cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEARxBpEB_fp16, hEARxBpEB_fp16.data(), hEARxBpEB_fp16.size() * 2,
                     cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEAL_fp16,      hEAL_fp16.data(),      hEAL_fp16.size() * 2,
                     cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEBR_fp16,      hEBR_fp16.data(),      hEBR_fp16.size() * 2,
                     cudaMemcpyHostToDevice));

    // ---- 4. Main GEMM + denoise epilogue ----------------------------------
    pearl::sm89::pearl_gemm_sm89_denoise_128x128x64_R64(
        dApEA, /*lda=*/K, dBpEB, /*ldb=*/K, dC, /*ldc=*/N,
        dAs, dBs,
        dEAL_fp16, dEBR_fp16, dAxEBL_fp16, dEARxBpEB_fp16,
        M, N, K, /*stream=*/0);
    CUCHK(cudaDeviceSynchronize());
    if (auto e = cudaGetLastError(); e != cudaSuccess) {
        fprintf(stderr, "noisy_gemm launch error: %s\n", cudaGetErrorString(e));
        return 1;
    }

    // ---- 5. CPU reference + compare ---------------------------------------
    std::vector<uint16_t> hC(size_t(M) * N), hCref(size_t(M) * N);
    CUCHK(cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost));

    ref_gemm(M, N, K, hA.data(), hB.data(), hAs.data(), hBs.data(),
             hCref.data());

    int worst_idx = 0;
    float worst_abs = 0.f, worst_rel = 0.f;
    long bad = 0;
    for (size_t i = 0; i < hC.size(); ++i) {
        float a = bf16_to_float(hC[i]);
        float b = bf16_to_float(hCref[i]);
        float d = std::fabs(a - b);
        float r = d / (std::fabs(b) + 1e-6f);
        if (d > worst_abs) { worst_abs = d; worst_rel = r; worst_idx = int(i); }
        if (d > spec.atol + spec.rtol * std::fabs(b)) ++bad;
    }

    printf("M=%d N=%d K=%d seed=%u  max|err|=%.3e  rel=%.3e  bad=%ld/%zu",
           M, N, K, spec.seed, worst_abs, worst_rel, bad, hC.size());
    if (bad == 0) {
        printf("   PASS\n");
    } else {
        printf("   FAIL  worst@idx=%d ref=%.4f got=%.4f\n",
               worst_idx, bf16_to_float(hCref[worst_idx]),
               bf16_to_float(hC[worst_idx]));
        int shown = 0;
        for (size_t i = 0; i < hC.size() && shown < 8; ++i) {
            float a = bf16_to_float(hC[i]);
            float b = bf16_to_float(hCref[i]);
            float d = std::fabs(a - b);
            if (d > spec.atol + spec.rtol * std::fabs(b)) {
                long r = long(i) / N, c = long(i) % N;
                printf("    [%4ld,%4ld] ref=%.4f got=%.4f diff=%.4f\n",
                       r, c, b, a, a - b);
                ++shown;
            }
        }
    }

    cudaFree(dA);          cudaFree(dB);
    cudaFree(dEAL);        cudaFree(dEBR);
    cudaFree(dEAR_KR);     cudaFree(dEAR_RK);
    cudaFree(dEBL_KR);     cudaFree(dEBL_RK);
    cudaFree(dApEA);       cudaFree(dBpEB);
    cudaFree(dAxEBL_i32);  cudaFree(dEARxBpEB_i32);
    cudaFree(dAxEBL_fp16); cudaFree(dEARxBpEB_fp16);
    cudaFree(dEAL_fp16);   cudaFree(dEBR_fp16);
    cudaFree(dAs);         cudaFree(dBs);
    cudaFree(dC);
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
    // Primary acceptance case: matches test_pearl_gemm.py's atol/rtol.
    rc |= run_case({128, 128, 512, 0, /*atol=*/1e-1f, /*rtol=*/1e-2f});
    if (rc != 0) {
        printf("\n(stopping after primary case failure)\n");
        return rc;
    }
    // Larger cases (same tolerance).
    rc |= run_case({256, 256, 256, 1, /*atol=*/1e-1f, /*rtol=*/1e-2f});
    rc |= run_case({512, 512, 512, 2, /*atol=*/1e-1f, /*rtol=*/1e-2f});

    if (rc == 0) printf("\nALL PASS\n");
    else         printf("\nFAIL\n");
    return rc;
}
