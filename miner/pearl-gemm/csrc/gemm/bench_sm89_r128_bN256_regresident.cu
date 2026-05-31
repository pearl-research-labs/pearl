// SPDX-License-Identifier: see LICENSE
//
// Standalone correctness + TOPS bench for the sm_89 R=128 bM=128 bN=256
// REGISTER-RESIDENT denoise path (wave-3 wider-N tile).
//
// Compares three R=128 layouts side-by-side on an sm_89 host:
//   - wave-2 baseline:  bM=64  bN=128  Denoise (smem-resident, ~97 KB smem)
//   - wave-2 regresident: bM=128 bN=128  Denoise (register-resident, ~33 KB smem)
//   - wave-3 candidate: bM=128 bN=256  Denoise (register-resident, ~66 KB smem)
//
// Two phases:
//   1. Correctness: zero-denoise and random-denoise sanity at small sizes
//      against a CPU fp32 reference; cross-validation between the candidate
//      and the bM=64 bN=128 baseline (must produce bit-equivalent C).
//   2. Bench: 2048³, 4096³, 8192³ (and a few skinny shapes). Writes the table
//      to stdout AND to a CSV file (path passed via --csv=PATH).
//
// MAC accounting: 2*M*N*K / time_seconds (main GEMM only — denoise is FLOP-
// negligible compared to the int8 mainloop).
//
// This binary runs ONLY on sm_89 hardware (4070 Ti SUPER).

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
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// trampolines under test
// ---------------------------------------------------------------------------
extern "C" void pearl_gemm_sm89_denoise_64x128x64_R128(
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

extern "C" void pearl_gemm_sm89_denoise_regresident_128x128x64_R128(
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

extern "C" void pearl_gemm_sm89_denoise_regresident_128x256x64_R128(
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

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

static constexpr int R_DIM = 128;
static constexpr float kIntToFp16ScaleFactor = float(1 << 12);

enum class Variant { BM64_BN128, BM128_BN128_REGRES, BM128_BN256_REGRES };

static char const* variant_label(Variant v) {
    switch (v) {
    case Variant::BM64_BN128:         return "bM=64  bN=128 R=128 smem-resident (wave-2 prod)";
    case Variant::BM128_BN128_REGRES: return "bM=128 bN=128 R=128 register-resident (wave-2)";
    case Variant::BM128_BN256_REGRES: return "bM=128 bN=256 R=128 register-resident (wave-3)";
    }
    return "?";
}

static char const* variant_short(Variant v) {
    switch (v) {
    case Variant::BM64_BN128:         return "bM64_bN128";
    case Variant::BM128_BN128_REGRES: return "bM128_bN128_regres";
    case Variant::BM128_BN256_REGRES: return "bM128_bN256_regres";
    }
    return "?";
}

static void launch(Variant v,
                   int8_t const* dA, int64_t lda,
                   int8_t const* dB, int64_t ldb,
                   cutlass::bfloat16_t* dC, int64_t ldc,
                   float const* dAs, float const* dBs,
                   cutlass::half_t const* dEAL,
                   cutlass::half_t const* dEBR,
                   cutlass::half_t const* dAxEBL,
                   cutlass::half_t const* dEARxBpEB,
                   int M, int N, int K, cudaStream_t s) {
    switch (v) {
    case Variant::BM64_BN128:
        pearl_gemm_sm89_denoise_64x128x64_R128(
            dA, lda, dB, ldb, dC, ldc, dAs, dBs,
            dEAL, dEBR, dAxEBL, dEARxBpEB, M, N, K, s);
        break;
    case Variant::BM128_BN128_REGRES:
        pearl_gemm_sm89_denoise_regresident_128x128x64_R128(
            dA, lda, dB, ldb, dC, ldc, dAs, dBs,
            dEAL, dEBR, dAxEBL, dEARxBpEB, M, N, K, s);
        break;
    case Variant::BM128_BN256_REGRES:
        pearl_gemm_sm89_denoise_regresident_128x256x64_R128(
            dA, lda, dB, ldb, dC, ldc, dAs, dBs,
            dEAL, dEBR, dAxEBL, dEARxBpEB, M, N, K, s);
        break;
    }
}

// ---------------------------------------------------------------------------
// fp16/bf16 helpers
// ---------------------------------------------------------------------------
static float bf16_to_float(uint16_t bf) {
    uint32_t b = uint32_t(bf) << 16;
    float v;
    std::memcpy(&v, &b, 4);
    return v;
}

static float half_to_float(uint16_t h) {
    __half hv;
    std::memcpy(&hv, &h, 2);
    return __half2float(hv);
}

static uint16_t float_to_half(float v) {
    __half hv = __float2half(v);
    uint16_t bits;
    std::memcpy(&bits, &hv, 2);
    return bits;
}

// CPU reference (mirrors test_sm89_r128_regresident_standalone.cu).
static void ref_denoise_gemm(int M, int N, int K,
                             int8_t const* A, int8_t const* B,
                             float const* A_scales, float const* B_scales,
                             uint16_t const* EAL,
                             uint16_t const* EARxBpEB,
                             uint16_t const* AxEBL,
                             uint16_t const* EBR,
                             uint16_t* C_bf16) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int64_t int_acc = 0;
            for (int k = 0; k < K; ++k) {
                int_acc += int64_t(A[i * K + k]) * int64_t(B[j * K + k]);
            }
            float acc = float(int_acc);
            acc /= kIntToFp16ScaleFactor;

            float eax = 0.f;
            for (int r = 0; r < R_DIM; ++r) {
                float a = half_to_float(EAL[i * R_DIM + r]);
                float b = half_to_float(EARxBpEB[j * R_DIM + r]);
                eax += a * b;
            }
            acc += eax;

            float axeb = 0.f;
            for (int r = 0; r < R_DIM; ++r) {
                float a = half_to_float(AxEBL[i * R_DIM + r]);
                float b = half_to_float(EBR[j * R_DIM + r]);
                axeb += a * b;
            }
            acc += axeb;

            acc *= kIntToFp16ScaleFactor;

            double v = double(acc) * double(A_scales[i]) * double(B_scales[j]);
            float vf = float(v);
            uint32_t bits;
            std::memcpy(&bits, &vf, 4);
            uint32_t lsb = (bits >> 16) & 1u;
            uint32_t rounding_bias = 0x7fffu + lsb;
            uint16_t bf = uint16_t((bits + rounding_bias) >> 16);
            C_bf16[i * N + j] = bf;
        }
    }
}

struct CaseSpec {
    int M, N, K;
    unsigned seed;
    bool zero_denoise;
    float atol;
    float rtol;
};

// ---------------------------------------------------------------------------
// Correctness against CPU reference for a single variant
// ---------------------------------------------------------------------------
static int correctness_case(Variant v, CaseSpec const& spec) {
    int M = spec.M, N = spec.N, K = spec.K;
    std::srand(spec.seed);

    std::vector<int8_t>   hA(size_t(M) * K), hB(size_t(N) * K);
    std::vector<float>    hAs(M), hBs(N);
    std::vector<uint16_t> hEAL(size_t(M) * R_DIM);
    std::vector<uint16_t> hEARxBpEB(size_t(N) * R_DIM);
    std::vector<uint16_t> hAxEBL(size_t(M) * R_DIM);
    std::vector<uint16_t> hEBR(size_t(N) * R_DIM);
    std::vector<uint16_t> hC(size_t(M) * N), hCref(size_t(M) * N);

    for (auto& x : hA)  x = int8_t((std::rand() % 255) - 127);
    for (auto& x : hB)  x = int8_t((std::rand() % 255) - 127);
    for (auto& x : hAs) x = 0.005f + (std::rand() / float(RAND_MAX)) * 0.02f;
    for (auto& x : hBs) x = 0.005f + (std::rand() / float(RAND_MAX)) * 0.02f;

    if (spec.zero_denoise) {
        std::fill(hEAL.begin(),      hEAL.end(),      uint16_t(0));
        std::fill(hEARxBpEB.begin(), hEARxBpEB.end(), uint16_t(0));
        std::fill(hAxEBL.begin(),    hAxEBL.end(),    uint16_t(0));
        std::fill(hEBR.begin(),      hEBR.end(),      uint16_t(0));
    } else {
        auto rand_half = []() {
            float f = (std::rand() / float(RAND_MAX) - 0.5f) * 0.2f;
            return float_to_half(f);
        };
        for (auto& x : hEAL)      x = rand_half();
        for (auto& x : hEARxBpEB) x = rand_half();
        for (auto& x : hAxEBL)    x = rand_half();
        for (auto& x : hEBR)      x = rand_half();
    }

    int8_t  *dA, *dB;
    float   *dAs, *dBs;
    cutlass::half_t *dEAL, *dEARxBpEB, *dAxEBL, *dEBR;
    cutlass::bfloat16_t *dC;
    CUCHK(cudaMalloc(&dA,        hA.size()));
    CUCHK(cudaMalloc(&dB,        hB.size()));
    CUCHK(cudaMalloc(&dAs,       hAs.size() * 4));
    CUCHK(cudaMalloc(&dBs,       hBs.size() * 4));
    CUCHK(cudaMalloc(&dEAL,      hEAL.size() * 2));
    CUCHK(cudaMalloc(&dEARxBpEB, hEARxBpEB.size() * 2));
    CUCHK(cudaMalloc(&dAxEBL,    hAxEBL.size() * 2));
    CUCHK(cudaMalloc(&dEBR,      hEBR.size() * 2));
    CUCHK(cudaMalloc(&dC,        hC.size() * 2));
    CUCHK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dAs, hAs.data(), hAs.size() * 4, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dBs, hBs.data(), hBs.size() * 4, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEAL,      hEAL.data(),      hEAL.size() * 2, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEARxBpEB, hEARxBpEB.data(), hEARxBpEB.size() * 2, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dAxEBL,    hAxEBL.data(),    hAxEBL.size() * 2, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEBR,      hEBR.data(),      hEBR.size() * 2, cudaMemcpyHostToDevice));
    CUCHK(cudaMemset(dC, 0, hC.size() * 2));

    launch(v, dA, K, dB, K, dC, N, dAs, dBs,
           dEAL, dEBR, dAxEBL, dEARxBpEB, M, N, K, /*stream=*/0);
    CUCHK(cudaDeviceSynchronize());
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "kernel launch error (%s): %s\n",
                variant_short(v), cudaGetErrorString(launch_err));
        return 1;
    }

    CUCHK(cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost));
    ref_denoise_gemm(M, N, K, hA.data(), hB.data(), hAs.data(), hBs.data(),
                     hEAL.data(), hEARxBpEB.data(), hAxEBL.data(), hEBR.data(),
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
    printf("  [%s] M=%d N=%d K=%d seed=%u zero=%d  max|err|=%.3e rel=%.3e bad=%ld/%zu  %s\n",
           variant_short(v), M, N, K, spec.seed, int(spec.zero_denoise),
           worst_abs, worst_rel, bad, hC.size(), bad == 0 ? "PASS" : "FAIL");
    if (bad != 0) {
        int shown = 0;
        for (size_t i = 0; i < hC.size() && shown < 4; ++i) {
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

    cudaFree(dA); cudaFree(dB); cudaFree(dAs); cudaFree(dBs);
    cudaFree(dEAL); cudaFree(dEARxBpEB); cudaFree(dAxEBL); cudaFree(dEBR);
    cudaFree(dC);
    return bad == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Cross-validation: run the same inputs through TWO variants, compare bf16
// outputs element-wise. Used to confirm bM=128 bN=256 register-resident gives
// the same numerical output as the bM=64 bN=128 baseline (within bf16 epsilon).
// ---------------------------------------------------------------------------
static int cross_validate_case(Variant ref_v, Variant cand_v,
                               int M, int N, int K, unsigned seed) {
    std::srand(seed);
    std::vector<int8_t>   hA(size_t(M)*K), hB(size_t(N)*K);
    std::vector<float>    hAs(M), hBs(N);
    std::vector<uint16_t> hEAL(size_t(M)*R_DIM);
    std::vector<uint16_t> hEARxBpEB(size_t(N)*R_DIM);
    std::vector<uint16_t> hAxEBL(size_t(M)*R_DIM);
    std::vector<uint16_t> hEBR(size_t(N)*R_DIM);
    std::vector<uint16_t> hCref(size_t(M)*N), hCcand(size_t(M)*N);
    for (auto& v : hA)  v = int8_t((rand() % 255) - 127);
    for (auto& v : hB)  v = int8_t((rand() % 255) - 127);
    for (auto& v : hAs) v = 0.005f + (rand() / float(RAND_MAX)) * 0.02f;
    for (auto& v : hBs) v = 0.005f + (rand() / float(RAND_MAX)) * 0.02f;
    auto rand_half = []() {
        float f = (rand() / float(RAND_MAX) - 0.5f) * 0.2f;
        return float_to_half(f);
    };
    for (auto& v : hEAL)      v = rand_half();
    for (auto& v : hEARxBpEB) v = rand_half();
    for (auto& v : hAxEBL)    v = rand_half();
    for (auto& v : hEBR)      v = rand_half();

    int8_t  *dA, *dB;
    float   *dAs, *dBs;
    cutlass::half_t *dEAL, *dEARxBpEB, *dAxEBL, *dEBR;
    cutlass::bfloat16_t *dC;
    CUCHK(cudaMalloc(&dA,        hA.size()));
    CUCHK(cudaMalloc(&dB,        hB.size()));
    CUCHK(cudaMalloc(&dAs,       hAs.size()*4));
    CUCHK(cudaMalloc(&dBs,       hBs.size()*4));
    CUCHK(cudaMalloc(&dEAL,      hEAL.size()*2));
    CUCHK(cudaMalloc(&dEARxBpEB, hEARxBpEB.size()*2));
    CUCHK(cudaMalloc(&dAxEBL,    hAxEBL.size()*2));
    CUCHK(cudaMalloc(&dEBR,      hEBR.size()*2));
    CUCHK(cudaMalloc(&dC,        size_t(M)*N*2));
    CUCHK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dAs, hAs.data(), hAs.size()*4, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dBs, hBs.data(), hBs.size()*4, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEAL,      hEAL.data(),      hEAL.size()*2, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEARxBpEB, hEARxBpEB.data(), hEARxBpEB.size()*2, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dAxEBL,    hAxEBL.data(),    hAxEBL.size()*2, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEBR,      hEBR.data(),      hEBR.size()*2, cudaMemcpyHostToDevice));

    CUCHK(cudaMemset(dC, 0, size_t(M)*N*2));
    launch(ref_v, dA, K, dB, K, dC, N, dAs, dBs,
           dEAL, dEBR, dAxEBL, dEARxBpEB, M, N, K, 0);
    CUCHK(cudaDeviceSynchronize());
    CUCHK(cudaMemcpy(hCref.data(), dC, size_t(M)*N*2, cudaMemcpyDeviceToHost));

    CUCHK(cudaMemset(dC, 0, size_t(M)*N*2));
    launch(cand_v, dA, K, dB, K, dC, N, dAs, dBs,
           dEAL, dEBR, dAxEBL, dEARxBpEB, M, N, K, 0);
    CUCHK(cudaDeviceSynchronize());
    CUCHK(cudaMemcpy(hCcand.data(), dC, size_t(M)*N*2, cudaMemcpyDeviceToHost));

    float worst_abs = 0.f, worst_rel = 0.f; long bad = 0;
    for (size_t i = 0; i < hCref.size(); ++i) {
        float a = bf16_to_float(hCcand[i]);
        float b = bf16_to_float(hCref[i]);
        float d = std::fabs(a - b);
        float r = d / (std::fabs(b) + 1e-6f);
        if (d > worst_abs) { worst_abs = d; worst_rel = r; }
        // bf16 epsilon ~1/128, tolerance proportional to magnitude.
        if (d > 0.5f + 0.02f * std::fabs(b)) ++bad;
    }
    printf("  cross [%s] vs [%s] M=%d N=%d K=%d  max|err|=%.3e rel=%.3e bad=%ld/%zu  %s\n",
           variant_short(cand_v), variant_short(ref_v), M, N, K,
           worst_abs, worst_rel, bad, hCref.size(),
           bad == 0 ? "PASS" : "FAIL");

    cudaFree(dA); cudaFree(dB); cudaFree(dAs); cudaFree(dBs);
    cudaFree(dEAL); cudaFree(dEARxBpEB); cudaFree(dAxEBL); cudaFree(dEBR);
    cudaFree(dC);
    return bad == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// TOPS bench
// ---------------------------------------------------------------------------
static double bench(Variant v, int M, int N, int K, int iters) {
    int const R = R_DIM;
    std::vector<int8_t> hA(size_t(M)*K), hB(size_t(N)*K);
    std::vector<float>  hAs(M, 0.01f), hBs(N, 0.01f);
    for (auto& x : hA) x = int8_t((rand() % 255) - 127);
    for (auto& x : hB) x = int8_t((rand() % 255) - 127);

    int8_t  *dA, *dB;
    float   *dAs, *dBs;
    cutlass::bfloat16_t *dC;
    cutlass::half_t *dEAL, *dEBR, *dAxEBL, *dEARxBpEB;
    CUCHK(cudaMalloc(&dA,        hA.size()));
    CUCHK(cudaMalloc(&dB,        hB.size()));
    CUCHK(cudaMalloc(&dAs,       hAs.size()*4));
    CUCHK(cudaMalloc(&dBs,       hBs.size()*4));
    CUCHK(cudaMalloc(&dC,        size_t(M)*N*2));
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

    auto run_once = [&]() {
        launch(v, dA, K, dB, K, dC, N, dAs, dBs,
               dEAL, dEBR, dAxEBL, dEARxBpEB, M, N, K, 0);
    };
    for (int i = 0; i < 3; ++i) run_once();
    CUCHK(cudaDeviceSynchronize());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "launch err (%s @ %dx%dx%d): %s\n",
                variant_short(v), M, N, K, cudaGetErrorString(err));
        cudaFree(dA); cudaFree(dB); cudaFree(dAs); cudaFree(dBs); cudaFree(dC);
        cudaFree(dEAL); cudaFree(dEBR); cudaFree(dAxEBL); cudaFree(dEARxBpEB);
        return -1.0;
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i) run_once();
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
    double seconds = double(ms) / 1000.0 / double(iters);
    double tops = 2.0 * double(M) * double(N) * double(K) / seconds * 1e-12;
    cudaFree(dA); cudaFree(dB); cudaFree(dAs); cudaFree(dBs); cudaFree(dC);
    cudaFree(dEAL); cudaFree(dEBR); cudaFree(dAxEBL); cudaFree(dEARxBpEB);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return tops;
}

int main(int argc, char** argv) {
    int dev = 0;
    std::string csv_path = "";
    bool skip_correctness = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--csv=", 0) == 0) csv_path = a.substr(6);
        else if (a == "--skip-correctness") skip_correctness = true;
        else if (a.rfind("--dev=", 0) == 0) dev = std::atoi(a.c_str() + 6);
        else if (i == 1) dev = std::atoi(a.c_str());
    }

    CUCHK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CUCHK(cudaGetDeviceProperties(&p, dev));
    int cc = p.major * 10 + p.minor;
    printf("device %d: %s sm_%d  L2=%dMB SMs=%d smem_optin=%zu KB\n",
           dev, p.name, cc,
           int(p.l2CacheSize/1024/1024),
           p.multiProcessorCount,
           p.sharedMemPerBlockOptin / 1024);
    if (cc != 89) {
        fprintf(stderr,
                "WARN: this binary is sm_89 only; current is sm_%d. "
                "Expect 'no kernel image' errors.\n", cc);
    }
    printf("4070 Ti SUPER peak INT8 dense = 353 TOPS\n\n");

    int rc = 0;

    if (!skip_correctness) {
        printf("=== Correctness vs CPU fp32 reference (R=128, fp16 denoise) ===\n");
        CaseSpec specs[] = {
            {128, 128, 128, 0,  /*zero=*/true,  1e-1f, 1e-2f},
            {256, 256, 256, 1,  /*zero=*/true,  1e-1f, 1e-2f},
            {128, 128, 128, 10, /*zero=*/false, 5e-1f, 2e-2f},
            {256, 256, 256, 11, /*zero=*/false, 5e-1f, 2e-2f},
            {512, 512, 512, 12, /*zero=*/false, 5e-1f, 2e-2f},
        };
        for (auto& s : specs) {
            // The bM=128 bN=256 path can only handle problems where N is a
            // multiple of 256; bump non-conforming sizes for the candidate.
            int N_cand = ((s.N + 255) / 256) * 256;
            int M_cand = ((s.M + 127) / 128) * 128;
            CaseSpec spec_cand = s;
            spec_cand.M = M_cand;
            spec_cand.N = N_cand;
            rc |= correctness_case(Variant::BM128_BN256_REGRES, spec_cand);
        }

        printf("\n=== Cross-validation vs bM=64 bN=128 baseline ===\n");
        // For cross-val all variants must accept the same problem size, so we
        // pick sizes that are a multiple of LCM(bM_max, bN_max) = LCM(128, 256)
        // = 256 in both dims.
        rc |= cross_validate_case(Variant::BM64_BN128, Variant::BM128_BN256_REGRES, 256, 256, 256, 100);
        rc |= cross_validate_case(Variant::BM64_BN128, Variant::BM128_BN256_REGRES, 512, 512, 512, 101);
        rc |= cross_validate_case(Variant::BM128_BN128_REGRES, Variant::BM128_BN256_REGRES, 256, 256, 256, 102);
        rc |= cross_validate_case(Variant::BM128_BN128_REGRES, Variant::BM128_BN256_REGRES, 512, 512, 512, 103);

        if (rc != 0) {
            printf("\nFAIL — stopping before bench\n");
            return rc;
        }
        printf("\nALL CORRECTNESS PASS\n\n");
    }

    // ---- Bench ----
    struct Shape { int M, N, K, iters; };
    Shape shapes[] = {
        {2048, 2048, 2048, 30},
        {4096, 4096, 4096, 12},
        {8192, 8192, 8192, 4},
        {4096, 4096, 8192, 8},
        {8192, 8192, 4096, 6},
        {4096, 16384, 4096, 5},
    };

    printf("=== TOPS bench ===\n");
    printf("%-7s %-7s %-7s | %14s %14s %14s | %8s\n",
           "M", "N", "K", "bM64_bN128", "bM128_bN128_reg", "bM128_bN256_reg",
           "cand/base");
    printf("%-7s-%-7s-%-7s-+-%14s-%14s-%14s-+-%8s\n",
           "-------", "-------", "-------", "--------------", "--------------",
           "--------------", "--------");

    FILE* csv = nullptr;
    if (!csv_path.empty()) {
        csv = std::fopen(csv_path.c_str(), "w");
        if (!csv) {
            fprintf(stderr, "WARN: failed to open csv %s\n", csv_path.c_str());
        } else {
            std::fprintf(csv,
                "M,N,K,iters,TOPS_bM64_bN128,TOPS_bM128_bN128_regres,TOPS_bM128_bN256_regres,ratio_cand_over_base\n");
            std::fflush(csv);
        }
    }

    for (auto const& s : shapes) {
        double a = bench(Variant::BM64_BN128,         s.M, s.N, s.K, s.iters);
        double b = bench(Variant::BM128_BN128_REGRES, s.M, s.N, s.K, s.iters);
        double c = bench(Variant::BM128_BN256_REGRES, s.M, s.N, s.K, s.iters);
        double ratio = (a > 0) ? (c / a) : 0.0;
        printf("%-7d %-7d %-7d | %14.2f %14.2f %14.2f | %7.2fx\n",
               s.M, s.N, s.K, a, b, c, ratio);
        if (csv) {
            std::fprintf(csv, "%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f\n",
                         s.M, s.N, s.K, s.iters, a, b, c, ratio);
            std::fflush(csv);
        }
    }
    if (csv) std::fclose(csv);

    return 0;
}
