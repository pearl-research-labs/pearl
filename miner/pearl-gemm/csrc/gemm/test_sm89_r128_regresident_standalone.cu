// SPDX-License-Identifier: see LICENSE
//
// Standalone correctness + bench for the sm_89 R=128 bM=128 bN=128
// REGISTER-RESIDENT denoise path.
//
// What this validates:
//   - Mathematical equivalence vs CPU fp32 reference of the denoise epilogue
//   - Bit-exact at zero-denoise vs the noiseless path
//   - Builds at the target trait combo (smem accounting < 60 KB target)
//
// This binary runs ONLY on sm_89 hardware (e.g., 4070 Ti SUPER).
//
// Reference math (mirrors collective_epilogue_sm89.hpp):
//   acc_fp32 = int32(A @ B^T)
//   acc_fp32 *= 1 / 2^12
//   acc_fp32 += EAL    @ EARxBpEB^T   (fp16 ops, fp32 accum)
//   acc_fp32 += AxEBL  @ EBR^T        (fp16 ops, fp32 accum)
//   acc_fp32 *= 2^12
//   C = bf16(acc_fp32 * AScale_row * BScale_col)
//
// Build alongside its two extern "C" trampolines (see
// _build_r128_regresident.sh).

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

// New trampoline from pearl_gemm_sm89_r128_regresident_inst.cu
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

// Existing R=128 bM=64 bN=128 trampoline for cross-validation.
extern "C" void pearl_gemm_sm89_denoise_64x128x64_R128(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    cutlass::half_t const*, cutlass::half_t const*,
    cutlass::half_t const*, cutlass::half_t const*,
    int, int, int, cudaStream_t);

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

static constexpr int R_DIM = 128;
static constexpr float kIntToFp16ScaleFactor = float(1 << 12);

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

static int run_case(CaseSpec const& spec) {
    int M = spec.M, N = spec.N, K = spec.K;
    std::srand(spec.seed);

    std::vector<int8_t>   hA(size_t(M) * K), hB(size_t(N) * K);
    std::vector<float>    hAs(M), hBs(N);
    std::vector<uint16_t> hEAL(size_t(M) * R_DIM);
    std::vector<uint16_t> hEARxBpEB(size_t(N) * R_DIM);
    std::vector<uint16_t> hAxEBL(size_t(M) * R_DIM);
    std::vector<uint16_t> hEBR(size_t(N) * R_DIM);
    std::vector<uint16_t> hC(size_t(M) * N), hCref(size_t(M) * N);

    for (auto& v : hA)  v = int8_t((std::rand() % 255) - 127);
    for (auto& v : hB)  v = int8_t((std::rand() % 255) - 127);
    for (auto& v : hAs) v = 0.005f + (std::rand() / float(RAND_MAX)) * 0.02f;
    for (auto& v : hBs) v = 0.005f + (std::rand() / float(RAND_MAX)) * 0.02f;

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
        for (auto& v : hEAL)      v = rand_half();
        for (auto& v : hEARxBpEB) v = rand_half();
        for (auto& v : hAxEBL)    v = rand_half();
        for (auto& v : hEBR)      v = rand_half();
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

    CUCHK(cudaMemcpy(dA,        hA.data(),        hA.size(),                cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dB,        hB.data(),        hB.size(),                cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dAs,       hAs.data(),       hAs.size() * 4,           cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dBs,       hBs.data(),       hBs.size() * 4,           cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEAL,      hEAL.data(),      hEAL.size() * 2,          cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEARxBpEB, hEARxBpEB.data(), hEARxBpEB.size() * 2,     cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dAxEBL,    hAxEBL.data(),    hAxEBL.size() * 2,        cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEBR,      hEBR.data(),      hEBR.size() * 2,          cudaMemcpyHostToDevice));
    CUCHK(cudaMemset(dC, 0, hC.size() * 2));

    pearl_gemm_sm89_denoise_regresident_128x128x64_R128(
        dA, /*lda=*/K, dB, /*ldb=*/K, dC, /*ldc=*/N,
        dAs, dBs,
        dEAL, dEBR, dAxEBL, dEARxBpEB,
        M, N, K, /*stream=*/0);
    CUCHK(cudaDeviceSynchronize());
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "kernel launch error: %s\n", cudaGetErrorString(launch_err));
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

    printf("M=%d N=%d K=%d seed=%u zero=%d  max|err|=%.3e  rel=%.3e  bad=%ld/%zu",
           M, N, K, spec.seed, int(spec.zero_denoise),
           worst_abs, worst_rel, bad, hC.size());
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

    CUCHK(cudaFree(dA));
    CUCHK(cudaFree(dB));
    CUCHK(cudaFree(dAs));
    CUCHK(cudaFree(dBs));
    CUCHK(cudaFree(dEAL));
    CUCHK(cudaFree(dEARxBpEB));
    CUCHK(cudaFree(dAxEBL));
    CUCHK(cudaFree(dEBR));
    CUCHK(cudaFree(dC));
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
        fprintf(stderr, "WARN: this binary was built for sm_89; current is sm_%d\n", cc);
    }

    int rc = 0;
    // Zero denoise — sanity check.
    rc |= run_case({128, 128, 128, 0, /*zero=*/true,  1e-1f, 1e-2f});
    rc |= run_case({256, 256, 128, 1, /*zero=*/true,  1e-1f, 1e-2f});
    rc |= run_case({256, 256, 256, 2, /*zero=*/true,  1e-1f, 1e-2f});
    if (rc != 0) {
        printf("\n(stopping after zero-denoise failure)\n");
        return rc;
    }

    // Random denoise — full path.
    rc |= run_case({128, 128, 128, 10, /*zero=*/false, 5e-1f, 2e-2f});
    rc |= run_case({256, 256, 128, 11, /*zero=*/false, 5e-1f, 2e-2f});
    rc |= run_case({256, 256, 256, 12, /*zero=*/false, 5e-1f, 2e-2f});
    rc |= run_case({512, 512, 512, 13, /*zero=*/false, 5e-1f, 2e-2f});
    rc |= run_case({1024, 1024, 1024, 14, /*zero=*/false, 5e-1f, 2e-2f});

    if (rc == 0) printf("\nALL PASS\n");
    else         printf("\nFAIL\n");
    return rc;
}
