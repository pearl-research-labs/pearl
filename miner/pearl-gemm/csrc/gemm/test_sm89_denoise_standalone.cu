// SPDX-License-Identifier: see LICENSE
//
// Self-contained sm_89 denoise-path GEMM correctness test.
//
// Mathematically the denoise epilogue computes (mirroring the Hopper version):
//
//   acc_fp32 = int32(A @ B^T)  (mainloop)
//   acc_fp32 *= 1 / 2^12
//   acc_fp32 += EAL    @ EARxBpEB^T   (fp16 MMA, fp32 accum)
//   acc_fp32 += AxEBL  @ EBR^T        (fp16 MMA, fp32 accum)
//   acc_fp32 *= 2^12
//   C = bf16(acc_fp32 * AScale_row * BScale_col)
//
// Test plan:
//   1. With all four denoise tensors == 0: result MUST match the noiseless
//      gemm bit-exactly (the 2^-12/2^12 pre/post-scale cancels exactly only at
//      large magnitudes; for any finite acc the round-trip introduces fp32
//      rounding error of <1 ULP per coordinate, which after the bf16 cast we
//      expect to be a no-op for all but the lowest-magnitude entries).
//   2. With random small non-zero denoise tensors: compare against a CPU
//      reference that performs the same fp32 math.

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

namespace pearl {
namespace sm89 {
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
}
}

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

static constexpr int R_DIM = 64;
static constexpr float kIntToFp16ScaleFactor = float(1 << 12);  // mirrors host constant

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

// CPU reference: replicates the Hopper denoise-path math in fp32.
//   acc = (A @ B^T) as int32 (cast to fp32 lossless for K up to ~16M)
//   acc /= 2^12
//   acc += (EAL @ EARxBpEB^T)  computed in fp32
//   acc += (AxEBL @ EBR^T)     computed in fp32
//   acc *= 2^12
//   C = bf16(acc * AScale_row * BScale_col)
static void ref_denoise_gemm(int M, int N, int K,
                             int8_t const* A, int8_t const* B,
                             float const* A_scales, float const* B_scales,
                             uint16_t const* EAL,        // fp16 bits, (M, R)
                             uint16_t const* EARxBpEB,   // fp16 bits, (N, R)
                             uint16_t const* AxEBL,      // fp16 bits, (M, R)
                             uint16_t const* EBR,        // fp16 bits, (N, R)
                             uint16_t* C_bf16) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // Mainloop int32 accum.
            int64_t int_acc = 0;
            for (int k = 0; k < K; ++k) {
                int_acc += int64_t(A[i * K + k]) * int64_t(B[j * K + k]);
            }
            float acc = float(int_acc);
            acc /= kIntToFp16ScaleFactor;

            // EAL @ EARxBpEB^T contribution.
            float eax = 0.f;
            for (int r = 0; r < R_DIM; ++r) {
                float a = half_to_float(EAL[i * R_DIM + r]);
                float b = half_to_float(EARxBpEB[j * R_DIM + r]);
                eax += a * b;
            }
            acc += eax;

            // AxEBL @ EBR^T contribution.
            float axeb = 0.f;
            for (int r = 0; r < R_DIM; ++r) {
                float a = half_to_float(AxEBL[i * R_DIM + r]);
                float b = half_to_float(EBR[j * R_DIM + r]);
                axeb += a * b;
            }
            acc += axeb;

            acc *= kIntToFp16ScaleFactor;

            // Apply scales then cast to bf16 (round-to-nearest-even).
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
    bool zero_denoise;  // if true, all four denoise tensors are zero
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
        std::fill(hEAL.begin(), hEAL.end(), uint16_t(0));
        std::fill(hEARxBpEB.begin(), hEARxBpEB.end(), uint16_t(0));
        std::fill(hAxEBL.begin(), hAxEBL.end(), uint16_t(0));
        std::fill(hEBR.begin(), hEBR.end(), uint16_t(0));
    } else {
        // Small random fp16 entries so the dot products stay well under fp16
        // overflow (max ~65504). Each dot product sums R_DIM (=64) terms, so
        // entries up to ~30 give safe products. We sample in [-0.1, 0.1].
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

    pearl::sm89::pearl_gemm_sm89_denoise_128x128x64_R64(
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
        // Dump a few mismatches for debugging.
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
    // First: zero denoise — should match noiseless path closely. We use the
    // production atol (matches test_sm89_standalone.cu) since the round-trip
    // through 2^-12/2^12 is lossless for the fp32 accumulator.
    rc |= run_case({128, 128, 128, 0, /*zero_denoise=*/true,  /*atol=*/1e-1f, /*rtol=*/1e-2f});
    rc |= run_case({256, 256, 128, 1, /*zero_denoise=*/true,  /*atol=*/1e-1f, /*rtol=*/1e-2f});
    if (rc != 0) {
        printf("\n(stopping after zero-denoise failure)\n");
        return rc;
    }

    // Next: random small fp16 denoise. The 3-chained-op path accumulates more
    // fp16 rounding than the pure noiseless path, so loosen the tolerance
    // per the spec.
    rc |= run_case({128, 128, 128, 10, /*zero_denoise=*/false, /*atol=*/5e-1f, /*rtol=*/2e-2f});
    rc |= run_case({256, 256, 128, 11, /*zero_denoise=*/false, /*atol=*/5e-1f, /*rtol=*/2e-2f});
    rc |= run_case({512, 512, 512, 12, /*zero_denoise=*/false, /*atol=*/5e-1f, /*rtol=*/2e-2f});
    rc |= run_case({1024, 1024, 1024, 13, /*zero_denoise=*/false, /*atol=*/5e-1f, /*rtol=*/2e-2f});

    if (rc == 0) printf("\nALL PASS\n");
    else         printf("\nFAIL\n");
    return rc;
}
