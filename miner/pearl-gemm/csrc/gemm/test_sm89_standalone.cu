// Self-contained sm_89 noiseless GEMM correctness test (no PyTorch, no
// pip install — links pearl_gemm_sm89_inst.cu directly and runs on rig04).
//
// Build (on a machine with nvcc 12.x):
//   nvcc -gencode arch=compute_89,code=sm_89 -std=c++20 -O3
//        -I . -I ../../third_party/cutlass/include
//        -I ../../third_party/cutlass/tools/util/include
//        -I ../../third_party/cutlass/examples/common
//        --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG
//        pearl_gemm_sm89_inst.cu test_sm89_standalone.cu
//        -o /tmp/test_sm89_standalone
//
// Then ship the binary to rig04 and run.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
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
extern "C" void pearl_gemm_sm89_noiseless_128x128x64_R64(
    int8_t const* A, int64_t lda,
    int8_t const* B, int64_t ldb,
    cutlass::bfloat16_t* C, int64_t ldc,
    float const* A_scales,
    float const* B_scales,
    int M, int N, int K,
    cudaStream_t stream);
extern "C" void pearl_gemm_sm89_noiseless_128x256x64_R64(
    int8_t const* A, int64_t lda,
    int8_t const* B, int64_t ldb,
    cutlass::bfloat16_t* C, int64_t ldc,
    float const* A_scales,
    float const* B_scales,
    int M, int N, int K,
    cudaStream_t stream);
}
}

using NoiselessFn = void(*)(int8_t const*, int64_t, int8_t const*, int64_t,
                            cutlass::bfloat16_t*, int64_t, float const*, float const*,
                            int, int, int, cudaStream_t);

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

// Reference: C = (A @ B^T) cast to fp32, multiplied by row * col scales,
// cast to bf16. Done in fp64 on CPU for max precision.
static void ref_gemm(int M, int N, int K,
                     int8_t const* A, int8_t const* B,
                     float const* A_scales, float const* B_scales,
                     uint16_t* C_bf16) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += int32_t(A[i * K + k]) * int32_t(B[j * K + k]);
            }
            double v = double(acc) * double(A_scales[i]) * double(B_scales[j]);
            // round-to-nearest-even fp32 -> bf16
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

static float bf16_to_float(uint16_t bf) {
    uint32_t b = uint32_t(bf) << 16;
    float v;
    std::memcpy(&v, &b, 4);
    return v;
}

static int run_case(int M, int N, int K, unsigned seed,
                    NoiselessFn fn, char const* tag) {
    std::srand(seed);
    std::vector<int8_t>   hA(size_t(M) * K), hB(size_t(N) * K);
    std::vector<float>    hAs(M), hBs(N);
    std::vector<uint16_t> hC(size_t(M) * N), hCref(size_t(M) * N);

    for (auto& v : hA)  v = int8_t((std::rand() % 255) - 127);
    for (auto& v : hB)  v = int8_t((std::rand() % 255) - 127);
    for (auto& v : hAs) v = 0.005f + (std::rand() / float(RAND_MAX)) * 0.02f;
    for (auto& v : hBs) v = 0.005f + (std::rand() / float(RAND_MAX)) * 0.02f;

    int8_t *dA, *dB;
    float *dAs, *dBs;
    cutlass::bfloat16_t *dC;
    CUCHK(cudaMalloc(&dA, hA.size()));
    CUCHK(cudaMalloc(&dB, hB.size()));
    CUCHK(cudaMalloc(&dAs, hAs.size() * 4));
    CUCHK(cudaMalloc(&dBs, hBs.size() * 4));
    CUCHK(cudaMalloc(&dC, hC.size() * 2));

    CUCHK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dAs, hAs.data(), hAs.size() * 4, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dBs, hBs.data(), hBs.size() * 4, cudaMemcpyHostToDevice));
    CUCHK(cudaMemset(dC, 0, hC.size() * 2));

    fn(dA, /*lda=*/K, dB, /*ldb=*/K, dC, /*ldc=*/N,
       dAs, dBs, M, N, K, /*stream=*/0);
    CUCHK(cudaDeviceSynchronize());
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "kernel launch error: %s\n", cudaGetErrorString(launch_err));
        return 1;
    }

    CUCHK(cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost));
    ref_gemm(M, N, K, hA.data(), hB.data(), hAs.data(), hBs.data(), hCref.data());

    int worst_idx = 0;
    float worst_abs = 0.f, worst_rel = 0.f;
    long bad = 0;
    for (size_t i = 0; i < hC.size(); ++i) {
        float a = bf16_to_float(hC[i]);
        float b = bf16_to_float(hCref[i]);
        float d = std::fabs(a - b);
        float r = d / (std::fabs(b) + 1e-6f);
        if (d > worst_abs) { worst_abs = d; worst_rel = r; worst_idx = int(i); }
        // pearl-gemm tolerance: atol=1e-1, rtol=1e-2
        if (d > 1e-1f + 1e-2f * std::fabs(b)) ++bad;
    }

    printf("[%s] M=%d N=%d K=%d seed=%u   max|err|=%.3e  rel=%.3e  bad=%ld/%zu",
           tag, M, N, K, seed, worst_abs, worst_rel, bad, hC.size());
    if (bad == 0) {
        printf("   PASS\n");
    } else {
        printf("   FAIL  worst@idx=%d ref=%.4f got=%.4f\n",
               worst_idx, bf16_to_float(hCref[worst_idx]),
               bf16_to_float(hC[worst_idx]));
    }

    CUCHK(cudaFree(dA));
    CUCHK(cudaFree(dB));
    CUCHK(cudaFree(dAs));
    CUCHK(cudaFree(dBs));
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
    // bN=128 baseline path. Tile shape must divide M and N for the
    // Is_Even_M/Is_Even_N=true fast path the trampoline declares.
    NoiselessFn fn128 = pearl::sm89::pearl_gemm_sm89_noiseless_128x128x64_R64;
    rc |= run_case(128, 128, 128, 0, fn128, "bN=128");
    rc |= run_case(256, 256, 128, 1, fn128, "bN=128");
    rc |= run_case(512, 512, 512, 2, fn128, "bN=128");
    rc |= run_case(1024, 1024, 1024, 3, fn128, "bN=128");

    // bN=256 wider-N path. Tile is (128, 256) so N must be multiple of 256.
    NoiselessFn fn256 = pearl::sm89::pearl_gemm_sm89_noiseless_128x256x64_R64;
    rc |= run_case(128, 256, 128, 100, fn256, "bN=256");
    rc |= run_case(256, 256, 128, 101, fn256, "bN=256");
    rc |= run_case(512, 512, 512, 102, fn256, "bN=256");
    rc |= run_case(1024, 1024, 1024, 103, fn256, "bN=256");

    if (rc == 0) printf("\nALL PASS\n");
    else         printf("\nFAIL\n");
    return rc;
}
