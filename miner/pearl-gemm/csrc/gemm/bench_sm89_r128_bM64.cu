// SPDX-License-Identifier: see LICENSE
//
// Standalone TOPS bench for the new sm_89 R=128 bM=64 path.
// Compares two layouts side-by-side at 2048³ and 4096³:
//   - existing baseline:  R=128 bM=64  bN=64  Denoise (SkipDenoising=false)
//   - new (this session): R=128 bM=64  bN=128 Denoise
//   - additional ref:     R=128 bM=64  bN=128 Noiseless (SkipDenoising=true)
//
// Build via _build_r128_bench.sh — runs only on real sm_89 hardware (4070 Ti
// SUPER). This program will crash with "no kernel image" on Hopper/Blackwell.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cutlass/numeric_types.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace pearl {
namespace sm89 {
// Existing R=128 (bM=bN=64) — from pearl_gemm_sm89_denoise_inst.cu
extern "C" void pearl_gemm_sm89_denoise_128x128x64_R64(  // R=64 ref
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    cutlass::half_t const*, cutlass::half_t const*,
    cutlass::half_t const*, cutlass::half_t const*,
    int, int, int, cudaStream_t);
// Existing R=128 (bM=bN=64) — not exported via extern "C", we use the new one only
// New (this session) R=128 bM=64 bN=128 — from pearl_gemm_sm89_r128_bM64_inst.cu
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
}
}

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

enum class Variant { R64_DENOISE, R128_BN128_NOISELESS, R128_BN128_DENOISE };

static double bench(Variant v, int M, int N, int K, int iters,
                    char const* label) {
    int const R = (v == Variant::R64_DENOISE) ? 64 : 128;
    (void)R;
    std::vector<int8_t> hA(size_t(M)*K), hB(size_t(N)*K);
    std::vector<float>  hAs(M, 0.01f), hBs(N, 0.01f);
    for (auto& v_ : hA) v_ = int8_t((rand() % 255) - 127);
    for (auto& v_ : hB) v_ = int8_t((rand() % 255) - 127);

    int8_t  *dA, *dB;
    float   *dAs, *dBs;
    cutlass::bfloat16_t *dC;
    cutlass::half_t *dEAL, *dEBR, *dAxEBL, *dEARxBpEB;
    int const R_used = R;
    CUCHK(cudaMalloc(&dA,  hA.size()));
    CUCHK(cudaMalloc(&dB,  hB.size()));
    CUCHK(cudaMalloc(&dAs, hAs.size()*4));
    CUCHK(cudaMalloc(&dBs, hBs.size()*4));
    CUCHK(cudaMalloc(&dC,  size_t(M)*N*2));
    CUCHK(cudaMalloc(&dEAL,      size_t(M)*R_used*2));
    CUCHK(cudaMalloc(&dEBR,      size_t(N)*R_used*2));
    CUCHK(cudaMalloc(&dAxEBL,    size_t(M)*R_used*2));
    CUCHK(cudaMalloc(&dEARxBpEB, size_t(N)*R_used*2));
    CUCHK(cudaMemset(dEAL,      0, size_t(M)*R_used*2));
    CUCHK(cudaMemset(dEBR,      0, size_t(N)*R_used*2));
    CUCHK(cudaMemset(dAxEBL,    0, size_t(M)*R_used*2));
    CUCHK(cudaMemset(dEARxBpEB, 0, size_t(N)*R_used*2));
    CUCHK(cudaMemcpy(dA,  hA.data(),  hA.size(),  cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dB,  hB.data(),  hB.size(),  cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dAs, hAs.data(), hAs.size()*4, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dBs, hBs.data(), hBs.size()*4, cudaMemcpyHostToDevice));

    auto run_once = [&]() {
        switch (v) {
        case Variant::R64_DENOISE:
            pearl::sm89::pearl_gemm_sm89_denoise_128x128x64_R64(
                dA, K, dB, K, dC, N, dAs, dBs,
                dEAL, dEBR, dAxEBL, dEARxBpEB, M, N, K, 0);
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
        }
    };
    for (int i = 0; i < 3; ++i) run_once();
    CUCHK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i) run_once();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);

    double seconds = double(ms) / 1000.0 / double(iters);
    double tops = 2.0 * double(M) * double(N) * double(K) / seconds * 1e-12;
    printf("  %-40s  M=%5d N=%5d K=%5d   %.3f ms   %6.2f TOPS\n",
           label, M, N, K, ms / iters, tops);

    cudaFree(dA); cudaFree(dB); cudaFree(dAs); cudaFree(dBs); cudaFree(dC);
    cudaFree(dEAL); cudaFree(dEBR); cudaFree(dAxEBL); cudaFree(dEARxBpEB);
    return tops;
}

int main(int argc, char** argv) {
    int dev = (argc >= 2) ? std::atoi(argv[1]) : 0;
    CUCHK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CUCHK(cudaGetDeviceProperties(&p, dev));
    printf("device %d: %s sm_%d%d smem_optin=%zu KB\n",
           dev, p.name, p.major, p.minor,
           p.sharedMemPerBlockOptin / 1024);
    if (!(p.major == 8 && p.minor == 9)) {
        printf("WARNING: this bench is sm_89 only; current device is sm_%d%d. "
               "Expect 'no kernel image' errors.\n", p.major, p.minor);
    }
    printf("Reference: 4070 Ti SUPER peak INT8 dense = 353 TOPS\n\n");

    int const SIZES[][3] = {{2048,2048,2048}, {4096,4096,4096}};
    for (auto const& s : SIZES) {
        int M=s[0], N=s[1], K=s[2];
        printf("=== %dx%dx%d ===\n", M, N, K);
        bench(Variant::R64_DENOISE,         M, N, K,  20, "R=64  bM=128 bN=128 Denoise (prod R=64)");
        bench(Variant::R128_BN128_NOISELESS, M, N, K, 20, "R=128 bM=64  bN=128 Noiseless (new)");
        bench(Variant::R128_BN128_DENOISE,   M, N, K, 20, "R=128 bM=64  bN=128 Denoise   (new)");
        printf("\n");
    }
    return 0;
}
