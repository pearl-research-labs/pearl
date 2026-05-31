// SPDX-License-Identifier: see LICENSE
//
// A/B bench: bN=128 (production) vs bN=256 (new wider-N variant) for both the
// noiseless and the denoise paths. Companion to _bench_ab.cu, but the A/B axis
// here is the *tile shape*, not the *tile scheduler*.
//
// MAC accounting matches _bench_ab.cu: "main TOPS" = 2*M*N*K / time_seconds.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---- forward decls (both tile shapes) ----
namespace pearl { namespace sm89 {
extern "C" void pearl_gemm_sm89_noiseless_128x128x64_R64(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    int, int, int, cudaStream_t);
extern "C" void pearl_gemm_sm89_noiseless_128x256x64_R64(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    int, int, int, cudaStream_t);
extern "C" void pearl_gemm_sm89_denoise_128x128x64_R64(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    cutlass::half_t const*, cutlass::half_t const*,
    cutlass::half_t const*, cutlass::half_t const*,
    int, int, int, cudaStream_t);
extern "C" void pearl_gemm_sm89_denoise_128x256x64_R64(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    cutlass::half_t const*, cutlass::half_t const*,
    cutlass::half_t const*, cutlass::half_t const*,
    int, int, int, cudaStream_t);
}}

extern "C" void pearl_noisingA_sm89_64x64x64_R64_int32(
    int8_t const*, int8_t const*, int8_t const*, int8_t const*,
    int8_t*, int32_t*, int, int, cudaStream_t);
namespace pearl { namespace sm89 {
extern "C" void pearl_noisingB_sm89_64x64x64_R64_int32(
    int8_t const*, int8_t const*, int8_t const*, int8_t const*,
    int8_t*, int32_t*, int, int, cudaStream_t);
}}

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

using NoiselessFn = void(*)(int8_t const*, int64_t, int8_t const*, int64_t,
                            cutlass::bfloat16_t*, int64_t, float const*, float const*,
                            int, int, int, cudaStream_t);
using DenoiseFn   = void(*)(int8_t const*, int64_t, int8_t const*, int64_t,
                            cutlass::bfloat16_t*, int64_t, float const*, float const*,
                            cutlass::half_t const*, cutlass::half_t const*,
                            cutlass::half_t const*, cutlass::half_t const*,
                            int, int, int, cudaStream_t);

static double bench_noiseless(NoiselessFn fn, int M, int N, int K, int iters) {
    std::vector<int8_t> hA(size_t(M)*K), hB(size_t(N)*K);
    std::vector<float>  hAs(M, 0.01f), hBs(N, 0.01f);
    for (auto& v : hA) v = int8_t((rand() % 255) - 127);
    for (auto& v : hB) v = int8_t((rand() % 255) - 127);

    int8_t *dA, *dB; float *dAs, *dBs; cutlass::bfloat16_t *dC;
    CUCHK(cudaMalloc(&dA, hA.size()));   CUCHK(cudaMalloc(&dB, hB.size()));
    CUCHK(cudaMalloc(&dAs, M*4));        CUCHK(cudaMalloc(&dBs, N*4));
    CUCHK(cudaMalloc(&dC, size_t(M)*N*2));
    CUCHK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dAs, hAs.data(), M*4, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dBs, hBs.data(), N*4, cudaMemcpyHostToDevice));
    for (int i = 0; i < 3; ++i) fn(dA, K, dB, K, dC, N, dAs, dBs, M, N, K, 0);
    CUCHK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i) fn(dA, K, dB, K, dC, N, dAs, dBs, M, N, K, 0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
    double tops = 2.0 * double(M)*double(N)*double(K) / (double(ms)/1000.0/iters) * 1e-12;
    cudaFree(dA); cudaFree(dB); cudaFree(dAs); cudaFree(dBs); cudaFree(dC);
    return tops;
}

static double bench_e2e_denoise(DenoiseFn fn, int M, int N, int K, int iters) {
    int const R = 64;
    int8_t  *dA, *dB, *dEAL, *dEBR, *dEAR_R, *dEBL_R, *dEAR_K, *dEBL_K;
    int8_t  *dApEA, *dBpEB;
    int32_t *dAxEBL_i32, *dEARxBpEB_i32;
    cutlass::half_t *dEAL_fp16, *dEBR_fp16, *dAxEBL_fp16, *dEARxBpEB_fp16;
    float *dAs, *dBs; cutlass::bfloat16_t *dC;
    CUCHK(cudaMalloc(&dA, size_t(M)*K));   CUCHK(cudaMalloc(&dB, size_t(N)*K));
    CUCHK(cudaMalloc(&dEAL, size_t(M)*R)); CUCHK(cudaMalloc(&dEBR, size_t(N)*R));
    CUCHK(cudaMalloc(&dEAR_R, size_t(K)*R)); CUCHK(cudaMalloc(&dEBL_R, size_t(K)*R));
    CUCHK(cudaMalloc(&dEAR_K, size_t(R)*K)); CUCHK(cudaMalloc(&dEBL_K, size_t(R)*K));
    CUCHK(cudaMalloc(&dApEA, size_t(M)*K)); CUCHK(cudaMalloc(&dBpEB, size_t(N)*K));
    CUCHK(cudaMalloc(&dAxEBL_i32,    size_t(M)*R*4));
    CUCHK(cudaMalloc(&dEARxBpEB_i32, size_t(N)*R*4));
    CUCHK(cudaMalloc(&dEAL_fp16,     size_t(M)*R*2));
    CUCHK(cudaMalloc(&dEBR_fp16,     size_t(N)*R*2));
    CUCHK(cudaMalloc(&dAxEBL_fp16,   size_t(M)*R*2));
    CUCHK(cudaMalloc(&dEARxBpEB_fp16,size_t(N)*R*2));
    CUCHK(cudaMalloc(&dAs, M*4));   CUCHK(cudaMalloc(&dBs, N*4));
    CUCHK(cudaMalloc(&dC, size_t(M)*N*2));

    auto run_iter = [&]() {
        pearl_noisingA_sm89_64x64x64_R64_int32(dA, dEAL, dEAR_R, dEBL_K, dApEA, dAxEBL_i32, M, K, 0);
        pearl::sm89::pearl_noisingB_sm89_64x64x64_R64_int32(dB, dEBR, dEBL_R, dEAR_K, dBpEB, dEARxBpEB_i32, N, K, 0);
        fn(dApEA, K, dBpEB, K, dC, N, dAs, dBs,
           dEAL_fp16, dEBR_fp16, dAxEBL_fp16, dEARxBpEB_fp16, M, N, K, 0);
    };
    for (int i = 0; i < 3; ++i) run_iter();
    CUCHK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i) run_iter();
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
    double tops_main = 2.0 * double(M)*double(N)*double(K) / (double(ms)/1000.0/iters) * 1e-12;
    cudaFree(dA); cudaFree(dB); cudaFree(dEAL); cudaFree(dEBR);
    cudaFree(dEAR_R); cudaFree(dEBL_R); cudaFree(dEAR_K); cudaFree(dEBL_K);
    cudaFree(dApEA); cudaFree(dBpEB);
    cudaFree(dAxEBL_i32); cudaFree(dEARxBpEB_i32);
    cudaFree(dEAL_fp16); cudaFree(dEBR_fp16); cudaFree(dAxEBL_fp16); cudaFree(dEARxBpEB_fp16);
    cudaFree(dAs); cudaFree(dBs); cudaFree(dC);
    return tops_main;
}

int main(int argc, char** argv) {
    int dev = (argc >= 2) ? std::atoi(argv[1]) : 0;
    CUCHK(cudaSetDevice(dev));
    cudaDeviceProp p; CUCHK(cudaGetDeviceProperties(&p, dev));
    printf("device %d: %s sm_%d%d  L2=%dMB SMs=%d\n", dev, p.name, p.major, p.minor,
           int(p.l2CacheSize/1024/1024), p.multiProcessorCount);
    printf("4070 Ti SUPER peak INT8 dense = 353 TOPS\n");

    int sizes[][3] = {
      { 1024,  1024, 1024}, { 2048,  2048, 2048}, { 4096,  4096, 4096},
      { 4096,  4096, 8192}, { 8192,  8192, 4096}, { 4096, 16384, 4096},
    };
    int iters[]   = { 50, 30, 10, 10, 5, 5 };

    printf("\n%-32s  | %-10s | %-10s | %-7s\n", "noiseless GEMM TOPS", "bN=128", "bN=256", "ratio");
    printf("%-32s  +-%-10s-+-%-10s-+-%-7s\n", "--------------------------------", "----------", "----------", "-------");
    for (size_t i = 0; i < sizeof(iters)/sizeof(iters[0]); ++i) {
        int M = sizes[i][0], N = sizes[i][1], K = sizes[i][2];
        double a = bench_noiseless(pearl::sm89::pearl_gemm_sm89_noiseless_128x128x64_R64, M, N, K, iters[i]);
        double b = bench_noiseless(pearl::sm89::pearl_gemm_sm89_noiseless_128x256x64_R64, M, N, K, iters[i]);
        printf("M=%5d N=%5d K=%5d         | %10.2f | %10.2f | %.2fx\n", M, N, K, a, b, b / a);
    }

    int eiters[]  = { 20, 10, 5, 5, 3, 3 };
    printf("\n%-32s  | %-10s | %-10s | %-7s\n", "denoise main GEMM TOPS", "bN=128", "bN=256", "ratio");
    printf("%-32s  +-%-10s-+-%-10s-+-%-7s\n", "--------------------------------", "----------", "----------", "-------");
    for (size_t i = 0; i < sizeof(eiters)/sizeof(eiters[0]); ++i) {
        int M = sizes[i][0], N = sizes[i][1], K = sizes[i][2];
        double a = bench_e2e_denoise(pearl::sm89::pearl_gemm_sm89_denoise_128x128x64_R64, M, N, K, eiters[i]);
        double b = bench_e2e_denoise(pearl::sm89::pearl_gemm_sm89_denoise_128x256x64_R64, M, N, K, eiters[i]);
        printf("M=%5d N=%5d K=%5d         | %10.2f | %10.2f | %.2fx\n", M, N, K, a, b, b / a);
    }
    return 0;
}
