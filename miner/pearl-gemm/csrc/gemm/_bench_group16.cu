// SPDX-License-Identifier: see LICENSE
//
// Single-shape, env-var-controlled bench for the group-of-16 swizzle A/B.
//
// CLI: ./bench_group16 <device> <M> <N> <K> <iters>
// Reads PEARL_SM89_GROUP16_SWIZZLE (and other PEARL_SM89_* envs) from the
// environment — those are baked into the launcher's static-cached path on
// first call, so each process invocation gets a clean env read.
//
// Reports MEDIAN over `iters` per-iter times (cudaEvent), plus min/max +
// main_gemm TOPS = 2*M*N*K / median.
//
// Two-process driver script runs this once with the env unset (adaptive) and
// once with the env set (group-of-16) to compare clean.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void pearl_noisingA_sm89_64x64x64_R64_int32(
    int8_t const* A, int8_t const* EAL, int8_t const* EAR, int8_t const* EBL,
    int8_t* ApEA, int32_t* AxEBL, int M, int K, cudaStream_t stream);

namespace pearl { namespace sm89 {
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
}}  // namespace pearl::sm89

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

static constexpr int R_DIM = 64;

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <dev> <M> <N> <K> <iters>\n", argv[0]);
        return 1;
    }
    int dev   = std::atoi(argv[1]);
    int M     = std::atoi(argv[2]);
    int N     = std::atoi(argv[3]);
    int K     = std::atoi(argv[4]);
    int iters = std::atoi(argv[5]);
    CUCHK(cudaSetDevice(dev));

    int8_t *dA, *dB, *dEAL, *dEBR, *dEAR_R, *dEBL_R, *dEAR_K, *dEBL_K;
    int8_t *dApEA, *dBpEB;
    int32_t *dAxEBL_i32, *dEARxBpEB_i32;
    cutlass::half_t *dEAL_fp16, *dEBR_fp16, *dAxEBL_fp16, *dEARxBpEB_fp16;
    float *dAs, *dBs; cutlass::bfloat16_t *dC;
    CUCHK(cudaMalloc(&dA, size_t(M)*K));   CUCHK(cudaMalloc(&dB, size_t(N)*K));
    CUCHK(cudaMalloc(&dEAL, size_t(M)*R_DIM));  CUCHK(cudaMalloc(&dEBR, size_t(N)*R_DIM));
    CUCHK(cudaMalloc(&dEAR_R, size_t(K)*R_DIM)); CUCHK(cudaMalloc(&dEBL_R, size_t(K)*R_DIM));
    CUCHK(cudaMalloc(&dEAR_K, size_t(R_DIM)*K)); CUCHK(cudaMalloc(&dEBL_K, size_t(R_DIM)*K));
    CUCHK(cudaMalloc(&dApEA, size_t(M)*K)); CUCHK(cudaMalloc(&dBpEB, size_t(N)*K));
    CUCHK(cudaMalloc(&dAxEBL_i32, size_t(M)*R_DIM*4));
    CUCHK(cudaMalloc(&dEARxBpEB_i32, size_t(N)*R_DIM*4));
    CUCHK(cudaMalloc(&dEAL_fp16, size_t(M)*R_DIM*2));
    CUCHK(cudaMalloc(&dEBR_fp16, size_t(N)*R_DIM*2));
    CUCHK(cudaMalloc(&dAxEBL_fp16, size_t(M)*R_DIM*2));
    CUCHK(cudaMalloc(&dEARxBpEB_fp16, size_t(N)*R_DIM*2));
    CUCHK(cudaMalloc(&dAs, M*4));   CUCHK(cudaMalloc(&dBs, N*4));
    CUCHK(cudaMalloc(&dC, size_t(M)*N*2));

    auto run = [&]() {
        pearl_noisingA_sm89_64x64x64_R64_int32(
            dA, dEAL, dEAR_R, dEBL_K, dApEA, dAxEBL_i32, M, K, 0);
        pearl::sm89::pearl_noisingB_sm89_64x64x64_R64_int32(
            dB, dEBR, dEBL_R, dEAR_K, dBpEB, dEARxBpEB_i32, N, K, 0);
        pearl::sm89::pearl_gemm_sm89_denoise_128x128x64_R64(
            dApEA, K, dBpEB, K, dC, N, dAs, dBs,
            dEAL_fp16, dEBR_fp16, dAxEBL_fp16, dEARxBpEB_fp16,
            M, N, K, 0);
    };

    // Warmup — drains static-init paths (cudaFuncSetAttribute, env-var caches)
    for (int i = 0; i < 5; ++i) run();
    CUCHK(cudaDeviceSynchronize());

    // Per-iter timed runs (one event-per-iter; lets us compute median)
    std::vector<float> ms_each;
    ms_each.reserve(iters);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(e0);
        run();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        ms_each.push_back(ms);
    }
    std::sort(ms_each.begin(), ms_each.end());
    float median = ms_each[ms_each.size() / 2];
    float minv   = ms_each.front();
    float maxv   = ms_each.back();
    double tops_main = 2.0 * double(M) * double(N) * double(K) /
                       (double(median) / 1000.0) * 1e-12;
    char const* env = std::getenv("PEARL_SM89_GROUP16_SWIZZLE");
    char const* swz = std::getenv("PEARL_SM89_SWIZZLE");
    printf("M=%5d N=%5d K=%5d  GROUP16=%s SWZ=%s  med=%.3f ms (min %.3f / max %.3f)  main_gemm=%.2f TOPS\n",
           M, N, K,
           env ? env : "0",
           swz ? swz : "(auto)",
           median, minv, maxv, tops_main);
    // CSV-friendly tail line so the driver script can parse without regex
    printf("CSV,%d,%d,%d,%s,%.4f,%.2f\n",
           M, N, K,
           (env && env[0] == '1') ? "group16" : "adaptive",
           median * 1000.0,  // microseconds
           tops_main);

    cudaFree(dA); cudaFree(dB);
    cudaFree(dEAL); cudaFree(dEBR);
    cudaFree(dEAR_R); cudaFree(dEBL_R);
    cudaFree(dEAR_K); cudaFree(dEBL_K);
    cudaFree(dApEA); cudaFree(dBpEB);
    cudaFree(dAxEBL_i32); cudaFree(dEARxBpEB_i32);
    cudaFree(dEAL_fp16); cudaFree(dEBR_fp16);
    cudaFree(dAxEBL_fp16); cudaFree(dEARxBpEB_fp16);
    cudaFree(dAs); cudaFree(dBs); cudaFree(dC);
    return 0;
}
