// SPDX-License-Identifier: see LICENSE
//
// Variant A — bM=128, bN=256, bK=64, R=64, kStages=2 (no persist-B path).
// Re-validation of the prior rejected wider-N config against the production
// 128x128 baseline. Both kernels go through PersistentSwizzledTileScheduler +
// L2 access policy window (production launcher).
//
// Sweep: 1024^3, 2048^3, 4096^3, 8192^3. 5 runs each, median TOPS reported.
// CSV written to /tmp/bench_128x256_nopersist.csv.
//
// MAC accounting: "TOPS" = 2*M*N*K / median(time_seconds_per_run).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace pearl { namespace sm89 {
extern "C" void pearl_gemm_sm89_noiseless_128x128x64_R64(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    int, int, int, cudaStream_t);
extern "C" void pearl_gemm_sm89_noiseless_128x256x64_R64(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    int, int, int, cudaStream_t);
}}

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

using NoiselessFn = void(*)(int8_t const*, int64_t, int8_t const*, int64_t,
                            cutlass::bfloat16_t*, int64_t, float const*, float const*,
                            int, int, int, cudaStream_t);

struct RunResult {
    double median_us;
    double tops;
};

// Run fn `runs` times each timing `iters` launches via a single CUDA event pair,
// then return median per-launch microseconds + TOPS.
static RunResult bench_noiseless(NoiselessFn fn, int M, int N, int K,
                                 int iters, int runs) {
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

    // Warmup.
    for (int i = 0; i < 5; ++i) fn(dA, K, dB, K, dC, N, dAs, dBs, M, N, K, 0);
    CUCHK(cudaDeviceSynchronize());

    std::vector<double> us_samples;
    us_samples.reserve(runs);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int r = 0; r < runs; ++r) {
        cudaEventRecord(e0);
        for (int i = 0; i < iters; ++i) fn(dA, K, dB, K, dC, N, dAs, dBs, M, N, K, 0);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        us_samples.push_back(double(ms) * 1000.0 / double(iters));
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);

    std::sort(us_samples.begin(), us_samples.end());
    double median_us = us_samples[us_samples.size() / 2];
    double tops = 2.0 * double(M)*double(N)*double(K) / (median_us / 1e6) * 1e-12;

    cudaFree(dA); cudaFree(dB); cudaFree(dAs); cudaFree(dBs); cudaFree(dC);
    return RunResult{median_us, tops};
}

int main(int argc, char** argv) {
    int dev = (argc >= 2) ? std::atoi(argv[1]) : 0;
    char const* csv_path = (argc >= 3) ? argv[2] : "/tmp/bench_128x256_nopersist.csv";
    CUCHK(cudaSetDevice(dev));
    cudaDeviceProp p; CUCHK(cudaGetDeviceProperties(&p, dev));
    printf("device %d: %s sm_%d%d  L2=%dMB SMs=%d\n", dev, p.name, p.major, p.minor,
           int(p.l2CacheSize/1024/1024), p.multiProcessorCount);
    printf("4070 Ti SUPER peak INT8 dense = 353 TOPS\n\n");

    // Task spec: 1024^2, 2048^2, 4096^2, 8192^2  (square M=N=K)
    int sizes[][3] = {
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8192},
    };
    // iters per timed window: keep wall ~50-500 ms per window
    int iters_per_run[] = { 100, 30, 8, 2 };
    int constexpr kRuns = 5;

    FILE* csv = std::fopen(csv_path, "w");
    if (!csv) { fprintf(stderr, "ERR: open %s\n", csv_path); std::exit(1); }
    std::fprintf(csv, "shape,variant,median_us,TOPS\n");

    printf("%-14s | %-12s | %12s | %10s\n", "shape", "variant", "median_us", "TOPS");
    printf("---------------+--------------+--------------+-----------\n");
    for (size_t i = 0; i < sizeof(iters_per_run)/sizeof(iters_per_run[0]); ++i) {
        int M = sizes[i][0], N = sizes[i][1], K = sizes[i][2];
        int iters = iters_per_run[i];

        RunResult a = bench_noiseless(
            pearl::sm89::pearl_gemm_sm89_noiseless_128x128x64_R64,
            M, N, K, iters, kRuns);
        RunResult b = bench_noiseless(
            pearl::sm89::pearl_gemm_sm89_noiseless_128x256x64_R64,
            M, N, K, iters, kRuns);

        char shape[32];
        std::snprintf(shape, sizeof(shape), "%dx%dx%d", M, N, K);
        printf("%-14s | %-12s | %12.2f | %10.2f\n", shape, "128x128 (prod)", a.median_us, a.tops);
        printf("%-14s | %-12s | %12.2f | %10.2f  (%.2fx)\n",
               shape, "128x256 (A)", b.median_us, b.tops, b.tops / a.tops);
        std::fprintf(csv, "%s,128x128_prod,%.4f,%.4f\n", shape, a.median_us, a.tops);
        std::fprintf(csv, "%s,128x256_A_nopersist,%.4f,%.4f\n", shape, b.median_us, b.tops);
    }

    std::fclose(csv);
    printf("\nCSV: %s\n", csv_path);
    return 0;
}
