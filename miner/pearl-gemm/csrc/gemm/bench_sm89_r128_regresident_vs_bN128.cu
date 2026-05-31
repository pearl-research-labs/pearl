// SPDX-License-Identifier: see LICENSE
//
// Head-to-head TOPS bench: R=128 bM=128 bN=128 REGISTER-RESIDENT denoise
//                   vs   R=128 bM=64  bN=128 smem-resident denoise (wave-2 winner)
//
// Outputs a CSV to stdout (also redirected by the runner). 5 reps per case;
// reports median ms / median TOPS. Tested sizes (per bench spec):
//   256, 512, 1024, 2048, 4096, 8192 (square M=N=K)
//   plus the production hot path 4096x4096x8192.
//
// sm_89 only. Built from inside the pearl-ab Docker container.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// regresident variant (this session's wave 2)
extern "C" void pearl_gemm_sm89_denoise_regresident_128x128x64_R128(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    cutlass::half_t const*, cutlass::half_t const*,
    cutlass::half_t const*, cutlass::half_t const*,
    int, int, int, cudaStream_t);

// reference wave-2 winner: smem-resident bM=64 bN=128
extern "C" void pearl_gemm_sm89_denoise_64x128x64_R128(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    cutlass::half_t const*, cutlass::half_t const*,
    cutlass::half_t const*, cutlass::half_t const*,
    int, int, int, cudaStream_t);

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

enum class Variant { BM128_REGRES, BM64_BN128 };

static double bench_once(Variant v, int M, int N, int K, int iters) {
    int const R = 128;
    std::vector<int8_t> hA(size_t(M) * K, 1), hB(size_t(N) * K, 1);
    std::vector<float>  hAs(M, 0.01f), hBs(N, 0.01f);

    int8_t  *dA, *dB;
    float   *dAs, *dBs;
    cutlass::bfloat16_t *dC;
    cutlass::half_t *dEAL, *dEBR, *dAxEBL, *dEARxBpEB;
    CUCHK(cudaMalloc(&dA,  hA.size()));
    CUCHK(cudaMalloc(&dB,  hB.size()));
    CUCHK(cudaMalloc(&dAs, hAs.size() * 4));
    CUCHK(cudaMalloc(&dBs, hBs.size() * 4));
    CUCHK(cudaMalloc(&dC,  size_t(M) * N * 2));
    CUCHK(cudaMalloc(&dEAL,      size_t(M) * R * 2));
    CUCHK(cudaMalloc(&dEBR,      size_t(N) * R * 2));
    CUCHK(cudaMalloc(&dAxEBL,    size_t(M) * R * 2));
    CUCHK(cudaMalloc(&dEARxBpEB, size_t(N) * R * 2));
    CUCHK(cudaMemset(dEAL,      0, size_t(M) * R * 2));
    CUCHK(cudaMemset(dEBR,      0, size_t(N) * R * 2));
    CUCHK(cudaMemset(dAxEBL,    0, size_t(M) * R * 2));
    CUCHK(cudaMemset(dEARxBpEB, 0, size_t(N) * R * 2));
    CUCHK(cudaMemcpy(dA,  hA.data(),  hA.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dB,  hB.data(),  hB.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dAs, hAs.data(), hAs.size() * 4, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dBs, hBs.data(), hBs.size() * 4, cudaMemcpyHostToDevice));

    auto run_once = [&]() {
        if (v == Variant::BM128_REGRES) {
            pearl_gemm_sm89_denoise_regresident_128x128x64_R128(
                dA, K, dB, K, dC, N, dAs, dBs,
                dEAL, dEBR, dAxEBL, dEARxBpEB, M, N, K, 0);
        } else {
            pearl_gemm_sm89_denoise_64x128x64_R128(
                dA, K, dB, K, dC, N, dAs, dBs,
                dEAL, dEBR, dAxEBL, dEARxBpEB, M, N, K, 0);
        }
    };
    // Warmup
    for (int i = 0; i < 3; ++i) run_once();
    CUCHK(cudaDeviceSynchronize());
    CUCHK(cudaGetLastError());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i) run_once();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);

    double sec_per = double(ms) / 1000.0 / double(iters);
    double tops = 2.0 * double(M) * double(N) * double(K) / sec_per * 1e-12;

    cudaFree(dA); cudaFree(dB); cudaFree(dAs); cudaFree(dBs); cudaFree(dC);
    cudaFree(dEAL); cudaFree(dEBR); cudaFree(dAxEBL); cudaFree(dEARxBpEB);
    return tops;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    int dev = (argc >= 2) ? std::atoi(argv[1]) : 0;
    CUCHK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CUCHK(cudaGetDeviceProperties(&p, dev));
    fprintf(stderr, "device %d: %s sm_%d%d  smem_optin=%zu KB\n",
            dev, p.name, p.major, p.minor, p.sharedMemPerBlockOptin / 1024);

    int const SIZES[][3] = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {4096, 4096, 8192},
        {8192, 8192, 8192},
    };
    int const N_REPS = 5;
    // CSV header
    printf("M,N,K,variant,median_tops,min_tops,max_tops\n");
    for (auto const& s : SIZES) {
        int M = s[0], N = s[1], K = s[2];

        for (auto v : {Variant::BM128_REGRES, Variant::BM64_BN128}) {
            // Per-variant iters: bM=128 regres is ~1000× slower than bM=64;
            // tune iters so each rep finishes in <=5s.
            long long flops = 2LL * M * N * K;
            int iters;
            if (v == Variant::BM128_REGRES) {
                // empirically ~0.1 TOPS on regres -> 1e11 flops takes 1s
                iters = (flops > 50LL * 1000 * 1000 * 1000) ? 1 :
                        (flops > 5LL  * 1000 * 1000 * 1000) ? 2 : 5;
            } else {
                iters = (flops > 200LL * 1024 * 1024 * 1024) ? 5 : 20;
            }
            std::vector<double> reps;
            for (int r = 0; r < N_REPS; ++r) {
                reps.push_back(bench_once(v, M, N, K, iters));
            }
            char const* tag = (v == Variant::BM128_REGRES)
                ? "bM128_bN128_regres" : "bM64_bN128_smem";
            double mn = *std::min_element(reps.begin(), reps.end());
            double mx = *std::max_element(reps.begin(), reps.end());
            double md = median(reps);
            printf("%d,%d,%d,%s,%.3f,%.3f,%.3f\n", M, N, K, tag, md, mn, mx);
            fflush(stdout);
            fprintf(stderr, "  %dx%dx%d  %s  iters/rep=%d  median=%.2f TOPS  range[%.2f,%.2f]\n",
                    M, N, K, tag, iters, md, mn, mx);
        }
    }
    return 0;
}
