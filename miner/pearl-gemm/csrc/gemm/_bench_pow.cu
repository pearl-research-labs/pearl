// Benchmark the overhead of the PoW transcript accumulator (SkipReduction=false)
// vs the production denoise variant (SkipReduction=true).
//
// Both run the FULL noisy_gemm pipeline (noisingA + noisingB + denoise GEMM).
// The only difference is whether TileHashAccumulator runs inside the mainloop
// and whether check_pow_target/write_host_signal_header run after.
//
// Uses an UNREACHABLE pow_target (all-zeros) so the signal never fires and the
// CPU side never has to handle a found block — keeps the bench clean.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "host_signal_header.hpp"

namespace pearl { namespace sm89 {
extern "C" void pearl_gemm_sm89_denoise_128x128x64_R64(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    cutlass::half_t const*, cutlass::half_t const*,
    cutlass::half_t const*, cutlass::half_t const*,
    int, int, int, cudaStream_t);
extern "C" void pearl_gemm_sm89_pow_128x128x64_R64(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    cutlass::half_t const*, cutlass::half_t const*,
    cutlass::half_t const*, cutlass::half_t const*,
    uint32_t const*, uint32_t const*,
    void*, void*,
    uint64_t*,
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

static double bench_one(bool use_pow, int M, int N, int K, int iters) {
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

    // PoW structures (only used in the use_pow path)
    uint32_t *dPowTarget, *dPowKey;
    void     *dSignalSync, *dSignalHeader;
    uint64_t *dHashCounter;
    CUCHK(cudaMalloc(&dPowTarget,    8*4));
    CUCHK(cudaMalloc(&dPowKey,       8*4));
    CUCHK(cudaMalloc(&dSignalSync,   sizeof(HostSignalSync)));
    CUCHK(cudaMalloc(&dSignalHeader, host_signal_header_size));
    CUCHK(cudaMalloc(&dHashCounter,  8));
    // pow_target = all zeros (unreachable, no block ever found)
    CUCHK(cudaMemset(dPowTarget, 0,  8*4));
    // pow_key = arbitrary (BLAKE3 keyed by these bytes)
    {
      std::vector<uint32_t> key(8);
      for (size_t i = 0; i < 8; ++i) key[i] = uint32_t(0xa5a5a5a5 + i);
      CUCHK(cudaMemcpy(dPowKey, key.data(), 8*4, cudaMemcpyHostToDevice));
    }
    CUCHK(cudaMemset(dSignalSync, 0,   sizeof(HostSignalSync)));
    CUCHK(cudaMemset(dSignalHeader, 0, host_signal_header_size));
    CUCHK(cudaMemset(dHashCounter, 0,  8));

    auto run_iter = [&]() {
        pearl_noisingA_sm89_64x64x64_R64_int32(dA, dEAL, dEAR_R, dEBL_K, dApEA, dAxEBL_i32, M, K, 0);
        pearl::sm89::pearl_noisingB_sm89_64x64x64_R64_int32(dB, dEBR, dEBL_R, dEAR_K, dBpEB, dEARxBpEB_i32, N, K, 0);
        if (use_pow) {
            pearl::sm89::pearl_gemm_sm89_pow_128x128x64_R64(
                dApEA, K, dBpEB, K, dC, N, dAs, dBs,
                dEAL_fp16, dEBR_fp16, dAxEBL_fp16, dEARxBpEB_fp16,
                dPowTarget, dPowKey,
                dSignalSync, dSignalHeader, dHashCounter,
                M, N, K, 0);
        } else {
            pearl::sm89::pearl_gemm_sm89_denoise_128x128x64_R64(
                dApEA, K, dBpEB, K, dC, N, dAs, dBs,
                dEAL_fp16, dEBR_fp16, dAxEBL_fp16, dEARxBpEB_fp16,
                M, N, K, 0);
        }
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
    cudaFree(dPowTarget); cudaFree(dPowKey);
    cudaFree(dSignalSync); cudaFree(dSignalHeader); cudaFree(dHashCounter);
    return tops_main;
}

int main(int argc, char** argv) {
    int dev = (argc >= 2) ? std::atoi(argv[1]) : 0;
    CUCHK(cudaSetDevice(dev));
    cudaDeviceProp p; CUCHK(cudaGetDeviceProperties(&p, dev));
    printf("device %d: %s sm_%d%d\n", dev, p.name, p.major, p.minor);

    int sizes[][3] = {
      { 1024,  1024, 1024}, { 2048,  2048, 2048}, { 4096,  4096, 4096},
      { 4096,  4096, 8192}, { 8192,  8192, 4096},
    };
    int iters[]   = { 20, 10, 5, 5, 3 };

    printf("\n%-32s  | %-13s | %-13s | %-7s\n",
           "denoise main TOPS", "SkipRed=true", "SkipRed=false", "ratio");
    printf("%-32s  +-%-13s-+-%-13s-+-%-7s\n",
           "--------------------------------",
           "-------------", "-------------", "-------");
    for (size_t i = 0; i < sizeof(iters)/sizeof(iters[0]); ++i) {
        int M = sizes[i][0], N = sizes[i][1], K = sizes[i][2];
        double tF = bench_one(false, M, N, K, iters[i]);
        double tT = bench_one(true,  M, N, K, iters[i]);
        printf("M=%5d N=%5d K=%5d         | %13.2f | %13.2f | %.3fx\n",
               M, N, K, tF, tT, tT / tF);
    }
    return 0;
}
