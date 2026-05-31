// SPDX-License-Identifier: see LICENSE
//
// Throughput bench for the sm_89 noiseless int8 GEMM. Measures TOPS
// (= 2 * M * N * K / time_seconds * 1e-12) for the problem sizes Pearl pool
// emits (M=N=131072 split into 1024-row chunks per iteration; K=4096).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cutlass/numeric_types.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
}
}

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

static double bench(int M, int N, int K, int iters) {
    std::vector<int8_t> hA(size_t(M)*K), hB(size_t(N)*K);
    std::vector<float>  hAs(M, 0.01f), hBs(N, 0.01f);
    for (auto& v : hA) v = int8_t((rand() % 255) - 127);
    for (auto& v : hB) v = int8_t((rand() % 255) - 127);

    int8_t  *dA, *dB;
    float   *dAs, *dBs;
    cutlass::bfloat16_t *dC;
    CUCHK(cudaMalloc(&dA,  hA.size()));
    CUCHK(cudaMalloc(&dB,  hB.size()));
    CUCHK(cudaMalloc(&dAs, hAs.size()*4));
    CUCHK(cudaMalloc(&dBs, hBs.size()*4));
    CUCHK(cudaMalloc(&dC,  size_t(M)*N*2));
    CUCHK(cudaMemcpy(dA,  hA.data(),  hA.size(),  cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dB,  hB.data(),  hB.size(),  cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dAs, hAs.data(), hAs.size()*4, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dBs, hBs.data(), hBs.size()*4, cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < 3; ++i)
        pearl::sm89::pearl_gemm_sm89_noiseless_128x128x64_R64(
            dA, K, dB, K, dC, N, dAs, dBs, M, N, K, 0);
    CUCHK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i)
        pearl::sm89::pearl_gemm_sm89_noiseless_128x128x64_R64(
            dA, K, dB, K, dC, N, dAs, dBs, M, N, K, 0);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);

    double seconds = double(ms) / 1000.0 / double(iters);
    double tops = 2.0 * double(M) * double(N) * double(K) / seconds * 1e-12;
    printf("  M=%6d N=%6d K=%6d   %.3f ms/call   %.2f TOPS\n",
           M, N, K, ms / iters, tops);

    cudaFree(dA); cudaFree(dB); cudaFree(dAs); cudaFree(dBs); cudaFree(dC);
    return tops;
}

int main(int argc, char** argv) {
    int dev = (argc >= 2) ? std::atoi(argv[1]) : 0;
    CUCHK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CUCHK(cudaGetDeviceProperties(&p, dev));
    printf("device %d: %s sm_%d%d\n", dev, p.name, p.major, p.minor);
    // Reference: 4070 Ti SUPER peak INT8 dense = 353 TOPS (per nvidia spec).
    printf("note: 4070 Ti SUPER peak INT8 dense = 353 TOPS\n\n");

    // Bench at sizes consistent with Pearl mining workloads.
    // Pool emits M=N=131072 K=4096; alpha-miner processes in chunks.
    // We bench at scales where our 128x128x64 tile saturates the GPU.
    bench( 1024,  1024, 1024, 50);
    bench( 2048,  2048, 2048, 30);
    bench( 4096,  4096, 4096, 10);
    bench( 4096,  4096, 8192, 10);
    bench( 8192,  8192, 4096, 5);
    bench( 4096, 16384, 4096, 5);
    bench(16384,  4096, 4096, 5);
    return 0;
}
