// SPDX-License-Identifier: see LICENSE
//
// Structured-input diagnostic for pearl_gemm_sm89_debug_int32_dump.
//
// Instead of random A/B, sets A and B to simple deterministic patterns that
// make ANY lane / row / col mismatch visible by inspection of the int32
// output tile.
//
// Test patterns:
//   #0: A[i,k] = 1, B[j,k] = 1            -> C[i,j] = K   (constant)
//   #1: A[i,k] = (k == 0) ? 1 : 0, B[j,k] = (k == 0) ? 1 : 0
//                                          -> C[i,j] = 1   (constant)
//   #2: A[i,k] = (i==k) ? 1 : 0, B[j,k] = (j==k) ? 1 : 0   (identity-like)
//                                          -> C[i,j] = (i==j) ? 1 : 0
//   #3: A[i,k] = 1, B[j,k] = j_small      -> C[i,j] = K * j_small   (col pattern)
//   #4: A[i,k] = i_small, B[j,k] = 1      -> C[i,j] = K * i_small   (row pattern)

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace pearl {
namespace sm89 {
extern "C" void pearl_gemm_sm89_debug_int32_dump(
    int8_t const* A, int64_t lda,
    int8_t const* B, int64_t ldb,
    int32_t* C, int64_t ldc,
    int M, int N, int K,
    cudaStream_t stream);
}
}

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

static void dump_int32_tile(const char* tag, const int32_t* C, int M, int N,
                            int max_rows, int max_cols) {
    printf("%s int32 dump (showing %dx%d of %dx%d):\n", tag,
           max_rows, max_cols, M, N);
    for (int r = 0; r < max_rows && r < M; ++r) {
        printf("  r=%3d: ", r);
        for (int c = 0; c < max_cols && c < N; ++c) {
            printf("%6d ", C[r * N + c]);
        }
        printf("\n");
    }
}

static void compute_ref(int M, int N, int K,
                        const int8_t* A, const int8_t* B,
                        int32_t* C) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k)
                acc += int32_t(A[i*K+k]) * int32_t(B[j*K+k]);
            C[i*N+j] = acc;
        }
}

static int run_pattern(int M, int N, int K, int pattern_id) {
    std::vector<int8_t> hA(size_t(M)*K), hB(size_t(N)*K);
    std::vector<int32_t> hC(size_t(M)*N), hRef(size_t(M)*N);

    auto setA = [&](int i, int k) -> int8_t {
        switch (pattern_id) {
            case 0: return 1;
            case 1: return (k == 0) ? 1 : 0;
            case 2: return (i == k) ? 1 : 0;
            case 3: return 1;
            case 4: return int8_t(i & 7);
            case 5: return int8_t((i + 1) & 0x3F);   // row indexed
            default: return 0;
        }
    };
    auto setB = [&](int j, int k) -> int8_t {
        switch (pattern_id) {
            case 0: return 1;
            case 1: return (k == 0) ? 1 : 0;
            case 2: return (j == k) ? 1 : 0;
            case 3: return int8_t(j & 7);
            case 4: return 1;
            case 5: return int8_t((j + 1) & 0x3F);   // col indexed
            default: return 0;
        }
    };

    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            hA[i*K+k] = setA(i, k);
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < K; ++k)
            hB[j*K+k] = setB(j, k);

    int8_t  *dA, *dB;
    int32_t *dC;
    CUCHK(cudaMalloc(&dA, hA.size()));
    CUCHK(cudaMalloc(&dB, hB.size()));
    CUCHK(cudaMalloc(&dC, hC.size() * sizeof(int32_t)));
    CUCHK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemset(dC, 0xAA, hC.size() * sizeof(int32_t)));

    pearl::sm89::pearl_gemm_sm89_debug_int32_dump(
        dA, K, dB, K, dC, N, M, N, K, /*stream=*/0);
    CUCHK(cudaDeviceSynchronize());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "launch err: %s\n", cudaGetErrorString(err));
        return 1;
    }

    CUCHK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(int32_t),
                     cudaMemcpyDeviceToHost));
    compute_ref(M, N, K, hA.data(), hB.data(), hRef.data());

    long bad = 0;
    for (size_t i = 0; i < hC.size(); ++i) if (hC[i] != hRef[i]) ++bad;

    printf("\n========== pattern %d (M=%d N=%d K=%d) bad=%ld/%zu ==========\n",
           pattern_id, M, N, K, bad, hC.size());
    dump_int32_tile("REF", hRef.data(), M, N, 16, 32);
    dump_int32_tile("GOT", hC.data(),   M, N, 16, 32);

    CUCHK(cudaFree(dA));
    CUCHK(cudaFree(dB));
    CUCHK(cudaFree(dC));
    return 0;
}

int main(int argc, char** argv) {
    int dev = (argc >= 2) ? std::atoi(argv[1]) : 0;
    CUCHK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CUCHK(cudaGetDeviceProperties(&p, dev));
    int cc = p.major * 10 + p.minor;
    printf("device %d: %s sm_%d\n", dev, p.name, cc);
    if (cc != 89) fprintf(stderr, "WARN: built for sm_89; running on sm_%d\n", cc);

    int M = 128, N = 128, K = 128;
    // Pattern 0: A=1, B=1 -> C[i,j] = K = 128 everywhere
    run_pattern(M, N, K, 0);
    // Pattern 5: A[i,k] = (i+1) & 63, B[j,k] = (j+1) & 63
    //   -> C[i,j] = K * (i+1) * (j+1) (after mask)
    //   -> row r is K*(r+1) * (col-pattern). Different per (row, col).
    run_pattern(M, N, K, 5);
    return 0;
}
