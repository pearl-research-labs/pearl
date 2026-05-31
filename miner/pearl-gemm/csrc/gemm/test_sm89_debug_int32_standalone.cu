// SPDX-License-Identifier: see LICENSE
//
// Self-contained correctness test for pearl_gemm_sm89_debug_int32_dump.
// Reads the raw int32 accumulator from the mainloop and compares against a
// CPU int32 reference (bit-exact for int8 x int8 -> int32; no rounding).
//
// If THIS test fails, the bug is in mainloop / MMA / smem layout / G2S / S2R.
// If THIS test passes but test_sm89_standalone.cu fails, the bug is in the
// epilogue (scale / cast / smem stage / S2G).
//
// Build (on a machine with nvcc 12.x):
//   nvcc -gencode arch=compute_89,code=sm_89 -std=c++20 -O3
//        -I . -I ../../third_party/cutlass/include
//        -I ../../third_party/cutlass/tools/util/include
//        -I ../../third_party/cutlass/examples/common
//        --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG
//        pearl_gemm_sm89_debug_inst.cu test_sm89_debug_int32_standalone.cu
//        -o /tmp/test_sm89_debug_int32

#include <cuda_runtime.h>

#include <cassert>
#include <cinttypes>
#include <cmath>
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

// int32 reference: C[i,j] = sum_k A[i,k] * B[j,k]. Bit-exact vs MMA.
static void ref_gemm_i32(int M, int N, int K,
                         int8_t const* A, int8_t const* B,
                         int32_t* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += int32_t(A[i * K + k]) * int32_t(B[j * K + k]);
            }
            C[i * N + j] = acc;
        }
    }
}

static int run_case(int M, int N, int K, unsigned seed,
                    bool dump_mismatches = true) {
    std::srand(seed);
    std::vector<int8_t>  hA(size_t(M) * K), hB(size_t(N) * K);
    std::vector<int32_t> hC(size_t(M) * N), hCref(size_t(M) * N);

    for (auto& v : hA) v = int8_t((std::rand() % 255) - 127);
    for (auto& v : hB) v = int8_t((std::rand() % 255) - 127);

    int8_t  *dA, *dB;
    int32_t *dC;
    CUCHK(cudaMalloc(&dA, hA.size()));
    CUCHK(cudaMalloc(&dB, hB.size()));
    CUCHK(cudaMalloc(&dC, hC.size() * sizeof(int32_t)));

    CUCHK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
    // Fill with a sentinel so untouched cells are visible.
    CUCHK(cudaMemset(dC, 0xAA, hC.size() * sizeof(int32_t)));

    pearl::sm89::pearl_gemm_sm89_debug_int32_dump(
        dA, /*lda=*/K, dB, /*ldb=*/K, dC, /*ldc=*/N, M, N, K, /*stream=*/0);
    CUCHK(cudaDeviceSynchronize());
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "kernel launch error: %s\n", cudaGetErrorString(launch_err));
        return 1;
    }

    CUCHK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(int32_t),
                     cudaMemcpyDeviceToHost));
    ref_gemm_i32(M, N, K, hA.data(), hB.data(), hCref.data());

    long bad = 0;
    long worst_idx = 0;
    int32_t worst_got = 0, worst_ref = 0;
    int32_t worst_diff = 0;
    const int32_t sentinel = int32_t(0xAAAAAAAA);
    long n_sentinel_left = 0;
    for (size_t i = 0; i < hC.size(); ++i) {
        if (hC[i] == sentinel) ++n_sentinel_left;
        if (hC[i] != hCref[i]) {
            ++bad;
            int32_t d = std::abs(hC[i] - hCref[i]);
            if (d > std::abs(worst_diff)) {
                worst_diff = hC[i] - hCref[i];
                worst_idx  = long(i);
                worst_got  = hC[i];
                worst_ref  = hCref[i];
            }
        }
    }

    printf("M=%d N=%d K=%d seed=%u   bad=%ld/%zu  sentinel_left=%ld",
           M, N, K, seed, bad, hC.size(), n_sentinel_left);
    if (bad == 0) {
        printf("   PASS\n");
    } else {
        long row = worst_idx / N;
        long col = worst_idx % N;
        printf("   FAIL  worst@[%ld,%ld]  ref=%d got=%d diff=%d\n",
               row, col, worst_ref, worst_got, worst_diff);
        if (dump_mismatches) {
            // Print a few representative mismatches across the tile to help
            // pattern-match the bug class (per-row, per-col, per-warp, etc).
            printf("  sample mismatches:\n");
            int shown = 0;
            for (size_t i = 0; i < hC.size() && shown < 20; ++i) {
                if (hC[i] != hCref[i]) {
                    long r = long(i) / N, c = long(i) % N;
                    printf("    [%4ld,%4ld] ref=%9d got=%9d diff=%9d\n",
                           r, c, hCref[i], hC[i], hC[i] - hCref[i]);
                    ++shown;
                }
            }
            // Also dump the first 4 row patterns: how many bad cells per row.
            printf("  bad-per-row (first 8):\n");
            for (long r = 0; r < (M < 8 ? M : 8); ++r) {
                long n_bad = 0;
                for (long c = 0; c < N; ++c) {
                    if (hC[r * N + c] != hCref[r * N + c]) ++n_bad;
                }
                printf("    row %4ld: %ld/%d bad\n", r, n_bad, N);
            }
            printf("  bad-per-col (first 8):\n");
            for (long c = 0; c < (N < 8 ? N : 8); ++c) {
                long n_bad = 0;
                for (long r = 0; r < M; ++r) {
                    if (hC[r * N + c] != hCref[r * N + c]) ++n_bad;
                }
                printf("    col %4ld: %ld/%d bad\n", c, n_bad, M);
            }
        }
    }

    CUCHK(cudaFree(dA));
    CUCHK(cudaFree(dB));
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
    // Single 128x128x128 tile: simplest case — one CTA, one k-tile pair.
    // If this fails, the failure pattern itself reveals which lane / warp
    // is producing wrong data.
    rc |= run_case(128, 128, 128, 0);
    if (rc != 0) {
        printf("\n(stopping after first failure for clarity)\n");
        return rc;
    }
    rc |= run_case(256, 256, 128, 1);
    rc |= run_case(512, 512, 512, 2);
    rc |= run_case(1024, 1024, 1024, 3);
    if (rc == 0) printf("\nALL PASS (int32 accumulator matches reference)\n");
    else         printf("\nFAIL\n");
    return rc;
}
