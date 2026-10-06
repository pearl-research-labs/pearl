// Hardware capture of the A100/GA100 (sm_80) HMMA.16816.F32 accumulation, for
// the FP16 proof-of-useful-work scheme (docs/fp16_scheme). This is the
// *hardware* oracle: it issues the real tensor-core `mma.sync` on silicon and
// dumps the FP32 result bits, so the software model in
// `zk-pow/src/api/fp16/accumulate.rs` can be checked against the device rather
// than against another copy of itself.
//
// Deliberately standalone: it links only the CUDA runtime (no torch, no
// pearl-gemm), so it builds and runs on an sm_80 box that cannot install the
// py3.12 miner stack. The mma asm and the distributed fragment layout are copied
// verbatim from the validated production kernel
// `miner/pearl-gemm/src/pearl_gemm/fp16_gemm/_kernel_sm80.cu`, so a capture here
// exercises exactly the datapath the miner uses.
//
// Build:  nvcc -arch=sm_80 -O2 -o a100_hmma_capture a100_hmma_capture.cu
//
// Input (stdin), one dot product per line:
//     k  a_bits[0..k)  b_bits[0..k)  c_bits
// where a_bits/b_bits are FP16 bit patterns (u16) and c_bits is the FP32
// carry-in bit pattern (u32). `k` must be a positive multiple of 16.
//
// Output (stdout), one line per input: the FP32 result bit pattern (u32),
// captured from the device D[0][0] of a 16x8 tile whose A-row 0 is `a`, B-row 0
// is `b`, and C[0][0] is the carry-in (all other tile entries zero).
//
// Exit non-zero on any CUDA error or malformed input.

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(_e), __FILE__, __LINE__);                \
            std::exit(2);                                                       \
        }                                                                       \
    } while (0)

__device__ __forceinline__ unsigned pack2(const __half* p, int i0, int i1) {
    __half2 h = __halves2half2(p[i0], p[i1]);
    return *reinterpret_cast<unsigned*>(&h);
}

// One warp computes one 16x8 output subtile D = A(16xK) . B(8xK)^T with FP32
// accumulation chained in ascending k order (groups of 16 per mma.sync, no
// split-k, no atomics), carrying the accumulator forward across the whole k
// axis -- the pinned reduction order that defines the device result. Identical
// to fp16_gemm_a100_kernel but single-tile (M=16, N=8).
__global__ void capture_kernel(const __half* __restrict__ A,
                               const __half* __restrict__ B,
                               const float* __restrict__ C,
                               float* __restrict__ D, int K) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;  // 0..7
    int t4 = lane & 3;    // 0..3

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    if (C != nullptr) {
        c0 = C[(gid) * 8 + t4 * 2];
        c1 = C[(gid) * 8 + t4 * 2 + 1];
        c2 = C[(gid + 8) * 8 + t4 * 2];
        c3 = C[(gid + 8) * 8 + t4 * 2 + 1];
    }
    for (int k0 = 0; k0 < K; k0 += 16) {
        unsigned a0 = pack2(A, (gid) * K + k0 + t4 * 2, (gid) * K + k0 + t4 * 2 + 1);
        unsigned a1 = pack2(A, (gid + 8) * K + k0 + t4 * 2, (gid + 8) * K + k0 + t4 * 2 + 1);
        unsigned a2 = pack2(A, (gid) * K + k0 + t4 * 2 + 8, (gid) * K + k0 + t4 * 2 + 9);
        unsigned a3 = pack2(A, (gid + 8) * K + k0 + t4 * 2 + 8, (gid + 8) * K + k0 + t4 * 2 + 9);
        unsigned b0 = pack2(B, (gid) * K + k0 + t4 * 2, (gid) * K + k0 + t4 * 2 + 1);
        unsigned b1 = pack2(B, (gid) * K + k0 + t4 * 2 + 8, (gid) * K + k0 + t4 * 2 + 9);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    D[(gid) * 8 + t4 * 2] = c0;
    D[(gid) * 8 + t4 * 2 + 1] = c1;
    D[(gid + 8) * 8 + t4 * 2] = c2;
    D[(gid + 8) * 8 + t4 * 2 + 1] = c3;
}

int main() {
    // Persistent device buffers sized to the largest tile we expect.
    int cap_k = 0;
    __half *dA = nullptr, *dB = nullptr;
    float *dC = nullptr, *dD = nullptr;
    std::vector<__half> hA, hB;
    float hC[128], hD[128];

    // Each record is whitespace-separated: k, then k a-bits, k b-bits, c-bits.
    long k;
    while (scanf("%ld", &k) == 1) {
        if (k <= 0) {
            fprintf(stderr, "bad k=%ld (must be positive)\n", k);
            return 1;
        }
        // The mma tile runs in units of 16 along the contraction axis. Any k is
        // zero-padded up to the next multiple of 16; within each group of 8 the
        // padded entries are zero (skipped by the model), and any whole trailing
        // all-zero group is a no-op, so the padded capture equals the model's
        // result for the true k -- which also validates those properties on
        // silicon (the committed corpus includes k not a multiple of 16).
        long kpad = (k + 15) / 16 * 16;
        std::vector<uint16_t> a(k), b(k);
        uint32_t cbits;
        for (long i = 0; i < k; i++) {
            unsigned v;
            if (scanf("%u", &v) != 1) { fprintf(stderr, "truncated a\n"); return 1; }
            a[i] = (uint16_t)v;
        }
        for (long i = 0; i < k; i++) {
            unsigned v;
            if (scanf("%u", &v) != 1) { fprintf(stderr, "truncated b\n"); return 1; }
            b[i] = (uint16_t)v;
        }
        if (scanf("%u", &cbits) != 1) { fprintf(stderr, "truncated c\n"); return 1; }

        if (kpad > cap_k) {
            if (dA) { cudaFree(dA); cudaFree(dB); }
            CUDA_CHECK(cudaMalloc(&dA, sizeof(__half) * 16 * kpad));
            CUDA_CHECK(cudaMalloc(&dB, sizeof(__half) * 8 * kpad));
            if (!dC) {
                CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * 128));
                CUDA_CHECK(cudaMalloc(&dD, sizeof(float) * 128));
            }
            hA.resize(16 * kpad);
            hB.resize(8 * kpad);
            cap_k = kpad;
        }
        // Zero the tile, place the vectors in row 0 of A and B, carry-in at [0][0].
        for (auto& h : hA) h = __ushort_as_half((unsigned short)0);
        for (auto& h : hB) h = __ushort_as_half((unsigned short)0);
        for (long i = 0; i < k; i++) hA[i] = __ushort_as_half((unsigned short)a[i]);
        for (long i = 0; i < k; i++) hB[i] = __ushort_as_half((unsigned short)b[i]);
        for (int i = 0; i < 128; i++) hC[i] = 0.f;
        float cf;
        memcpy(&cf, &cbits, 4);
        hC[0] = cf;

        CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(__half) * 16 * kpad, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(__half) * 8 * kpad, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dC, hC, sizeof(float) * 128, cudaMemcpyHostToDevice));
        capture_kernel<<<1, 32>>>(dA, dB, dC, dD, (int)kpad);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hD, dD, sizeof(float) * 128, cudaMemcpyDeviceToHost));

        uint32_t dbits;
        memcpy(&dbits, &hD[0], 4);
        printf("%u\n", dbits);
    }
    return 0;
}
