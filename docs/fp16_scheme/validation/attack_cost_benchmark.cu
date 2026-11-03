// Truncation-correction attack-cost micro-benchmark (FP16 scheme, A100/sm_80).
//
// The hardness argument (whitepaper Section "Hardness"/"Attack vectors") is: the
// device output D_ij = RZ( S_ij - sum_u rho_u - sum_g gamma_g ), where S_ij is
// the exact dot product (an attacker can get it, and every eta, cheaply from the
// low-rank structure), but the per-product truncation corrections rho_u are a
// *per-product nonlinearity applied before the reduction* -- not a contraction,
// so no tensor-core matmul evaluates them. The attacker must compute them
// elementwise on CUDA cores. This benchmark measures that cost against the
// honest tensor-core GEMM, reproducing the paper's "6-13x honest
// (compute-bound-optimistic)" and "memory-bound, far more" figures on silicon.
//
// We grant the attacker EVERYTHING the paper grants and more: the exact sums and
// the breakpoint oracle are free; the attack kernel only does the unavoidable
// work of touching each of the M*N*K products and computing its truncation
// remainder on CUDA cores (decompose both FP16 operands, multiply significands,
// per-group eta, extract the discarded low bits). The honest kernel is the real
// `mma.sync.m16n8k16.f32.f16` tensor-core GEMM (same datapath as the miner).
//
// Build:  nvcc -arch=sm_80 -O3 -o attack_cost_benchmark attack_cost_benchmark.cu
// Run:    ./attack_cost_benchmark [M N K iters]     (defaults 512 512 1024 50)

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(e) do{cudaError_t _e=(e); if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(_e),__LINE__);std::exit(2);} }while(0)

#define W 24
#define GROUP 8

// ---------- honest: tensor-core GEMM (one warp per 16x8 subtile) ----------
__device__ __forceinline__ unsigned pack2(const __half* p, long i0, long i1) {
    __half2 h = __halves2half2(p[i0], p[i1]);
    return *reinterpret_cast<unsigned*>(&h);
}
__global__ void honest_gemm(const __half* __restrict__ A, const __half* __restrict__ B,
                            float* __restrict__ D, int M, int N, int K) {
    int m0 = blockIdx.y * 16, n0 = blockIdx.x * 8;
    if (m0 >= M || n0 >= N) return;
    int lane = threadIdx.x & 31, gid = lane >> 2, t4 = lane & 3;
    const __half* Am = A + (long)m0 * K;
    const __half* Bn = B + (long)n0 * K;
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (int k0 = 0; k0 < K; k0 += 16) {
        unsigned a0 = pack2(Am, (gid) * K + k0 + t4 * 2, (gid) * K + k0 + t4 * 2 + 1);
        unsigned a1 = pack2(Am, (gid + 8) * K + k0 + t4 * 2, (gid + 8) * K + k0 + t4 * 2 + 1);
        unsigned a2 = pack2(Am, (gid) * K + k0 + t4 * 2 + 8, (gid) * K + k0 + t4 * 2 + 9);
        unsigned a3 = pack2(Am, (gid + 8) * K + k0 + t4 * 2 + 8, (gid + 8) * K + k0 + t4 * 2 + 9);
        unsigned b0 = pack2(Bn, (gid) * K + k0 + t4 * 2, (gid) * K + k0 + t4 * 2 + 1);
        unsigned b1 = pack2(Bn, (gid) * K + k0 + t4 * 2 + 8, (gid) * K + k0 + t4 * 2 + 9);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    float* Dt = D + (long)m0 * N + n0;
    Dt[(gid) * N + t4 * 2] = c0; Dt[(gid) * N + t4 * 2 + 1] = c1;
    Dt[(gid + 8) * N + t4 * 2] = c2; Dt[(gid + 8) * N + t4 * 2 + 1] = c3;
}

// ---------- attack: per-product truncation correction on CUDA cores ----------
// (sign, significand, stored_exp) of an FP16 code, matching dtype::decompose_fp16.
__device__ __forceinline__ void decompose(unsigned short bits, int& sign, int& m, int& eps) {
    int exp = (bits >> 10) & 0x1F, man = bits & 0x3FF;
    sign = (bits & 0x8000) ? -1 : 1;
    if (exp == 0) { m = man; eps = -14; }           // subnormal/zero
    else { m = 0x400 | man; eps = exp - 15; }       // normal
}
// One thread per output cell; compute sum_u rho_u (the correction) for its row/col.
__global__ void attack_corrections(const __half* __restrict__ A, const __half* __restrict__ B,
                                    float* __restrict__ OUT, int M, int N, int K) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)M * N) return;
    int i = idx / N, j = idx % N;
    const unsigned short* a = reinterpret_cast<const unsigned short*>(A) + (long)i * K;
    const unsigned short* b = reinterpret_cast<const unsigned short*>(B) + (long)j * K;
    long correction = 0;  // sum of discarded low bits (the attacker's unavoidable work)
    for (int g0 = 0; g0 < K; g0 += GROUP) {
        int g1 = g0 + GROUP < K ? g0 + GROUP : K;
        // eta = max over nonzero products of (ea+eb)  (accumulator granted free)
        int eta = -1000000;
        for (int u = g0; u < g1; u++) {
            int sa, ma, ea, sb, mb, eb; decompose(a[u], sa, ma, ea); decompose(b[u], sb, mb, eb);
            if (ma && mb) { int e = ea + eb; if (e > eta) eta = e; }
        }
        if (eta == -1000000) continue;
        int unit = eta - W;
        for (int u = g0; u < g1; u++) {
            int sa, ma, ea, sb, mb, eb; decompose(a[u], sa, ma, ea); decompose(b[u], sb, mb, eb);
            if (!ma || !mb) continue;
            long P = (long)ma * mb;                 // product significand (exact, < 2^22)
            int sh = (ea + eb) - 20 - unit;         // product LSB is 2^(ea+eb-20)
            if (sh < 0) {
                int s = -sh; if (s > 62) s = 62;
                long discarded = P & ((1L << s) - 1);  // the bits the device truncates
                correction += (sa * sb) * discarded;   // rho_u contribution
            }
        }
    }
    OUT[idx] = (float)correction;  // prevents dead-code elimination
}

static double time_ms(cudaEvent_t s, cudaEvent_t e) { float ms = 0; cudaEventElapsedTime(&ms, s, e); return ms; }

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 512;
    int N = argc > 2 ? atoi(argv[2]) : 512;
    int K = argc > 3 ? atoi(argv[3]) : 1024;
    int iters = argc > 4 ? atoi(argv[4]) : 50;
    if (M % 16 || N % 8 || K % 16) { fprintf(stderr, "need M%%16==0, N%%8==0, K%%16==0\n"); return 1; }

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d), %d SMs\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("tile M=%d N=%d K=%d, %d timed iters; products = M*N*K = %.3g\n\n", M, N, K, iters, (double)M * N * K);

    // Random finite FP16 operands.
    std::vector<unsigned short> hA((long)M * K), hB((long)N * K);
    srand(1);
    auto rnd_fp16 = []() { int e = 1 + rand() % 29, m = rand() % 1024, s = (rand() & 1) << 15; return (unsigned short)(s | (e << 10) | m); };
    for (auto& x : hA) x = rnd_fp16();
    for (auto& x : hB) x = rnd_fp16();

    __half *dA, *dB; float *dD, *dOUT;
    CK(cudaMalloc(&dA, sizeof(__half) * hA.size()));
    CK(cudaMalloc(&dB, sizeof(__half) * hB.size()));
    CK(cudaMalloc(&dD, sizeof(float) * (long)M * N));
    CK(cudaMalloc(&dOUT, sizeof(float) * (long)M * N));
    CK(cudaMemcpy(dA, hA.data(), sizeof(__half) * hA.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), sizeof(__half) * hB.size(), cudaMemcpyHostToDevice));

    dim3 gg(N / 8, M / 16), gb(32, 1);
    int tpb = 256; long cells = (long)M * N; int ab = (int)((cells + tpb - 1) / tpb);

    cudaEvent_t s, e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    // warmup
    honest_gemm<<<gg, gb>>>(dA, dB, dD, M, N, K);
    attack_corrections<<<ab, tpb>>>(dA, dB, dOUT, M, N, K);
    CK(cudaDeviceSynchronize());

    CK(cudaEventRecord(s));
    for (int t = 0; t < iters; t++) honest_gemm<<<gg, gb>>>(dA, dB, dD, M, N, K);
    CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
    double honest = time_ms(s, e) / iters;

    CK(cudaEventRecord(s));
    for (int t = 0; t < iters; t++) attack_corrections<<<ab, tpb>>>(dA, dB, dOUT, M, N, K);
    CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
    double attack = time_ms(s, e) / iters;
    CK(cudaGetLastError());

    double macs = 2.0 * (double)M * N * K;
    printf("honest tensor-core GEMM : %8.3f ms/iter  (%.1f GFLOP/s effective)\n", honest, macs / (honest * 1e6));
    printf("attack per-product corr : %8.3f ms/iter  (CUDA cores, elementwise)\n", attack);
    printf("\nMEASURED attack / honest wall-time ratio : %.1fx\n", attack / honest);
    printf("  (the attacker pays this to reconstruct one tile's corrections even\n");
    printf("   with the exact sums and breakpoint oracle granted for free)\n");
    return 0;
}
