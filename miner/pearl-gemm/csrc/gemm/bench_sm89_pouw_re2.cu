// SPDX-License-Identifier: see LICENSE
//
// FULL-PoUW throughput bench for the sm_89 Pearl pipeline at the PRODUCTION
// mining config (tile 128x256x128, R=256). This is the apples-to-apples
// counterpart to lpminer's `--pearl-bench` (full PoUW: noisingA + noisingB +
// int8 GEMM + denoise + BLAKE3 PoW transcript hash + target check), which on a
// real 4070 Ti SUPER reports ~134 tmac_s / ~1.9 attempts/sec at
// M=N=131072, K=4096, R=256.
//
// Pipeline (4 logical kernels, matching bench_sm89_noisy_gemm_e2e.cu):
//   1. noisingA : ApEA = A + EAL@EAR ; AxEBL = A@EBL          (per-CTA, full K)
//   2. noisingB : BpEB = B + EBR@EBL ; EARxBpEB = EAR@BpEB    (symmetric)
//   3. denoise+GEMM+PoW : C = (ApEA @ BpEB^T) denoised by the four R-factor
//        corrections, with the BLAKE3 transcript accumulator folded over every
//        k_block (SkipReduction=false) and a final keyed-BLAKE3 target check.
//        This is pearl_gemm_sm89_pow_128x256x128_R256 (the despilled mainloop).
//
// R=256 noising: the existing sm_89 noising kernels are validated only for
// R in {64,128} (single-pass, register-resident AxEBL accumulator; a single
// R=256 pass would blow the register file). We therefore run noising as TWO
// R=128 passes (R-halves [0:128) and [128:256)), reusing the validated R=128
// kernels and writing each half into the (M,256)/(N,256) factor buffers. The
// noising cost is < 1% of total pipeline MACs at the mining shape, so the
// two-launch overhead is negligible for the tmac_s figure.
//
// MAC accounting (per attempt = one full pipeline pass), identical to the
// e2e bench so the tmac_s is directly comparable to lpminer:
//   noisingA:           2*M*K*R
//   noisingB:           2*N*K*R
//   denoise+GEMM:       M*N*(K + 2*R)
//   Total MACs = 2*K*R*(M+N) + M*N*(K + 2*R)
//   tmac_s     = Total MACs / time / 1e12
// (lpminer reports tmac_s on this same total-work definition; the dominant
//  term by far is the main GEMM M*N*K.)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>
#include "cute/tensor.hpp"
#include "kernel_traits_sm89.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---- noising trampolines (R=128 int32 variants) --------------------------
extern "C" void pearl_noisingA_sm89_64x128x64_R128_int32(
    int8_t const* A, int8_t const* EAL, int8_t const* EAR, int8_t const* EBL,
    int8_t* ApEA, int32_t* AxEBL, int M, int K, cudaStream_t stream);

namespace pearl {
namespace sm89 {
extern "C" void pearl_noisingB_sm89_64x128x64_R128_int32(
    int8_t const* B, int8_t const* EBR, int8_t const* EBL, int8_t const* EAR,
    int8_t* BpEB, int32_t* EARxBpEB, int N, int K, cudaStream_t stream);

// ---- full-PoUW main GEMM (denoise + BLAKE3 transcript + target check) ----
extern "C" void pearl_gemm_sm89_pow_128x256x128_R256(
    int8_t const* A, int64_t lda,
    int8_t const* B, int64_t ldb,
    cutlass::half_t* C, int64_t ldc,
    float const* A_scales,
    float const* B_scales,
    cutlass::half_t const* EAL,
    cutlass::half_t const* EBR,
    cutlass::half_t const* AxEBL,
    cutlass::half_t const* EARxBpEB,
    uint32_t const* pow_target,
    uint32_t const* pow_key,
    void* host_signal_sync,
    void* host_signal_header_pinned,
    uint64_t* inner_hash_counter,
    int M, int N, int K, cudaStream_t stream);

// Trait alias (must match pearl_gemm_sm89_pow_inst_128x256x128.cu) so we can
// statically probe sizeof(SharedStorage) for the PoW tile vs the 99 KB cap.
using PowTraits128x256x128_R256 = KernelTraitsSm89<
    int8_t, cutlass::half_t, cutlass::half_t, float,
    cute::Shape<cute::Int<128>, cute::Int<256>, cute::Int<128>, cute::Int<256>>,
    true, true, 1, 1, /*SkipReduction=*/false, /*SkipDenoising=*/false,
    /*kStages=*/2, /*EnableDebug=*/false, /*kRegisterResidentDenoise=*/true>;
}  // namespace sm89
}  // namespace pearl

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

static constexpr int R_DIM = 256;

// One full-PoUW pipeline pass at (M,N,K). Returns tmac_s; on the first shape
// optionally sanity-checks the output is finite + non-zero (5090 smoke).
static double bench(int M, int N, int K, int iters, bool sanity_check) {
    // ---- host buffers (int7-style random fill; bench times, denoise factor
    //      values don't affect timing or the finiteness of the main GEMM) ----
    std::vector<int8_t> hA(size_t(M)*K), hB(size_t(N)*K);
    std::vector<float>  hAs(M, 0.01f), hBs(N, 0.01f);
    std::srand(0);
    for (auto& v : hA) v = int8_t((std::rand() % 127) - 64);
    for (auto& v : hB) v = int8_t((std::rand() % 127) - 64);

    int8_t  *dA, *dB, *dApEA, *dBpEB;
    int8_t  *dEAL_i8, *dEBR_i8, *dEAR_R, *dEBL_R, *dEAR_K, *dEBL_K;
    int32_t *dAxEBL_i32, *dEARxBpEB_i32;
    cutlass::half_t *dEAL_fp16, *dEBR_fp16, *dAxEBL_fp16, *dEARxBpEB_fp16;
    float   *dAs, *dBs;
    cutlass::half_t *dC;
    uint32_t *dTarget, *dKey;
    void *dSignalSync, *dSignalHeader;

    CUCHK(cudaMalloc(&dA, hA.size()));
    CUCHK(cudaMalloc(&dB, hB.size()));
    CUCHK(cudaMalloc(&dApEA, size_t(M)*K));
    CUCHK(cudaMalloc(&dBpEB, size_t(N)*K));
    // noising int8 factor inputs (R-major / K-major). Values irrelevant for timing.
    CUCHK(cudaMalloc(&dEAL_i8, size_t(M)*R_DIM));
    CUCHK(cudaMalloc(&dEBR_i8, size_t(N)*R_DIM));
    CUCHK(cudaMalloc(&dEAR_R,  size_t(K)*R_DIM));
    CUCHK(cudaMalloc(&dEBL_R,  size_t(K)*R_DIM));
    CUCHK(cudaMalloc(&dEAR_K,  size_t(R_DIM)*K));
    CUCHK(cudaMalloc(&dEBL_K,  size_t(R_DIM)*K));
    // noising int32 outputs (M,R)/(N,R).
    CUCHK(cudaMalloc(&dAxEBL_i32,    size_t(M)*R_DIM*4));
    CUCHK(cudaMalloc(&dEARxBpEB_i32, size_t(N)*R_DIM*4));
    // denoise fp16 factor buffers consumed by the PoW epilogue.
    CUCHK(cudaMalloc(&dEAL_fp16,      size_t(M)*R_DIM*2));
    CUCHK(cudaMalloc(&dEBR_fp16,      size_t(N)*R_DIM*2));
    CUCHK(cudaMalloc(&dAxEBL_fp16,    size_t(M)*R_DIM*2));
    CUCHK(cudaMalloc(&dEARxBpEB_fp16, size_t(N)*R_DIM*2));
    CUCHK(cudaMalloc(&dAs, size_t(M)*4));
    CUCHK(cudaMalloc(&dBs, size_t(N)*4));
    CUCHK(cudaMalloc(&dC, size_t(M)*N*2));
    CUCHK(cudaMalloc(&dTarget, 8*4));
    CUCHK(cudaMalloc(&dKey, 8*4));
    CUCHK(cudaMalloc(&dSignalSync, 64));
    CUCHK(cudaMalloc(&dSignalHeader, 4096));

    CUCHK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dAs, hAs.data(), size_t(M)*4, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dBs, hBs.data(), size_t(N)*4, cudaMemcpyHostToDevice));
    // Init misc buffers to small finite values.
    CUCHK(cudaMemset(dEAL_i8, 1, size_t(M)*R_DIM));
    CUCHK(cudaMemset(dEBR_i8, 1, size_t(N)*R_DIM));
    CUCHK(cudaMemset(dEAR_R, 0, size_t(K)*R_DIM));
    CUCHK(cudaMemset(dEBL_R, 0, size_t(K)*R_DIM));
    CUCHK(cudaMemset(dEAR_K, 0, size_t(R_DIM)*K));
    CUCHK(cudaMemset(dEBL_K, 0, size_t(R_DIM)*K));
    CUCHK(cudaMemset(dEAL_fp16, 0, size_t(M)*R_DIM*2));
    CUCHK(cudaMemset(dEBR_fp16, 0, size_t(N)*R_DIM*2));
    CUCHK(cudaMemset(dAxEBL_fp16, 0, size_t(M)*R_DIM*2));
    CUCHK(cudaMemset(dEARxBpEB_fp16, 0, size_t(N)*R_DIM*2));
    // PoW target = all-zeros => hash > target => block NOT found => steady-state
    // mining (the BLAKE3 hash is still fully computed every k-tile; only the
    // rare header-writeback path is skipped, exactly as in real mining).
    CUCHK(cudaMemset(dTarget, 0, 8*4));
    CUCHK(cudaMemset(dKey, 0, 8*4));
    CUCHK(cudaMemset(dSignalSync, 0, 64));
    CUCHK(cudaMemset(dSignalHeader, 0, 4096));

    auto run_iter = [&]() {
        // noisingA / noisingB as two R=128 passes (R-halves), reusing the
        // validated R=128 kernels. Half h writes columns [h*128, h*128+128) of
        // the (M,256)/(N,256) factor outputs.
        for (int h = 0; h < 2; ++h) {
            int roff = h * 128;
            pearl_noisingA_sm89_64x128x64_R128_int32(
                dA,
                dEAL_i8 + size_t(0)*R_DIM + roff,   // EAL[:, roff:]  (R-major, ld=R)
                dEAR_R  + size_t(0)*R_DIM + roff,   // EAR[:, roff:]
                dEBL_K  + size_t(roff)*K,           // EBL[roff:, :]  (K-major, ld=K)
                dApEA,
                dAxEBL_i32 + size_t(0)*R_DIM + roff,
                M, K, 0);
            pearl::sm89::pearl_noisingB_sm89_64x128x64_R128_int32(
                dB,
                dEBR_i8 + size_t(0)*R_DIM + roff,
                dEBL_R  + size_t(0)*R_DIM + roff,
                dEAR_K  + size_t(roff)*K,
                dBpEB,
                dEARxBpEB_i32 + size_t(0)*R_DIM + roff,
                N, K, 0);
        }
        pearl::sm89::pearl_gemm_sm89_pow_128x256x128_R256(
            dApEA, K, dBpEB, K, dC, N, dAs, dBs,
            dEAL_fp16, dEBR_fp16, dAxEBL_fp16, dEARxBpEB_fp16,
            dTarget, dKey, dSignalSync, dSignalHeader, nullptr,
            M, N, K, 0);
    };

    for (int i = 0; i < 3; ++i) run_iter();
    cudaError_t le = cudaDeviceSynchronize();
    if (le != cudaSuccess) {
        printf("  M=%d N=%d K=%d  LAUNCH FAILED: %s\n",
               M, N, K, cudaGetErrorString(le));
        cudaGetLastError();
        return -1.0;
    }

    if (sanity_check) {
        std::vector<cutlass::half_t> hC(16);
        CUCHK(cudaMemcpy(hC.data(), dC, 16*2, cudaMemcpyDeviceToHost));
        int nonzero = 0, finite = 0;
        for (int i = 0; i < 16; ++i) {
            float f = float(hC[i]);
            if (std::isfinite(f)) ++finite;
            if (f != 0.f) ++nonzero;
        }
        printf("  [smoke] C[0:16] finite=%d/16 nonzero=%d/16  C[0]=%.3f C[1]=%.3f  %s\n",
               finite, nonzero, float(hC[0]), float(hC[1]),
               (finite == 16 && nonzero > 0) ? "OK" : "SUSPECT");
    }

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i) run_iter();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);

    double seconds = double(ms) / 1000.0 / double(iters);
    double macs = 2.0 * double(K) * double(R_DIM) * (double(M) + double(N))
                + double(M) * double(N) * (double(K) + 2.0 * double(R_DIM));
    double tmac_s = macs / seconds * 1e-12;
    double main_tops = 2.0 * double(M) * double(N) * double(K) / seconds * 1e-12;
    double attempts_per_sec = 1.0 / seconds;

    printf("  M=%6d N=%6d K=%5d  %.3f ms/attempt  full_PoUW=%7.2f tmac_s  "
           "main_gemm=%7.2f TOPS  %.3f attempts/s\n",
           M, N, K, ms / iters, tmac_s, main_tops, attempts_per_sec);

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(dA); cudaFree(dB); cudaFree(dApEA); cudaFree(dBpEB);
    cudaFree(dEAL_i8); cudaFree(dEBR_i8);
    cudaFree(dEAR_R); cudaFree(dEBL_R); cudaFree(dEAR_K); cudaFree(dEBL_K);
    cudaFree(dAxEBL_i32); cudaFree(dEARxBpEB_i32);
    cudaFree(dEAL_fp16); cudaFree(dEBR_fp16);
    cudaFree(dAxEBL_fp16); cudaFree(dEARxBpEB_fp16);
    cudaFree(dAs); cudaFree(dBs); cudaFree(dC);
    cudaFree(dTarget); cudaFree(dKey); cudaFree(dSignalSync); cudaFree(dSignalHeader);
    return tmac_s;
}

int main(int argc, char** argv) {
    using namespace pearl::sm89;
    int dev = (argc >= 2) ? std::atoi(argv[1]) : 0;
    CUCHK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CUCHK(cudaGetDeviceProperties(&p, dev));
    printf("device %d: %s sm_%d%d  (%.1f GB)\n", dev, p.name, p.major, p.minor,
           p.totalGlobalMem / 1073741824.0);
    printf("reference: lpminer --pearl-bench full PoUW on 4070 Ti SUPER = "
           "~134 tmac_s / ~1.9 attempts/s @ M=N=131072 K=4096 R=256\n");

    // ---- Static smem-fit probe for the PoW tile vs sm_89 99 KB opt-in cap ----
    const size_t SMEM_POW = sizeof(typename PowTraits128x256x128_R256::SharedStorage);
    const size_t ADA_CAP  = 101376;  // 99 KB
    printf("\n=== PoW (128x256x128 R=256, SkipReduction=false) SharedStorage ===\n");
    printf("  sizeof(SharedStorage) = %zu bytes (%.3f KB)   sm_89 cap = %zu  -> %s\n\n",
           SMEM_POW, SMEM_POW / 1024.0, ADA_CAP,
           SMEM_POW <= ADA_CAP ? "FITS" : "EXCEEDS CAP");

    // ---- Mining shape + a couple smaller shapes ----
    // C = M*N*half is the largest buffer; 131072^2 * 2 = 32 GB does NOT fit in
    // 16 GB. The full mining shape only fits on >=34 GB cards (5090=32 GB also
    // cannot hold 131072^2 C). We therefore run the true mining shape ONLY if it
    // fits, else the user runs it on the 4070 Ti SUPER (16 GB) — wait, C alone
    // is 32 GB there too. lpminer does NOT materialize full C: it streams C per
    // tile and never stores 131072^2. Our epilogue DOES store C, so for the
    // bench we use the largest M=N that fits the card while keeping K=4096,R=256
    // (the per-attempt tmac_s is shape-stable once the GEMM is large enough to
    // saturate). The user can pass a custom shape via argv.
    printf("Full-PoUW pipeline sweep (noisingA + noisingB + GEMM+denoise+PoW):\n");

    // Custom single shape: argv[2]=M argv[3]=N argv[4]=K [argv[5]=iters].
    // When provided, runs ONLY this shape (useful for a watchdog-safe smoke or
    // to bench the exact mining shape on a card that can hold C).
    if (argc >= 5) {
        int M = std::atoi(argv[2]), N = std::atoi(argv[3]), K = std::atoi(argv[4]);
        int it = (argc >= 6) ? std::atoi(argv[5]) : 3;
        bench(M, N, K, it, true);
        return 0;
    }

    struct Shape { int M, N, K, iters; };
    // C buffer sizes: M*N*2 bytes. 16384^2*2 = 512 MB; 24576^2*2 = 1.1 GB.
    Shape shapes[] = {
        { 4096,  4096, 4096, 10},
        { 8192,  8192, 4096,  5},
        {16384, 16384, 4096,  3},
        {24576, 24576, 4096,  2},
    };
    bool first = true;
    for (auto& s : shapes) {
        bench(s.M, s.N, s.K, s.iters, first);
        first = false;
    }
    return 0;
}
