// SPDX-License-Identifier: see LICENSE
//
// Wave-4 bench: group-of-16 swizzle × persistent-over-output-tiles combination.
//
// CLI: ./bench_group16_plus_persistent <device> <M> <N> <K> <iters>
//
// Reads these env vars (all consumed inside pearl_gemm_sm89_run on first call):
//   PEARL_SM89_SWIZZLE        — fixed swizzle width (1..256), overrides adaptive.
//   PEARL_SM89_SWIZZLE_NMAJ   — '0' = M-major, '1' = N-major, unset = adaptive.
//   PEARL_SM89_GROUP16_SWIZZLE — '1' = alpha-style group16 N-major (sugar for
//                                SWIZZLE=16 + SWIZZLE_NMAJ=1).
//
// The PersistentSwizzledTileScheduler is *always* in use here — it's the
// production sm_89 path. So this bench measures group-of-K_swizzle (any K) IN
// COMBINATION with persistent-over-output-tiles, exactly the combo the wave-3
// bench could not isolate from "group16 alone" because wave-3 only varied the
// PEARL_SM89_GROUP16_SWIZZLE flag.
//
// Reports MEDIAN over `iters` per-iter times (cudaEvent), plus min/max +
// main_gemm TOPS = 2*M*N*K / median.

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

    // Warmup — drains static-init paths AND warms the GPU clock state.
    // On idle systems the clock drops to ~855 MHz between launches; need
    // ~20+ iters of work to lock the boost clock. 30 iters is a safety margin.
    for (int i = 0; i < 30; ++i) run();
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
    char const* env_g16 = std::getenv("PEARL_SM89_GROUP16_SWIZZLE");
    char const* env_swz = std::getenv("PEARL_SM89_SWIZZLE");
    char const* env_nmj = std::getenv("PEARL_SM89_SWIZZLE_NMAJ");
    printf("M=%6d N=%6d K=%5d  G16=%s SWZ=%s NMAJ=%s  med=%.3f ms (min %.3f / max %.3f)  main_gemm=%.2f TOPS\n",
           M, N, K,
           env_g16 ? env_g16 : "0",
           env_swz ? env_swz : "(auto)",
           env_nmj ? env_nmj : "(auto)",
           median, minv, maxv, tops_main);
    // CSV-friendly tail line so the driver script can parse without regex.
    // Variant tag: <swz>x<nmaj-side> (e.g. 16xN, 32xM, autoxauto, g16xN)
    char tag[32];
    if (env_g16 && env_g16[0] == '1') {
        std::snprintf(tag, sizeof(tag), "g16xN");
    } else {
        char const* swz_s = env_swz ? env_swz : "auto";
        char const* nmj_s = "auto";
        if (env_nmj) {
            if (env_nmj[0] == '0') nmj_s = "M";
            else if (env_nmj[0] == '1') nmj_s = "N";
        }
        std::snprintf(tag, sizeof(tag), "%sx%s", swz_s, nmj_s);
    }
    printf("CSV,%d,%d,%d,%s,%.4f,%.2f\n",
           M, N, K, tag,
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
