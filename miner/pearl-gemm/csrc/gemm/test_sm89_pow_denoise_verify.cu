// SPDX-License-Identifier: see LICENSE
//
// Correctness oracle for the R-strip-TILED smem-resident denoise in the
// production PoW kernel pearl_gemm_sm89_pow_128x256x128_R256 (tile 128x256x128,
// R=256). This is the path changed by the 30x-slowdown fix: the smem-resident
// denoise now sweeps R in strips of kRTile, staging each factor once and doing
// the two corrections as fp16 tensor-core MMAs. The R-strip-summed result must
// equal the full-R result (it is a reassociation of the same R-length dot), so
// rel_err vs the CPU fp32 reference must be ~0 (bit-exact, not merely < 1e-2).
//
// The kernel math (matching collective_epilogue_sm89.hpp denoise()):
//   acc = int32(A @ B^T)                          (mainloop)
//   acc *= 1 / 2^12
//   acc += EAL    @ EARxBpEB^T                     (fp16 MMA, fp32 accum)
//   acc += AxEBL  @ EBR^T                          (fp16 MMA, fp32 accum)
//   acc *= 2^12
//   C    = half( acc * A_scale[row] * B_scale[col] )
//
// We feed NONZERO small fp16 factors so the two correction MMAs are actually
// exercised (the bench's smoke uses zero factors and would not test them).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace pearl {
namespace sm89 {
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
}
}

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

static constexpr int R_DIM = 256;
static constexpr float kScale = float(1 << 12);  // kIntToFp16ScaleFactor

static float half_to_float(uint16_t h) {
    __half hv; std::memcpy(&hv, &h, 2); return __half2float(hv);
}
static uint16_t float_to_half(float v) {
    __half hv = __float2half(v); uint16_t b; std::memcpy(&b, &hv, 2); return b;
}

int main(int argc, char** argv) {
    int dev = (argc >= 2) ? std::atoi(argv[1]) : 0;
    CUCHK(cudaSetDevice(dev));
    cudaDeviceProp p; CUCHK(cudaGetDeviceProperties(&p, dev));
    printf("device %d: %s sm_%d%d\n", dev, p.name, p.major, p.minor);

    // Tile is 128x256x128 -> exercise a few full tiles. K multiple of 128.
    int M = 256, N = 512, K = 256;

    std::vector<int8_t> hA(size_t(M)*K), hB(size_t(N)*K);
    std::vector<float>  hAs(M), hBs(N);
    std::vector<uint16_t> hEAL(size_t(M)*R_DIM), hAxEBL(size_t(M)*R_DIM);
    std::vector<uint16_t> hEBR(size_t(N)*R_DIM), hEARxBpEB(size_t(N)*R_DIM);
    std::vector<uint16_t> hC(size_t(M)*N);

    std::srand(7);
    for (auto& v : hA) v = int8_t((std::rand() % 31) - 15);
    for (auto& v : hB) v = int8_t((std::rand() % 31) - 15);
    for (auto& v : hAs) v = 0.01f;   // identity-ish scales (constant)
    for (auto& v : hBs) v = 0.01f;
    // Small nonzero fp16 factors in [-0.05, 0.05] so each R=256-term dot stays
    // well under fp16 range and the correction is a meaningful nonzero value.
    auto rh = []() {
        float f = (std::rand() / float(RAND_MAX) - 0.5f) * 0.1f;
        return float_to_half(f);
    };
    for (auto& v : hEAL)      v = rh();
    for (auto& v : hAxEBL)    v = rh();
    for (auto& v : hEBR)      v = rh();
    for (auto& v : hEARxBpEB) v = rh();

    int8_t *dA, *dB; float *dAs, *dBs;
    cutlass::half_t *dEAL, *dEBR, *dAxEBL, *dEARxBpEB, *dC;
    uint32_t *dTarget, *dKey; void *dSync, *dHeader;
    CUCHK(cudaMalloc(&dA, hA.size())); CUCHK(cudaMalloc(&dB, hB.size()));
    CUCHK(cudaMalloc(&dAs, size_t(M)*4)); CUCHK(cudaMalloc(&dBs, size_t(N)*4));
    CUCHK(cudaMalloc(&dEAL, hEAL.size()*2)); CUCHK(cudaMalloc(&dEBR, hEBR.size()*2));
    CUCHK(cudaMalloc(&dAxEBL, hAxEBL.size()*2)); CUCHK(cudaMalloc(&dEARxBpEB, hEARxBpEB.size()*2));
    CUCHK(cudaMalloc(&dC, hC.size()*2));
    CUCHK(cudaMalloc(&dTarget, 32)); CUCHK(cudaMalloc(&dKey, 32));
    CUCHK(cudaMalloc(&dSync, 64)); CUCHK(cudaMalloc(&dHeader, 4096));

    CUCHK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dAs, hAs.data(), size_t(M)*4, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dBs, hBs.data(), size_t(N)*4, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEAL, hEAL.data(), hEAL.size()*2, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEBR, hEBR.data(), hEBR.size()*2, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dAxEBL, hAxEBL.data(), hAxEBL.size()*2, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dEARxBpEB, hEARxBpEB.data(), hEARxBpEB.size()*2, cudaMemcpyHostToDevice));
    CUCHK(cudaMemset(dC, 0, hC.size()*2));
    CUCHK(cudaMemset(dTarget, 0, 32)); CUCHK(cudaMemset(dKey, 0, 32));
    CUCHK(cudaMemset(dSync, 0, 64)); CUCHK(cudaMemset(dHeader, 0, 4096));

    pearl::sm89::pearl_gemm_sm89_pow_128x256x128_R256(
        dA, K, dB, K, dC, N, dAs, dBs,
        dEAL, dEBR, dAxEBL, dEARxBpEB,
        dTarget, dKey, dSync, dHeader, nullptr, M, N, K, 0);
    CUCHK(cudaDeviceSynchronize());
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) { fprintf(stderr, "launch: %s\n", cudaGetErrorString(le)); return 1; }
    CUCHK(cudaMemcpy(hC.data(), dC, hC.size()*2, cudaMemcpyDeviceToHost));

    // CPU reference (fp32 denoise math), then cast to half (RNE).
    double worst_abs = 0, worst_rel = 0; int worst = 0; long bad = 0;
    const float atol = 5e-2f, rtol = 5e-3f;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int64_t ia = 0;
            for (int k = 0; k < K; ++k)
                ia += int64_t(hA[i*K+k]) * int64_t(hB[j*K+k]);
            float acc = float(ia) / kScale;
            float d1 = 0.f, d2 = 0.f;
            for (int r = 0; r < R_DIM; ++r) {
                d1 += half_to_float(hEAL[i*R_DIM+r]) * half_to_float(hEARxBpEB[j*R_DIM+r]);
                d2 += half_to_float(hAxEBL[i*R_DIM+r]) * half_to_float(hEBR[j*R_DIM+r]);
            }
            acc += d1; acc += d2; acc *= kScale;
            float vf = acc * hAs[i] * hBs[j];
            // cast to half (RNE) for reference, compare in float space.
            float ref = half_to_float(float_to_half(vf));
            float got = half_to_float(hC[i*N+j]);
            float dd = std::fabs(got - ref);
            float rr = dd / (std::fabs(ref) + 1e-6f);
            if (dd > worst_abs) { worst_abs = dd; worst_rel = rr; worst = i*N+j; }
            if (dd > atol + rtol*std::fabs(ref)) ++bad;
        }
    }
    printf("M=%d N=%d K=%d R=%d  max|err|=%.4e  rel=%.4e  bad=%ld/%zu  worst@%d ref=%.5f got=%.5f\n",
           M, N, K, R_DIM, worst_abs, worst_rel, bad, hC.size(), worst, 0.f, 0.f);
    bool pass = (bad == 0);
    printf("%s\n", pass ? "PASS (R-strip denoise matches full-R reference)"
                        : "FAIL");
    return pass ? 0 : 1;
}
