// Bit-exact A100 (sm_80) FP16 fused noisy quantization.
//
// Reproduces the FP16 scheme verifier's per-row fused noisy quantize
// (zk-pow/src/api/fp16/quantization.rs) element-for-element:
//
//   row_norms(row):      sumsq = sequential-f32 sum of fp16_to_f32(x)^2
//                        l2  = grid4(RNE_bf16(sqrtf(sumsq / k)))
//                        linf = RNE_bf16(max |x|)
//   floor:               l2, linf = bf16_max(.,  f32_to_bf16(2^-32))
//   derive_row_scales:   noised_bound = bf16_fma(delta_r, l2, linf)
//                        alpha = bf16_div(MAX_FP16, noised_bound)
//                        beta  = bf16_mul(bf16_mul(alpha, l2), delta_over_std)
//   elementwise:         noised  = fmaf(alpha_f32, fp16_to_f32(x), beta_f32 * N)
//                        clamped = clamp(noised, -65504, 65504)
//                        code    = RNE_fp16(clamped)
//
// The noise tile N = E @ F^T is produced by the committed fp16_gemm_a100 kernel
// (passed in as an (num_rows x k) f32 tensor) and consumed here as N[i*k + j].
//
// Every BF16 op is done in f32 (operands decoded exactly) and rounded back to
// BF16 with the reference's exact RNE; the elementwise FMA is a single-rounding
// f32 fmaf; the final cast is RNE-to-FP16. FP32 ops that the reference rounds
// separately use __f*_rn intrinsics so nvcc never contracts a mul+add into an
// fma and changes a bit.

#include <cuda_fp16.h>
#include <math_constants.h>
#include <torch/extension.h>

// ---- BF16 helpers (bit-exact to crate::api::fp8::{compute,dtype}) ----

__device__ __forceinline__ float bf16_to_f32(unsigned short bits) {
    return __uint_as_float((unsigned int)bits << 16);
}

// f32 -> BF16 round-to-nearest-ties-to-even (dtype::f32_to_bf16).
__device__ __forceinline__ unsigned short f32_to_bf16(float x) {
    unsigned int bits = __float_as_uint(x);
    unsigned int round_bit = (bits >> 16) & 1u;
    return (unsigned short)((bits + 0x7FFFu + round_bit) >> 16);
}

__device__ __forceinline__ unsigned short bf16_mul(unsigned short a, unsigned short b) {
    return f32_to_bf16(bf16_to_f32(a) * bf16_to_f32(b));
}

__device__ __forceinline__ unsigned short bf16_div(unsigned short a, unsigned short b) {
    return f32_to_bf16(bf16_to_f32(a) / bf16_to_f32(b));
}

// torch.maximum: exact, returns one of the inputs (compute::bf16_max).
__device__ __forceinline__ unsigned short bf16_max(unsigned short a, unsigned short b) {
    return (bf16_to_f32(a) >= bf16_to_f32(b)) ? a : b;
}

// Single-rounding FMA a*b + c in BF16 (compute::bf16_fma): TwoSum in f64 +
// round-to-odd fixup, then f32 -> BF16 RNE. BF16 operands make a*b exact in f64.
__device__ __forceinline__ unsigned short bf16_fma(unsigned short a, unsigned short b,
                                                   unsigned short c) {
    double a64 = (double)bf16_to_f32(a);
    double b64 = (double)bf16_to_f32(b);
    double c64 = (double)bf16_to_f32(c);
    double p = a64 * b64;          // exact
    double s = p + c64;            // f64 RNE of the exact sum x = a*b + c
    double t = s - c64;
    double r = (p - t) + (c64 - (s - t));  // TwoSum residual: x - s, exact
    float s32 = (float)s;
    double err = (s - (double)s32) + r;    // sign(x - s32); nonzero iff x != s32
    bool fix = (err != 0.0) && ((__float_as_uint(s32) & 1u) == 0u) && isfinite(s32);
    float rounded = s32;
    if (fix) {
        rounded = (err > 0.0) ? nextafterf(s32, CUDART_INF_F)
                              : nextafterf(s32, -CUDART_INF_F);
    }
    return f32_to_bf16(rounded);
}

// prequant::round_l2_to_grid: round to nearest multiple of 4 ulps, ties up.
__device__ __forceinline__ unsigned short round_l2_to_grid(unsigned short l2) {
    return (unsigned short)((l2 + 2u) & ~3u);
}

__constant__ float MAX_FP16_F = 65504.0f;

// ---- Kernel 1: per-row norms + scale derivation (one thread per row) ----
//
// delta_r, delta_over_std, max_fp16_bf, floor_bf are host-precomputed BF16 bits
// (identical f64 path to the reference's delta_r_bf16 / delta_over_std_bf16).

__global__ void fp16_row_scales_kernel(const __half* __restrict__ rows, int num_rows, int k,
                                       unsigned short delta_r, unsigned short delta_over_std,
                                       unsigned short max_fp16_bf, unsigned short floor_bf,
                                       unsigned short* __restrict__ alpha_out,
                                       unsigned short* __restrict__ beta_out,
                                       unsigned short* __restrict__ l2_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows) return;

    const __half* row = rows + (long)i * k;
    float sumsq = 0.0f;
    float absmax = 0.0f;
    for (int j = 0; j < k; ++j) {
        float v = __half2float(row[j]);
        // Sequential f32 fold; forbid fma-contraction so mul and add round apart.
        sumsq = __fadd_rn(sumsq, __fmul_rn(v, v));
        absmax = fmaxf(absmax, fabsf(v));
    }
    float rms = __fsqrt_rn(__fdiv_rn(sumsq, (float)k));
    unsigned short l2 = round_l2_to_grid(f32_to_bf16(rms));
    unsigned short linf = f32_to_bf16(absmax);

    l2 = bf16_max(l2, floor_bf);
    linf = bf16_max(linf, floor_bf);

    unsigned short noised_bound = bf16_fma(delta_r, l2, linf);
    unsigned short alpha = bf16_div(max_fp16_bf, noised_bound);
    unsigned short beta = bf16_mul(bf16_mul(alpha, l2), delta_over_std);

    alpha_out[i] = alpha;
    beta_out[i] = beta;
    l2_out[i] = l2;
}

// ---- Kernel 2: fused elementwise noised quantize (one thread per element) ----

__global__ void fp16_noised_elementwise_kernel(const __half* __restrict__ rows,
                                               const float* __restrict__ noise,
                                               const unsigned short* __restrict__ alpha,
                                               const unsigned short* __restrict__ beta,
                                               int num_rows, int k, __half* __restrict__ out) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)num_rows * k;
    if (idx >= total) return;
    int i = (int)(idx / k);

    float af = bf16_to_f32(alpha[i]);
    float bf = bf16_to_f32(beta[i]);
    float x = __half2float(rows[idx]);
    // beta*N rounded first (f32), then a single-rounding fmaf = Rust mul_add.
    float bn = __fmul_rn(bf, noise[idx]);
    float noised = fmaf(af, x, bn);
    float clamped = noised < -MAX_FP16_F ? -MAX_FP16_F : (noised > MAX_FP16_F ? MAX_FP16_F : noised);
    out[idx] = __float2half_rn(clamped);
}

// ---- Host launchers ----

// delta_r_bf16 / delta_over_std_bf16: computed on host with the same f64 path.
static unsigned short host_f32_to_bf16(float x) {
    unsigned int bits;
    memcpy(&bits, &x, sizeof(bits));
    unsigned int round_bit = (bits >> 16) & 1u;
    return (unsigned short)((bits + 0x7FFFu + round_bit) >> 16);
}

std::vector<torch::Tensor> fp16_row_scales(torch::Tensor rows, long r) {
    TORCH_CHECK(rows.is_cuda() && rows.scalar_type() == torch::kFloat16, "rows must be fp16 CUDA");
    TORCH_CHECK(rows.dim() == 2 && rows.is_contiguous(), "rows must be 2D contiguous");
    int num_rows = rows.size(0), k = rows.size(1);

    const double NOISE_TARGET_NORM = 256.0;
    const double DELTA = 0.5;
    double sqrt_r = sqrt((double)r);
    unsigned short delta_r = host_f32_to_bf16((float)(DELTA * sqrt_r));
    unsigned short delta_over_std =
        host_f32_to_bf16((float)(DELTA * sqrt_r / (NOISE_TARGET_NORM * NOISE_TARGET_NORM)));
    unsigned short max_fp16_bf = host_f32_to_bf16(65504.0f);
    unsigned short floor_bf = host_f32_to_bf16(1.0f / 4294967296.0f);  // 2^-32

    auto opts = torch::dtype(torch::kInt16).device(rows.device());
    auto alpha = torch::empty({num_rows}, opts);
    auto beta = torch::empty({num_rows}, opts);
    auto l2 = torch::empty({num_rows}, opts);

    int threads = 128;
    int blocks = (num_rows + threads - 1) / threads;
    fp16_row_scales_kernel<<<blocks, threads>>>(
        (const __half*)rows.data_ptr(), num_rows, k, delta_r, delta_over_std, max_fp16_bf, floor_bf,
        (unsigned short*)alpha.data_ptr(), (unsigned short*)beta.data_ptr(),
        (unsigned short*)l2.data_ptr());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fp16_row_scales launch failed");
    return {alpha, beta, l2};
}

torch::Tensor fp16_noised_elementwise(torch::Tensor rows, torch::Tensor noise, torch::Tensor alpha,
                                      torch::Tensor beta) {
    TORCH_CHECK(rows.is_cuda() && rows.scalar_type() == torch::kFloat16, "rows must be fp16 CUDA");
    TORCH_CHECK(noise.is_cuda() && noise.scalar_type() == torch::kFloat32, "noise must be f32 CUDA");
    TORCH_CHECK(rows.dim() == 2 && rows.is_contiguous(), "rows must be 2D contiguous");
    TORCH_CHECK(noise.sizes() == rows.sizes() && noise.is_contiguous(), "noise must match rows");
    int num_rows = rows.size(0), k = rows.size(1);
    TORCH_CHECK(alpha.numel() == num_rows && beta.numel() == num_rows, "scales must be per-row");
    alpha = alpha.contiguous();
    beta = beta.contiguous();

    auto out = torch::empty({num_rows, k}, torch::dtype(torch::kFloat16).device(rows.device()));
    long total = (long)num_rows * k;
    int threads = 256;
    long blocks = (total + threads - 1) / threads;
    fp16_noised_elementwise_kernel<<<(unsigned)blocks, threads>>>(
        (const __half*)rows.data_ptr(), (const float*)noise.data_ptr(),
        (const unsigned short*)alpha.data_ptr(), (const unsigned short*)beta.data_ptr(), num_rows, k,
        (__half*)out.data_ptr());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fp16_noised_elementwise launch failed");
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp16_row_scales", &fp16_row_scales,
          "A100 sm_80 per-row FP16 noisy-quant norms+scales (bf16 alpha,beta,l2)",
          pybind11::arg("rows"), pybind11::arg("r"));
    m.def("fp16_noised_elementwise", &fp16_noised_elementwise,
          "A100 sm_80 fused noised elementwise quantize to FP16", pybind11::arg("rows"),
          pybind11::arg("noise"), pybind11::arg("alpha"), pybind11::arg("beta"));
}
