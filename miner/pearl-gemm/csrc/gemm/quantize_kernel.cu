// Based on vLLM:
//  * https://github.com/vllm-project/vllm/blob/main/csrc/quantization/w8a8/int8/scaled_quant.c
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include <torch/all.h>
#include <cub/cub.cuh>
#include <cuda/functional>
#include "quantization_util.cuh"
#include "quantize_kernel.hpp"

using pearl::vectorize_read_with_alignment;
using pearl::vectorize_read_with_aux_global;
using pearl::vectorize_with_alignment;
using pearl::vectorize_with_aux_global;

constexpr int COMPILE_TIME_STRIDE = 256;

static CUTLASS_DEVICE int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
  float dst = std::nearbyint(x);

  // saturate
  // See https://github.com/pytorch/pytorch/issues/127666
  // See https://github.com/llvm/llvm-project/issues/95183
  // hip-clang std::clamp __glibcxx_assert_fail host function when building on
  // Arch/gcc14. The following replaces std::clamp usage with similar logic
  // dst = std::clamp(dst, i8_min, i8_max);
  dst = (dst < i8_min) ? i8_min : (dst > i8_max) ? i8_max : dst;
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

// Divide in an IEEE754 compatible way, even if compiling with fast-math
static CUTLASS_DEVICE float div_ieee754(const float x, const float y) {
  float r;
  asm volatile("div.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(x), "f"(y));
  return r;
}

// Original single-row-per-block kernel (kept for small token counts)
template <typename scalar_t, typename scale_t, typename SmoothScaleT,
          bool FastMath, bool UseSmoothScale, int MaxVal>
CUTLASS_GLOBAL void dynamic_scaled_quant_kernel(
    const scalar_t* __restrict__ input,
    const SmoothScaleT* __restrict__ smooth_scale, int8_t* __restrict__ output,
    scale_t* __restrict__ scale_out, const int hidden_size) {
  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  const int64_t token_idx = blockIdx.x;

  const scalar_t* row_in = input + token_idx * hidden_size;
  int8_t* row_out = output + token_idx * hidden_size;

  // Find row maximum for dynamic scaling
  // VEC_SIZE is number of elements per 32-byte vector load
  constexpr int VEC_SIZE = 32 / sizeof(scalar_t);
  float thread_max = 0.f;
  if constexpr (UseSmoothScale) {
    vectorize_read_with_aux_global<VEC_SIZE>(
        row_in, smooth_scale, hidden_size, tid, stride,
        [&thread_max] __device__(const scalar_t& src,
                                 const SmoothScaleT& smooth_val) {
          const float v =
              fabsf(static_cast<float>(src) * static_cast<float>(smooth_val));
          thread_max = fmaxf(thread_max, v);
        });
  } else {
    vectorize_read_with_alignment<VEC_SIZE>(
        row_in, hidden_size, tid, stride,
        [&thread_max] __device__(const scalar_t& src) {
          float v = fabsf(static_cast<float>(src));
          thread_max = fmaxf(thread_max, v);
        });
  }

  // CUB block reduction across all threads
  using BlockReduce = cub::BlockReduce<float, COMPILE_TIME_STRIDE>;
  __shared__ typename BlockReduce::TempStorage tmp;
  float block_max =
      BlockReduce(tmp).Reduce(thread_max, ::cuda::maximum<>{}, blockDim.x);

  __shared__ float scale;
  __shared__ bool is_zero;
  if (tid == 0) {
    is_zero = (block_max == 0.f);
    // testing showed that the fast-math operator is accurate in this operation
    scale = is_zero ? 0.f : (block_max / MaxVal);
    scale_out[token_idx] = scale;
  }
  __syncthreads();

  // Copy shared to constant (hopefully register)
  const float scale_ = scale;
  const bool is_zero_ = is_zero;

  // Quantize with smooth scale applied
  if constexpr (UseSmoothScale) {
    vectorize_with_aux_global<VEC_SIZE>(
        row_in, smooth_scale, row_out, hidden_size, tid, stride,
        [is_zero_, scale_] __device__(int8_t & dst, const scalar_t& src,
                                      const SmoothScaleT& smooth_val) {
          if (is_zero_) {
            dst = 0;
          } else {
            // smooth_val contains reciprocal, so multiply instead of divide
            const float dividend =
                static_cast<float>(src) * static_cast<float>(smooth_val);
            const float quotient = [&]() {
              if constexpr (FastMath) {
                return dividend / scale_;
              } else {
                return div_ieee754(dividend, scale_);
              }
            }();
            dst = float_to_int8_rn(quotient);
          }
        });
  } else {
    vectorize_with_alignment<VEC_SIZE>(
        row_in, row_out, hidden_size, tid, stride,
        [is_zero_, scale_] __device__(int8_t & dst, const scalar_t& src) {
          if (is_zero_) {
            dst = 0;
          } else {
            const float dividend = static_cast<float>(src);
            const float quotient = [&]() {
              if constexpr (FastMath) {
                return dividend / scale_;
              } else {
                return div_ieee754(dividend, scale_);
              }
            }();
            dst = float_to_int8_rn(quotient);
          }
        });
  }
}

template <typename scalar_t, typename SmoothScaleT, bool FastMath,
          bool UseSmoothScale, int MaxVal>
void run_quantize_kernel(const scalar_t* ptr_data,
                         const SmoothScaleT* ptr_smooth_scale, int8_t* ptr_x_q,
                         float* ptr_x_s, int num_tokens, int hidden_size,
                         cudaStream_t stream) {
  dim3 grid(num_tokens);
  dim3 block(COMPILE_TIME_STRIDE);

  dynamic_scaled_quant_kernel<scalar_t, float, SmoothScaleT, FastMath,
                              UseSmoothScale, MaxVal>
      <<<grid, block, 0, stream>>>(ptr_data, ptr_smooth_scale, ptr_x_q, ptr_x_s,
                                   hidden_size);
}

// ============================================================================
// X-Macro based explicit template instantiations
// ============================================================================
//
// Total instantiations: 32
// - UseSmoothScale=false: 2 scalar_t × 1 SmoothScaleT(float) × 2 FastMath × 2 MaxVal = 8
// - UseSmoothScale=true:  2 scalar_t × 3 SmoothScaleT × 2 FastMath × 2 MaxVal = 24

// Base instantiation macro
#define INSTANTIATE_QUANTIZE_KERNEL(scalar_t, smooth_t, FastMath, UseSmooth, \
                                    MaxVal)                                  \
  template void                                                              \
  run_quantize_kernel<scalar_t, smooth_t, FastMath, UseSmooth, MaxVal>(      \
      const scalar_t*, const smooth_t*, int8_t*, float*, int, int,           \
      cudaStream_t);

// Expand FastMath × MaxVal combinations for a given (scalar_t, smooth_t, UseSmooth)
#define INSTANTIATE_FAST_MATH_MAX_VAL(scalar_t, smooth_t, UseSmooth) \
  INSTANTIATE_QUANTIZE_KERNEL(scalar_t, smooth_t, false, UseSmooth,  \
                              MAX_VAL_7BIT)                          \
  INSTANTIATE_QUANTIZE_KERNEL(scalar_t, smooth_t, false, UseSmooth,  \
                              MAX_VAL_8BIT)                          \
  INSTANTIATE_QUANTIZE_KERNEL(scalar_t, smooth_t, true, UseSmooth,   \
                              MAX_VAL_7BIT)                          \
  INSTANTIATE_QUANTIZE_KERNEL(scalar_t, smooth_t, true, UseSmooth, MAX_VAL_8BIT)

// UseSmoothScale=false: only float as dummy SmoothScaleT
#define INSTANTIATE_NO_SMOOTH(scalar_t) \
  INSTANTIATE_FAST_MATH_MAX_VAL(scalar_t, float, false)

// UseSmoothScale=true: expand for a specific SmoothScaleT across all scalar_t
#define INSTANTIATE_WITH_SMOOTH_TYPE(smooth_t)                   \
  INSTANTIATE_FAST_MATH_MAX_VAL(cutlass::half_t, smooth_t, true) \
  INSTANTIATE_FAST_MATH_MAX_VAL(cutlass::bfloat16_t, smooth_t, true)

// Expand UseSmoothScale=false for all scalar types
INSTANTIATE_NO_SMOOTH(cutlass::half_t)
INSTANTIATE_NO_SMOOTH(cutlass::bfloat16_t)

// Expand UseSmoothScale=true for all SmoothScaleT types
INSTANTIATE_WITH_SMOOTH_TYPE(float)
INSTANTIATE_WITH_SMOOTH_TYPE(cutlass::half_t)
INSTANTIATE_WITH_SMOOTH_TYPE(cutlass::bfloat16_t)

// Clean up macros
#undef INSTANTIATE_QUANTIZE_KERNEL
#undef INSTANTIATE_FAST_MATH_MAX_VAL
#undef INSTANTIATE_NO_SMOOTH
#undef INSTANTIATE_WITH_SMOOTH_TYPE
