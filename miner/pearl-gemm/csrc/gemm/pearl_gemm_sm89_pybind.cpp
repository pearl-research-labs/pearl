// SPDX-License-Identifier: see LICENSE
//
// Minimal PyBind wrapper to call the sm_89 noiseless GEMM from Python.
// Used for bring-up validation against TestGEMM::test_noiseless_int7_gemm.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

namespace pearl {
namespace sm89 {

extern "C" void pearl_gemm_sm89_noiseless_128x128x128_R64(
    int8_t const* A, int64_t lda,
    int8_t const* B, int64_t ldb,
    cutlass::bfloat16_t* C, int64_t ldc,
    float const* A_scales,
    float const* B_scales,
    int M, int N, int K,
    cudaStream_t stream);

}  // namespace sm89
}  // namespace pearl

void gemm_sm89(torch::Tensor A, torch::Tensor B,
               torch::Tensor A_scales, torch::Tensor B_scales,
               torch::Tensor C) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
  TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
  TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bfloat16");
  TORCH_CHECK(A_scales.dtype() == torch::kFloat32, "A_scales must be fp32");
  TORCH_CHECK(B_scales.dtype() == torch::kFloat32, "B_scales must be fp32");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "2D tensors");

  int const M = A.size(0);
  int const K = A.size(1);
  int const N = B.size(0);
  TORCH_CHECK(B.size(1) == K, "B's K must match A's K (B is (N,K))");
  TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C shape mismatch");
  TORCH_CHECK(A_scales.size(0) == M, "A_scales size must equal M");
  TORCH_CHECK(B_scales.size(0) == N, "B_scales size must equal N");

  int const lda = A.stride(0);
  int const ldb = B.stride(0);
  int const ldc = C.stride(0);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  pearl::sm89::pearl_gemm_sm89_noiseless_128x128x128_R64(
      reinterpret_cast<int8_t const*>(A.data_ptr<int8_t>()), lda,
      reinterpret_cast<int8_t const*>(B.data_ptr<int8_t>()), ldb,
      reinterpret_cast<cutlass::bfloat16_t*>(C.data_ptr<at::BFloat16>()), ldc,
      A_scales.data_ptr<float>(), B_scales.data_ptr<float>(),
      M, N, K, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_sm89", &gemm_sm89,
        "Pearl sm_89 noiseless int8 GEMM, tile (128,128,128), R=64");
}
