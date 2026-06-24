// Include these 2 headers instead of torch/extension.h since we don't need all
// of the torch headers.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/nn/functional.h>
#include <torch/python.h>

#include <cutlass/arch/arch.h>
#include <cutlass/numeric_types.h>

#include <cute/container/array.hpp>
#include "heuristics.hpp"
#include "host_signal_header.hpp"
#include "inner_hash_kernel.h"
#include "pearl_api_params.h"
#include "pearl_gemm_constants.hpp"
#include "pearl_gemm_decl.h"
#include "static_switch.h"

#include <cstdio>
#include "../blake3/blake3_constants.hpp"
#include "../moe/build_routing_data.cuh"
#include "../tensor_hash/tensor_hash_api.hpp"

#include <optional>

#include <iostream>

namespace PYBIND11_NAMESPACE {
namespace detail {

template <class T, size_t N>
struct type_caster<cute::array<T, N>> : array_caster<cute::array<T, N>, T, N> {
  static constexpr auto name =
      _("cute.Array[") + make_caster<T>::name + _<N>() + _("]");
};

}  // namespace detail
}  // namespace PYBIND11_NAMESPACE

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

int64_t get_host_signal_header_size() {
  return host_signal_header_size;
}

int64_t get_host_signal_sync_size() {
  return sizeof(HostSignalSync);
}

std::string print_range(const auto& begin, const auto& end, const char* name) {
  using T =
      typename std::iterator_traits<std::decay_t<decltype(begin)>>::value_type;
  std::stringstream ss;
  ss << name << "=(";
  for (auto it = begin; it != end; it++) {
    if constexpr (std::is_same_v<T, uint8_t>) {
      ss << static_cast<size_t>(*it);
    } else {
      ss << *it;
    }
    ss << (it != end - 1 ? ", " : ")");
  }
  return ss.str();
}

template <typename T, size_t N>
std::string print_array(const cute::array<T, N>& array, const char* name) {
  return print_range(array.begin(), array.end(), name);
}

// checks tensor is on a cuda device, is contiguous, and has (one of) the expected dtype(s)
void check_common(
    at::Tensor& tensor, at::ScalarType expected_dtype,
    std::optional<at::ScalarType> allowed_2nd_dtype = std::nullopt) {
  CHECK_DEVICE(tensor);
  CHECK_CONTIGUOUS(tensor);
  // build expected dtype message, may include a second allowed dtype
  std::string expected_dtypes_str = c10::toString(expected_dtype);
  if (allowed_2nd_dtype) {
    expected_dtypes_str += " or ";
    expected_dtypes_str += c10::toString(*allowed_2nd_dtype);
  }

  at::ScalarType tensor_dtype = tensor.scalar_type();

  TORCH_CHECK(tensor_dtype == expected_dtype ||
                  (allowed_2nd_dtype && tensor_dtype == *allowed_2nd_dtype),
              "expected dtype: ", expected_dtypes_str,
              "but got this dtype: ", c10::toString(tensor_dtype));
}

// in addition to check_common, also checks that the shape of the tensor is as expected (overloads for 1D,2D)
void check_tensor(
    at::Tensor& tensor, int x, at::ScalarType expected_dtype,
    std::optional<at::ScalarType> allowed_2nd_dtype = std::nullopt) {
  check_common(tensor, expected_dtype, allowed_2nd_dtype);
  CHECK_SHAPE(tensor, x);
}

void check_tensor(
    at::Tensor& tensor, int x, int y, at::ScalarType expected_dtype,
    std::optional<at::ScalarType> allowed_2nd_dtype = std::nullopt) {
  check_common(tensor, expected_dtype, allowed_2nd_dtype);
  CHECK_SHAPE(tensor, x, y);
}

void denoise_converter(
    const std::optional<at::Tensor>& EARxBpEB_in_,   //n x r, int32
    const std::optional<at::Tensor>& AxEBL_in_,      //m x r, int32
    const std::optional<at::Tensor>& EARxBpEB_out_,  //n x r, fp16
    const std::optional<at::Tensor>& AxEBL_out_      //m x r, fp16
) {

  at::Tensor EARxBpEB_in, EARxBpEB_out, AxEBL_in, AxEBL_out;
  int m{}, n{}, r{};
  bool convert_EARxBpEB = EARxBpEB_out_.has_value() && EARxBpEB_in_.has_value();
  bool convert_AxEBL = AxEBL_out_.has_value() && AxEBL_in_.has_value();
  TORCH_CHECK(convert_EARxBpEB || convert_AxEBL,
              "Need at least one int32 denoising factor to convert");

  if (convert_AxEBL) {
    AxEBL_in = AxEBL_in_.value();
    AxEBL_out = AxEBL_out_.value();
    m = int(AxEBL_in.size(0));
    r = int(AxEBL_in.size(1));
    check_tensor(AxEBL_in, m, r, torch::kInt32);
    check_tensor(AxEBL_out, m, r, torch::kFloat16);
  }

  if (convert_EARxBpEB) {
    EARxBpEB_in = EARxBpEB_in_.value();
    EARxBpEB_out = EARxBpEB_out_.value();
    n = int(EARxBpEB_in.size(0));
    r = int(EARxBpEB_in.size(1));
    check_tensor(EARxBpEB_in, n, r, torch::kInt32);
    check_tensor(EARxBpEB_out, n, r, torch::kFloat16);
  }
  PearlAPIParams params;

  params.ptr_AxEBL_int32 = convert_AxEBL ? AxEBL_in.data_ptr() : nullptr;
  params.ptr_AxEBL_mma = convert_AxEBL ? AxEBL_out.data_ptr() : nullptr;
  params.ptr_EARxBpEB_int32 =
      convert_EARxBpEB ? EARxBpEB_in.data_ptr() : nullptr;
  params.ptr_EARxBpEB_mma =
      convert_EARxBpEB ? EARxBpEB_out.data_ptr() : nullptr;
  params.m = m;
  params.n = n;
  params.r = r;

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (params.r == 64) {
    run_denoise_converter<64>(params, stream);
  } else if (params.r == 128) {
    run_denoise_converter<128>(params, stream);
  } else {
    TORCH_CHECK(false,
                "No denoise converter kernel found with given config: R = ", r);
  }
}

void noise_gen(const int64_t R, const int64_t num_threads,
               const std::optional<at::Tensor>& EAL_,          // m x r
               const std::optional<at::Tensor>& EAL_fp16_,     // m x r
               const std::optional<at::Tensor>& EAR_R_major_,  // k x r
               const std::optional<at::Tensor>& EAR_K_major_,  // r x k
               const std::optional<at::Tensor>& EBL_R_major_,  // k x r
               const std::optional<at::Tensor>& EBL_K_major_,  // r x k
               const std::optional<at::Tensor>& EBR_,          // n x r
               const std::optional<at::Tensor>& EBR_fp16_,     // n x r
               const std::optional<at::Tensor>& key_A_,        // KEY_LEN
               const std::optional<at::Tensor>& key_B_,        // KEY_LEN
               const std::optional<at::Tensor>& aux_buffer_)   // buffer_size
{
  at::Tensor EAL, EAL_fp16, EAR_R_major, EAR_K_major, EBL_R_major, EBL_K_major,
      EBR, EBR_fp16, key_A, key_B, aux_buffer;
  int m{}, n{}, k{};
  int device_id = -1;
  Noise_gen_params params;
  static constexpr int KEY_LEN = blake3::KEY_SIZE;
  // Because many combinations of tensors are possible in noisegen, each optional tensor needs to
  //  independently be checked if it has data or not, have problem shape and device info extracted
  //  and set for the kernel, and then checked for cuda device (and that all tensors are on the
  //  same device), contiguity, shape, and dtype.  We use these setter lambdas to do this.
  auto set_noise_matrix_ptr = [&](const std::optional<at::Tensor>& tensor_,
                                  at::Tensor& tensor, int& x, int& y,
                                  bool is_int8 = true) -> void* {
    if (tensor_.has_value()) {
      tensor = tensor_.value();
      int tensor_device_id = tensor.get_device();
      if (device_id == -1) {
        device_id = tensor_device_id;
      }
      TORCH_CHECK(device_id == tensor_device_id, "Device id mismatch");
      if (is_int8) {
        TORCH_CHECK(tensor.dtype() == torch::kInt8);
      }
      if (x == 0) {
        x = tensor.size(0);
      }
      if (y == 0) {
        y = tensor.size(1);
      }
      at::ScalarType expected_dtype = is_int8 ? torch::kInt8 : torch::kFloat16;
      check_tensor(tensor, x, y, expected_dtype);
      return tensor.data_ptr();
    } else {
      return nullptr;
    }
  };
  auto set_8b_ptr = [&](const std::optional<at::Tensor>& tensor_,
                        at::Tensor& tensor, bool expect_value) -> void* {
    TORCH_CHECK(expect_value == tensor_.has_value());
    if (tensor_.has_value()) {
      tensor = tensor_.value();
      int tensor_device_id = tensor.get_device();
      TORCH_CHECK(device_id == tensor_device_id, "Device id mismatch");
      check_tensor(tensor, KEY_LEN, torch::kUInt8);
      return tensor.data_ptr();
    } else {
      return nullptr;
    }
  };

  // non-const copy of R for setter functions
  int R_ = R;
  params.ptr_EAL = set_noise_matrix_ptr(EAL_, EAL, m, R_);
  params.ptr_EAL_fp16 = set_noise_matrix_ptr(EAL_fp16_, EAL_fp16, m, R_, false);

  params.ptr_EAR_R_major =
      set_noise_matrix_ptr(EAR_R_major_, EAR_R_major, k, R_);
  params.ptr_EAR_K_major =
      set_noise_matrix_ptr(EAR_K_major_, EAR_K_major, R_, k);
  params.ptr_EBL_R_major =
      set_noise_matrix_ptr(EBL_R_major_, EBL_R_major, k, R_);
  params.ptr_EBL_K_major =
      set_noise_matrix_ptr(EBL_K_major_, EBL_K_major, R_, k);
  params.ptr_EBR = set_noise_matrix_ptr(EBR_, EBR, n, R_);
  params.ptr_EBR_fp16 = set_noise_matrix_ptr(EBR_fp16_, EBR_fp16, n, R_, false);
  TORCH_CHECK(device_id != -1, "No noise matrices supplied");
  TORCH_CHECK(
      k % 16 == 0,
      "K must be divisible by 16 (the size of a 128b vectorized copy atom)");
  // could modify if we want B to generate EAR for example
  bool const expect_A =
      params.ptr_EAL || params.ptr_EAR_R_major || params.ptr_EAR_K_major;
  bool const expect_B =
      params.ptr_EBL_R_major || params.ptr_EBL_K_major || params.ptr_EBR;

  params.ptr_key_A = set_8b_ptr(key_A_, key_A, expect_A);
  params.ptr_key_B = set_8b_ptr(key_B_, key_B, expect_B);

  params.m = m;
  params.n = n;
  params.k = k;
  params.r = R;

  int aux_buffer_size{};
  if (aux_buffer_.has_value()) {
    aux_buffer = aux_buffer_.value();
    TORCH_CHECK(device_id == aux_buffer.get_device(), "Device id mismatch");
    aux_buffer_size = aux_buffer.size(0);
    TORCH_CHECK(aux_buffer_size % 4 == 0, "Aux buffer must be 16 byte aligned");
    check_tensor(aux_buffer, aux_buffer_size, torch::kUInt32, torch::kInt32);
    params.ptr_aux_buffer = aux_buffer.data_ptr();
  } else {
    params.ptr_aux_buffer = nullptr;
  }
  params.aux_buffer_size = aux_buffer_size;

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)device_id};

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  bool kernel_found = false;
  NUM_THREADS_SWITCH(
      num_threads, NumThreads,
      if (R == 64) {
        kernel_found = true;
        run_noise_generation<64, NumThreads>(params, stream);
      } else if (R == 128) {
        kernel_found = true;
        run_noise_generation<128, NumThreads>(params, stream);
      });

  if (!kernel_found) {
    TORCH_CHECK(false, "No noise_gen kernel found with given config: R = ", R,
                ", num_threads = ", num_threads);
  }
}

void noise_A(at::Tensor& A,                          // m x k
             at::Tensor& EAL,                        // m x r
             at::Tensor& AxEBL,                      // m x r
             at::Tensor& ApEA,                       // m x n
             const std::optional<at::Tensor>& EAR_,  // k x r
             const std::optional<at::Tensor>& EBL_,  // r x k
             std::optional<int64_t> tile_size_m_ = std::nullopt,
             std::optional<int64_t> tile_size_k_ = std::nullopt,
             int64_t pipeline_stages = 2,
             std::optional<int64_t> k_blocks_per_split_ = std::nullopt) {

  at::cuda::CUDAGuard device_guard{(char)A.get_device()};
  auto dprops = at::cuda::getCurrentDeviceProperties();

  at::Tensor EAR, EBL;

  int m = int(A.size(0));
  int n = 1;  // dummy value
  int k = int(A.size(1));
  int r = int(EAL.size(1));

  TORCH_CHECK(
      k % 16 == 0,
      "K must be divisible by 16 (the size of a 128b vectorized copy atom)");
  if (k_blocks_per_split_.has_value() && k_blocks_per_split_.value() > 0) {
    TORCH_CHECK(AxEBL.scalar_type() == torch::kInt32,
                "AxEBL should have int32 dtype for split-K. It currently has ",
                c10::toString(AxEBL.scalar_type()), ".");
  }
  TORCH_CHECK(EAR_.has_value() && EBL_.has_value(),
              "Expected both of EAR, EBL.");
  EAR = EAR_.value();
  EBL = EBL_.value();
  check_tensor(EAR, k, r, torch::kInt8);
  check_tensor(EBL, r, k, torch::kInt8);

  check_tensor(A, m, k, torch::kInt8);
  check_tensor(EAL, m, r, torch::kInt8);
  check_tensor(ApEA, m, k, torch::kInt8);
  check_tensor(AxEBL, m, r, torch::kFloat16, torch::kInt32);

  const int64_t tile_size_m = tile_size_m_.value_or(kDefaultNoisingTileSizeMN);
  const int64_t tile_size_k = tile_size_k_.value_or(kDefaultNoisingTileSizeK);

  int k_blocks_per_split;
  if (k_blocks_per_split_.has_value()) {
    k_blocks_per_split = k_blocks_per_split_.value();
  } else if (AxEBL.scalar_type() != torch::kInt32) {
    k_blocks_per_split = 0;
  } else {
    // Use heuristic
    k_blocks_per_split =
        get_num_k_blocks(m, tile_size_m, k, tile_size_k, dprops);
  }

  PearlAPIParams params;
  params.m = m;
  params.n = n;
  params.k = k;
  params.r = r;
  params.host_signal_sync = nullptr;

  params.ptr_A = A.data_ptr();
  params.ptr_B = nullptr;
  params.ptr_EAL = EAL.data_ptr();
  params.ptr_EAR_R_major = EAR.data_ptr();
  params.ptr_EBL_K_major = EBL.data_ptr();
  params.ptr_EAR_K_major = nullptr;
  params.ptr_EBL_R_major = nullptr;
  params.ptr_EBR = nullptr;
  params.ptr_AxEBL = AxEBL.data_ptr();
  params.ptr_EARxBpEB = nullptr;
  params.ptr_ApEA = ApEA.data_ptr();
  params.ptr_BpEB = nullptr;
  params.ptr_A_scales = nullptr;
  params.ptr_B_scales = nullptr;
  params.ptr_C = nullptr;
  params.host_signal_header_pinned = nullptr;

  params.k_blocks_per_split_noising_A = k_blocks_per_split;
  params.k_blocks_per_split_noising_B = 0;
  params.inner_hash_counter = nullptr;
  params.ptr_pow_target = nullptr;
  params.ptr_pow_key = nullptr;

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  bool kernel_found = false;
  NOISING_A_CONFIG_SWITCH(
      tile_size_m, tile_size_k, r, pipeline_stages, AxEBL.scalar_type(),
      kernel_found = true;
      run_pearl_noising_A_<ElementDenoise_AxEBL, R_, bM_, bK_, stages_>(
          params, stream););

  TORCH_CHECK(kernel_found,
              "No noise_A kernel found with given config: ", "R = ", r,
              ", bM_noising = ", tile_size_m, ", AxEBL of type ",
              c10::toString(AxEBL.scalar_type()));
}

void noise_B(at::Tensor& B,                          // n x k
             at::Tensor& EBR,                        // n x r
             at::Tensor& EARxBpEB,                   // n x r
             at::Tensor& BpEB,                       // n x k
             const std::optional<at::Tensor>& EAR_,  // k x r
             const std::optional<at::Tensor>& EBL_,  // r x k
             std::optional<int64_t> tile_size_n_ = std::nullopt,
             std::optional<int64_t> tile_size_k_ = std::nullopt,
             int64_t pipeline_stages = 2,
             std::optional<int64_t> k_blocks_per_split_ = std::nullopt) {

  at::cuda::CUDAGuard device_guard{(char)B.get_device()};
  auto dprops = at::cuda::getCurrentDeviceProperties();
  at::Tensor EAR, EBL;

  if (k_blocks_per_split_.has_value() && k_blocks_per_split_.value() > 0) {
    TORCH_CHECK(
        EARxBpEB.scalar_type() == torch::kInt32,
        "EARxBpEB should have int32 dtype for split-K. It currently has ",
        c10::toString(EARxBpEB.scalar_type()), ".");
  }

  int m = 1;  // dummy value
  int n = int(B.size(0));
  int k = int(B.size(1));
  int r = int(EBR.size(1));
  TORCH_CHECK(
      k % 16 == 0,
      "K must be divisible by 16 (the size of a 128b vectorized copy atom)");
  TORCH_CHECK(EAR_.has_value() && EBL_.has_value(),
              "Expected both of EAR, EBL.");
  EAR = EAR_.value();
  EBL = EBL_.value();
  check_tensor(EBL, k, r, torch::kInt8);
  check_tensor(EAR, r, k, torch::kInt8);

  check_tensor(B, n, k, torch::kInt8);
  check_tensor(EBR, n, r, torch::kInt8);
  check_tensor(BpEB, n, k, torch::kInt8);
  check_tensor(EARxBpEB, n, r, torch::kFloat16, torch::kInt32);

  const int64_t tile_size_n = tile_size_n_.value_or(kDefaultNoisingTileSizeMN);
  const int64_t tile_size_k = tile_size_k_.value_or(kDefaultNoisingTileSizeK);

  int k_blocks_per_split;
  if (k_blocks_per_split_.has_value()) {
    k_blocks_per_split = k_blocks_per_split_.value();
  } else if (EARxBpEB.scalar_type() != torch::kInt32) {
    k_blocks_per_split = 0;
  } else {
    // Use heuristic
    k_blocks_per_split =
        get_num_k_blocks(n, tile_size_n, k, tile_size_k, dprops);
  }

  PearlAPIParams params;

  params.m = m;
  params.n = n;
  params.k = k;
  params.r = r;

  params.ptr_A = nullptr;
  params.ptr_B = B.data_ptr();
  params.ptr_EAL = nullptr;
  params.ptr_EAR_K_major = EAR.data_ptr();
  params.ptr_EBL_R_major = EBL.data_ptr();
  params.ptr_EAR_R_major = nullptr;
  params.ptr_EBL_K_major = nullptr;
  params.ptr_EBR = EBR.data_ptr();
  params.ptr_AxEBL = nullptr;
  params.ptr_EARxBpEB = EARxBpEB.data_ptr();
  params.ptr_ApEA = nullptr;
  params.ptr_BpEB = BpEB.data_ptr();
  params.ptr_A_scales = nullptr;
  params.ptr_B_scales = nullptr;
  params.ptr_C = nullptr;
  params.host_signal_header_pinned = nullptr;
  params.host_signal_sync = nullptr;

  params.k_blocks_per_split_noising_A = 0;
  params.k_blocks_per_split_noising_B = k_blocks_per_split;
  params.inner_hash_counter = nullptr;
  params.ptr_pow_target = nullptr;
  params.ptr_pow_key = nullptr;

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  bool kernel_found = false;
  NOISING_B_CONFIG_SWITCH(
      tile_size_n, tile_size_k, r, pipeline_stages, EARxBpEB.scalar_type(),

      kernel_found = true;
      run_pearl_noising_B_<ElementDenoise_EARxBpEB, R_, bN_, bK_, stages_>(
          params, stream););

  TORCH_CHECK(kernel_found,
              "No noise_B kernel found with given config: ", "R = ", r,
              ", bN_noising = ", tile_size_n, ", EARxBpEB of type ",
              c10::toString(EARxBpEB.scalar_type()));
}

void gemm(at::Tensor& A,         // m x k
          at::Tensor& B,         // n x k
          at::Tensor& A_scales,  // m
          at::Tensor& B_scales,  // n
          at::Tensor& C,         // m x n
          int64_t bM, int64_t bN, int64_t bK, int64_t cM, int64_t cN,
          std::optional<int64_t> pipeline_stages_ = std::nullopt,
          std::optional<int64_t> swizzle = std::nullopt,
          bool swizzle_n_maj = true) {

  at::cuda::CUDAGuard device_guard{(char)A.get_device()};
  auto dprops = at::cuda::getCurrentDeviceProperties();
  CHECK_DEVICE(A);
  CHECK_DEVICE(B);
  CHECK_DEVICE(A_scales);
  CHECK_DEVICE(B_scales);
  CHECK_DEVICE(C);

  CHECK_CONTIGUOUS(A);
  CHECK_CONTIGUOUS(B);
  CHECK_CONTIGUOUS(A_scales);
  CHECK_CONTIGUOUS(B_scales);
  CHECK_CONTIGUOUS(C);

  int m = int(A.size(0));
  int n = int(B.size(0));
  int k = int(A.size(1));

  check_tensor(A, m, k, torch::kInt8);
  check_tensor(B, n, k, torch::kInt8);
  check_tensor(A_scales, m, torch::kFloat32);
  check_tensor(B_scales, n, torch::kFloat32);
  check_tensor(C, m, n, at::kBFloat16);

  TORCH_CHECK(k % bK == 0, "K must be divisible by bK");

  PearlAPIParams params;
  using ElementOut = cutlass::bfloat16_t;

  static constexpr int align_n = 128 / (sizeof(ElementOut) * 8);
  TORCH_CHECK(n % align_n == 0,
              "N must be divisible by " + std::to_string(align_n));

  params.m = m;
  params.n = n;
  params.k = k;

  if (swizzle.has_value()) {
    params.swizzle = static_cast<int>(swizzle.value());
  } else {
    int b_maj = swizzle_n_maj ? bN : bM;
    params.swizzle = get_swizzle_size(params.k, b_maj, dprops);
  }

  int const pipeline_stages =
      pipeline_stages_.has_value()
          ? pipeline_stages_.value()
          : get_pipeline_stages(bM, bN, bK, /*R=*/64, /*skip_denoising=*/true,
                                dprops);

  params.swizzle_n_maj = swizzle_n_maj;
  params.ptr_A = nullptr;
  params.ptr_B = nullptr;
  params.ptr_EAL = nullptr;
  params.ptr_EAL_mma = nullptr;
  params.ptr_EBR = nullptr;
  params.ptr_EBR_mma = nullptr;
  params.ptr_AxEBL = nullptr;
  params.ptr_EARxBpEB = nullptr;
  params.ptr_ApEA = A.data_ptr();
  params.ptr_BpEB = B.data_ptr();
  params.ptr_A_scales = A_scales.data_ptr();
  params.ptr_B_scales = B_scales.data_ptr();
  params.ptr_C = C.data_ptr();
  params.host_signal_header_pinned = nullptr;
  params.host_signal_sync = nullptr;
  params.ptr_EAR_R_major = nullptr;
  params.ptr_EBL_R_major = nullptr;
  params.ptr_EAR_K_major = nullptr;
  params.ptr_EBL_K_major = nullptr;
  params.ptr_AxEBL_int32 = nullptr;
  params.ptr_EARxBpEB_int32 = nullptr;
  params.ptr_AxEBL_mma = nullptr;
  params.ptr_EARxBpEB_mma = nullptr;

  params.k_blocks_per_split_noising_A = 0;
  params.k_blocks_per_split_noising_B = 0;

  params.ptr_pow_target = nullptr;
  params.ptr_pow_key = nullptr;

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  constexpr bool SkipReduction = true;
  constexpr bool SkipDenoising = true;
  constexpr bool EnableDebug = false;

  // Awkwardly, the template signature and thus the existence of a compatible kernel
  // depends on R and the denoise dtypes, but these aren't used for noiseless gemm.
  // We iterate over all possibilities to try to find a compiled kernel.
  bool kernel_found = false;
  for (int const R : {64, 128}) {
    MATMUL_CONFIG_SWITCH(
        bM, bN, bK, R, pipeline_stages, cM, cN, kernel_found = true;
        run_pearl_gemm_<ElementOut, R_, bM_, bN_, bK_, stages_, cM_, cN_,
                        SkipReduction, SkipDenoising, EnableDebug>(params,
                                                                   stream);
        goto done;);
  }

  TORCH_CHECK(kernel_found,
              "No gemm kernel found with given config: ", "bM = ", bM,
              ", bN = ", bN, ", bK = ", bK, ", cM = ", cM, ", cN = ", cN,
              ", stages = ", pipeline_stages);
done:;
}

void noisy_gemm(
    at::Tensor& A,                                  // m x k
    at::Tensor& B,                                  // n x k
    at::Tensor& EAL,                                // m x r
    const std::optional<at::Tensor>& EAL_fp16_,     // m x r
    at::Tensor& EBR,                                // n x r
    const std::optional<at::Tensor>& EBR_fp16_,     // n x r
    const std::optional<at::Tensor>& EAR_R_major_,  // k x r
    const std::optional<at::Tensor>& EBL_R_major_,  // k x r
    const std::optional<at::Tensor>& EAR_K_major_,  // r x k
    const std::optional<at::Tensor>& EBL_K_major_,  // r x k
    at::Tensor& AxEBL_fp16,                         // m x r fp16
    at::Tensor& EARxBpEB_fp16,                      // n x r fp16
    at::Tensor& ApEA,                               // m x k
    at::Tensor& BpEB,                               // n x k
    at::Tensor& A_scales,                           // m
    at::Tensor& B_scales,                           // n
    at::Tensor& C,                                  // m x n
    at::Tensor& host_signal_header_pinned, at::Tensor& host_signal_sync,
    at::Tensor& pow_target,  // blake3::CHAINING_VALUE (uint256, LE word order)
    at::Tensor& pow_key,     // blake3::KEY_SIZE (32 bytes)
    //m x r, int32, provide if want noising to produce in int32
    const std::optional<at::Tensor>& AxEBL_int32_,
    //n x r, int32, provide if want noising to produce in int32
    const std::optional<at::Tensor>& EARxBpEB_int32_, int64_t bM, int64_t bN,
    int64_t bK, int64_t cM, int64_t cN,
    std::optional<int64_t> pipeline_stages_ = std::nullopt,
    std::optional<int64_t> swizzle = std::nullopt, bool swizzle_n_maj = true,
    std::optional<int64_t> tile_size_m_noising_A_ = std::nullopt,
    std::optional<int64_t> tile_size_n_noising_B_ = std::nullopt,
    std::optional<int64_t> tile_size_k_noising_A_ = std::nullopt,
    std::optional<int64_t> tile_size_k_noising_B_ = std::nullopt,
    int64_t pipeline_stages_noising_A = 2,
    int64_t pipeline_stages_noising_B = 2,
    std::optional<int64_t> k_blocks_per_split_noising_A_ = std::nullopt,
    std::optional<int64_t> k_blocks_per_split_noising_B_ = std::nullopt,
    bool run_noising_a = true, bool run_noising_b = true,
    bool skip_reduction = false, bool skip_denoising = false,
    std::optional<at::Tensor> inner_hash_counter = std::nullopt,
    bool enable_debug = false) {
  auto dprops = at::cuda::getCurrentDeviceProperties();

  at::Tensor EAL_fp16, EBR_fp16, EAR_R_major, EBL_R_major, EAR_K_major,
      EBL_K_major, EARxBpEB_int32, AxEBL_int32;
  bool int32_noising_EARxBpEB = EARxBpEB_int32_.has_value();
  bool int32_noising_AxEBL = AxEBL_int32_.has_value();
  auto AxEBL_noising_dtype =
      int32_noising_AxEBL ? torch::kInt32 : torch::kFloat16;
  auto EARxBpEB_noising_dtype =
      int32_noising_EARxBpEB ? torch::kInt32 : torch::kFloat16;
  TORCH_CHECK(
      host_signal_header_pinned.scalar_type() == torch::kInt8,
      "host_signal_header_pinned should have int8 dtype. It currently has ",
      c10::toString(host_signal_header_pinned.scalar_type()), ".");
  TORCH_CHECK(host_signal_sync.scalar_type() == torch::kInt8,
              "host_signal_sync should have int8 dtype. It currently has ",
              c10::toString(host_signal_sync.scalar_type()), ".");
  // Validate pow_target
  CHECK_DEVICE(pow_target);
  CHECK_CONTIGUOUS(pow_target);
  TORCH_CHECK(pow_target.scalar_type() == torch::kUInt32,
              "pow_target must be uint32 dtype. It currently has ",
              c10::toString(pow_target.scalar_type()), ".");
  CHECK_SHAPE(pow_target, blake3::CHAINING_VALUE_SIZE_U32);

  // Validate pow_key
  CHECK_DEVICE(pow_key);
  CHECK_CONTIGUOUS(pow_key);
  TORCH_CHECK(pow_key.scalar_type() == torch::kUInt32,
              "pow_key must be uint32 dtype. It currently has ",
              c10::toString(pow_key.scalar_type()), ".");
  CHECK_SHAPE(pow_key, blake3::CHAINING_VALUE_SIZE_U32);

  TORCH_CHECK(bM <= 256,
              "bN must be less than or equal to 256, host signal header limits "
              "thread rows to uint8");
  TORCH_CHECK(bN <= 256,
              "bM must be less than or equal to 256, host signal header limits "
              "thread cols to uint8");

  int m = int(A.size(0));
  int n = int(B.size(0));
  int k = int(A.size(1));
  int r = int(EAL.size(1));

  TORCH_CHECK(
      EAR_R_major_.has_value() && EBL_R_major_.has_value() &&
          EAR_K_major_.has_value() && EBL_K_major_.has_value(),
      "Expected all of EAR_R_major, EBL_R_major, EAR_K_major, EBL_K_major.");
  EAR_R_major = EAR_R_major_.value();
  EBL_R_major = EBL_R_major_.value();
  EAR_K_major = EAR_K_major_.value();
  EBL_K_major = EBL_K_major_.value();
  check_tensor(EAR_R_major, k, r, torch::kInt8);
  check_tensor(EBL_R_major, k, r, torch::kInt8);
  check_tensor(EAR_K_major, r, k, torch::kInt8);
  check_tensor(EBL_K_major, r, k, torch::kInt8);

  TORCH_CHECK(EAL_fp16_.has_value() && EBR_fp16_.has_value(),
              "Expected both of EAL_fp16, EBR_fp16.");
  EAL_fp16 = EAL_fp16_.value();
  EBR_fp16 = EBR_fp16_.value();
  check_tensor(EAL_fp16, m, r, torch::kFloat16);
  check_tensor(EBR_fp16, n, r, torch::kFloat16);

  if (int32_noising_AxEBL) {
    AxEBL_int32 = AxEBL_int32_.value();
    check_tensor(AxEBL_int32, m, r, torch::kInt32);
  }

  if (int32_noising_EARxBpEB) {
    EARxBpEB_int32 = EARxBpEB_int32_.value();
    check_tensor(EARxBpEB_int32, n, r, torch::kInt32);
  }

  check_tensor(A, m, k, torch::kInt8);
  check_tensor(B, n, k, torch::kInt8);
  check_tensor(EAL, m, r, torch::kInt8);
  check_tensor(EBR, n, r, torch::kInt8);
  check_tensor(ApEA, m, k, torch::kInt8);
  check_tensor(BpEB, n, k, torch::kInt8);
  check_tensor(AxEBL_fp16, m, r, torch::kFloat16);
  check_tensor(EARxBpEB_fp16, n, r, torch::kFloat16);
  check_tensor(A_scales, m, torch::kFloat32);
  check_tensor(B_scales, n, torch::kFloat32);
  check_tensor(C, m, n, at::kBFloat16);

  CHECK_SHAPE(host_signal_header_pinned, get_host_signal_header_size());
  CHECK_SHAPE(host_signal_sync, get_host_signal_sync_size());

  TORCH_CHECK(k % bK == 0, "K must be divisible by bK");

  const int64_t tile_size_m_noising_A =
      tile_size_m_noising_A_.value_or(kDefaultNoisingTileSizeMN);
  const int64_t tile_size_k_noising_A =
      tile_size_k_noising_A_.value_or(kDefaultNoisingTileSizeK);
  const int64_t tile_size_n_noising_B =
      tile_size_n_noising_B_.value_or(kDefaultNoisingTileSizeMN);
  const int64_t tile_size_k_noising_B =
      tile_size_k_noising_B_.value_or(kDefaultNoisingTileSizeK);

  int k_blocks_per_split_noising_A, k_blocks_per_split_noising_B;
  if (k_blocks_per_split_noising_A_.has_value()) {
    k_blocks_per_split_noising_A = k_blocks_per_split_noising_A_.value();
  } else if (AxEBL_noising_dtype != torch::kInt32) {
    k_blocks_per_split_noising_A = 0;
  } else {
    // Use heuristic
    k_blocks_per_split_noising_A = get_num_k_blocks(
        m, tile_size_m_noising_A, k, tile_size_k_noising_A, dprops);
  }

  if (k_blocks_per_split_noising_B_.has_value()) {
    k_blocks_per_split_noising_B = k_blocks_per_split_noising_B_.value();
  } else if (EARxBpEB_noising_dtype != torch::kInt32) {
    k_blocks_per_split_noising_B = 0;
  } else {
    // Use heuristic
    k_blocks_per_split_noising_B = get_num_k_blocks(
        n, tile_size_n_noising_B, k, tile_size_k_noising_B, dprops);
  }

  if (k_blocks_per_split_noising_A > 0) {
    TORCH_CHECK(AxEBL_noising_dtype == torch::kInt32,
                "AxEBL should have int32 dtype for split-K. It currently has ",
                c10::toString(AxEBL_noising_dtype), ".");
  }

  if (k_blocks_per_split_noising_B > 0) {
    TORCH_CHECK(
        EARxBpEB_noising_dtype == torch::kInt32,
        "EARxBpEB should have int32 dtype for split-K. It currently has ",
        c10::toString(EARxBpEB_noising_dtype), ".");
  }

  int const pipeline_stages = pipeline_stages_.value_or(
      get_pipeline_stages(bM, bN, bK, r, skip_denoising, dprops));

  PearlAPIParams params;
  using ElementOut = cutlass::bfloat16_t;

  static constexpr int align_n = 128 / (sizeof(ElementOut) * 8);
  TORCH_CHECK(n % align_n == 0,
              "N must be divisible by " + std::to_string(align_n));

  params.m = m;
  params.n = n;
  params.k = k;
  params.r = r;

  if (swizzle.has_value()) {
    params.swizzle = static_cast<int>(swizzle.value());
  } else {
    int b_maj = swizzle_n_maj ? bN : bM;
    params.swizzle = get_swizzle_size(params.k, b_maj, dprops);
  }

  params.ptr_EARxBpEB = int32_noising_EARxBpEB ? EARxBpEB_int32.data_ptr()
                                               : EARxBpEB_fp16.data_ptr();
  params.ptr_EARxBpEB_int32 =
      int32_noising_EARxBpEB ? EARxBpEB_int32.data_ptr() : nullptr;
  params.ptr_AxEBL =
      int32_noising_AxEBL ? AxEBL_int32.data_ptr() : AxEBL_fp16.data_ptr();
  params.ptr_AxEBL_int32 =
      int32_noising_AxEBL ? AxEBL_int32.data_ptr() : nullptr;
  params.swizzle_n_maj = swizzle_n_maj;
  params.ptr_A = A.data_ptr();
  params.ptr_B = B.data_ptr();
  params.ptr_EAL = EAL.data_ptr();
  params.ptr_EAL_mma = EAL_fp16.data_ptr();
  params.ptr_EAR_R_major = EAR_R_major.data_ptr();
  params.ptr_EBL_R_major = EBL_R_major.data_ptr();
  params.ptr_EAR_K_major = EAR_K_major.data_ptr();
  params.ptr_EBL_K_major = EBL_K_major.data_ptr();
  params.ptr_EBR = EBR.data_ptr();
  params.ptr_EBR_mma = EBR_fp16.data_ptr();
  params.ptr_AxEBL_mma = AxEBL_fp16.data_ptr();
  params.ptr_EARxBpEB_mma = EARxBpEB_fp16.data_ptr();
  params.ptr_ApEA = ApEA.data_ptr();
  params.ptr_BpEB = BpEB.data_ptr();
  params.ptr_A_scales = A_scales.data_ptr();
  params.ptr_B_scales = B_scales.data_ptr();
  params.ptr_C = C.data_ptr();
  params.host_signal_header_pinned = host_signal_header_pinned.data_ptr();
  params.host_signal_sync = host_signal_sync.data_ptr();

  params.k_blocks_per_split_noising_A = k_blocks_per_split_noising_A;
  params.k_blocks_per_split_noising_B = k_blocks_per_split_noising_B;

  // Optional counter for validating inner hash calls
  if (inner_hash_counter.has_value()) {
    params.inner_hash_counter =
        static_cast<uint64_t*>(inner_hash_counter.value().data_ptr());
  } else {
    params.inner_hash_counter = nullptr;
  }

  // PoW target and key
  params.ptr_pow_target = pow_target.data_ptr();
  params.ptr_pow_key = pow_key.data_ptr();

  // Validate that inner_hash_counter is provided when enable_debug is true
  TORCH_CHECK(!enable_debug || params.inner_hash_counter != nullptr,
              "inner_hash_counter must be provided when enable_debug is True");

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  bool kernel_found_matmul = false;
  bool kernel_found_noising_a = false;
  bool kernel_found_noising_b = false;

  bool do_denoise_conversion =
      (int32_noising_EARxBpEB || int32_noising_AxEBL) && (!skip_denoising);

  DEBUG_MODE_SWITCH(
      enable_debug, EnableDebug,
      SKIP_REDUCTION_SWITCH(
          skip_reduction, SkipReduction,
          SKIP_DENOISING_SWITCH(
              skip_denoising, SkipDenoising,

              if (run_noising_a) {
                NOISING_A_CONFIG_SWITCH(
                    tile_size_m_noising_A, tile_size_k_noising_A, r,
                    pipeline_stages_noising_A, AxEBL_noising_dtype,
                    kernel_found_noising_a = true;
                    run_pearl_noising_A_<ElementDenoise_AxEBL, R_, bM_, bK_,
                                         stages_>(params, stream););
              } else { kernel_found_noising_a = true; }

              if (run_noising_b) {
                NOISING_B_CONFIG_SWITCH(
                    tile_size_n_noising_B, tile_size_k_noising_B, r,
                    pipeline_stages_noising_B, EARxBpEB_noising_dtype,
                    kernel_found_noising_b = true;
                    run_pearl_noising_B_<ElementDenoise_EARxBpEB, R_, bN_, bK_,
                                         stages_>(params, stream););
              } else { kernel_found_noising_b = true; }

              if (do_denoise_conversion) {
                if (params.r == 64) {
                  run_denoise_converter<64>(params, stream);
                } else if (params.r == 128) {
                  run_denoise_converter<128>(params, stream);
                } else {
                  TORCH_CHECK(false,
                              "No denoise converter kernel found with "
                              "given config: R = ",
                              r);
                }
              } MATMUL_CONFIG_SWITCH(bM, bN, bK, r, pipeline_stages, cM, cN,
                                     kernel_found_matmul = true;
                                     run_pearl_gemm_<
                                         ElementOut, R_, bM_, bN_, bK_, stages_,
                                         cM_, cN_, SkipReduction, SkipDenoising,
                                         EnableDebug>(params, stream);););););

  TORCH_CHECK(kernel_found_matmul,
              "No noisy_gemm kernel found with given config: ", "bM = ", bM,
              ", bN = ", bN, ", bK = ", bK, ", R = ", r,
              ", stages = ", pipeline_stages, ", cM = ", cM, ", cN = ", cN,
              ", SkipReduction = ", skip_reduction ? "true" : "false",
              ", SkipDenoising = ", skip_denoising ? "true" : "false",
              ", DebugMode = ", enable_debug ? "true" : "false",
              ", AxEBL of type ", c10::toString(AxEBL_noising_dtype),
              ", EARxBpEB of type ", c10::toString(EARxBpEB_noising_dtype));
  TORCH_CHECK(kernel_found_noising_a,
              "No noise_A kernel found with given config: ", "R = ", r,
              ", tile_size_m_noising_A = ", tile_size_m_noising_A,
              ", tile_size_k_noising_A = ", tile_size_k_noising_A,
              ", pipeline_stages_noising_A = ", pipeline_stages_noising_A,
              ", AxEBL of type ", c10::toString(AxEBL_noising_dtype));
  TORCH_CHECK(kernel_found_noising_b,
              "No noise_B kernel found with given config: ", "R = ", r,
              ", tile_size_n_noising_B = ", tile_size_n_noising_B,
              ", tile_size_k_noising_B = ", tile_size_k_noising_B,
              ", pipeline_stages_noising_B = ", pipeline_stages_noising_B,
              ", EARxBpEB of type ", c10::toString(EARxBpEB_noising_dtype));
}

HostSignalHeader get_host_signal_header(at::Tensor& host_signal_header_pinned) {
  HostSignalHeader* header =
      reinterpret_cast<HostSignalHeader*>(host_signal_header_pinned.data_ptr());
  return *header;
}

// Python binding for inner_hash function
at::Tensor inner_hash(at::Tensor input_buffer, int64_t iterations) {
  CHECK_DEVICE(input_buffer);
  CHECK_CONTIGUOUS(input_buffer);

  // Ensure input is uint32 dtype and has correct size
  TORCH_CHECK(input_buffer.dtype() == torch::kUInt32,
              "Input must be uint32 tensor");
  {
    static const cute::array<int64_t, 5> valid_sizes = {64, 96, 128, 192, 256};
    TORCH_CHECK(std::find(valid_sizes.begin(), valid_sizes.end(),
                          input_buffer.size(0)) != valid_sizes.end(),
                "Input must have size equal to " +
                    print_array(valid_sizes, "valid_sizes"));
  }
  TORCH_CHECK(input_buffer.dim() == 1, "Input must be 1D tensor");

  TORCH_CHECK(iterations == 1 || iterations == 100000,
              "Iterations must be 1 or 100000");

  at::cuda::CUDAGuard device_guard{(char)input_buffer.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  // Create output tensor
  auto output = torch::empty({1}, input_buffer.options().dtype(torch::kUInt32));

  // Launch the inner_hash kernel
  launch_inner_hash_kernel(input_buffer.data_ptr<uint32_t>(),
                           input_buffer.size(0), output.data_ptr<uint32_t>(),
                           iterations, stream);

  // Synchronize to ensure kernel completion
  cudaStreamSynchronize(stream);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Pearl GEMM with noising/denoising and PoW extraction";
  m.def("denoise_converter", &denoise_converter,
        "Convert denoising factors from int32 to fp16");
  m.def("noisy_gemm", &noisy_gemm, "Noisy GEMM");
  m.def("gemm", &gemm, "GEMM without noising steps");
  m.def("noise_A", &noise_A, "Noise A (activations)");
  m.def("noise_B", &noise_B, "Noise B (weights)");
  m.def("get_host_signal_header", &get_host_signal_header,
        "Get host signal header");
  m.def("noise_gen", &noise_gen, "Noise generation");
  m.def("inner_hash", &inner_hash, "Inner hash function");
  m.def("tensor_hash", &run_tensor_hash,
        "CUDA hash function with configurable kernel parameters",
        py::arg("data"), py::arg("key"), py::arg("out"), py::arg("roots"),
        py::arg("threads_per_block") = DEFAULT_THREADS_PER_BLOCK,
        py::arg("num_stages") = DEFAULT_NUM_STAGES,
        py::arg("leaves_per_mt_block") = DEFAULT_LEAVES_PER_MT_BLOCK);
  m.def("commitment_hash_from_merkle_roots",
        &run_commitment_hash_from_merkle_roots,
        "Commitment hash from Merkle roots", py::arg("A_merkle_root"),
        py::arg("B_merkle_root"), py::arg("key"), py::arg("A_commitment_hash"),
        py::arg("B_commitment_hash"), py::arg("routing_root") = py::none(),
        py::arg("offsets_hash") = py::none());
  m.def("get_host_signal_header_size", &get_host_signal_header_size,
        "Calculate host signal header buffer size");
  m.def("get_host_signal_sync_size", &get_host_signal_sync_size,
        "Calculate host signal sync buffer size");
  m.def("get_required_scratchpad_bytes", &get_required_scratchpad_bytes,
        "Calculate required scratchpad bytes for given matrix size",
        py::arg("matrix_bytes"),
        py::arg("threads_per_block") = DEFAULT_THREADS_PER_BLOCK);
  m.def("build_routing_data", &build_routing_data,
        "Stable counting sort for MoE routing data");
  m.def("get_build_routing_data_scratchpad_bytes",
        &get_build_routing_data_scratchpad_bytes,
        "Required scratchpad size for build_routing_data", py::arg("numel"));
  m.attr("kEALScaleFactorDenoise") = pearl::kEALScaleFactorDenoise;
  m.attr("kEBRScaleFactorDenoise") = pearl::kEBRScaleFactorDenoise;
  py::enum_<HostSignalStatus>(m, "HostSignalStatus")
      .value("kSignalIdle", HostSignalStatus::kSignalIdle)
      .value("kSignalTriggered", HostSignalStatus::kSignalTriggered);

  py::class_<MMASize>(m, "MMASize")
      .def_readonly("m", &MMASize::m)
      .def_readonly("n", &MMASize::n)
      .def_readonly("k", &MMASize::k);

  py::class_<HostSignalHeader>(m, "HostSignalHeader")
      .def_readonly("status", &HostSignalHeader::status)
      .def_readonly("gridDim", &HostSignalHeader::gridDim)
      .def_readonly("blockDim", &HostSignalHeader::blockDim)
      .def_readonly("blockIdx", &HostSignalHeader::blockIdx)
      .def_readonly("tileCoord", &HostSignalHeader::tileCoord)
      .def_readonly("threadIdx", &HostSignalHeader::threadIdx)
      .def_property_readonly(
          "thread_rows",
          [](const HostSignalHeader& self) {
            return py::memoryview::from_memory(
                self.thread_rows.data(),
                self.num_registers_per_thread * sizeof(uint8_t));
          })
      .def_property_readonly(
          "thread_cols",
          [](const HostSignalHeader& self) {
            return py::memoryview::from_memory(
                self.thread_cols.data(),
                self.num_registers_per_thread * sizeof(uint8_t));
          })
      .def_readonly("mma_size", &HostSignalHeader::mma_size)
      .def_readonly("mma_tile_size", &HostSignalHeader::mma_tile_size)
      .def_readonly("target", &HostSignalHeader::target)
      .def_readonly("num_registers_per_thread",
                    &HostSignalHeader::num_registers_per_thread)
      .def("__repr__", [](const HostSignalHeader& self) {
        std::stringstream ss;
        ss << "HostSignalHeader(status=" << self.status;
        ss << ", " << print_array(self.gridDim, "gridDim");
        ss << ", " << print_array(self.blockDim, "blockDim");
        ss << ", " << print_array(self.blockIdx, "blockIdx");
        ss << ", " << print_array(self.tileCoord, "tileCoord");
        ss << ", " << print_array(self.threadIdx, "threadIdx");
        ss << ", "
           << print_range(
                  self.thread_rows.begin(),
                  self.thread_rows.begin() + self.num_registers_per_thread,
                  "thread_rows");
        ss << ", "
           << print_range(
                  self.thread_cols.begin(),
                  self.thread_cols.begin() + self.num_registers_per_thread,
                  "thread_cols");
        ss << ", mma_size=(" << self.mma_size.m << ", " << self.mma_size.n
           << ", " << self.mma_size.k << ")";
        ss << ", mma_tile_size=(" << self.mma_tile_size.m << ", "
           << self.mma_tile_size.n << ", " << self.mma_tile_size.k << ")";
        ss << ", " << print_array(self.target, "target");
        ss << ", num_registers_per_thread="
           << static_cast<size_t>(self.num_registers_per_thread);
        return ss.str();
      });
}

// For torch ops. Using the same default values as ones used in the pearl_gemm_interface.py
TORCH_LIBRARY(pearl_gemm, m) {
  m.def(
      "noisy_gemm("
      "    Tensor A, "
      "    Tensor B, "
      "    Tensor EAL, "
      "    Tensor? EAL_fp16, "
      "    Tensor EBR, "
      "    Tensor? EBR_fp16, "
      "    Tensor? EAR_R_major, "
      "    Tensor? EBL_R_major, "
      "    Tensor? EAR_K_major, "
      "    Tensor? EBL_K_major, "
      "    Tensor(AxEBL_fp16!) AxEBL_fp16, "
      "    Tensor(EARxBpEB_fp16!) EARxBpEB_fp16, "
      "    Tensor(ApEA!) ApEA, "
      "    Tensor(BpEB!) BpEB, "
      "    Tensor A_scales, "
      "    Tensor B_scales, "
      "    Tensor(C!) C, "
      "    Tensor(host_signal_header_pinned!) host_signal_header_pinned, "
      "    Tensor(host_signal_sync!) host_signal_sync, "
      "    Tensor pow_target, "
      "    Tensor pow_key, "
      "    Tensor(AxEBL_int32!)? AxEBL_int32 = None,"
      "    Tensor(EARxBpEB_int32!)? EARxBpEB_int32 = None,"
      "    int tile_size_m = 128, "
      "    int tile_size_n = 256, "
      "    int tile_size_k = 128, "
      "    int cluster_size_m = 1, "
      "    int cluster_size_n = 1, "
      "    int? pipeline_stages = None, "
      "    int? swizzle = None, "
      "    bool swizzle_n_maj = True, "
      "    int? tile_size_m_noising_A = None, "
      "    int? tile_size_n_noising_B = None, "
      "    int? tile_size_k_noising_A = None, "
      "    int? tile_size_k_noising_B = None, "
      "    int pipeline_stages_noising_A = 2, "
      "    int pipeline_stages_noising_B = 2, "
      "    int? k_blocks_per_split_noising_A = None, "
      "    int? k_blocks_per_split_noising_B = None, "
      "    bool run_noising_A = True, "
      "    bool run_noising_B = False, "
      "    bool skip_reduction = True, "
      "    bool skip_denoising = False, "
      "    Tensor(inner_hash_counter!)? inner_hash_counter = None, "
      "    bool enable_debug = False"
      ") -> ()",
      {at::Tag::pt2_compliant_tag});

  m.def(
      "gemm(Tensor A, Tensor B, Tensor A_scales, Tensor B_scales, Tensor(C!) "
      "C, "
      "int tile_size_m = 128, int tile_size_n = 256, int tile_size_k = 128, "
      "int cluster_size_m = 1, "
      "int cluster_size_n = 1, "
      "int? pipeline_stages = None, "
      "int? swizzle = None, "
      "bool swizzle_n_maj = True) -> "
      "()",
      {at::Tag::pt2_compliant_tag});

  m.def(
      "noise_A("
      "    Tensor A, "
      "    Tensor EAL, "
      "    Tensor(AxEBL!) AxEBL, "
      "    Tensor(ApEA!) ApEA, "
      "    Tensor? EAR, "
      "    Tensor? EBL, "
      "    int? tile_size_m = None, "
      "    int? tile_size_k = None, "
      "    int pipeline_stages = 2, "
      "    int? k_blocks_per_split = None"
      ") -> ()",
      {at::Tag::pt2_compliant_tag});

  m.def(
      "noise_B("
      "    Tensor B, "
      "    Tensor EBR, "
      "    Tensor(EARxBpEB!) EARxBpEB, "
      "    Tensor(BpEB!) BpEB, "
      "    Tensor? EAR, "
      "    Tensor? EBL, "
      "    int? tile_size_n = None, "
      "    int? tile_size_k = None, "
      "    int pipeline_stages = 2, "
      "    int? k_blocks_per_split = None"
      ") -> ()",
      {at::Tag::pt2_compliant_tag});

  m.def(
      "denoise_converter("
      "     Tensor? EARxBpEB_in,"
      "     Tensor? AxEBL_in,"
      "     Tensor(EARxBpEB_out!)? EARxBpEB_out,"
      "     Tensor(AxEBL_out!)? AxEBL_out"
      ") -> ()",
      {at::Tag::pt2_compliant_tag});

  m.def(
      "noise_gen("
      "    int R, "
      "    int num_threads = 64,"
      "    Tensor(EAL!)? EAL = None,"
      "    Tensor(EAL_fp16!)? EAL_fp16 = None,"
      "    Tensor(EAR_R_major!)? EAR_R_major = None,"
      "    Tensor(EAR_K_major!)? EAR_K_major = None,"
      "    Tensor(EBL_R_major!)? EBL_R_major = None,"
      "    Tensor(EBL_K_major!)? EBL_K_major = None,"
      "    Tensor(EBR!)? EBR = None,"
      "    Tensor(EBR_fp16!)? EBR_fp16 = None,"
      "    Tensor? key_A = None,"
      "    Tensor? key_B = None,"
      "    Tensor(aux_buffer!)? aux_buffer = None"
      ") -> ()",
      {at::Tag::pt2_compliant_tag});

  m.def("inner_hash(Tensor input_buffer, int iterations = 1) -> Tensor");
  m.def(
      "tensor_hash(Tensor data, Tensor key, Tensor(out!) out, Tensor(roots!) "
      "roots, "
      "int num_threads = 128, int num_stages = 2, "
      "int leaves_per_mt_block = 512) -> ()");
  m.def(
      "commitment_hash_from_merkle_roots("
      "    Tensor A_merkle_root, "
      "    Tensor B_merkle_root, "
      "    Tensor key, "
      "    Tensor(A_commitment_hash!) A_commitment_hash, "
      "    Tensor(B_commitment_hash!) B_commitment_hash, "
      "    Tensor? routing_root=None, "
      "    Tensor? offsets_hash=None"
      ") -> ()");
  m.def(
      "build_routing_data("
      "    Tensor topk_ids, "
      "    Tensor(routing_data!) routing_data, "
      "    Tensor(slot_indices!) slot_indices, "
      "    Tensor(routing_offsets!) routing_offsets, "
      "    Tensor(scratchpad!) scratchpad, "
      "    int num_experts"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(pearl_gemm, CUDA, m) {
  m.impl("noisy_gemm", &noisy_gemm);
  m.impl("gemm", &gemm);
  m.impl("noise_A", &noise_A);
  m.impl("noise_B", &noise_B);
  m.impl("denoise_converter", &denoise_converter);
  m.impl("noise_gen", &noise_gen);
  m.impl("inner_hash", &inner_hash);
  m.impl("tensor_hash", &run_tensor_hash);
  m.impl("commitment_hash_from_merkle_roots",
         &run_commitment_hash_from_merkle_roots);
  m.impl("build_routing_data", &build_routing_data);
}
