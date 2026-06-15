// Include these 2 headers instead of torch/extension.h since we don't need all
// of the torch headers.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/python.h>
#include <cstddef>
#include <cstdint>
#include <optional>
// Include only the host function declaration
#include "blake3/blake3_constants.hpp"
#include "tensor_hash_decl.hpp"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Default values for kernel parameters
constexpr size_t DEFAULT_THREADS_PER_BLOCK = 128;    // merkle_tree_roots_kernel
constexpr size_t DEFAULT_NUM_STAGES = 2;             // merkle_tree_roots_kernel
constexpr size_t DEFAULT_LEAVES_PER_MT_BLOCK = 512;  // compute_blake_mt_kernel

// merkle_tree_roots_kernel builds a TMA descriptor over `data`;
// cuTensorMapEncodeTiled requires the global address to be 16-byte aligned.
constexpr uintptr_t TMA_GLOBAL_ALIGNMENT_BYTES = 16;

size_t get_required_scratchpad_bytes(
    size_t matrix_bytes, size_t threads_per_block = DEFAULT_THREADS_PER_BLOCK) {
  size_t bytes_per_block = threads_per_block * blake3::CHUNK_SIZE;
  size_t required_blocks =
      (matrix_bytes + bytes_per_block - 1) / bytes_per_block;
  return required_blocks * blake3::CHAINING_VALUE_SIZE;
}

// Tensor hash with configurable kernel parameters:
//   threads_per_block: Threads per block for merkle_tree_roots_kernel (128, 256, 512)
//   num_stages: Pipeline stages for merkle_tree_roots_kernel (2)
//   leaves_per_mt_block: Threads for compute_blake_mt_kernel (256, 512, 1024)
void run_tensor_hash(
    at::Tensor& data,  // input data tensor
    at::Tensor& key, at::Tensor& out, at::Tensor& roots,
    int64_t threads_per_block = DEFAULT_THREADS_PER_BLOCK,
    int64_t num_stages = DEFAULT_NUM_STAGES,
    int64_t leaves_per_mt_block = DEFAULT_LEAVES_PER_MT_BLOCK) {
  CHECK_DEVICE(data);
  CHECK_DEVICE(key);
  CHECK_DEVICE(out);
  CHECK_DEVICE(roots);
  CHECK_CONTIGUOUS(data);
  CHECK_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(roots);

  TORCH_CHECK(key.dtype() == at::kByte, "key must be uint8");
  TORCH_CHECK(out.dtype() == at::kByte, "out must be uint8");
  TORCH_CHECK(roots.dtype() == at::kByte, "roots must be uint8");
  TORCH_CHECK(data.dim() == 2, "data must be 2D tensor");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(data.data_ptr()) %
                      TMA_GLOBAL_ALIGNMENT_BYTES ==
                  0,
              "data must be ", TMA_GLOBAL_ALIGNMENT_BYTES,
              "-byte aligned for TMA");
  TORCH_CHECK(key.numel() == blake3::KEY_SIZE, "key must have exactly",
              blake3::KEY_SIZE, "bytes");
  TORCH_CHECK(out.numel() == blake3::CHAINING_VALUE_SIZE,
              "out must have exactly", blake3::CHAINING_VALUE_SIZE, "bytes");
  TORCH_CHECK(roots.numel() % blake3::CHAINING_VALUE_SIZE == 0,
              "roots must have a multiple of", blake3::CHAINING_VALUE_SIZE,
              "bytes");

  // Validate threads_per_block (merkle_tree_roots_kernel)
  TORCH_CHECK(threads_per_block == 128 || threads_per_block == 256 ||
                  threads_per_block == 512,
              "threads_per_block must be 128, 256, or 512");

  // Validate num_stages
  TORCH_CHECK(num_stages == 2 || num_stages == 3 || num_stages == 4,
              "num_stages must be 2, 3, or 4");

  // Validate leaves_per_mt_block (compute_blake_mt_kernel)
  TORCH_CHECK(leaves_per_mt_block == 256 || leaves_per_mt_block == 512 ||
                  leaves_per_mt_block == 1024,
              "leaves_per_mt_block must be 256, 512, or 1024");

  constexpr size_t chunk_size = 1024;

  // We split data into chunks of size C (chunk_size)
  size_t num_chunks = (data.numel() + chunk_size - 1) / chunk_size;
  // We split chunks into blocks based on threads_per_block
  size_t num_blocks = (num_chunks + threads_per_block - 1) / threads_per_block;

  TORCH_INTERNAL_ASSERT(
      num_blocks * blake3::CHAINING_VALUE_SIZE ==
          get_required_scratchpad_bytes(data.numel(), threads_per_block),
      "num_blocks=", num_blocks, " get_required_scratchpad_bytes=",
      get_required_scratchpad_bytes(data.numel(), threads_per_block));
  TORCH_CHECK((size_t)roots.numel() >= get_required_scratchpad_bytes(
                                           data.numel(), threads_per_block),
              "roots must have at least ", num_blocks, " * ",
              blake3::CHAINING_VALUE_SIZE, "bytes");
  TORCH_CHECK(data.numel() > 0, "data must be non-empty");

  auto stream = at::cuda::getCurrentCUDAStream();
  auto dprops = at::cuda::getCurrentDeviceProperties();

  tensor_hash(data.data_ptr<uint8_t>(), data.numel(), out.data_ptr<uint8_t>(),
              key.data_ptr<uint8_t>(), num_blocks,
              static_cast<uint32_t>(threads_per_block),
              static_cast<uint32_t>(num_stages),
              static_cast<uint32_t>(leaves_per_mt_block),
              roots.data_ptr<uint8_t>(), *dprops, stream);
}

// Computes both A and B commitment hashes from their merkle roots
// Should be the same as commitment_hash_from_merkle_roots from Commitment_hash.py
//
// routing_root and offsets_hash are optional. When both are provided the MoE
// routing commitment is folded into A's seed (see the kernel); they must be
// supplied together or not at all (the dense case passes neither).
void run_commitment_hash_from_merkle_roots(
    at::Tensor& A_merkle_root, at::Tensor& B_merkle_root, at::Tensor& key,
    at::Tensor& A_commitment_hash, at::Tensor& B_commitment_hash,
    std::optional<at::Tensor> routing_root = std::nullopt,
    std::optional<at::Tensor> offsets_hash = std::nullopt) {
  CHECK_DEVICE(A_merkle_root);
  CHECK_DEVICE(B_merkle_root);
  CHECK_DEVICE(key);
  CHECK_CONTIGUOUS(A_merkle_root);
  CHECK_CONTIGUOUS(B_merkle_root);
  CHECK_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(A_commitment_hash);
  CHECK_CONTIGUOUS(B_commitment_hash);

  TORCH_CHECK(A_merkle_root.dtype() == at::kByte,
              "A_merkle_root must be uint8");
  TORCH_CHECK(B_merkle_root.dtype() == at::kByte,
              "B_merkle_root must be uint8");
  TORCH_CHECK(key.dtype() == at::kByte, "key must be uint8");
  TORCH_CHECK(A_commitment_hash.dtype() == at::kByte,
              "A_commitment_hash must be uint8");
  TORCH_CHECK(B_commitment_hash.dtype() == at::kByte,
              "B_commitment_hash must be uint8");

  TORCH_CHECK(A_merkle_root.numel() == blake3::CHAINING_VALUE_SIZE,
              "A_merkle_root must have exactly", blake3::CHAINING_VALUE_SIZE,
              "bytes");
  TORCH_CHECK(B_merkle_root.numel() == blake3::CHAINING_VALUE_SIZE,
              "B_merkle_root must have exactly", blake3::CHAINING_VALUE_SIZE,
              "bytes");
  TORCH_CHECK(key.numel() == blake3::KEY_SIZE, "key must have exactly",
              blake3::KEY_SIZE, "bytes");
  TORCH_CHECK(A_commitment_hash.numel() == blake3::CHAINING_VALUE_SIZE,
              "A_commitment_hash must have exactly",
              blake3::CHAINING_VALUE_SIZE, "bytes");
  TORCH_CHECK(B_commitment_hash.numel() == blake3::CHAINING_VALUE_SIZE,
              "B_commitment_hash must have exactly",
              blake3::CHAINING_VALUE_SIZE, "bytes");

  TORCH_CHECK(routing_root.has_value() == offsets_hash.has_value(),
              "routing_root and offsets_hash must be provided together");

  const uint8_t* routing_root_ptr = nullptr;
  const uint8_t* offsets_hash_ptr = nullptr;
  if (routing_root.has_value()) {
    auto& routing_root_tensor = routing_root.value();
    auto& offsets_hash_tensor = offsets_hash.value();
    CHECK_DEVICE(routing_root_tensor);
    CHECK_DEVICE(offsets_hash_tensor);
    CHECK_CONTIGUOUS(routing_root_tensor);
    CHECK_CONTIGUOUS(offsets_hash_tensor);
    TORCH_CHECK(routing_root_tensor.dtype() == at::kByte,
                "routing_root must be uint8");
    TORCH_CHECK(offsets_hash_tensor.dtype() == at::kByte,
                "offsets_hash must be uint8");
    TORCH_CHECK(routing_root_tensor.numel() == blake3::CHAINING_VALUE_SIZE,
                "routing_root must have exactly", blake3::CHAINING_VALUE_SIZE,
                "bytes");
    TORCH_CHECK(offsets_hash_tensor.numel() == blake3::CHAINING_VALUE_SIZE,
                "offsets_hash must have exactly", blake3::CHAINING_VALUE_SIZE,
                "bytes");
    routing_root_ptr = routing_root_tensor.data_ptr<uint8_t>();
    offsets_hash_ptr = offsets_hash_tensor.data_ptr<uint8_t>();
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  auto dprops = at::cuda::getCurrentDeviceProperties();

  commitment_hash_from_merkle_roots(
      A_merkle_root.data_ptr<uint8_t>(), B_merkle_root.data_ptr<uint8_t>(),
      key.data_ptr<uint8_t>(), A_commitment_hash.data_ptr<uint8_t>(),
      B_commitment_hash.data_ptr<uint8_t>(), routing_root_ptr, offsets_hash_ptr,
      *dprops, stream);
}

#undef CHECK_DEVICE
#undef CHECK_CONTIGUOUS
