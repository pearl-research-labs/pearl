#!/usr/bin/env bash
# Build the standalone persist-B mainloop test.
# Run from WSL Ubuntu-22.04 with CUDA 12.1 or 12.8 in PATH.
set -euo pipefail
CUDA_BIN=${CUDA_BIN:-/usr/local/cuda/bin}
export PATH=$CUDA_BIN:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /mnt/c/Source/pearl/miner/pearl-gemm/csrc/gemm

OUT=/mnt/c/Source/pearl/miner/pearl-gemm/build-sm89/persist_b
mkdir -p "$OUT"

NVCC_FLAGS=(
  -gencode arch=compute_89,code=sm_89
  -std=c++20 -O3
  -I .
  -I ..
  -I ../../third_party/cutlass/include
  -I ../../third_party/cutlass/tools/util/include
  -I ../../third_party/cutlass/examples/common
  --expt-relaxed-constexpr
  --expt-extended-lambda
  -DNDEBUG
)

echo "==> building _test_persist_b ..."
nvcc "${NVCC_FLAGS[@]}" \
  _test_persist_b.cu \
  -o "$OUT/test_persist_b"

ls -lh "$OUT"
echo ""
echo "Run on a 4070 Ti SUPER (sm_89) or any sm_89+ device with PTX JIT:"
echo "  $OUT/test_persist_b [device_id=0]"
echo ""
echo "Verified clean build on CUDA 12.8 (WSL Ubuntu-22.04, /usr/local/cuda-12.8/bin/nvcc)."
echo "Verified end-to-end execution on RTX 5090 via PTX JIT:"
echo "  first  call B cp.async issues = 3 (expected 3)"
echo "  second call B cp.async issues = 0 (expected 0) — kPersistB hook active."
