#!/usr/bin/env bash
# CPU02-native build for r128 regresident standalone test.
# Runs INSIDE the pearl-ab Docker container (nvcc 12.1).
# Source mount: /host_home/pearl-deploy/pearl-gemm
set -euo pipefail

PEARL=/host_home/pearl-deploy/pearl-gemm
cd "$PEARL/csrc/gemm"

OUT="$PEARL/build-sm89/r128_regresident"
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
  -DPEARL_GEMM_BUILD_SM89
  --ptxas-options=-v
)

echo "==> Building test_sm89_r128_regresident_standalone (sm_89, nvcc $(nvcc --version | grep release | awk '{print $5}'))"
nvcc "${NVCC_FLAGS[@]}" \
  pearl_gemm_sm89_r128_regresident_inst.cu \
  pearl_gemm_sm89_r128_bM64_inst.cu \
  test_sm89_r128_regresident_standalone.cu \
  -o "$OUT/test_r128_regresident" 2>&1 | tail -80
echo "==> Binary: $OUT/test_r128_regresident"
ls -lh "$OUT/test_r128_regresident"
