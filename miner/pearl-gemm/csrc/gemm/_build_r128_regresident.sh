#!/usr/bin/env bash
# Build the R=128 bM=128 bN=128 register-resident denoise standalone
# correctness test. Runs ONLY on sm_89 hardware (4070 Ti SUPER).
set -euo pipefail
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /mnt/c/Source/pearl/miner/pearl-gemm/csrc/gemm

OUT=/mnt/c/Source/pearl/miner/pearl-gemm/build-sm89/r128_regresident
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

echo "==> Building test_sm89_r128_regresident_standalone"
nvcc "${NVCC_FLAGS[@]}" \
  pearl_gemm_sm89_r128_regresident_inst.cu \
  pearl_gemm_sm89_r128_bM64_inst.cu \
  test_sm89_r128_regresident_standalone.cu \
  -o "$OUT/test_r128_regresident" 2>&1 | tail -40
echo "==> Binary: $OUT/test_r128_regresident"
ls -lh "$OUT/test_r128_regresident"
echo
echo "Run on rig04 (or any sm_89 host):"
echo "    $OUT/test_r128_regresident"
