#!/usr/bin/env bash
# Build the head-to-head bench: regresident bM=128 bN=128 vs bM=64 bN=128.
# Runs INSIDE pearl-ab Docker container on CPU02.
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
)

echo "==> Building bench_sm89_r128_regresident_vs_bN128"
nvcc "${NVCC_FLAGS[@]}" \
  pearl_gemm_sm89_r128_regresident_inst.cu \
  pearl_gemm_sm89_r128_bM64_inst.cu \
  bench_sm89_r128_regresident_vs_bN128.cu \
  -o "$OUT/bench_r128_regresident_vs_bN128" 2>&1 | tail -10
echo "==> Binary: $OUT/bench_r128_regresident_vs_bN128"
ls -lh "$OUT/bench_r128_regresident_vs_bN128"
