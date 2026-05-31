#!/usr/bin/env bash
# Build the R=128 bM=64 bN=128 standalone bench.
# Resulting binary runs ONLY on sm_89 hardware (4070 Ti SUPER).
set -euo pipefail
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /mnt/c/Source/pearl/miner/pearl-gemm/csrc/gemm

OUT=/mnt/c/Source/pearl/miner/pearl-gemm/build-sm89/r128_unblock
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

echo "==> Building bench_sm89_r128_bM64"
nvcc "${NVCC_FLAGS[@]}" \
  pearl_gemm_sm89_denoise_inst.cu \
  pearl_gemm_sm89_r128_bM64_inst.cu \
  bench_sm89_r128_bM64.cu \
  -o "$OUT/bench_sm89_r128_bM64" 2>&1 | tail -30
echo "==> Binary: $OUT/bench_sm89_r128_bM64"
ls -lh "$OUT/bench_sm89_r128_bM64"
