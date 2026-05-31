#!/usr/bin/env bash
# Build the unified C++ bench. Companion to _build_tier1a.sh / _build_bN256.sh.
# Run from WSL Ubuntu-22.04 where CUDA 12.8 is installed.
set -euo pipefail
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /mnt/c/Source/pearl/miner/pearl-gemm/csrc/gemm

OUT=/mnt/c/Source/pearl/miner/pearl-gemm/build-sm89/unified
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

echo "==> building _bench_unified..."
nvcc "${NVCC_FLAGS[@]}" \
  pearl_noisingA_sm89_inst.cu \
  pearl_noisingB_sm89_inst.cu \
  pearl_gemm_sm89_denoise_inst.cu \
  pearl_gemm_sm89_pow_inst.cu \
  _bench_unified.cu \
  -o "$OUT/bench_unified"

ls -lh "$OUT"
