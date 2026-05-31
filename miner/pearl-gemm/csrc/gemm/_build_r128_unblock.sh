#!/usr/bin/env bash
# Compile-only validation pass for R=128 sm_89 instantiations.
# Run from WSL Ubuntu-22.04 where CUDA 12.8 is installed.
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

echo "==> [1/4] Compile-only: existing R=128 noiseless (bM=bN=64)"
nvcc "${NVCC_FLAGS[@]}" -c pearl_gemm_sm89_inst.cu \
  -o "$OUT/sm89_inst_existing.o" 2>&1 | tail -30
echo "==> OK"

echo "==> [2/4] Compile-only: existing R=128 denoise (bM=bN=64)"
nvcc "${NVCC_FLAGS[@]}" -c pearl_gemm_sm89_denoise_inst.cu \
  -o "$OUT/sm89_denoise_inst_existing.o" 2>&1 | tail -30
echo "==> OK"

echo "==> [3/4] Compile-only: existing R=128 pow (bM=bN=64)"
nvcc "${NVCC_FLAGS[@]}" -c pearl_gemm_sm89_pow_inst.cu \
  -o "$OUT/sm89_pow_inst_existing.o" 2>&1 | tail -30
echo "==> OK"

echo "==> [4/6] Compile-only: NEW R=128 bM=64 bN=128 (this file's deliverable)"
nvcc "${NVCC_FLAGS[@]}" -c pearl_gemm_sm89_r128_bM64_inst.cu \
  -o "$OUT/sm89_r128_bM64_inst.o" 2>&1 | tail -40
echo "==> OK"

echo "==> [5/6] Skipped launch-template autogen test (needs torch/c10 headers)."
echo "    The new file pearl_gemm_sm89_r128_bM64_inst.cu exercises the same"
echo "    KTraits + Mainloop + Epilogue templates that the launch template uses."

echo "==> [6/6] All R=128 instantiations compile cleanly."
ls -lh "$OUT"
