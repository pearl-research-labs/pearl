#!/usr/bin/env bash
# Build standalone tests + benches for the Tier 1a (PersistentSwizzledTileScheduler) change.
# Run from WSL Ubuntu-22.04 where CUDA 12.8 is installed.
set -euo pipefail
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /mnt/c/Source/pearl/miner/pearl-gemm/csrc/gemm

OUT=/mnt/c/Source/pearl/miner/pearl-gemm/build-sm89/tier1a
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

echo "==> 1/3 building test_sm89_standalone (noiseless correctness)..."
nvcc "${NVCC_FLAGS[@]}" \
  pearl_gemm_sm89_inst.cu pearl_gemm_sm89_inst_128x256.cu test_sm89_standalone.cu \
  -o "$OUT/test_sm89_standalone"

echo "==> 2/3 building bench_sm89_standalone (noiseless TOPS)..."
nvcc "${NVCC_FLAGS[@]}" \
  pearl_gemm_sm89_inst.cu bench_sm89_standalone.cu \
  -o "$OUT/bench_sm89_standalone"

echo "==> 3/4 building bench_sm89_noisy_gemm_e2e (full pipeline TOPS)..."
nvcc "${NVCC_FLAGS[@]}" \
  pearl_noisingA_sm89_inst.cu pearl_noisingB_sm89_inst.cu \
  pearl_gemm_sm89_denoise_inst.cu bench_sm89_noisy_gemm_e2e.cu \
  -o "$OUT/bench_sm89_noisy_gemm_e2e"

echo "==> 4/5 building _bench_ab (head-to-head NEW vs BASELINE)..."
nvcc "${NVCC_FLAGS[@]}" \
  pearl_gemm_sm89_inst.cu          pearl_gemm_sm89_inst_baseline.cu \
  pearl_gemm_sm89_denoise_inst.cu  pearl_gemm_sm89_denoise_inst_baseline.cu \
  pearl_noisingA_sm89_inst.cu      pearl_noisingB_sm89_inst.cu \
  _bench_ab.cu \
  -o "$OUT/bench_ab"

echo "==> 5/6 building _bench_pow (PoW SkipRed=true vs false overhead)..."
nvcc "${NVCC_FLAGS[@]}" \
  pearl_gemm_sm89_denoise_inst.cu  pearl_gemm_sm89_pow_inst.cu \
  pearl_noisingA_sm89_inst.cu      pearl_noisingB_sm89_inst.cu \
  _bench_pow.cu \
  -o "$OUT/bench_pow"

echo "==> 6/6 building _bench_group16 (single-shape env-controlled, drives group-of-16 A/B)..."
nvcc "${NVCC_FLAGS[@]}" \
  pearl_gemm_sm89_denoise_inst.cu \
  pearl_noisingA_sm89_inst.cu      pearl_noisingB_sm89_inst.cu \
  _bench_group16.cu \
  -o "$OUT/bench_group16"

ls -lh "$OUT"
