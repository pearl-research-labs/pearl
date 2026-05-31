#!/usr/bin/env bash
# Build sm_89 bN=256 refactor: noiseless correctness test + bN=128 vs bN=256 A/B bench.
# Run from WSL Ubuntu-22.04 (CUDA 12.8 at /usr/local/cuda-12.8).
set -euo pipefail
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /mnt/c/Source/pearl/miner/pearl-gemm/csrc/gemm

OUT=/mnt/c/Source/pearl/miner/pearl-gemm/build-sm89/bN256
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
  --resource-usage
  -DNDEBUG
)

echo "==> 1/2 building test_sm89_standalone (noiseless bN=128 AND bN=256 correctness)..."
nvcc "${NVCC_FLAGS[@]}" \
  pearl_gemm_sm89_inst.cu pearl_gemm_sm89_inst_128x256.cu \
  test_sm89_standalone.cu \
  -o "$OUT/test_sm89_standalone" 2>&1 | tee "$OUT/build_test.log"

echo "==> 2/2 building bench_bN256 (A/B bN=128 vs bN=256, noiseless + denoise)..."
nvcc "${NVCC_FLAGS[@]}" \
  pearl_gemm_sm89_inst.cu          pearl_gemm_sm89_inst_128x256.cu \
  pearl_gemm_sm89_denoise_inst.cu  pearl_gemm_sm89_denoise_inst_128x256.cu \
  pearl_noisingA_sm89_inst.cu      pearl_noisingB_sm89_inst.cu \
  _bench_bN256.cu \
  -o "$OUT/bench_bN256" 2>&1 | tee "$OUT/build_bench.log"

ls -lh "$OUT"
