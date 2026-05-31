#!/usr/bin/env bash
# Build _bench_group16_plus_persistent inside pearl-ab container on CPU02.
# (vastai/pytorch:latest with CUDA 12.1 in /usr/local/cuda)
set -euo pipefail
cd /host_home/pearl-deploy/pearl-gemm/csrc/gemm

NVCC=/usr/local/cuda/bin/nvcc

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

OUT=/tmp/bench_group16_plus_persistent
echo "==> building $OUT ..."
"$NVCC" "${NVCC_FLAGS[@]}" \
  pearl_noisingA_sm89_inst.cu \
  pearl_noisingB_sm89_inst.cu \
  pearl_gemm_sm89_denoise_inst.cu \
  _bench_group16_plus_persistent.cu \
  -o "$OUT" 2>&1 | tail -50
ls -lh "$OUT"
