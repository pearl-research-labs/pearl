#!/usr/bin/env bash
# Build sm_89 128x256 standalone A/B bench against production 128x128 R=64.
# Targets the pearl-ab docker container (vastai/pytorch:latest, /usr/local/cuda).
#
# Args:
#   $1 = output dir (default: /tmp/bN256_bench)
set -euo pipefail

OUT="${1:-/tmp/bN256_bench}"
mkdir -p "$OUT"
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
  --resource-usage
  -DNDEBUG
)

echo "==> building bench_128x256_nopersist (Variant A: bN=256, no persist-B, 128x128 baseline)..."
"$NVCC" "${NVCC_FLAGS[@]}" \
  pearl_gemm_sm89_inst.cu \
  pearl_gemm_sm89_inst_128x256.cu \
  _bench_128x256_nopersist.cu \
  -o "$OUT/bench_128x256_nopersist" 2>&1 | tee "$OUT/build_nopersist.log"

ls -lh "$OUT"
echo "DONE: $OUT/bench_128x256_nopersist"
