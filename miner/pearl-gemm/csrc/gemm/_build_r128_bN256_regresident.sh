#!/usr/bin/env bash
# CPU02-native build for wave-3 R=128 bM=128 bN=256 register-resident bench.
# Runs INSIDE the pearl-ab Docker container (nvcc 12.1) on CPU02.
# Source mount: /host_home/pearl-deploy/pearl-gemm
#
# Usage on CPU02 host (192.168.68.32):
#   docker exec pearl-ab bash /host_home/pearl-deploy/pearl-gemm/csrc/gemm/_build_r128_bN256_regresident.sh
set -euo pipefail

PEARL=/host_home/pearl-deploy/pearl-gemm
cd "$PEARL/csrc/gemm"

OUT="$PEARL/build-sm89/r128_bN256_regresident"
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
  --resource-usage
)

echo "==> Building bench_sm89_r128_bN256_regresident (sm_89, nvcc $(nvcc --version | grep release | awk '{print $5}'))"
nvcc "${NVCC_FLAGS[@]}" \
  pearl_gemm_sm89_r128_bN256_regresident_inst.cu \
  pearl_gemm_sm89_r128_regresident_inst.cu \
  pearl_gemm_sm89_r128_bM64_inst.cu \
  bench_sm89_r128_bN256_regresident.cu \
  -o "$OUT/bench_r128_bN256_regresident" 2>&1 | tee "$OUT/build.log"
echo
echo "==> Binary: $OUT/bench_r128_bN256_regresident"
ls -lh "$OUT/bench_r128_bN256_regresident" 2>&1 || true

echo
echo "==> ptxas register / spill summary (grep 'bN256\\|Registers\\|spill'):"
grep -E "Used [0-9]+ registers|stack frame|spill stores|spill loads|ptxas info.*regresident_128x256" "$OUT/build.log" || true
