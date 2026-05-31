#!/usr/bin/env bash
# Build _bench_streamk in the pearl-ab container path. Mirrors
# _build_128x256.sh's nvcc invocation.
set -euo pipefail

OUT="${1:-/tmp/streamk_bench}"
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
  -DNDEBUG
  -DPEARL_GEMM_BUILD_SM89
)

echo "==> building _bench_streamk..."
"$NVCC" "${NVCC_FLAGS[@]}" \
  pearl_gemm_sm89_inst.cu \
  _bench_streamk.cu \
  -o "$OUT/bench_streamk" 2>&1 | tee "$OUT/build.log"

ls -lh "$OUT"
echo "DONE: $OUT/bench_streamk"
