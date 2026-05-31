#!/usr/bin/env bash
# Build the RE2 gemm-only bench for both sm_89 (res-usage / 4070 Ti SUPER) and
# sm_120 (correctness verification on the local 5090). Iterative tuning driver.
set -euo pipefail
NVCC=/usr/local/cuda-12.8/bin/nvcc
cd /mnt/c/Source/pearl/miner/pearl-gemm/csrc/gemm
OUT=/mnt/c/Source/pearl/miner/pearl-gemm/build-sm89/re2
mkdir -p "$OUT"

ARCH="${1:-89}"        # 89 or 120
SUFFIX="${2:-tuned}"

if [ "$ARCH" = "120" ]; then
  GENCODE="-gencode arch=compute_120,code=sm_120"
else
  GENCODE="-gencode arch=compute_89,code=sm_89"
fi

FLAGS=(
  $GENCODE
  -std=c++20 -O3
  -I . -I ..
  -I ../../third_party/cutlass/include
  -I ../../third_party/cutlass/tools/util/include
  -I ../../third_party/cutlass/examples/common
  --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG
)

$NVCC "${FLAGS[@]}" \
  bench_sm89_gemmonly_re2.cu \
  pearl_gemm_sm89_inst_128x256x128.cu \
  pearl_gemm_sm89_inst_128x256.cu \
  -o "$OUT/bench_sm89_gemmonly_re2_sm${ARCH}_${SUFFIX}"
echo "built $OUT/bench_sm89_gemmonly_re2_sm${ARCH}_${SUFFIX}"

# ---- Full-PoUW bench (noisingA + noisingB + GEMM+denoise+PoW) -------------
$NVCC "${FLAGS[@]}" \
  bench_sm89_pouw_re2.cu \
  pearl_gemm_sm89_pow_inst_128x256x128.cu \
  pearl_noisingA_sm89_inst.cu \
  pearl_noisingB_sm89_inst.cu \
  -o "$OUT/bench_sm89_pouw_re2_sm${ARCH}_${SUFFIX}"
echo "built $OUT/bench_sm89_pouw_re2_sm${ARCH}_${SUFFIX}"
