#!/usr/bin/env bash
# Build the standalone sm_89 Pearl miner `pearl_miner_sm89`.
# ARCH=89 (4070 Ti SUPER, production) or 120 (local 5090, compile-only smoke).
set -euo pipefail
NVCC=/usr/local/cuda-12.8/bin/nvcc
cd /mnt/c/Source/pearl/miner/pearl-gemm/csrc/gemm
OUT=/mnt/c/Source/pearl/miner/pearl-gemm/build-sm89/re2
mkdir -p "$OUT"

ARCH="${1:-89}"
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

# Noise + noised operands (ApEA/BpEB) are computed host-side (pearl_miner_host.hpp,
# validated bit-exact vs miner-base). The on-device noise_generation.cu /
# denoise_converter.cu / noisingA/B are NOT linked: the first two pull c10/torch
# via error_check.hpp, and the R<=128 noising kernels cannot do a correct R=256
# noise in a single pass. Only the load-bearing PoW mainloop kernel is linked.
# OpenMP parallelizes the host noising.
$NVCC "${FLAGS[@]}" \
  -Xcompiler -fopenmp \
  pearl_miner_sm89.cu \
  pearl_gemm_sm89_pow_inst_128x256x128.cu \
  -o "$OUT/pearl_miner_sm89_sm${ARCH}"
echo "built $OUT/pearl_miner_sm89_sm${ARCH}"
