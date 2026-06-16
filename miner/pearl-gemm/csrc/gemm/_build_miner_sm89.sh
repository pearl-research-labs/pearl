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

# Linked translation units:
#   pearl_miner_sm89.cu                          driver + host derivation + verify
#   pearl_gemm_sm89_pow_inst_128x256x128.cu      C-store PoW kernel (verify-mode)
#   pearl_gemm_sm89_pow_inst_128x256x128_nostore.cu  no-C-store PoW (mine-mode;
#                                                fixes the 131072^2 32 GB OOM)
#   pearl_miner_noisegen_sm89.cu                 torch-free R=256 noise-gen +
#                                                device fill_AB (mine-mode)
#   pearl_noisingA_sm89_inst.cu / ...B_inst.cu   validated R=128 noising kernels,
#                                                run as two CHAINED R-halves to
#                                                build the R=256 noised operands
#                                                on device (bit-exact; mod-256
#                                                wrap is associative).
# verify-mode still materializes the noised operands host-side (OpenMP) and
# cross-checks the GPU transcript; mine-mode is fully on-device (no host
# materialization, no (M,N) C buffer). None of these TUs pull c10/torch.
# -fopenmp: parallel host Merkle reduce (blake3_root_from_chunk_cvs) + verify-mode materialize.
# NO -mavx2 / -march: rig04 & rig05 are Intel Pentium G4560 (Kaby Lake) which have AVX/AVX2
#   FUSED OFF (Intel segments it out of Pentium/Celeron). -mavx2 emits illegal instructions ->
#   SIGILL core dump on those rigs. Keep the portable x86-64 (SSE2) baseline so ONE binary runs
#   on the whole fleet (Ryzen, EPYC, AND Pentium). The OpenMP parallel fold is the dominant win.
$NVCC "${FLAGS[@]}" \
  -Xcompiler -fopenmp \
  pearl_miner_sm89.cu \
  pearl_gemm_sm89_pow_inst_128x256x128.cu \
  pearl_gemm_sm89_pow_inst_128x256x128_nostore.cu \
  pearl_gemm_sm89_pow_inst_128x256x128_nodenoise_nostore.cu \
  pearl_gemm_sm89_pow_inst_128x256x64_nodenoise_nostore.cu \
  pearl_blake3_root_sm89.cu \
  pearl_miner_noisegen_sm89.cu \
  pearl_noising_fused_sm89.cu \
  pearl_noisingA_sm89_inst.cu \
  pearl_noisingB_sm89_inst.cu \
  -o "$OUT/pearl_miner_sm89_sm${ARCH}"
echo "built $OUT/pearl_miner_sm89_sm${ARCH}"
