#!/usr/bin/env bash
# Compile the specialized AVX-512 16-way batched BLAKE3 solver for
# `pearl.challenge`. This is self-contained — it implements the single-block
# BLAKE3 compression directly with AVX-512 intrinsics, specialized for our
# 40-byte (seed||nonce_le8) input. No external BLAKE3 library needed.
#
# Run inside the pearl-build container with /host_home mounted.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -mavx512f provides _mm512_*; -mavx512vl is needed for some 256-bit ops.
# -mavx512bw is harmless (the kernel only uses _f instructions but the host
# has all of avx512{f,dq,bw,vl,vbmi,vbmi2}).
# -O3 lets gcc inline the inner kernel into the OpenMP body.
gcc -O3 -march=native -mavx512f -mavx512vl -fopenmp -Wall \
  _pearl_challenge_solver_simd.c \
  -o pearl_challenge_solver_simd

echo "OK: $(ls -lh "$SCRIPT_DIR/pearl_challenge_solver_simd" | awk '{print $5, $9}')"
