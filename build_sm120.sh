#!/bin/bash
set -e
echo "=== PEARL-GEMM sm_120a BUILD (PR#118 sm80-fallback) ==="
date
cd /work
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export PEARL_GEMM_ARCH=sm_120a
export PEARL_GEMM_FORCE_BUILD=1
export MAX_JOBS=8
echo "ARCH=$PEARL_GEMM_ARCH  CUDA_HOME=$CUDA_HOME"
nvcc --version | tail -2
echo "=== Starting build_ext ==="
python3 setup.py build_ext --inplace 2>&1
echo "=== BUILD EXIT: $? ==="
echo "=== Resulting .so ==="
ls -la src/*.so build/lib*/*.so 2>/dev/null
date
