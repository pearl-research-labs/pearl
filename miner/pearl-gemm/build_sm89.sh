#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_HOME=/usr/local/cuda-12.8
# Sanitize PATH — Windows PATH has parens that confuse bash heredocs.
export PATH=/usr/local/cuda-12.8/bin:.venv-sm89/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
exec ./.venv-sm89/bin/python csrc/gemm/test_sm89_noiseless.py "$@"
