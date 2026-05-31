#!/usr/bin/env bash
# Recreate pearl-ab container's Python venv after restart.
# pearl-gemm .so already built on /host_home/pearl-deploy/pearl-gemm/src/.
set -e
apt-get update -qq
apt-get install -y -qq software-properties-common curl >/dev/null
add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1
apt-get install -y -qq python3.12 python3.12-dev python3.12-venv >/dev/null
python3.12 -m venv /opt/pearl-venv
. /opt/pearl-venv/bin/activate
pip install --quiet --upgrade pip uv
pip install --quiet --index-url https://download.pytorch.org/whl/cu121 torch
# Patch torch's CUDA version check (we have nvcc 12.1, torch built for 13.0)
sed -i 's|raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)|pass  # Patched|' \
  /opt/pearl-venv/lib/python3.12/site-packages/torch/utils/cpp_extension.py
# Rust toolchain for py-pearl-mining (must build BEFORE miner-base which depends on it)
if [ ! -e /root/.cargo/bin/cargo ]; then
  curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs > /tmp/rustup.sh
  sh /tmp/rustup.sh -y --default-toolchain stable --profile minimal >/dev/null
fi
. /root/.cargo/env
pip install --quiet maturin blake3 fastjsonschema bitcoin-utils prometheus-client aiohttp numpy
# Build py-pearl-mining first so miner-base can resolve it
cd /host_home/pearl-deploy/py-pearl-mining
maturin develop --release 2>&1 | tail -3
# Install pearl Python packages (path-resolved together)
cd /host_home/pearl-deploy
uv pip install --quiet -e ./miner-utils -e ./pearl-gemm-build-utils
uv pip install --quiet -e ./miner-base -e ./pearl-gateway -e ./pearl-stratum
# Install pearl-gemm (using the already-built .so cached on disk)
cd /host_home/pearl-deploy/pearl-gemm
PEARL_GEMM_TARGET_ARCH=89 TORCH_CUDA_ARCH_LIST=8.9 \
  uv pip install --quiet --no-build-isolation -e . 2>&1 | tail -3
# Patch torch.version.cuda for pearl-gateway upgrade (it pulls cu130)
sed -i 's|raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)|pass  # Patched|' \
  /opt/pearl-venv/lib/python3.12/site-packages/torch/utils/cpp_extension.py 2>/dev/null || true
echo === DONE: import check ===
. /opt/pearl-venv/bin/activate
python -c "import pearl_gemm_cuda, pearl_gemm, miner_base, pearl_gateway, pearl_stratum, pearl_mining; print('all OK')"
