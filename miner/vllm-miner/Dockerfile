# Version pins for reproducible builds (override with docker build --build-arg)
# CUDA 13.0 user-space + PyTorch cu130 (see miner/vllm-miner/pyproject.toml).
ARG CUDA_VERSION=13.0.3

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS builder

# libatomic1: gcc's atomics runtime; pearl_gemm's HitSignal binds it for the
# doorbell's native acquire load (see pow/_hit_signal.py).
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates git libatomic1 && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=docker.io/astral/uv:latest /uv /usr/local/bin/uv

# Install Rust toolchain (required for zk-pow and py-pearl-mining)
COPY --from=rust:latest /usr/local/cargo /usr/local/cargo
COPY --from=rust:latest /usr/local/rustup /usr/local/rustup
ENV PATH="/usr/local/cargo/bin:${PATH}" \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo
RUN rustup self update && rustup default stable

ENV UV_PYTHON_INSTALL_DIR=/usr/local/uv-python \
    UV_PYTHON_BIN_DIR=/usr/local/bin \
    UV_PROJECT_ENVIRONMENT=/usr/local/venv \
    UV_LINK_MODE=copy \
    UV_TORCH_BACKEND=cu130 \
    UV_COMPILE_BYTECODE=true

# Install Python via uv
RUN uv python install 3.12 --default

WORKDIR /app

# Copy all pyproject.toml and uv.lock files first for better layer caching
COPY pyproject.toml uv.lock ./
COPY miner/pearl-gemm/pyproject.toml ./miner/pearl-gemm/
COPY py-pearl-mining/pyproject.toml py-pearl-mining/Cargo.toml ./py-pearl-mining/
COPY miner/vllm-miner/pyproject.toml ./miner/vllm-miner/
COPY miner/miner-utils/pyproject.toml ./miner/miner-utils/
COPY miner/miner-base/pyproject.toml ./miner/miner-base/
COPY miner/pearl-gateway/pyproject.toml ./miner/pearl-gateway/

# Sync remote packages ONLY (mainly download torch and vLLM)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    uv sync --package vllm-miner --no-editable --no-dev --no-install-workspace

# Copy kernel source (CuTe DSL JIT-compiles at runtime; no C++ build)
COPY miner/pearl-gemm/ ./miner/pearl-gemm/

# Copy the rest of the source code
COPY miner/miner-utils/ ./miner/miner-utils/
COPY miner/miner-base/ ./miner/miner-base/
COPY miner/pearl-gateway/ ./miner/pearl-gateway/
COPY miner/vllm-miner/src ./miner/vllm-miner/src
COPY pearl-blake3/ ./pearl-blake3/
COPY py-pearl-mining/ ./py-pearl-mining/
COPY zk-pow/ ./zk-pow/
COPY plonky2/ ./plonky2/

# Install vllm-miner and remaining workspace dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    uv sync --package vllm-miner --no-editable --no-dev --inexact

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

# Runtime needs libatomic1 for the HitSignal doorbell (same as the builder).
RUN apt-get update && \
    apt-get install -y --no-install-recommends libatomic1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy uv-managed Python and the virtual environment
COPY --from=builder /usr/local/uv-python /usr/local/uv-python
COPY --from=builder /usr/local/venv /usr/local/venv

# Copy the entrypoint script directly from context (not builder, to avoid rebuild on changes)
COPY miner/vllm-miner/entrypoint.sh /app/entrypoint.sh

# Venv bin/ contains python symlink and installed scripts (pearl-gateway, etc.)
ENV PATH="/usr/local/venv/bin:${PATH}"

# vLLM native libs need libcudart from the NVIDIA CUDA image; PyTorch ships its own libs under torch/lib.
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

ENTRYPOINT ["./entrypoint.sh"]
