# syntax=docker/dockerfile:1

FROM rust:1.96.1-alpine AS rust-toolchain

# ===== Go + Rust + Task toolchain: rarely changes, fully cached =====
FROM golang:1.26.5-alpine AS toolchain

RUN apk add --no-cache \
    curl \
    build-base \
    musl-dev \
    openssl-dev \
    pkgconfig

COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
ENV PATH="/usr/local/cargo/bin:${PATH}" \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo

RUN sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -b /usr/local/bin v3.46.4

WORKDIR /app

# ===== Rust & C libraries (cached until Rust/C source changes) =====
FROM toolchain AS libs

COPY Taskfile.yml Taskfile.yml
COPY plonky2/ plonky2/
COPY pearl-blake3/ pearl-blake3/
COPY zk-pow/ zk-pow/
COPY xmss/ xmss/

RUN task build:zk-gobind build:libxmss

# ===== Go binaries =====
FROM libs AS builder

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN task build:pearl

# ===== Runtime =====
FROM alpine:3.24 AS runtime

RUN apk add --no-cache \
    ca-certificates \
    libgcc \
    libstdc++

COPY --from=builder /app/bin/pearld /usr/local/bin/pearld
COPY --from=builder /app/bin/prlctl /usr/local/bin/prlctl

VOLUME ["/root/.pearld"]

# 44108  Mainnet Pearl peer-to-peer port
# 44107  Mainnet RPC port
EXPOSE 44108 44107

ENTRYPOINT ["pearld"]
