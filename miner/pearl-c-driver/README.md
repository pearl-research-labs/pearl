# pearl-c-driver — SKELETON

C/C++ hot-loop driver for the Pearl W19R miner. Replaces Python orchestration
for the per-nonce GEMM dispatch path. SKELETON only — bench mode is functional,
stratum integration is the next phase.

## Status

- [x] Build against prebuilt `pearl_gemm_w19_cuda.so` (no libtorch link required at C call sites)
- [x] Hot-loop driving `pearl_gemm_sm89_w19r_64x64x128_R128_prod` directly
- [x] CUDA Graph capture mode (`--graph`) — single `cudaGraphLaunch` for all 256 GEMMs per attempt
- [x] Bench at production shape (M=N=2048, K=4096, R=128, batch=256)
- [ ] noise_gen_blake3_persistent integration (next; symbol exported as
      `pearl_w19r::launch_blake3_noise_gen_persistent<bM,bN>`)
- [ ] apply_sparse_noise / extract_sparse_indices (need to compile launcher
      from .cuh headers — not exposed via C linkage in .so)
- [ ] stratum v1.5 client (placeholder in `src/stratum_client.c`)
- [ ] Share derivation (commitment_hash + tensor_hash) — port from Python

## Build

In CPU01's `pearl-build-hostnet` docker container:

```bash
cd /host_home/pearl-deploy/pearl-c-driver && make
```

## Run

```bash
export LD_LIBRARY_PATH=/opt/pearl-venv/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpython3.12.so \
  ./pearl_c_driver --shape prod --batch 256 --iters 100 --warmup 5 [--graph]
```

`LD_PRELOAD libpython3.12` is required because `pearl_gemm_w19_cuda.so` has a
DT_NEEDED on `libtorch_python.so`, which resolves Python C-API symbols at
runtime. We never call any of those, but the loader must satisfy them.

## Bench result (CPU01, 2026-05-19)

GEMM-only ceiling at M=N=2048, K=4096, R=128, batch=256:

| metric              | value     |
|---------------------|-----------|
| per-attempt latency | 95.0 ms   |
| nonces/sec          | 2,691     |
| effective TOPS      | 92.45     |
| vs Python (725 n/s) | 3.71x     |

CUDA Graph mode does not change the result (compute-bound, not launch-bound).
This matches the user's note that GEMM-only achieves 92.42 TOPS in the unified
harness.
