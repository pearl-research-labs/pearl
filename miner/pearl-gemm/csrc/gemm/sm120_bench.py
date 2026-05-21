"""Production-shape bench for pearl-gemm on sm_120 (RTX 50-series).

Mirrors the sm_89 bench/wave-15 methodology: time pg.gemm at production
mining shapes, report TOPS = 2 * M * N * K / time_seconds / 1e12.

Production shape used by alphapool mining: M=N=2048, K=4096, R=128, batch=256.
We bench the noiseless GEMM (skip_reduction + skip_denoising) at the
underlying tile configs available in our build.
"""

import time

import torch
import pearl_gemm_cuda as pg

assert torch.cuda.is_available()
dev = torch.device("cuda:0")
cap = torch.cuda.get_device_capability(0)
gpu_name = torch.cuda.get_device_name(0)
print(f"GPU: {gpu_name}  capability {cap}  pearl_gemm_cuda min_cc={pg._min_compute_capability}")
print(f"torch {torch.__version__}  cuda {torch.version.cuda}")
print()

# Tile configs available in the all-consumer build (matches
# default_compiled_kernels.py narrow grid).
TILE_CONFIGS = [
    # (bM, bN, bK, pipeline_stages, label)
    (128, 128, 64, 3, "R=64 128x128x64 stages=3"),
    (128, 128, 64, 2, "R=64 128x128x64 stages=2"),
]

# Problem sizes -- noiseless GEMM only (we're benching the kernel,
# not the noisy_gemm pipeline).
SIZES = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (2048, 2048, 4096),     # production shape ratio M=N, K=2*M
]

WARMUP = 3
ITERS = 20

print(f"{'shape (MxNxK)':<22} {'tile':<32} {'avg ms':>10} {'TOPS':>10}")
print("-" * 76)

for M, N, K in SIZES:
    A = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev)
    B = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=dev)
    A_scales = torch.ones(M, dtype=torch.float32, device=dev)
    B_scales = torch.ones(N, dtype=torch.float32, device=dev)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)

    best_tops = 0.0
    best_label = ""
    for bM, bN, bK, stages, label in TILE_CONFIGS:
        if M % bM != 0 or N % bN != 0 or K % bK != 0:
            continue
        try:
            for _ in range(WARMUP):
                pg.gemm(A, B, A_scales, B_scales, C, bM, bN, bK, 1, 1, stages, None, True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                pg.gemm(A, B, A_scales, B_scales, C, bM, bN, bK, 1, 1, stages, None, True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            ms = 1000.0 * (t1 - t0) / ITERS
            tops = 2 * M * N * K / ((t1 - t0) / ITERS) / 1e12
            print(f"{M}x{N}x{K:<14} {label:<32} {ms:>10.3f} {tops:>10.2f}")
            if tops > best_tops:
                best_tops = tops
                best_label = label
        except Exception as e:
            print(f"{M}x{N}x{K:<14} {label:<32} FAILED: {e!r}")
    print(f"{'':22} {'best:':<32} {'':>10} {best_tops:>10.2f}  ({best_label})")
    print()

# 5090 peak int8 TOPS (dense, marketing): 838 TOPS.
# alpha-miner on 4070 Ti S (sm_89): 72.7 TOPS effective.
# Our wave-18 best on 4070 Ti S: 65.15 TOPS.
print("Reference points:")
print(f"  RTX 5090 dense int8 peak (marketing): ~838 TOPS")
print(f"  alpha-miner on 4070 Ti S (sm_89):      72.7 TOPS effective")
print(f"  wave-18 best on 4070 Ti S (sm_89):     65.15 TOPS")
