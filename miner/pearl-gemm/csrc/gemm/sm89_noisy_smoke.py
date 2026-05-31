"""Direct pg.noisy_gemm() smoke test on sm_89 with int32 noising path.

The sm_89 inst files have:
  - noisingA: 64×64 R=64 int32 stages=2 (NoReduction=true)
  - noisingB: 64×64 R=64 int32 stages=2 (NoReduction=true)
  - gemm: 128×128×64 R=64 (Noiseless/Denoise stages=3, Pow stages=2)

So we pass int32 *_int32_ tensors → api.cpp dispatch picks int32 noising →
denoise_converter runs (int32 → fp16) → gemm with fp16 denoise inputs.

Goal: verify the full noisy_gemm pipeline runs on sm_89 without error.
"""

import torch
import pearl_gemm_cuda as pg

dev = torch.device("cuda:0")
print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("min_cc:", pg._min_compute_capability)

# Shape: K=4096 to match pool's set_mining_params; rank R=64 (sm_89 inst).
# Chunk M=N=2048 (Tier 1a memo: this is where sm_89 wins).
M, N, K, R = 2048, 2048, 4096, 64

A = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev)
B = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=dev)
A_scales = torch.rand(M, dtype=torch.float32, device=dev) * 0.02 + 0.005
B_scales = torch.rand(N, dtype=torch.float32, device=dev) * 0.02 + 0.005
C = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)

# Noise factors (allocated empty — actual values come from set_mining_params
# in production; for smoke test just zero them).
EAL = torch.zeros(M, R, dtype=torch.int8, device=dev)
EBR = torch.zeros(N, R, dtype=torch.int8, device=dev)
EAL_fp16 = torch.zeros(M, R, dtype=torch.float16, device=dev)
EBR_fp16 = torch.zeros(N, R, dtype=torch.float16, device=dev)
EAR_R_major = torch.zeros(K, R, dtype=torch.int8, device=dev)
EBL_R_major = torch.zeros(K, R, dtype=torch.int8, device=dev)
EAR_K_major = torch.zeros(R, K, dtype=torch.int8, device=dev)
EBL_K_major = torch.zeros(R, K, dtype=torch.int8, device=dev)

# Both fp16 (allocated, may be unused) and int32 (used by sm_89 path).
AxEBL_fp16 = torch.zeros(M, R, dtype=torch.float16, device=dev)
EARxBpEB_fp16 = torch.zeros(N, R, dtype=torch.float16, device=dev)
AxEBL_int32 = torch.zeros(M, R, dtype=torch.int32, device=dev)
EARxBpEB_int32 = torch.zeros(N, R, dtype=torch.int32, device=dev)

ApEA = torch.zeros(M, K, dtype=torch.int8, device=dev)
BpEB = torch.zeros(N, K, dtype=torch.int8, device=dev)

# PoW target/key + host signal scratch.
hh_size = pg.get_host_signal_header_size()
hs_size = pg.get_host_signal_sync_size()
host_signal_header = torch.zeros(hh_size, dtype=torch.int8, pin_memory=True)
host_signal_sync = torch.zeros(hs_size, dtype=torch.int8, device=dev)
pow_target = torch.full((8,), 0xFFFFFFFF, dtype=torch.uint32, device=dev)  # trivial
pow_key = torch.zeros(8, dtype=torch.uint32, device=dev)

import time
print(f"=== Calling pg.noisy_gemm M={M} N={N} K={K} R={R} sm_89 int32 path ===")

def call():
    pg.noisy_gemm(
        A, B,
        EAL, EAL_fp16,
        EBR, EBR_fp16,
        EAR_R_major, EBL_R_major,
        EAR_K_major, EBL_K_major,
        AxEBL_fp16, EARxBpEB_fp16,
        ApEA, BpEB,
        A_scales, B_scales, C,
        host_signal_header, host_signal_sync,
        pow_target, pow_key,
        AxEBL_int32, EARxBpEB_int32,   # int32 noising → sm_89 path
        128, 128, 64,                  # bM, bN, bK
        1, 1,                          # cM, cN
        2,                             # pipeline_stages (PoW = 2)
        None, True,                    # swizzle, swizzle_n_maj
        64, 64,                        # tile_size_m_noising_A, tile_size_n_noising_B
        64, 64,                        # tile_size_k_noising_A, tile_size_k_noising_B
        2, 2,                          # pipeline_stages_noising_A/B
        None, None,                    # k_blocks_per_split_*
        True, True,                    # run_noising_a, run_noising_b
        False, False,                  # skip_reduction, skip_denoising
        None,                          # inner_hash_counter
        False,                         # enable_debug
    )

try:
    # Warmup
    for _ in range(3):
        call()
    torch.cuda.synchronize()

    # Bench 30 iterations
    t0 = time.perf_counter()
    N_ITER = 30
    for _ in range(N_ITER):
        call()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N_ITER
    ops_per_iter = 2.0 * M * N * K
    tops = ops_per_iter / (dt * 1e12)
    rate = 1.0 / dt
    print(f"=== noisy_gemm BENCH ===")
    print(f"per-iter:  {dt*1e3:.2f} ms ({rate:.1f}/s)")
    print(f"main_TOPS: {tops:.1f}  (alpha-miner baseline: 66.4 TMAC/s)")
    print(f"C max abs: {round(C.abs().max().item(), 2)}")
    print(f"ApEA nnz : {(ApEA != 0).sum().item()}/{ApEA.numel()}")
    print(f"BpEB nnz : {(BpEB != 0).sum().item()}/{BpEB.numel()}")
except Exception as e:
    print(f"!!! noisy_gemm FAILED: {type(e).__name__}: {str(e)[:500]}")
    raise
