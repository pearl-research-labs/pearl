"""Profile noisy_gemm subcomponents on sm_89."""

import torch
import time
import pearl_gemm_cuda as pg

dev = torch.device("cuda:0")
print("min_cc:", pg._min_compute_capability)

M, N, K, R = 2048, 2048, 4096, 64

A = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev)
B = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=dev)
A_scales = torch.rand(M, dtype=torch.float32, device=dev) * 0.02 + 0.005
B_scales = torch.rand(N, dtype=torch.float32, device=dev) * 0.02 + 0.005
C = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)

def bench(name, fn, n_iter=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iter
    print(f"{name:25} {dt*1e3:8.3f} ms")
    return dt

bench("gemm (no noise)",
      lambda: pg.gemm(A, B, A_scales, B_scales, C, 128, 128, 64, 1, 1, 3, None, True))

# noise_A: int32 AxEBL output
EAL = torch.zeros(M, R, dtype=torch.int8, device=dev)
AxEBL_int32 = torch.zeros(M, R, dtype=torch.int32, device=dev)
ApEA = torch.zeros(M, K, dtype=torch.int8, device=dev)
EAR_R_major = torch.zeros(K, R, dtype=torch.int8, device=dev)
EBL_K_major = torch.zeros(R, K, dtype=torch.int8, device=dev)
bench("noise_A int32 (64x64)",
      lambda: pg.noise_A(A, EAL, AxEBL_int32, ApEA, EAR_R_major, EBL_K_major, 64, 64, 2, None))

# noise_B: int32 EARxBpEB output
EBR = torch.zeros(N, R, dtype=torch.int8, device=dev)
EARxBpEB_int32 = torch.zeros(N, R, dtype=torch.int32, device=dev)
BpEB = torch.zeros(N, K, dtype=torch.int8, device=dev)
EAR_K_major = torch.zeros(R, K, dtype=torch.int8, device=dev)
EBL_R_major = torch.zeros(K, R, dtype=torch.int8, device=dev)
bench("noise_B int32 (64x64)",
      lambda: pg.noise_B(B, EBR, EARxBpEB_int32, BpEB, EAR_K_major, EBL_R_major, 64, 64, 2, None))
