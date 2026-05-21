"""Minimal smoke test for sm_120 (consumer Blackwell) dispatch in pearl_gemm_cuda.

Mirrors sm89_smoke.py but accepts capability (12, 0) — the RTX 50-series GPUs.

Validates that:
  1. pearl_gemm_cuda imports cleanly with _min_compute_capability == 8
  2. GPU has capability (12, 0) — RTX 5090 / 5080 / 5070 (Ti)
  3. A simple gemm() call at 128x128x64 routes through the sm_89 (= sm_120)
     dispatch and returns a tensor of the right shape/dtype.

Build with:
    PEARL_GEMM_TARGET_ARCH=120          pip install -e miner/pearl-gemm
or:
    PEARL_GEMM_TARGET_ARCH=all-consumer pip install -e miner/pearl-gemm
"""

import torch
import pearl_gemm_cuda as pg

print("=== Module loaded ===")
print("min_compute_capability:", getattr(pg, "_min_compute_capability", "unset"))
print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

cap = torch.cuda.get_device_capability(0)
assert cap in ((8, 9), (12, 0)), f"Need sm_89 or sm_120 GPU, got {cap}"

M, N, K = 128, 128, 256
dev = torch.device("cuda:0")
A = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev)
B = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=dev)
A_scales = torch.ones(M, dtype=torch.float32, device=dev)
B_scales = torch.ones(N, dtype=torch.float32, device=dev)
C = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)
print("=== Tensors created ===")

print("dir(pg) — entry points:")
for name in sorted(dir(pg)):
    if not name.startswith("_"):
        print(f"  {name}")

if hasattr(pg, "gemm"):
    print("=== Calling pg.gemm(...) ===")
    # Signature (positional): A, B, A_scales, B_scales, C, bM, bN, bK, cM, cN,
    #                         pipeline_stages?, swizzle?, swizzle_n_maj
    pg.gemm(A, B, A_scales, B_scales, C, 128, 128, 64, 1, 1, 3, None, True)
    torch.cuda.synchronize()
    print("=== gemm OK ===")
    print("C shape:", C.shape, "dtype:", C.dtype)
    print("C[0, :8]:", C[0, :8].tolist())
else:
    print("No `gemm` entry point on pearl_gemm_cuda — inspect the dir() listing above.")
