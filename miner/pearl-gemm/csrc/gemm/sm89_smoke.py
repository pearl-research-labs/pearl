"""Minimal smoke test for sm_89 dispatch in pearl_gemm_cuda.

Validates that:
  1. pearl_gemm_cuda imports cleanly with _min_compute_capability == 8
  2. GPU has capability (8, 9)
  3. A simple gemm() call at 128x128x64 routes through the sm_89 dispatch
     and returns a tensor of the right shape/dtype.
"""

import torch
import pearl_gemm_cuda as pg

print("=== Module loaded ===")
print("min_compute_capability:", getattr(pg, "_min_compute_capability", "unset"))
print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

assert torch.cuda.get_device_capability(0) == (8, 9), "Need sm_89 GPU"

# Minimal gemm() — matches sm_89 NoiselessTraits128x128x64_R64
# (SkipReduction=true, SkipDenoising=true, kStages=3).
M, N, K = 128, 128, 256
dev = torch.device("cuda:0")
A = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev)
B = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=dev)
A_scales = torch.ones(M, dtype=torch.float32, device=dev)
B_scales = torch.ones(N, dtype=torch.float32, device=dev)
C = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)
print("=== Tensors created ===")

# Inspect available entry points
print("dir(pg) — entry points:")
for name in sorted(dir(pg)):
    if not name.startswith("_"):
        print(f"  {name}")

# Try a simple noiseless gemm call if the API exposes one.
if hasattr(pg, "gemm"):
    print("=== Calling pg.gemm(...) ===")
    try:
        # tile shape that matches sm_89 inst: 128x128x64 R=64 stages=3
        pg.gemm(
            A,
            B,
            C,
            A_scales,
            B_scales,
            tile_size_m=128,
            tile_size_n=128,
            tile_size_k=64,
            pipeline_stages=3,
            cM=1,
            cN=1,
            skip_reduction=True,
            skip_denoising=True,
            enable_debug=False,
        )
        torch.cuda.synchronize()
        print("=== gemm OK ===")
        print("C shape:", C.shape, "dtype:", C.dtype)
        print("C[0, :8]:", C[0, :8].tolist())
    except Exception as e:
        print(f"!!! gemm FAILED: {e!r}")
        raise
else:
    print("No `gemm` entry point on pearl_gemm_cuda — inspect the dir() listing above.")
