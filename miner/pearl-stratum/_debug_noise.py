"""Standalone test of pearl_gemm_cuda.noise_gen to isolate the sm_89 issue.

If this fails the same way the full pearl_gemm_noisy() did, the bug is in
noise_gen itself (and we know what to patch). If it succeeds, the bug is in
some other kernel called by pearl_gemm_noisy.
"""
import sys, torch, traceback

import pearl_gemm
import pearl_gemm_cuda as pgc
from pearl_gemm import get_required_scratchpad_bytes

print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} cap={torch.cuda.get_device_capability(0)}")
print(f"pearl_gemm_cuda from: {pgc.__file__}")
print(f"pearl_gemm has noise_gen: {hasattr(pgc, 'noise_gen')}")
print(f"pearl_gemm helpers: {[n for n in dir(pgc) if not n.startswith('_')][:20]}")

DEVICE = "cuda:0"
torch.cuda.set_device(0)

# Match the values pearl_gemm_noisy uses for 2048x2048x4096, R=128
M, N, K, R = 2048, 2048, 4096, 128

# These shapes are documented in pearl_gemm_interface.noise_gen()
EAL          = torch.empty((M, R),    dtype=torch.int8,    device=DEVICE)
EAL_fp16     = torch.empty((M, R),    dtype=torch.float16, device=DEVICE)
EAR_R_major  = torch.empty((K, R),    dtype=torch.int8,    device=DEVICE)
EAR_K_major  = torch.empty((R, K),    dtype=torch.int8,    device=DEVICE)
EBL_R_major  = torch.empty((K, R),    dtype=torch.int8,    device=DEVICE)
EBL_K_major  = torch.empty((R, K),    dtype=torch.int8,    device=DEVICE)
EBR          = torch.empty((N, R),    dtype=torch.int8,    device=DEVICE)
EBR_fp16     = torch.empty((N, R),    dtype=torch.float16, device=DEVICE)
key_A        = torch.zeros(32, dtype=torch.uint8, device=DEVICE)
key_B        = torch.zeros(32, dtype=torch.uint8, device=DEVICE)
# noise_gen accumulates BLAKE3 in an aux buffer of size S (returned by
# get_required_scratchpad_bytes for max(M,N)*K matrices)
matrix_bytes = max(M*K, N*K)
aux_size = get_required_scratchpad_bytes(matrix_bytes)
aux_buffer   = torch.empty(aux_size, dtype=torch.int32, device=DEVICE)

print(f"Buffers allocated. aux_size={aux_size} bytes")

print("Calling pgc.noise_gen(R=128, num_threads=64, ...)")
torch.cuda.synchronize()
try:
    pgc.noise_gen(
        128, 64,
        EAL, EAL_fp16,
        EAR_R_major, EAR_K_major,
        EBL_R_major, EBL_K_major,
        EBR, EBR_fp16,
        key_A, key_B,
        aux_buffer,
    )
    torch.cuda.synchronize()
    print("✓ noise_gen SUCCEEDED")
    print(f"EAL sample: {EAL[0, :8].cpu().tolist()}")
    print(f"EBR sample: {EBR[0, :8].cpu().tolist()}")
except Exception as e:
    print(f"✗ noise_gen FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)
