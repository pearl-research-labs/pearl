"""Per-iter timing for noisy_gemm to find the overhead source."""

import torch, time
import pearl_gemm_cuda as pg

dev = torch.device("cuda:0")
M, N, K, R = 2048, 2048, 4096, 64

A = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev)
B = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=dev)
A_scales = torch.rand(M, dtype=torch.float32, device=dev) * 0.02 + 0.005
B_scales = torch.rand(N, dtype=torch.float32, device=dev) * 0.02 + 0.005
C = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)
EAL = torch.zeros(M, R, dtype=torch.int8, device=dev)
EBR = torch.zeros(N, R, dtype=torch.int8, device=dev)
EAL_fp16 = torch.zeros(M, R, dtype=torch.float16, device=dev)
EBR_fp16 = torch.zeros(N, R, dtype=torch.float16, device=dev)
EAR_R_major = torch.zeros(K, R, dtype=torch.int8, device=dev)
EBL_R_major = torch.zeros(K, R, dtype=torch.int8, device=dev)
EAR_K_major = torch.zeros(R, K, dtype=torch.int8, device=dev)
EBL_K_major = torch.zeros(R, K, dtype=torch.int8, device=dev)
AxEBL_fp16 = torch.zeros(M, R, dtype=torch.float16, device=dev)
EARxBpEB_fp16 = torch.zeros(N, R, dtype=torch.float16, device=dev)
AxEBL_int32 = torch.zeros(M, R, dtype=torch.int32, device=dev)
EARxBpEB_int32 = torch.zeros(N, R, dtype=torch.int32, device=dev)
ApEA = torch.zeros(M, K, dtype=torch.int8, device=dev)
BpEB = torch.zeros(N, K, dtype=torch.int8, device=dev)

hh_size = pg.get_host_signal_header_size()
hs_size = pg.get_host_signal_sync_size()
print(f"hh_size={hh_size}  hs_size={hs_size}")
host_signal_header = torch.zeros(hh_size, dtype=torch.int8, pin_memory=True)
host_signal_sync = torch.zeros(hs_size, dtype=torch.int8, device=dev)
pow_target = torch.full((8,), 0xFFFFFFFF, dtype=torch.uint32, device=dev)
pow_key = torch.zeros(8, dtype=torch.uint32, device=dev)

def call_pow():
    pg.noisy_gemm(
        A, B, EAL, EAL_fp16, EBR, EBR_fp16,
        EAR_R_major, EBL_R_major, EAR_K_major, EBL_K_major,
        AxEBL_fp16, EARxBpEB_fp16, ApEA, BpEB,
        A_scales, B_scales, C,
        host_signal_header, host_signal_sync,
        pow_target, pow_key,
        AxEBL_int32, EARxBpEB_int32,
        128, 128, 64, 1, 1, 2,
        None, True, 64, 64, 64, 64, 2, 2, None, None,
        True, True, False, False, None, False,
    )

def call_noreduce():
    # skip_reduction=True → Noiseless (no PoW search loop)
    pg.noisy_gemm(
        A, B, EAL, EAL_fp16, EBR, EBR_fp16,
        EAR_R_major, EBL_R_major, EAR_K_major, EBL_K_major,
        AxEBL_fp16, EARxBpEB_fp16, ApEA, BpEB,
        A_scales, B_scales, C,
        host_signal_header, host_signal_sync,
        pow_target, pow_key,
        AxEBL_int32, EARxBpEB_int32,
        128, 128, 64, 1, 1, 3,         # stages=3 for Noiseless
        None, True, 64, 64, 64, 64, 2, 2, None, None,
        True, True, True, False, None, False,   # skip_reduction=True
    )

def call_no_denoise():
    # skip_reduction=True + skip_denoising=True → bare gemm-after-noising
    pg.noisy_gemm(
        A, B, EAL, EAL_fp16, EBR, EBR_fp16,
        EAR_R_major, EBL_R_major, EAR_K_major, EBL_K_major,
        AxEBL_fp16, EARxBpEB_fp16, ApEA, BpEB,
        A_scales, B_scales, C,
        host_signal_header, host_signal_sync,
        pow_target, pow_key,
        AxEBL_int32, EARxBpEB_int32,
        128, 128, 64, 1, 1, 3,
        None, True, 64, 64, 64, 64, 2, 2, None, None,
        True, True, True, True, None, False,    # skip_reduction=skip_denoising=True
    )

call = call_pow

# Test pow with different pow_target settings
def make_target(level):
    """level=0 → target=MAX (trivially pass), level=8 → target=0 (impossible)"""
    t = torch.zeros(8, dtype=torch.uint32, device=dev)
    if level < 8:
        t[level:] = 0xFFFFFFFF
    return t

print("\n=== PoW with different targets ===")
for level, name in [(0, "target=MAX"), (4, "target=middle"), (7, "target=hard")]:
    pow_target.copy_(make_target(level))
    try:
        call_pow(); torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call_pow()
        end.record()
        end.synchronize()
        t = start.elapsed_time(end)
        print(f"  level={level} {name:15} = {t:8.3f} ms")
    except Exception as e:
        print(f"  level={level} FAILED: {e}")

# Reset to trivial target
pow_target.copy_(make_target(0))
print("\n=== Path comparison @ M=N=2048 K=4096 ===")
for name, fn in [("PoW (skip_red=F)", call_pow),
                 ("Noiseless (skip_red=T)", call_noreduce),
                 ("Bare (skip_red+den=T)", call_no_denoise)]:
    try:
        fn(); torch.cuda.synchronize()  # warmup
        times = []
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        avg = sum(times)/len(times)
        ops = 2.0 * M * N * K
        tops = ops / (avg * 1e-3 * 1e12)
        print(f"{name:28} avg={avg:8.3f} ms  TOPS={tops:7.2f}")
    except Exception as e:
        print(f"{name:28} FAILED: {type(e).__name__}: {str(e)[:120]}")
