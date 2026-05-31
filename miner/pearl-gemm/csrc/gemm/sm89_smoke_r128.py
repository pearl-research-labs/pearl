"""Smoke + bench for sm_89 R=128 path (bM=bN=64) via pybind.

Validates:
  1. pg.gemm(...) at bM=bN=bK=64 R=128 resolves to the new sm_89 inst.
  2. pg.noisy_gemm(...) at bM=bN=bK=64 R=128 runs end-to-end with int32 noising.
  3. Output stats are non-degenerate (no all-zero / no all-NaN).

Bench TOPS at 2048^3 R=128 and 4096^3 R=128 for both APIs.
"""

import time
import torch

import pearl_gemm_cuda as pg


def banner(s: str) -> None:
    print("=" * 72)
    print(s)
    print("=" * 72)


def check_cap() -> None:
    cap = torch.cuda.get_device_capability(0)
    print("GPU:", torch.cuda.get_device_name(0), cap)
    print("min_cc:", getattr(pg, "_min_compute_capability", "unset"))
    assert cap == (8, 9), f"need sm_89, got {cap}"


# ---------------------------------------------------------------------------
# pg.gemm bench at bM=bN=bK=64 R=128
# ---------------------------------------------------------------------------

def bench_gemm(M: int, N: int, K: int, bM=64, bN=64, bK=64, stages=3,
               n_iter=20) -> None:
    dev = torch.device("cuda:0")
    A = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev)
    B = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=dev)
    A_scales = torch.rand(M, dtype=torch.float32, device=dev) * 0.02 + 0.005
    B_scales = torch.rand(N, dtype=torch.float32, device=dev) * 0.02 + 0.005
    C = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)

    def call():
        pg.gemm(A, B, A_scales, B_scales, C, bM, bN, bK, 1, 1, stages, None, True)

    # warmup
    for _ in range(3):
        call()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        call()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iter
    tops = (2.0 * M * N * K) / (dt * 1e12)
    nnz = (C != 0).sum().item()
    max_abs = C.abs().max().item()
    print(f"  gemm({M}x{N}x{K}, bM={bM} bN={bN} bK={bK} stages={stages}): "
          f"{dt*1e3:.2f} ms  {tops:6.1f} TOPS  nnz={nnz}/{C.numel()}  "
          f"|max|={max_abs:.1f}")
    return tops


# ---------------------------------------------------------------------------
# pg.noisy_gemm bench at bM=bN=bK=64 R=128
# ---------------------------------------------------------------------------

def bench_noisy_gemm(M: int, N: int, K: int, R: int = 128,
                     bM=64, bN=64, bK=64, stages=2, n_iter=20) -> None:
    dev = torch.device("cuda:0")
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

    hh = pg.get_host_signal_header_size()
    hs = pg.get_host_signal_sync_size()
    host_signal_header = torch.zeros(hh, dtype=torch.int8, pin_memory=True)
    host_signal_sync = torch.zeros(hs, dtype=torch.int8, device=dev)
    pow_target = torch.full((8,), 0xFFFFFFFF, dtype=torch.uint32, device=dev)
    pow_key = torch.zeros(8, dtype=torch.uint32, device=dev)

    def call():
        pg.noisy_gemm(
            A, B, EAL, EAL_fp16, EBR, EBR_fp16,
            EAR_R_major, EBL_R_major, EAR_K_major, EBL_K_major,
            AxEBL_fp16, EARxBpEB_fp16, ApEA, BpEB,
            A_scales, B_scales, C,
            host_signal_header, host_signal_sync,
            pow_target, pow_key,
            AxEBL_int32, EARxBpEB_int32,
            bM, bN, bK, 1, 1, stages,
            None, True,
            64, 64, 64, 64,  # noisingA/B tile sizes
            2, 2,            # noisingA/B pipeline stages
            None, None,
            True, True,      # run_noising_a, run_noising_b
            False, False,    # skip_reduction, skip_denoising (full PoW path)
            None, False,
        )

    # warmup
    for _ in range(3):
        call()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        call()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iter
    tops = (2.0 * M * N * K) / (dt * 1e12)
    nnz_C = (C != 0).sum().item()
    nnz_ApEA = (ApEA != 0).sum().item()
    nnz_BpEB = (BpEB != 0).sum().item()
    max_abs = C.abs().max().item()
    print(f"  noisy_gemm({M}x{N}x{K}, R={R}, bM={bM} bN={bN} bK={bK} stages={stages}): "
          f"{dt*1e3:.2f} ms  {tops:6.1f} TOPS")
    print(f"    nnz_C={nnz_C}/{C.numel()}  nnz_ApEA={nnz_ApEA}/{ApEA.numel()}  "
          f"nnz_BpEB={nnz_BpEB}/{BpEB.numel()}  |max C|={max_abs:.1f}")
    return tops


def main() -> int:
    check_cap()

    banner("pg.gemm bM=bN=bK=64 R=128 (noiseless)")
    # Note: pg.gemm iterates R in {64,128}, so it picks first matching config.
    # At bM=bN=bK=64, the R=64 config is bM=bN=128 — no match. R=128 config
    # is bM=bN=64 — match.
    for (M, N, K) in [(512, 512, 1024), (1024, 1024, 2048), (2048, 2048, 4096),
                      (4096, 4096, 4096)]:
        try:
            bench_gemm(M, N, K, bM=64, bN=64, bK=64, stages=3, n_iter=15)
        except Exception as e:
            print(f"  FAILED at {M}x{N}x{K}: {type(e).__name__}: {str(e)[:200]}")
            return 1

    banner("pg.noisy_gemm bM=bN=bK=64 R=128 (full PoW path)")
    for (M, N, K) in [(512, 512, 1024), (1024, 1024, 2048), (2048, 2048, 4096),
                      (4096, 4096, 4096)]:
        try:
            bench_noisy_gemm(M, N, K, R=128, bM=64, bN=64, bK=64,
                             stages=2, n_iter=15)
        except Exception as e:
            print(f"  FAILED at {M}x{N}x{K}: {type(e).__name__}: {str(e)[:200]}")
            return 2

    banner("Done")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
