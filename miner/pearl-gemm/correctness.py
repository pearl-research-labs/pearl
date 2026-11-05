"""Standalone correctness harness — exercises the REAL mining path (noisy_gemm)
that PR#118 fixed for sm_120. Mirrors test_int7_noisy_gemm from the suite.
No pytest needed.
"""

import os
import sys

sys.path.insert(0, "/work/src")
os.environ.setdefault("PEARL_GEMM_DISABLE_R32", "TRUE")
import torch
from pearl_gemm import noisy_gemm
from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
from pearl_gemm_build_utils.kernel_configs.default_compiled_kernels import KERNEL_CONFIGS

cc = torch.cuda.get_device_capability(0)
print(f"Device: {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]})")


def compute_ref_tensor(tg):
    A = tg.A.clone()
    B = tg.B.clone()
    As = tg.A_scales.clone()
    Bs = tg.B_scales.clone()
    AB = torch._int_mm(A, B.t())
    return torch.einsum("mn,m,n->mn", AB.to(torch.float32), As, Bs).cpu()


def run_noisy(tg, gp):
    noisy_gemm(
        A=tg.A,
        B=tg.B,
        EAL=tg.EAL,
        EAL_fp16=tg.EAL_fp16,
        EAR_R_major=tg.EAR_R_major,
        EBL_R_major=tg.EBL_R_major,
        EAR_K_major=tg.EAR_K_major,
        EBL_K_major=tg.EBL_K_major,
        EBR=tg.EBR,
        EBR_fp16=tg.EBR_fp16,
        AxEBL_fp16=tg.AxEBL_fp16,
        EARxBpEB_fp16=tg.EARxBpEB_fp16,
        ApEA=tg.ApEA,
        BpEB=tg.BpEB,
        A_scales=tg.A_scales,
        B_scales=tg.B_scales,
        C=tg.C,
        host_signal_header_pinned=tg.host_signal_header_pinned,
        host_signal_sync=tg.host_signal_sync,
        AxEBL_int32=tg.AxEBL_int32,
        EARxBpEB_int32=tg.EARxBpEB_int32,
        tile_size_m=gp.tile_size_m,
        tile_size_n=gp.tile_size_n,
        tile_size_k=gp.tile_size_k,
        pipeline_stages=gp.pipeline_stages,
        cluster_size_m=gp.cluster_size_m,
        cluster_size_n=gp.cluster_size_n,
        swizzle=gp.swizzle,
        swizzle_n_maj=gp.swizzle_n_maj,
        tile_size_m_noising_A=gp.tile_size_m_noising_A,
        tile_size_n_noising_B=gp.tile_size_n_noising_B,
        tile_size_k_noising_A=gp.tile_size_k_noising_A,
        tile_size_k_noising_B=gp.tile_size_k_noising_B,
        k_blocks_per_split_noising_A=gp.k_blocks_per_split_noising_A,
        k_blocks_per_split_noising_B=gp.k_blocks_per_split_noising_B,
        run_noising_A=not gp.skip_noising_a,
        run_noising_B=not gp.skip_noising_b,
        skip_reduction=gp.skip_reduction,
        skip_denoising=False,
        pow_target=tg.pow_target,
        pow_key=tg.pow_key,
    )
    torch.cuda.synchronize()


passed = failed = 0
print("\n=== noisy_gemm end-to-end (the mining path) ===")
for mc in KERNEL_CONFIGS.matmul_kernels:
    for m, n, k in [(128, 128, 256), (1024, 1024, 512), (1025, 1032, 512), (8192, 6144, 4096)]:
        for dt in ["fp16", "int32"]:
            try:
                gp = GEMMParam(
                    m,
                    n,
                    k,
                    skip_noising_a=False,
                    skip_noising_b=False,
                    EARxBpEB_type_noising=dt,
                    AxEBL_type_noising=dt,
                    matmul_config=mc,
                    use_variable_scales=False,
                )
                tg = GemmTensorGenerator(gp)
                tg.generate()
                run_noisy(tg, gp)
                C_ref = compute_ref_tensor(tg)
                torch.testing.assert_close(
                    tg.C.cpu(), C_ref.to(torch.bfloat16), atol=1e-1, rtol=1e-2
                )
                print(f"  R{mc.R} {m}x{n}x{k} {dt}: PASS")
                passed += 1
            except AssertionError:
                print(f"  R{mc.R} {m}x{n}x{k} {dt}: FAIL (numeric mismatch)")
                failed += 1
            except Exception as e:
                print(f"  R{mc.R} {m}x{n}x{k} {dt}: ERROR {type(e).__name__}: {str(e)[:100]}")
                failed += 1

print(f"\n=== RESULT: {passed} passed, {failed} failed ===")
sys.exit(0 if failed == 0 else 1)
