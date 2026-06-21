"""Phase-3 minimal test: is the clean-GEMM bug present for ALL tile shapes,
or only multi-warpgroup (bM>64) tiles? Discriminates atom-level bug vs
N-permutation/warp-tiling bug. Clean GEMM only (no noise, no denoise blame)."""
import sys, os
sys.path.insert(0, "/work/src")
os.environ.setdefault("PEARL_GEMM_DISABLE_R32", "TRUE")
import torch
from pearl_gemm import noisy_gemm
from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
from pearl_gemm_build_utils.kernel_configs.default_compiled_kernels import KERNEL_CONFIGS

cc = torch.cuda.get_device_capability(0)
print(f"Device: {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]})\n")

def clean_ref(tg):
    AB = torch._int_mm(tg.A.clone(), tg.B.clone().t())
    return torch.einsum("mn,m,n->mn", AB.to(torch.float32), tg.A_scales.clone(), tg.B_scales.clone()).cpu()

def run_clean(tg, gp):
    noisy_gemm(
        A=tg.A, B=tg.B, EAL=tg.EAL, EAL_fp16=tg.EAL_fp16,
        EAR_R_major=tg.EAR_R_major, EBL_R_major=tg.EBL_R_major,
        EAR_K_major=tg.EAR_K_major, EBL_K_major=tg.EBL_K_major,
        EBR=tg.EBR, EBR_fp16=tg.EBR_fp16, AxEBL_fp16=tg.AxEBL_fp16,
        EARxBpEB_fp16=tg.EARxBpEB_fp16, ApEA=tg.ApEA, BpEB=tg.BpEB,
        A_scales=tg.A_scales, B_scales=tg.B_scales, C=tg.C,
        host_signal_header_pinned=tg.host_signal_header_pinned,
        host_signal_sync=tg.host_signal_sync,
        AxEBL_int32=tg.AxEBL_int32, EARxBpEB_int32=tg.EARxBpEB_int32,
        tile_size_m=gp.tile_size_m, tile_size_n=gp.tile_size_n, tile_size_k=gp.tile_size_k,
        pipeline_stages=gp.pipeline_stages, cluster_size_m=gp.cluster_size_m,
        cluster_size_n=gp.cluster_size_n, swizzle=gp.swizzle, swizzle_n_maj=gp.swizzle_n_maj,
        tile_size_m_noising_A=gp.tile_size_m_noising_A, tile_size_n_noising_B=gp.tile_size_n_noising_B,
        tile_size_k_noising_A=gp.tile_size_k_noising_A, tile_size_k_noising_B=gp.tile_size_k_noising_B,
        k_blocks_per_split_noising_A=gp.k_blocks_per_split_noising_A,
        k_blocks_per_split_noising_B=gp.k_blocks_per_split_noising_B,
        run_noising_A=False, run_noising_B=False,
        skip_reduction=gp.skip_reduction, skip_denoising=True,
        pow_target=tg.pow_target, pow_key=tg.pow_key,
    )
    torch.cuda.synchronize()

m, n, k = 128, 256, 256
print(f"Clean GEMM (no noise, skip_denoising) at {m}x{n}x{k}, per matmul config:\n")
print(f"{'tile':>14} {'WGs':>4} {'kWarpRows':>9}  result")
for mc in sorted(KERNEL_CONFIGS.matmul_kernels, key=lambda c:(c.tile_size_m, c.tile_size_n, c.R)):
    bM = mc.tile_size_m
    wgs = bM // 64
    kwr = 4 * wgs
    gp = GEMMParam(m, n, k, skip_noising_a=True, skip_noising_b=True,
                   EARxBpEB_type_noising="int32", AxEBL_type_noising="int32",
                   matmul_config=mc, use_variable_scales=False)
    tg = GemmTensorGenerator(gp); tg.generate()
    try:
        run_clean(tg, gp)
        out = tg.C.cpu().to(torch.float32); ref = clean_ref(tg).to(torch.float32)
        d = (out - ref).abs()
        ok = torch.allclose(out, ref, atol=1e-1, rtol=1e-2)
        # also check: how many ROWS are correct (localizes warp-in-M corruption)
        row_ok = torch.isclose(out, ref, atol=1e-1, rtol=1e-2).all(dim=1)
        ncorrect = int(row_ok.sum())
        tag = f"{bM}x{mc.tile_size_n}x{mc.tile_size_k} R{mc.R}"
        print(f"{tag:>14} {wgs:>4} {kwr:>9}  {'PASS' if ok else 'FAIL'}  max|d|={d.max():.2f}  rows_ok={ncorrect}/{out.shape[0]}")
    except Exception as e:
        print(f"{bM}x{mc.tile_size_n} R{mc.R}: ERROR {type(e).__name__}: {str(e)[:60]}")
