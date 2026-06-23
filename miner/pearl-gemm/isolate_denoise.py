"""ISOLATION diagnostic for the sm_120 noisy_gemm numeric bug.
Following systematic-debugging Phase 1: gather evidence, find WHICH stage breaks.
Leading hypothesis: PR#118 two-phase denoise (DenoisePhase1Consumed barrier) race.

Tests on ONE small shape (128x128x256), ONE config, int32 noising:
  T1: clean GEMM (skip both noising)  -> must match clean ref  (isolates base matmul)
  T2: skip_denoising=True             -> match NOISED ref       (isolates matmul-w-noise vs denoise)
  T3: noise A only                    -> which path corrupts
  T4: noise B only
  T5: full path (baseline repro)
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
print(f"Device: {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]})\n")

mc = sorted(KERNEL_CONFIGS.matmul_kernels, key=lambda c: (c.tile_size_m * c.tile_size_n, c.R))[0]
print(
    f"Config: {mc.tile_size_m}x{mc.tile_size_n}x{mc.tile_size_k} stage={mc.pipeline_stages} R={mc.R}\n"
)
m, n, k = 128, 128, 256


def clean_ref(tg):
    AB = torch._int_mm(tg.A.clone(), tg.B.clone().t())
    return torch.einsum(
        "mn,m,n->mn", AB.to(torch.float32), tg.A_scales.clone(), tg.B_scales.clone()
    ).cpu()


def noised_ref(tg):
    # (A+EA) @ (B+EB)^T  with int8 wrap, then scales — what kernel computes BEFORE denoise
    A = tg.A.clone().to(torch.int32)
    B = tg.B.clone().to(torch.int32)
    EA = torch._int_mm(tg.EAL.clone(), tg.EAR_R_major.clone().t())  # (m,k)
    EB = torch._int_mm(tg.EBR.clone(), tg.EBL_R_major.clone().t())  # (n,k)
    ApEA = ((A + EA).to(torch.int8)).to(torch.int32)
    BpEB = ((B + EB).to(torch.int8)).to(torch.int32)
    AB = ApEA @ BpEB.t()
    return torch.einsum(
        "mn,m,n->mn", AB.to(torch.float32), tg.A_scales.clone(), tg.B_scales.clone()
    ).cpu()


def run(tg, gp, skip_denoising):
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
        skip_denoising=skip_denoising,
        pow_target=tg.pow_target,
        pow_key=tg.pow_key,
    )
    torch.cuda.synchronize()


def report(tag, out, ref):
    out = out.to(torch.float32)
    ref = ref.to(torch.float32)
    d = (out - ref).abs()
    ok = torch.allclose(out, ref, atol=1e-1, rtol=1e-2)
    print(
        f"[{tag}] {'PASS' if ok else 'FAIL'}  max|d|={d.max():.3f} mean|d|={d.mean():.4f}  "
        f"ref|.|mean={ref.abs().mean():.3f} out|.|mean={out.abs().mean():.3f}"
    )
    return ok


def make(skip_a, skip_b):
    gp = GEMMParam(
        m,
        n,
        k,
        skip_noising_a=skip_a,
        skip_noising_b=skip_b,
        EARxBpEB_type_noising="int32",
        AxEBL_type_noising="int32",
        matmul_config=mc,
        use_variable_scales=False,
    )
    tg = GemmTensorGenerator(gp)
    tg.generate()
    return gp, tg


print("=== T1: clean GEMM (skip A & B noising, denoise on) vs clean ref ===")
gp, tg = make(True, True)
run(tg, gp, skip_denoising=False)
report("T1 clean", tg.C.cpu(), clean_ref(tg))

print("\n=== T5: full path (baseline repro) vs clean ref ===")
gp, tg = make(False, False)
run(tg, gp, skip_denoising=False)
report("T5 full", tg.C.cpu(), clean_ref(tg))

print("\n=== T2: full noise but skip_denoising=True vs NOISED ref ===")
gp, tg = make(False, False)
run(tg, gp, skip_denoising=True)
report("T2 nodenoise", tg.C.cpu(), noised_ref(tg))

print("\n=== T3: noise A only (skip B), denoise on, vs clean ref ===")
gp, tg = make(False, True)
run(tg, gp, skip_denoising=False)
report("T3 Aonly", tg.C.cpu(), clean_ref(tg))

print("\n=== T4: noise B only (skip A), denoise on, vs clean ref ===")
gp, tg = make(True, False)
run(tg, gp, skip_denoising=False)
report("T4 Bonly", tg.C.cpu(), clean_ref(tg))
