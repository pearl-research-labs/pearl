"""
Default compiled kernels configuration.

Minimal set of kernels for development and PR CI.
This replaces default_compiled_kernels.jsonnet with Python + pydantic.

When PEARL_GEMM_TARGET_ARCH is a consumer-GPU build (89 / 120 / all-consumer)
the matmul/noising grids are narrowed to the shapes that fit the ~100 KB
opt-in smem cap shared by Ada (sm_89) and consumer Blackwell (sm_120), and
that have explicit sm_89 instantiations in pearl-gemm/csrc/gemm/pearl_*_sm89_inst.cu.
sm_120 reuses the sm_89 source path; see pearl_gemm_launch_template.h.
"""

import os

from pearl_gemm_build_utils.kernel_configs import (
    KernelCompilationGrid,
    MatmulKernelConfig,
    NoisingAKernelConfig,
    NoisingBKernelConfig,
)

_TARGET_ARCH = os.environ.get("PEARL_GEMM_TARGET_ARCH", "90a")
# Consumer-GPU builds (Ada + consumer Blackwell) share the same ~100 KB
# opt-in smem cap and use the same sm_89 source path, so they take the
# narrow tile grid below.
_NARROW_SMEM_ARCH = _TARGET_ARCH in ("89", "120", "all-consumer")

if _NARROW_SMEM_ARCH:
    # sm_89 (Ada) — bK=128 won't fit in 99 KB opt-in smem cap; bN=256 was
    # benched and rejected. Two pipeline depths to match Noiseless/Denoise
    # (stages=3) and Pow (stages=2) instantiations in pearl_gemm_sm89_*_inst.cu.
    #
    # R=64  -> bM=bN=128 (best TOPS; alphapool rank=64 mining params).
    # R=128 -> bM=64, bN ∈ {64, 128} (alphapool rank=128 mining params; bM=128
    #          R=128 would put SharedStorageDenoise arm-2 at 128 KB > 99 KB
    #          sm_89 opt-in cap).
    #   - bN=64  Denoise SharedStorage = 65 KB (existing safe baseline)
    #   - bN=128 Denoise SharedStorage = 97 KB (fits with 2 KB headroom under
    #            99 KB cap; 2× output tile area per CTA → higher TOPS).
    # kernel_traits_sm89.hpp derives kNumWarps = bM/16 so both bM=128 (8 warps,
    # 2x4 grid) and bM=64 (4 warps, 2x2 grid) compile. R2S epilogue thread
    # geometry validates: (bM/16)*32 = kNumThreads, S2GValueLayoutC = (16, bN/32).
    _matmul_kernels = [
        # R=64: bM=bN=128 production path
        MatmulKernelConfig(
            tile_size_m=128,
            tile_size_n=128,
            tile_size_k=64,
            R=64,
            pipeline_stages=stages,
            cM=1,
            cN=1,
        )
        for stages in (2, 3)
    ] + [
        # R=128: bM=64, bN ∈ {64, 128} — alphapool rank=128 paths
        MatmulKernelConfig(
            tile_size_m=64,
            tile_size_n=bN,
            tile_size_k=64,
            R=128,
            pipeline_stages=stages,
            cM=1,
            cN=1,
        )
        for bN in (64, 128)
        for stages in (2, 3)
    ] + [
        # Wave-10: bM=128 bN=256 R=128 — the ONLY combo whose per-thread output
        # footprint matches MinerSettings.cols_pattern = [0,1,8,9,...,248,249]
        # (64 cols, period 256) when paired with the wave-9 kernel_traits patch
        # (kWarpRows=bM/16, kWarpCols=1).  Smem fits via kRegisterResidentDenoise
        # which is auto-enabled inside KernelTraitsSm89 for this combo (see
        # kernel_traits_sm89.hpp line ~81).  Without this entry, MATMUL_CONFIG_SWITCH
        # has no case for (128,256,64,128) so noisy_gemm dispatch fails at runtime
        # and zero shares are credited.
        MatmulKernelConfig(
            tile_size_m=128,
            tile_size_n=256,
            tile_size_k=64,
            R=128,
            pipeline_stages=stages,
            cM=1,
            cN=1,
        )
        for stages in (2, 3)
    ]

    # sm_89 noising kernels support R in {64, 128} (template static_assert
    # in pearl_noising{A,B}_kernel_sm89.h:97/100) and both fp16 and int32
    # denoise outputs. The sm_89 port runs with NoReduction=true always — the
    # cross-CTA atomic-add reduction path used by Hopper int32 noising is not
    # implemented here, so each CTA accumulates the full K dimension in
    # registers (1D grid over M/N). For fp16 this matches the kernel's
    # static_assert `denoise_dtype_bits == 32 || NoReduction`; for int32 the
    # behavior is the same NoReduction-true path the existing kernel ships
    # with. Both dtypes flow through the same register accumulator + epilogue
    # (registers → gmem for noisingA, registers → smem → gmem via union for
    # noisingB) — smem budget is unchanged because the epilogue smem arm in
    # noisingB is sized in `ElementDenoise` (fp16 is strictly smaller than
    # int32, so the mainloop arm of the union always dominates the total).
    _noising_a_kernels = [
        NoisingAKernelConfig(
            tile_size_m=64,
            tile_size_k=64,
            R=R,
            pipeline_stages=2,
            AxEBL_type=dtype,
        )
        for R in (64, 128)
        for dtype in ("fp16", "int32")
    ]
    _noising_b_kernels = [
        NoisingBKernelConfig(
            tile_size_n=64,
            tile_size_k=64,
            R=R,
            pipeline_stages=2,
            EARxBpEB_type=dtype,
        )
        for R in (64, 128)
        for dtype in ("fp16", "int32")
    ]
else:
    # Hopper / fat-binary path — original config.
    _matmul_kernels = []
    for R in [64, 128]:
        _matmul_kernels.append(
            MatmulKernelConfig(
                tile_size_m=128,
                tile_size_n=256,
                tile_size_k=128,
                R=R,
                pipeline_stages=3,
                cM=1,
                cN=1,
            )
        )

    _noising_a_kernels = [
        NoisingAKernelConfig(
            tile_size_m=64,
            tile_size_k=64,
            R=R,
            pipeline_stages=2,
            AxEBL_type=dtype,
        )
        for R in [64, 128]
        for dtype in ["fp16", "int32"]
    ]

    _noising_b_kernels = [
        NoisingBKernelConfig(
            tile_size_n=64,
            tile_size_k=64,
            R=R,
            pipeline_stages=2,
            EARxBpEB_type=dtype,
        )
        for R in [64, 128]
        for dtype in ["fp16", "int32"]
    ]

KERNEL_CONFIGS = KernelCompilationGrid(
    matmul_kernels=_matmul_kernels,
    noising_a_kernels=_noising_a_kernels,
    noising_b_kernels=_noising_b_kernels,
)
