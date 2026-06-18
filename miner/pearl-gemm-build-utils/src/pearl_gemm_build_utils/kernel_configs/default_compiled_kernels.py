"""
Default compiled kernels configuration.

Minimal set of kernels for development and PR CI.
This replaces default_compiled_kernels.jsonnet with Python + pydantic.
"""

from pearl_gemm_build_utils.kernel_configs import (
    KernelCompilationGrid,
    MatmulKernelConfig,
    NoisingAKernelConfig,
    NoisingBKernelConfig,
)

# Build matmul kernels
_matmul_kernels = []

# 128x256x128 stages=3 — the original Hopper-optimal config (~210 KB SMEM).
# Does NOT fit on Blackwell consumer / GB10 (99 KB SMEM cap per CTA), but
# keep it as the default for sm_90 (H100/H200) builds.
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

# 64x128x64 stages=2 — Blackwell consumer config (~50 KB SMEM). Required
# for sm_120/121 (RTX 50-series, GB10) which have a 99 KB SMEM-per-CTA cap.
for R in [64, 128]:
    _matmul_kernels.append(
        MatmulKernelConfig(
            tile_size_m=64,
            tile_size_n=128,
            tile_size_k=64,
            R=R,
            pipeline_stages=2,
            cM=1,
            cN=1,
        )
    )

# 128x128x64 stages=2 — preserves 2 MMA warpgroups (kernel layout assumes
# <=2 MMA WGs tiled in M). ~80 KB SMEM, fits GB10 cap.
for R in [64, 128]:
    _matmul_kernels.append(
        MatmulKernelConfig(
            tile_size_m=128,
            tile_size_n=128,
            tile_size_k=64,
            R=R,
            pipeline_stages=2,
            cM=1,
            cN=1,
        )
    )

# 64x64x64 stages=2 — Blackwell consumer minimum tile: comfortably fits in
# 99 KB SMEM even with 4 denoise buffers and full PoUW reductions.
for R in [64, 128]:
    _matmul_kernels.append(
        MatmulKernelConfig(
            tile_size_m=64,
            tile_size_n=64,
            tile_size_k=64,
            R=R,
            pipeline_stages=2,
            cM=1,
            cN=1,
        )
    )

# 128x128x64 stages=3 — Blackwell consumer; matches the noisy_gemm heuristic
# get_pipeline_stages() which computes 3 stages fit in ~99 KB SMEM with R=128 and
# skip_denoising=false (the canonical mining path).
for R in [64, 128]:
    _matmul_kernels.append(
        MatmulKernelConfig(
            tile_size_m=128,
            tile_size_n=128,
            tile_size_k=64,
            R=R,
            pipeline_stages=3,
            cM=1,
            cN=1,
        )
    )

# Noising A: 64x64, fp16/int32
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

# Noising B: 64x64, fp16/int32
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

# Path 3: canonical 128x256x128 stages=2 — now fits GB10 after Step 1-3 SMEM
# restructuring (denoise serialized, smem_C removed). ~98 KB SMEM usage.
for R in [64, 128]:
    _matmul_kernels.append(
        MatmulKernelConfig(
            tile_size_m=128,
            tile_size_n=256,
            tile_size_k=128,
            R=R,
            pipeline_stages=2,
            cM=1,
            cN=1,
        )
    )

KERNEL_CONFIGS = KernelCompilationGrid(
    matmul_kernels=_matmul_kernels,
    noising_a_kernels=_noising_a_kernels,
    noising_b_kernels=_noising_b_kernels,
)
