"""Receipt-sidechannel native-global-journal test grid.

Opt in with:

    PEARL_GEMM_KERNEL_CONFIG_MODULE=pearl_gemm_build_utils.kernel_configs.rsc_native_global_journal_kernels

This keeps the build small while covering the accepted baseline shape and the
RSC-003 missing 128x64x128 native-global-journal candidate.
"""

from pearl_gemm_build_utils.kernel_configs import (
    KernelCompilationGrid,
    MatmulKernelConfig,
    NoisingAKernelConfig,
    NoisingBKernelConfig,
)


_matmul_kernels = [
    MatmulKernelConfig(
        tile_size_m=128,
        tile_size_n=256,
        tile_size_k=128,
        R=128,
        pipeline_stages=3,
        cM=1,
        cN=1,
    ),
    MatmulKernelConfig(
        tile_size_m=128,
        tile_size_n=64,
        tile_size_k=128,
        R=128,
        pipeline_stages=3,
        cM=1,
        cN=1,
    ),
]

_noising_a_kernels = [
    NoisingAKernelConfig(
        tile_size_m=64,
        tile_size_k=64,
        R=128,
        pipeline_stages=2,
        AxEBL_type=dtype,
    )
    for dtype in ("fp16", "int32")
]

_noising_b_kernels = [
    NoisingBKernelConfig(
        tile_size_n=64,
        tile_size_k=64,
        R=128,
        pipeline_stages=2,
        EARxBpEB_type=dtype,
    )
    for dtype in ("fp16", "int32")
]

KERNEL_CONFIGS = KernelCompilationGrid(
    matmul_kernels=_matmul_kernels,
    noising_a_kernels=_noising_a_kernels,
    noising_b_kernels=_noising_b_kernels,
)
