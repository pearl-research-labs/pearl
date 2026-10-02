"""SM100 FP8 grouped GEMM (CuTe DSL blockwise-scaled grouped kernel)."""

from ._host import (
    MAX_GROUPED_ROWS,
    SCALE_GRANULARITY_MNK,
    WIDE_TILE_MIN_TOKENS_PER_GROUP,
    GroupedFp8GemmConfig,
    grouped_fp8_gemm,
    grouped_fp8_gemm_scale_shapes,
)

__all__ = [
    "MAX_GROUPED_ROWS",
    "SCALE_GRANULARITY_MNK",
    "WIDE_TILE_MIN_TOKENS_PER_GROUP",
    "GroupedFp8GemmConfig",
    "grouped_fp8_gemm",
    "grouped_fp8_gemm_scale_shapes",
]
