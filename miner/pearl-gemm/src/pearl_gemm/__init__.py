"""pearl_gemm: CuTe DSL kernels for Pearl's FP8 scheme."""

from .grouped_fp8_gemm import (
    GroupedFp8GemmConfig,
    grouped_fp8_gemm,
    grouped_fp8_gemm_scale_shapes,
)
from .grouped_mixed_gemm import (
    GroupedMixedGemmConfig,
    grouped_mixed_gemm,
    supports_grouped_mixed_gemm,
)
from .mixed_gemm import (
    MixedGemmConfig,
    default_mixed_gemm_config,
    mixed_gemm,
    supports_lottery_family,
    validate_mixed_gemm_config,
)
from .noise_lines import (
    LABEL_E1,
    LABEL_E2,
    LABEL_F1,
    LABEL_F2,
    noise_lines,
)
from .noisy_quant import (
    NoiseLoadMode,
    NoisyQuantConfig,
    default_noisy_quant_config,
    noisy_quant,
    pack_noise_factor,
    validate_noisy_quant_config,
)
from .noisy_quant_b import (
    NoisyQuantBConfig,
    noisy_quant_b,
)
from .pow import (
    Hit,
    HitRecordLayout,
    HitSignal,
    HitSignalConfig,
    HitSignalPoisonedError,
)
from .pre_quant import (
    PreQuantConfig,
    pre_quant,
    pre_quant_output_shapes,
)
from .protocol_constants import (
    BLOCK_SCALE_GROUP,
    NOISE_TARGET_NORM,
    PACKED_NOISE_K,
    PEEL_COLS,
    QUANT_MAX,
    R,
)
from .tensor_hash_plus_stats import (
    TensorHashConfig,
    get_tensor_hash_plus_stats_config,
    tensor_hash,
    tensor_hash_plus_stats,
    tensor_hash_plus_stats_b,
    tensor_hash_plus_stats_record_is_legal,
    tensor_hash_scratchpad_bytes,
    tensor_hash_workspace_bytes,
)

__all__ = [
    "BLOCK_SCALE_GROUP",
    "GroupedFp8GemmConfig",
    "GroupedMixedGemmConfig",
    "Hit",
    "HitRecordLayout",
    "HitSignal",
    "HitSignalConfig",
    "HitSignalPoisonedError",
    "LABEL_E1",
    "LABEL_E2",
    "LABEL_F1",
    "LABEL_F2",
    "MixedGemmConfig",
    "NOISE_TARGET_NORM",
    "PACKED_NOISE_K",
    "PEEL_COLS",
    "NoiseLoadMode",
    "NoisyQuantBConfig",
    "NoisyQuantConfig",
    "PreQuantConfig",
    "QUANT_MAX",
    "R",
    "TensorHashConfig",
    "default_mixed_gemm_config",
    "default_noisy_quant_config",
    "get_tensor_hash_plus_stats_config",
    "grouped_fp8_gemm",
    "grouped_fp8_gemm_scale_shapes",
    "grouped_mixed_gemm",
    "mixed_gemm",
    "noise_lines",
    "noisy_quant",
    "noisy_quant_b",
    "pack_noise_factor",
    "pre_quant",
    "pre_quant_output_shapes",
    "supports_grouped_mixed_gemm",
    "supports_lottery_family",
    "tensor_hash",
    "tensor_hash_plus_stats",
    "tensor_hash_plus_stats_b",
    "tensor_hash_plus_stats_record_is_legal",
    "tensor_hash_scratchpad_bytes",
    "tensor_hash_workspace_bytes",
    "validate_mixed_gemm_config",
    "validate_noisy_quant_config",
]
