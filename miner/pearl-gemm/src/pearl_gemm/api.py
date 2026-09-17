"""Caller-owned functional API for the new-scheme operations."""

from .mixed_gemm import (
    MixedGemmConfig,
    mixed_gemm,
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
    noisy_quant,
    pack_noise_factor,
    validate_noisy_quant_config,
)
from .noisy_quant_b import (
    NoisyQuantBConfig,
    b_peel_for_a,
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
    "NoiseLoadMode",
    "NoisyQuantBConfig",
    "NoisyQuantConfig",
    "PreQuantConfig",
    "TensorHashConfig",
    "b_peel_for_a",
    "get_tensor_hash_plus_stats_config",
    "mixed_gemm",
    "validate_mixed_gemm_config",
    "noise_lines",
    "noisy_quant",
    "noisy_quant_b",
    "validate_noisy_quant_config",
    "pack_noise_factor",
    "pre_quant",
    "pre_quant_output_shapes",
    "tensor_hash",
    "tensor_hash_plus_stats",
    "tensor_hash_plus_stats_b",
    "tensor_hash_plus_stats_record_is_legal",
    "tensor_hash_scratchpad_bytes",
    "tensor_hash_workspace_bytes",
]
