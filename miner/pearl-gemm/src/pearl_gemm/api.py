"""Caller-owned functional API for the new-scheme operations."""

# FP16 (A100 / sm_80) scheme primitives, exposed as submodule namespaces so the
# bit-exact PoUW datapath is reachable as `pearl_gemm.fp16_*` without colliding
# with the FP8-family flat exports below (e.g. both schemes have a `noise_lines`).
from . import (
    fp16_commit,
    fp16_gemm,
    fp16_miner,
    fp16_noise_lines,
    fp16_noisy_quant,
    fp16_pipeline,
    fp16_policy,
    fp16_search,
)
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
    "fp16_commit",
    "fp16_gemm",
    "fp16_miner",
    "fp16_noise_lines",
    "fp16_noisy_quant",
    "fp16_pipeline",
    "fp16_policy",
    "fp16_search",
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
    "NoiseLoadMode",
    "NoisyQuantBConfig",
    "NoisyQuantConfig",
    "PreQuantConfig",
    "TensorHashConfig",
    "get_tensor_hash_plus_stats_config",
    "default_mixed_gemm_config",
    "grouped_fp8_gemm",
    "grouped_fp8_gemm_scale_shapes",
    "grouped_mixed_gemm",
    "supports_grouped_mixed_gemm",
    "mixed_gemm",
    "supports_lottery_family",
    "validate_mixed_gemm_config",
    "noise_lines",
    "default_noisy_quant_config",
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
