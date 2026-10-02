"""``noisy_quant``: stats combine + E1 gen + noising + peel, one fused kernel.

``_quantization_ops.py`` holds quantization device helpers,
``_kernel_common.py`` the architecture-neutral kernel phases,
``_kernel_register_peel.py`` the register-peel topology SM90 and SM120 share,
``_kernel.py`` / ``_kernel_sm90.py`` / ``_kernel_sm120.py`` the per-family
fused device kernels, and ``_host.py`` the functional caller-owned launch.
"""

from ._host import (
    NoiseLoadMode,
    NoisyQuantConfig,
    default_noisy_quant_config,
    noisy_quant,
    pack_noise_factor,
    validate_noisy_quant_config,
)

__all__ = [
    "NoiseLoadMode",
    "NoisyQuantConfig",
    "default_noisy_quant_config",
    "noisy_quant",
    "pack_noise_factor",
    "validate_noisy_quant_config",
]
