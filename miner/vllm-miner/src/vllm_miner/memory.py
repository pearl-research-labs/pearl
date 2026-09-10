"""Conservative post-profile GPU memory reservation for mining work."""

import math

from .config import config as gpu_config
from .mining_config import max_mineable_k
from .settings import runtime_settings
from .state import all_states

_MIB = 1024**2
_RESERVE_ALIGNMENT = 256 * _MIB
_FRAGMENTATION_FACTOR = 1.10
# Layerwise checkpoint reload materializes one BF16 layer and re-encodes FP10
# through FP32 eager intermediates before building a chunked FP8 copy. Sixteen
# bytes per source element conservatively covers that peak alongside the old
# graph-visible encoding; the global fragmentation factor is applied again.
_RELOAD_TRANSIENT_BYTES_PER_ELEMENT = 16


def _elements(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def _launch_bytes(m: int, n: int, k: int, device) -> int:
    from pearl_gemm import (
        R,
        pre_quant_output_shapes,
        tensor_hash_workspace_bytes,
    )

    from .pipeline import _configs
    from .tuning import device_config_name

    _, commit, _, _ = _configs(device_config_name(device), m, n, k)
    codes_shape, scales_shape = pre_quant_output_shapes(m, k)

    return sum(
        (
            2 * m * k,  # padded BF16 A
            _elements(codes_shape),  # int8 codes
            2 * _elements(scales_shape),  # BF16 scales
            32 + 32,  # roots
            tensor_hash_workspace_bytes(m, k, commit),
            96,  # A keys (seedA || noise-line keyA || jackpot key)
            4 * 2 * (m * k // 512),  # hash statistics
            k * R,  # FP8 F_A lines, drawn per launch under this A's key
            2 * m + 2 * m,  # alpha/beta A
            m * R,  # FP8 E1
            m * k,  # FP8 A prime
            2 * m * 2 * R,  # BF16 A peel
            # The per-launch B peel and b_peel_for_a's intermediates (F_A
            # transposed in FP8; F32 B'F_A^T, E_B, E_B@gram, the scaled
            # product and mid -- five (n, R); gram (R, R); F32 F_B and F_A;
            # F32 beta_b), all bounded as if simultaneously live.
            2 * n * 2 * R,  # BF16 B peel
            k * R + 4 * (5 * n * R + R * R + 2 * R * k + n),
            2 * m * n,  # BF16 output
        )
    )


def _raw_peak_mining_bytes() -> int:
    """The pre-margin, pre-alignment reservation total (see the public wrapper)."""
    settings = runtime_settings()
    states = [state for state in all_states() if state.mineable]
    if gpu_config.settings.no_mining or not states:
        return 0

    # HitSignal is deliberately sized for the complete protocol-legal K domain
    # on first use, not only the layers registered before memory profiling.
    signal_k = max_mineable_k()
    max_m = max(settings.m_buckets)
    signal_payload = max_m * signal_k + 2 * max_m * (signal_k // 8)
    signal_bytes = 4 + signal_payload
    owned_hit_bytes = signal_payload

    max_launch = max(
        _launch_bytes(m, state.n, state.k, state.weight.device)
        for state in states
        for m in settings.m_buckets
    )
    # One completion lease is acquired before every launch allocation, so the
    # process-wide completion bound is the launch population. Startup warmup
    # runs on the serving thread before any launch, so it needs no slot of its
    # own.
    launch_slots = settings.completion_inflight_limit
    reload_refresh = max(
        _RELOAD_TRANSIENT_BYTES_PER_ELEMENT * state.n * state.k for state in states
    )
    return signal_bytes + owned_hit_bytes + launch_slots * max_launch + reload_refresh


def estimate_peak_mining_bytes() -> int:
    """Bytes vLLM must not assign to KV cache after profiling.

    Layer-static FP10/FP8/B-operand storage is already resident when vLLM
    profiles free memory. This estimate covers allocations that appear later:
    the persistent hit signal, a possible owned hit payload, the maximum
    number of overlapping A-side launch workspaces, and one layerwise
    checkpoint-reload re-encoding transient.
    """
    raw = _raw_peak_mining_bytes()
    if raw == 0:
        return 0
    with_margin = math.ceil(raw * _FRAGMENTATION_FACTOR) + _RESERVE_ALIGNMENT
    return math.ceil(with_margin / _RESERVE_ALIGNMENT) * _RESERVE_ALIGNMENT
