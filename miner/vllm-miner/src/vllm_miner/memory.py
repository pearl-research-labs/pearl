"""Conservative post-profile GPU memory reservation for bounded mining work."""

import math

from .mining_config import expert_n, max_mineable_k
from .settings import miner_settings, runtime_settings
from .state import all_states, device_config_name

_MIB = 1024**2
_RESERVE_ALIGNMENT = 256 * _MIB
_FRAGMENTATION_FACTOR = 1.10


def _moe_launch_bytes(m: int, k: int, experts: int, top_k: int) -> int:
    from pearl_gemm import R, TensorHashConfig, tensor_hash_scratchpad_bytes

    cum_m = m * top_k
    chunk = TensorHashConfig().chunk_size

    def committed(values: int) -> int:
        payload = -(-4 * values // chunk) * chunk
        return payload + tensor_hash_scratchpad_bytes(payload, TensorHashConfig())

    return sum(
        (
            cum_m * k,  # gathered FP8 A prime
            2 * cum_m * 2 * R,  # gathered BF16 A peel
            2 * cum_m,  # gathered BF16 alpha A
            16 * cum_m,  # sorted expert ids and slots (int64)
            4 * cum_m,  # tokens (Rflat)
            4 * (experts + 1) + 4 * experts + 64,  # m_indptr, m_valid, HR || HO
            committed(cum_m) + committed(experts),
        )
    )


def _moe_tail_bytes(m: int, n: int, k: int, experts: int, top_k: int) -> int:
    cum_m = m * top_k
    n_e = n // experts
    return sum(
        (
            2 * cum_m * (n_e // 2),  # activation output
            2 * cum_m * k,  # w2 grouped GEMM output
            6 * cum_m,  # gathered router weights (source + BF16)
            4 * cum_m * k,  # weighted rows + scattered slot-order copy
            2 * m * k,  # reduced output
        )
    )


def _launch_bytes(m: int, n: int, k: int, device, experts: int = 0, top_k: int = 0) -> int:
    from miner_base.protocol import local_device
    from pearl_gemm import R, pre_quant_output_shapes, tensor_hash_workspace_bytes

    from .pipeline import _configs

    lottery_n = expert_n(n, experts)
    _, commit, _, _ = _configs(
        device_config_name(device), m, lottery_n, k, device=local_device(device)
    )
    codes_shape, scales_shape = pre_quant_output_shapes(m, k)
    moe = _moe_launch_bytes(m, k, experts, top_k) if experts else 0
    output_rows = m * top_k if experts else m

    return moe + sum(
        (
            2 * m * k,  # BF16 A
            math.prod(codes_shape),  # int8 codes
            2 * math.prod(scales_shape),  # BF16 scales
            64,  # roots
            tensor_hash_workspace_bytes(m, k, commit),
            96,  # A keys
            4 * 2 * (m * k // 512),  # hash statistics
            4 * m,  # alpha/beta A
            m * R,  # FP8 E1
            m * k,  # FP8 A prime
            2 * m * 2 * R,  # BF16 A peel
            2 * output_rows * lottery_n,  # BF16 output
        )
    )


def _raw_peak_mining_bytes() -> int:
    settings = runtime_settings()
    states = [state for state in all_states() if state.mineable]
    if miner_settings().no_mining or not states:
        return 0

    signal_k = max_mineable_k()
    max_m = max(settings.m_buckets)
    signal_payload = max_m * signal_k + 2 * max_m * (signal_k // 8)
    signal_bytes = 4 + signal_payload
    owned_hit_bytes = signal_payload

    max_launch = max(
        _launch_bytes(m, state.n, state.k, state.weight.device, state.experts, state.top_k)
        for state in states
        for m in settings.m_buckets
    )
    moe_tail = max(
        (
            _moe_tail_bytes(max_m, state.n, state.k, state.experts, state.top_k)
            for state in states
            if state.experts
        ),
        default=0,
    )
    return (
        signal_bytes + owned_hit_bytes + settings.completion_inflight_limit * max_launch + moe_tail
    )


def estimate_peak_mining_bytes() -> int:
    """Bytes vLLM must keep off the KV cache for hit signal + overlapping launch workspaces."""
    raw = _raw_peak_mining_bytes()
    if raw == 0:
        return 0
    with_margin = math.ceil(raw * _FRAGMENTATION_FACTOR) + _RESERVE_ALIGNMENT
    return math.ceil(with_margin / _RESERVE_ALIGNMENT) * _RESERVE_ALIGNMENT
