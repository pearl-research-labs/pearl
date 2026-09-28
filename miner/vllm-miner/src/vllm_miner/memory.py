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


def _moe_launch_bytes(m: int, k: int, experts: int, top_k: int) -> int:
    """The MoE launch's extra per-launch buffers (``pipeline._grouped_mixed_gemm``
    and ``moe.route_tokens``): the noised A-side operands gathered into expert
    order over ``cum_m = m * top_k`` rows (the committed planes are not
    gathered: MoE hits publish payload-less), and the routing table with its
    two commitments."""
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
            8 * cum_m + 8 * cum_m,  # sorted expert ids and slots (int64)
            4 * cum_m,  # tokens (Rflat)
            4 * (experts + 1) + 4 * experts + 64,  # m_indptr, m_valid, HR || HO
            committed(cum_m) + committed(experts),  # the two routing commitments
        )
    )


def _moe_tail_bytes(m: int, n: int, k: int, experts: int, top_k: int) -> int:
    """The framework adapter's tail after a mined MoE launch returns
    (``vllm_pearl_config.PearlMoEMethod._mined_tail``): the activation
    ``(cum_m, n_e / 2)``, the ``w2`` grouped GEMM output ``(cum_m, k)``, the
    router weights gathered to expert order, the weighted rows ``(cum_m, k)``
    (``moe.combine_routed_rows``: the product is materialized while its
    slot-order destination is already allocated), that slot-order copy
    ``(cum_m, k)`` and the reduced ``(m, k)`` output, all BF16 and bounded as
    if simultaneously live. It runs on the serving thread after the launch's
    own buffers may still be in flight, so it is counted once on top of the
    launch slots."""
    cum_m = m * top_k
    n_e = n // experts
    return sum(
        (
            2 * cum_m * (n_e // 2),  # activation output
            2 * cum_m * k,  # w2 grouped GEMM output
            4 * cum_m + 2 * cum_m,  # gathered router weights (source dtype, then BF16)
            2 * cum_m * k,  # weighted rows (rows * weights) in expert order
            2 * cum_m * k,  # weighted rows scattered to slot order
            2 * m * k,  # reduced output
        )
    )


def _launch_bytes(m: int, n: int, k: int, device, experts: int = 0, top_k: int = 0) -> int:
    from pearl_gemm import (
        R,
        pre_quant_output_shapes,
        tensor_hash_workspace_bytes,
    )

    from .mining_config import expert_n
    from .pipeline import _configs
    from .tuning import device_config_name

    # The pipeline resolves its configuration against the lottery width: the
    # stacked ``n`` dense, one expert's ``n_e`` for MoE (pipeline._launch_stages).
    lottery_n = expert_n(n, experts)
    _, commit, _, _ = _configs(device_config_name(device), m, lottery_n, k)
    codes_shape, scales_shape = pre_quant_output_shapes(m, k)

    # MoE layers gather the A side into expert order over ``m * top_k`` rows
    # on top of the dense planes below, and their output is the permuted
    # ``(m * top_k, n_e)`` expert output rather than a dense ``(m, n)``.
    moe = _moe_launch_bytes(m, k, experts, top_k) if experts else 0
    output_rows = m * top_k if experts else m

    return moe + sum(
        (
            2 * m * k,  # BF16 A
            _elements(codes_shape),  # int8 codes
            2 * _elements(scales_shape),  # BF16 scales
            32 + 32,  # roots
            tensor_hash_workspace_bytes(m, k, commit),
            96,  # A keys (seedA || noise-line keyA || jackpot key)
            4 * 2 * (m * k // 512),  # hash statistics
            2 * m + 2 * m,  # alpha/beta A
            m * R,  # FP8 E1
            m * k,  # FP8 A prime
            2 * m * 2 * R,  # BF16 A peel
            2 * output_rows * lottery_n,  # BF16 output
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
        _launch_bytes(m, state.n, state.k, state.weight.device, state.experts, state.top_k)
        for state in states
        for m in settings.m_buckets
    )
    # One completion lease is acquired before every launch allocation, so the
    # process-wide completion bound is the launch population. Startup warmup
    # runs on the serving thread before any launch, so it needs no slot of its
    # own.
    launch_slots = settings.completion_inflight_limit
    # The serving thread finishes one mined MoE forward at a time, so its
    # tail is one extra live set, not one per launch slot.
    moe_tail = max(
        (
            _moe_tail_bytes(max_m, state.n, state.k, state.experts, state.top_k)
            for state in states
            if state.experts
        ),
        default=0,
    )
    reload_refresh = max(
        _RELOAD_TRANSIENT_BYTES_PER_ELEMENT * state.n * state.k for state in states
    )
    return signal_bytes + owned_hit_bytes + launch_slots * max_launch + moe_tail + reload_refresh


def estimate_peak_mining_bytes() -> int:
    """Bytes vLLM must not assign to KV cache after profiling.

    Layer-static FP10/FP8/B-operand storage is already resident when vLLM
    profiles free memory. This estimate covers allocations that appear later:
    the persistent hit signal, a possible owned hit payload, the maximum
    number of overlapping A-side launch workspaces, one mined MoE forward's
    activation/down/combine tail, and one layerwise checkpoint-reload
    re-encoding transient.
    """
    raw = _raw_peak_mining_bytes()
    if raw == 0:
        return 0
    with_margin = math.ceil(raw * _FRAGMENTATION_FACTOR) + _RESERVE_ALIGNMENT
    return math.ceil(with_margin / _RESERVE_ALIGNMENT) * _RESERVE_ALIGNMENT
