"""Per-layer GPU mining state: committed FP10 planes, steady per-job buffers,
the published job context, and the weight-keyed registry the torch op uses."""

import itertools
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import torch
from miner_base.block_submission import (
    PROTOCOL_COMMITMENT_LEAF,
    PrebuiltCommitment,
    commit_planes_for_leaf,
)
from miner_base.commitment import MiningConfiguration, PlanarCommitment
from miner_base.commitment_hash import noise_seed_b
from miner_base.prequant import PrequantMatrix
from miner_utils import get_logger
from pearl_gateway.comm.dataclasses import MiningJob

from .fp8_fallback import build_fp8_fallback
from .mining_config import PACKED_NOISE_K, RANK, SCALE_BLOCK, is_mineable_shape
from .tuning import device_config_name, tuned, with_committed_leaf

_LOGGER = get_logger(__name__)
_NEXT_LAYER_ID = itertools.count(1)


@dataclass
class LayerBuffers:
    """Per-layer GPU buffers for the current job's B-side operands.

    Every tensor is allocated once at load (before KV-cache profiling) and
    rewritten in place, on the mining stream, whenever the job changes
    (:func:`~vllm_miner.job_prep.prepare_layer`). Launches read these exact
    tensors, so their addresses never change.
    """

    b_prime: torch.Tensor  # (n, k) float8_e4m3fn
    # Columns [R, 2R) hold -(beta_b (.) E_B), the per-job half of B's peel. The
    # mid half [0, R) depends on the per-nonce F_A and is recomputed per launch
    # (pipeline._b_peel_for_launch); the kernel writes a zero placeholder here.
    b_peel: torch.Tensor  # (n, 2R) bfloat16
    alpha_b: torch.Tensor  # (n,) bfloat16
    beta_b: torch.Tensor  # (n,) bfloat16
    inv_alpha_b: torch.Tensor  # (n,) float32
    e2: torch.Tensor  # (n, R) float8_e4m3fn  E_B rows
    f1: torch.Tensor  # (R, k) float8_e4m3fn  all-zero F_A stand-in for noisy_quant_b's peel side
    f2: torch.Tensor  # (R, k) float8_e4m3fn  F_B
    f2_hl: torch.Tensor  # (k, PACKED_NOISE_K) int8 packed F_B
    noise_lines: torch.Tensor  # (k, R) float8_e4m3fn scratch
    root_codes: torch.Tensor  # (32,) uint8
    root_scales: torch.Tensor  # (32,) uint8
    tensor_hash_workspace: torch.Tensor  # uint8 Merkle scratch
    commit_stats: torch.Tensor  # (2 * n*k/512,) float32
    gram: torch.Tensor  # (R, R) float32 noisy_quant_b epilogue workspace
    commit_config: Any
    prepare_config: Any
    key_a_dev: torch.Tensor  # (32,) uint8 keyA (the header's A-side opening key)
    key_b_dev: torch.Tensor  # (32,) uint8 keyB (B's Merkle key)
    seed_b_dev: torch.Tensor  # (32,) uint8 noise seedB (finalize input, hit-record stamp)
    noise_key_b_dev: torch.Tensor  # (32,) uint8 Subkey("noise-line", seedB): E_B/F_B draw key
    threshold_dev: torch.Tensor  # (32,) uint8 little-endian lottery threshold


@dataclass
class BProofContext:
    """Lazily build and retain the B opening tree for one job's ``keyB``.

    ``commit_leaf`` is the Merkle leaf the GPU commit ran at (the resolved
    commit config's ``chunk_size``); the CPU rebuild must open the same
    tree, so it resolves through ``commit_planes_for_leaf``. ``p_b`` is the
    committed ``pB`` the GPU-derived ``seed_b`` was chained under; the
    rebuild re-derives the seed from the CPU tree to catch a GPU/CPU
    commitment disagreement before a proof is built on it.
    """

    key_b: bytes
    seed_b: bytes
    p_b: bytes
    codes: torch.Tensor
    scales: torch.Tensor
    commit_leaf: int = PROTOCOL_COMMITMENT_LEAF
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _commitment: PlanarCommitment | None = field(default=None, init=False, repr=False)

    def prebuilt_commitment(self) -> PrebuiltCommitment:
        with self._lock:
            if self._commitment is None:
                commitment = commit_planes_for_leaf(
                    [self.codes, self.scales], self.key_b, self.commit_leaf
                )
                derived = noise_seed_b(commitment.digest, self.key_b, self.p_b)
                if derived != self.seed_b:
                    raise RuntimeError("CPU B commitment disagrees with GPU plane roots")
                self._commitment = commitment
            return PrebuiltCommitment(self._commitment, self.key_b)


@dataclass(frozen=True)
class JobContext:
    """The host-side description of the job one layer's B operands were
    prepared for.

    The device tensors alias the layer's steady :class:`LayerBuffers`; the
    bytes/ints are what the winner consumer needs to validate a hit against
    the launch's job (``winners.py``). Published under ``LayerState.lock``
    by :func:`~vllm_miner.job_prep.prepare_layer`, which rewrites the
    operands on the mining stream before publishing, so a launch enqueued
    after publication on that stream reads exactly this job's operands.
    """

    job: MiningJob
    config: MiningConfiguration
    key_a: bytes
    key_b: bytes
    seed_b: bytes
    target: int
    key_a_dev: torch.Tensor
    seed_b_dev: torch.Tensor
    threshold_dev: torch.Tensor
    f2: torch.Tensor
    e2: torch.Tensor
    beta_b: torch.Tensor
    b_prime: torch.Tensor
    b_peel: torch.Tensor
    alpha_b: torch.Tensor
    inv_alpha_b: torch.Tensor
    b_proof: BProofContext


@dataclass
class LayerState:
    """Everything one mining-enabled linear layer carries across forwards."""

    layer_name: str
    layer_id: int
    weight: torch.Tensor  # (n, k) int8, contiguous, CUDA
    weight_scale: torch.Tensor  # (n, k/8) bfloat16, contiguous, CUDA
    # Stable host planes for reference winner checks and lazy proof commitments.
    weight_cpu: torch.Tensor
    weight_scale_cpu: torch.Tensor
    n: int
    k: int
    mineable: bool
    buffers: LayerBuffers | None
    w_fp8: torch.Tensor  # (n, k) float8_e4m3fn fallback operand
    w_fp8_scale: torch.Tensor  # (1, n) float32
    # The job the steady buffers currently hold, or None before the first
    # preparation / after unpublication. Read and replaced under ``lock``.
    job_ctx: JobContext | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    mining_error_logged: bool = False
    disabled_reason: str | None = None

    def disable_mining(self, reason: str) -> None:
        """Permanently isolate one deterministic layer failure from serving.

        The layer serves the FP8 fallback from here on; crediting and hit
        validation stop immediately.
        """
        with self.lock:
            first_failure = self.disabled_reason is None
            if first_failure:
                self.disabled_reason = reason
            self.mineable = False
            self.job_ctx = None
        if first_failure:
            _LOGGER.error(f"mining disabled for {self.layer_name}: {reason}")


def _is_sm100(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device)[0] == 10


def _kernels_available() -> bool:
    try:
        import pearl_gemm  # noqa: F401
    except Exception:
        _LOGGER.opt(exception=True).warning("pearl_gemm import failed; layer will not mine")
        return False
    return True


@lru_cache(maxsize=256)
def _b_configs(config_name: str, n: int, k: int):
    from pearl_gemm import (
        NoisyQuantBConfig,
        TensorHashConfig,
        tensor_hash_plus_stats_record_is_legal,
        validate_noisy_quant_config,
    )

    commit = TensorHashConfig(
        **with_committed_leaf(
            tuned(
                config_name,
                "tensor_hash_plus_stats",
                legal=tensor_hash_plus_stats_record_is_legal,
                m=n,
                k=k,
            ),
            PROTOCOL_COMMITMENT_LEAF,
        )
    )
    prepare = NoisyQuantBConfig(**tuned(config_name, "noisy_quant", m=n, k=k))
    validate_noisy_quant_config(n, k, prepare)
    return commit, prepare


def buffer_field_specs(
    n: int, k: int, device: torch.device
) -> tuple[dict[str, tuple[tuple[int, ...], torch.dtype]], Any, Any]:
    """(tensor-field geometry, commit config, prepare config) for one shape.

    The single source of truth for LayerBuffers' tensor fields:
    :func:`_allocate_buffers` materializes exactly these tensors, and
    ``memory._graph_scratch_bytes`` sizes the graph-publication scratch
    reservation from the same table, so a future field addition cannot
    silently under-reserve.
    """
    from pearl_gemm import tensor_hash_workspace_bytes

    commit_config, prepare_config = _b_configs(device_config_name(device), n, k)
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        "b_prime": ((n, k), torch.float8_e4m3fn),
        "b_peel": ((n, 2 * RANK), torch.bfloat16),
        "alpha_b": ((n,), torch.bfloat16),
        "beta_b": ((n,), torch.bfloat16),
        "inv_alpha_b": ((n,), torch.float32),
        "e2": ((n, RANK), torch.float8_e4m3fn),
        "f1": ((RANK, k), torch.float8_e4m3fn),
        "f2": ((RANK, k), torch.float8_e4m3fn),
        "f2_hl": ((k, PACKED_NOISE_K), torch.int8),
        "noise_lines": ((k, RANK), torch.float8_e4m3fn),
        "root_codes": ((32,), torch.uint8),
        "root_scales": ((32,), torch.uint8),
        "tensor_hash_workspace": (
            (tensor_hash_workspace_bytes(n, k, commit_config),),
            torch.uint8,
        ),
        "commit_stats": ((2 * (n * k // 512),), torch.float32),
        "gram": ((RANK, RANK), torch.float32),
        "key_a_dev": ((32,), torch.uint8),
        "key_b_dev": ((32,), torch.uint8),
        "seed_b_dev": ((32,), torch.uint8),
        "noise_key_b_dev": ((32,), torch.uint8),
        "threshold_dev": ((32,), torch.uint8),
    }
    return specs, commit_config, prepare_config


def _allocate_buffers(n: int, k: int, device: torch.device) -> LayerBuffers:
    specs, commit_config, prepare_config = buffer_field_specs(n, k, device)
    tensors = {
        name: torch.zeros(shape, dtype=dtype, device=device)
        for name, (shape, dtype) in specs.items()
    }
    tensors["alpha_b"].fill_(1)
    tensors["inv_alpha_b"].fill_(1)
    return LayerBuffers(commit_config=commit_config, prepare_config=prepare_config, **tensors)


def supports_layer_shape(n: int, k: int) -> bool:
    """Whether a (n, k) linear can carry FP10 planes and the FP8 fallback."""
    return k % SCALE_BLOCK == 0 and n % 16 == 0 and k % 16 == 0


def can_mine_layer(n: int, k: int, device: torch.device) -> bool:
    """Whether this device/shape runs the mined pipeline. Layers that cannot
    mine must stay on their original unquantized path: encoding them would
    trade serving quality for zero protocol value."""
    return (
        supports_layer_shape(n, k)
        and is_mineable_shape(n, k)
        and _is_sm100(device)
        and _kernels_available()
    )


@dataclass(frozen=True)
class _StaticLayerEncoding:
    weight: torch.Tensor
    weight_scale: torch.Tensor
    weight_cpu: torch.Tensor
    weight_scale_cpu: torch.Tensor
    w_fp8: torch.Tensor
    w_fp8_scale: torch.Tensor


def _validate_bf16_weight(bf16_weight: torch.Tensor) -> tuple[int, int]:
    if bf16_weight.dtype != torch.bfloat16 or bf16_weight.dim() != 2 or not bf16_weight.is_cuda:
        raise ValueError(
            f"expected a 2D CUDA bfloat16 weight, got {bf16_weight.dtype} "
            f"{tuple(bf16_weight.shape)} on {bf16_weight.device}"
        )
    n, k = bf16_weight.shape
    if not supports_layer_shape(n, k):
        raise ValueError(f"unsupported mining shape n={n}, k={k}")
    return n, k


def _encode_static_weight(bf16_weight: torch.Tensor) -> _StaticLayerEncoding:
    pq = PrequantMatrix.encode(bf16_weight)
    weight = pq.int_values.contiguous()
    weight_scale = pq.scales.contiguous()
    w_fp8, w_fp8_scale = build_fp8_fallback(weight, weight_scale)
    return _StaticLayerEncoding(
        weight=weight,
        weight_scale=weight_scale,
        weight_cpu=weight.cpu(),
        weight_scale_cpu=weight_scale.cpu(),
        w_fp8=w_fp8,
        w_fp8_scale=w_fp8_scale,
    )


def create_layer_state(layer_name: str, bf16_weight: torch.Tensor) -> LayerState:
    """Encode a loaded BF16 weight and build one graph-stable layer state."""
    n, k = _validate_bf16_weight(bf16_weight)
    encoded = _encode_static_weight(bf16_weight)
    # Callers gate on can_mine_layer, so reaching here without it is a bug.
    mineable = can_mine_layer(n, k, encoded.weight.device)
    if not mineable:
        raise RuntimeError(f"layer {layer_name} (n={n}, k={k}) is not mineable on this device")
    return LayerState(
        layer_name=layer_name,
        layer_id=next(_NEXT_LAYER_ID),
        weight=encoded.weight,
        weight_scale=encoded.weight_scale,
        weight_cpu=encoded.weight_cpu,
        weight_scale_cpu=encoded.weight_scale_cpu,
        n=n,
        k=k,
        mineable=mineable,
        buffers=_allocate_buffers(n, k, encoded.weight.device),
        w_fp8=encoded.w_fp8,
        w_fp8_scale=encoded.w_fp8_scale,
    )


def refresh_layer_state(state: LayerState, bf16_weight: torch.Tensor) -> None:
    """Re-encode a checkpoint weight into an inactive graph-stable state.

    Every tensor reachable from captured serving graphs is updated in place.
    The layer's job context must already be unpublished and mining GPU work
    drained by the framework lifecycle coordinator.
    """
    n, k = _validate_bf16_weight(bf16_weight)
    if (n, k) != (state.n, state.k) or bf16_weight.device != state.weight.device:
        raise ValueError(
            f"reload changed {state.layer_name} from {(state.n, state.k)} on "
            f"{state.weight.device} to {(n, k)} on {bf16_weight.device}"
        )
    with state.lock:
        if state.job_ctx is not None:
            raise RuntimeError(f"cannot refresh active mining layer {state.layer_name}")
    encoded = _encode_static_weight(bf16_weight)
    with torch.no_grad():
        state.weight.copy_(encoded.weight)
        state.weight_scale.copy_(encoded.weight_scale)
        state.w_fp8.copy_(encoded.w_fp8)
        state.w_fp8_scale.copy_(encoded.w_fp8_scale)
    with state.lock:
        if state.job_ctx is not None:
            raise RuntimeError(f"mining layer {state.layer_name} became active during reload")
        state.weight_cpu = encoded.weight_cpu
        state.weight_scale_cpu = encoded.weight_scale_cpu
        state.mineable = True
        state.disabled_reason = None
        state.mining_error_logged = False


_STATE_REGISTRY: dict[int, LayerState] = {}
_LAYER_ID_REGISTRY: dict[int, LayerState] = {}
_REGISTRY_LOCK = threading.RLock()


def register_state(state: LayerState) -> None:
    if not state.weight.is_contiguous():
        raise RuntimeError("mining weight must be contiguous for data_ptr lookup")
    if not 0 < state.layer_id < 2**32:
        raise RuntimeError(f"mining layer id must fit uint32, got {state.layer_id}")
    pointer = state.weight.data_ptr()
    with _REGISTRY_LOCK:
        pointer_owner = _STATE_REGISTRY.get(pointer)
        id_owner = _LAYER_ID_REGISTRY.get(state.layer_id)
        if pointer_owner is not None and pointer_owner is not state:
            raise RuntimeError(f"mining weight pointer {pointer} is already registered")
        if id_owner is not None and id_owner is not state:
            raise RuntimeError(f"mining layer id {state.layer_id} is already registered")
        _STATE_REGISTRY[pointer] = state
        _LAYER_ID_REGISTRY[state.layer_id] = state


def unregister_state(state: LayerState) -> None:
    with _REGISTRY_LOCK:
        pointer = state.weight.data_ptr()
        if _STATE_REGISTRY.get(pointer) is state:
            del _STATE_REGISTRY[pointer]
        if _LAYER_ID_REGISTRY.get(state.layer_id) is state:
            del _LAYER_ID_REGISTRY[state.layer_id]


def lookup_state(weight: torch.Tensor) -> LayerState:
    with _REGISTRY_LOCK:
        state = _STATE_REGISTRY.get(weight.data_ptr())
    if state is None:
        raise RuntimeError(
            "mining state for weight tensor was not registered; "
            "process_weights_after_loading() must run before execution"
        )
    return state


def lookup_state_by_layer_id(layer_id: int) -> LayerState:
    with _REGISTRY_LOCK:
        state = _LAYER_ID_REGISTRY.get(layer_id)
    if state is None:
        raise RuntimeError(f"unknown mining layer id {layer_id}")
    return state


def all_states() -> list[LayerState]:
    with _REGISTRY_LOCK:
        return list(_STATE_REGISTRY.values())


def clear_state_registry() -> None:
    """Drop dispatch indexes only after terminal GPU-runtime quiescence."""
    with _REGISTRY_LOCK:
        _STATE_REGISTRY.clear()
        _LAYER_ID_REGISTRY.clear()
