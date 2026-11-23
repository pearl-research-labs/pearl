"""Per-layer GPU mining state, FP8 serving fallback, autotune lookup, and dispatch registry."""

import itertools
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import cache, lru_cache
from typing import Any

import torch
from miner_base.block_submission import PrebuiltCommitment
from miner_base.commitment import PlanarCommitment, commit_planes
from miner_base.prequant import PrequantMatrix
from miner_base.protocol import (
    COMMITMENT_CHUNK_SIZE,
    Device,
    MiningConfiguration,
    local_device,
    noise_seed_b,
)
from miner_utils import get_logger
from pearl_gateway.comm.dataclasses import MiningJob

from .capture import cuda_device_index
from .cpu_b import shared_copy
from .mining_config import (
    MOE_LOTTERY_N,
    PACKED_NOISE_K,
    RANK,
    SCALE_BLOCK,
    expert_n,
    is_mineable_shape,
    select_tile,
)
from .moe import validate_moe_dims

_LOGGER = get_logger(__name__)
_NEXT_LAYER_ID = itertools.count(1)
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
_BUILD_CHUNK_ROWS = 4096


# ---------------------------------------------------------------------------
# Autotune config helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _device_config_name(index: int) -> str:
    return torch.cuda.get_device_name(index).lower().replace(" ", "_").replace("-", "_")


def device_config_name(device: torch.device) -> str:
    return _device_config_name(cuda_device_index(device))


@cache
def _load_autotune_config(name: str) -> dict:
    from pearl_gemm.autotune import load_config

    return load_config(name) or {}


def tuned(
    config_name: str,
    kernel: str,
    *,
    legal: Callable[[dict], bool] | None = None,
    require_legal: bool = False,
    exact: bool = False,
    **shape: int,
) -> dict:
    from pearl_gemm.autotune import get_tuned

    return get_tuned(
        kernel,
        _load_autotune_config(config_name),
        legal=legal,
        require_legal=require_legal,
        exact=exact,
        **shape,
    )


def with_committed_leaf(kwargs: dict, chunk_size: int) -> dict:
    """Keep autotune launch knobs while pinning ``chunk_size`` to the committed leaf."""
    from pearl_gemm.tensor_hash_plus_stats._merkle_host import SUPPORTED_THREAD_LOAD_SIZES

    pinned = dict(kwargs)
    pinned["chunk_size"] = chunk_size
    load = pinned.get("thread_load_size", 128)
    if chunk_size % load:
        pinned["thread_load_size"] = max(
            size for size in SUPPORTED_THREAD_LOAD_SIZES if chunk_size % size == 0
        )
    return pinned


# ---------------------------------------------------------------------------
# Job-independent W8A8-FP8 fallback GEMM (used by non-mined forwards & graphs)
# ---------------------------------------------------------------------------


def build_fp8_fallback(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row-scaled FP8 copy of the opened FP10 weight for ``torch._scaled_mm``."""
    n, k = weight.shape
    w_fp8 = torch.empty(n, k, dtype=torch.float8_e4m3fn, device=weight.device)
    w_scale = torch.empty(1, n, dtype=torch.float32, device=weight.device)
    for start in range(0, n, _BUILD_CHUNK_ROWS):
        end = min(start + _BUILD_CHUNK_ROWS, n)
        rows = end - start
        blocks = weight[start:end].to(torch.float32).reshape(rows, -1, SCALE_BLOCK)
        opened = (blocks * weight_scale[start:end].to(torch.float32).unsqueeze(-1)).reshape(rows, k)
        scale = opened.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / _FP8_MAX
        w_fp8[start:end] = (opened / scale).to(torch.float8_e4m3fn)
        w_scale[0, start:end] = scale.squeeze(1)
    return w_fp8, w_scale


@cache
def _resolve_vllm_fp8_quant():
    try:
        from vllm import _custom_ops as ops

        return ops.scaled_fp8_quant
    except Exception:
        return None


def fp8_fallback_gemm(
    x2d: torch.Tensor,
    w_fp8: torch.Tensor,
    w_fp8_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Dynamic per-token FP8 quant + rowwise ``torch._scaled_mm``."""
    vllm_quant = _resolve_vllm_fp8_quant()
    if vllm_quant is not None:
        x_fp8, x_scale = vllm_quant(x2d, use_per_token_if_dynamic=True)
    else:
        x_f32 = x2d.to(torch.float32)
        x_scale = x_f32.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / _FP8_MAX
        x_fp8 = (x_f32 / x_scale).to(torch.float8_e4m3fn)
    return torch._scaled_mm(
        x_fp8,
        w_fp8.t(),
        scale_a=x_scale,
        scale_b=w_fp8_scale,
        bias=bias,
        out_dtype=out_dtype,
    )


# ---------------------------------------------------------------------------
# Per-layer steady buffers, job context, and state registry
# ---------------------------------------------------------------------------


@dataclass
class LayerBuffers:
    """Per-layer GPU buffers allocated once at load and updated in place per job."""

    b_prime: torch.Tensor  # (n, k) float8_e4m3fn
    b_peel: torch.Tensor  # (n, 2R) bfloat16
    alpha_b: torch.Tensor  # (n,) bfloat16
    beta_b: torch.Tensor  # (n,) bfloat16
    inv_alpha_b: torch.Tensor  # (n,) float32
    e2: torch.Tensor  # (n, R) float8_e4m3fn
    f1: torch.Tensor  # (R, k) float8_e4m3fn
    f1_hl: torch.Tensor  # (k, PACKED_NOISE_K) int8
    f2: torch.Tensor  # (R, k) float8_e4m3fn
    f2_hl: torch.Tensor  # (k, PACKED_NOISE_K) int8
    noise_lines: torch.Tensor  # (k, R) float8_e4m3fn
    root_codes: torch.Tensor  # (32,) uint8
    root_scales: torch.Tensor  # (32,) uint8
    tensor_hash_workspace: torch.Tensor  # uint8
    commit_stats: torch.Tensor  # (2 * n*k/512,) float32
    gram: torch.Tensor  # (R, R) float32
    commit_config: Any
    prepare_config: Any
    key_a_dev: torch.Tensor  # (32,) uint8
    key_b_dev: torch.Tensor  # (32,) uint8
    seed_b_dev: torch.Tensor  # (32,) uint8
    noise_key_b_dev: torch.Tensor  # (32,) uint8
    threshold_dev: torch.Tensor  # (32,) uint8


@dataclass
class BProofContext:
    """Lazy B-side CPU Merkle tree verified against GPU/CPU ``seed_b``."""

    key_b: bytes
    ancestor_header: bytes
    seed_b: bytes
    p_b: bytes
    codes: torch.Tensor
    scales: torch.Tensor
    commit_leaf: int = COMMITMENT_CHUNK_SIZE
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _commitment: PlanarCommitment | None = field(default=None, init=False, repr=False)

    def prebuilt_commitment(self) -> PrebuiltCommitment:
        with self._lock:
            if self._commitment is None:
                commitment = commit_planes([self.codes, self.scales], self.key_b, self.commit_leaf)
                if noise_seed_b(commitment.digest, self.key_b, self.p_b) != self.seed_b:
                    raise RuntimeError("CPU B commitment disagrees with GPU plane roots")
                self._commitment = commitment
            return PrebuiltCommitment(self._commitment, self.key_b)


@dataclass(frozen=True)
class JobContext:
    """Host-side metadata of the job currently loaded into ``LayerState.buffers``."""

    job: MiningJob
    config: MiningConfiguration
    key_a: bytes
    key_b: bytes
    seed_b: bytes
    target: int
    b_proof: BProofContext


@dataclass
class LayerState:
    """Everything one mining-enabled linear or MoE layer carries across forwards."""

    layer_name: str
    layer_id: int
    weight: torch.Tensor  # (n, k) int8, contiguous, CUDA
    weight_scale: torch.Tensor  # (n, k/8) bfloat16, contiguous, CUDA
    weight_cpu: torch.Tensor
    weight_scale_cpu: torch.Tensor
    n: int
    k: int
    mineable: bool
    buffers: LayerBuffers | None
    w_fp8: torch.Tensor | None
    w_fp8_scale: torch.Tensor | None
    job_ctx: JobContext | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    disabled_reason: str | None = None
    experts: int = 0
    top_k: int = 0

    @property
    def lottery_n(self) -> int:
        return expert_n(self.n, self.experts)

    @property
    def committed_device(self) -> Device:
        return local_device(self.weight.device)

    def disable_mining(self, reason: str) -> None:
        with self.lock:
            first_failure = self.disabled_reason is None
            if first_failure:
                self.disabled_reason = reason
            self.mineable = False
            self.job_ctx = None
        if first_failure:
            _LOGGER.error(f"mining disabled for {self.layer_name}: {reason}")


def _mines_on(device: torch.device) -> bool:
    from pearl_gemm.protocol_constants import SM90_CC_MAJOR, SM100_CC_MAJOR, SM120_CC_MAJOR

    return torch.cuda.get_device_capability(device)[0] in (
        SM90_CC_MAJOR,
        SM100_CC_MAJOR,
        SM120_CC_MAJOR,
    )


def _kernels_available() -> bool:
    try:
        import pearl_gemm  # noqa: F401
    except Exception:
        _LOGGER.opt(exception=True).warning("pearl_gemm import failed; layer will not mine")
        return False
    return True


def _grouped_kernel_available(device: torch.device) -> bool:
    if not _kernels_available():
        return False
    from pearl_gemm import supports_grouped_mixed_gemm

    return supports_grouped_mixed_gemm(device)


@lru_cache(maxsize=256)
def _b_configs(config_name: str, n: int, k: int):
    from pearl_gemm import (
        TensorHashConfig,
        default_noisy_quant_config,
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
            COMMITMENT_CHUNK_SIZE,
        )
    )
    prepare = replace(default_noisy_quant_config(), **tuned(config_name, "noisy_quant", m=n, k=k))
    validate_noisy_quant_config(n, k, prepare)
    return commit, prepare


def _allocate_buffers(n: int, k: int, device: torch.device) -> LayerBuffers:
    from pearl_gemm import tensor_hash_workspace_bytes

    commit_config, prepare_config = _b_configs(device_config_name(device), n, k)
    z = lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=device)  # noqa: E731
    return LayerBuffers(
        b_prime=z((n, k), torch.float8_e4m3fn),
        b_peel=z((n, 2 * RANK), torch.bfloat16),
        alpha_b=torch.ones((n,), dtype=torch.bfloat16, device=device),
        beta_b=z((n,), torch.bfloat16),
        inv_alpha_b=torch.ones((n,), dtype=torch.float32, device=device),
        e2=z((n, RANK), torch.float8_e4m3fn),
        f1=z((RANK, k), torch.float8_e4m3fn),
        f1_hl=z((k, PACKED_NOISE_K), torch.int8),
        f2=z((RANK, k), torch.float8_e4m3fn),
        f2_hl=z((k, PACKED_NOISE_K), torch.int8),
        noise_lines=z((k, RANK), torch.float8_e4m3fn),
        root_codes=z((32,), torch.uint8),
        root_scales=z((32,), torch.uint8),
        tensor_hash_workspace=z((tensor_hash_workspace_bytes(n, k, commit_config),), torch.uint8),
        commit_stats=z((2 * (n * k // 512),), torch.float32),
        gram=z((RANK, RANK), torch.float32),
        commit_config=commit_config,
        prepare_config=prepare_config,
        key_a_dev=z((32,), torch.uint8),
        key_b_dev=z((32,), torch.uint8),
        seed_b_dev=z((32,), torch.uint8),
        noise_key_b_dev=z((32,), torch.uint8),
        threshold_dev=z((32,), torch.uint8),
    )


def supports_layer_shape(n: int, k: int) -> bool:
    return k % SCALE_BLOCK == 0 and n % 16 == 0 and k % 16 == 0


def _lottery_family_available(n: int, k: int, device: torch.device) -> bool:
    from pearl_gemm import supports_lottery_family

    tile = select_tile(n, k, device=local_device(device))
    return tile is not None and supports_lottery_family(tile.rows, device)


def can_mine_layer(n: int, k: int, device: torch.device, experts: int = 0) -> bool:
    if not _mines_on(device) or not _kernels_available():
        return False
    if experts and (n % (experts * MOE_LOTTERY_N) or not _grouped_kernel_available(device)):
        return False
    n_e = expert_n(n, experts)
    return (
        supports_layer_shape(n, k)
        and is_mineable_shape(n_e, k, device=local_device(device))
        and _lottery_family_available(n_e, k, device)
    )


def create_layer_state(
    layer_name: str, bf16_weight: torch.Tensor, *, experts: int = 0, top_k: int = 0
) -> LayerState:
    """Encode a loaded BF16 weight and allocate steady per-job mining buffers."""
    if bf16_weight.dtype != torch.bfloat16 or bf16_weight.dim() != 2 or not bf16_weight.is_cuda:
        raise ValueError(
            f"expected a 2D CUDA bfloat16 weight, got {bf16_weight.dtype} "
            f"{tuple(bf16_weight.shape)} on {bf16_weight.device}"
        )
    n, k = bf16_weight.shape
    if not supports_layer_shape(n, k):
        raise ValueError(f"unsupported mining shape n={n}, k={k}")
    if experts or top_k:
        validate_moe_dims(experts, top_k)

    pq = PrequantMatrix.encode(bf16_weight)
    weight = pq.int_values.contiguous()
    weight_scale = pq.scales.contiguous()
    w_fp8 = w_fp8_scale = None
    if not experts:
        w_fp8, w_fp8_scale = build_fp8_fallback(weight, weight_scale)

    if not can_mine_layer(n, k, weight.device, experts):
        raise RuntimeError(f"layer {layer_name} (n={n}, k={k}) is not mineable on this device")

    return LayerState(
        experts=experts,
        top_k=top_k,
        layer_name=layer_name,
        layer_id=next(_NEXT_LAYER_ID),
        weight=weight,
        weight_scale=weight_scale,
        weight_cpu=shared_copy(weight),
        weight_scale_cpu=shared_copy(weight_scale),
        n=n,
        k=k,
        mineable=True,
        buffers=_allocate_buffers(n, k, weight.device),
        w_fp8=w_fp8,
        w_fp8_scale=w_fp8_scale,
    )


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
    with _REGISTRY_LOCK:
        _STATE_REGISTRY.clear()
        _LAYER_ID_REGISTRY.clear()
