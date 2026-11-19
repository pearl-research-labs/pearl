"""The ``pearl`` vLLM quantization method: FP10 encoding and mined GEMM dispatch."""

from typing import Any, override

import torch
from miner_utils import get_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    apply_moe_activation,
    apply_moe_activation_supported,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter

from .capture import suspend_submissions_until_capture_complete
from .moe import MoeRouting, combine_routed_rows
from .settings import is_layer_ignored, runtime_settings
from .state import LayerState, can_mine_layer, create_layer_state, register_state, unregister_state
from .upcast import UpcastBudgetError, reconstruct_bf16

_LOGGER = get_logger("vllm.pearl_miner")
_DIRECT_WEIGHT_READ_SUFFIXES = (".kv_b_proj",)
_TensorSnapshot = tuple[dict[str, Any], dict[str, Any]]


def _validate_source_config(method: str, config: dict[str, Any]) -> None:
    block_size = config.get("weight_block_size")
    if block_size is None:
        return
    valid = (
        isinstance(block_size, list | tuple)
        and len(block_size) == 2
        and all(type(dim) is int and dim > 0 for dim in block_size)
    )
    if not valid:
        raise ValueError(
            f"pearl: checkpoint declares quant_method={method!r} with an invalid "
            f"weight_block_size={block_size!r}; expected two positive integers."
        )


def _build_source_config(config: dict[str, Any]) -> QuantizationConfig | None:
    if not config or not (method := config.get("quant_method")):
        return None
    _validate_source_config(method, config)
    try:
        from vllm.model_executor.layers.quantization import get_quantization_config

        return get_quantization_config(method).from_config(config)
    except Exception as exc:
        raise ValueError(
            f"pearl: checkpoint declares quant_method={method!r} but it could not be "
            "delegated (unregistered or incompatible config); refusing to misload it as BF16."
        ) from exc


def _snapshot_tensors(layer: torch.nn.Module) -> _TensorSnapshot:
    return (
        dict(getattr(layer, "_parameters", None) or {}),
        dict(getattr(layer, "_buffers", None) or {}),
    )


def _restore_tensors(layer: torch.nn.Module, snapshot: _TensorSnapshot) -> None:
    for attribute, saved in zip(("_parameters", "_buffers"), snapshot, strict=True):
        target = getattr(layer, attribute, None)
        if target is not None:
            target.clear()
            target.update(saved)


class PearlLinearMethod(UnquantizedLinearMethod):
    """Encodes a mineable linear layer's BF16 weight to FP10 and routes apply through pearl::apply_linear."""

    def __init__(self, prefix: str, source: LinearMethodBase | None = None):
        super().__init__()
        self.prefix = prefix
        self.source = source
        self._state: LayerState | None = None
        self._serve_via_source = False
        self._n = 0
        self._k = 0

    @override
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        self._n = sum(output_partition_sizes)
        self._k = input_size_per_partition
        target = self.source if self.source is not None else super()
        target.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    @staticmethod
    def _install_state_parameters(layer: torch.nn.Module, state: LayerState) -> None:
        weight_param = torch.nn.Parameter(state.weight, requires_grad=False)
        if "weight" in getattr(layer, "_parameters", {}):
            replace_parameter(layer, "weight", weight_param)
        else:
            layer.register_parameter("weight", weight_param)
        if "weight_scale" in getattr(layer, "_parameters", {}):
            delattr(layer, "weight_scale")
        layer.register_parameter(
            "weight_scale", torch.nn.Parameter(state.weight_scale, requires_grad=False)
        )

    def _device(self, layer: torch.nn.Module) -> torch.device:
        weight = getattr(layer, "weight", None)
        data = getattr(weight, "data", weight)
        device = getattr(data, "device", None)
        if device is not None:
            return device
        for param in layer.parameters(recurse=False):
            return param.device
        return torch.device(torch.cuda.current_device())

    def _serve_via_source_or_unquantized(self, layer: torch.nn.Module) -> None:
        if self.source is not None:
            self._serve_via_source = True
        else:
            super().process_weights_after_loading(layer)

    def _clear_source_tensors(self, layer: torch.nn.Module) -> _TensorSnapshot:
        snapshot = _snapshot_tensors(layer)
        for name in list(getattr(layer, "_parameters", None) or {}):
            if name != "bias":
                del layer._parameters[name]
        for name in list(getattr(layer, "_buffers", None) or {}):
            del layer._buffers[name]
        return snapshot

    def _load_bf16_weight(
        self, layer: torch.nn.Module, n: int, k: int, device: torch.device
    ) -> torch.Tensor | None:
        if self.source is None:
            weight = layer.weight.data
        else:
            try:
                weight = reconstruct_bf16(layer, self.source, n, k, device)
            except UpcastBudgetError as exc:
                _LOGGER.warning(
                    f"layer {self.prefix} (n={n}, k={k}) upcast workspace exceeds memory budget "
                    f"on {device}; serving via source scheme. ({exc})"
                )
                return None
        if tuple(weight.shape) != (n, k):
            raise ValueError(
                f"pearl: {self.prefix} was gated as mineable at (n={n}, k={k}) but the "
                f"weight to encode has shape {tuple(weight.shape)}"
            )
        return weight

    @override
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.source is not None:
            self.source.process_weights_after_loading(layer)
        from .runtime import mining_disabled_by_runtime_failure

        if mining_disabled_by_runtime_failure():
            _LOGGER.warning(
                "Process-local mining startup rolled back; Pearl layers keep their source scheme."
            )
            self._serve_via_source_or_unquantized(layer)
            return

        n, k = self._n, self._k
        device = self._device(layer)
        if not can_mine_layer(n, k, device):
            _LOGGER.info(
                f"layer {self.prefix} (n={n}, k={k}) not mineable here; serving unquantized"
            )
            self._serve_via_source_or_unquantized(layer)
            return

        weight = self._load_bf16_weight(layer, n, k, device)
        if weight is None:
            self._serve_via_source_or_unquantized(layer)
            return

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        snapshot = (
            self._clear_source_tensors(layer)
            if self.source is not None
            else _snapshot_tensors(layer)
        )
        try:
            state = create_layer_state(self.prefix, weight)
            try:
                self._install_state_parameters(layer, state)
                register_state(state)
            except BaseException:
                unregister_state(state)
                _restore_tensors(layer, snapshot)
                raise
        except torch.cuda.OutOfMemoryError:
            if self.source is not None:
                _restore_tensors(layer, snapshot)
            _LOGGER.warning(
                f"Pearl state allocation OOM for {self.prefix}; serving via source scheme."
            )
            self._serve_via_source_or_unquantized(layer)
            return

        self._state = state
        if state.mineable:
            suspend_submissions_until_capture_complete()

    @override
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._state is not None:
            from . import linear_op  # noqa: F401

            return torch.ops.pearl.apply_linear(x, layer.weight, bias)
        if self._serve_via_source and self.source is not None:
            return self.source.apply(layer, x, bias)
        return super().apply(layer, x, bias)


def moe_config_unsupported_reason(moe: FusedMoEConfig) -> str | None:
    if not moe.is_act_and_mul or not apply_moe_activation_supported(moe.activation):
        return f"activation {moe.activation.value}"
    if moe.swiglu_limit is not None or moe.swiglu_alpha is not None or moe.swiglu_beta is not None:
        return "swiglu gate parameters"
    if moe.has_bias:
        return "expert biases"
    if moe.use_ep:
        return "expert parallelism"
    if moe.is_lora_enabled:
        return "LoRA experts"
    return None


class PearlMoEMethod(UnquantizedFusedMoEMethod):
    """Mines the first grouped GEMM (w13) of a BF16 routed-expert MoE layer."""

    def __init__(self, moe: FusedMoEConfig, prefix: str):
        if moe.moe_backend not in ("auto", "triton"):
            _LOGGER.info(f"{prefix}: forcing moe_backend=triton (was {moe.moe_backend})")
        moe.moe_backend = "triton"
        super().__init__(moe)
        self.prefix = prefix
        self._state: LayerState | None = None

    @override
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if is_layer_ignored(self.prefix, runtime_settings().ignored_layers):
            return
        w13 = layer.w13_weight.data
        experts, n_e, k = w13.shape
        weight = w13.reshape(experts * n_e, k)
        reason = moe_config_unsupported_reason(self.moe)
        if reason is None and layer.expert_map is not None:
            reason = "expert parallelism"
        if reason is None and layer.apply_router_weight_on_input:
            reason = "router weights applied on the input"
        if reason is None and not can_mine_layer(experts * n_e, k, w13.device, experts):
            reason = f"shape (experts={experts}, n_e={n_e}, k={k})"
        if reason is not None:
            _LOGGER.info(f"MoE layer {self.prefix} not mineable here ({reason}); serving unmined")
            return
        try:
            state = create_layer_state(
                self.prefix, weight, experts=experts, top_k=self.moe.experts_per_token
            )
            register_state(state)
        except torch.cuda.OutOfMemoryError:
            _LOGGER.warning(
                f"Pearl state allocation OOM for {self.prefix}; serving MoE layer unmined."
            )
            return
        self._state = state
        if state.mineable:
            suspend_submissions_until_capture_complete()

    @override
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if self._state is not None and shared_experts is None:
            if x.dtype != torch.bfloat16:
                raise TypeError(f"Pearl mined MoE requires bfloat16 activations, got {x.dtype}")
            from .linear_op import try_mine_moe

            out = try_mine_moe(
                self._state,
                x,
                topk_ids,
                lambda gate_up, routing: self._mined_tail(layer, topk_weights, gate_up, routing),
            )
            if out is not None:
                return out
        return super().apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)

    def _mined_tail(
        self,
        layer: torch.nn.Module,
        topk_weights: torch.Tensor,
        gate_up: torch.Tensor,
        routing: MoeRouting,
    ) -> torch.Tensor:
        hidden = torch.empty(
            gate_up.shape[0], gate_up.shape[1] // 2, dtype=gate_up.dtype, device=gate_up.device
        )
        apply_moe_activation(
            self.moe.activation,
            hidden,
            gate_up,
            activation_config=ApplyMoEActivationConfig.from_configs(
                self.moe, self.moe_quant_config
            ),
        )
        # torch._grouped_mm is private API (pinned torch==2.13.0).
        down = torch._grouped_mm(hidden, layer.w2_weight.transpose(1, 2), offs=routing.m_indptr[1:])
        return combine_routed_rows(down, topk_weights, routing)


class PearlConfig(QuantizationConfig):
    """Engine-selected config over BF16 or quantized HuggingFace checkpoints."""

    def __init__(self, source_config: QuantizationConfig | None = None):
        super().__init__()
        self.source_config = source_config
        if source_config is not None:
            self._ignore_unexpected_suffixes = list(
                dict.fromkeys(
                    [
                        *self._ignore_unexpected_suffixes,
                        *getattr(source_config, "_ignore_unexpected_suffixes", ()),
                    ]
                )
            )

    @property
    def weight_block_size(self) -> list[int] | None:
        if self.source_config is None:
            raise AttributeError("weight_block_size")
        return self.source_config.weight_block_size

    def has_blocked_weights(self) -> bool:
        source = self.source_config
        if source is None:
            return False
        if hasattr(source, "weight_block_size"):
            return source.weight_block_size is not None
        if hasattr(source, "has_blocked_weights"):
            return bool(source.has_blocked_weights())
        return False

    @override
    def get_cache_scale_mapper(self) -> Any:
        if self.source_config is not None:
            return self.source_config.get_cache_scale_mapper()
        return super().get_cache_scale_mapper()

    @override
    def get_name(self) -> QuantizationMethods:
        return "pearl"

    @override
    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    @override
    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @override
    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @override
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PearlConfig":
        return cls(source_config=_build_source_config(config))

    @override
    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg: dict[str, Any], user_quant: str | None, hf_config: Any = None
    ) -> QuantizationMethods | None:
        if user_quant == "pearl" and hf_quant_cfg:
            return "pearl"
        return None

    @override
    def apply_vllm_mapper(self, hf_to_vllm_mapper: Any) -> None:
        if self.source_config is not None:
            self.source_config.apply_vllm_mapper(hf_to_vllm_mapper)

    @override
    def maybe_update_config(
        self, model_name: str, hf_config: Any = None, revision: str | None = None
    ) -> None:
        if self.source_config is not None:
            self.source_config.maybe_update_config(model_name, hf_config, revision)

    @staticmethod
    def _non_linear_method(
        layer: torch.nn.Module, prefix: str, source_method: QuantizeMethodBase | None
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts) and source_method is None:
            reason = moe_config_unsupported_reason(layer.moe_config)
            if reason is None and getattr(layer, "apply_router_weight_on_input", False):
                reason = "router weights applied on the input"
            if reason is not None:
                _LOGGER.info(f"MoE layer {prefix} not mineable here ({reason}); serving unmined")
                return None
            return PearlMoEMethod(layer.moe_config, prefix)
        return source_method

    @override
    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        source_method: QuantizeMethodBase | None = None
        if self.source_config is not None:
            self.source_config.packed_modules_mapping = self.packed_modules_mapping
            source_method = self.source_config.get_quant_method(layer, prefix)

        if not isinstance(layer, LinearBase):
            return self._non_linear_method(layer, prefix, source_method)

        if prefix == "kv_b_proj" or prefix.endswith(_DIRECT_WEIGHT_READ_SUFFIXES):
            _LOGGER.warning("Pearl leaves direct-read MLA kv_b_proj weights on the source scheme.")
            return source_method if source_method is not None else UnquantizedLinearMethod()

        if is_layer_ignored(prefix, runtime_settings().ignored_layers):
            return source_method if source_method is not None else UnquantizedLinearMethod()

        if source_method is None:
            return PearlLinearMethod(prefix)
        if isinstance(source_method, UnquantizedLinearMethod):
            if self.source_config is not None:
                raise ValueError(
                    f"pearl: quantized checkpoint left {prefix!r} (LinearBase) on the "
                    "delegate's unquantized path; refusing to mine a packed weight as BF16."
                )
            return PearlLinearMethod(prefix)
        if isinstance(source_method, LinearMethodBase):
            return PearlLinearMethod(prefix, source=source_method)
        return source_method
