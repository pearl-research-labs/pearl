"""The ``pearl`` vLLM quantization method: load-time FP10 encoding of selected
linear layers, mined GEMM dispatch through ``pearl::apply_linear``.

Selected with ``--quantization pearl``. Two source shapes are supported:

* **Plain BF16 checkpoints** carry no quantization config, so vLLM
  default-constructs this config and each mineable linear is encoded straight
  from its BF16 weight.
* **Quantized checkpoints** (plain ``fp8`` block/per-tensor, or
  ``compressed-tensors`` FP8 / NVFP4 / MXFP4) are delegated to the matching
  vLLM quant method: mineable dense linears are decoded back to BF16
  (:mod:`vllm_miner.upcast`) then encoded to FP10, while every
  non-mined layer -- MoE experts, KV cache, embeddings, ignored/non-mineable
  linears -- keeps its compact source scheme so a quantized model still fits.
  Upcasting is lossy: it mines over already-quantized weights.

  ``--quantization pearl`` takes over such checkpoints via
  ``override_quantization_method`` (vLLM would otherwise refuse the mismatch).

Each CUDA worker mines the dense 2-D shard loaded in that process. Layers matched
by ``PEARL_IGNORED_LAYERS`` and non-linear layers keep the source/unquantized
path.
"""

from typing import Any, override

import torch
from miner_utils import get_logger
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

from .settings import is_layer_ignored, runtime_settings
from .state import (
    LayerState,
    can_mine_layer,
    create_layer_state,
    refresh_layer_state,
    register_state,
    unregister_state,
)
from .upcast import UpcastBudgetError, reconstruct_bf16

_LOGGER = get_logger("vllm.pearl_miner")
_DIRECT_WEIGHT_READ_SUFFIXES = (".kv_b_proj",)


def _validate_source_config(method: str, config: dict[str, Any]) -> None:
    """Screen checkpoint metadata that would misconfigure delegate weight
    allocation. vLLM's ``from_config`` accepts a malformed ``weight_block_size``
    (non-pair, zero, or negative dims) and only trips much later inside a
    kernel, by which point the packed parameters are already allocated."""
    block_size = config.get("weight_block_size")
    if block_size is None:
        return
    valid = (
        isinstance(block_size, list | tuple)
        and len(block_size) == 2
        and all(
            isinstance(dim, int) and not isinstance(dim, bool) and dim > 0 for dim in block_size
        )
    )
    if not valid:
        raise ValueError(
            f"pearl: checkpoint declares quant_method={method!r} with an invalid "
            f"weight_block_size={block_size!r}; expected two positive integers."
        )


def _build_source_config(config: dict[str, Any]) -> QuantizationConfig | None:
    """Build a delegate quant config from a checkpoint's ``quantization_config``
    so its layers load in their compact format (and mineable dense linears can
    be decoded back to BF16). Returns ``None`` only for genuinely unquantized
    checkpoints (plain BF16). Raises if a real quant method is declared but
    cannot be delegated, rather than silently misloading a quantized checkpoint
    as BF16.

    Supported delegates are whatever vLLM registers for the checkpoint's
    ``quant_method`` -- notably ``fp8`` (block/per-tensor) and
    ``compressed-tensors`` (FP8 / NVFP4 / MXFP4)."""
    if not config:
        return None
    method = config.get("quant_method")
    if not method:
        return None
    _validate_source_config(method, config)
    try:
        from vllm.model_executor.layers.quantization import get_quantization_config

        return get_quantization_config(method).from_config(config)
    except Exception as exc:
        # The checkpoint declares a real quantization method we could not
        # delegate to. Silently treating it as BF16 would let create_weights
        # allocate an unquantized weight and mine packed quantized tensors,
        # which is a correctness bug -- fail loudly instead.
        raise ValueError(
            f"pearl: checkpoint declares quant_method={method!r} but it could not be "
            "delegated (unregistered or incompatible config); refusing to misload it as "
            "BF16. Add the delegate, exclude the model, or fix the checkpoint config."
        ) from exc


def _set_param(layer: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    """(Re)register ``name`` as a plain non-grad parameter, dropping any prior."""
    layer._parameters.pop(name, None)
    layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))


# Everything a layer holds that we may delete while installing Pearl state:
# (parameters, buffers). Kept so a failed install can put the layer back.
_TensorSnapshot = tuple[dict[str, Any], dict[str, Any]]


def _snapshot_tensors(layer: torch.nn.Module) -> _TensorSnapshot:
    return (
        dict(getattr(layer, "_parameters", None) or {}),
        dict(getattr(layer, "_buffers", None) or {}),
    )


def _restore_tensors(layer: torch.nn.Module, snapshot: _TensorSnapshot) -> None:
    for attribute, saved in zip(("_parameters", "_buffers"), snapshot, strict=True):
        target = getattr(layer, attribute, None)
        if target is None:
            continue
        target.clear()
        target.update(saved)


class PearlLinearMethod(UnquantizedLinearMethod):
    """Mines a linear layer by encoding its BF16 weight to the schema FP10 planes
    in ``process_weights_after_loading`` and registering the GPU mining state.

    With a compressed-tensors ``source`` delegate the weight is loaded in the
    checkpoint's packed format and decoded to BF16 before encoding; without one
    the BF16 weight is used directly. Local shards with an unsupported shape or
    device stay on the source scheme, or the inherited unquantized path when
    there is no source.
    """

    def __init__(self, prefix: str, source: LinearMethodBase | None = None):
        super().__init__()
        self.prefix = prefix
        self.source = source
        self._mined = False
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
        # vLLM supplies process-local partition sizes here; that dense 2-D shard
        # is the mining operand for this CUDA worker.
        self._n = sum(output_partition_sizes)
        self._k = input_size_per_partition
        if self.source is not None:
            self.source.create_weights(
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                **extra_weight_attrs,
            )
            return
        super().create_weights(
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
        # The int8 codes plane replaces BF16 ``weight`` and is the dispatch key.
        # replace_parameter needs an existing param (it updates in place to keep
        # the reload graph pointer); after a source upcast the packed weight was
        # dropped, so register it fresh instead.
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
        # Derive from the (packed) ``weight`` the delegate's ``apply`` reads, not
        # the first registered parameter: a CPU scale/bias registered before the
        # weight would otherwise make ``reconstruct_bf16`` build its identity and
        # destination on CPU while ``layer.weight`` is on CUDA, and
        # ``source.apply`` would fail with a device mismatch. Read through
        # ``.data`` so a Parameter, a plain tensor, and the reload path's stand-in
        # all resolve the same device the weight is read from elsewhere.
        weight = getattr(layer, "weight", None)
        data = getattr(weight, "data", weight)
        device = getattr(data, "device", None)
        if device is not None:
            return device
        for param in layer.parameters(recurse=False):
            return param.device
        return torch.device(torch.cuda.current_device())

    @staticmethod
    def _synchronize_layer_device(device: torch.device) -> None:
        """Drain in-flight CUDA work on the layer's device before in-place tensor
        mutation.

        ``process_weights_after_loading`` deletes source parameters/buffers and
        replaces the dispatch weight while a previous forward's kernels may
        still be running on the compute stream (the worker-level reload
        lifecycle in :mod:`vllm_miner.lifecycle` suspends *mining*
        producers, but vLLM's own reload-time kernels and any draining serving
        work share the default stream). Waiting here matches the contract
        :func:`refresh_layer_state` already assumes -- "all layer readers drained
        by the framework lifecycle coordinator" -- for the first-install path
        too, so a concurrent forward cannot observe a half-installed layer.

        A CPU device (the CPU tests) is a no-op: there is no async stream to
        drain and the host-side mutation is already ordered.
        """
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _serve_via_source_or_unquantized(self, layer: torch.nn.Module) -> None:
        """Keep serving via the source scheme (weights already processed) or,
        without a source, the inherited unquantized path."""
        if self.source is not None:
            self._serve_via_source = True
        else:
            super().process_weights_after_loading(layer)

    def _clear_source_tensors(self, layer: torch.nn.Module) -> _TensorSnapshot:
        """Free the packed source weights/scales *and buffers* once decoded;
        keep ``bias`` (registered by the layer, not the source) and any pearl
        params. Returns a snapshot of the layer's tensors as they were before
        the removal, so a later failure can restore a usable layer."""
        snapshot = _snapshot_tensors(layer)
        for name in list(getattr(layer, "_parameters", None) or {}):
            if name != "bias":
                del layer._parameters[name]
        for name in list(getattr(layer, "_buffers", None) or {}):
            del layer._buffers[name]
        return snapshot

    def _reset_mining_state(self, layer: torch.nn.Module) -> None:
        """Retire any registered mining state and the dispatch markers, so
        ``apply`` stops routing through ``pearl::apply_linear``. Used before
        installing a plain BF16 weight over a layer that a previous pass (a
        reload) already encoded."""
        if self._state is not None:
            unregister_state(self._state)
            self._state = None
        self._mined = False
        self._serve_via_source = False
        if "weight_scale" in (getattr(layer, "_parameters", None) or {}):
            del layer._parameters["weight_scale"]

    def _create_and_install_state(
        self,
        layer: torch.nn.Module,
        weight: torch.Tensor,
        restore: _TensorSnapshot | None = None,
    ) -> LayerState:
        """Create, install, and register a fresh mining state for a layer that
        has no existing state yet.

        Contract: replacement Pearl parameters are installed
        (``_install_state_parameters``) *before* ``register_state`` so the
        registry never observes partially-installed state. On failure the state
        is unregistered and the layer's parameters and buffers are restored
        from ``restore`` -- the snapshot taken before ``_clear_source_tensors``
        dropped the packed source tensors -- or from one taken here when the
        weight came straight from the layer. Either way a failed install leaves
        the layer with the tensors it had on entry."""
        snapshot = restore if restore is not None else _snapshot_tensors(layer)
        state = create_layer_state(self.prefix, weight)
        try:
            self._install_state_parameters(layer, state)
            register_state(state)
        except BaseException:
            unregister_state(state)
            _restore_tensors(layer, snapshot)
            raise
        self._state = state
        self._mined = True
        return state

    def _reconstruct_mining_weight(
        self, layer: torch.nn.Module, n: int, k: int, device: torch.device
    ) -> torch.Tensor | None:
        """Decode the quantized source weight to BF16 for encoding, or return
        ``None`` to signal the caller should serve via the source scheme.

        ``reconstruct_bf16`` raises :class:`UpcastBudgetError` before allocating
        when the transient workspace would exceed the device's free-memory
        budget; that is not an encoding failure, just a layer too large to
        upcast alongside its packed source weights, so fall back to serving
        rather than aborting startup.
        """
        try:
            return reconstruct_bf16(layer, self.source, n, k, device)
        except UpcastBudgetError as exc:
            _LOGGER.warning(
                f"layer {self.prefix} (n={n}, k={k}) is mineable by shape but the "
                f"upcast workspace does not fit on {device}; serving via the source "
                f"scheme and not mining. ({exc})"
            )
            return None

    def _fall_back_after_runtime_failure(self, layer: torch.nn.Module) -> bool:
        if self._state is not None:
            return False
        from .mining_state import mining_disabled_by_runtime_failure

        if not mining_disabled_by_runtime_failure():
            return False
        _LOGGER.warning(
            "Process-local mining startup rolled back; Pearl layers keep their source scheme.",
        )
        self._serve_via_source_or_unquantized(layer)
        return True

    def _refresh_state_or_restore(
        self,
        layer: torch.nn.Module,
        weight: torch.Tensor,
    ) -> LayerState:
        state = self._state
        if state is None:
            raise RuntimeError("Pearl reload lost its registered layer state")
        try:
            refresh_layer_state(state, weight)
        except torch.cuda.OutOfMemoryError:
            # Encoding allocates all replacement planes before mutating the
            # graph-stable state. Put those previous state tensors back on the
            # layer so the failed reload remains coherent while the lifecycle
            # keeps this worker paused for restart.
            self._install_state_parameters(layer, state)
            raise
        self._install_state_parameters(layer, state)
        return state

    def _create_state_or_fall_back(
        self,
        layer: torch.nn.Module,
        weight: torch.Tensor,
        source_snapshot: _TensorSnapshot | None,
    ) -> LayerState | None:
        try:
            return self._create_and_install_state(layer, weight, restore=source_snapshot)
        except torch.cuda.OutOfMemoryError:
            if source_snapshot is not None:
                _restore_tensors(layer, source_snapshot)
            _LOGGER.warning(
                f"Pearl state allocation ran out of memory for {self.prefix}; "
                "serving this layer via its source scheme.",
            )
            self._serve_via_source_or_unquantized(layer)
            return None

    @override
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Finalise the source's packed weights before mining or serving it.
        if self.source is not None:
            self.source.process_weights_after_loading(layer)
        if self._fall_back_after_runtime_failure(layer):
            return

        n, k = self._n, self._k
        device = self._device(layer)
        if self._state is None and not can_mine_layer(n, k, device):
            _LOGGER.info(
                f"layer {self.prefix} (n={n}, k={k}) not mineable here; serving unquantized"
            )
            self._serve_via_source_or_unquantized(layer)
            return

        if self.source is not None:
            # Lossy: decode already-quantized weights back to BF16 for encoding.
            # Returns None when the upcast workspace would OOM the worker, in
            # which case serve via the source scheme instead of mining.
            weight = self._reconstruct_mining_weight(layer, n, k, device)
            if weight is None:
                self._serve_via_source_or_unquantized(layer)
                return
        else:
            weight = layer.weight.data

        # The mineability gate above keyed on create_weights' (n, k); encoding a
        # differently shaped weight would mine a tensor that was never vetted.
        # Checked before the packed source tensors are dropped, so a mismatch
        # leaves the layer as the loader built it.
        if tuple(weight.shape) != (n, k):
            raise ValueError(
                f"pearl: {self.prefix} was gated as mineable at (n={n}, k={k}) but the "
                f"weight to encode has shape {tuple(weight.shape)}"
            )

        # Drain in-flight CUDA work before mutating layer tensors in place.
        # The worker reload lifecycle gates *mining* producers, but a draining
        # serving forward (or vLLM's own reload-time kernels) may still hold the
        # default compute stream; deleting/replacing source tensors underneath
        # it would let a concurrent ``apply`` observe a half-installed layer.
        self._synchronize_layer_device(device)
        source_snapshot = self._clear_source_tensors(layer) if self.source is not None else None

        if self._state is not None:
            # vLLM 0.28 layerwise reload restores checkpoint-shaped BF16
            # parameters, calls this method, then copies kernel-format values
            # back into the original graph-visible storage. Refresh every Pearl
            # operand in place so those graph pointers and registry keys survive.
            state = self._refresh_state_or_restore(layer, weight)
        else:
            state = self._create_state_or_fall_back(layer, weight, source_snapshot)
            if state is None:
                return

        if state.mineable:
            from .cuda_graph_submission_gate import suspend_submissions_until_capture_complete

            suspend_submissions_until_capture_complete()

    @override
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._mined:
            if x.dtype != torch.bfloat16:
                raise TypeError(f"Pearl mined linear requires bfloat16 activations, got {x.dtype}")
            from . import linear_op  # noqa: F401  (registers pearl::apply_linear)

            return torch.ops.pearl.apply_linear(x, layer.weight, bias)
        if self._serve_via_source and self.source is not None:
            return self.source.apply(layer, x, bias)
        return super().apply(layer, x, bias)


class PearlConfig(QuantizationConfig):
    """Engine-selected config over BF16 or compressed-tensors checkpoints.

    ``source_config`` is a compressed-tensors delegate built from the
    checkpoint's ``quantization_config`` (``None`` for plain BF16). Mineable
    dense linears are wrapped by :class:`PearlLinearMethod`; every other layer
    is served by the delegate so a quantized model keeps its compact footprint.
    """

    def __init__(self, source_config: QuantizationConfig | None = None):
        super().__init__()
        self.source_config = source_config
        # ``AutoWeightsLoader`` reads this attribute off the engine-selected
        # config only, so extend -- never replace -- pearl's inherited allowlist
        # with the delegate's extra suffixes. Replacing it would drop the base
        # KV-cache scale suffixes for any delegate shipping a narrower list.
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
        """Surface the delegate's block shape so vLLM's ``has_blocked_weights``
        check enables the ``+quant_fp8`` custom op (block-fp8 activation quant).
        Without this the MLA KV / DSA-indexer / MoE activation math is wrong.

        That check probes ``hasattr(quant_config, "weight_block_size")`` *before*
        falling back to ``has_blocked_weights()``, so a delegate that answers
        only through the latter (compressed-tensors) must not see this attribute:
        raise ``AttributeError`` there rather than reporting a misleading
        ``None``, which would silently disable ``+quant_fp8``.
        """
        if self.source_config is None:
            raise AttributeError("weight_block_size")
        # Propagates AttributeError when the delegate has no block-size concept.
        return self.source_config.weight_block_size

    def has_blocked_weights(self) -> bool:
        """Blockiness probe for delegates without a ``weight_block_size``."""
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
        # Delegate KV-cache scale name remapping (e.g. ``.k_proj.output_scale``
        # -> ``.attn.k_scale``) so fp8 kv scales load onto the right params.
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
        return 100  # SM100 only: the mining kernels and the FP8 fallback GEMM

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
        # vLLM otherwise rejects ``--quantization pearl`` on a checkpoint whose
        # own quant_method differs (e.g. fp8). Claim any quantized checkpoint
        # when the user explicitly asked for pearl so it can be mined + served
        # via the delegate; plain BF16 (no hf config) needs no override.
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

    @override
    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        source_method: QuantizeMethodBase | None = None
        if self.source_config is not None:
            # The model updates the mapping on the engine-selected config only;
            # forward it so the delegate resolves fused/packed targets too.
            self.source_config.packed_modules_mapping = self.packed_modules_mapping
            source_method = self.source_config.get_quant_method(layer, prefix)

        # Non-linear layers (MoE experts, KV cache, embeddings): serve via the
        # source scheme so they keep the checkpoint's compact format. No mining.
        if not isinstance(layer, LinearBase):
            return source_method

        # MLA reads kv_b_proj weights directly (not via ``apply``): leave them on
        # the source scheme (or BF16) rather than encoding to FP10 planes.
        if prefix == "kv_b_proj" or prefix.endswith(_DIRECT_WEIGHT_READ_SUFFIXES):
            _LOGGER.warning(
                "Pearl leaves direct-read MLA kv_b_proj weights on the source scheme.",
            )
            return source_method if source_method is not None else UnquantizedLinearMethod()

        if is_layer_ignored(prefix, runtime_settings().ignored_layers):
            return source_method if source_method is not None else UnquantizedLinearMethod()

        # Mineable dense linear: mine a real BF16 weight directly (no source),
        # or wrap a quantized source so process_weights_after_loading can
        # reconstruct BF16 before encoding. Guard against a quantized checkpoint
        # leaving a LinearBase on the unquantized path with a still-packed
        # weight -- encoding that as BF16 would silently mine garbage, so fail
        # closed with an actionable message rather than mis-mine.
        if source_method is None:
            return PearlLinearMethod(prefix)
        if isinstance(source_method, UnquantizedLinearMethod):
            if self.source_config is not None:
                raise ValueError(
                    f"pearl: quantized checkpoint left {prefix!r} (LinearBase) on the "
                    "delegate's unquantized path, but its packed weight is not a real "
                    "BF16 tensor -- refusing to mine it as BF16. Add the suffix to the "
                    "delegate's packed_modules_mapping, or exclude the layer with "
                    "PEARL_IGNORED_LAYERS."
                )
            return PearlLinearMethod(prefix)
        if isinstance(source_method, LinearMethodBase):
            return PearlLinearMethod(prefix, source=source_method)
        # Unexpected delegate kind: serve it, do not mine.
        return source_method
