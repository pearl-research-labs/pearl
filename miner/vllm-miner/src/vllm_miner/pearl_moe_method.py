from typing import TYPE_CHECKING

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from miner_utils import get_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEActivationFormat,
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.utils import set_weight_attrs

from .mining_state import ensure_pinned_pool_at_least

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

_LOGGER = get_logger("vllm.pearl_miner")

WEIGHT_DTYPE = torch.int8
SCALE_DTYPE = torch.float32
SMOOTH_SCALE_DTYPE = torch.bfloat16
SCALE_LAST_DIM = 1
QUANT_METHOD_CHANNEL = "channel"
W13_WEIGHT_SHARDS_WITH_ACT_AND_MUL = 2
W13_WEIGHT_SHARDS_WITHOUT_ACT_AND_MUL = 1
# Fused gate+up smooth scale: checkpoint provides one half; we replicate to the other.
W13_SMOOTH_PROJ_HALVES = 2
# Expert row holding the shared gate/up smooth scale in checkpoints.
W13_SMOOTH_SHARED_EXPERT_INDEX = 0
# Per-layer pinned PoW headers held while a forward's callback is pending.
MOE_POW_HEADER_POOL_DEPTH = 16


class PearlMoEMethod(FusedMoEMethodBase):
    def __init__(self, moe_config: FusedMoEConfig):
        super().__init__(moe_config)

    @classmethod
    def from_config(cls, moe_config: FusedMoEConfig) -> "PearlMoEMethod":
        return cls(moe_config)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        w13_num_shards = (
            W13_WEIGHT_SHARDS_WITH_ACT_AND_MUL
            if self.moe.is_act_and_mul
            else W13_WEIGHT_SHARDS_WITHOUT_ACT_AND_MUL
        )

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size,
                dtype=WEIGHT_DTYPE,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size_per_partition, dtype=WEIGHT_DTYPE
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                SCALE_LAST_DIM,
                dtype=SCALE_DTYPE,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, SCALE_LAST_DIM, dtype=SCALE_DTYPE),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update({"quant_method": QUANT_METHOD_CHANNEL})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        smooth_attrs = dict(extra_weight_attrs)
        w2_smooth_quant_scale = torch.nn.Parameter(
            torch.ones(num_experts, intermediate_size_per_partition, dtype=SMOOTH_SCALE_DTYPE),
            requires_grad=False,
        )
        layer.register_parameter("w2_smooth_quant_scale", w2_smooth_quant_scale)
        set_weight_attrs(w2_smooth_quant_scale, smooth_attrs)

        w13_smooth_quant_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2 * hidden_size, dtype=SMOOTH_SCALE_DTYPE),
            requires_grad=False,
        )
        layer.register_parameter("w13_smooth_quant_scale", w13_smooth_quant_scale)
        set_weight_attrs(w13_smooth_quant_scale, smooth_attrs)

        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13_smooth = getattr(layer, "w13_smooth_quant_scale", None)
        if w13_smooth is not None:
            half = w13_smooth.shape[1] // W13_SMOOTH_PROJ_HALVES
            w13_smooth.data[:, half:] = w13_smooth.data[:, :half]
            # Checkpoint stores one shared gate/up smooth scale on this row.
            src = W13_SMOOTH_SHARED_EXPERT_INDEX
            if w13_smooth.shape[0] > src + 1:
                w13_smooth.data[src + 1 :] = w13_smooth.data[src : src + 1]

        num_experts = layer.w13_weight.shape[0]
        ensure_pinned_pool_at_least(num_experts * MOE_POW_HEADER_POOL_DEPTH)
        self._setup_kernel(layer)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        return FusedMoEQuantConfig.make(
            quant_dtype=None,
            per_act_token_quant=True,
            per_out_ch_quant=True,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        )

    def _setup_kernel(self, layer: torch.nn.Module) -> None:
        """Build and store the modular MoE kernel so ``apply`` can delegate."""
        from .pearl_moe_experts import PearlMoEExperts

        try:
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            prepare_finalize = MoEPrepareAndFinalizeNoDPEPModular()
            experts = PearlMoEExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                layer=layer,
            )
            self.moe_kernel = mk.FusedMoEKernel(
                prepare_finalize=prepare_finalize,
                fused_experts=experts,
            )
            _LOGGER.info("Using PearlMoEExperts for MoE layer")
        except Exception:
            # Leave moe_kernel unset; layer may initialise via maybe_init_modular_kernel.
            _LOGGER.exception("Failed to set up PearlMoE kernel; will rely on lazy init")

    def apply(
        self,
        layer: "FusedMoE",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.moe_kernel is not None, (
            "PearlMoEMethod.apply called before moe_kernel was initialised"
        )
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts_input=shared_experts_input,
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEExpertsModular:
        if prepare_finalize.activation_format == FusedMoEActivationFormat.BatchedExperts:
            raise NotImplementedError(
                "BatchedExperts activation format is not supported for PearlMoE"
            )
        from .pearl_moe_experts import PearlMoEExperts

        return PearlMoEExperts(
            moe_config=self.moe,
            quant_config=self.moe_quant_config,
            layer=layer,
        )
