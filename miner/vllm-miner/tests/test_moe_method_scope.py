"""``PearlMoEMethod`` scope: which ``FusedMoEConfig`` deployments mine and
which keep the engine's own MoE method (host-only, no CUDA context)."""

from dataclasses import replace

import pytest
import torch
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    MoEActivation,
    RoutingMethodType,
)


def _parallel(*, use_ep: bool) -> FusedMoEParallelConfig:
    return FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=2 if use_ep else 1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=use_ep,
        all2all_backend="naive",
        enable_eplb=False,
    )


def _moe_config(**overrides) -> FusedMoEConfig:
    base = FusedMoEConfig(
        num_experts=64,
        experts_per_token=8,
        hidden_dim=2048,
        intermediate_size=1024,
        num_local_experts=64,
        num_logical_experts=64,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=_parallel(use_ep=False),
        in_dtype=torch.bfloat16,
    )
    return replace(base, **overrides)


def test_bf16_silu_tp_only_layer_is_admitted():
    from vllm_miner.vllm_pearl_config import moe_config_unsupported_reason

    assert moe_config_unsupported_reason(_moe_config()) is None


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"moe_parallel_config": _parallel(use_ep=True)}, "expert parallelism"),
        ({"has_bias": True}, "expert biases"),
        ({"is_lora_enabled": True}, "LoRA experts"),
        ({"swiglu_limit": 7.0}, "swiglu gate parameters"),
        ({"swiglu_alpha": 1.702}, "swiglu gate parameters"),
        ({"activation": MoEActivation.SILU_NO_MUL}, "activation silu_no_mul"),
    ],
    ids=["ep", "bias", "lora", "swiglu-limit", "swiglu-alpha", "non-act-and-mul"],
)
def test_out_of_scope_deployments_are_declined_by_the_reason_predicate(overrides, reason):
    """The reason is decided from the config alone (dispatch is covered below)."""
    from vllm_miner.vllm_pearl_config import moe_config_unsupported_reason

    assert moe_config_unsupported_reason(_moe_config(**overrides)) == reason


class _FakeRoutedExperts:
    """Stands in for ``RoutedExperts`` at the dispatch boundary: the config
    decides, so only ``moe_config`` (and the router-weight flag) are read."""

    def __init__(self, moe_config: FusedMoEConfig) -> None:
        self.moe_config = moe_config
        self.apply_router_weight_on_input = False


def _dispatch(monkeypatch, moe_config: FusedMoEConfig, prefix: str = "model.layers.0.mlp.experts"):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_miner import vllm_pearl_config as vpc

    monkeypatch.setattr(vpc, "RoutedExperts", _FakeRoutedExperts)
    layer = _FakeRoutedExperts(moe_config)
    # ``PearlMoEMethod`` is a vLLM ``CustomOp``: constructing it needs the
    # engine config context, as it would inside ``get_quant_method``.
    with set_current_vllm_config(VllmConfig()):
        return vpc.PearlConfig().get_quant_method(layer, prefix)


def test_ep_deployment_keeps_the_engine_method_and_backend_at_dispatch(monkeypatch):
    """An EP layer on a FlashInfer backend is declined at dispatch:
    ``get_quant_method`` returns None (vLLM keeps its own unquantized method),
    no ``PearlMoEMethod`` is constructed and the backend is untouched."""
    from vllm_miner import vllm_pearl_config as vpc

    constructed = []
    real_method = vpc.PearlMoEMethod

    class RecordingMethod(real_method):
        def __init__(self, moe, prefix):
            constructed.append(prefix)
            super().__init__(moe, prefix)

    monkeypatch.setattr(vpc, "PearlMoEMethod", RecordingMethod)
    moe = _moe_config(moe_parallel_config=_parallel(use_ep=True))
    moe.moe_backend = "flashinfer_cutlass"

    assert _dispatch(monkeypatch, moe) is None
    assert constructed == []
    assert moe.moe_backend == "flashinfer_cutlass"


def test_router_weights_on_the_input_keep_the_engine_method(monkeypatch):
    moe = _moe_config()

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_miner import vllm_pearl_config as vpc

    monkeypatch.setattr(vpc, "RoutedExperts", _FakeRoutedExperts)
    layer = _FakeRoutedExperts(moe)
    layer.apply_router_weight_on_input = True
    with set_current_vllm_config(VllmConfig()):
        assert vpc.PearlConfig().get_quant_method(layer, "model.layers.0.mlp.experts") is None


def test_ignored_mineable_experts_stay_on_triton_pearl_method(monkeypatch):
    """An ignored but mineable expert layer still takes PearlMoEMethod so
    every expert layer of the model runs on the Triton backend."""
    from vllm_miner import vllm_pearl_config as vpc
    from vllm_miner.settings import RuntimeSettings

    settings = RuntimeSettings(ignored_layers=("model.layers.1.mlp.experts",))
    monkeypatch.setattr(vpc, "runtime_settings", lambda: settings)
    moe = _moe_config()
    moe.moe_backend = "flashinfer_cutlass"
    method = _dispatch(monkeypatch, moe, prefix="model.layers.1.mlp.experts")
    assert isinstance(method, vpc.PearlMoEMethod)
    assert moe.moe_backend == "triton"


def test_admitted_deployment_gets_pearl_method_with_triton_backend(monkeypatch):
    """Control for the dispatch test: the admitted BF16 TP-only layer is
    served by ``PearlMoEMethod``, which forces the Triton backend."""
    from vllm_miner import vllm_pearl_config as vpc

    moe = _moe_config()
    moe.moe_backend = "flashinfer_cutlass"

    method = _dispatch(monkeypatch, moe)
    assert isinstance(method, vpc.PearlMoEMethod)
    assert moe.moe_backend == "triton"
