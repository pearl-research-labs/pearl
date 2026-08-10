from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import vllm_miner.pearl_moe_experts as moe_experts
from vllm_miner.config import config as pearl_config
from vllm_miner.pearl_moe_experts import (
    _EXPERT_LOCAL_MIN_M_ENV,
    _EXPERT_LOCAL_MINING_ENV,
    PearlMoEExperts,
    _expert_local_min_m,
    _use_expert_local_mining_threshold,
)


class _FakeRoutingLayout:
    def __init__(self, expert_counts: list[int]) -> None:
        self._expert_counts = expert_counts
        self.num_experts = len(expert_counts)

    def expert_slice(self, expert_index: int) -> tuple[int, int]:
        return sum(self._expert_counts[:expert_index]), self._expert_counts[expert_index]


def _global_min_m() -> int:
    return int(pearl_config.matrix_multiplication_config["use_simplified_gemm"]["min_m"])


def test_expert_local_mining_threshold_defaults_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_EXPERT_LOCAL_MINING_ENV, raising=False)
    assert _use_expert_local_mining_threshold() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_expert_local_mining_threshold_can_opt_out(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(_EXPERT_LOCAL_MINING_ENV, value)
    assert _use_expert_local_mining_threshold() is False


def test_expert_local_min_m_defaults_to_global_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_EXPERT_LOCAL_MIN_M_ENV, raising=False)
    assert _expert_local_min_m() == _global_min_m()


def test_expert_local_min_m_accepts_tile_aligned_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_EXPERT_LOCAL_MIN_M_ENV, "1536")
    assert _expert_local_min_m() == 1536


@pytest.mark.parametrize("value", ["1023", "1153", "not-an-int"])
def test_expert_local_min_m_rejects_invalid_override(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(_EXPERT_LOCAL_MIN_M_ENV, value)
    with pytest.raises(ValueError, match=_EXPERT_LOCAL_MIN_M_ENV):
        _expert_local_min_m()


@pytest.mark.parametrize(
    ("expert_counts", "expected"),
    [
        pytest.param([0], [False], id="inactive"),
        pytest.param([1023], [False], id="below-threshold"),
        pytest.param([1024], [True], id="exactly-threshold"),
        pytest.param([1025], [True], id="above-threshold"),
        pytest.param([0, 1023, 1024, 2048], [False, False, True, True], id="mixed"),
        pytest.param([1, 1023, 64], [False, False, False], id="all-cold"),
    ],
)
def test_expert_mining_mask_gate_cases(
    monkeypatch: pytest.MonkeyPatch,
    expert_counts: list[int],
    expected: list[bool],
) -> None:
    monkeypatch.setenv(_EXPERT_LOCAL_MIN_M_ENV, str(_global_min_m()))
    layout = _FakeRoutingLayout(expert_counts)

    assert PearlMoEExperts._expert_mining_mask(layout, n=1024, k=1024) == expected


def test_forward_mining_mask_can_mine_every_active_expert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_EXPERT_LOCAL_MINING_ENV, "0")
    layout = _FakeRoutingLayout([0, 1, 1023, 1024])

    assert PearlMoEExperts._forward_mining_mask(layout, n=1024, k=1024) == [
        False,
        True,
        True,
        True,
    ]


def test_expert_mining_mask_keeps_global_nk_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_EXPERT_LOCAL_MIN_M_ENV, "1024")
    layout = _FakeRoutingLayout([2048, 2048])

    assert PearlMoEExperts._expert_mining_mask(layout, n=1023, k=1024) == [False, False]
    assert PearlMoEExperts._expert_mining_mask(layout, n=1024, k=1023) == [False, False]


def test_all_cold_apply_uses_grouped_vanilla_without_preparing_noising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_EXPERT_LOCAL_MINING_ENV, raising=False)
    monkeypatch.setenv(_EXPERT_LOCAL_MIN_M_ENV, str(_global_min_m()))
    layout = _FakeRoutingLayout([_global_min_m() - 1, 0])

    grouped_gemm = Mock()
    prepare_noising = Mock(side_effect=AssertionError("all-cold gate prepared MoE noising"))
    apply_per_expert = Mock(side_effect=AssertionError("all-cold gate used per-expert GEMM"))

    monkeypatch.setattr(
        moe_experts,
        "get_async_manager",
        lambda: SimpleNamespace(_conf=SimpleNamespace(no_mining=False)),
    )
    monkeypatch.setattr(moe_experts.pearl_config, "should_use_noisy_gemm", lambda m, n, k: True)
    monkeypatch.setattr(moe_experts, "build_moe_routing_layout", lambda topk_ids, e: layout)
    monkeypatch.setattr(moe_experts, "prepare_moe_noising", prepare_noising)
    monkeypatch.setattr(moe_experts, "invoke_fused_moe_triton_kernel", grouped_gemm)
    monkeypatch.setattr(
        moe_experts,
        "try_get_optimal_moe_config",
        lambda *args, **kwargs: {"BLOCK_SIZE_M": 128},
    )
    monkeypatch.setattr(
        moe_experts,
        "moe_align_block_size",
        lambda *args, **kwargs: (
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
        ),
    )

    fake_experts = SimpleNamespace(
        moe_problem_size=lambda hidden_states, w1, w2, topk_ids: (
            layout.num_experts,
            1,
            1024,
            1024,
            1,
        ),
        _quant_a_7bit=lambda hidden_states, k: (
            torch.zeros((1, k), dtype=torch.int8),
            torch.ones((1, 1), dtype=torch.float32),
            None,
        ),
        adjust_N_for_activation=lambda n, activation: n,
        _forward_mining_mask=PearlMoEExperts._forward_mining_mask,
        _apply_per_expert_gemm1=apply_per_expert,
        _int8_w8a8_triton_kwargs=lambda: {},
        activation=lambda activation, intermediate_cache2, intermediate_cache1: None,
        _gemm2_triton=lambda **kwargs: None,
        w1_scale=torch.ones((layout.num_experts, 1024, 1), dtype=torch.float32),
        w2_scale=torch.ones((layout.num_experts, 8, 8), dtype=torch.float32),
        per_act_token_quant=True,
    )

    PearlMoEExperts.apply(
        fake_experts,
        output=torch.empty((1, 1024), dtype=torch.float32),
        hidden_states=torch.empty((1, 1024), dtype=torch.float32),
        w1=torch.empty((layout.num_experts, 1024, 1024), dtype=torch.int8),
        w2=torch.empty((layout.num_experts, 1024, 1024), dtype=torch.float32),
        topk_weights=torch.ones((1, 1), dtype=torch.float32),
        topk_ids=torch.zeros((1, 1), dtype=torch.int32),
        activation=None,
        global_num_experts=layout.num_experts,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=torch.empty((1, 1, 1024), dtype=torch.float32),
        workspace2=torch.empty((1, 1, 1024), dtype=torch.float32),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    grouped_gemm.assert_called_once()
    apply_per_expert.assert_not_called()
    prepare_noising.assert_not_called()
