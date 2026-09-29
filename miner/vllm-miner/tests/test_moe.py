"""Host-side MoE routing and gating contracts (``vllm_miner.moe`` and friends)."""

import pytest
import torch
from miner_base.commitment import Device
from vllm_miner.mining_config import (
    MOE_LOTTERY_N,
    effective_work_per_matmul,
    expert_n,
    mining_configuration,
    select_tile,
)
from vllm_miner.moe import topk_ids_shape_error, validate_moe_dims


@pytest.mark.parametrize("experts,top_k", [(4, 2), (1, 1), (128, 8), (8, 8)])
def test_validate_moe_dims_accepts_positive_ints(experts, top_k):
    validate_moe_dims(experts, top_k)


@pytest.mark.parametrize(
    "experts,top_k",
    [(0, 1), (4, 0), (-4, 2), (4, -1), (True, 1), (4, True), (4.0, 2), (4, 2.0), (2, 4)],
    ids=[
        "no-experts",
        "no-topk",
        "neg-experts",
        "neg-topk",
        "bool-e",
        "bool-k",
        "float-e",
        "float-k",
        "k>e",
    ],
)
def test_validate_moe_dims_rejects_non_dims(experts, top_k):
    with pytest.raises(ValueError):
        validate_moe_dims(experts, top_k)


def test_topk_ids_shape_error_names_the_problem():
    good = torch.zeros(16, 2, dtype=torch.int64)
    assert topk_ids_shape_error(good, 16, 2) is None
    assert topk_ids_shape_error(good.to(torch.int32), 16, 2) is None
    assert "(16, 2)" in topk_ids_shape_error(good[:8], 16, 2)
    assert "(16, 2)" in topk_ids_shape_error(good[:, :1], 16, 2)
    assert "(16, 2)" in topk_ids_shape_error(good.reshape(-1), 16, 2)
    assert "int32/int64" in topk_ids_shape_error(good.float(), 16, 2)
    assert "tensor" in topk_ids_shape_error([[0, 1]] * 16, 16, 2)


@pytest.mark.parametrize("device", list(Device), ids=lambda d: d.name.lower())
def test_moe_configuration_commits_the_expert_local_tile(device):
    """The tile is selected on ``n_e``, and the expert count reaches ``pB``."""
    experts, n_e, k = 8, 3 * MOE_LOTTERY_N, 2048
    config = mining_configuration(k, experts * n_e, experts, device=device)
    dense_e = mining_configuration(k, n_e, device=device)
    assert config.device is device
    assert config.experts == experts and dense_e.experts == 0
    assert config.rows_pattern == dense_e.rows_pattern
    assert config.cols_pattern == dense_e.cols_pattern
    assert config.p_b(experts * n_e)[-2:] == experts.to_bytes(2, "little")
    assert expert_n(experts * n_e, experts) == n_e
    assert expert_n(n_e, 0) == n_e
    with pytest.raises(ValueError, match="multiple"):
        expert_n(experts * n_e + 1, experts)
    # Credit over whole expert-local tiles is the dense formula at n_e.
    tile = select_tile(n_e, k, device=device)
    assert effective_work_per_matmul(4 * tile.rows, n_e, k, device=device) == (
        4 * tile.rows * n_e * k
    )
