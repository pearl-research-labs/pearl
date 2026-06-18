import pytest
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
    global_min_m = pearl_config.matrix_multiplication_config["use_simplified_gemm"]["min_m"]
    assert _expert_local_min_m() == global_min_m


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


def test_expert_mining_mask_uses_expert_local_min_m(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_EXPERT_LOCAL_MIN_M_ENV, "1536")
    layout = _FakeRoutingLayout([1024, 1535, 1536, 2048])

    assert PearlMoEExperts._expert_mining_mask(layout, n=1024, k=1024) == [
        False,
        False,
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
