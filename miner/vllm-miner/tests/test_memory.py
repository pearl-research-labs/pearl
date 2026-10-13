from types import SimpleNamespace

import pytest
from pearl_gemm import R
from vllm_miner import memory


def test_peak_reservation_uses_process_wide_completion_bound(monkeypatch):
    state = SimpleNamespace(
        mineable=True,
        n=16,
        k=8,
        experts=0,
        top_k=0,
        weight=SimpleNamespace(device="cuda:0"),
    )
    settings = SimpleNamespace(
        m_buckets=(2,),
        completion_inflight_limit=7,
        warmup_compile=True,
    )
    monkeypatch.setattr(memory, "all_states", lambda: [state])
    monkeypatch.setattr(memory, "runtime_settings", lambda: settings)
    monkeypatch.setattr(memory.gpu_config.settings, "no_mining", False)
    monkeypatch.setattr(memory, "max_mineable_k", lambda: 8)
    monkeypatch.setattr(memory, "_launch_bytes", lambda *_args: 100)
    monkeypatch.setattr(memory, "_FRAGMENTATION_FACTOR", 1.0)
    monkeypatch.setattr(memory, "_RESERVE_ALIGNMENT", 1)

    # signal payload=20, signal lock+payload=24, owned CPU snapshot allowance=20,
    # seven launch leases=700 (warmup runs before any launch and needs no slot),
    # reload transient=2048, signal/owned payload=44, then the fixed one-byte
    # cushion.
    assert memory.estimate_peak_mining_bytes() == 2793


def test_moe_launch_bytes_cover_the_gathered_operands():
    """An MoE launch gathers the noised A operands over ``m * top_k`` rows on
    top of the dense planes: the reservation grows by at least those copies,
    but not by the committed planes, which MoE hits do not snapshot."""
    m, k, experts, top_k = 64, 2048, 8, 2
    extra = memory._moe_launch_bytes(m, k, experts, top_k)
    cum_m = m * top_k
    gathered = cum_m * k + 2 * cum_m * 2 * R + 2 * cum_m  # A', A peel, alpha A
    assert extra >= gathered + 4 * cum_m  # + at least the routing table
    assert extra < gathered + cum_m * k  # no int8 codes copy


@pytest.mark.parametrize(
    "experts,top_k,n_e", [(4, 2, 256), (64, 8, 2048), (128, 6, 1536), (8, 1, 512)]
)
def test_moe_tail_bytes_are_the_adapter_tail(experts, top_k, n_e):
    """The tail is the activation, the ``w2`` output, the weighted rows, the
    slot-order copy and the reduced output over ``cum_m`` rows -- not a dense
    ``(m, E * n_e)``."""
    m, k = 512, 4096
    cum_m = m * top_k
    tail = memory._moe_tail_bytes(m, experts * n_e, k, experts, top_k)
    assert tail == 2 * cum_m * (n_e // 2) + 3 * (2 * cum_m * k) + 6 * cum_m + 2 * m * k
    assert tail < 2 * m * experts * n_e + 6 * cum_m * k + 2 * m * k


def test_peak_reservation_counts_one_moe_tail(monkeypatch):
    """The MoE tail is added once (serving thread), not per launch slot."""
    dense = SimpleNamespace(
        mineable=True, n=16, k=8, experts=0, top_k=0, weight=SimpleNamespace(device="cuda:0")
    )
    moe = SimpleNamespace(
        mineable=True, n=16, k=8, experts=4, top_k=2, weight=SimpleNamespace(device="cuda:0")
    )
    settings = SimpleNamespace(m_buckets=(2,), completion_inflight_limit=7, warmup_compile=True)
    monkeypatch.setattr(memory, "runtime_settings", lambda: settings)
    monkeypatch.setattr(memory.gpu_config.settings, "no_mining", False)
    monkeypatch.setattr(memory, "max_mineable_k", lambda: 8)
    monkeypatch.setattr(memory, "_launch_bytes", lambda *_args: 100)
    monkeypatch.setattr(memory, "_moe_tail_bytes", lambda *_args: 1000)
    monkeypatch.setattr(memory, "_FRAGMENTATION_FACTOR", 1.0)
    monkeypatch.setattr(memory, "_RESERVE_ALIGNMENT", 1)

    monkeypatch.setattr(memory, "all_states", lambda: [dense])
    without = memory.estimate_peak_mining_bytes()
    monkeypatch.setattr(memory, "all_states", lambda: [dense, moe])
    assert memory.estimate_peak_mining_bytes() == without + 1000


def test_peak_reservation_is_zero_when_mining_is_disabled(monkeypatch):
    monkeypatch.setattr(memory.gpu_config.settings, "no_mining", True)
    monkeypatch.setattr(
        memory,
        "all_states",
        lambda: [SimpleNamespace(mineable=True)],
    )

    assert memory.estimate_peak_mining_bytes() == 0
