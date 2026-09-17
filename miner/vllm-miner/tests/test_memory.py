from types import SimpleNamespace

from vllm_miner import memory


def test_peak_reservation_uses_process_wide_completion_bound(monkeypatch):
    state = SimpleNamespace(
        mineable=True,
        n=16,
        k=8,
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


def test_peak_reservation_is_zero_when_mining_is_disabled(monkeypatch):
    monkeypatch.setattr(memory.gpu_config.settings, "no_mining", True)
    monkeypatch.setattr(
        memory,
        "all_states",
        lambda: [SimpleNamespace(mineable=True)],
    )

    assert memory.estimate_peak_mining_bytes() == 0
