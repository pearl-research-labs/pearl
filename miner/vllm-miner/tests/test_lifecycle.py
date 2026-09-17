"""Portable ownership tests for vLLM reload and sleep/offload hooks."""

import pytest
from vllm_miner import lifecycle as lifecycle_module


@pytest.fixture
def owner(monkeypatch):
    events = []
    pointers = {(1, "weight"): 101}
    monkeypatch.setattr(lifecycle_module, "_state_pointers", lambda: dict(pointers))
    monkeypatch.setattr(
        lifecycle_module,
        "_synchronize_weight_streams",
        lambda: events.append("sync-weight-streams"),
    )
    monkeypatch.setattr(
        lifecycle_module,
        "suspend_mining_producers_until_resumed",
        lambda reason: events.append(("suspend", reason)) or 17,
    )
    monkeypatch.setattr(
        lifecycle_module,
        "resume_mining_producers",
        lambda token: events.append(("resume", token)),
    )
    monkeypatch.setattr(lifecycle_module, "wait_for_mining_producers_idle", lambda _t: True)
    monkeypatch.setattr(lifecycle_module, "unpublish_contexts", lambda: events.append("unpublish"))
    return lifecycle_module.PearlVllmLifecycle(), events, pointers


def test_quiesce_waits_for_accounting_and_winner_continuations(owner, monkeypatch):
    lifecycle, events, _pointers = owner
    monkeypatch.setattr(
        "vllm_miner.pipeline.wait_for_mining_continuations_idle",
        lambda _timeout: False,
    )

    with pytest.raises(TimeoutError, match="accounting/winner continuations"):
        lifecycle.before_sleep(1)

    assert "unpublish" not in events, "contexts were dropped before GPU work quiesced"
    lifecycle.reset_for_tests()


def test_level_one_sleep_holds_gpu_gate_until_all_tags_wake(owner):
    lifecycle, events, _pointers = owner

    lifecycle.before_sleep(1)
    lifecycle.after_wake(["weights"])
    assert not any(event == ("resume", 17) for event in events)
    lifecycle.after_wake(["kv_cache"])

    assert events[:2] == [("suspend", "sleep(level=1)"), "unpublish"]
    assert ("resume", 17) in events


def test_reload_stream_fence_precedes_mining_admission(owner):
    lifecycle, events, _pointers = owner

    assert lifecycle.before_reload(True)
    lifecycle.after_reload(owns_interval=True)

    assert events.index("unpublish") < events.index("sync-weight-streams")
    assert events.index("sync-weight-streams") < events.index(("resume", 17))


def test_level_two_sleep_requires_reload_before_final_wake(owner):
    lifecycle, events, _pointers = owner

    lifecycle.before_sleep(2)
    lifecycle.after_wake(["weights"])
    assert lifecycle.before_reload(True) is False
    lifecycle.after_reload(owns_interval=False)
    lifecycle.after_wake(["kv_cache"])

    assert ("resume", 17) in events


def test_level_two_final_wake_without_reload_stays_failed_and_suspended(owner):
    lifecycle, events, _pointers = owner

    lifecycle.before_sleep(2)
    lifecycle.after_wake(["weights"])
    with pytest.raises(RuntimeError, match="requires checkpoint reload_weights"):
        lifecycle.after_wake(["kv_cache"])

    assert ("resume", 17) not in events
    lifecycle.reset_for_tests()


def test_standalone_checkpoint_reload_owns_one_reversible_interval(owner):
    lifecycle, events, _pointers = owner

    assert lifecycle.before_reload(True) is True
    lifecycle.after_reload(owns_interval=True)

    assert ("suspend", "reload_weights") in events
    assert ("resume", 17) in events


def test_kernel_format_reload_is_rejected_before_quiescence(owner):
    lifecycle, events, _pointers = owner

    with pytest.raises(ValueError, match="checkpoint-format"):
        lifecycle.before_reload(False)

    assert not any(isinstance(event, tuple) and event[0] == "suspend" for event in events)


def test_pointer_change_fails_closed(owner):
    lifecycle, events, pointers = owner
    lifecycle.before_sleep(1)
    pointers[(1, "weight")] = 202

    with pytest.raises(RuntimeError, match="changed Pearl graph-visible storage"):
        lifecycle.after_wake(["weights"])

    assert ("resume", 17) not in events
    lifecycle.reset_for_tests()
