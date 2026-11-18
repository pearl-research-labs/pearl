"""Portable scheduler/cache boundary tests for standalone vLLM reload."""

from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from vllm_miner.engine_lifecycle import _make_collective_rpc_wrapper


def _engine(*, paused: bool, sleeping: bool = False, pause_result=None):
    events = []
    engine = SimpleNamespace(model_executor=SimpleNamespace(is_sleeping=sleeping))
    engine.is_scheduler_paused = lambda: paused
    engine.pause_scheduler = lambda **kwargs: events.append(("pause", kwargs)) or pause_result
    engine.resume_scheduler = lambda: events.append("resume")
    return engine, events


def test_idle_standalone_reload_pauses_clears_cache_and_resumes():
    engine, events = _engine(paused=False)

    def original(_self, method, timeout, args, kwargs):
        events.append(("rpc", method, timeout, args, kwargs))
        return ["ok"]

    wrapped = _make_collective_rpc_wrapper(original)

    assert wrapped(engine, "reload_weights", 9.0, (1,), {"x": 2}) == ["ok"]
    assert events == [
        ("pause", {"mode": "abort", "clear_cache": True}),
        ("rpc", "reload_weights", 9.0, (1,), {"x": 2}),
        "resume",
    ]


def test_active_reload_waits_asynchronously_for_engine_idle():
    pause_future = Future()
    engine, events = _engine(paused=False, pause_result=pause_future)
    wrapped = _make_collective_rpc_wrapper(lambda *_args: events.append("rpc") or ["ok"])

    reload_result = wrapped(engine, "reload_weights")
    assert isinstance(reload_result, Future)
    assert not reload_result.done()
    assert not reload_result.cancel()
    assert not reload_result.cancelled()
    assert "rpc" not in events
    with pytest.raises(RuntimeError, match="already waiting"):
        wrapped(engine, "reload_weights")

    pause_future.set_result(None)

    assert reload_result.result() == ["ok"]
    assert events[-2:] == ["rpc", "resume"]
    assert not hasattr(engine, "_pearl_reload_future")


def test_deferred_reload_failure_leaves_scheduler_paused_and_latched():
    pause_future = Future()
    engine, events = _engine(paused=False, pause_result=pause_future)

    def fail(*_args):
        raise RuntimeError("reload failed")

    reload_result = _make_collective_rpc_wrapper(fail)(engine, "reload_weights")
    pause_future.set_result(None)

    with pytest.raises(RuntimeError, match="reload failed"):
        reload_result.result()
    assert "resume" not in events
    assert engine._pearl_reload_future is reload_result


def test_reload_failure_leaves_scheduler_paused():
    engine, events = _engine(paused=False)

    def fail(*_args):
        raise RuntimeError("reload failed")

    wrapped = _make_collective_rpc_wrapper(fail)
    with pytest.raises(RuntimeError, match="reload failed"):
        wrapped(engine, "reload_weights")

    assert "resume" not in events


def test_level_zero_pause_is_cleared_but_preserved_after_reload():
    engine, events = _engine(paused=True)
    wrapped = _make_collective_rpc_wrapper(lambda *_args: events.append("rpc") or ["ok"])

    assert wrapped(engine, "reload_weights") == ["ok"]
    assert events == [
        ("pause", {"mode": "abort", "clear_cache": True}),
        "rpc",
    ]


def test_deep_sleep_reload_preserves_external_scheduler_pause():
    engine, events = _engine(paused=True, sleeping=True)
    wrapped = _make_collective_rpc_wrapper(lambda *_args: ["ok"])

    assert wrapped(engine, "reload_weights") == ["ok"]
    assert events == []


def test_non_reload_rpc_is_untouched():
    engine, events = _engine(paused=False)
    wrapped = _make_collective_rpc_wrapper(
        lambda _self, method, timeout, args, kwargs: (method, timeout, args, kwargs)
    )

    assert wrapped(engine, "health", 1.0, (), None) == ("health", 1.0, (), None)
    assert events == []
