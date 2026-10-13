"""Engine-side scheduler/cache boundary for standalone checkpoint reload."""

from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

from miner_utils import get_logger

from .hooks import wrap_callable_once

_LOGGER = get_logger("vllm.pearl_miner.engine_lifecycle")


class _ReloadFuture(Future[object]):
    """Server-owned reload work cannot be cancelled by a departing RPC waiter."""

    def cancel(self) -> bool:
        return False


def _is_reload_method(method: str | Callable[..., Any]) -> bool:
    return method == "reload_weights" or getattr(method, "__name__", None) == "reload_weights"


def _run_reload(
    self: object,
    original: object,
    method: str | Callable[..., Any],
    timeout: float | None,
    args: tuple,
    kwargs: dict[str, Any] | None,
    *,
    resume_after: bool,
) -> object:
    try:
        result = original(self, method, timeout, args, kwargs)
        if resume_after:
            self.resume_scheduler()
    except BaseException:
        _LOGGER.exception(
            "Checkpoint reload failed; vLLM scheduling remains paused and the "
            "worker must be restarted."
        )
        raise
    return result


def _defer_reload_until_paused(
    self: object,
    pause_future: Future[object],
    run_reload: Callable[[], object],
) -> Future[object]:
    reload_future: Future[object] = _ReloadFuture()
    self._pearl_reload_future = reload_future

    def paused(future: Future[object]) -> None:
        try:
            future.result()
            reload_future.set_result(run_reload())
        except BaseException as exc:
            reload_future.set_exception(exc)
        else:
            delattr(self, "_pearl_reload_future")

    pause_future.add_done_callback(paused)
    return reload_future


def _reload_with_boundary(
    self: object,
    original: object,
    method: str | Callable[..., Any],
    timeout: float | None,
    args: tuple,
    kwargs: dict[str, Any] | None,
) -> object:
    if getattr(self, "_pearl_reload_future", None) is not None:
        raise RuntimeError("a checkpoint reload is already waiting for vLLM to become idle")

    scheduler_was_paused = self.is_scheduler_paused()
    executor_sleeping = bool(getattr(getattr(self, "model_executor", None), "is_sleeping", False))

    def run_reload() -> object:
        return _run_reload(
            self,
            original,
            method,
            timeout,
            args,
            kwargs,
            resume_after=not scheduler_was_paused,
        )

    if scheduler_was_paused and executor_sleeping:
        # EngineCore.sleep already owns scheduler/request/KV quiescence.
        return run_reload()

    # An externally level-0-paused scheduler can still own live requests and
    # caches. Abort/clear it as well, but preserve its paused state.
    pause_future = self.pause_scheduler(mode="abort", clear_cache=True)
    if pause_future is None:
        return run_reload()

    # EngineCore utility dispatch understands Future results and keeps its RPC
    # pending while the busy loop drains the last model-executor step. Running
    # the worker RPC from that callback avoids blocking the thread that makes
    # the pause future ready.
    return _defer_reload_until_paused(self, pause_future, run_reload)


def _make_collective_rpc_wrapper(original: object) -> object:
    def collective_rpc_with_reload_boundary(
        self: object,
        method: str | Callable[..., Any],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> object:
        if _is_reload_method(method):
            return _reload_with_boundary(self, original, method, timeout, args, kwargs)
        return original(self, method, timeout, args, kwargs)

    return collective_rpc_with_reload_boundary


def install_engine_reload_boundary() -> None:
    """Patch the vLLM 0.28 EngineCore in every process, without touching CUDA."""
    from vllm.v1.engine.core import EngineCore

    wrap_callable_once(EngineCore, "collective_rpc", _make_collective_rpc_wrapper)
