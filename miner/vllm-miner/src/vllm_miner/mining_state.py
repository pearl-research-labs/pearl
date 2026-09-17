"""Process-global Torch/CUDA mining runtime state."""

import threading
import time

from miner_base.async_loop_manager import AsyncLoopManager
from miner_base.settings import MinerSettings
from miner_utils import get_logger
from pearl_gateway.config import MinerRpcConfig

from .capture import (
    close_mining_admission,
    mining_gpu_work_idle,
    open_mining_admission,
)
from .config import config

_LOGGER = get_logger(__name__)

_async_manager: AsyncLoopManager | None = None
_runtime_poisoned = False
_runtime_startup_disabled = False
_RUNTIME_LIFECYCLE_LOCK = threading.RLock()
_GPU_RUNTIME_DRAIN_TIMEOUT_SECONDS = 60.0


def get_async_manager() -> AsyncLoopManager:
    if _runtime_poisoned:
        raise RuntimeError("GPU runtime is poisoned after an incomplete teardown")
    if _async_manager is None:
        raise RuntimeError("Async Loop Manager has not been initialized yet")
    return _async_manager


def try_get_async_manager() -> AsyncLoopManager | None:
    """Return the manager, or ``None`` before serving-runtime initialization."""
    if _runtime_poisoned:
        raise RuntimeError("GPU runtime is poisoned after an incomplete teardown")
    return _async_manager


def mining_disabled_by_runtime_failure() -> bool:
    """Whether safe startup rollback permanently disabled mining in this worker."""
    return _runtime_startup_disabled


def disable_mining_after_runtime_failure(reason: str) -> None:
    """Permanently fail this process to serving-only after safe rollback.

    Callers must invoke this only after all partial mining owners were stopped or
    proven absent. It deliberately does not alter framework-owned CUDA state.
    """
    global _runtime_startup_disabled
    with _RUNTIME_LIFECYCLE_LOCK:
        _runtime_startup_disabled = True
        config.settings = config.settings.model_copy(update={"no_mining": True})
        close_mining_admission()
    _LOGGER.error(f"{reason}; process-local mining is disabled while serving continues")


def init_async_manager(miner_settings: MinerSettings | None = None) -> None:
    """Initialize the reusable serving runtime after a framework binds CUDA."""
    with _RUNTIME_LIFECYCLE_LOCK:
        _init_async_manager_locked(miner_settings)


def _init_async_manager_locked(miner_settings: MinerSettings | None) -> None:
    global _async_manager, _runtime_startup_disabled

    if _runtime_poisoned:
        raise RuntimeError("GPU runtime cannot restart after an incomplete teardown")
    if _runtime_startup_disabled:
        return
    if _async_manager is not None and _async_manager.started:
        return

    settings = miner_settings if miner_settings is not None else MinerSettings()
    manager = AsyncLoopManager(
        # MinerRpcConfig remains environment-backed: transport/host/port take
        # MINER_RPC_* values while this socket path is the intentional YAML
        # fallback for local UDS deployments.
        MinerRpcConfig(socket_path=config.gateway_socket_path),
        settings,
        startup_fail_open=True,
    )
    from .pipeline import fallback_winner_checks_idle

    manager.register_drain_predicate(mining_gpu_work_idle)
    manager.register_drain_predicate(fallback_winner_checks_idle)
    manager.register_drain_barrier(close_mining_admission)
    manager.register_drain_release(open_mining_admission)

    # Keep process-global producer admission closed until manager.start() has
    # finished starting the async loop.
    close_mining_admission()
    manager.start()
    _async_manager = manager
    config.settings = settings
    # Reopen producer admission only after the new manager is running.
    open_mining_admission()
    _LOGGER.info(f"Mining state initialized, {settings=}")


def delete_state() -> None:
    """Tear down the process-global GPU runtime in dependency order."""
    with _RUNTIME_LIFECYCLE_LOCK:
        _delete_state_locked()


def _gpu_runtime_safe_to_release(timeout: float) -> tuple[bool, str | None]:
    """Wait only for process-local CUDA and host-continuation memory safety."""
    from .capture import wait_for_mining_producers_idle
    from .pipeline import wait_for_mining_continuations_idle

    deadline = time.monotonic() + timeout
    if not wait_for_mining_producers_idle(timeout):
        return False, "mining GPU work did not quiesce during runtime teardown"
    remaining = max(0.0, deadline - time.monotonic())
    if not wait_for_mining_continuations_idle(remaining):
        return False, "mining host continuations did not quiesce during runtime teardown"
    return True, None


def _stop_manager_after_gpu_teardown(
    manager: AsyncLoopManager | None,
    failure: BaseException | None,
) -> None:
    if manager is None:
        return
    try:
        # Shutdown does not promise final hash accounting or proof delivery.
        manager.stop(wait_for_submissions=False)
    except BaseException:
        if failure is None:
            raise
        _LOGGER.opt(exception=True).error(
            "Async manager teardown also failed after GPU runtime failure"
        )


def _delete_state_locked() -> None:
    """Teardown after serializing against initialization and other owners.

    Producer admission closes first, then mining GPU work and host
    continuations drain, then every layer forgets its job. A timed-out runtime
    is poisoned and retains its contexts rather than exposing live CUDA memory
    to unsafe in-process reuse.
    """
    global _async_manager, _runtime_poisoned

    from .job_prep import unpublish_contexts

    manager = _async_manager
    if manager is not None:
        # Fence reusable-flush release before closing the process-global GPU
        # gate. Otherwise a flush already in progress could reopen producers
        # between this close and the eventual manager.stop(). Teardown permits
        # abandoning proofs, so close proof admission and wake RPCs immediately.
        manager.abort_pending_submissions()
    close_mining_admission()
    failure: BaseException | None = None
    try:
        if manager is not None:
            safe, error = _gpu_runtime_safe_to_release(_GPU_RUNTIME_DRAIN_TIMEOUT_SECONDS)
            if not safe:
                raise RuntimeError(error)
        unpublish_contexts()
    except BaseException as exc:
        _runtime_poisoned = True
        failure = exc

    try:
        _stop_manager_after_gpu_teardown(manager, failure)
    except BaseException:
        _runtime_poisoned = True
        raise
    if failure is not None:
        raise failure
    _async_manager = None
