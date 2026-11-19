"""Process-global CUDA mining runtime: AsyncLoopManager lifecycle and vLLM worker hooks.

Kept separate from ``plugin.py`` so importing the plugin stays CUDA-free until
vLLM binds each worker's CUDA device.
"""

import threading
import time
from collections.abc import Callable, Iterator

from miner_base.async_loop_manager import AsyncLoopManager
from miner_base.settings import MinerSettings
from miner_utils import get_logger
from pearl_gateway.config import MinerRpcConfig

from .capture import (
    close_mining_admission,
    mining_gpu_work_idle,
    mining_suspended_for_capture,
    open_mining_admission,
    resume_submissions_after_capture,
    suspend_mining_launches,
)
from .plugin import wrap_callable_once
from .settings import miner_settings, set_miner_settings

_LOGGER = get_logger("vllm.pearl_miner")

_init_lock = threading.Lock()
_lifecycle_lock = threading.RLock()
_async_manager: AsyncLoopManager | None = None
_runtime_poisoned = False
_runtime_startup_disabled = False
_GPU_RUNTIME_DRAIN_TIMEOUT_SECONDS = 60.0
_CUTLASS_COMPAT_ATTR = "_vllm_miner_cutlass46_compat_applied"


# ---------------------------------------------------------------------------
# Process-global AsyncLoopManager lifecycle
# ---------------------------------------------------------------------------


def get_async_manager() -> AsyncLoopManager:
    if _runtime_poisoned:
        raise RuntimeError("GPU runtime is poisoned after an incomplete teardown")
    if _async_manager is None:
        raise RuntimeError("Async Loop Manager has not been initialized yet")
    return _async_manager


def try_get_async_manager() -> AsyncLoopManager | None:
    if _runtime_poisoned:
        raise RuntimeError("GPU runtime is poisoned after an incomplete teardown")
    return _async_manager


def mining_disabled_by_runtime_failure() -> bool:
    return _runtime_startup_disabled


def disable_mining_after_runtime_failure(reason: str) -> None:
    global _runtime_startup_disabled
    with _lifecycle_lock:
        _runtime_startup_disabled = True
        set_miner_settings(miner_settings().model_copy(update={"no_mining": True}))
        close_mining_admission()
    _LOGGER.error(f"{reason}; process-local mining is disabled while serving continues")


def init_async_manager(settings: MinerSettings | None = None) -> None:
    global _async_manager
    with _lifecycle_lock:
        if _runtime_poisoned:
            raise RuntimeError("GPU runtime cannot restart after an incomplete teardown")
        if _runtime_startup_disabled:
            return
        if _async_manager is not None and _async_manager.started:
            return

        resolved = settings if settings is not None else MinerSettings()
        manager = AsyncLoopManager(MinerRpcConfig(), resolved, startup_fail_open=True)
        from .pipeline import completions_idle

        manager.register_drain_predicate(mining_gpu_work_idle)
        manager.register_drain_predicate(completions_idle)
        manager.register_drain_barrier(close_mining_admission)
        manager.register_drain_release(open_mining_admission)

        close_mining_admission()
        manager.start()
        _async_manager = manager
        set_miner_settings(resolved)
        open_mining_admission()
        _LOGGER.info(f"Mining state initialized, settings={resolved}")


def _gpu_runtime_safe_to_release(timeout: float) -> tuple[bool, str | None]:
    from .capture import wait_for_mining_producers_idle
    from .pipeline import wait_for_mining_continuations_idle

    deadline = time.monotonic() + timeout
    if not wait_for_mining_producers_idle(timeout):
        return False, "mining GPU work did not quiesce during runtime teardown"
    remaining = max(0.0, deadline - time.monotonic())
    if not wait_for_mining_continuations_idle(remaining):
        return False, "mining host continuations did not quiesce during runtime teardown"
    return True, None


def delete_state() -> None:
    """Tear down the process-global GPU runtime in dependency order."""
    global _async_manager, _runtime_poisoned
    with _lifecycle_lock:
        from .cpu_b import stop_cpu_b_service
        from .job_prep import unpublish_contexts

        manager = _async_manager
        if manager is not None:
            manager.abort_pending_submissions()
        close_mining_admission()
        failure: BaseException | None = None
        try:
            if manager is not None:
                safe, error = _gpu_runtime_safe_to_release(_GPU_RUNTIME_DRAIN_TIMEOUT_SECONDS)
                if not safe:
                    raise RuntimeError(error)
            stop_cpu_b_service()
            unpublish_contexts()
        except BaseException as exc:
            _runtime_poisoned = True
            failure = exc

        if manager is not None:
            try:
                manager.stop(wait_for_submissions=False)
            except BaseException:
                _runtime_poisoned = True
                if failure is None:
                    raise
                _LOGGER.opt(exception=True).error(
                    "Async manager teardown also failed after GPU runtime failure"
                )
        if failure is not None:
            raise failure
        _async_manager = None


# ---------------------------------------------------------------------------
# CUTLASS 4.6 compatibility & vLLM worker/runner hooks
# ---------------------------------------------------------------------------


def apply_cutlass_46_compat() -> None:
    """Restore ``cute.make_fragment`` alias removed in cutlass-dsl 4.6 if wheels still call it."""
    try:
        import cutlass.cute as cute
    except ImportError:
        return
    if getattr(cute, _CUTLASS_COMPAT_ATTR, False):
        return
    if not hasattr(cute, "make_fragment") and hasattr(cute, "make_rmem_tensor"):
        cute.make_fragment = cute.make_rmem_tensor
        _LOGGER.info("Applied cutlass-dsl 4.6 alias: cute.make_fragment")
    setattr(cute, _CUTLASS_COMPAT_ATTR, True)


def ensure_pearl_runtime_initialized() -> None:
    """Boot the mining runtime and vLLM hooks after the worker selects its CUDA device."""
    with _init_lock:
        apply_cutlass_46_compat()
        # Register pearl::apply_linear before model load for vLLM 0.28 AOT compile cache hits.
        from . import linear_op  # noqa: F401

        init_async_manager()
        _install_capture_phase_hook()
        _install_shutdown_hook()
        _install_serving_activity_hook()


def rollback_pearl_runtime_startup() -> bool:
    from .state import clear_state_registry

    try:
        if try_get_async_manager() is not None:
            delete_state()
        clear_state_registry()
    except BaseException:
        _LOGGER.opt(exception=True).error("Pearl runtime startup rollback failed")
        return False
    return True


def iter_gpu_model_runner_classes() -> Iterator[type[object]]:
    """Yield importable vLLM GPU model runner classes (V1 legacy and V2/MRv2)."""
    for module_path in (
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker.gpu.model_runner",
    ):
        try:
            mod = __import__(module_path, fromlist=["GPUModelRunner"])
            yield mod.GPUModelRunner
        except ImportError:
            _LOGGER.opt(exception=True).warning(
                f"Could not import GPUModelRunner from {module_path}."
            )


def _install_on_gpu_model_runners(install: Callable[[type[object]], None]) -> bool:
    installed = False
    for runner_cls in iter_gpu_model_runner_classes():
        install(runner_cls)
        installed = True
    return installed


def _make_capture_model_wrapper(original: object) -> object:
    def capture_model_no_mining(self: object, *args: object, **kwargs: object) -> object:
        captured = False
        try:
            with mining_suspended_for_capture("capture_model"):
                result = original(self, *args, **kwargs)
                captured = True
                return result
        finally:
            if captured:
                try:
                    resume_submissions_after_capture()
                except Exception:
                    _LOGGER.opt(exception=True).warning(
                        "Failed to resume winner checks after CUDA-graph capture."
                    )

    return capture_model_no_mining


def _make_memory_profile_wrapper(
    entry_point: str, *, reserve_mining_memory: bool = False
) -> Callable[[object], object]:
    def make_wrapper(original: object) -> object:
        def profile_no_mining(self: object, *args: object, **kwargs: object) -> object:
            with mining_suspended_for_capture(entry_point):
                result = original(self, *args, **kwargs)
                if reserve_mining_memory:
                    from .memory import estimate_peak_mining_bytes

                    reserve = estimate_peak_mining_bytes()
                    if reserve:
                        available = int(result)
                        result = max(0, available - reserve)
                        _LOGGER.info(
                            f"Reserved {reserve / 1024**3:.2f} GiB for bounded "
                            f"Pearl mining work; KV availability {available} -> {result} bytes"
                        )
                return result

        return profile_no_mining

    return make_wrapper


def _prepare_mining() -> bool:
    from .cpu_b import start_cpu_b_service
    from .job_prep import current_job, prepare_mining

    try:
        ready = prepare_mining(current_job())
    except Exception:
        _LOGGER.opt(exception=True).warning(
            "Initial mining preparation failed; serving continues on the FP8 fallback."
        )
        return False
    manager = try_get_async_manager()
    if manager is not None:
        try:
            start_cpu_b_service(manager)
        except Exception:
            _LOGGER.opt(exception=True).warning(
                "CPU B preparation failed to start; B stays on inline GPU preparation."
            )
    return ready


def _make_worker_warmup_wrapper(original: object) -> object:
    def warmup_no_mining(self: object, *args: object, **kwargs: object) -> object:
        with suspend_mining_launches():
            result = original(self, *args, **kwargs)
            _prepare_mining()
            return result

    return warmup_no_mining


def _install_capture_phase_hook() -> None:
    def install(runner_cls: type[object]) -> None:
        wrap_callable_once(runner_cls, "capture_model", _make_capture_model_wrapper)
        for entry_point in (
            "profile_cudagraph_memory",
            "determine_available_memory",
            "profile_run",
        ):
            wrap_callable_once(
                runner_cls,
                entry_point,
                _make_memory_profile_wrapper(entry_point),
                require=False,
            )

    if not _install_on_gpu_model_runners(install):
        _LOGGER.warning("No GPUModelRunner class importable; CUDA graph hooks not installed.")

    try:
        from vllm.v1.worker.gpu_worker import Worker as gpu_worker_cls
    except ImportError:
        _LOGGER.opt(exception=True).warning(
            "Could not import the GPU worker; initialization forwards may mine."
        )
        return
    wrap_callable_once(
        gpu_worker_cls,
        "determine_available_memory",
        _make_memory_profile_wrapper("determine_available_memory", reserve_mining_memory=True),
    )
    wrap_callable_once(gpu_worker_cls, "compile_or_warm_up_model", _make_worker_warmup_wrapper)


def _teardown_worker_mining() -> BaseException | None:
    from .state import clear_state_registry

    error: BaseException | None = None
    for step, label in (
        (close_mining_admission, "mining-admission closure"),
        (delete_state, "process-local runtime teardown"),
        (clear_state_registry, "state-registry cleanup"),
    ):
        try:
            step()
        except BaseException as exc:
            if error is None:
                error = exc
            else:
                _LOGGER.opt(exception=exc).error(f"Pearl {label} also failed")
    return error


def _make_worker_shutdown_wrapper(original: object) -> object:
    def shutdown_with_pearl(self: object, *args: object, **kwargs: object) -> object:
        mining_error = _teardown_worker_mining()
        if mining_error is not None:
            _LOGGER.opt(exception=mining_error).error(
                "Pearl worker teardown failed; continuing native vLLM shutdown"
            )
        result = original(self, *args, **kwargs)
        if mining_error is not None:
            raise mining_error
        return result

    return shutdown_with_pearl


def _install_shutdown_hook() -> None:
    from vllm.v1.worker.gpu_worker import Worker as gpu_worker_cls

    wrap_callable_once(gpu_worker_cls, "shutdown", _make_worker_shutdown_wrapper)


def _install_serving_activity_hook() -> None:
    def make_activity_wrapper(original: object) -> object:
        def execute_model_with_activity(
            self: object, scheduler_output: object, *args: object, **kwargs: object
        ) -> object:
            # V2 routes profiling forwards through execute_model(dummy_run=True).
            if not kwargs.get("dummy_run"):
                resume_submissions_after_capture()
            return original(self, scheduler_output, *args, **kwargs)

        return execute_model_with_activity

    if not _install_on_gpu_model_runners(
        lambda cls: wrap_callable_once(cls, "execute_model", make_activity_wrapper)
    ):
        _LOGGER.warning("No GPUModelRunner class importable; execution hook not installed.")
