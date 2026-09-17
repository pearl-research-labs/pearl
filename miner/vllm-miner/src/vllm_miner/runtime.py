"""Process-global mining runtime initialization for the vLLM plugin.

Kept separate from registration because the runtime touches CUDA and must be
booted only after vLLM binds the worker device (see :mod:`worker_hooks`).
"""

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from miner_utils import get_logger

from .capture import (
    mining_suspended_for_capture,
    suspend_mining_launches,
)
from .cuda_graph_submission_gate import resume_submissions_after_capture
from .cutlass_compat import apply_cutlass_46_compat
from .hooks import wrap_callable_once
from .mining_state import init_async_manager

_LOGGER = get_logger("vllm.pearl_miner")

_init_lock = threading.Lock()


@contextmanager
def _mining_suspended_for_capture(entry_point: str) -> Iterator[None]:
    """Adapter seam over the shared gate-first capture-quiescence protocol."""
    with mining_suspended_for_capture(entry_point, framework="vLLM"):
        yield


def ensure_pearl_runtime_initialized() -> None:
    """Boot the mining runtime (async manager plus worker hooks).

    Thread-safe and re-entrant: delegates to the idempotent shared-runtime
    initializers. Must run after vLLM has selected the worker's CUDA device.
    """
    with _init_lock:
        # Restore the CuTe alias removed in cutlass-dsl 4.6 before vLLM's
        # wheel-bundled flash-attn kernels use it.
        apply_cutlass_46_compat()
        # vLLM 0.28's AOT compile cache can serve a compiled graph that calls
        # pearl::apply_linear during engine warmup, before any Python-level
        # PearlLinearMethod.apply (whose lazy import registers the op on a
        # cold compile) has run. Register the op before the model loads.
        from . import linear_op  # noqa: F401  (registers pearl::apply_linear)

        _install_fused_fp8_activation_quantizer()
        # TODO: Export process-local proof-backpressure state through vLLM's
        # /metrics once the adapter has a supported worker-to-engine bridge.
        init_async_manager()
        _install_capture_phase_hook()
        _install_weight_lifecycle_hooks()
        # Always install the real-execute fallback: a resolved NONE graph mode
        # or a failed/absent capture hook must not leave launches suspended.
        _install_serving_activity_hook()


def rollback_pearl_runtime_startup() -> bool:
    """Best-effort rollback for adapter startup before model loading begins."""
    from .mining_state import delete_state, try_get_async_manager
    from .state import clear_state_registry

    try:
        if try_get_async_manager() is not None:
            delete_state()
        clear_state_registry()
    except BaseException:
        _LOGGER.opt(exception=True).error("Pearl runtime startup rollback failed")
        return False
    return True


def _install_fused_fp8_activation_quantizer() -> None:
    """Inject vLLM's fused dynamic per-token quantizer into vllm_miner."""
    from vllm import _custom_ops as ops

    from .fp8_fallback import register_activation_quantizer

    def quantize(x):
        return ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)

    register_activation_quantizer(quantize)


def iter_gpu_model_runner_classes() -> Iterator[type[object]]:
    """Yield the importable GPU model runner classes (V1 legacy, V2/MRv2).

    vLLM picks one of the two at worker init (dense models and DSpark select
    V2), so hooks must land on both for mining to engage regardless of the
    selection. Imports are guarded: a build lacking one class degrades to the
    other instead of failing the install.
    """
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as v1_runner_cls

        yield v1_runner_cls
    except ImportError:
        _LOGGER.opt(exception=True).warning("Could not import the V1 GPUModelRunner.")
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner as v2_runner_cls

        yield v2_runner_cls
    except ImportError:
        _LOGGER.opt(exception=True).warning("Could not import the V2 GPUModelRunner.")


def _install_on_gpu_model_runners(install: Callable[[type[object]], None]) -> bool:
    """Run ``install`` for each importable GPU model runner. Return whether any ran."""
    installed = False
    for runner_cls in iter_gpu_model_runner_classes():
        install(runner_cls)
        installed = True
    return installed


def _make_capture_model_wrapper(original: object) -> object:
    def capture_model_no_mining(self: object, *args: object, **kwargs: object) -> object:
        captured = False
        try:
            with _mining_suspended_for_capture("capture_model"):
                result = original(self, *args, **kwargs)
                captured = True
                return result
        finally:
            # Capture failure is worker-fatal. Do not reopen mining against a
            # partial/invalid graph epoch while that failure propagates.
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
    """Suspend mining across profiling and optionally reserve post-profile work."""

    def make_wrapper(original: object) -> object:
        def profile_no_mining(self: object, *args: object, **kwargs: object) -> object:
            with _mining_suspended_for_capture(entry_point):
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
    """Warm every layer's variants and prepare B for the current job, if any.

    Runs synchronously on the worker thread; a failure never blocks serving
    (the affected layers stay on the FP8 fallback).
    """
    from .job_prep import current_job, prepare_mining

    try:
        return prepare_mining(current_job())
    except Exception:
        _LOGGER.opt(exception=True).warning(
            "Initial mining preparation failed; serving continues on the FP8 fallback."
        )
        return False


def _make_worker_warmup_wrapper(original: object) -> object:
    """Run initial mining preparation after the framework's own warmup."""

    def warmup_no_mining(self: object, *args: object, **kwargs: object) -> object:
        with suspend_mining_launches():
            result = original(self, *args, **kwargs)
            _prepare_mining()
            return result

    return warmup_no_mining


def _install_capture_phase_hook() -> None:
    """Keep mining out of vLLM profiling, warmup, and CUDA-graph capture."""

    def install(runner_cls: type[object]) -> None:
        wrap_callable_once(runner_cls, "capture_model", _make_capture_model_wrapper)
        # Absent on some vLLM builds, hence require=False.
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
        _LOGGER.warning(
            "No GPUModelRunner class importable; CUDA graph mining-submission hook "
            "not installed. Mining may stay suspended under CUDA-graph capture."
        )

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
        _make_memory_profile_wrapper(
            "determine_available_memory",
            reserve_mining_memory=True,
        ),
    )
    wrap_callable_once(
        gpu_worker_cls,
        "compile_or_warm_up_model",
        _make_worker_warmup_wrapper,
    )


def _make_worker_sleep_lifecycle_wrapper(original: object) -> object:
    def sleep_with_pearl(self: object, level: int = 1) -> object:
        from .lifecycle import lifecycle

        owner = lifecycle()
        try:
            owner.before_sleep(level)
            return original(self, level=level)
        except BaseException:
            owner.sleep_failed()
            raise

    return sleep_with_pearl


def _make_worker_wake_lifecycle_wrapper(original: object) -> object:
    def wake_with_pearl(self: object, tags: list[str] | None = None) -> object:
        from .lifecycle import lifecycle

        owner = lifecycle()
        try:
            result = original(self, tags=tags)
            owner.after_wake(tags)
            return result
        except BaseException:
            owner.wake_failed()
            raise

    return wake_with_pearl


def _make_worker_reload_lifecycle_wrapper(original: object) -> object:
    def reload_with_pearl(self: object, *args: object, **kwargs: object) -> object:
        from .lifecycle import lifecycle, reload_is_checkpoint_format

        owner = lifecycle()
        try:
            owns_interval = owner.before_reload(reload_is_checkpoint_format(args, kwargs))
            result = original(self, *args, **kwargs)
            owner.after_reload(owns_interval=owns_interval)
            return result
        except BaseException:
            owner.reload_failed()
            raise

    return reload_with_pearl


def _record_secondary_teardown_error(
    current: BaseException | None,
    operation: Callable[[], object],
    message: str,
) -> BaseException | None:
    try:
        operation()
    except BaseException as exc:
        if current is None:
            return exc
        _LOGGER.opt(exception=exc).error(message)
    return current


def _teardown_worker_mining() -> BaseException | None:
    from .capture import close_mining_admission
    from .mining_state import delete_state
    from .state import clear_state_registry

    error = _record_secondary_teardown_error(
        None,
        close_mining_admission,
        "Pearl mining-admission closure failed",
    )

    try:
        delete_state()
    except BaseException as exc:
        if error is None:
            return exc
        _LOGGER.opt(exception=exc).error("Pearl process-local runtime teardown also failed")
        return error
    return _record_secondary_teardown_error(
        error,
        clear_state_registry,
        "Pearl state-registry cleanup also failed",
    )


def _make_worker_shutdown_lifecycle_wrapper(original: object) -> object:
    def shutdown_with_pearl(self: object, *args: object, **kwargs: object) -> object:
        try:
            mining_error = _teardown_worker_mining()
        except BaseException as exc:
            mining_error = exc
        if mining_error is not None:
            _LOGGER.opt(exception=mining_error).error(
                "Pearl worker teardown failed; continuing native vLLM shutdown"
            )

        try:
            result = original(self, *args, **kwargs)
        except BaseException:
            if mining_error is not None:
                _LOGGER.opt(exception=mining_error).error(
                    "Pearl worker teardown also failed before native vLLM shutdown"
                )
            raise
        if mining_error is not None:
            raise mining_error
        return result

    return shutdown_with_pearl


def _install_weight_lifecycle_hooks() -> None:
    """Wrap vLLM 0.28 worker sleep/wake/reload with Pearl GPU ownership."""
    from vllm.v1.worker.gpu_worker import Worker as gpu_worker_cls

    wrap_callable_once(gpu_worker_cls, "sleep", _make_worker_sleep_lifecycle_wrapper)
    wrap_callable_once(gpu_worker_cls, "wake_up", _make_worker_wake_lifecycle_wrapper)
    wrap_callable_once(
        gpu_worker_cls,
        "reload_weights",
        _make_worker_reload_lifecycle_wrapper,
    )
    wrap_callable_once(
        gpu_worker_cls,
        "shutdown",
        _make_worker_shutdown_lifecycle_wrapper,
    )


def _install_serving_activity_hook() -> None:
    """Install the unified real-execute lifecycle hook.

    The first non-dummy execution releases any pre-capture suspension even if
    ``capture_model`` never ran.
    """

    def make_activity_wrapper(original: object) -> object:
        def execute_model_with_activity(
            self: object, scheduler_output: object, *args: object, **kwargs: object
        ) -> object:
            # V2 routes profiling forwards through execute_model(dummy_run=True).
            if kwargs.get("dummy_run"):
                return original(self, scheduler_output, *args, **kwargs)

            resume_submissions_after_capture()
            return original(self, scheduler_output, *args, **kwargs)

        return execute_model_with_activity

    def install(runner_cls: type[object]) -> None:
        wrap_callable_once(runner_cls, "execute_model", make_activity_wrapper)

    if not _install_on_gpu_model_runners(install):
        _LOGGER.warning("No GPUModelRunner class importable; execution hook not installed.")
