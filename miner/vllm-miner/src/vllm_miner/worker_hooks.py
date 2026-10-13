"""Post-``init_device`` hook that boots the Pearl mining runtime.

vLLM selects each worker's CUDA device inside ``Worker.init_device`` (reached via
``WorkerWrapperBase.init_device``). Booting the runtime right after that binds it
to the correct per-data-parallel-rank device instead of the default one.
"""

import threading

from miner_utils import get_logger

from .hooks import wrap_callable_once

_LOGGER = get_logger("vllm.pearl_miner")

_install_lock = threading.Lock()


def install_pearl_worker_runtime_hook() -> None:
    """Wrap ``WorkerWrapperBase.init_device`` to boot the runtime after it.

    Idempotent per process; a no-op only when vLLM's worker base is missing
    (the wrapper-model path then still boots the runtime lazily). Listener-start
    failure disables mining while native serving continues; failures that cannot
    roll back safely still fail worker init.
    """
    # Guarded import of a vLLM internal: tolerate builds without this seam.
    try:
        from vllm.v1.worker.worker_base import WorkerWrapperBase
    except ImportError:
        _LOGGER.opt(exception=True).warning(
            "Could not import WorkerWrapperBase; Pearl runtime will init lazily."
        )
        return

    def make_wrapper(original: object) -> object:
        def init_device_then_pearl_runtime(self: object, *args: object, **kwargs: object) -> object:
            result = original(self, *args, **kwargs)
            _init_runtime_if_cuda(self)
            return result

        return init_device_then_pearl_runtime

    with _install_lock:
        wrap_callable_once(WorkerWrapperBase, "init_device", make_wrapper)


def _init_runtime_if_cuda(wrapper: object) -> None:
    vllm_config = getattr(wrapper, "vllm_config", None)
    device_config = getattr(vllm_config, "device_config", None)
    if getattr(device_config, "device_type", None) != "cuda":
        return
    # Each CUDA worker owns a process-local runtime after vLLM has selected its
    # device. Dense TP/DP/EP shards are admitted later by their local shape.
    # Imported here so non-worker processes never pull in the mining runtime.
    from .mining_state import (
        disable_mining_after_runtime_failure,
        mining_disabled_by_runtime_failure,
    )
    from .runtime import ensure_pearl_runtime_initialized, rollback_pearl_runtime_startup

    try:
        ensure_pearl_runtime_initialized()
    except Exception as exc:
        if not rollback_pearl_runtime_startup():
            _LOGGER.opt(exception=True).error(
                "Pearl runtime startup could not roll back safely; vLLM worker init is aborted"
            )
            raise
        disable_mining_after_runtime_failure(f"Pearl runtime startup failed ({type(exc).__name__})")
    if mining_disabled_by_runtime_failure():
        _LOGGER.error("Pearl runtime startup rolled back; native vLLM serving continues")
