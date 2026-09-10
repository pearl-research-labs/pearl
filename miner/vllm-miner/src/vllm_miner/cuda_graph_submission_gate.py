"""vLLM CUDA-graph gate: mined launches stay suspended through capture."""

import threading

from miner_utils import get_logger

from .capture import (
    resume_mining_launches,
    suspend_mining_launches_until_resumed,
)

_LOGGER = get_logger("vllm.pearl_miner")
_CAPTURE_SUSPENSION_ID: int | None = None
_CAPTURE_GATE_LOCK = threading.Lock()


def _will_capture_cuda_graphs() -> bool:
    """Return whether the resolved vLLM config will invoke graph capture."""
    try:
        from vllm.config import get_current_vllm_config
        from vllm.config.compilation import CUDAGraphMode

        config = get_current_vllm_config()
        if bool(config.model_config.enforce_eager):
            return False
        mode = config.compilation_config.cudagraph_mode
        return mode is not None and mode != CUDAGraphMode.NONE
    except Exception:
        # A false positive can suppress useful work forever if no capture hook
        # runs. Fail open here; capture itself still raises the hard producer
        # gate and quiesces all admitted work before touching CUDA.
        _LOGGER.warning(
            "Could not resolve vLLM CUDA-graph mode; not pre-suspending mining.",
            exc_info=True,
        )
        return False


def suspend_submissions_until_capture_complete() -> None:
    """Suspend mined launches once, but only when vLLM will really capture."""
    global _CAPTURE_SUSPENSION_ID
    if not _will_capture_cuda_graphs():
        return
    with _CAPTURE_GATE_LOCK:
        if _CAPTURE_SUSPENSION_ID is None:
            _CAPTURE_SUSPENSION_ID = suspend_mining_launches_until_resumed(
                "vllm_cuda_graph_capture"
            )


def resume_submissions_after_capture() -> None:
    """Resume mined launches after capture or on the first real execute step."""
    global _CAPTURE_SUSPENSION_ID
    with _CAPTURE_GATE_LOCK:
        suspension_id, _CAPTURE_SUSPENSION_ID = _CAPTURE_SUSPENSION_ID, None
    if suspension_id is not None:
        resume_mining_launches(suspension_id)
