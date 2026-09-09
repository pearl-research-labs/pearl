"""Reversible vLLM weight/sleep ownership for graph-stable Pearl states."""

import threading
from dataclasses import fields
from typing import Any

import torch
from miner_utils import get_logger

from .capture import (
    resume_mining_producers,
    suspend_mining_producers_until_resumed,
    wait_for_mining_producers_idle,
)
from .job_prep import unpublish_contexts
from .state import LayerBuffers, all_states

_LOGGER = get_logger("vllm.pearl_miner.lifecycle")
_LIFECYCLE_QUIESCE_TIMEOUT_S = 60.0
_SLEEP_TAGS = {"weights", "kv_cache"}


def _synchronize_weight_streams() -> None:
    """Wait exact current-stream fences before other streams read refreshed state."""
    events: list[torch.cuda.Event] = []
    seen_devices: set[torch.device] = set()
    for state in all_states():
        device = state.weight.device
        if device in seen_devices:
            continue
        seen_devices.add(device)
        with torch.cuda.device(device):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
        events.append(event)
    for event in events:
        event.synchronize()


def _state_pointers() -> dict[tuple[int, str], int]:
    """Snapshot every graph- or cache-visible CUDA tensor address."""
    pointers: dict[tuple[int, str], int] = {}
    for state in all_states():
        for name in ("weight", "weight_scale", "w_fp8", "w_fp8_scale"):
            tensor = getattr(state, name)
            pointers[(state.layer_id, name)] = tensor.data_ptr()
        buffers = state.buffers
        if buffers is not None:
            for descriptor in fields(LayerBuffers):
                value = getattr(buffers, descriptor.name)
                if isinstance(value, torch.Tensor):
                    pointers[(state.layer_id, f"buffers.{descriptor.name}")] = value.data_ptr()
    return pointers


class PearlVllmLifecycle:
    """Own one worker's reversible reload and sleep/offload interval.

    vLLM 0.28 preserves kernel tensor storage during checkpoint reload and
    CuMem virtual addresses during sleep. Pearl therefore refreshes derived
    operands in place and verifies that every pointer remains stable before
    mining is admitted again.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._suspension_id: int | None = None
        self._reason: str | None = None
        self._sleep_level: int | None = None
        self._sleeping_tags: set[str] = set()
        self._deep_reload_complete = False
        self._pointer_snapshot: dict[tuple[int, str], int] = {}
        self._failed = False

    def _mark_failed(self, operation: str) -> None:
        with self._lock:
            self._failed = True
        _LOGGER.error(
            f"Pearl vLLM lifecycle failed during {operation}; mining remains suspended "
            "and the worker must be restarted."
        )

    def _quiesce(self, reason: str) -> None:
        with self._lock:
            if self._suspension_id is not None:
                raise RuntimeError(
                    f"Pearl lifecycle already owns {self._reason}; cannot begin {reason}"
                )
            self._reason = reason
            self._suspension_id = suspend_mining_producers_until_resumed(reason)

        if not wait_for_mining_producers_idle(_LIFECYCLE_QUIESCE_TIMEOUT_S):
            self._mark_failed(reason)
            raise TimeoutError(
                f"mining GPU work did not quiesce within {_LIFECYCLE_QUIESCE_TIMEOUT_S}s "
                f"before vLLM {reason}"
            )
        from .pipeline import wait_for_mining_continuations_idle

        if not wait_for_mining_continuations_idle(_LIFECYCLE_QUIESCE_TIMEOUT_S):
            self._mark_failed(reason)
            raise TimeoutError(
                f"mining accounting/winner continuations did not quiesce before {reason}"
            )

        # Mining GPU work is drained: the buffers may now be rewritten and the
        # job they hold can no longer be trusted. The next launch re-prepares.
        unpublish_contexts()
        with self._lock:
            self._pointer_snapshot = _state_pointers()

    def _verify_pointers(self) -> None:
        current = _state_pointers()
        with self._lock:
            expected = self._pointer_snapshot
        if current != expected:
            missing = sorted(set(expected) - set(current))
            changed = sorted(
                key for key in expected.keys() & current.keys() if expected[key] != current[key]
            )
            raise RuntimeError(
                "vLLM changed Pearl graph-visible storage across reload/sleep "
                f"(missing={missing[:4]}, changed={changed[:4]})"
            )

    def before_sleep(self, level: int) -> None:
        if level not in (1, 2):
            raise ValueError(f"unsupported vLLM sleep level {level}; expected 1 or 2")
        self._quiesce(f"sleep(level={level})")
        with self._lock:
            self._sleep_level = level
            self._sleeping_tags = set(_SLEEP_TAGS)
            self._deep_reload_complete = False

    def sleep_failed(self) -> None:
        self._mark_failed("sleep")

    def after_wake(self, tags: list[str] | None) -> None:
        awakened = set(_SLEEP_TAGS if tags is None else tags)
        if "weights" in awakened:
            _synchronize_weight_streams()
            self._verify_pointers()
        with self._lock:
            if self._sleep_level is None:
                raise RuntimeError("vLLM wake_up without an active Pearl sleep lifecycle")
            self._sleeping_tags.difference_update(awakened)
            finished = not self._sleeping_tags
            deep_reload_missing = (
                self._sleep_level == 2 and finished and not self._deep_reload_complete
            )
        if deep_reload_missing:
            self._mark_failed("deep wake without reload")
            raise RuntimeError(
                "vLLM level-2 wake requires checkpoint reload_weights before final wake"
            )
        if finished:
            self._finish_successfully()

    def wake_failed(self) -> None:
        self._mark_failed("wake_up")

    def before_reload(self, is_checkpoint_format: bool) -> bool:
        if not is_checkpoint_format:
            raise ValueError(
                "Pearl supports checkpoint-format vLLM reload only; kernel-format "
                "reload omits derived FP10/FP8 state"
            )
        with self._lock:
            if self._failed:
                raise RuntimeError("Pearl lifecycle is failed; restart the worker")
            if self._suspension_id is not None:
                if self._sleep_level != 2 or "weights" in self._sleeping_tags:
                    raise RuntimeError(
                        "reload_weights may reuse only a level-2 sleep lifecycle after "
                        "the weights tag has been awakened"
                    )
                return False
        self._quiesce("reload_weights")
        return True

    def after_reload(self, *, owns_interval: bool) -> None:
        _synchronize_weight_streams()
        self._verify_pointers()
        with self._lock:
            if self._sleep_level == 2:
                self._deep_reload_complete = True
                return
        if owns_interval:
            self._finish_successfully()

    def reload_failed(self) -> None:
        self._mark_failed("reload_weights")

    def _finish_successfully(self) -> None:
        self._verify_pointers()
        with self._lock:
            if self._failed:
                raise RuntimeError("cannot resume a failed Pearl lifecycle")
            suspension_id = self._suspension_id
            if suspension_id is None:
                raise RuntimeError("Pearl lifecycle has no producer suspension to release")

        with self._lock:
            if self._failed or self._suspension_id != suspension_id:
                raise RuntimeError("Pearl lifecycle ownership changed during restart")
            self._suspension_id = None
            self._reason = None
            self._sleep_level = None
            self._sleeping_tags.clear()
            self._deep_reload_complete = False
            self._pointer_snapshot = {}
        resume_mining_producers(suspension_id)

    def reset_for_tests(self) -> None:
        """Release only an unused test suspension; production failures stay fatal."""
        with self._lock:
            suspension_id = self._suspension_id
            self._suspension_id = None
            self._reason = None
            self._sleep_level = None
            self._sleeping_tags.clear()
            self._deep_reload_complete = False
            self._pointer_snapshot = {}
            self._failed = False
        if suspension_id is not None:
            resume_mining_producers(suspension_id)


_LIFECYCLE = PearlVllmLifecycle()


def lifecycle() -> PearlVllmLifecycle:
    return _LIFECYCLE


def reload_is_checkpoint_format(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """Resolve GPUWorker.reload_weights' forwarded format argument."""
    if "is_checkpoint_format" in kwargs:
        return bool(kwargs["is_checkpoint_format"])
    # GPUModelRunner signature: iterator, path, is_checkpoint_format.
    return bool(args[2]) if len(args) >= 3 else True
