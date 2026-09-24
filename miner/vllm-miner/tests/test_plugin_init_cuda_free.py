"""Plugin load must not create a CUDA context, and must defer runtime init.

Regression tests for the bug where loading the plugin in a data-parallel worker
created a CUDA context on the default device (GPU 0) before the worker bound its
per-rank device. The root cause is a premature context at registration time,
which is observable on a single GPU via ``torch.cuda.is_initialized()``.
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

_NO_CUDA_CONTEXT_PROBE = Path(__file__).parent / "_no_cuda_context_probe.py"


def test_plugin_registration_creates_no_cuda_context():
    # Fresh interpreter: a CUDA context is process-global, so other GPU tests in
    # this session would otherwise have already created one. The probe loads the
    # plugin and asserts no context was created (see _no_cuda_context_probe.py).
    result = subprocess.run(
        [sys.executable, str(_NO_CUDA_CONTEXT_PROBE)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_worker_hook_runs_runtime_after_init_device(monkeypatch):
    """The deferral hook must boot the runtime *after* the wrapped ``init_device``,
    and be idempotent (a second install must not double-wrap)."""
    import vllm.v1.worker.worker_base as worker_base
    import vllm_miner.runtime as runtime
    import vllm_miner.worker_hooks as worker_hooks

    order = []

    # Fresh local class: install_* rebinds init_device on the class itself (not
    # via monkeypatch), so a shared class would leak the patch.
    class _DummyWrapper:
        def __init__(self):
            self.vllm_config = SimpleNamespace(device_config=SimpleNamespace(device_type="cuda"))

        def init_device(self):
            order.append("init_device")

    monkeypatch.setattr(worker_base, "WorkerWrapperBase", _DummyWrapper)
    monkeypatch.setattr(
        runtime, "ensure_pearl_runtime_initialized", lambda: order.append("runtime")
    )

    worker_hooks.install_pearl_worker_runtime_hook()
    worker_hooks.install_pearl_worker_runtime_hook()  # idempotent: no double-wrap

    _DummyWrapper().init_device()

    assert order == ["init_device", "runtime"]


def test_worker_runtime_failure_rolls_back_and_preserves_native_serving(monkeypatch):
    import vllm_miner.mining_state as gpu_runtime
    import vllm_miner.runtime as runtime
    from vllm_miner.worker_hooks import _init_runtime_if_cuda

    disabled = []
    monkeypatch.setattr(
        runtime,
        "ensure_pearl_runtime_initialized",
        lambda: (_ for _ in ()).throw(RuntimeError("injected mining failure")),
    )
    monkeypatch.setattr(runtime, "rollback_pearl_runtime_startup", lambda: True)
    monkeypatch.setattr(
        gpu_runtime,
        "disable_mining_after_runtime_failure",
        disabled.append,
    )
    monkeypatch.setattr(gpu_runtime, "mining_disabled_by_runtime_failure", lambda: True)
    wrapper = SimpleNamespace(
        vllm_config=SimpleNamespace(device_config=SimpleNamespace(device_type="cuda"))
    )

    _init_runtime_if_cuda(wrapper)

    assert len(disabled) == 1
    assert "RuntimeError" in disabled[0]


def test_worker_listener_failure_preserves_native_serving(monkeypatch):
    import vllm_miner.mining_state as gpu_runtime
    import vllm_miner.runtime as runtime
    from vllm_miner.worker_hooks import _init_runtime_if_cuda

    monkeypatch.setattr(runtime, "ensure_pearl_runtime_initialized", lambda: None)
    monkeypatch.setattr(gpu_runtime, "mining_disabled_by_runtime_failure", lambda: True)
    wrapper = SimpleNamespace(
        vllm_config=SimpleNamespace(device_config=SimpleNamespace(device_type="cuda"))
    )

    _init_runtime_if_cuda(wrapper)


def test_tp_workers_boot_process_local_mining_runtime(monkeypatch):
    import vllm_miner.runtime as runtime
    from vllm_miner.worker_hooks import _init_runtime_if_cuda

    initialized = []
    monkeypatch.setattr(
        runtime,
        "ensure_pearl_runtime_initialized",
        lambda: initialized.append("runtime"),
    )
    wrapper = SimpleNamespace(
        vllm_config=SimpleNamespace(
            device_config=SimpleNamespace(device_type="cuda"),
            parallel_config=SimpleNamespace(tensor_parallel_size=8),
        )
    )

    _init_runtime_if_cuda(wrapper)

    assert initialized == ["runtime"]
