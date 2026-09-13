"""``MINER_NO_MINING`` must stop mining work, not just the launches that credit
hashes.

Two boundaries own the switch and both are host logic, so both are testable
without a GPU. The serving op declines before any GPU preparation
(``linear_op._try_mine``), and startup preparation skips every layer
(``job_prep.prepare_mining``) -- otherwise an operator who disabled mining
still pays the GPU B-preparation chain and the kernel warmup launches per
layer, none of which credit anything. Layers stay FP10-encoded and serve on
the FP8 fallback either way: encoding happens at load, before any of this.

Both tests carry a positive control, because "mining did not happen" is the
assertion a broken guard chain satisfies for free.
"""

import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import vllm_miner.capture as gcs
from miner_base.settings import MinerSettings
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.dataclasses import MiningJob
from vllm_miner import health, job_prep, linear_op, pipeline
from vllm_miner import settings as settings_module
from vllm_miner.capture import suspend_mining_launches
from vllm_miner.config import config as gpu_config
from vllm_miner.settings import RuntimeSettings
from vllm_miner.state import (
    LayerState,
    lookup_state,
    register_state,
    unregister_state,
)

_N, _K = 128, 1024
# Small enough to keep the probe activation cheap; the exact value is only ever
# compared against itself.
_MIN_MINING_TOKENS = 16
_HEADER = bytes(range(80))


@pytest.fixture(autouse=True)
def pinned_settings(monkeypatch):
    """Pin the GPU runtime settings this suite reasons about.

    ``warmup_compile`` is on: warmup is the most expensive thing the kill switch
    has to stop, so the sweep test must be able to observe it.
    """
    monkeypatch.setattr(
        settings_module,
        "_settings",
        RuntimeSettings(min_mining_tokens=_MIN_MINING_TOKENS, warmup_compile=True),
    )


@pytest.fixture
def layer():
    """A registered, mineable, FP10-encoded layer with no context published yet."""
    weight = torch.zeros(_N, _K, dtype=torch.int8)
    state = LayerState(
        layer_name="test.mined.layer",
        layer_id=1,
        weight=weight,
        weight_scale=torch.ones(_N, _K // 8, dtype=torch.bfloat16),
        weight_cpu=weight,
        weight_scale_cpu=torch.ones(_N, _K // 8, dtype=torch.bfloat16),
        n=_N,
        k=_K,
        mineable=True,
        buffers=SimpleNamespace(),
        w_fp8=torch.zeros(_N, _K, dtype=torch.float8_e4m3fn),
        w_fp8_scale=torch.ones(1, _N, dtype=torch.float32),
    )
    register_state(state)
    yield state
    unregister_state(state)


def test_registry_rejects_duplicate_owners_without_removing_original(layer):
    duplicate_id = SimpleNamespace(weight=layer.weight.clone(), layer_id=layer.layer_id)
    with pytest.raises(RuntimeError, match="layer id"):
        register_state(duplicate_id)
    unregister_state(duplicate_id)
    assert lookup_state(layer.weight) is layer

    duplicate_pointer = SimpleNamespace(weight=layer.weight, layer_id=layer.layer_id + 10_000)
    with pytest.raises(RuntimeError, match="weight pointer"):
        register_state(duplicate_pointer)
    unregister_state(duplicate_pointer)
    assert lookup_state(layer.weight) is layer


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_apply_linear_rejects_non_bf16_before_dispatch(monkeypatch, dtype):
    lookup = Mock(side_effect=AssertionError("invalid dtype reached state lookup"))
    monkeypatch.setattr(linear_op, "lookup_state", lookup)
    x = torch.zeros(2, 8, dtype=dtype)
    weight = torch.zeros(4, 8, dtype=torch.int8)

    with pytest.raises(TypeError, match="requires bfloat16"):
        linear_op._apply_linear_impl(x, weight, None)
    with pytest.raises(TypeError, match="requires bfloat16"):
        linear_op._apply_linear_fake(x, weight, None)

    lookup.assert_not_called()


def test_apply_linear_rejects_non_bf16_bias_before_dispatch(monkeypatch):
    lookup = Mock(side_effect=AssertionError("invalid bias reached state lookup"))
    monkeypatch.setattr(linear_op, "lookup_state", lookup)
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    weight = torch.zeros(4, 8, dtype=torch.int8)
    bias = torch.zeros(4, dtype=torch.float32)

    with pytest.raises(TypeError, match="requires bfloat16 bias"):
        linear_op._apply_linear_impl(x, weight, bias)
    with pytest.raises(TypeError, match="requires bfloat16 bias"):
        linear_op._apply_linear_fake(x, weight, bias)

    lookup.assert_not_called()


def test_apply_linear_bf16_fallback_contract(monkeypatch):
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    weight = torch.zeros(4, 8, dtype=torch.int8)
    state = SimpleNamespace(
        k=8,
        weight=weight,
        w_fp8=object(),
        w_fp8_scale=object(),
    )
    fallback = Mock(return_value=torch.zeros(2, 4, dtype=torch.bfloat16))
    monkeypatch.setattr(linear_op, "lookup_state", lambda _weight: state)
    monkeypatch.setattr(linear_op, "_try_mine", lambda *_args: None)
    monkeypatch.setattr(linear_op, "fp8_fallback_gemm", fallback)

    output = linear_op._apply_linear_impl(x, weight, None)

    assert output.dtype == torch.bfloat16
    assert tuple(output.shape) == (2, 4)
    assert fallback.call_args.args[-1] == torch.bfloat16


def test_no_mining_gates_the_serving_path(layer, monkeypatch):
    """The switch must decline the forward, and be the *only* reason it declines.

    ``pick_bucket`` is the first thing past the whole guard chain, so it doubles
    as a reach detector: with the switch off the same forward must reach it. Were
    it unreachable for another reason -- an unmineable layer, too few tokens, the
    capture gate -- the "returns None" assertion would hold with or without the
    behaviour under test.
    """
    reached: list[int] = []

    def reach_detector(m_tokens: int) -> None:
        reached.append(m_tokens)
        return None  # no bucket: the caller falls back without touching the GPU

    monkeypatch.setattr(pipeline, "pick_bucket", reach_detector)
    # A host-only run has no stream to query; the capture guards have their own
    # suite (test_capture_gate) and are not what this test is about.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    x2d = torch.zeros(_MIN_MINING_TOKENS, _K, dtype=torch.bfloat16)

    monkeypatch.setattr(gpu_config, "settings", MinerSettings(no_mining=True))
    assert linear_op._try_mine(layer, x2d, None) is None
    assert reached == [], "MINER_NO_MINING did not gate the serving path"

    monkeypatch.setattr(gpu_config, "settings", MinerSettings(no_mining=False))
    assert linear_op._try_mine(layer, x2d, None) is None  # no bucket: fallback
    assert reached == [_MIN_MINING_TOKENS], "the guard chain blocked for some other reason"


def _prepare_failing_mined_forward(layer, monkeypatch, error):
    attempts = []
    job = object()
    layer.job_ctx = SimpleNamespace(job=job)
    monkeypatch.setattr(gpu_config, "settings", MinerSettings(no_mining=False))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(job_prep, "current_job", lambda: job)
    monkeypatch.setattr(pipeline, "pick_bucket", lambda _tokens: 256)
    monkeypatch.setattr(pipeline, "variant_ready", lambda *_args: True)

    def fail(*_args, **_kwargs):
        attempts.append(object())
        raise error

    monkeypatch.setattr(pipeline, "run_mining_forward", fail)
    health._BREAKER.reset_for_tests()
    return attempts


def test_serving_oom_opens_device_cooldown_without_disabling_layer(layer, monkeypatch):
    attempts = _prepare_failing_mined_forward(
        layer,
        monkeypatch,
        torch.cuda.OutOfMemoryError("synthetic OOM"),
    )
    x = torch.zeros(_MIN_MINING_TOKENS, _K, dtype=torch.bfloat16)

    assert linear_op._try_mine(layer, x, None) is None
    assert linear_op._try_mine(layer, x, None) is None

    assert len(attempts) == 1
    assert layer.mineable
    assert layer.disabled_reason is None


def test_deterministic_serving_failure_disables_only_layer(layer, monkeypatch):
    attempts = _prepare_failing_mined_forward(
        layer,
        monkeypatch,
        RuntimeError("synthetic deterministic failure"),
    )
    x = torch.zeros(_MIN_MINING_TOKENS, _K, dtype=torch.bfloat16)

    assert linear_op._try_mine(layer, x, None) is None
    assert linear_op._try_mine(layer, x, None) is None

    assert len(attempts) == 1
    assert not layer.mineable
    assert "deterministic forward failure" in layer.disabled_reason


def test_framework_warmup_gate_stops_serving_launches(layer, monkeypatch):
    """A dummy model forward declines before pipeline work, then the same input
    reaches the pipeline as soon as framework initialization ends."""
    reached: list[int] = []

    def reach_detector(m_tokens: int) -> None:
        reached.append(m_tokens)
        return None

    monkeypatch.setattr(pipeline, "pick_bucket", reach_detector)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(gpu_config, "settings", MinerSettings(no_mining=False))
    x2d = torch.zeros(_MIN_MINING_TOKENS, _K, dtype=torch.bfloat16)

    with suspend_mining_launches():
        assert linear_op._try_mine(layer, x2d, None) is None
    assert reached == [], "a framework warmup forward entered the mining pipeline"

    assert linear_op._try_mine(layer, x2d, None) is None
    assert reached == [_MIN_MINING_TOKENS], "the warmup gate stayed closed after its scope"


def test_hit_signal_allocates_normal_tensors_during_inference_mode(monkeypatch):
    """Model loading may run under InferenceMode, but the consumer mutates the
    persistent signal from a normal worker thread."""
    modes: list[bool] = []

    class FakeConfig:
        def __init__(self, max_m, max_k):
            self.max_m = max_m
            self.max_k = max_k

    class FakeSignal:
        def __init__(self, cfg, device):
            modes.append(torch.is_inference_mode_enabled())
            self.cfg = cfg
            self.device = device

    monkeypatch.setitem(
        sys.modules,
        "pearl_gemm",
        SimpleNamespace(HitSignal=FakeSignal, HitSignalConfig=FakeConfig),
    )
    monkeypatch.setattr(pipeline, "_HIT_SIGNALS", {})

    with torch.inference_mode():
        acquired = pipeline._hit_signal_for(torch.device("cuda:0"))

    assert isinstance(acquired, FakeSignal)
    # Hit-signal buffers size to the largest k any committed tile can prove
    # (max_mineable_k across the 4x128 and 16x32 tiles), not the 4x128 tile alone.
    from vllm_miner.mining_config import max_mineable_k

    assert acquired.cfg.max_k == max_mineable_k()
    assert modes == [False]

    class PrimedLookupMustNotLock:
        def __enter__(self):
            raise AssertionError("primed hit-signal lookup reacquired construction lock")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(pipeline, "_hit_signal_lock", PrimedLookupMustNotLock())
    assert pipeline._hit_signal_for(torch.device("cuda:0")) is acquired


def test_hit_signal_concurrent_first_access_constructs_once(monkeypatch):
    constructions = []

    class FakeConfig:
        def __init__(self, max_m, max_k):
            self.max_m = max_m
            self.max_k = max_k

    class FakeSignal:
        def __init__(self, cfg, device):
            constructions.append(object())
            self.cfg = cfg
            self.device = device

    monkeypatch.setitem(
        sys.modules,
        "pearl_gemm",
        SimpleNamespace(HitSignal=FakeSignal, HitSignalConfig=FakeConfig),
    )
    monkeypatch.setattr(pipeline, "_HIT_SIGNALS", {})
    start = threading.Barrier(9)
    results = []

    def acquire():
        start.wait()
        results.append(pipeline._hit_signal_for(torch.device("cuda:0")))

    threads = [threading.Thread(target=acquire) for _ in range(8)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert len(constructions) == 1
    assert len({id(signal) for signal in results}) == 1


@pytest.mark.parametrize(
    ("interrupt", "expected_launches", "expected_ready"),
    [
        # One mixed-GEMM compile variant per bucket, two buckets.
        (None, 2, True),
        ("process-gate", 1, False),
        ("caller-cancel", 1, False),
    ],
)
def test_pipeline_warmup_reports_completion_or_stops(
    monkeypatch, interrupt, expected_launches, expected_ready
):
    """A full warmup reports ready; interruption stops after the current bucket."""
    launched: list[object] = []
    locally_cancelled = False

    class FakeEvent:
        def record(self):
            pass

        def synchronize(self):
            pass

    def launch_once(*_args, **_kwargs):
        nonlocal locally_cancelled
        launched.append(object())
        if interrupt == "process-gate":
            gcs.close_mining_admission()
        elif interrupt == "caller-cancel":
            locally_cancelled = True

    monkeypatch.setattr(gcs, "_admission_closed", False)
    monkeypatch.setattr(
        settings_module,
        "_settings",
        RuntimeSettings(m_buckets=(256, 512), warmup_compile=True),
    )
    monkeypatch.setattr(pipeline, "_ready_variants", set())
    monkeypatch.setattr(pipeline, "_failed_variants", set())
    monkeypatch.setattr(pipeline, "_hit_signal_for", lambda _device: object())
    monkeypatch.setattr(pipeline, "_launch_stages", launch_once)
    monkeypatch.setattr(pipeline.torch, "zeros", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(pipeline.torch.cuda, "Event", FakeEvent)
    state = SimpleNamespace(
        weight=SimpleNamespace(device="cuda:0"),
        buffers=object(),
        n=_N,
        k=_K,
        layer_name="test.mined.layer",
        layer_id=1,
    )

    ready = pipeline.warmup_layer_variants(state, cancelled=lambda: locally_cancelled)

    assert len(launched) == expected_launches
    assert ready is expected_ready


def test_no_mining_stops_b_preparation_and_warmup(layer, monkeypatch):
    """The switch must stop startup preparation, not only credited launches.

    ``prepare_mining`` is the boundary the adapter reaches at the framework's
    readiness barrier. With a job available and a registered mineable layer,
    the gated preparation must prepare nothing and warm up nothing, while the
    same call with the switch off does both.
    """
    job = MiningJob(
        incomplete_header_bytes=_HEADER,
        target=100,
        cert_version=CertificateVersion.PLAIN_FP8,
    )
    prepared: list[LayerState] = []
    warmed: list[str] = []

    def fake_prepare(state: LayerState, prepare_job: MiningJob) -> object:
        assert prepare_job is job
        prepared.append(state)
        return SimpleNamespace(job=prepare_job)

    def fake_warmup(state, cancelled=None):
        warmed.append(state.layer_name)
        return True

    monkeypatch.setattr(job_prep, "prepare_layer", fake_prepare)
    monkeypatch.setattr(pipeline, "warmup_layer_variants", fake_warmup)

    monkeypatch.setattr(gpu_config, "settings", MinerSettings(no_mining=True))
    assert job_prep.prepare_mining(job) is False

    assert prepared == [], "MINER_NO_MINING did not stop B preparation"
    assert warmed == [], "MINER_NO_MINING did not stop kernel warmup"

    monkeypatch.setattr(gpu_config, "settings", MinerSettings(no_mining=False))
    assert job_prep.prepare_mining(job) is True

    assert prepared == [layer], "preparation was blocked for some reason other than the switch"
    assert warmed == [layer.layer_name]
