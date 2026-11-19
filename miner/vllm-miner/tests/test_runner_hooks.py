import contextlib
from types import SimpleNamespace

import pytest
import torch


@contextlib.contextmanager
def _capture_scope(_entry_point: str):
    from vllm_miner.capture import graph_setup_no_mining

    with graph_setup_no_mining():
        yield


def _runner_classes() -> tuple[type[object], type[object]]:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as v2_runner_cls
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as v1_runner_cls

    return v1_runner_cls, v2_runner_cls


def _worker_class() -> type[object]:
    from vllm.v1.worker.gpu_worker import Worker

    return Worker


def _is_wrapped(cls: type[object], method_name: str) -> bool:
    return bool(getattr(getattr(cls, method_name), "_pearl_wrapped_once", False))


def test_iter_gpu_model_runner_classes_yields_both():
    from vllm_miner.runtime import iter_gpu_model_runner_classes

    assert tuple(iter_gpu_model_runner_classes()) == _runner_classes()


def test_worker_phase_hooks_choose_hard_profile_and_soft_warmup_gates(monkeypatch):
    from vllm_miner import runtime
    from vllm_miner.capture import (
        gpu_mining_producer,
        in_graph_setup_no_mining,
        mining_launches_suspended,
        open_mining_admission,
    )

    observed: list[tuple[str, bool, bool, bool]] = []
    open_mining_admission()

    def record(phase):
        def original(_self):
            with gpu_mining_producer() as admitted:
                observed.append(
                    (
                        phase,
                        in_graph_setup_no_mining(),
                        mining_launches_suspended(),
                        admitted,
                    )
                )
            return 0

        return original

    worker_cls = _worker_class()
    monkeypatch.setattr(worker_cls, "determine_available_memory", record("profile"))
    monkeypatch.setattr(worker_cls, "compile_or_warm_up_model", record("warmup"))
    monkeypatch.setattr(runtime, "mining_suspended_for_capture", _capture_scope)
    monkeypatch.setattr(runtime, "_prepare_mining", lambda: record("ready")(None))

    runtime._install_capture_phase_hook()
    worker = object.__new__(worker_cls)
    worker.determine_available_memory()
    worker.compile_or_warm_up_model()

    assert observed == [
        ("profile", True, False, False),
        ("warmup", False, True, True),
        ("ready", False, True, True),
    ]


def test_local_shard_loading_defers_b_preparation_until_worker_readiness(monkeypatch):
    import vllm.distributed as distributed
    from vllm_miner import job_prep
    from vllm_miner import vllm_pearl_config as quant_config

    state = SimpleNamespace(
        weight=torch.zeros(128, 512, dtype=torch.int8),
        weight_scale=torch.ones(128, dtype=torch.bfloat16),
        mineable=True,
    )
    layer = SimpleNamespace(
        weight=SimpleNamespace(data=torch.zeros(128, 512, dtype=torch.bfloat16)),
        register_parameter=lambda *_args: None,
        parameters=lambda recurse=False: iter([torch.zeros(1)]),
    )
    registered = []
    monkeypatch.setattr(
        distributed,
        "get_tensor_model_parallel_world_size",
        lambda: pytest.fail("local shard selection queried global TP topology"),
    )
    monkeypatch.setattr(quant_config, "can_mine_layer", lambda *_args: True)
    monkeypatch.setattr(quant_config, "create_layer_state", lambda *_args: state)
    monkeypatch.setattr(quant_config, "register_state", registered.append)
    monkeypatch.setattr(
        quant_config,
        "suspend_submissions_until_capture_complete",
        lambda: None,
    )
    monkeypatch.setattr(
        job_prep,
        "prepare_layer",
        lambda *_args: pytest.fail("layer loading started B preparation before readiness"),
    )
    monkeypatch.setattr(
        job_prep,
        "prepare_mining",
        lambda *_args: pytest.fail("layer loading started B preparation before readiness"),
    )

    method = quant_config.PearlLinearMethod("model.layers.0.self_attn.qkv_proj")
    # create_weights() records the (n, k) the mineability gate and the encode
    # shape check key on; this layer is a stub, so mirror its weight shape.
    method._n, method._k = 128, 512
    method.process_weights_after_loading(layer)

    assert registered == [state]


def test_encode_refuses_a_weight_that_does_not_match_the_gated_shape(monkeypatch):
    from vllm_miner import vllm_pearl_config as quant_config

    layer = SimpleNamespace(
        weight=SimpleNamespace(data=torch.zeros(128, 512, dtype=torch.bfloat16)),
        parameters=lambda recurse=False: iter([torch.zeros(1)]),
    )
    monkeypatch.setattr(quant_config, "can_mine_layer", lambda *_args: True)
    monkeypatch.setattr(
        quant_config,
        "create_layer_state",
        lambda *_args: pytest.fail("encoded a weight the mineability gate never vetted"),
    )

    method = quant_config.PearlLinearMethod("model.layers.0.mlp.down_proj")
    method._n, method._k = 128, 256  # disagrees with the layer's 512-wide weight

    with pytest.raises(ValueError, match="was gated as mineable"):
        method.process_weights_after_loading(layer)


def test_blocked_weight_probe_follows_the_delegate():
    from vllm_miner import vllm_pearl_config as quant_config

    # No delegate (plain BF16): must look unblocked to vLLM, and must not report
    # a misleading ``weight_block_size`` of None either.
    plain = quant_config.PearlConfig()
    assert not hasattr(plain, "weight_block_size")
    assert plain.has_blocked_weights() is False

    # fp8-style delegate: answers through ``weight_block_size``.
    blocked = quant_config.PearlConfig(source_config=SimpleNamespace(weight_block_size=[128, 128]))
    assert blocked.weight_block_size == [128, 128]
    assert blocked.has_blocked_weights() is True

    per_tensor = quant_config.PearlConfig(source_config=SimpleNamespace(weight_block_size=None))
    assert per_tensor.weight_block_size is None
    assert per_tensor.has_blocked_weights() is False

    # compressed-tensors-style delegate: answers only through the method, so the
    # attribute must stay absent for vLLM's hasattr-first probe to reach it.
    ct = quant_config.PearlConfig(source_config=SimpleNamespace(has_blocked_weights=lambda: True))
    assert not hasattr(ct, "weight_block_size")
    assert ct.has_blocked_weights() is True


def test_source_delegate_extends_rather_than_replaces_ignored_suffixes():
    from vllm_miner import vllm_pearl_config as quant_config

    inherited = tuple(quant_config.PearlConfig()._ignore_unexpected_suffixes)
    assert inherited  # vLLM ships KV-cache scale suffixes on the base config

    config = quant_config.PearlConfig(
        source_config=SimpleNamespace(_ignore_unexpected_suffixes=[".weight_shape"])
    )

    assert set(inherited) <= set(config._ignore_unexpected_suffixes)
    assert ".weight_shape" in config._ignore_unexpected_suffixes


def _packed_source_layer() -> torch.nn.Module:
    """A layer as a quantized delegate leaves it: packed weight, scales, and a
    kernel buffer, plus the layer-owned bias."""
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(torch.zeros(128, 512, dtype=torch.uint8), requires_grad=False)
    )
    layer.register_parameter(
        "weight_scale", torch.nn.Parameter(torch.ones(128, 1), requires_grad=False)
    )
    layer.register_parameter(
        "bias", torch.nn.Parameter(torch.zeros(128, dtype=torch.bfloat16), requires_grad=False)
    )
    layer.register_buffer("workspace", torch.zeros(4))
    return layer


def _fake_source_delegate(n: int) -> SimpleNamespace:
    return SimpleNamespace(
        process_weights_after_loading=lambda _layer: None,
        apply=lambda _layer, x, bias=None: torch.zeros(x.shape[0], n, dtype=torch.bfloat16),
    )


def test_failed_state_install_restores_the_dropped_source_tensors(monkeypatch):
    # The upcast path frees the packed source tensors before encoding, so a
    # failure while installing/registering must put them all back rather than
    # leaving a layer with no weights at all.
    from vllm_miner import vllm_pearl_config as quant_config

    layer = _packed_source_layer()
    packed = layer.weight
    scale = layer.weight_scale
    bias = layer.bias
    workspace = layer.workspace

    state = SimpleNamespace(
        weight=torch.zeros(128, 512, dtype=torch.int8),
        weight_scale=torch.ones(128, dtype=torch.bfloat16),
        mineable=True,
    )
    monkeypatch.setattr(quant_config, "can_mine_layer", lambda *_args: True)
    monkeypatch.setattr(quant_config, "create_layer_state", lambda *_args: state)
    monkeypatch.setattr(quant_config, "unregister_state", lambda *_args: None)
    monkeypatch.setattr(
        quant_config,
        "register_state",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("registry rejected the state")),
    )

    method = quant_config.PearlLinearMethod(
        "model.layers.0.mlp.down_proj", source=_fake_source_delegate(128)
    )
    method._n, method._k = 128, 512

    with pytest.raises(RuntimeError, match="registry rejected"):
        method.process_weights_after_loading(layer)

    assert layer.weight is packed
    assert layer.weight_scale is scale
    assert layer.bias is bias
    assert layer.workspace is workspace
    assert method._state is None


def test_device_is_taken_from_the_weight_not_an_earlier_registered_param():
    # A delegate can register a scale/bias before the packed weight, and those
    # can live on a different device. _device must follow the weight (what the
    # delegate's apply reads and where reconstruct_bf16 must build its identity),
    # not whichever parameter happens to be registered first.
    from vllm_miner import vllm_pearl_config as quant_config

    layer = torch.nn.Module()
    # Registered first, on "meta" to stand in for a differently-placed param.
    layer.register_parameter(
        "input_scale", torch.nn.Parameter(torch.ones(1, device="meta"), requires_grad=False)
    )
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(torch.zeros(4, 6, dtype=torch.bfloat16), requires_grad=False),
    )

    method = quant_config.PearlLinearMethod("model.layers.0.mlp.down_proj")

    assert method._device(layer) == layer.weight.device


@pytest.mark.parametrize(
    "block_size", [[0, 128], [128, -1], [128], [128, 128, 128], "128", [128, 1.5], [128, True]]
)
def test_source_config_rejects_a_malformed_weight_block_size(block_size):
    # vLLM's from_config accepts these and only trips inside a kernel, long
    # after the delegate has allocated parameters from the bad block shape.
    from vllm_miner import vllm_pearl_config as quant_config

    with pytest.raises(ValueError, match="weight_block_size"):
        quant_config._build_source_config(
            {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": block_size}
        )


def test_source_config_builds_a_delegate_for_a_well_formed_fp8_checkpoint():
    from vllm_miner import vllm_pearl_config as quant_config

    config = quant_config._build_source_config(
        {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}
    )

    assert config is not None
    assert config.weight_block_size == [128, 128]
    # Plain BF16 checkpoints carry no quantization config at all.
    assert quant_config._build_source_config({}) is None


def test_source_config_refuses_an_undelegatable_quant_method():
    from vllm_miner import vllm_pearl_config as quant_config

    with pytest.raises(ValueError, match="could not be "):
        quant_config._build_source_config({"quant_method": "not-a-real-quant-method"})


def test_quantized_source_returning_unquantized_for_linearbase_is_rejected():
    # A quantized checkpoint's delegate that returns UnquantizedLinearMethod for
    # a LinearBase it did not claim (e.g. a fused target missing from
    # packed_modules_mapping) leaves a packed (uint8 + scales) weight under
    # layer.weight. Mining that as if it were BF16 would encode garbage; the
    # config must fail closed instead of wrapping a sourceless PearlLinearMethod.
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm_miner import vllm_pearl_config as quant_config

    monkeypatch_target = "vllm_miner.vllm_pearl_config.is_layer_ignored"
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(monkeypatch_target, lambda *_a: False)
        config = quant_config.PearlConfig(
            source_config=SimpleNamespace(
                get_quant_method=lambda _layer, _prefix: UnquantizedLinearMethod(),
                packed_modules_mapping={},
                _ignore_unexpected_suffixes=(),
            )
        )
        layer = LinearBase(64, 64, disable_tp=True)  # a real LinearBase
        assert isinstance(layer, LinearBase)

        with pytest.raises(ValueError, match="refusing to mine it as BF16"):
            config.get_quant_method(layer, "model.layers.0.mlp.gate_up_proj")


def test_plain_bf16_source_returning_unquantized_still_mines():
    # The failure case above is gated on a quantized source being present: a
    # plain BF16 checkpoint (no source_config) returning UnquantizedLinearMethod
    # is the normal path and must still wrap a sourceless PearlLinearMethod.
    from vllm.model_executor.layers.linear import LinearBase
    from vllm_miner import vllm_pearl_config as quant_config

    monkeypatch_target = "vllm_miner.vllm_pearl_config.is_layer_ignored"
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(monkeypatch_target, lambda *_a: False)
        config = quant_config.PearlConfig()  # source_config=None -> plain BF16
        layer = LinearBase(64, 64, disable_tp=True)
        assert isinstance(layer, LinearBase)

        method = config.get_quant_method(layer, "model.layers.0.mlp.gate_up_proj")

        assert isinstance(method, quant_config.PearlLinearMethod)
        assert method.source is None


def test_unmineable_local_shard_falls_through_without_mining_state(monkeypatch):
    from vllm_miner import vllm_pearl_config as quant_config

    fallthrough: list[object] = []
    layer = SimpleNamespace(
        weight=SimpleNamespace(data=torch.zeros(64, 512, dtype=torch.bfloat16)),
        parameters=lambda recurse=False: iter([torch.zeros(1)]),
    )
    monkeypatch.setattr(quant_config, "can_mine_layer", lambda *_args: False)
    monkeypatch.setattr(
        quant_config.UnquantizedLinearMethod,
        "process_weights_after_loading",
        lambda _self, value: fallthrough.append(value),
    )
    monkeypatch.setattr(
        quant_config,
        "create_layer_state",
        lambda *_args: pytest.fail("unmineable shard created mining state"),
    )
    monkeypatch.setattr(
        quant_config,
        "register_state",
        lambda *_args: pytest.fail("unmineable shard registered mining state"),
    )

    method = quant_config.PearlLinearMethod("model.layers.0.self_attn.qkv_proj")
    method._n, method._k = 64, 512
    method.process_weights_after_loading(layer)

    assert fallthrough == [layer]
    assert method._state is None


def test_capture_phase_hook_wraps_both_runners():
    from vllm_miner.runtime import _install_capture_phase_hook

    _install_capture_phase_hook()

    for cls in _runner_classes():
        assert _is_wrapped(cls, "capture_model"), cls
        assert _is_wrapped(cls, "profile_cudagraph_memory"), cls
    assert _is_wrapped(_worker_class(), "determine_available_memory")
    assert _is_wrapped(_worker_class(), "compile_or_warm_up_model")


def test_worker_shutdown_runs_native_cleanup_after_admission_close_failure(monkeypatch):
    events = []
    from vllm_miner import runtime

    def fail_admission_close():
        events.append("close-admission")
        raise RuntimeError("close failed")

    monkeypatch.setattr(runtime, "close_mining_admission", fail_admission_close)
    monkeypatch.setattr(runtime, "delete_state", lambda: events.append("delete-state"))
    monkeypatch.setattr(
        "vllm_miner.state.clear_state_registry",
        lambda: events.append("clear-registry"),
    )
    wrapped = runtime._make_worker_shutdown_wrapper(
        lambda _self: events.append("native-shutdown") or "stopped"
    )

    with pytest.raises(RuntimeError, match="close failed"):
        wrapped(object())

    assert events == [
        "close-admission",
        "delete-state",
        "clear-registry",
        "native-shutdown",
    ]


def test_worker_shutdown_preserves_native_failure_after_mining_failure(monkeypatch):
    from vllm_miner import runtime
    from vllm_miner import state as runtime_state

    events = []

    def fail_mining_cleanup():
        events.append("delete-state")
        raise RuntimeError("mining cleanup failed")

    def fail_native_cleanup(_self):
        events.append("native-shutdown")
        raise ValueError("native cleanup failed")

    monkeypatch.setattr(runtime, "delete_state", fail_mining_cleanup)
    monkeypatch.setattr(
        runtime_state,
        "clear_state_registry",
        lambda: events.append("clear-registry"),
    )
    wrapped = runtime._make_worker_shutdown_wrapper(fail_native_cleanup)

    with pytest.raises(ValueError, match="native cleanup failed"):
        wrapped(object())

    assert events == ["delete-state", "clear-registry", "native-shutdown"]


def test_worker_memory_profile_wrapper_uses_the_hard_gate(monkeypatch):
    from vllm_miner import runtime
    from vllm_miner.capture import (
        gpu_mining_producer,
        in_graph_setup_no_mining,
    )

    observed: list[tuple[bool, bool]] = []

    def original(_self):
        with gpu_mining_producer() as admitted:
            observed.append((in_graph_setup_no_mining(), admitted))
        return "profiled"

    monkeypatch.setattr(runtime, "mining_suspended_for_capture", _capture_scope)
    wrapped = runtime._make_memory_profile_wrapper("determine_available_memory")(original)

    assert wrapped(object()) == "profiled"
    assert observed == [(True, False)]


@pytest.mark.parametrize(("available", "reserve", "expected"), [(1000, 400, 600), (100, 400, 0)])
def test_worker_memory_profile_reserves_bounded_mining_work(
    monkeypatch, available, reserve, expected
):
    from vllm_miner import memory, runtime

    monkeypatch.setattr(runtime, "mining_suspended_for_capture", _capture_scope)
    monkeypatch.setattr(memory, "estimate_peak_mining_bytes", lambda: reserve)
    wrapped = runtime._make_memory_profile_wrapper(
        "determine_available_memory", reserve_mining_memory=True
    )(lambda _self: available)

    assert wrapped(object()) == expected


def test_capture_timeout_refuses_to_call_vllm_capture(monkeypatch):
    """A quiescence deadline is a capture-safety failure, not a warning."""
    import vllm_miner.capture as capture
    from vllm_miner import runtime
    from vllm_miner.capture import in_graph_setup_no_mining

    events: list[object] = []

    def never_idle(timeout: float) -> bool:
        events.append(("wait", timeout, in_graph_setup_no_mining()))
        return False

    def original(_self):
        events.append("capture")

    monkeypatch.setattr(capture, "_producer_permanently_suspended", False)
    monkeypatch.setattr(capture, "wait_for_mining_producers_idle", never_idle)
    monkeypatch.setattr(
        runtime,
        "resume_submissions_after_capture",
        lambda: events.append("resume-submissions"),
    )

    wrapped = runtime._make_capture_model_wrapper(original)
    with pytest.raises(TimeoutError, match="vLLM CUDA-graph capture"):
        wrapped(object())

    assert events == [
        ("wait", 60.0, True),
    ]
    assert not in_graph_setup_no_mining()
    assert capture.mining_launches_suspended()
    with capture.gpu_mining_producer() as admitted:
        assert not admitted


def test_successful_capture_reopens_submissions(monkeypatch):
    from vllm_miner import runtime

    events = []

    monkeypatch.setattr(runtime, "mining_suspended_for_capture", _capture_scope)
    monkeypatch.setattr(
        runtime,
        "resume_submissions_after_capture",
        lambda: events.append("resume-submissions"),
    )
    wrapped = runtime._make_capture_model_wrapper(
        lambda _self: events.append("capture") or "captured"
    )

    assert wrapped(object()) == "captured"
    assert events == ["capture", "resume-submissions"]


def test_worker_warmup_wrapper_suppresses_mined_launches(monkeypatch):
    from vllm_miner.capture import mining_launches_suspended
    from vllm_miner.runtime import _make_worker_warmup_wrapper

    observed: list[bool] = []
    ready: list[bool] = []
    monkeypatch.setattr(
        "vllm_miner.runtime._prepare_mining",
        lambda: ready.append(mining_launches_suspended()),
    )

    def original(_self):
        observed.append(mining_launches_suspended())
        return "done"

    wrapped = _make_worker_warmup_wrapper(original)
    assert wrapped(object()) == "done"
    assert observed == [True]
    assert ready == [True]
    assert not mining_launches_suspended()


def test_serving_activity_hook_wraps_both_runners():
    from vllm_miner.runtime import _install_serving_activity_hook

    _install_serving_activity_hook()

    for cls in _runner_classes():
        assert _is_wrapped(cls, "execute_model"), cls


def test_none_cudagraph_mode_never_pre_suspends_launches(monkeypatch):
    from vllm_miner import capture as gate

    monkeypatch.setattr(gate, "_capture_launch_suspended", False)
    monkeypatch.setattr(gate, "_will_capture_cuda_graphs", lambda: False)

    gate.suspend_submissions_until_capture_complete()
    assert gate._capture_launch_suspended is False


def test_first_real_execute_resumes_when_capture_hook_never_runs(monkeypatch):
    from vllm_miner import runtime

    resumed: list[bool] = []

    class FakeRunner:
        def execute_model(self, scheduler_output, *args, **kwargs):
            return "executed"

    monkeypatch.setattr(runtime, "iter_gpu_model_runner_classes", lambda: iter((FakeRunner,)))
    monkeypatch.setattr(runtime, "resume_submissions_after_capture", lambda: resumed.append(True))

    runtime._install_serving_activity_hook()
    runner = FakeRunner()
    scheduled = SimpleNamespace(total_num_scheduled_tokens=1)
    assert runner.execute_model(scheduled, dummy_run=True) == "executed"
    assert resumed == []
    assert runner.execute_model(scheduled) == "executed"
    assert resumed == [True]
