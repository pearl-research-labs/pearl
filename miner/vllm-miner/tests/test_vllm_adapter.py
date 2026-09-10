"""vLLM adapter on SM100: pearl method routing, load-time encoding, and
dispatch. No hardware skip guards: runs only on the B200 job."""

import pytest
import torch
from vllm_miner import settings as settings_module
from vllm_miner.settings import RuntimeSettings
from vllm_miner.state import lookup_state, unregister_state

pytestmark = pytest.mark.gpu

_N, _K = 256, 2048


@pytest.fixture(scope="module", autouse=True)
def vllm_default_config():
    """vLLM 0.26 asserts an active config context both when model-parallel
    groups are created and when CustomOp/Linear layers are instantiated;
    hold one open for the module the way vLLM's own tests'
    ``default_vllm_config`` fixture does (tests/conftest.py upstream).
    """
    from vllm.config import VllmConfig, set_current_vllm_config

    config = VllmConfig()
    with set_current_vllm_config(config):
        yield config


@pytest.fixture(scope="module", autouse=True)
def vllm_single_rank(vllm_default_config):
    import torch.distributed
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if not torch.distributed.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://127.0.0.1:29511",
            backend="nccl",
        )
    if not model_parallel_is_initialized():
        initialize_model_parallel(1, 1)
    yield


@pytest.fixture
def mined_settings(monkeypatch):
    settings = RuntimeSettings(
        m_buckets=(256,),
        min_mining_tokens=4,
        warmup_compile=False,
        ignored_layers=("re:.*\\.skipme", "exact.layer"),
    )
    monkeypatch.setattr(settings_module, "_settings", settings)
    return settings


def _linear(prefix: str, quant_config, *, n: int = _N, k: int = _K):
    from vllm.model_executor.layers.linear import ReplicatedLinear

    # params_dtype must be explicit: vLLM otherwise falls back to
    # torch.get_default_dtype() (fp32), which the pearl method refuses.
    return ReplicatedLinear(
        k, n, bias=False, params_dtype=torch.bfloat16, quant_config=quant_config, prefix=prefix
    ).cuda()


def test_get_quant_method_routes_by_layer_and_exclusion(mined_settings):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm_miner.vllm_pearl_config import PearlConfig, PearlLinearMethod

    config = PearlConfig()
    assert config.get_config_filenames() == []
    assert config.get_supported_act_dtypes() == [torch.bfloat16]
    assert PearlConfig.from_config({}).get_name() == "pearl"

    mined = _linear("model.layers.0.mlp.up_proj", config)
    assert isinstance(mined.quant_method, PearlLinearMethod)

    excluded = _linear("model.layers.0.mlp.skipme", config)
    assert type(excluded.quant_method) is UnquantizedLinearMethod
    mla_direct_read = _linear("model.layers.0.self_attn.kv_b_proj", config)
    assert type(mla_direct_read.quant_method) is UnquantizedLinearMethod
    exact_direct_read = _linear("kv_b_proj", config)
    assert type(exact_direct_read.quant_method) is UnquantizedLinearMethod
    near_match = _linear("model.layers.0.self_attn.kv_b_project", config)
    assert isinstance(near_match.quant_method, PearlLinearMethod)

    assert config.get_quant_method(torch.nn.Linear(4, 4), prefix="not.a.vllm.linear") is None


@pytest.mark.parametrize(
    ("shape_kind", "n", "k"),
    [("column-like", 128, 4096), ("row-like", 4096, 2048)],
)
def test_representative_local_shard_encodes_and_dispatches(
    mined_settings, async_manager, shape_kind, n, k
):
    """Exercise the local matrices produced by row/column sharding.

    Distributed collective ordering is exercised separately on multiple ranks;
    this test owns real encoding, registration, dispatch, and serving numerics
    for each process-local operand shape.
    """
    from vllm_miner.vllm_pearl_config import PearlConfig

    torch.manual_seed(7)
    layer = _linear(f"model.layers.1.mlp.{shape_kind}_proj", PearlConfig(), n=n, k=k)
    source = (torch.randn(n, k, dtype=torch.bfloat16) * 1.2).cuda()
    layer.weight.data.copy_(source)
    reference = layer.weight.data.clone()
    state = None

    try:
        layer.quant_method.process_weights_after_loading(layer)
        state = lookup_state(layer.weight)
        assert layer.weight.dtype == torch.int8
        assert layer.weight_scale.dtype == torch.bfloat16
        assert torch.equal(state.weight, layer.weight.data)

        x = torch.randn(4, k, dtype=torch.bfloat16, device="cuda")
        out, _ = layer(x)
        torch.cuda.synchronize()
        assert out.shape == (4, n)
        # No job context exists: the FP8 fallback must approximate x @ W.T.
        expected = x.float() @ reference.float().T
        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), expected.flatten(), dim=0
        )
        assert cos.item() > 0.99
    finally:
        if state is not None:
            unregister_state(state)


def test_checkpoint_reload_refreshes_graph_visible_state_in_place(mined_settings, async_manager):
    from dataclasses import fields

    from vllm_miner.state import LayerBuffers
    from vllm_miner.vllm_pearl_config import PearlConfig

    layer = _linear("model.layers.2.mlp.up_proj", PearlConfig())
    first = torch.randn(_N, _K, dtype=torch.bfloat16, device="cuda")
    second = torch.randn(_N, _K, dtype=torch.bfloat16, device="cuda")
    layer.weight.data.copy_(first)
    state = None
    try:
        layer.quant_method.process_weights_after_loading(layer)
        state = lookup_state(layer.weight)
        pointers = {
            "weight": state.weight.data_ptr(),
            "weight_scale": state.weight_scale.data_ptr(),
            "w_fp8": state.w_fp8.data_ptr(),
            "w_fp8_scale": state.w_fp8_scale.data_ptr(),
        }
        assert state.buffers is not None
        pointers.update(
            {
                f"buffers.{descriptor.name}": value.data_ptr()
                for descriptor in fields(LayerBuffers)
                if isinstance((value := getattr(state.buffers, descriptor.name)), torch.Tensor)
            }
        )

        # Mirror vLLM 0.26 layerwise reload: checkpoint-shaped BF16 parameters
        # are materialized temporarily, processed, then copied to old storage.
        layer.register_parameter("weight", torch.nn.Parameter(second.clone(), requires_grad=False))
        if "weight_scale" in layer._parameters:
            delattr(layer, "weight_scale")
        layer.quant_method.process_weights_after_loading(layer)

        assert lookup_state(layer.weight) is state
        assert state.weight.data_ptr() == pointers["weight"]
        assert state.weight_scale.data_ptr() == pointers["weight_scale"]
        assert state.w_fp8.data_ptr() == pointers["w_fp8"]
        assert state.w_fp8_scale.data_ptr() == pointers["w_fp8_scale"]
        for descriptor in fields(LayerBuffers):
            value = getattr(state.buffers, descriptor.name)
            if isinstance(value, torch.Tensor):
                assert value.data_ptr() == pointers[f"buffers.{descriptor.name}"]

        x = torch.randn(4, _K, dtype=torch.bfloat16, device="cuda")
        out, _ = layer(x)
        expected = x.float() @ second.float().T
        cosine = torch.nn.functional.cosine_similarity(
            out.float().flatten(), expected.flatten(), dim=0
        )
        assert cosine.item() > 0.99
    finally:
        if state is not None:
            unregister_state(state)


def test_runtime_startup_rollback_keeps_initial_weights_unquantized(
    mined_settings, async_manager, monkeypatch
):
    import vllm_miner.mining_state as gpu_runtime
    from vllm_miner.vllm_pearl_config import PearlConfig

    monkeypatch.setattr(gpu_runtime, "mining_disabled_by_runtime_failure", lambda: True)
    layer = _linear("model.layers.3.mlp.up_proj", PearlConfig())
    layer.weight.data.normal_()
    reference = layer.weight.data.clone()
    x = torch.randn(4, _K, dtype=torch.bfloat16, device="cuda")

    layer.quant_method.process_weights_after_loading(layer)
    out, _ = layer(x)

    assert layer.weight.dtype == torch.bfloat16
    assert not hasattr(layer, "weight_scale")
    expected = x.float() @ reference.float().T
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), expected.flatten(), dim=0)
    assert cos.item() > 0.999


def test_layer_state_creation_failure_stays_unquantized(mined_settings, async_manager, monkeypatch):
    from vllm_miner import vllm_pearl_config as pearl_config

    layer = _linear("model.layers.4.mlp.up_proj", pearl_config.PearlConfig())
    layer.weight.data.normal_()
    reference = layer.weight.data.clone()
    monkeypatch.setattr(
        pearl_config,
        "create_layer_state",
        lambda *_args: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("injected")),
    )

    layer.quant_method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.bfloat16
    assert not hasattr(layer, "weight_scale")
    torch.testing.assert_close(layer.weight.data, reference)


@pytest.mark.parametrize(("k", "n"), [(1000, 24), (2048, 48), (1024, 64)])
def test_unmineable_shapes_stay_unquantized(mined_settings, async_manager, k, n):
    """Unsupported (misaligned k), encodable-but-unmineable (n % 32, so below
    even the tall 16x32 tile's alignment) and below-the-cert-v4-floor
    (k < 2048) shapes must all keep the original BF16 path: encoding without
    mining is pure loss."""
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm_miner.vllm_pearl_config import PearlConfig

    layer = ReplicatedLinear(
        k,
        n,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=PearlConfig(),
        prefix="model.odd_proj",
    ).cuda()
    # vLLM allocates weights uninitialized; give the forward finite values the
    # way a checkpoint load would (recycled GPU memory can even decode as NaN).
    torch.manual_seed(11)
    layer.weight.data.copy_(torch.randn(n, k, dtype=torch.bfloat16, device="cuda"))
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.bfloat16  # untouched: serves unquantized

    x = torch.randn(2, k, dtype=torch.bfloat16, device="cuda")
    out, _ = layer(x)
    expected = x.float() @ layer.weight.data.float().T
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
