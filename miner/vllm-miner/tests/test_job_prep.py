"""Per-job B preparation of a mined layer (``vllm_miner.job_prep``).

What a job change costs must depend on *what* changed: a target-only change
rewrites the 32-byte threshold in place and reuses every B-side artifact, a
header change reruns the GPU B chain, and an unchanged job does nothing.

Host logic only. The expensive GPU B-preparation collaborator is stubbed and
counted.
"""

import itertools
from types import SimpleNamespace

import pytest
import torch
from miner_base.commitment_hash import noise_seed_b
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.dataclasses import MiningJob
from vllm_miner import health, job_prep
from vllm_miner import settings as settings_module
from vllm_miner.job_prep import (
    current_context,
    current_job,
    layer_job_keys,
    prepare_layer,
    prepare_mining,
    unpublish_contexts,
)
from vllm_miner.mining_config import (
    PACKED_NOISE_K,
    RANK,
    mining_configuration,
    threshold_bytes_for,
)
from vllm_miner.settings import RuntimeSettings
from vllm_miner.state import (
    LayerBuffers,
    LayerState,
    register_state,
    unregister_state,
)

_N, _K = 128, 2048
# Past the 4x64 tile's verifier limit (30720), so this shape commits the tall
# 16x32 tile instead of the preferred 4x64.
_TALL_K = 43520
_HEADER = bytes(range(80))
_OTHER_HEADER = bytes(range(1, 81))
_LAYER_IDS = itertools.count(1)

_ALIASED_OPERANDS = (
    "key_a_dev",
    "seed_b_dev",
    "threshold_dev",
    "f2",
    "e2",
    "beta_b",
    "b_prime",
    "b_peel",
    "alpha_b",
    "inv_alpha_b",
)


def _job(target: int = 100, header: bytes = _HEADER) -> MiningJob:
    return MiningJob(
        incomplete_header_bytes=header,
        target=target,
        cert_version=CertificateVersion.PLAIN_FP8,
    )


def _buffers(k: int = _K) -> LayerBuffers:
    return LayerBuffers(
        b_prime=torch.zeros(_N, k, dtype=torch.float8_e4m3fn),
        b_peel=torch.zeros(_N, 2 * RANK, dtype=torch.bfloat16),
        alpha_b=torch.ones(_N, dtype=torch.bfloat16),
        beta_b=torch.zeros(_N, dtype=torch.bfloat16),
        inv_alpha_b=torch.ones(_N, dtype=torch.float32),
        e2=torch.zeros(_N, RANK, dtype=torch.float8_e4m3fn),
        f1=torch.zeros(RANK, k, dtype=torch.float8_e4m3fn),
        f2=torch.zeros(RANK, k, dtype=torch.float8_e4m3fn),
        f2_hl=torch.zeros(k, PACKED_NOISE_K, dtype=torch.int8),
        noise_lines=torch.zeros(k, RANK, dtype=torch.float8_e4m3fn),
        root_codes=torch.zeros(32, dtype=torch.uint8),
        root_scales=torch.zeros(32, dtype=torch.uint8),
        tensor_hash_workspace=torch.zeros(1, dtype=torch.uint8),
        commit_stats=torch.zeros(2 * (_N * k // 512), dtype=torch.float32),
        gram=torch.zeros(1),
        commit_config=SimpleNamespace(chunk_size=1024),
        prepare_config=object(),
        key_a_dev=torch.zeros(32, dtype=torch.uint8),
        key_b_dev=torch.zeros(32, dtype=torch.uint8),
        seed_b_dev=torch.zeros(32, dtype=torch.uint8),
        noise_key_b_dev=torch.zeros(32, dtype=torch.uint8),
        threshold_dev=torch.zeros(32, dtype=torch.uint8),
    )


def _layer_state(k: int = _K) -> LayerState:
    """A registered mined layer with no published context yet (post-load state)."""
    weight = torch.zeros(_N, k, dtype=torch.int8)
    return LayerState(
        layer_name=f"test.mined.layer.{k}",
        layer_id=next(_LAYER_IDS),
        weight=weight,
        weight_scale=torch.ones(_N, k // 8, dtype=torch.bfloat16),
        weight_cpu=weight,
        weight_scale_cpu=torch.ones(_N, k // 8, dtype=torch.bfloat16),
        n=_N,
        k=k,
        mineable=True,
        buffers=_buffers(k),
        w_fp8=torch.zeros(_N, k, dtype=torch.float8_e4m3fn),
        w_fp8_scale=torch.ones(1, _N, dtype=torch.float32),
    )


@pytest.fixture(autouse=True)
def host_only_runtime(monkeypatch):
    """No CUDA device to pin; kernel warmup is not part of the preparation decision."""
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(job_prep.gpu_config, "settings", SimpleNamespace(no_mining=False))
    monkeypatch.setattr(settings_module, "_settings", RuntimeSettings(warmup_compile=False))
    health._BREAKER.reset_for_tests()
    yield
    health._BREAKER.reset_for_tests()


@pytest.fixture
def preps(monkeypatch) -> list[tuple[LayerState, bytes]]:
    """Stub the GPU B chain and record ``(layer, keyB)`` per full preparation.

    The stub derives ``seedB`` from the key the way the real chain does (over
    a fixed stand-in commitment), so seed assertions are meaningful.
    """
    recorded: list[tuple[LayerState, bytes]] = []

    def fake_prepare(state, key_a, key_b, p_b, buffers) -> bytes:
        recorded.append((state, key_b))
        buffers.key_a_dev.copy_(torch.frombuffer(bytearray(key_a), dtype=torch.uint8))
        seed_b = noise_seed_b(b"\x00" * 32, key_b, p_b)
        buffers.seed_b_dev.copy_(torch.frombuffer(bytearray(seed_b), dtype=torch.uint8))
        return seed_b

    monkeypatch.setattr(job_prep, "_prepare_b_on_gpu", fake_prepare)
    return recorded


@pytest.fixture
def layer():
    state = _layer_state()
    register_state(state)
    yield state
    unregister_state(state)


def _threshold_words(state: LayerState) -> bytes:
    assert state.buffers is not None
    return bytes(state.buffers.threshold_dev.numpy())


def test_first_job_runs_the_b_chain_and_publishes_in_place(layer, preps):
    job = _job()
    ctx = current_context(layer, job)

    assert ctx is not None and layer.job_ctx is ctx
    assert [key for _, key in preps] == [layer_job_keys(job)[1]]
    assert ctx.job is job and ctx.target == job.target
    assert ctx.config == mining_configuration(_K, _N)
    assert ctx.seed_b == noise_seed_b(b"\x00" * 32, ctx.key_b, ctx.config.p_b(_N))
    assert ctx.b_proof.commit_leaf == layer.buffers.commit_config.chunk_size
    # Every device operand is the steady buffer itself: launches read the
    # same addresses forever.
    for name in _ALIASED_OPERANDS:
        assert getattr(ctx, name) is getattr(layer.buffers, name), name
    assert _threshold_words(layer) == threshold_bytes_for(job, _K, _N)
    assert bytes(layer.buffers.seed_b_dev.numpy()) == ctx.seed_b


def test_same_header_target_change_rewrites_only_the_threshold(layer, preps):
    """The 32 bytes that changed must not cost a full-model B prep.

    Only the lottery threshold depends on the target: keyA/keyB come from the
    header, seedB from keyB, the commitment and pB, and every B operand from
    seedB. A target-only change therefore rewrites the threshold in place
    (stream-ordered with the launches that read it) and republishes.
    """
    first_job = _job(target=100)
    first = current_context(layer, first_job)
    second_job = _job(target=200)
    published = current_context(layer, second_job)

    assert len(preps) == 1, "a target-only change reran the B chain"
    assert published is not first and layer.job_ctx is published
    assert published.job is second_job and published.target == 200
    assert (published.key_a, published.key_b) == (first.key_a, first.key_b)
    assert published.seed_b == first.seed_b
    assert published.b_proof is first.b_proof
    for name in _ALIASED_OPERANDS:
        assert getattr(published, name) is getattr(first, name), name
    assert _threshold_words(layer) == threshold_bytes_for(second_job, _K, _N)


def test_tall_shape_target_change_reuses_the_b_side(preps):
    """A wide-k layer commits the tall 16x32 tile, whose ``pB`` enters seedB.
    The fast path must compare the keys exactly as the full path derived them
    (n included) or a target-only change would re-prepare precisely the layers
    the tall tile made mineable."""
    state = _layer_state(_TALL_K)
    config = mining_configuration(_TALL_K, state.n)
    assert (config.rows_pattern.tile_size, config.cols_pattern.tile_size) == (16, 32)

    first = prepare_layer(state, _job(target=100))
    published = prepare_layer(state, _job(target=200))

    assert len(preps) == 1, "a target-only change re-prepared the tall B side"
    assert published.seed_b == first.seed_b
    assert _threshold_words(state) == threshold_bytes_for(_job(target=200), _TALL_K, _N)


def test_header_change_reruns_the_b_chain(layer, preps):
    first = current_context(layer, _job())
    published = current_context(layer, _job(header=_OTHER_HEADER))

    assert len(preps) == 2
    assert (published.key_a, published.key_b) != (first.key_a, first.key_b)
    assert published.seed_b != first.seed_b
    assert _threshold_words(layer) == threshold_bytes_for(_job(header=_OTHER_HEADER), _K, _N)


def test_unchanged_job_is_a_noop(layer, preps):
    job = _job()
    ctx = current_context(layer, job)
    # Equal, not identical: the manager republishes equal jobs as new objects.
    again = current_context(layer, _job())

    assert again is ctx
    assert len(preps) == 1


def test_no_job_or_disabled_layer_yields_no_context(layer, preps):
    assert current_context(layer, None) is None
    layer.mineable = False
    assert current_context(layer, _job()) is None
    assert preps == []


def test_preparation_failure_publishes_nothing(layer, monkeypatch):
    def boom(*_args):
        raise RuntimeError("kernel failed")

    monkeypatch.setattr(job_prep, "_prepare_b_on_gpu", boom)
    with pytest.raises(RuntimeError, match="kernel failed"):
        prepare_layer(layer, _job())
    assert layer.job_ctx is None


def test_layer_disabled_during_preparation_is_not_published(layer, monkeypatch):
    def disable_then_prepare(state, key_a, key_b, p_b, buffers) -> bytes:
        state.disable_mining("test")
        return b"\x11" * 32

    monkeypatch.setattr(job_prep, "_prepare_b_on_gpu", disable_then_prepare)
    assert prepare_layer(layer, _job()) is None
    assert layer.job_ctx is None


def test_unpublish_forgets_every_registered_layer(layer, preps):
    current_context(layer, _job())
    other = _layer_state(_TALL_K)
    register_state(other)
    try:
        current_context(other, _job())
        unpublish_contexts()
        assert layer.job_ctx is None and other.job_ctx is None
        # The next launch prepares again from scratch.
        current_context(layer, _job())
        assert len(preps) == 3
    finally:
        unregister_state(other)


def test_current_job_is_none_without_a_runtime(monkeypatch):
    from vllm_miner import mining_state as runtime

    monkeypatch.setattr(runtime, "_async_manager", None)
    assert current_job() is None
    monkeypatch.setattr(runtime, "_runtime_poisoned", True)
    assert current_job() is None


def test_startup_preparation_readies_every_layer(layer, preps, monkeypatch):
    warmed: list[LayerState] = []
    monkeypatch.setattr(settings_module, "_settings", RuntimeSettings(warmup_compile=True))
    monkeypatch.setattr(
        "vllm_miner.pipeline.warmup_layer_variants",
        lambda state, cancelled=None: warmed.append(state) or True,
    )
    job = _job()

    assert prepare_mining(job) is True
    assert warmed == [layer]
    assert layer.job_ctx is not None and layer.job_ctx.job is job


def test_startup_preparation_without_a_job_warms_but_is_not_ready(layer, preps, monkeypatch):
    warmed: list[LayerState] = []
    monkeypatch.setattr(settings_module, "_settings", RuntimeSettings(warmup_compile=True))
    monkeypatch.setattr(
        "vllm_miner.pipeline.warmup_layer_variants",
        lambda state, cancelled=None: warmed.append(state) or True,
    )

    assert prepare_mining(None) is False
    assert warmed == [layer]
    assert layer.job_ctx is None and preps == []


def test_startup_preparation_isolates_a_failing_layer(layer, monkeypatch):
    other = _layer_state(_TALL_K)
    register_state(other)
    try:

        def prepare_or_fail(state, key_a, key_b, p_b, buffers) -> bytes:
            if state is layer:
                raise RuntimeError("deterministic failure")
            return b"\x22" * 32

        monkeypatch.setattr(job_prep, "_prepare_b_on_gpu", prepare_or_fail)
        assert prepare_mining(_job()) is False
        assert not layer.mineable and layer.disabled_reason is not None
        assert other.mineable and other.job_ctx is not None
    finally:
        unregister_state(other)


def test_startup_preparation_incomplete_warmup_disables_only_that_layer(layer, preps, monkeypatch):
    monkeypatch.setattr(settings_module, "_settings", RuntimeSettings(warmup_compile=True))
    monkeypatch.setattr(
        "vllm_miner.pipeline.warmup_layer_variants", lambda state, cancelled=None: False
    )
    assert prepare_mining(_job()) is False
    assert not layer.mineable
    assert preps == []


def test_kill_switch_skips_preparation(layer, preps, monkeypatch):
    monkeypatch.setattr(job_prep.gpu_config, "settings", SimpleNamespace(no_mining=True))
    assert prepare_mining(_job()) is False
    assert preps == [] and layer.job_ctx is None
