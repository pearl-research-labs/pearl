"""Mined-linear runtime on Blackwell: load-time encoding, per-job B preparation
parity, launch accounting, winner round-trip, and staleness.

Intentionally no hardware skip guards: this suite runs only on the B200 job.
"""

import hashlib
import json
import threading
from pathlib import Path

import pytest
import torch
from miner_base.block_submission import commit_planes_for_leaf
from miner_base.commitment import BlockHeader
from miner_base.commitment_hash import noise_seed_b
from miner_base.prequant import PrequantMatrix
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.dataclasses import MiningJob
from pearl_gemm import pack_noise_factor, supports_lottery_family
from vllm_miner import settings as settings_module
from vllm_miner.job_prep import prepare_layer
from vllm_miner.mining_config import (
    RANK,
    commitment_keys_for,
    effective_work_per_matmul,
    mining_configuration,
    threshold_bytes_for,
)
from vllm_miner.settings import RuntimeSettings
from vllm_miner.state import (
    create_layer_state,
    lookup_state,
    register_state,
    supports_layer_shape,
    unregister_state,
)

pytestmark = pytest.mark.gpu

_M_BUCKET = 256
_N, _K = 256, 2048
_TALL_K = 16384  # GLM-5.2 o_proj: high k past the 4x128 proof-size cap (15872)
_MAX_256 = (1 << 256) - 1

# Both mineable shapes commit the preferred 4x64 tile (4x128 is now only the
# n-omitted back-compat commitment, never used by the runtime). The o_proj entry
# carries the real GLM-5.2 shape -- (6144, 16384) -- so it exercises the true
# wide-n allocation/indexing path plus the high-k B-build/kernel path the 4x128
# proof-size cap (15872) could not reach. ``(name, n, k)``; used to parametrize
# the shared shape checks below.
_MINED_SHAPES = {
    "low_k": ("test.mined.layer", _N, _K),
    "o_proj_high_k": ("test.mined.o_proj", 6144, _TALL_K),
}
_MINED_SHAPE_PARAMS = list(_MINED_SHAPES.values())
_MINED_SHAPE_IDS = list(_MINED_SHAPES)


def _header() -> BlockHeader:
    return BlockHeader(
        version=1,
        prev_block=b"\x11" * 32,
        merkle_root=b"\x22" * 32,
        timestamp=1_700_000_000,
        nbits=0x1D3FFFFF,
    )


def _job(
    target: int = 1,
    cert_version: CertificateVersion = CertificateVersion.PLAIN_FP8,
) -> MiningJob:
    # target=1: winning is effectively impossible, keeping runs deterministic.
    return MiningJob(
        incomplete_header_bytes=bytes(_header().to_bytes()),
        target=target,
        cert_version=cert_version,
    )


def _wait_for_polled_job(async_manager, monkeypatch, job: MiningJob, *, no_gateway: bool) -> None:
    async_manager._conf = async_manager._conf.model_copy(update={"no_gateway": no_gateway})
    assert async_manager._client is not None
    published = threading.Event()

    def mark_published() -> None:
        if async_manager.get_mining_job() == job:
            published.set()

    async_manager.register_mining_job_changed_callback(mark_published)
    monkeypatch.setattr(async_manager._client, "get_mining_info", lambda: job)
    assert published.wait(timeout=5), "patched gateway job was not published"
    assert async_manager.get_mining_job() == job


@pytest.fixture
def mined_settings(monkeypatch):
    settings = RuntimeSettings(
        m_buckets=(_M_BUCKET,),
        min_mining_tokens=4,
        warmup_compile=False,
    )
    monkeypatch.setattr(settings_module, "_settings", settings)
    return settings


@pytest.fixture
def layer_state(request, mined_settings):
    # Unparametrized users get the low-k default shape (256 x 2048); the shared
    # shape checks parametrize this indirectly with ``_MINED_SHAPE_PARAMS``.
    name, n, k = getattr(request, "param", _MINED_SHAPES["low_k"])
    torch.manual_seed(3)
    state = create_layer_state(name, (torch.randn(n, k, dtype=torch.bfloat16) * 1.5).cuda())
    register_state(state)
    yield state
    unregister_state(state)
    from vllm_miner import pipeline

    pipeline._HIT_SIGNALS.pop(state.weight.device.index or 0, None)


def _cpu_pq(state) -> PrequantMatrix:
    return PrequantMatrix(state.weight_cpu, state.weight_scale_cpu)


# SHA-256 digests of the protocol's job-constant B operands for the
# ``layer_state`` weights, recorded from the former bit-exact CPU reference
# implementation. Each shape
# also stores ``peel_tol``, the mid peel half's bound against its fp64-exact
# product.
_B_DIGESTS = json.loads(
    (Path(__file__).parent / "fixtures" / "b_preparation_digests.json").read_text()
)


def _digest(t: torch.Tensor) -> str:
    """SHA-256 of ``t``'s values: FP8 as raw bytes, other floats as float32
    with ``-0.0`` folded into ``+0.0``."""
    t = t.detach().cpu().contiguous()
    data = t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t.float() + 0.0
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# state / load-time encoding
# --------------------------------------------------------------------------- #


def test_create_layer_state_encodes_reference_planes(layer_state):
    torch.manual_seed(3)
    source = torch.randn(_N, _K, dtype=torch.bfloat16) * 1.5
    reference = PrequantMatrix.encode(source)

    # GPU encoding must be bit-identical to the CPU consensus encoder.
    assert torch.equal(layer_state.weight.cpu(), reference.int_values)
    assert torch.equal(layer_state.weight_scale.cpu(), reference.scales)
    # Host copies mirror the committed planes exactly.
    assert torch.equal(layer_state.weight_cpu, reference.int_values)
    assert torch.equal(layer_state.weight_scale_cpu, reference.scales)
    assert layer_state.mineable
    assert layer_state.buffers is not None
    assert lookup_state(layer_state.weight) is layer_state


def test_create_layer_state_validates_input(mined_settings):
    with pytest.raises(ValueError, match="bfloat16"):
        create_layer_state("bad.dtype", torch.zeros(64, 512, dtype=torch.int8, device="cuda"))
    with pytest.raises(ValueError, match="unsupported"):
        create_layer_state("bad.shape", torch.zeros(60, 1032, dtype=torch.bfloat16, device="cuda"))
    assert supports_layer_shape(256, 1024)
    assert not supports_layer_shape(255, 1024)
    assert not supports_layer_shape(256, 1000)


def test_fp8_fallback_matches_opened_planes(layer_state):
    from vllm_miner.fp8_fallback import fp8_fallback_gemm

    torch.manual_seed(5)
    x = torch.randn(8, _K, dtype=torch.bfloat16, device="cuda")
    out = fp8_fallback_gemm(x, layer_state.w_fp8, layer_state.w_fp8_scale, None, torch.bfloat16)
    opened = _cpu_pq(layer_state).open().to(torch.float32)
    expected = x.cpu().to(torch.float32) @ opened.T
    err = out.cpu().to(torch.float32) - expected
    # e4m3 keeps 3 mantissa bits (~2^-4 relative step); two independently
    # quantized operands leave every product term with ~4% relative error that
    # is independent across K, so the output's relative Frobenius error stays
    # at that level rather than averaging away (measured 0.039, K-invariant).
    rel = (err.norm() / expected.norm().clamp_min(1e-30)).item()
    assert rel < 5e-2
    # A norm bound can hide one corrupted element: each element's error is the
    # same ~4%-of-output-RMS quantization noise, so cap its max at ~6 sigma
    # (measured 0.16 across seeds).
    assert err.abs().max().item() < 0.25 * expected.square().mean().sqrt().item()


# --------------------------------------------------------------------------- #
# per-job B preparation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("layer_state", _MINED_SHAPE_PARAMS, ids=_MINED_SHAPE_IDS, indirect=True)
def test_prepare_layer_matches_pinned_commitment_chain(layer_state):
    """Load-time commitment + B preparation parity against the protocol's
    operands, for both the low-k shape and the high-k o_proj shape (k=16384) the 4x128 tile
    could not prove. Both commit the preferred 4x64 tile, which is bound into
    ``pB`` and so into noise seedB -- so the expected seed must pass the layer's
    ``n``."""
    n, k = layer_state.n, layer_state.k
    job = _job()
    ctx = prepare_layer(layer_state, job)

    key_a, key_b = commitment_keys_for(job)
    weight_pq = _cpu_pq(layer_state)
    config = mining_configuration(k, n)
    comm_b = commit_planes_for_leaf(weight_pq.planes(), key_b, config.chunk_size)
    # The committed tile (4x64) is bound into pB, hence into seedB.
    expected_seed_b = noise_seed_b(comm_b.digest, key_b, config.p_b(n))
    assert (ctx.key_a, ctx.key_b) == (key_a, key_b)
    assert ctx.seed_b == expected_seed_b
    assert expected_seed_b != noise_seed_b(comm_b.digest, key_b, mining_configuration(k).p_b(n))

    # Every mineable shape here commits the preferred 4x64 tile.
    assert (config.rows_pattern.tile_size, config.cols_pattern.tile_size) == (4, 64)
    assert config.common_dim == k
    # Cert-v4 can only open the native 1024-byte leaf.
    assert config.chunk_size == 1024
    assert config.a_chunk_size == 1024

    buffers = layer_state.buffers
    assert bytes(buffers.key_a_dev.cpu().numpy()) == key_a
    assert bytes(buffers.key_b_dev.cpu().numpy()) == key_b
    assert bytes(buffers.seed_b_dev.cpu().numpy()) == expected_seed_b
    assert bytes(buffers.threshold_dev.cpu().numpy()) == threshold_bytes_for(job, k, n)

    # Uploaded operands equal the protocol's B side. Both F bases are keyed by
    # seedB, so the whole B side -- the complete peel included -- is a job
    # constant fixed by seedB alone.
    expected = _B_DIGESTS[f"{n}x{k}"]
    assert _digest(buffers.f1) == expected["f1"], "F_A"
    assert _digest(buffers.f2) == expected["f2"], "F_B"
    assert _digest(buffers.e2) == expected["e2"], "E_B"
    assert _digest(buffers.alpha_b) == expected["alpha_b"], "alpha_b"
    assert _digest(buffers.beta_b) == expected["beta_b"], "beta_b"
    assert _digest(buffers.b_prime) == expected["b_prime"], "B'"
    assert torch.equal(buffers.f1_hl.cpu(), pack_noise_factor(buffers.f1.cpu()))
    assert torch.equal(ctx.f1_hl, buffers.f1_hl)
    b_peel = buffers.b_peel.cpu()
    # The element-wise second half (-(beta_b (.) E_B)) is bit-exact; the mid
    # half is the peel matmul, which the fused kernel's epilogue reassociates,
    # so it is held to tolerance against the fp64-exact product of the
    # (pinned) operands.
    assert _digest(b_peel[:, RANK:]) == expected["peel_beta_e"], "-(beta_b (.) E_B)"
    f1_64, f2_64 = buffers.f1.cpu().double(), buffers.f2.cpu().double()
    beta_64 = buffers.beta_b.cpu().double().reshape(-1, 1)
    diff_64 = beta_64 * (buffers.e2.cpu().double() @ f2_64) - buffers.b_prime.cpu().double()
    mid_64 = diff_64 @ f1_64.t()
    mid_relative_error = (
        (b_peel[:, :RANK].double() - mid_64).norm() / mid_64.norm().clamp_min(1e-30)
    ).item()
    assert mid_relative_error < expected["peel_tol"], mid_relative_error
    prebuilt = ctx.b_proof.prebuilt_commitment()
    assert prebuilt.key == key_b
    assert prebuilt.commitment.digest == comm_b.digest

    assert layer_state.job_ctx is ctx


def test_prepare_dispatches_gpu_b_chain(layer_state, monkeypatch):
    """Production B preparation runs the GPU chain, in order."""
    import pearl_gemm

    calls: list[str] = []

    def observed(name, implementation):
        def wrapper(*args, **kwargs):
            calls.append(name)
            return implementation(*args, **kwargs)

        return wrapper

    for kernel_name in ("tensor_hash_plus_stats_b", "noise_lines", "noisy_quant_b"):
        monkeypatch.setattr(
            pearl_gemm,
            kernel_name,
            observed(kernel_name, getattr(pearl_gemm, kernel_name)),
        )

    ctx = prepare_layer(layer_state, _job())
    torch.cuda.synchronize()

    assert ctx is not None
    # Both F bases are keyed by seedB, so B preparation draws F_A and F_B.
    assert calls == [
        "tensor_hash_plus_stats_b",
        "noise_lines",
        "noise_lines",
        "noisy_quant_b",
    ]


def test_prepare_fails_closed_on_gpu_b_failure(layer_state, monkeypatch):
    """A failed GPU B stage must leave the layer unpublished and propagate."""
    import pearl_gemm

    noise_launches = 0

    def fail_after_partial_gpu_chain(*args, **kwargs):
        nonlocal noise_launches
        noise_launches += 1
        raise RuntimeError("synthetic GPU B-preparation failure")

    monkeypatch.setattr(pearl_gemm, "noise_lines", fail_after_partial_gpu_chain)

    with pytest.raises(RuntimeError, match="synthetic GPU B-preparation failure"):
        prepare_layer(layer_state, _job())

    assert noise_launches == 1
    assert layer_state.job_ctx is None


def test_prepare_republishes_on_new_job(layer_state):
    first = prepare_layer(layer_state, _job())
    second = prepare_layer(layer_state, _job(target=7))
    assert (second.key_a, second.key_b) == (first.key_a, first.key_b)  # same header, same keys
    assert second.seed_b == first.seed_b
    assert bytes(layer_state.buffers.threshold_dev.cpu().numpy()) == threshold_bytes_for(
        _job(target=7), _K, _N
    )
    assert layer_state.job_ctx is second


# --------------------------------------------------------------------------- #
# pipeline accounting and winners
# --------------------------------------------------------------------------- #


def test_mine_launch_credits_protocol_hashes(layer_state, async_manager, monkeypatch):
    from vllm_miner.pipeline import mine_launch

    job = _job()
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=True)
    ctx = prepare_layer(layer_state, job)
    before = async_manager._inner_hash_counter
    c = mine_launch(layer_state, ctx, _M_BUCKET, 16, lambda a: a.zero_())
    assert async_manager.wait_until_drained(timeout=30)
    assert c.shape == (_M_BUCKET, _N)
    credited = async_manager._inner_hash_counter - before
    assert credited == effective_work_per_matmul(_M_BUCKET, _N, _K)


def test_failed_completion_record_fences_real_launch_before_fallback(
    layer_state, async_manager, monkeypatch
):
    import vllm_miner.pipeline as pipeline
    from vllm_miner import linear_op
    from vllm_miner.capture import wait_for_mining_producers_idle

    job = _job()
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=True)
    prepare_layer(layer_state, job)
    x = torch.randn(16, _K, dtype=torch.bfloat16, device="cuda")
    record_event = pipeline._record_stream_event
    register_completion = pipeline.register_mining_completion
    attempts = 0
    registered = []

    def fail_first_record():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("synthetic event record failure")
        return record_event()

    def observe_completion(event, keepalive=None):
        registered.append(event)
        register_completion(event, keepalive)

    monkeypatch.setattr(pipeline, "_record_stream_event", fail_first_record)
    monkeypatch.setattr(pipeline, "register_mining_completion", observe_completion)

    output = linear_op._apply_linear_impl(x, layer_state.weight, None)

    assert attempts == 2
    assert len(registered) == 1
    assert wait_for_mining_producers_idle(timeout=30)
    torch.cuda.synchronize()
    assert output.shape == (16, _N)
    assert output.isfinite().all()


def test_failed_warmup_quiesces_before_publishing_failure(layer_state, async_manager, monkeypatch):
    import vllm_miner.pipeline as pipeline

    job = _job()
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=True)
    prepare_layer(layer_state, job)
    launch_stages = pipeline._launch_stages

    class QuiescenceCheckingSet(set):
        def add(self, item):
            assert torch.cuda.current_stream().query()
            super().add(item)

    def fail_after_real_launch(*args, **kwargs):
        launch_stages(*args, **kwargs)
        torch.cuda._sleep(200_000_000)
        raise RuntimeError("synthetic post-enqueue warmup failure")

    failed = QuiescenceCheckingSet()
    monkeypatch.setattr(pipeline, "_ready_variants", set())
    monkeypatch.setattr(pipeline, "_failed_variants", failed)
    monkeypatch.setattr(pipeline, "_launch_stages", fail_after_real_launch)

    assert not pipeline.warmup_layer_variants(layer_state)
    assert (_M_BUCKET, _N, _K, 0, 0) in failed


def test_mine_launch_reuses_and_rearms_the_persistent_hit_signal(
    layer_state, async_manager, monkeypatch
):
    """Two guaranteed hits reuse one real HitSignal and both leave it armed."""
    import pearl_gemm
    import vllm_miner.pipeline as pipeline
    from vllm_miner.winners import WinnerCheckCallback

    job = _job(target=_MAX_256)
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=False)
    ctx = prepare_layer(layer_state, job)
    signal = pipeline._hit_signal_for(layer_state.weight.device)
    observed_signals = []
    observed_hits = []
    hit_seen = [threading.Event(), threading.Event()]
    original_mixed_gemm = pearl_gemm.mixed_gemm

    def observed_mixed_gemm(*args, **kwargs):
        observed_signals.append((args[9], kwargs["record_hits"]))
        return original_mixed_gemm(*args, **kwargs)

    def observe_hit(_callback, hit):
        index = len(observed_hits)
        observed_hits.append(hit)
        hit_seen[index].set()

    monkeypatch.setattr(pearl_gemm, "mixed_gemm", observed_mixed_gemm)
    monkeypatch.setattr(WinnerCheckCallback, "_handle_hit", observe_hit)
    leases_before = pipeline.WINNER_LEASES.held()

    for index in range(2):
        assert (
            pipeline.mine_launch(
                layer_state,
                ctx,
                _M_BUCKET,
                16,
                lambda activation: activation.zero_(),
            )
            is not None
        )
        assert hit_seen[index].wait(timeout=30), f"persistent hit {index + 1} was not consumed"
        assert async_manager.wait_until_drained(timeout=30)
        assert not signal.doorbell(), f"persistent hit {index + 1} did not re-arm"

    assert observed_signals == [(signal, True), (signal, True)]
    assert pipeline._hit_signal_for(layer_state.weight.device) is signal
    assert len(observed_hits) == 2
    assert all(
        hit.valid and hit.codes is not None and hit.scales is not None for hit in observed_hits
    )
    assert pipeline.WINNER_LEASES.held() == leases_before


@pytest.mark.parametrize("layer_state", _MINED_SHAPE_PARAMS, ids=_MINED_SHAPE_IDS, indirect=True)
def test_real_winner_is_detected_and_reports_committed_tile(
    layer_state, async_manager, monkeypatch
):
    """An always-win mine must produce a valid winner whose hit record carries
    the committed 4x64 tile geometry and the layer's k -- covering the low-k
    path and the high-k o_proj path (k=16384) the 4x128 tile could not prove."""
    import vllm_miner.pipeline as pipeline
    from vllm_miner.winners import WinnerCheckCallback

    job = _job(target=_MAX_256)
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=False)
    ctx = prepare_layer(layer_state, job)

    observed_hits = []
    hit_seen = threading.Event()

    def observe_hit(_callback, hit):
        observed_hits.append(hit)
        hit_seen.set()

    monkeypatch.setattr(WinnerCheckCallback, "_handle_hit", observe_hit)

    assert (
        pipeline.mine_launch(layer_state, ctx, _M_BUCKET, 16, lambda activation: activation.zero_())
        is not None
    )
    assert hit_seen.wait(timeout=60), "no winner detected for the always-win mine"
    assert async_manager.wait_until_drained(timeout=30)

    hit = observed_hits[0]
    assert hit.valid and hit.codes is not None and hit.scales is not None
    assert (hit.ltile_rows, hit.ltile_cols) == (4, 64)
    assert hit.k == layer_state.k


def test_decode_bucket_routes_to_the_64_row_tile_and_wins(layer_state, async_manager, monkeypatch):
    """A decode bucket (m=64 < 256) resolves through the small-m heuristic to
    the (64, 64) kernel tile and mines end-to-end: an always-win launch on the
    4x64-committed layer publishes a valid winner."""
    import vllm_miner.pipeline as pipeline
    from vllm_miner.tuning import device_config_name
    from vllm_miner.winners import WinnerCheckCallback

    *_, gemm = pipeline._configs(
        device_config_name(layer_state.weight.device), 64, layer_state.n, layer_state.k
    )
    assert (gemm.tile_m, gemm.tile_n) == (64, 64)
    assert (gemm.ltile_rows, gemm.ltile_cols) == (4, 64)

    job = _job(target=_MAX_256)
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=False)
    ctx = prepare_layer(layer_state, job)

    observed_hits = []
    hit_seen = threading.Event()

    def observe_hit(_callback, hit):
        observed_hits.append(hit)
        hit_seen.set()

    monkeypatch.setattr(WinnerCheckCallback, "_handle_hit", observe_hit)

    assert pipeline.mine_launch(layer_state, ctx, 64, 16, lambda a: a.zero_()) is not None
    assert hit_seen.wait(timeout=60), "no winner detected for the always-win decode mine"
    assert async_manager.wait_until_drained(timeout=30)

    hit = observed_hits[0]
    assert hit.valid and hit.codes is not None and hit.scales is not None
    assert hit.m == 64
    assert (hit.ltile_rows, hit.ltile_cols) == (4, 64)


@pytest.mark.skipif(
    not supports_lottery_family(16), reason="no 16x32 lottery kernel on this device"
)
@pytest.mark.parametrize(
    "layer_state", [("test.mined.tall", 96, 2048)], ids=["tall_16x32"], indirect=True
)
def test_decode_bucket_on_a_tall_tile_layer_mines_through_saved_records(
    layer_state, async_manager, monkeypatch
):
    """m=64 on a 16x32-committed layer: the small-m heuristic declines, a
    saved wide-tile record launches, and the partial-M (m < CTA rows) 16-row
    fold still publishes a valid winner -- an edge 64-multiple buckets made
    reachable (pre-change buckets were at least 256)."""
    import vllm_miner.pipeline as pipeline
    from vllm_miner.tuning import device_config_name
    from vllm_miner.winners import WinnerCheckCallback

    *_, gemm = pipeline._configs(
        device_config_name(layer_state.weight.device), 64, layer_state.n, layer_state.k
    )
    assert (gemm.ltile_rows, gemm.ltile_cols) == (16, 32)
    assert gemm.tile_m != 64

    job = _job(target=_MAX_256)
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=False)
    ctx = prepare_layer(layer_state, job)

    observed_hits = []
    hit_seen = threading.Event()

    def observe_hit(_callback, hit):
        observed_hits.append(hit)
        hit_seen.set()

    monkeypatch.setattr(WinnerCheckCallback, "_handle_hit", observe_hit)

    assert pipeline.mine_launch(layer_state, ctx, 64, 16, lambda a: a.zero_()) is not None
    assert hit_seen.wait(timeout=60), "no winner detected for the always-win tall-tile mine"
    assert async_manager.wait_until_drained(timeout=30)

    hit = observed_hits[0]
    assert hit.valid and hit.codes is not None and hit.scales is not None
    assert hit.m == 64
    assert (hit.ltile_rows, hit.ltile_cols) == (16, 32)


@pytest.mark.parametrize("cert_version", [CertificateVersion.ZK_DENSE, CertificateVersion.ZK_MOE])
def test_legacy_certificate_job_credits_hashrate_without_submission(
    layer_state, async_manager, cert_version, monkeypatch
):
    """A non-V4 job from the node still measures hashrate; it must not submit."""
    import vllm_miner.pipeline as pipeline
    from vllm_miner.pipeline import (
        WINNER_LEASES,
        fallback_winner_checks_idle,
        mine_launch,
    )

    monkeypatch.setattr(
        pipeline,
        "_start_launch_completion",
        lambda *_args, **_kwargs: pytest.fail("legacy job scheduled a winner callback"),
    )

    job = _job(target=_MAX_256, cert_version=cert_version)
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=False)
    ctx = prepare_layer(layer_state, job)
    hashes_before = async_manager._inner_hash_counter
    leases_before = WINNER_LEASES.held()
    assert fallback_winner_checks_idle()

    assert mine_launch(layer_state, ctx, _M_BUCKET, 16, lambda a: a.zero_()) is not None
    assert async_manager.wait_until_drained(timeout=30)

    assert async_manager._inner_hash_counter - hashes_before == effective_work_per_matmul(
        _M_BUCKET, _N, _K
    )
    assert WINNER_LEASES.held() == leases_before
    assert fallback_winner_checks_idle()
    assert async_manager.submission_acknowledgements == 0


def test_a_failed_activation_build_does_not_retire_a_lease(layer_state, async_manager, monkeypatch):
    """An OOM on the launch activation must not consume a retention lease: enough
    of those would silently disable mining for the whole process."""
    from vllm_miner.pipeline import WINNER_LEASES, mine_launch

    job = _job()
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=False)
    ctx = prepare_layer(layer_state, job)
    held_before = WINNER_LEASES.held()

    def explode(_a: torch.Tensor) -> None:
        raise torch.cuda.OutOfMemoryError("synthetic")

    for _ in range(20):
        with pytest.raises(torch.cuda.OutOfMemoryError):
            mine_launch(layer_state, ctx, _M_BUCKET, 16, explode)
    assert WINNER_LEASES.held() == held_before
    assert mine_launch(layer_state, ctx, _M_BUCKET, 16, lambda a: a.zero_()) is not None


def test_mined_forward_approximates_fallback(layer_state, async_manager, monkeypatch):
    from vllm_miner.fp8_fallback import fp8_fallback_gemm
    from vllm_miner.pipeline import run_mining_forward

    job = _job()
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=True)
    ctx = prepare_layer(layer_state, job)
    torch.manual_seed(11)
    x = torch.randn(32, _K, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(_N, dtype=torch.bfloat16, device="cuda")
    mined = run_mining_forward(layer_state, ctx, x, bias=bias, m_bucket=_M_BUCKET)
    torch.cuda.synchronize()
    fallback = fp8_fallback_gemm(
        x, layer_state.w_fp8, layer_state.w_fp8_scale, bias, torch.bfloat16
    )
    assert mined.shape == fallback.shape == (32, _N)
    # Both approximate x @ W.T under independent quantization/noise error.
    cos = torch.nn.functional.cosine_similarity(
        mined.float().flatten(), fallback.float().flatten(), dim=0
    )
    assert cos.item() > 0.99


def test_real_winner_submission_does_not_consume_launch_capacity(
    layer_state, async_manager, monkeypatch
):
    """Exercise independent winner-retention and proof-submission bounds."""
    from vllm_miner.pipeline import WINNER_LEASES, mine_launch

    settings = settings_module._settings.model_copy(update={"winner_check_inflight_limit": 1})
    monkeypatch.setattr(settings_module, "_settings", settings)
    started = threading.Event()
    submission_release = threading.Event()
    lease_released = threading.Event()
    release_lease = WINNER_LEASES.release

    def observed_lease_release():
        release_lease()
        lease_released.set()

    def blocked_submission(*_args):
        started.set()
        assert submission_release.wait(timeout=30)
        return True

    monkeypatch.setattr(WINNER_LEASES, "release", observed_lease_release)
    monkeypatch.setattr(async_manager, "_submit_block", blocked_submission)
    async_manager._conf = async_manager._conf.model_copy(update={"submission_inflight_limit": 1})
    job = _job(target=_MAX_256)
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=False)
    ctx = prepare_layer(layer_state, job)
    held_before = WINNER_LEASES.held()
    acknowledgements_before = async_manager.submission_acknowledgements

    try:
        assert (
            mine_launch(layer_state, ctx, _M_BUCKET, 16, lambda activation: activation.zero_())
            is not None
        )
        assert started.wait(timeout=30), "real winner never reached the submission worker"
        # Callback and proof submission run on separate executors. Observe the
        # actual release instead of racing it against the submission's start.
        assert lease_released.wait(timeout=30), "winner callback did not release its event lease"
        assert WINNER_LEASES.held() == held_before

        # Proof backpressure does not consume launch capacity; a second launch
        # may complete and its winner continuation waits for the one proof slot.
        assert (
            mine_launch(layer_state, ctx, _M_BUCKET, 16, lambda activation: activation.zero_())
            is not None
        )
        assert async_manager._pending_submissions == 1
    finally:
        submission_release.set()

    assert async_manager.wait_until_drained(timeout=30)
    assert WINNER_LEASES.held() == held_before
    assert async_manager.submission_acknowledgements == acknowledgements_before + 2


def test_launch_reprepares_a_stale_context_for_the_live_job(
    layer_state, async_manager, monkeypatch
):
    """A context prepared for a retired job is replaced in place by the
    launch path before mining, so the launch mines the live job."""
    from vllm_miner.linear_op import _try_mine

    stale = prepare_layer(layer_state, _job(target=3))
    live = _job(target=7)
    _wait_for_polled_job(async_manager, monkeypatch, live, no_gateway=True)
    x = torch.randn(_M_BUCKET, _K, dtype=torch.bfloat16, device="cuda")
    assert _try_mine(layer_state, x, None) is not None
    assert layer_state.job_ctx is not stale
    assert layer_state.job_ctx.job == live
    assert bytes(layer_state.buffers.threshold_dev.cpu().numpy()) == threshold_bytes_for(
        live, _K, _N
    )
    assert async_manager.wait_until_drained(timeout=30)


def test_linear_op_falls_back_without_job(layer_state, async_manager):
    from vllm_miner import linear_op  # noqa: F401

    x = torch.randn(4, _K, dtype=torch.bfloat16, device="cuda")
    out = torch.ops.pearl.apply_linear(x, layer_state.weight, None)
    torch.cuda.synchronize()
    assert out.shape == (4, _N)  # no job: FP8 fallback served


# --------------------------------------------------------------------------- #
# MoE: the grouped mining GEMM over a stacked expert weight
# --------------------------------------------------------------------------- #

_EXPERTS, _TOP_K, _N_E = 4, 2, 256


@pytest.fixture
def moe_layer_state(mined_settings):
    torch.manual_seed(5)
    weight = (torch.randn(_EXPERTS * _N_E, _K, dtype=torch.bfloat16) * 1.5).cuda()
    state = create_layer_state("test.mined.experts", weight, experts=_EXPERTS, top_k=_TOP_K)
    register_state(state)
    yield state
    unregister_state(state)
    from vllm_miner import pipeline

    pipeline._HIT_SIGNALS.pop(state.weight.device.index or 0, None)


def _always_win_job() -> MiningJob:
    """A job whose every lottery tile wins and whose header nbits admit the
    resulting jackpot, so its proofs verify and submit."""
    header = BlockHeader(
        version=1,
        prev_block=b"\x11" * 32,
        merkle_root=b"\x22" * 32,
        timestamp=1_700_000_000,
        nbits=0x207FFFFF,
    )
    return MiningJob(
        incomplete_header_bytes=bytes(header.to_bytes()),
        target=_MAX_256,
        cert_version=CertificateVersion.PLAIN_FP8,
    )


def _moe_topk_ids(m_tokens: int) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.stack([torch.randperm(_EXPERTS, device="cuda")[:_TOP_K] for _ in range(m_tokens)])


def test_moe_layers_mine_only_where_the_grouped_kernel_exists(monkeypatch):
    """An expert layer with a lottery tile the dense kernels accept mines only
    where ``grouped_mixed_gemm`` exists (SM100, SM120), and only when its
    ``n_e`` sits on the 128-column MoE lottery lattice."""
    from pearl_gemm import supports_grouped_mixed_gemm
    from vllm_miner.mining_config import MOE_LOTTERY_N
    from vllm_miner.state import can_mine_layer

    device = torch.device("cuda")
    experts, n_e, k = 8, MOE_LOTTERY_N, 2048
    assert can_mine_layer(experts * n_e, k, device, experts) is supports_grouped_mixed_gemm(device)
    assert can_mine_layer(n_e, k, device)
    # 64 columns per expert commit a dense 4x64 tile but straddle the lattice.
    assert can_mine_layer(experts * 64, k, device)
    assert not can_mine_layer(experts * 64, k, device, experts)

    for major in (10, 12):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None, m=major: (m, 0))
        assert can_mine_layer(experts * n_e, k, device, experts)

    assert supports_grouped_mixed_gemm((8, 9)) is False
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 9))
    assert not can_mine_layer(experts * n_e, k, device, experts)


def test_moe_layer_state_commits_the_stacked_weight_without_dense_fallback(moe_layer_state):
    state = moe_layer_state
    assert (state.n, state.experts, state.top_k, state.lottery_n) == (
        _EXPERTS * _N_E,
        _EXPERTS,
        _TOP_K,
        _N_E,
    )
    assert state.w_fp8 is None and state.w_fp8_scale is None
    ctx = prepare_layer(state, _job())
    assert ctx.config.experts == _EXPERTS
    assert ctx.config.p_b(state.n)[-2:] == _EXPERTS.to_bytes(2, "little")
    assert bytes(state.buffers.threshold_dev.cpu().numpy()) == threshold_bytes_for(_job(), _K, _N_E)


def test_moe_warmup_compiles_the_grouped_variant(moe_layer_state):
    import vllm_miner.pipeline as pipeline

    assert pipeline.warmup_layer_variants(moe_layer_state)
    assert pipeline.variant_ready(_M_BUCKET, moe_layer_state)
    assert (_M_BUCKET, _EXPERTS * _N_E, _K, _EXPERTS, _TOP_K) in pipeline._ready_variants


def test_moe_mined_forward_approximates_each_experts_gemm(
    moe_layer_state, async_manager, monkeypatch
):
    """``run_mining_moe_forward`` returns every routed (token, expert) pair's
    ``x_t @ W_e^T`` in expert-major, token-ascending order, credited as work
    against the per-expert lottery lattice."""
    from vllm_miner.mining_config import select_tile
    from vllm_miner.pipeline import run_mining_moe_forward

    state = moe_layer_state
    job = _job()
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=True)
    ctx = prepare_layer(state, job)
    m_tokens = 48
    torch.manual_seed(11)
    x = torch.randn(m_tokens, _K, dtype=torch.bfloat16, device="cuda")
    topk_ids = _moe_topk_ids(m_tokens)
    before = async_manager._inner_hash_counter

    mined = run_mining_moe_forward(state, ctx, x, topk_ids, m_bucket=_M_BUCKET)
    assert mined is not None
    c, routing = mined
    assert async_manager.wait_until_drained(timeout=30)
    torch.cuda.synchronize()

    assert routing.cum_m == m_tokens * _TOP_K
    assert c.shape == (routing.cum_m, _N_E)
    opened = _cpu_pq(state).open().to(torch.float32).view(_EXPERTS, _N_E, _K)
    indptr = routing.m_indptr.tolist()
    tokens = routing.tokens.tolist()
    for e in range(_EXPERTS):
        rows = tokens[indptr[e] : indptr[e + 1]]
        assert rows == sorted(rows) and all((topk_ids[t] == e).any() for t in rows)
        expected = x.cpu()[rows].to(torch.float32) @ opened[e].T
        cos = torch.nn.functional.cosine_similarity(
            c[indptr[e] : indptr[e + 1]].cpu().float().flatten(), expected.flatten(), dim=0
        )
        assert cos.item() > 0.99
    # Every (token, slot) is a distinct permuted row and maps back to its expert.
    slots = routing.slots.tolist()
    assert sorted(slots) == list(range(routing.cum_m))
    assert all(
        topk_ids.reshape(-1)[s].item() == e
        for e in range(_EXPERTS)
        for s in slots[indptr[e] : indptr[e + 1]]
    )

    credited = async_manager._inner_hash_counter - before
    # Exact: the whole 4-row tiles of every expert, read from the device
    # counts once the launch's event completed.
    tile_rows = select_tile(_N_E, _K).rows
    whole_rows = sum(count // tile_rows * tile_rows for count in routing.m_valid.tolist())
    assert 0 < whole_rows <= routing.cum_m
    assert credited == effective_work_per_matmul(whole_rows, _N_E, _K)


def test_moe_routing_commitments_match_the_host_tree(moe_layer_state):
    """``HR || HO`` computed on the device equal the CPU routing-tree root and
    offsets hash the proof opens."""
    from miner_base.commitment import HashId, commit_routing, hash_offsets
    from vllm_miner.moe import route_tokens

    key = torch.arange(32, dtype=torch.uint8, device="cuda")
    for m_tokens in (48, 700):
        topk_ids = _moe_topk_ids(m_tokens)
        routing = route_tokens(topk_ids, _EXPERTS, key)
        torch.cuda.synchronize()
        key_bytes = bytes(key.cpu().tolist())
        hr = bytes(
            commit_routing(routing.tokens.tolist(), key_bytes, HashId.BLAKE3_CHUNK_1024).root
        )
        ho = hash_offsets(routing.m_indptr[1:].tolist(), key_bytes, HashId.BLAKE3_CHUNK_1024)
        assert bytes(routing.commitments.cpu().tolist()) == hr + ho


@pytest.mark.parametrize(
    "counts,whole_rows",
    [
        ([24, 24, 24, 24], 96),  # balanced: every row lies in a whole tile
        ([25, 23, 30, 18], 24 + 20 + 28 + 16),  # ragged: one partial tile per expert
        ([96, 0, 0, 0], 96),  # empty experts credit nothing and cost nothing
        ([3, 3, 3, 3], 0),  # sub-tile experts: no publishable tile, no credit
    ],
    ids=["balanced", "ragged", "empty", "sub-tile"],
)
def test_moe_launch_credit_counts_whole_expert_tiles(moe_layer_state, counts, whole_rows):
    from vllm_miner.pipeline import moe_launch_credit

    expected = effective_work_per_matmul(whole_rows, _N_E, _K) if whole_rows else 0
    assert moe_launch_credit(moe_layer_state, counts) == expected
    assert moe_launch_credit(moe_layer_state, torch.tensor(counts, dtype=torch.int32)) == expected


def test_moe_tail_runs_inside_the_launch_and_holds_drain(
    moe_layer_state, async_manager, monkeypatch
):
    """The adapter's tail executes inside the admitted launch: while it runs,
    ``wait_for_mining_producers_idle`` does not report quiescence, and the
    launch's terminal event is recorded only after it returns."""
    import vllm_miner.pipeline as pipeline
    from vllm_miner.capture import wait_for_mining_producers_idle
    from vllm_miner.linear_op import try_mine_moe

    state = moe_layer_state
    job = _job()
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=True)
    prepare_layer(state, job)
    x = torch.randn(16, _K, dtype=torch.bfloat16, device="cuda")
    topk_ids = _moe_topk_ids(16)
    order: list[str] = []
    record_event = pipeline._record_stream_event

    def observed_record():
        order.append("event")
        return record_event()

    monkeypatch.setattr(pipeline, "_record_stream_event", observed_record)
    tail_entered = threading.Event()
    release_tail = threading.Event()
    result: list[object] = []

    def tail(gate_up, routing):
        order.append("tail")
        tail_entered.set()
        assert release_tail.wait(30)
        return gate_up.shape[0], routing.cum_m

    worker = threading.Thread(
        target=lambda: result.append(try_mine_moe(state, x, topk_ids, tail)), name="moe-forward"
    )
    worker.start()
    assert tail_entered.wait(60)
    assert not wait_for_mining_producers_idle(0.2), "drain reported quiescence during the tail"
    release_tail.set()
    worker.join(60)
    assert not worker.is_alive()
    assert result == [(16 * _TOP_K, 16 * _TOP_K)]
    assert order == ["tail", "event"]
    assert wait_for_mining_producers_idle(30)
    assert async_manager.wait_until_drained(timeout=30)


def test_moe_tail_oom_declines_the_forward_and_opens_the_cooldown(
    moe_layer_state, async_manager, monkeypatch
):
    """A tail OOM is the launch's OOM: the forward is declined (the adapter
    serves unmined), the device cooldown opens like for any launch OOM, the
    layer stays enabled, no credit is booked, and no lease leaks."""
    import vllm_miner.pipeline as pipeline
    from vllm_miner import health
    from vllm_miner.linear_op import try_mine_moe

    state = moe_layer_state
    job = _job()
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=True)
    prepare_layer(state, job)
    x = torch.randn(16, _K, dtype=torch.bfloat16, device="cuda")
    before = async_manager._inner_hash_counter

    def oom_tail(gate_up, routing):
        raise torch.cuda.OutOfMemoryError("synthetic tail OOM")

    try:
        assert try_mine_moe(state, x, _moe_topk_ids(16), oom_tail) is None
        assert state.mineable and state.disabled_reason is None
        with health.mining_attempt(state.weight.device) as attempt:
            assert not attempt, "the device cooldown did not open after the tail OOM"
        assert async_manager.wait_until_drained(timeout=30)
        assert async_manager._inner_hash_counter == before
        assert pipeline._COMPLETION_LEASES.in_use() == 0
    finally:
        health._BREAKER.reset_for_tests()  # the cooldown is process-wide


def test_moe_forward_declines_a_malformed_router_table(moe_layer_state, async_manager, monkeypatch):
    """A router table of the wrong shape, dtype or device declines the call
    (the inherited path serves it) instead of disabling the layer."""
    from vllm_miner.linear_op import try_mine_moe

    state = moe_layer_state
    job = _job()
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=True)
    prepare_layer(state, job)
    x = torch.randn(16, _K, dtype=torch.bfloat16, device="cuda")
    good = _moe_topk_ids(16)
    tail = lambda out, routing: (out, routing)  # noqa: E731
    for bad in (
        good[:8],  # row count
        good[:, :1],  # top_k
        good.float(),  # dtype
        good.cpu(),  # device
        good.reshape(-1),  # rank
    ):
        assert try_mine_moe(state, x, bad, tail) is None
        assert state.mineable and state.disabled_reason is None
    assert try_mine_moe(state, x, good, tail) is not None
    assert async_manager.wait_until_drained(timeout=30)


def test_combine_routed_rows_matches_the_per_token_weighted_sum():
    """The deterministic scatter + slot reduction equals the textbook
    per-token ``sum_k w[t, k] * y[t, k]`` in fp32, and is bitwise
    reproducible across calls."""
    from vllm_miner.moe import combine_routed_rows, route_tokens

    m_tokens, n = 40, 96
    topk_ids = _moe_topk_ids(m_tokens)
    key = torch.zeros(32, dtype=torch.uint8, device="cuda")
    routing = route_tokens(topk_ids, _EXPERTS, key)
    torch.manual_seed(3)
    rows = torch.randn(routing.cum_m, n, dtype=torch.bfloat16, device="cuda")
    weights = torch.rand(m_tokens, _TOP_K, dtype=torch.float32, device="cuda")

    combined = combine_routed_rows(rows, weights, routing)
    again = combine_routed_rows(rows, weights, routing)
    torch.cuda.synchronize()

    assert combined.shape == (m_tokens, n) and combined.dtype == rows.dtype
    assert torch.equal(combined, again)
    # Per-token oracle: undo the permutation slot by slot in fp32.
    slots = routing.slots.tolist()
    y = torch.zeros(m_tokens, _TOP_K, n, dtype=torch.float32)
    for row, slot in enumerate(slots):
        y[slot // _TOP_K, slot % _TOP_K] = rows[row].float().cpu()
    w_bf16 = weights.to(torch.bfloat16).float().cpu()  # the combine weights in the rows' dtype
    expected = (y * w_bf16.unsqueeze(-1)).sum(1)
    torch.testing.assert_close(combined.float().cpu(), expected, atol=6e-2, rtol=2e-2)


def test_moe_routing_is_enqueued_only_for_admitted_launches(
    moe_layer_state, async_manager, monkeypatch
):
    """A declined MoE forward enqueues no routing work: the sort and the keyed
    commitments run inside the admitted launch's completion fence."""
    import vllm_miner.pipeline as pipeline
    from vllm_miner import moe

    state = moe_layer_state
    job = _job()
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=True)
    ctx = prepare_layer(state, job)
    routed = []

    def counting_route(*args, **kwargs):
        routed.append(args)
        return moe.route_tokens(*args, **kwargs)

    monkeypatch.setattr(pipeline, "route_tokens", counting_route)
    monkeypatch.setattr(pipeline._COMPLETION_LEASES, "try_acquire", lambda *_: None)
    x = torch.randn(16, _K, dtype=torch.bfloat16, device="cuda")
    assert pipeline.run_mining_moe_forward(state, ctx, x, _moe_topk_ids(16), _M_BUCKET) is None
    assert routed == []


def _capture_completions(pipeline, monkeypatch) -> list:
    works = []

    def capture(event, callback, release, **kwargs):
        with pipeline._fallback_check_lock:
            pipeline._fallback_checks += 1
        works.append(
            pipeline._FallbackWork(
                event=event,
                callback=callback,
                release=release,
                completion=pipeline._FallbackCompletion(),
                **kwargs,
            )
        )

    monkeypatch.setattr(pipeline, "_start_launch_completion", capture)
    return works


def _recording_submission(async_manager, monkeypatch):
    submitted = []
    done = threading.Event()

    def submit_block(opening, _job):
        submitted.append(opening)
        done.set()
        return True

    monkeypatch.setattr(async_manager, "_submit_block", submit_block)
    return submitted, done


@pytest.mark.parametrize("order", ["dense-consumer-first", "moe-consumer-first", "two-routings"])
def test_winner_check_consumes_only_its_own_launch_record(
    layer_state, moe_layer_state, async_manager, monkeypatch, order
):
    """Two retained launches share the device-wide signal; the first wins and
    the second's hit is lost to the closed latch. When the second launch's
    completion runs first it must leave the record for its owner: a dense
    consumer must not open an MoE record without its witness, an MoE consumer
    must not attach its routing to a dense record, and a same-layer MoE
    consumer must not attach its own routing to the other launch's record."""
    import vllm_miner.pipeline as pipeline
    from miner_base.block_submission import create_proof
    from pearl_mining import (
        CERT_VERSION_PLAIN_FP8,
        IncompleteBlockHeader,
        verify_plain_proof_for_cert_version,
    )

    works = _capture_completions(pipeline, monkeypatch)
    submitted, done = _recording_submission(async_manager, monkeypatch)
    job = _always_win_job()
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=False)
    dense_ctx = prepare_layer(layer_state, job)
    moe_ctx = prepare_layer(moe_layer_state, job)
    m_tokens = 48
    torch.manual_seed(11)
    x = torch.randn(m_tokens, _K, dtype=torch.bfloat16, device="cuda")
    topk_ids = _moe_topk_ids(m_tokens)

    def dense_launch():
        out = pipeline.mine_launch(layer_state, dense_ctx, _M_BUCKET, 16, lambda a: a.zero_())
        assert out is not None
        return None

    def moe_launch(ids):
        mined = pipeline.run_mining_moe_forward(
            moe_layer_state, moe_ctx, x, ids, m_bucket=_M_BUCKET
        )
        assert mined is not None
        return mined[1]

    if order == "dense-consumer-first":
        winner_routing, _ = moe_launch(topk_ids), dense_launch()
    elif order == "moe-consumer-first":
        winner_routing, _ = dense_launch(), moe_launch(topk_ids)
    else:
        winner_routing, _ = moe_launch(topk_ids), moe_launch((topk_ids + 1) % _EXPERTS)
    torch.cuda.synchronize()
    assert len(works) == 2
    signal = pipeline._HIT_SIGNALS[layer_state.weight.device.index or 0]
    assert signal.doorbell(), "the first launch should have published under the always-win target"

    # The second launch's consumer runs first: the record is not its own.
    pipeline._run_fallback_work(works[1])
    assert signal.doorbell() and submitted == []

    pipeline._run_fallback_work(works[0])
    assert done.wait(timeout=60), "the owning launch's consumer did not submit"
    assert async_manager.wait_until_drained(timeout=30)
    assert not signal.doorbell()
    (opening,) = submitted
    header = IncompleteBlockHeader.from_bytes(job.incomplete_header_bytes)
    if winner_routing is None:
        assert opening.moe is None and opening.a_codes.shape == (_M_BUCKET, _K)
    else:
        assert opening.moe is not None
        assert tuple(opening.moe.routing) == tuple(winner_routing.tokens.tolist())
        assert tuple(opening.moe.offsets) == tuple(winner_routing.m_indptr[1:].tolist())
    accepted, message = verify_plain_proof_for_cert_version(
        CERT_VERSION_PLAIN_FP8, header, create_proof(opening, header)
    )
    assert accepted, message


def _inject_post_publication_failure(pipeline, monkeypatch, point: str) -> None:
    """Make the next retained launch fail after its publisher was enqueued:
    at the terminal event's record (the cleanup event that follows must
    still succeed) or at the winner continuation's construction."""
    if point == "event":
        record_event = pipeline._record_stream_event
        calls = 0

        def fail_first_record():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("synthetic terminal event failure")
            return record_event()

        monkeypatch.setattr(pipeline, "_record_stream_event", fail_first_record)
    else:
        continuation = pipeline._winner_continuation
        armed = [True]

        def fail_once(**kwargs):
            if armed:
                armed.clear()
                raise RuntimeError("synthetic continuation failure")
            return continuation(**kwargs)

        monkeypatch.setattr(pipeline, "_winner_continuation", fail_once)


@pytest.mark.parametrize("point", ["event", "continuation"])
def test_failed_continuation_after_publication_frees_the_latch(
    layer_state, moe_layer_state, async_manager, monkeypatch, point
):
    """A retained MoE launch whose host continuation fails after its publisher
    ran leaves a payload-less record under its own key. No other consumer may
    take it (every winner check holds another key), so the failed launch must
    consume it itself, event gated, or the device-wide latch stays closed for
    every later winner. Then a launch under another key publishes and its
    proof is submitted."""
    import vllm_miner.pipeline as pipeline

    submitted, done = _recording_submission(async_manager, monkeypatch)
    job = _always_win_job()
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=False)
    dense_ctx = prepare_layer(layer_state, job)
    moe_ctx = prepare_layer(moe_layer_state, job)
    m_tokens = 48
    torch.manual_seed(11)
    x = torch.randn(m_tokens, _K, dtype=torch.bfloat16, device="cuda")
    signal = pipeline._hit_signal_for(layer_state.weight.device)
    winner_leases_before = pipeline.WINNER_LEASES.held()

    _inject_post_publication_failure(pipeline, monkeypatch, point)
    with pytest.raises(RuntimeError, match="synthetic"):
        pipeline.run_mining_moe_forward(
            moe_layer_state, moe_ctx, x, _moe_topk_ids(m_tokens), m_bucket=_M_BUCKET
        )
    # The discard is event gated on the completion worker, like a winner check.
    assert pipeline.wait_for_mining_continuations_idle(30)
    assert async_manager.wait_until_drained(timeout=30)
    assert not signal.doorbell(), "the failed launch's own record was left latched"
    assert pipeline.WINNER_LEASES.held() == winner_leases_before
    assert pipeline._COMPLETION_LEASES.in_use() == 0
    assert submitted == []

    out = pipeline.mine_launch(layer_state, dense_ctx, _M_BUCKET, 16, lambda a: a.zero_())
    assert out is not None
    assert done.wait(timeout=60), "a later launch could not publish through the shared signal"
    assert async_manager.wait_until_drained(timeout=30)
    assert len(submitted) == 1 and submitted[0].moe is None
    assert not signal.doorbell()


def test_failed_continuation_discard_leaves_a_foreign_record_in_place(
    layer_state, moe_layer_state, async_manager, monkeypatch
):
    """The failed launch's compensating discard is owner selective: when the
    latched record belongs to another (live) launch it is left for that
    launch's consumer, which then submits it."""
    import vllm_miner.pipeline as pipeline

    works = _capture_completions(pipeline, monkeypatch)
    submitted, done = _recording_submission(async_manager, monkeypatch)
    job = _always_win_job()
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=False)
    dense_ctx = prepare_layer(layer_state, job)
    moe_ctx = prepare_layer(moe_layer_state, job)
    m_tokens = 48
    torch.manual_seed(11)
    x = torch.randn(m_tokens, _K, dtype=torch.bfloat16, device="cuda")

    out = pipeline.mine_launch(layer_state, dense_ctx, _M_BUCKET, 16, lambda a: a.zero_())
    assert out is not None
    torch.cuda.synchronize()
    signal = pipeline._HIT_SIGNALS[layer_state.weight.device.index or 0]
    assert signal.doorbell(), "the first launch should have published under the always-win target"

    _inject_post_publication_failure(pipeline, monkeypatch, "continuation")
    with pytest.raises(RuntimeError, match="synthetic"):
        pipeline.run_mining_moe_forward(
            moe_layer_state, moe_ctx, x, _moe_topk_ids(m_tokens), m_bucket=_M_BUCKET
        )
    torch.cuda.synchronize()
    assert len(works) == 2
    assert isinstance(works[1].callback, pipeline._OwnedRecordDiscard)

    pipeline._run_fallback_work(works[1])  # the failed launch's discard: not its record
    assert signal.doorbell() and submitted == []
    pipeline._run_fallback_work(works[0])  # the owner's winner check
    assert done.wait(timeout=60), "the owning launch's consumer did not submit"
    assert async_manager.wait_until_drained(timeout=30)
    assert not signal.doorbell()
    assert len(submitted) == 1 and submitted[0].moe is None


def test_moe_real_winner_yields_a_verifiable_moe_proof(moe_layer_state, async_manager, monkeypatch):
    """An always-win MoE launch publishes one expert-local hit; the winner check
    resolves it to the tile's global tokens and stacked weight rows, and the
    opening it builds carries a routing witness the cert-v4 verifier accepts."""
    from miner_base.block_submission import create_proof, submit_opened_block
    from pearl_mining import (
        CERT_VERSION_PLAIN_FP8,
        IncompleteBlockHeader,
        verify_plain_proof_for_cert_version,
    )
    from vllm_miner.pipeline import run_mining_moe_forward

    state = moe_layer_state
    submitted = []
    done = threading.Event()

    class RecordingClient:
        def submit_plain_proof(self, proof, _job):
            submitted.append(proof)

    def submit_block(opening, mining_job):
        try:
            assert opening.moe is not None
            submitted.append(opening)
            return submit_opened_block(opening, mining_job, RecordingClient()) is not None
        finally:
            done.set()

    monkeypatch.setattr(async_manager, "_submit_block", submit_block)
    job = _always_win_job()
    _wait_for_polled_job(async_manager, monkeypatch, job, no_gateway=False)
    ctx = prepare_layer(state, job)
    m_tokens = 48
    torch.manual_seed(11)
    x = torch.randn(m_tokens, _K, dtype=torch.bfloat16, device="cuda")
    topk_ids = _moe_topk_ids(m_tokens)

    mined = run_mining_moe_forward(state, ctx, x, topk_ids, m_bucket=_M_BUCKET)
    assert mined is not None
    assert done.wait(timeout=60), "no winner reached submission for the always-win MoE mine"
    assert async_manager.wait_until_drained(timeout=30)

    opening, proof = submitted
    routing = mined[1]
    assert tuple(opening.moe.routing) == tuple(routing.tokens.tolist())
    assert tuple(opening.moe.offsets) == tuple(routing.m_indptr[1:].tolist())
    w = opening.moe.expert_index
    assert all(w * _N_E <= col < (w + 1) * _N_E for col in opening.b_column_indices)
    assert all((topk_ids[t] == w).any() for t in opening.a_row_indices)
    assert proof.moe.experts == _EXPERTS and proof.moe_witness.w == w
    header = IncompleteBlockHeader.from_bytes(job.incomplete_header_bytes)
    accepted, message = verify_plain_proof_for_cert_version(CERT_VERSION_PLAIN_FP8, header, proof)
    assert accepted, message
    # Reconstructing the same host opening produces identical proof bytes.
    assert create_proof(opening, header).to_base64() == proof.to_base64()
