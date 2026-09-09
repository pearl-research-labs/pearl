"""Mined-linear runtime on SM100: load-time encoding, per-job B preparation
parity, launch accounting, winner round-trip, and staleness.

Intentionally no hardware skip guards: this suite runs only on the B200 job.
"""

import threading

import pytest
import torch
from miner_base.block_submission import commit_planes_for_leaf
from miner_base.commitment import BlockHeader
from miner_base.commitment_hash import noise_seed_b
from miner_base.hardware import hardware_for
from miner_base.noise import OperandNoiser, Side
from miner_base.prequant import PrequantMatrix
from miner_base.quantization import Fp8QuantScheme
from miner_base.scheme import PearlScheme
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.dataclasses import MiningJob
from pearl_gemm import b_peel_for_a
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
def test_prepare_layer_matches_reference_commitment_chain(layer_state):
    """Load-time commitment + B preparation parity against the CPU reference,
    for both the low-k shape and the high-k o_proj shape (k=16384) the 4x128 tile
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

    # Uploaded operands equal the canonical CPU reference build. v4 draws F_A
    # from each launch's own seedA, so the B side is checked under a probe
    # seedA: the job-invariant parts from the buffers, the per-launch mid half
    # of the peel through the same ``b_peel_for_a`` the pipeline runs.
    hardware = hardware_for(config.device)
    scheme = PearlScheme(hardware, Fp8QuantScheme(), k, RANK)
    seed_a_probe = b"\x00" * 32
    noise_a = OperandNoiser(seed_a_probe, Side.A, RANK, k, hardware.compute)
    noise_b = OperandNoiser(expected_seed_b, Side.B, RANK, k, hardware.compute)
    stacked = scheme.build_b_rows(
        weight_pq.open(), noise_a, noise_b, list(range(n)), weight_pq.exact_norms()
    )
    assert torch.equal(buffers.b_prime.cpu(), stacked.quant_part)
    assert torch.equal(buffers.f2.cpu(), noise_b.F())
    b_peel = buffers.b_peel.cpu()
    # The element-wise second half (the job-invariant -(beta_b (.) E_B)) is
    # bit-exact; the first half is the per-launch reference-only peel matmul,
    # which the buffers leave zero (noisy_quant_b runs against F_A = 0).
    assert torch.equal(b_peel[:, RANK:], stacked.peel_part[:, RANK:])
    assert not b_peel[:, :RANK].any()
    f1_lines = noise_a.F().t().contiguous().cuda()  # (k, R) noise_lines layout
    full = b_peel_for_a(
        buffers.b_prime, buffers.e2, buffers.f2, buffers.beta_b, buffers.b_peel, f1_lines
    ).cpu()
    assert torch.equal(full[:, RANK:], stacked.peel_part[:, RANK:])
    # CPU and GPU reduction order may differ on the matmul half.
    mid_reference = stacked.peel_part[:, :RANK].float()
    mid_relative_error = (
        (full[:, :RANK].float() - mid_reference).norm() / mid_reference.norm().clamp_min(1e-30)
    ).item()
    assert mid_relative_error < 1e-2, mid_relative_error
    assert torch.equal(
        buffers.alpha_b.cpu().reshape(-1),
        stacked.alpha.reshape(-1),
    )
    prebuilt = ctx.b_proof.prebuilt_commitment()
    assert prebuilt.key == key_b
    assert prebuilt.commitment.digest == comm_b.digest

    assert layer_state.job_ctx is ctx


def test_prepare_dispatches_gpu_b_chain_without_reference_builder(layer_state, monkeypatch):
    """Production B preparation must never fall back to PearlScheme.build_b_rows."""
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
    monkeypatch.setattr(
        PearlScheme,
        "build_b_rows",
        lambda *_args, **_kwargs: pytest.fail("B rebuild used the CPU reference builder"),
    )

    ctx = prepare_layer(layer_state, _job())
    torch.cuda.synchronize()

    assert ctx is not None
    # v4: F_A is per launch (drawn under each A's seedA), so B preparation
    # draws only F_B here.
    assert calls == [
        "tensor_hash_plus_stats_b",
        "noise_lines",
        "noisy_quant_b",
    ]


def test_prepare_fails_closed_instead_of_using_reference_b_fallback(layer_state, monkeypatch):
    """A failed GPU B stage must leave the layer unpublished and propagate."""
    import pearl_gemm

    reference_called = False
    noise_launches = 0

    def fail_after_partial_gpu_chain(*args, **kwargs):
        nonlocal noise_launches
        noise_launches += 1
        raise RuntimeError("synthetic GPU B-preparation failure")

    def reject_reference_fallback(*_args, **_kwargs):
        nonlocal reference_called
        reference_called = True
        pytest.fail("GPU B-preparation failure fell back to CPU build_b_rows")

    monkeypatch.setattr(pearl_gemm, "noise_lines", fail_after_partial_gpu_chain)
    monkeypatch.setattr(PearlScheme, "build_b_rows", reject_reference_fallback)

    with pytest.raises(RuntimeError, match="synthetic GPU B-preparation failure"):
        prepare_layer(layer_state, _job())

    assert noise_launches == 1
    assert not reference_called
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
    assert (_M_BUCKET, _N, _K) in failed


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
    """An OOM on the padded activation must not consume a retention lease: enough
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
