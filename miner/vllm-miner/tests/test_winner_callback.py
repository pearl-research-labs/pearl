import sys
import threading
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
from miner_base.block_submission import commit_planes_for_leaf
from miner_base.commitment import Device
from miner_base.commitment_hash import a_keys, noise_seed_b
from miner_base.mining_config import activation_leaf
from miner_base.prequant import PrequantMatrix
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.dataclasses import MiningJob
from vllm_miner import pipeline, winners
from vllm_miner.mining_config import (
    commitment_keys_for,
    effective_work_per_matmul,
    lottery_threshold,
    mining_configuration,
    threshold_bytes_for,
    tile_indices,
)
from vllm_miner.winners import WinnerCheckCallback

# Host-only: every configuration commits one explicit device.
_DEVICE = Device.BLACKWELL


@pytest.fixture(autouse=True)
def async_manager():
    """These host-only unit tests install their own manager seams when needed."""
    yield None


class _Leases:
    def __init__(self) -> None:
        self.releases = 0

    def release(self) -> None:
        self.releases += 1


class _Signal:
    def __init__(self, hit=None) -> None:
        self.hit = hit
        self.resets = 0

    def take_owned_hit(self, owner=None):
        hit = self.hit
        if hit is None:
            return None
        self.resets += 1
        self.hit = None
        owned_copy = getattr(hit, "owned_copy", None)
        return hit if owned_copy is None else owned_copy()


@pytest.mark.parametrize("published", [False, True])
def test_winner_callback_always_releases_its_winner_lease(published):
    hit = SimpleNamespace(valid=False) if published else None
    signal = _Signal(hit)
    leases = _Leases()
    callback = WinnerCheckCallback(
        signal=signal,
        event=object(),
        leases=leases,
        manager=SimpleNamespace(),
    )

    callback()

    assert leases.releases == 1
    assert signal.resets == int(published)


def test_winner_callback_rearms_before_slow_validation(monkeypatch):
    order = []

    class Hit:
        valid = True

        def owned_copy(self):
            order.append("copy")
            return self

    class Signal(_Signal):
        def take_owned_hit(self, owner=None):
            hit = super().take_owned_hit(owner)
            order.append("reset")
            return hit

    signal = Signal(Hit())
    leases = _Leases()
    callback = WinnerCheckCallback(
        signal=signal,
        event=object(),
        leases=leases,
        manager=SimpleNamespace(),
    )
    monkeypatch.setattr(callback, "_handle_hit", lambda _hit: order.append("validate"))

    callback()

    assert order == ["copy", "reset", "validate"]
    assert leases.releases == 1


def _stub_hit_signal_error(monkeypatch):
    class HitSignalPoisonedError(RuntimeError):
        pass

    module = ModuleType("pearl_gemm")
    module.HitSignalPoisonedError = HitSignalPoisonedError
    monkeypatch.setitem(sys.modules, "pearl_gemm", module)
    return HitSignalPoisonedError


def test_winner_callback_releases_its_lease_when_signal_consume_raises(monkeypatch):
    _stub_hit_signal_error(monkeypatch)
    leases = _Leases()
    signal = Mock()
    signal.take_owned_hit.side_effect = RuntimeError("synthetic consume failure")
    callback = WinnerCheckCallback(
        signal=signal,
        event=object(),
        leases=leases,
        manager=SimpleNamespace(),
    )

    callback()

    assert leases.releases == 1


def test_hit_signal_poison_disables_device_and_releases_lease(monkeypatch):
    HitSignalPoisonedError = _stub_hit_signal_error(monkeypatch)

    leases = _Leases()
    signal = Mock()
    signal.device = torch.device("cuda:0")
    signal.take_owned_hit.side_effect = HitSignalPoisonedError("poisoned")
    state = Mock()
    state.weight.device = signal.device
    disable_device = Mock()
    monkeypatch.setattr(winners, "all_states", lambda: [state])
    monkeypatch.setattr(winners, "disable_device_mining", disable_device)

    WinnerCheckCallback(
        signal=signal,
        event=object(),
        leases=leases,
        manager=SimpleNamespace(),
    )()

    assert leases.releases == 1
    disable_device.assert_called_once_with(
        signal.device,
        "persistent hit signal could not be re-armed",
    )
    state.disable_mining.assert_called_once()


@pytest.mark.parametrize("job_replaced_during_handoff", [False, True])
def test_validated_persistent_hit_builds_canonical_opening(
    monkeypatch,
    job_replaced_during_handoff,
):
    k = 2048
    n = 256
    # Non-saturating: 4x64 vs 4x128 thresholds stay distinct, so omitting n
    # from mining_configuration / the pB in seedB would fail this test.
    target = 1 << 128
    config = mining_configuration(k, n, device=_DEVICE)
    job = MiningJob(
        incomplete_header_bytes=bytes(range(76)),
        target=target,
        cert_version=CertificateVersion.PLAIN_FP8,
    )
    a = PrequantMatrix.encode(torch.zeros(8, k, dtype=torch.bfloat16))
    b = PrequantMatrix.encode(torch.zeros(n, k, dtype=torch.bfloat16))
    key_a, key_b = commitment_keys_for(job)
    assert lottery_threshold(target, k, n, device=_DEVICE) != lottery_threshold(
        target, k, device=_DEVICE
    )
    comm_a = commit_planes_for_leaf(a.planes(), key_a, activation_leaf(config))
    comm_b = commit_planes_for_leaf(b.planes(), key_b, config.chunk_size)
    seed_b = noise_seed_b(comm_b.digest, key_b, config.p_b(n))
    keys = a_keys(comm_a.digest, seed_b, key_a, config.p_a(a.int_values.shape[0]))
    prebuilt = object()
    ctx = SimpleNamespace(
        config=config,
        key_a=key_a,
        key_b=key_b,
        seed_b=seed_b,
        target=job.target,
        job=job,
        b_proof=SimpleNamespace(prebuilt_commitment=lambda: prebuilt),
    )
    state = SimpleNamespace(
        layer_name="test.layer",
        layer_id=7,
        n=b.int_values.shape[0],
        lottery_n=b.int_values.shape[0],
        k=k,
        weight_cpu=b.int_values,
        weight_scale_cpu=b.scales,
        lock=threading.Lock(),
        job_ctx=ctx,
    )
    hit = SimpleNamespace(
        valid=True,
        layer_id=state.layer_id,
        m=a.int_values.shape[0],
        n=state.n,
        k=k,
        ltile_rows=config.rows_pattern.tile_size,
        ltile_cols=config.cols_pattern.tile_size,
        tile_row=1,
        tile_column=0,
        commitment_hash_A=keys.jackpot_key,
        commitment_hash_B=seed_b,
        target=threshold_bytes_for(job, k, state.n, device=_DEVICE),
        codes=a.int_values,
        scales=a.scales,
    )
    submitted = []
    handoff_timeouts = []

    class Manager:
        live_job = job

        @classmethod
        def get_mining_job(cls):
            return cls.live_job

        @staticmethod
        def handle_submit_block(opening, submitted_job, *, timeout):
            submitted.append((opening, submitted_job))
            handoff_timeouts.append(timeout)
            return True

    monkeypatch.setattr(winners, "lookup_state_by_layer_id", lambda _layer_id: state)
    original_commit = winners.commit_planes_for_leaf

    def commit(*args, **kwargs):
        result = original_commit(*args, **kwargs)
        if job_replaced_during_handoff:
            Manager.live_job = MiningJob(
                incomplete_header_bytes=bytes(range(1, 77)),
                target=target,
                cert_version=CertificateVersion.PLAIN_FP8,
            )
        return result

    monkeypatch.setattr(winners, "commit_planes_for_leaf", commit)
    callback = WinnerCheckCallback(
        signal=_Signal(hit),
        event=object(),
        leases=_Leases(),
        manager=Manager(),
    )

    callback._handle_hit(hit)

    if job_replaced_during_handoff:
        assert submitted == []
        return

    assert len(submitted) == 1
    assert handoff_timeouts == [winners._WINNER_HANDOFF_TIMEOUT_S]
    opening, submitted_job = submitted[0]
    assert submitted_job is job
    assert opening.a_row_indices == tuple(tile_indices(config.rows_pattern, 1))
    assert opening.b_column_indices == tuple(tile_indices(config.cols_pattern, 0))
    assert opening.a_codes is a.int_values
    assert opening.a_scales is a.scales
    assert opening.b_codes is b.int_values
    assert opening.b_scales is b.scales
    assert opening.b_commitment is prebuilt
    assert opening.a_commitment is not None
    assert opening.a_commitment.key == key_a
    assert opening.a_commitment.commitment.digest == comm_a.digest


def test_hit_without_a_published_context_is_dropped(monkeypatch):
    state = SimpleNamespace(layer_name="test.layer", lock=threading.Lock(), job_ctx=None)
    monkeypatch.setattr(winners, "lookup_state_by_layer_id", lambda _layer_id: state)
    submit = Mock(side_effect=AssertionError("stale hit reached submission"))
    callback = WinnerCheckCallback(
        signal=object(),
        event=object(),
        leases=_Leases(),
        manager=SimpleNamespace(
            get_mining_job=lambda: Mock(spec=MiningJob),
            handle_submit_block=submit,
        ),
    )

    callback._handle_hit(SimpleNamespace(layer_id=3))

    submit.assert_not_called()


def test_hit_for_a_replaced_job_is_dropped(monkeypatch):
    """A context prepared for a job the manager has since replaced cannot
    submit a hit, whatever the record's stamped words say."""
    retired = Mock(spec=MiningJob)
    ctx = SimpleNamespace(job=retired)
    state = SimpleNamespace(layer_name="test.layer", lock=threading.Lock(), job_ctx=ctx)
    monkeypatch.setattr(winners, "lookup_state_by_layer_id", lambda _layer_id: state)
    submit = Mock(side_effect=AssertionError("retired context reached submission"))
    callback = WinnerCheckCallback(
        signal=object(),
        event=object(),
        leases=_Leases(),
        manager=SimpleNamespace(
            get_mining_job=lambda: Mock(spec=MiningJob),
            handle_submit_block=submit,
        ),
    )

    callback._handle_hit(SimpleNamespace(layer_id=3))

    submit.assert_not_called()


def test_hit_stamped_with_a_stale_b_seed_is_dropped(monkeypatch):
    """The kernel stamps seedB from the device buffer at launch time; a hit
    produced before the layer was re-prepared for the live job carries the
    old seed and must not validate against the new context."""
    job = Mock(spec=MiningJob)
    config = mining_configuration(2048, 256, device=_DEVICE)
    ctx = SimpleNamespace(job=job, config=config, seed_b=b"\x01" * 32)
    state = SimpleNamespace(
        layer_name="test.layer", n=256, lottery_n=256, k=2048, lock=threading.Lock(), job_ctx=ctx
    )
    hit = SimpleNamespace(
        layer_id=3,
        m=8,
        n=256,
        k=2048,
        ltile_rows=config.rows_pattern.tile_size,
        ltile_cols=config.cols_pattern.tile_size,
        tile_row=0,
        tile_column=0,
        commitment_hash_B=b"\x02" * 32,
    )
    assert (
        winners._matching_context(hit, state, SimpleNamespace(get_mining_job=lambda: job)) is None
    )


def test_winner_callback_consumes_only_its_launch_record():
    """The callback passes its launch's pinned ``pow_key`` to the signal, so a
    record another launch published is left for that launch's check."""
    owners = []

    class Signal(_Signal):
        def take_owned_hit(self, owner=None):
            owners.append(owner)
            return None

    key = bytes(range(32))
    leases = _Leases()
    WinnerCheckCallback(
        signal=Signal(),
        event=object(),
        leases=leases,
        manager=SimpleNamespace(),
        pow_key_host=torch.tensor(list(key), dtype=torch.uint8),
    )()
    assert owners == [key] and leases.releases == 1
    with pytest.raises(ValueError, match="pow_key"):
        WinnerCheckCallback(
            signal=Signal(), event=object(), leases=leases, manager=SimpleNamespace(), moe=object()
        )


def _moe_winner_fixture(group_id: int):
    """A retained MoE launch's host view: 4 experts of ``n_e = 256`` stacked
    rows, round-robin top-2 routing of 64 tokens, and an always-win job."""
    from miner_base.block_submission import PrebuiltCommitment
    from miner_base.commitment import BlockHeader
    from vllm_miner.moe import MoeLaunch, MoeRouting

    k, experts, n_e, top_k, m = 2048, 4, 256, 2, 64
    config = mining_configuration(k, experts * n_e, experts, device=_DEVICE)
    header = BlockHeader(
        version=1,
        prev_block=b"\x11" * 32,
        merkle_root=b"\x22" * 32,
        timestamp=1_700_000_000,
        nbits=0x207FFFFF,
    )
    job = MiningJob(
        incomplete_header_bytes=bytes(header.to_bytes()),
        target=(1 << 256) - 1,
        cert_version=CertificateVersion.PLAIN_FP8,
    )
    torch.manual_seed(3)
    a = PrequantMatrix.encode(torch.randn(m, k, dtype=torch.bfloat16))
    b = PrequantMatrix.encode(torch.randn(experts * n_e, k, dtype=torch.bfloat16))
    key_a, key_b = commitment_keys_for(job)
    comm_b = commit_planes_for_leaf(b.planes(), key_b, config.chunk_size)
    seed_b = noise_seed_b(comm_b.digest, key_b, config.p_b(experts * n_e))
    per_expert = [
        [t for t in range(m) if any((t + slot) % experts == e for slot in range(top_k))]
        for e in range(experts)
    ]
    counts = [len(rows) for rows in per_expert]
    m_indptr = torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], dtype=torch.int32)
    routing = MoeRouting(
        experts=experts,
        top_k=top_k,
        tokens=torch.tensor([t for rows in per_expert for t in rows], dtype=torch.int32),
        m_indptr=m_indptr,
        m_valid=m_indptr.diff(),
        slots=torch.zeros(m * top_k, dtype=torch.int64),
        commitments=torch.zeros(64, dtype=torch.uint8),
    )
    ctx = SimpleNamespace(
        config=config,
        key_a=key_a,
        key_b=key_b,
        seed_b=seed_b,
        target=job.target,
        job=job,
        b_proof=SimpleNamespace(prebuilt_commitment=lambda: PrebuiltCommitment(comm_b, key_b)),
    )
    state = SimpleNamespace(
        layer_name="test.experts",
        layer_id=9,
        n=experts * n_e,
        lottery_n=n_e,
        experts=experts,
        k=k,
        weight_cpu=b.int_values,
        weight_scale_cpu=b.scales,
        lock=threading.Lock(),
        job_ctx=ctx,
    )
    hit = SimpleNamespace(
        valid=True,
        layer_id=state.layer_id,
        m=counts[min(group_id, experts - 1)],
        n=n_e,
        k=k,
        ltile_rows=config.rows_pattern.tile_size,
        ltile_cols=config.cols_pattern.tile_size,
        tile_row=2,
        tile_column=1,
        group_id=group_id,
        commitment_hash_A=b"\x07" * 32,
        commitment_hash_B=seed_b,
        target=threshold_bytes_for(job, k, n_e, device=_DEVICE),
        codes=None,
        scales=None,
    )
    moe = MoeLaunch(routing, a.int_values, a.scales)
    return job, state, hit, moe, per_expert


def test_moe_hit_opens_the_routed_tokens_and_expert_rows(monkeypatch):
    """A payload-less MoE record resolves through the launch's retained
    routing: its expert-local tile becomes the expert's routed tokens and its
    stacked weight rows, and the opening carries a routing witness the cert-v4
    verifier accepts."""
    from miner_base.block_submission import create_proof
    from pearl_mining import (
        CERT_VERSION_PLAIN_FP8,
        IncompleteBlockHeader,
        verify_plain_proof_for_cert_version,
    )

    w = 2
    job, state, hit, moe, per_expert = _moe_winner_fixture(w)
    submitted = []

    class Manager:
        @staticmethod
        def get_mining_job():
            return job

        @staticmethod
        def handle_submit_block(opening, _job, *, timeout):
            submitted.append(opening)
            return True

    monkeypatch.setattr(winners, "lookup_state_by_layer_id", lambda _layer_id: state)
    callback = WinnerCheckCallback(
        signal=_Signal(hit),
        event=object(),
        leases=_Leases(),
        manager=Manager(),
        pow_key_host=torch.zeros(32, dtype=torch.uint8),
        moe=moe,
    )
    callback._handle_hit(hit)

    (opening,) = submitted
    config = state.job_ctx.config
    inner = tile_indices(config.rows_pattern, hit.tile_row)
    assert opening.moe is not None and opening.moe.expert_index == w
    assert opening.moe.inner_a_rows == tuple(inner)
    assert opening.a_row_indices == tuple(per_expert[w][i] for i in inner)
    assert opening.b_column_indices == tuple(
        w * state.lottery_n + c for c in tile_indices(config.cols_pattern, hit.tile_column)
    )
    assert opening.a_codes is moe.codes and opening.b_codes is state.weight_cpu
    header = IncompleteBlockHeader.from_bytes(job.incomplete_header_bytes)
    proof = create_proof(opening, header)
    assert proof.moe is not None and proof.moe_witness.w == w
    accepted, message = verify_plain_proof_for_cert_version(CERT_VERSION_PLAIN_FP8, header, proof)
    assert accepted, message


def test_moe_hit_outside_the_expert_range_is_dropped():
    job, state, hit, moe, _ = _moe_winner_fixture(4)
    manager = SimpleNamespace(get_mining_job=lambda: job)
    assert winners._matching_context(hit, state, manager, moe=moe) is None
    # A dense check never opens a payload-less record.
    assert winners._matching_context(hit, state, manager) is None


def test_launch_asks_submission_gate_about_its_captured_job(monkeypatch):
    legacy_job = MiningJob(
        incomplete_header_bytes=bytes(range(76)),
        target=1,
        cert_version=CertificateVersion.ZK_DENSE,
    )
    checked_jobs = []

    class Manager:
        def __init__(self):
            self.credited = 0

        def mining_launch_admission(self, job):
            checked_jobs.append(job)
            return nullcontext(SimpleNamespace(launch=True, retain_winner=False))

        def increment_credited_hashes(self, count):
            self.credited += count

    order = []

    class Event:
        def record(self):
            order.append("event")

    manager = Manager()
    output = MagicMock()
    output.__getitem__.return_value = output
    bias = object()
    output.add_.side_effect = lambda candidate: order.append(("bias", candidate)) or output
    launch_args = []
    monkeypatch.setattr(pipeline, "get_async_manager", lambda: manager)
    monkeypatch.setattr(pipeline, "_hit_signal_for", lambda _device: object())
    monkeypatch.setattr(pipeline, "_ensure_fallback_runner_started", lambda: None)
    activation = MagicMock()
    monkeypatch.setattr(pipeline.torch, "empty", lambda *_args, **_kwargs: activation)

    def launch(*_args, **kwargs):
        launch_args.append(kwargs)
        return object(), object(), object(), output

    monkeypatch.setattr(pipeline, "_launch_stages", launch)
    monkeypatch.setattr(pipeline.torch.cuda, "Event", Event)
    monkeypatch.setattr(pipeline, "register_mining_completion", lambda *_args, **_kwargs: None)

    def complete(
        _event,
        callback,
        _release,
        *,
        manager,
        credited_hashes,
        completion_lease,
        **_kwargs,
    ):
        try:
            manager.increment_credited_hashes(credited_hashes)
            callback()
        finally:
            completion_lease.release()

    monkeypatch.setattr(pipeline, "_start_launch_completion", complete)
    state = SimpleNamespace(
        layer_id=5,
        k=2048,
        n=128,
        experts=0,
        top_k=0,
        committed_device=_DEVICE,
        weight=SimpleNamespace(device="cuda:0"),
    )
    ctx = SimpleNamespace(
        job=legacy_job,
        config=object(),
        threshold_dev=object(),
    )
    result = pipeline.mine_launch(
        state,
        ctx,
        256,
        8,
        lambda _activation: None,
        bias=bias,
    )

    assert result is output
    assert order == [("bias", bias), "event"]
    assert checked_jobs == [legacy_job]
    assert launch_args == [{"layer_id": 5, "record_hits": False, "routing": None}]
    assert manager.credited == effective_work_per_matmul(256, 128, 2048, device=_DEVICE)


def test_completion_registration_failure_synchronizes_exact_event(monkeypatch):
    order = []

    class Event:
        @staticmethod
        def synchronize():
            order.append("event")

    register_global = Mock(side_effect=RuntimeError("global registry failed"))
    monkeypatch.setattr(pipeline, "register_mining_completion", register_global)

    pipeline._register_launch_event(Event())

    assert order == ["event"]
    register_global.assert_called_once()


def test_serving_activation_writes_live_prefix_and_zeros_only_tail(monkeypatch):
    captured = []

    class Event:
        def record(self):
            pass

        def synchronize(self):
            pass

    manager = SimpleNamespace(increment_credited_hashes=lambda _count: None)
    decision = SimpleNamespace(retain_winner=False)
    state = SimpleNamespace(
        layer_id=5,
        k=8,
        n=128,
        experts=0,
        top_k=0,
        committed_device=_DEVICE,
        weight=SimpleNamespace(device=torch.device("cpu")),
    )
    ctx = SimpleNamespace(config=object())
    monkeypatch.setattr(pipeline, "_ensure_fallback_runner_started", lambda: None)
    monkeypatch.setattr(pipeline, "_hit_signal_for", lambda _device: object())
    monkeypatch.setattr(pipeline, "register_mining_completion", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline.torch.cuda, "Event", Event)
    # This test drives the activation/callback plumbing with a toy (n, k); the
    # credited-work path is exercised elsewhere, so stub it out of the shape.
    monkeypatch.setattr(pipeline, "effective_work_per_matmul", lambda *_a, **_k: 0)

    def complete(_event, callback, _release, *, completion_lease, **_kwargs):
        try:
            callback()
        finally:
            completion_lease.release()

    monkeypatch.setattr(pipeline, "_start_launch_completion", complete)

    def launch(_ctx, _config, activation, *_args, **_kwargs):
        captured.append(activation.clone())
        return object(), object(), object(), torch.zeros(activation.shape[0], 128)

    monkeypatch.setattr(pipeline, "_launch_stages", launch)
    live = torch.arange(24, dtype=torch.bfloat16).view(3, 8)

    pipeline._mine_admitted_launch(
        manager,
        decision,
        state,
        ctx,
        m_bucket=8,
        m_tokens=3,
        fill=lambda prefix: prefix.copy_(live),
    )
    full = torch.arange(64, dtype=torch.bfloat16).view(8, 8)
    pipeline._mine_admitted_launch(
        manager,
        decision,
        state,
        ctx,
        m_bucket=8,
        m_tokens=8,
        fill=lambda prefix: prefix.copy_(full),
    )

    assert torch.equal(captured[0][:3], live)
    assert torch.count_nonzero(captured[0][3:]) == 0
    assert torch.equal(captured[1], full)


def test_partial_fill_failure_registers_cleanup_before_releasing_slot(monkeypatch):
    pool = pipeline._CompletionLeasePool()
    cleanup_event = object()

    def observe_registration(_event):
        assert pool.in_use() == 1

    register = Mock(side_effect=observe_registration)
    state = SimpleNamespace(
        layer_id=5,
        k=8,
        n=128,
        experts=0,
        top_k=0,
        committed_device=_DEVICE,
        weight=SimpleNamespace(device=torch.device("cpu")),
    )
    ctx = SimpleNamespace(config=object())
    decision = SimpleNamespace(retain_winner=False)
    manager = SimpleNamespace(increment_credited_hashes=lambda _count: None)
    monkeypatch.setattr(pipeline, "_COMPLETION_LEASES", pool)
    monkeypatch.setattr(
        pipeline,
        "runtime_settings",
        lambda: SimpleNamespace(completion_inflight_limit=2),
    )
    monkeypatch.setattr(pipeline, "_ensure_fallback_runner_started", lambda: None)
    monkeypatch.setattr(pipeline.torch, "empty", lambda *_args, **_kwargs: MagicMock())
    monkeypatch.setattr(pipeline, "_record_stream_event", lambda: cleanup_event)
    monkeypatch.setattr(pipeline, "_register_launch_event", register)

    def partial_fill(_activation):
        raise RuntimeError("partial fill failed")

    with pytest.raises(RuntimeError, match="partial fill failed"):
        pipeline._mine_admitted_launch(
            manager,
            decision,
            state,
            ctx,
            8,
            8,
            partial_fill,
        )

    register.assert_called_once_with(cleanup_event)
    assert pool.in_use() == 0


def test_unretained_launches_decline_before_allocation_when_completions_are_full(
    monkeypatch,
):
    held = []
    pool = pipeline._CompletionLeasePool()
    allocate = Mock(return_value=MagicMock())
    output = object()
    state = SimpleNamespace(
        layer_id=5,
        k=8,
        n=128,
        experts=0,
        top_k=0,
        committed_device=_DEVICE,
        weight=SimpleNamespace(device=torch.device("cpu")),
    )
    ctx = SimpleNamespace(config=object())
    decision = SimpleNamespace(retain_winner=False)
    manager = SimpleNamespace(increment_credited_hashes=lambda _count: None)
    monkeypatch.setattr(pipeline, "_COMPLETION_LEASES", pool)
    monkeypatch.setattr(
        pipeline,
        "runtime_settings",
        lambda: SimpleNamespace(completion_inflight_limit=2),
    )
    monkeypatch.setattr(pipeline, "_ensure_fallback_runner_started", lambda: None)
    monkeypatch.setattr(pipeline.torch, "empty", allocate)
    # Toy (n, k) here drives only lease/decline plumbing; credited work is
    # covered elsewhere, so keep it out of the shape.
    monkeypatch.setattr(pipeline, "effective_work_per_matmul", lambda *_a, **_k: 0)
    monkeypatch.setattr(pipeline, "_hit_signal_for", lambda _device: object())
    monkeypatch.setattr(
        pipeline,
        "_launch_stages",
        lambda *_args, **_kwargs: (object(), object(), object(), output),
    )
    monkeypatch.setattr(pipeline, "_record_stream_event", lambda: object())
    monkeypatch.setattr(pipeline, "_register_launch_event", lambda *_args: None)

    def hold_completion(_event, _callback, _release, *, completion_lease, **_kwargs):
        held.append(completion_lease)

    monkeypatch.setattr(pipeline, "_start_launch_completion", hold_completion)

    def launch():
        launched = pipeline._mine_admitted_launch(
            manager, decision, state, ctx, 8, 8, lambda _a: None
        )
        return launched is not None

    assert launch()
    assert launch()
    assert not launch()
    assert allocate.call_count == 2
    assert pool.in_use() == 2

    for lease in held:
        lease.release()
    assert pool.in_use() == 0


def test_launch_declines_a_job_replaced_before_launch_admission(monkeypatch):
    stale_job = Mock(spec=MiningJob)

    class Manager:
        credited = 0

        @staticmethod
        def mining_launch_admission(job):
            assert job is stale_job
            return nullcontext(SimpleNamespace(launch=False, retain_winner=False))

        def increment_credited_hashes(self, count):
            self.credited += count

    manager = Manager()
    monkeypatch.setattr(pipeline, "get_async_manager", lambda: manager)
    allocate = Mock(side_effect=AssertionError("stale launch allocated GPU work"))
    monkeypatch.setattr(pipeline.torch, "empty", allocate)
    state = SimpleNamespace(
        k=2048, n=128, experts=0, top_k=0, weight=SimpleNamespace(device="cuda:0")
    )
    ctx = SimpleNamespace(job=stale_job, threshold_dev=object())

    assert pipeline.mine_launch(state, ctx, 256, 8, lambda _activation: None) is None
    allocate.assert_not_called()
    assert manager.credited == 0


def test_winner_check_lease_underflow_is_an_error():
    leases = pipeline.WinnerCheckLeases()

    with pytest.raises(RuntimeError, match="without an owner"):
        leases.release()


def test_fallback_runner_failure_happens_before_retained_gpu_launch(monkeypatch):
    job = Mock(spec=MiningJob)

    class BrokenThread:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def start():
            raise RuntimeError("synthetic thread exhaustion")

    manager = SimpleNamespace(
        mining_launch_admission=lambda candidate: nullcontext(
            SimpleNamespace(
                launch=candidate is job,
                retain_winner=True,
            )
        )
    )
    monkeypatch.setattr(pipeline, "get_async_manager", lambda: manager)
    monkeypatch.setattr(pipeline, "_fallback_thread", None)
    monkeypatch.setattr(pipeline.threading, "Thread", BrokenThread)
    allocate = Mock(side_effect=AssertionError("GPU work started before fallback setup"))
    monkeypatch.setattr(pipeline.torch, "empty", allocate)
    state = SimpleNamespace(
        k=2048, n=128, experts=0, top_k=0, weight=SimpleNamespace(device="cuda:0")
    )
    ctx = SimpleNamespace(job=job, threshold_dev=object())

    with pytest.raises(RuntimeError, match="thread exhaustion"):
        pipeline.mine_launch(state, ctx, 256, 8, lambda _activation: None)

    allocate.assert_not_called()


def _run_fallback_for_test(
    event,
    callback,
    release,
    *,
    manager=None,
    credited_hashes=0,
    completion_lease=None,
):
    completion = pipeline._FallbackCompletion()
    with pipeline._fallback_check_lock:
        pipeline._fallback_checks += 1
    pipeline._run_fallback_work(
        pipeline._FallbackWork(
            event=event,
            callback=callback,
            release=release,
            completion=completion,
            manager=manager,
            credited_hashes=credited_hashes,
            completion_lease=completion_lease,
        )
    )
    assert completion.done.is_set()


def test_lifecycle_wait_includes_event_gated_host_continuations():
    with pipeline._fallback_check_lock:
        pipeline._fallback_checks += 1
    try:
        assert not pipeline.wait_for_mining_continuations_idle(0)
    finally:
        with pipeline._fallback_check_lock:
            pipeline._fallback_checks -= 1
            pipeline._fallback_check_lock.notify_all()
    assert pipeline.wait_for_mining_continuations_idle(0)


def test_lifecycle_wait_blocks_on_a_real_host_continuation():
    callback_entered = threading.Event()
    allow_callback = threading.Event()

    class Event:
        @staticmethod
        def synchronize():
            return None

    def callback():
        callback_entered.set()
        assert allow_callback.wait(1)

    pipeline._ensure_fallback_runner_started()
    pipeline._start_launch_completion(Event(), callback, lambda: None)
    assert callback_entered.wait(1)
    assert not pipeline.wait_for_mining_continuations_idle(0)

    allow_callback.set()
    assert pipeline.wait_for_mining_continuations_idle(1)


def test_fallback_check_runs_after_its_event():
    order = []

    class Event:
        @staticmethod
        def synchronize():
            order.append("event")

    _run_fallback_for_test(
        Event(),
        lambda: order.append("callback"),
        lambda: order.append("release"),
    )

    assert order == ["event", "callback"]
    assert pipeline.fallback_winner_checks_idle()


def test_credit_is_published_only_after_successful_event_completion():
    order = []
    manager = SimpleNamespace(
        increment_credited_hashes=lambda count: order.append(("credit", count))
    )

    class Event:
        @staticmethod
        def synchronize():
            order.append(("event", None))

    _run_fallback_for_test(
        Event(),
        lambda: order.append(("callback", None)),
        lambda: None,
        manager=manager,
        credited_hashes=17,
    )

    assert order == [("event", None), ("credit", 17), ("callback", None)]


def test_fallback_check_releases_when_event_wait_fails():
    releases = []
    pool = pipeline._CompletionLeasePool()
    completion_lease = pool.try_acquire(1)
    assert completion_lease is not None

    class Event:
        @staticmethod
        def synchronize():
            raise RuntimeError("synthetic event failure")

    manager = Mock()
    _run_fallback_for_test(
        Event(),
        lambda: pytest.fail("callback ran without a completed event"),
        lambda: releases.append(True),
        manager=manager,
        credited_hashes=17,
        completion_lease=completion_lease,
    )

    assert releases == [True]
    assert pool.in_use() == 0
    manager.increment_credited_hashes.assert_not_called()
    assert pipeline.fallback_winner_checks_idle()


@pytest.mark.parametrize("failure", ["register", "publish"])
def test_failed_fallback_publication_runs_check_synchronously(monkeypatch, failure):
    order = []

    class Event:
        @staticmethod
        def synchronize():
            order.append("event")

    def release():
        order.append("release")

    def callback():
        order.append("callback")
        release()

    register = Mock()
    publish = Mock()
    if failure == "register":
        register.side_effect = RuntimeError("synthetic registration failure")
    else:
        publish.side_effect = RuntimeError("synthetic publication failure")
    monkeypatch.setattr(pipeline, "_ensure_fallback_runner_started", lambda: None)
    monkeypatch.setattr(pipeline, "register_mining_completion", register)
    monkeypatch.setattr(pipeline, "_fallback_queue", SimpleNamespace(put_nowait=publish))

    pipeline._start_launch_completion(Event(), callback, release)

    assert order == ["event", "callback", "release"]
    assert pipeline.fallback_winner_checks_idle()


def test_winner_retention_uses_its_own_configured_capacity(monkeypatch):
    job = Mock(spec=MiningJob)
    manager = SimpleNamespace(
        mining_launch_admission=lambda candidate: nullcontext(
            SimpleNamespace(
                launch=candidate is job,
                retain_winner=True,
            )
        )
    )
    leases = Mock()
    leases.try_acquire.return_value = False
    monkeypatch.setattr(pipeline, "get_async_manager", lambda: manager)
    monkeypatch.setattr(pipeline, "WINNER_LEASES", leases)
    monkeypatch.setattr(
        pipeline,
        "runtime_settings",
        lambda: SimpleNamespace(
            winner_check_inflight_limit=9,
            completion_inflight_limit=8,
        ),
    )
    allocate = Mock(side_effect=AssertionError("unretained launch allocated GPU work"))
    monkeypatch.setattr(pipeline.torch, "empty", allocate)
    state = SimpleNamespace(
        k=2048, n=128, experts=0, top_k=0, weight=SimpleNamespace(device="cuda:0")
    )
    ctx = SimpleNamespace(job=job, threshold_dev=object())

    assert pipeline.mine_launch(state, ctx, 256, 8, lambda _activation: None) is None
    leases.try_acquire.assert_called_once_with(9)
    allocate.assert_not_called()
