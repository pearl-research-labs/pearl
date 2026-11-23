"""Per-job B-side preparation: in place, synchronous, on the caller's stream."""

import time
from dataclasses import replace

import torch
from miner_base.protocol import noise_line_key, noise_seed_b, operand_digest
from miner_utils import get_logger
from pearl_gateway.comm.dataclasses import MiningJob

from .capture import mining_attempt
from .cpu_b import CpuPrepared, cpu_prepared_waiting, discard_cpu_prepared, take_cpu_prepared
from .mining_config import commitment_keys_for, mining_configuration, threshold_bytes_for
from .settings import miner_settings, runtime_settings
from .state import BProofContext, JobContext, LayerBuffers, LayerState, all_states

_LOGGER = get_logger(__name__)
_CPU_UPLOADED = (
    "f1",
    "f1_hl",
    "f2",
    "e2",
    "alpha_b",
    "beta_b",
    "inv_alpha_b",
    "b_prime",
    "b_peel",
)


def _as_bytes_tensor(payload: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8)


def _prepare_b_on_gpu(
    state: LayerState, key_a: bytes, key_b: bytes, p_b: bytes, buffers: LayerBuffers
) -> bytes:
    """Run B-side commit, noise lines, and noisy quantization into ``buffers``."""
    from pearl_gemm import (
        LABEL_F1,
        LABEL_F2,
        noise_lines,
        noisy_quant_b,
        pack_noise_factor,
        tensor_hash_plus_stats_b,
    )

    buffers.key_a_dev.copy_(_as_bytes_tensor(key_a))
    buffers.key_b_dev.copy_(_as_bytes_tensor(key_b))
    tensor_hash_plus_stats_b(
        state.weight,
        state.weight_scale,
        buffers.key_b_dev,
        buffers.root_codes,
        buffers.root_scales,
        buffers.tensor_hash_workspace,
        buffers.commit_stats,
        config=buffers.commit_config,
    )
    hash_b = operand_digest(
        [bytes(buffers.root_codes.cpu()), bytes(buffers.root_scales.cpu())], key_b
    )
    seed_b = noise_seed_b(hash_b, key_b, p_b)

    buffers.seed_b_dev.copy_(_as_bytes_tensor(seed_b))
    buffers.noise_key_b_dev.copy_(_as_bytes_tensor(noise_line_key(seed_b)))
    for label, factor, packed in (
        (LABEL_F1, buffers.f1, buffers.f1_hl),
        (LABEL_F2, buffers.f2, buffers.f2_hl),
    ):
        noise_lines(buffers.noise_key_b_dev, label, buffers.noise_lines)
        factor.copy_(buffers.noise_lines.t())
        packed.copy_(pack_noise_factor(factor))

    noisy_quant_b(
        state.weight,
        state.weight_scale,
        buffers.noise_key_b_dev,
        buffers.commit_stats,
        buffers.f2_hl,
        buffers.f1,
        buffers.alpha_b,
        buffers.beta_b,
        buffers.e2,
        buffers.b_prime,
        buffers.b_peel,
        buffers.gram,
        config=buffers.prepare_config,
    )
    buffers.inv_alpha_b.copy_(torch.reciprocal(buffers.alpha_b.float()))
    return seed_b


def _publish(state: LayerState, ctx: JobContext) -> bool:
    with state.lock:
        if not state.mineable:
            return False
        state.job_ctx = ctx
    return True


def _publish_cpu_prepared(
    state: LayerState, job: MiningJob, ready: CpuPrepared, threshold: torch.Tensor
) -> JobContext | None:
    buffers = state.buffers
    assert buffers is not None
    prepared = ready.prepared
    key_a, _ = commitment_keys_for(job, prepared.b_proof.ancestor_header)
    uploaded = torch.cuda.Event()
    try:
        for name in _CPU_UPLOADED:
            getattr(buffers, name).copy_(getattr(prepared, name), non_blocking=True)
        buffers.key_a_dev.copy_(_as_bytes_tensor(key_a))
        buffers.key_b_dev.copy_(_as_bytes_tensor(prepared.key_b))
        buffers.seed_b_dev.copy_(_as_bytes_tensor(prepared.seed_b))
        buffers.noise_key_b_dev.copy_(_as_bytes_tensor(noise_line_key(prepared.seed_b)))
        buffers.threshold_dev.copy_(threshold)
    finally:
        uploaded.record()
        ready.uploaded(uploaded)
    ctx = JobContext(
        job=job,
        config=prepared.config,
        key_a=key_a,
        key_b=prepared.key_b,
        seed_b=prepared.seed_b,
        target=job.target,
        b_proof=prepared.b_proof,
    )
    if not _publish(state, ctx):
        return None
    _LOGGER.debug(f"{state.layer_name}: published CPU-prepared B")
    return ctx


def prepare_layer(state: LayerState, job: MiningJob) -> JobContext | None:
    """Update this layer's steady B buffers for ``job`` and publish the new JobContext."""
    assert state.buffers is not None
    buffers = state.buffers
    torch.cuda.set_device(state.weight.device)
    started = time.monotonic()

    device = state.committed_device
    threshold = _as_bytes_tensor(threshold_bytes_for(job, state.k, state.lottery_n, device=device))
    with state.lock:
        current = state.job_ctx

    ready = take_cpu_prepared(
        state, job, None if current is None else current.b_proof.ancestor_header
    )
    if ready is not None:
        return _publish_cpu_prepared(state, job, ready, threshold)

    if current is not None and job.keeps_ancestor(current.b_proof.ancestor_header):
        key_a, _ = commitment_keys_for(job, current.b_proof.ancestor_header)
        if current.key_a != key_a:
            buffers.key_a_dev.copy_(_as_bytes_tensor(key_a))
        buffers.threshold_dev.copy_(threshold)
        ctx = replace(current, job=job, key_a=key_a, target=job.target)
        if not _publish(state, ctx):
            return None
        _LOGGER.debug(f"{state.layer_name}: reused B (target={job.target})")
        return ctx

    key_a, key_b = commitment_keys_for(job)
    config = mining_configuration(state.k, state.n, state.experts, device=device)
    p_b = config.p_b(state.n)
    seed_b = _prepare_b_on_gpu(state, key_a, key_b, p_b, buffers)
    buffers.threshold_dev.copy_(threshold)
    ctx = JobContext(
        job=job,
        config=config,
        key_a=key_a,
        key_b=key_b,
        seed_b=seed_b,
        target=job.target,
        b_proof=BProofContext(
            key_b=key_b,
            ancestor_header=bytes(job.parent_header),
            seed_b=seed_b,
            p_b=p_b,
            codes=state.weight_cpu,
            scales=state.weight_scale_cpu,
            commit_leaf=buffers.commit_config.chunk_size,
        ),
    )
    if not _publish(state, ctx):
        return None
    _LOGGER.debug(
        f"prepared {state.layer_name} (n={state.n}, k={state.k}) in "
        f"{time.monotonic() - started:.3f}s"
    )
    return ctx


def current_job() -> MiningJob | None:
    from .runtime import try_get_async_manager

    try:
        manager = try_get_async_manager()
    except RuntimeError:
        return None
    return None if manager is None else manager.get_mining_job()


def current_context(state: LayerState, job: MiningJob | None) -> JobContext | None:
    if job is None or not state.mineable:
        return None
    with state.lock:
        ctx = state.job_ctx
    if ctx is not None and ctx.job == job and not cpu_prepared_waiting(state):
        return ctx
    return prepare_layer(state, job)


def unpublish_contexts() -> None:
    for state in all_states():
        with state.lock:
            state.job_ctx = None
    discard_cpu_prepared()


def prepare_mining(job: MiningJob | None) -> bool:
    """Compile configured variants and prepare B for ``job`` on the worker thread."""
    from .pipeline import warmup_layer_variants

    if miner_settings().no_mining:
        _LOGGER.warning(
            "MINER_NO_MINING is set: skipping B preparation and kernel warmup. "
            "Encoded layers serve on the FP8 fallback.",
        )
        return False
    states = [state for state in all_states() if state.mineable]
    if not states:
        return job is not None
    started = time.monotonic()
    ready = job is not None
    for state in states:
        try:
            with mining_attempt(state.weight.device) as attempt:
                if not attempt:
                    ready = False
                    continue
                if runtime_settings().warmup_compile and not warmup_layer_variants(state):
                    state.disable_mining("configured pipeline warmup could not become ready")
                    ready = False
                    continue
                if job is not None and current_context(state, job) is None:
                    ready = False
                    continue
                attempt.mark_success()
        except torch.cuda.OutOfMemoryError:
            _LOGGER.warning(f"mining preparation OOM on {state.layer_name}; device cooldown")
            ready = False
        except Exception as exc:
            state.disable_mining(f"mining preparation failed ({type(exc).__name__})")
            ready = False
    _LOGGER.info(
        f"Mining preparation of {len(states)} layer(s) took {time.monotonic() - started:.1f}s "
        f"(job_ready={job is not None}, all_ready={ready})"
    )
    return ready
