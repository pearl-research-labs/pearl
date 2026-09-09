"""Per-job B-side preparation: in place, synchronous, on the caller's stream.

v4 keys B under the ancestor header (the miner proposes at depth 0, so that is
the proposed header itself): ``B'``, its peel, ``E_B``/``F_B`` and noise
``seedB`` change with every job. :func:`prepare_layer` runs the GPU B chain
(``tensor_hash_plus_stats_b -> seedB -> noise_lines -> noisy_quant_b``) into
the layer's steady :class:`~vllm_miner.state.LayerBuffers` on the
current CUDA stream and publishes a :class:`~vllm_miner.state.JobContext`
describing the job those operands now hold.

:func:`current_context` is the single entry every launch path takes first:
it returns the published context when it already describes ``job`` and
prepares in place otherwise. Preparation and mined launches are all
enqueued by the serving worker thread on its stream, so stream order alone
guarantees a launch never reads torn operands -- no background thread,
reader quiesce, or scratch publication is involved. The cost is that the
first launch of each layer after a job change pays that layer's B
preparation inline (a target-only change skips the chain and rewrites just
the 32-byte threshold).
"""

import time
from dataclasses import replace

import torch
from miner_base.commitment_hash import noise_line_key, noise_seed_b, operand_digest
from miner_utils import get_logger
from pearl_gateway.comm.dataclasses import MiningJob

from .config import config as gpu_config
from .mining_config import (
    commitment_keys_for,
    mining_configuration,
    threshold_bytes_for,
)
from .state import BProofContext, JobContext, LayerBuffers, LayerState, all_states

_LOGGER = get_logger(__name__)


def _as_bytes_tensor(payload: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8)


def _prepare_b_on_gpu(
    state: LayerState, key_a: bytes, key_b: bytes, p_b: bytes, buffers: LayerBuffers
) -> bytes:
    """Run commit, noise generation, and noisy quantization on the current stream.

    Returns B's noise seed. The v4 B chain: ``HB = blake3(roots, key=keyB)``,
    ``seedB = H_"seed-B"(HB || keyB || pB)``, and every B-side line (``E_B``
    in-kernel, ``F_B`` here) is drawn under ``Subkey("noise-line", seedB)``.
    ``F_A`` depends on each launch's A commitment, so B's peel mid half cannot
    be prepared per job: ``noisy_quant_b`` runs against an all-zero ``F_A``
    stand-in (``buffers.f1``, giving a zero mid half) and the pipeline rebuilds
    that half per launch. The root readback (``.cpu()``) synchronizes the
    stream, which is what makes ``seedB`` available to derive the noise key.
    """
    from pearl_gemm import (
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
    noise_lines(buffers.noise_key_b_dev, LABEL_F2, buffers.noise_lines)
    buffers.f2.copy_(buffers.noise_lines.t())
    buffers.f2_hl.copy_(pack_noise_factor(buffers.f2))
    buffers.f1.zero_()

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


def layer_job_keys(job: MiningJob) -> tuple[bytes, bytes]:
    """``(keyA, keyB)`` under ``job``. v4 keys depend on the header alone; the
    layer's committed tile enters through ``pB`` in ``seedB`` instead."""
    return commitment_keys_for(job)


def _publish(state: LayerState, ctx: JobContext) -> bool:
    with state.lock:
        if not state.mineable:
            return False
        state.job_ctx = ctx
    return True


def prepare_layer(state: LayerState, job: MiningJob) -> JobContext | None:
    """Rewrite this layer's B operands for ``job`` and publish the new context.

    Runs on the caller's current stream. A job that only changed the lottery
    target (same header, hence the same keys, ``seedB`` and every B operand)
    rewrites just the 32-byte threshold. Returns None only when the layer was
    disabled meanwhile; failures propagate so the caller can isolate the layer.
    """
    assert state.buffers is not None
    buffers = state.buffers
    torch.cuda.set_device(state.weight.device)
    started = time.monotonic()

    key_a, key_b = layer_job_keys(job)
    threshold = _as_bytes_tensor(threshold_bytes_for(job, state.k, state.n))
    with state.lock:
        current = state.job_ctx
    if current is not None and (current.key_a, current.key_b) == (key_a, key_b):
        buffers.threshold_dev.copy_(threshold)
        ctx = replace(current, job=job, target=job.target)
        if not _publish(state, ctx):
            return None
        _LOGGER.debug(f"{state.layer_name}: target-only refresh (target={job.target})")
        return ctx

    config = mining_configuration(state.k, state.n)
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
        key_a_dev=buffers.key_a_dev,
        seed_b_dev=buffers.seed_b_dev,
        threshold_dev=buffers.threshold_dev,
        f2=buffers.f2,
        e2=buffers.e2,
        beta_b=buffers.beta_b,
        b_prime=buffers.b_prime,
        b_peel=buffers.b_peel,
        alpha_b=buffers.alpha_b,
        inv_alpha_b=buffers.inv_alpha_b,
        b_proof=BProofContext(
            key_b=key_b,
            seed_b=seed_b,
            p_b=p_b,
            codes=state.weight_cpu,
            scales=state.weight_scale_cpu,
            # The leaf the GPU commit above ran at; the proof-side rebuild
            # must open the same tree.
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
    """The manager's live job, or None before the runtime is up (or after a
    poisoned teardown)."""
    from .mining_state import try_get_async_manager

    try:
        manager = try_get_async_manager()
    except RuntimeError:
        return None
    return None if manager is None else manager.get_mining_job()


def current_context(state: LayerState, job: MiningJob | None) -> JobContext | None:
    """The layer's context for ``job``, preparing in place when it is not current.

    Returns None when there is no job or the layer cannot mine. Must be called
    on the stream the caller's launches use.
    """
    if job is None or not state.mineable:
        return None
    with state.lock:
        ctx = state.job_ctx
    if ctx is not None and ctx.job == job:
        return ctx
    return prepare_layer(state, job)


def unpublish_contexts() -> None:
    """Forget every layer's job after mining GPU work has been drained.

    The next launch re-prepares. Used by lifecycle boundaries (runtime
    teardown, weight reload) where the buffers may be rewritten or the job
    they hold can no longer be trusted.
    """
    for state in all_states():
        with state.lock:
            state.job_ctx = None


def prepare_mining(job: MiningJob | None) -> bool:
    """Startup preparation on the calling (serving worker) thread.

    Compiles every configured pipeline variant of each mineable layer
    (``PEARL_WARMUP_COMPILE``) and, when a job is already known, prepares each
    layer's B operands for it so the first serving forwards mine without an
    inline preparation stall. Returns whether every mineable layer is ready
    to mine ``job`` (always False without a job). Layer failures are isolated
    to the layer; serving keeps the FP8 fallback for it.
    """
    from .health import mining_attempt
    from .pipeline import warmup_layer_variants
    from .settings import runtime_settings

    if gpu_config.settings.no_mining:
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
