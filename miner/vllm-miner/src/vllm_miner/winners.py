"""Persistent hit-signal decoding and proof handoff."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from miner_base.async_loop_manager import AsyncLoopManager
from miner_base.block_submission import (
    MoEBlockInfo,
    OpenedBlockInfo,
    PrebuiltCommitment,
    commit_planes_for_leaf,
)
from miner_base.mining_config import activation_leaf
from miner_utils import get_logger

from .health import disable_device_mining
from .mining_config import threshold_bytes_for, tile_indices
from .state import (
    JobContext,
    LayerState,
    all_states,
    lookup_state_by_layer_id,
)

if TYPE_CHECKING:
    from pearl_gemm import Hit, HitSignal

    from .moe import MoeLaunch
    from .pipeline import WinnerCheckLeases

_LOGGER = get_logger(__name__)
# Established gateway reads are intentionally unbounded, but a winner must not
# wait forever behind saturated proof capacity. This matches the
# integration/measurement flush budget and remains below the framework's
# 60-second mining-quiescence bound.
_WINNER_HANDOFF_TIMEOUT_S = 30.0


def _matching_context(
    hit: "Hit",
    state: LayerState,
    manager: AsyncLoopManager,
    *,
    moe: "MoeLaunch | None" = None,
) -> JobContext | None:
    """Return the live context described by a persistent record, if any.

    The seedB and target words below were stamped by the kernel from the
    device buffers at launch time, so they pin the hit to the job the layer
    was prepared for when it launched. A dense record must carry its payload
    (the tile's committed A rows); an MoE record is payload-less by design
    and is opened from the launch's retained planes (``moe``).
    """
    with state.lock:
        ctx = state.job_ctx
    if ctx is None:
        _LOGGER.info(f"dropping stale winner for {state.layer_name}")
        return None
    if ctx.job != manager.get_mining_job():
        _LOGGER.info(f"dropping winner for replaced job on {state.layer_name}")
        return None
    if (
        (hit.n, hit.k) != (state.lottery_n, state.k)
        or hit.ltile_rows != ctx.config.rows_pattern.tile_size
        or hit.ltile_cols != ctx.config.cols_pattern.tile_size
    ):
        _LOGGER.warning(
            "dropping hit with mismatched layer geometry: "
            f"layer_id={hit.layer_id} dims=({hit.m}, {hit.n}, {hit.k}) "
            f"tile=({hit.tile_row}, {hit.tile_column}) "
            f"ltile=({hit.ltile_rows}, {hit.ltile_cols})"
        )
        return None
    if hit.commitment_hash_B != ctx.seed_b:
        _LOGGER.info(f"dropping stale B seed for {state.layer_name}")
        return None
    if hit.target != threshold_bytes_for(ctx.job, state.k, state.lottery_n):
        _LOGGER.info(f"dropping stale target for {state.layer_name}")
        return None
    if moe is None and (hit.codes is None or hit.scales is None):
        _LOGGER.warning(f"dropping payload-less hit for {state.layer_name}")
        return None
    if moe is not None and hit.group_id >= moe.routing.experts:
        _LOGGER.warning(
            f"dropping MoE hit with expert {hit.group_id} outside {moe.routing.experts} "
            f"experts for {state.layer_name}"
        )
        return None
    return ctx


@dataclass
class WinnerCheckCallback:
    """Consume at most one record after an admitted launch event completes."""

    signal: "HitSignal"
    event: torch.cuda.Event
    leases: "WinnerCheckLeases"
    manager: AsyncLoopManager
    # The launch's jackpot key (``pow_key``), copied to pinned host memory on
    # the launch stream ahead of ``event``: the record's ``HASH_A`` names the
    # launch that published it, and this callback consumes only that
    # launch's record (``HitSignal.take_owned_hit``).
    pow_key_host: torch.Tensor | None = None
    # MoE launches keep their routing and unpermuted planes alive until the
    # check: the record is payload-less (the expert's rows are a slice of
    # the committed activation), and the proof opens the full committed
    # activation at the tile's tokens.
    moe: "MoeLaunch | None" = None

    def __post_init__(self) -> None:
        if self.moe is not None and self.pow_key_host is None:
            raise ValueError("an MoE winner check must own its launch's pow_key")

    def __call__(self) -> None:
        try:
            # One signal-level critical section owns the reusable payload and
            # re-arms before slow CPU validation. Concurrent manager/fallback
            # callbacks cannot both consume one record or reset a newer claim,
            # and a record another launch published is left for its owner.
            owner = None if self.pow_key_host is None else bytes(self.pow_key_host.tolist())
            hit = self.signal.take_owned_hit(owner)
            if hit is None:
                return
            if not hit.valid:
                _LOGGER.warning("dropping malformed persistent hit record (valid=False)")
                return
            self._handle_hit(hit)
        except Exception as exc:
            from pearl_gemm import HitSignalPoisonedError

            if isinstance(exc, HitSignalPoisonedError):
                disable_mining_after_poisoning(self.signal)
            else:
                _LOGGER.opt(exception=True).warning("persistent winner validation crashed")
        finally:
            self.leases.release()

    def _handle_hit(self, hit: "Hit") -> None:
        try:
            state = lookup_state_by_layer_id(hit.layer_id)
        except RuntimeError:
            _LOGGER.warning(f"dropping hit for unknown layer id {hit.layer_id}")
            return

        ctx = _matching_context(hit, state, self.manager, moe=self.moe)
        if ctx is None:
            return

        config = ctx.config
        a_rows = tile_indices(config.rows_pattern, hit.tile_row)
        b_rows = tile_indices(config.cols_pattern, hit.tile_column)
        codes_cpu = hit.codes
        scales_cpu = hit.scales
        m = hit.m
        moe_info = None
        if self.moe is not None:
            # The record's ``m`` and ``tile_row`` are the winning expert's; the
            # committed A is the launch's full activation, and the
            # tile's rows are that expert's tokens at the expert-local rows.
            routing = self.moe.routing
            moe_info = MoEBlockInfo(
                expert_index=hit.group_id,
                inner_a_rows=tuple(a_rows),
                routing=tuple(routing.tokens.tolist()),
                offsets=tuple(routing.m_indptr[1:].tolist()),
            )
            expert_rows = moe_info.expert_rows()
            a_rows = [expert_rows[i] for i in a_rows]
            b_rows = [hit.group_id * state.lottery_n + r for r in b_rows]
            codes_cpu, scales_cpu = self.moe.codes.cpu(), self.moe.scales.cpu()
            m = codes_cpu.shape[0]
        pow_key = hit.commitment_hash_A  # the launch's jackpot key
        expert = "" if moe_info is None else f" expert={moe_info.expert_index}"
        _LOGGER.info(
            f"block candidate! layer={state.layer_name} "
            f"tile=({hit.tile_row}, {hit.tile_column}){expert} "
            f"m={m} n={state.n} k={state.k} jackpot_key={pow_key.hex()[:16]}..."
        )

        # The hit's GPU commit ran at the committed ACTIVATION leaf (forced on
        # every launchable A-side TensorHashConfig); rebuild the same tree for
        # the opening.
        comm_a = commit_planes_for_leaf(
            [codes_cpu, scales_cpu], ctx.key_a, activation_leaf(ctx.config)
        )

        opening = OpenedBlockInfo(
            a_row_indices=tuple(a_rows),
            b_column_indices=tuple(b_rows),
            a_codes=codes_cpu,
            a_scales=scales_cpu,
            b_codes=state.weight_cpu,
            b_scales=state.weight_scale_cpu,
            mining_config=config,
            b_commitment=ctx.b_proof.prebuilt_commitment(),
            a_commitment=PrebuiltCommitment(comm_a, ctx.key_a),
            moe=moe_info,
        )
        submitted = ctx.job == self.manager.get_mining_job() and self.manager.handle_submit_block(
            opening,
            ctx.job,
            timeout=_WINNER_HANDOFF_TIMEOUT_S,
        )
        if submitted:
            _LOGGER.info(f"winner handed off for canonical proof validation ({state.layer_name})")
        else:
            _LOGGER.error(
                "validated lottery winner became stale or was not queued for submission "
                f"({state.layer_name})"
            )


def disable_mining_after_poisoning(signal: "HitSignal") -> None:
    """A signal that could not be re-armed stops mining on its device: the
    device breaker and every layer living there. Called from the handler of
    the ``HitSignalPoisonedError`` so the log carries its traceback."""
    reason = "persistent hit signal could not be re-armed"
    disable_device_mining(signal.device, reason)
    for state in all_states():
        if state.weight.device == signal.device:
            state.disable_mining(reason)
    _LOGGER.opt(exception=True).error(
        f"device mining disabled after hit-signal poisoning on {signal.device}"
    )
