"""Persistent hit-signal decoding and proof handoff."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from miner_base.async_loop_manager import AsyncLoopManager
from miner_base.block_submission import MoEBlockInfo, OpenedBlockInfo, PrebuiltCommitment
from miner_base.commitment import commit_planes
from miner_base.protocol import a_keys
from miner_utils import get_logger

from .capture import disable_device_mining
from .mining_config import threshold_bytes_for, tile_indices
from .state import JobContext, LayerState, all_states, lookup_state_by_layer_id

if TYPE_CHECKING:
    from pearl_gemm import Hit, HitSignal

    from .moe import MoeLaunch
    from .pipeline import LeasePool

_LOGGER = get_logger(__name__)
_WINNER_HANDOFF_TIMEOUT_S = 30.0


def _matching_context(
    hit: "Hit",
    state: LayerState,
    manager: AsyncLoopManager,
    *,
    moe: "MoeLaunch | None" = None,
) -> JobContext | None:
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
        or hit.ltile_rows != ctx.config.tile_rows
        or hit.ltile_cols != ctx.config.tile_cols
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
    if hit.target != threshold_bytes_for(
        ctx.job, state.k, state.lottery_n, device=ctx.config.device
    ):
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
    leases: "LeasePool"
    manager: AsyncLoopManager
    pow_key_host: torch.Tensor | None = None
    moe: "MoeLaunch | None" = None

    def __post_init__(self) -> None:
        if self.moe is not None and self.pow_key_host is None:
            raise ValueError("an MoE winner check must own its launch's pow_key")

    def __call__(self) -> None:
        try:
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
        a_rows = tile_indices(config.tile_rows, hit.tile_row)
        b_rows = tile_indices(config.tile_cols, hit.tile_column)
        codes_cpu = hit.codes
        scales_cpu = hit.scales
        m = hit.m
        moe_info = None
        routing_commitments = b""
        if self.moe is not None:
            routing = self.moe.routing
            routing_commitments = bytes(routing.commitments.cpu())
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

        comm_a = commit_planes([codes_cpu, scales_cpu], ctx.key_a, config.a_hash_id)
        expected_key = a_keys(
            comm_a.digest, ctx.seed_b, ctx.key_a, config.p_a(m), routing_commitments
        ).jackpot_key
        if expected_key != hit.commitment_hash_A:
            _LOGGER.info(f"dropping winner keyed by another header on {state.layer_name}")
            return

        expert = "" if moe_info is None else f" expert={moe_info.expert_index}"
        _LOGGER.info(
            f"block candidate! layer={state.layer_name} "
            f"tile=({hit.tile_row}, {hit.tile_column}){expert} "
            f"m={m} n={state.n} k={state.k} jackpot_key={expected_key.hex()[:16]}..."
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
            b_ancestor_header=ctx.b_proof.ancestor_header,
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
    reason = "persistent hit signal could not be re-armed"
    disable_device_mining(signal.device, reason)
    for state in all_states():
        if state.weight.device == signal.device:
            state.disable_mining(reason)
    _LOGGER.opt(exception=True).error(
        f"device mining disabled after hit-signal poisoning on {signal.device}"
    )
