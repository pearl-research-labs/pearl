"""Persistent hit-signal decoding and proof handoff."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from miner_base.async_loop_manager import AsyncLoopManager
from miner_base.block_submission import (
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
) -> JobContext | None:
    """Return the live context described by a persistent record, if any.

    The seedB and target words below were stamped by the kernel from the
    device buffers at launch time, so they pin the hit to the job the layer
    was prepared for when it launched.
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
        (hit.n, hit.k) != (state.n, state.k)
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
    if hit.target != threshold_bytes_for(ctx.job, state.k, state.n):
        _LOGGER.info(f"dropping stale target for {state.layer_name}")
        return None
    if hit.codes is None or hit.scales is None:
        _LOGGER.warning(f"dropping payload-less hit for {state.layer_name}")
        return None
    return ctx


@dataclass
class WinnerCheckCallback:
    """Consume at most one record after an admitted launch event completes."""

    signal: "HitSignal"
    event: torch.cuda.Event
    leases: "WinnerCheckLeases"
    manager: AsyncLoopManager

    def __call__(self) -> None:
        try:
            # One signal-level critical section owns the reusable payload and
            # re-arms before slow CPU validation. Concurrent manager/fallback
            # callbacks cannot both consume one record or reset a newer claim.
            hit = self.signal.take_owned_hit()
            if hit is None:
                return
            if not hit.valid:
                _LOGGER.warning("dropping malformed persistent hit record (valid=False)")
                return
            self._handle_hit(hit)
        except Exception as exc:
            from pearl_gemm import HitSignalPoisonedError

            if isinstance(exc, HitSignalPoisonedError):
                reason = "persistent hit signal could not be re-armed"
                disable_device_mining(self.signal.device, reason)
                for state in all_states():
                    if state.weight.device == self.signal.device:
                        state.disable_mining(reason)
                _LOGGER.opt(exception=True).error(
                    f"device mining disabled after hit-signal poisoning on {self.signal.device}"
                )
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

        ctx = _matching_context(hit, state, self.manager)
        if ctx is None:
            return

        config = ctx.config
        a_rows = tile_indices(config.rows_pattern, hit.tile_row)
        b_rows = tile_indices(config.cols_pattern, hit.tile_column)
        codes_cpu = hit.codes
        scales_cpu = hit.scales
        pow_key = hit.commitment_hash_A  # the launch's jackpot key
        _LOGGER.info(
            f"block candidate! layer={state.layer_name} "
            f"tile=({hit.tile_row}, {hit.tile_column}) "
            f"m={hit.m} n={state.n} k={state.k} jackpot_key={pow_key.hex()[:16]}..."
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
