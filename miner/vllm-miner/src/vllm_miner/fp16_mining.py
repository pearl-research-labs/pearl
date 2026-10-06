"""Per-forward FP16 (A100) search + winner submission.

The FP16 analogue of the FP8 ``pipeline.run_mining_forward`` + ``winners``:
``pearl_gemm.fp16_miner.search_block`` already fuses the whole per-attempt chain
(commit -> seed chain -> noise -> noisy-quant -> full-matrix lottery search ->
hit decode + host policy), so this module only has to:

1. admit the launch through the shared ``AsyncLoopManager`` gate (retain-winner,
   job freshness) exactly as FP8 does;
2. build the FP16 job statement (``Fp16JobParams``) from the committed patterns
   and the proposed header's ``nbits``, run ``search_block`` under the
   process-wide producer/attempt gates;
3. on an admissible ``WinningTile``, assemble the committed-operand opening
   (:class:`miner_base.fp16_block_submission.Fp16OpenedBlock`) at the winning
   *contiguous* tile and hand it to the manager's bounded submission executor
   (``handle_submit_block``) -- the same executor the FP8 winner uses; the
   executor dispatches ``submit_fp16_block`` by the job's cert version.

Unlike FP8 this is fully synchronous: ``search_block`` returns the decoded
winner on the calling stream, so there is no persistent hit signal, no
completion lease, and no event-gated winner callback. The layer's serving
output is a plain BF16 linear produced by the caller; the search is a side
effect that never feeds serving.

Everything here needs ``torch`` + ``pearl_gemm`` (the A100 kernels) and an
sm_80 device, so it is import-time light and GPU-work lazy: nothing runs off an
A100 with a V5 job.
"""

from __future__ import annotations

import struct

import torch
from miner_base.fp16_block_submission import Fp16OpenedBlock
from miner_utils import get_logger
from pearl_gateway.comm.dataclasses import MiningJob

from .capture import gpu_mining_producer, mining_launches_suspended
from .fp16_layer import FP16_TILE_H, Fp16LayerState
from .health import mining_attempt
from .mining_state import get_async_manager

_LOGGER = get_logger(__name__)

# Winner handoff must not wait forever behind saturated proof capacity; matches
# the FP8 winner handoff budget (winners.py::_WINNER_HANDOFF_TIMEOUT_S).
_WINNER_HANDOFF_TIMEOUT_S = 30.0

# FP16 difficulty comes from the proposed header's compact ``nbits`` (last u32
# of the 76-byte header), exactly as the verifier takes its default difficulty.
_HEADER_NBITS_OFFSET = 72


def _header_nbits(header_bytes: bytes) -> int:
    if len(header_bytes) != 76:
        raise ValueError(f"proposed header must be 76 bytes, got {len(header_bytes)}")
    return struct.unpack_from("<I", header_bytes, _HEADER_NBITS_OFFSET)[0]


def _search_patterns():
    """The committed A100 patterns as ``pearl_gemm.fp16_miner`` expects them.

    ``search_block`` keys on ``pearl_gemm.fp16_pipeline.AxisPattern`` (FOLD=1,
    BLAKE=2), a distinct type from ``miner_base.layout.AxisPattern`` used by the
    certificate opening; both describe ``P_A=[(4,Blake)]`` / ``P_B=[(4,Blake),
    (16,Fold)]``."""
    from pearl_gemm.fp16_pipeline import AxisPattern as GemmAxisPattern

    _FOLD, _BLAKE = 1, 2
    rows = GemmAxisPattern.new(((4, _BLAKE),))
    cols = GemmAxisPattern.new(((4, _BLAKE), (16, _FOLD)))
    return rows, cols


def _job_params(state: Fp16LayerState, m: int, nbits: int):
    """The FP16 public statement for this layer's committed tile and difficulty."""
    from pearl_gemm.fp16_miner import Fp16JobParams, Fp16OperandParams

    rows, cols = _search_patterns()
    return Fp16JobParams(
        k=state.k,
        a=Fp16OperandParams(num_rows=m, pattern=rows),
        b=Fp16OperandParams(num_rows=state.n, pattern=cols),
        nbits=nbits,
        r=state.r,
    )


def _to_committed_activation(x2d: torch.Tensor, device: torch.device) -> torch.Tensor:
    """The FP16 activation operand ``A`` (m x k), its rows truncated to a whole
    number of h=4 row tiles.

    The committed A is exactly what ``search_block`` commits and what the
    certificate opens, so any rows beyond the last full tile (never searchable)
    are dropped rather than zero-padded: a padded row could latch a winning tile
    on zeros that the opened real activation would not reproduce."""
    usable = x2d.shape[0] - (x2d.shape[0] % FP16_TILE_H)
    if usable <= 0:
        return x2d[:0]
    a = x2d[:usable]
    if a.dtype == torch.float16:
        return a.contiguous()
    return a.to(torch.float16).contiguous()


def run_fp16_mining_forward(state: Fp16LayerState, job: MiningJob, x2d: torch.Tensor) -> None:
    """Run the FP16 search for one forward and submit an admissible winner.

    ``x2d`` is the ``(m_tokens, k)`` serving activation (any float dtype viewable
    as FP16); only whole h=4 row tiles are mined. Never raises into serving: a
    deterministic kernel failure disables the layer, an OOM is swallowed (the
    shared device cooldown owns backoff), and a declined admission is a no-op.
    """
    if not state.mineable or mining_launches_suspended():
        return
    if torch.cuda.is_current_stream_capturing():
        # Per-launch host machinery is illegal inside a CUDA-graph capture.
        return
    try:
        _attempt_fp16_launch(state, job, x2d)
    except torch.cuda.OutOfMemoryError:
        _LOGGER.warning(f"FP16 mining OOM on {state.weight.device}; serving continues unmined")
    except Exception as exc:
        state.disable_mining(f"deterministic FP16 forward failure ({type(exc).__name__})")


def _attempt_fp16_launch(state: Fp16LayerState, job: MiningJob, x2d: torch.Tensor) -> None:
    from pearl_gemm.fp16_miner import search_block

    manager = get_async_manager()
    with manager.mining_launch_admission(job) as decision:
        # retain_winner is off when no proof could be submitted (no-gateway,
        # skip-submission, a non-V5 job, or closed admission); running the
        # search then only burns an A100 forward, so decline like FP8 does.
        if not decision.launch or not decision.retain_winner:
            return

    device = state.weight.device
    a16 = _to_committed_activation(x2d, device)
    if a16.shape[0] == 0:
        return
    header_bytes = bytes(job.incomplete_header_bytes)
    params = _job_params(state, a16.shape[0], _header_nbits(header_bytes))

    with gpu_mining_producer() as admitted:
        if not admitted:
            return
        with mining_attempt(device) as attempt:
            if not attempt:
                return
            winner = search_block(header_bytes, a16, state.weight, params, device)
            attempt.mark_success(winner.found)

    if not winner.found:
        return
    if winner.report is not None and not winner.report.accept:
        _LOGGER.info(
            f"FP16 lottery winner failed host policy on {state.layer_name} "
            f"(tile={winner.tile_row},{winner.tile_col}); filtered before submission"
        )
        return

    _submit_fp16_winner(state, job, a16, winner)


def _submit_fp16_winner(
    state: Fp16LayerState, job: MiningJob, a16: torch.Tensor, winner
) -> None:
    """Assemble the committed-operand opening for the winning tile and hand it to
    the manager's bounded submission executor.

    The committed A is exactly the ``a16`` the search ran on, and B is the
    layer's stable host weight copy; the winning tile is contiguous, so the
    grid coordinates ``(tile_row, tile_col)`` name the opened global rows
    directly (``create_fp16_proof`` resolves ``base + tile_offsets``)."""
    manager = get_async_manager()
    if job != manager.get_mining_job():
        _LOGGER.info(f"dropping FP16 winner for replaced job on {state.layer_name}")
        return

    opened = Fp16OpenedBlock(
        a=a16.detach().to("cpu"),
        b=state.weight_cpu,
        k=state.k,
        m=a16.shape[0],
        n=state.n,
        rows_pattern=state.rows_pattern,
        cols_pattern=state.cols_pattern,
        tile_row=winner.tile_row,
        tile_col=winner.tile_col,
        r=state.r,
    )
    _LOGGER.info(
        f"FP16 block candidate! layer={state.layer_name} "
        f"tile=({winner.tile_row}, {winner.tile_col}) m={opened.m} n={state.n} k={state.k}"
    )
    submitted = manager.handle_submit_block(opened, job, timeout=_WINNER_HANDOFF_TIMEOUT_S)
    if submitted:
        _LOGGER.info(f"FP16 winner handed off for proof construction ({state.layer_name})")
    else:
        _LOGGER.error(f"FP16 winner was not queued for submission ({state.layer_name})")
