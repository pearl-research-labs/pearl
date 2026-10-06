"""End-to-end FP16 (A100 / ``sm_80``) tile-pipeline driver.

Chains the already-built, GA100-validated sibling kernels to reproduce
``zk-pow/src/api/fp16/verify.rs::verify_tile`` bit-for-bit on real GA100 silicon:

1. :func:`pearl_gemm.fp16_noisy_quant.noisy_quantize` rebuilds each noised FP16
   operand ``A' = Q(alpha*A + beta*(E_a@F_a^T))`` (and ``B'``), with the
   ``E@F^T`` noise on the committed :func:`pearl_gemm.fp16_gemm.fp16_gemm_a100`
   datapath. (``row_norms`` is folded into ``noisy_quantize`` here.)
2. :func:`pearl_gemm.fp16_policy.replay_and_evaluate` replays the ``h x w`` tile
   with the same bit-exact A100 accumulation and scores the unpredictable-
   accumulation-steps policy -> ``(tile, report)``.
3. The scheme-neutral lottery extractor folds the tile into the jackpot ticket:
   :func:`pearl_gemm.fp16_pipeline._layout.lane_assignment` ->
   :func:`~pearl_gemm.fp16_pipeline._layout.xor_fold_extract` ->
   :func:`~pearl_gemm.fp16_pipeline._layout.compute_jackpot_ticket`.

The module *drives* the sibling hosts (it does not reimplement their kernels) and
keeps the noised operands on-device between stages 1 and 2. Everything is gated
to ``sm_80`` via ``require_arch``. The native GEMM tile grid needs ``m % 16 == 0``
for the ``E@F^T`` matmul, so the A side (``h`` as small as 4) is zero-padded up to
a multiple of 16 and sliced back -- bit-exact, since GEMM rows are independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .._utils._arch import Arch, require_arch
from ..fp16_noisy_quant import BuiltRows16, noisy_quantize
from ..fp16_policy import PolicyReport, replay_and_evaluate
from ._layout import (
    AxisPattern,
    check_jackpot_difficulty,
    compute_jackpot_ticket,
    lane_assignment,
    xor_fold_extract,
)

_SUPPORTED_ARCHS = (Arch.SM80,)


@dataclass
class TileVerify:
    """The pipeline output (mirrors the Rust ``TileVerify``, with the extracted
    64-byte ``message`` surfaced alongside the ticket).

    ``tile`` is the ``(h, w)`` float32 replayed tile; ``report`` the policy
    report; ``message`` the 64-byte XOR-fold extract; ``ticket`` the 32-byte
    keyed-BLAKE3 jackpot digest.
    """

    tile: torch.Tensor
    report: PolicyReport
    message: bytes
    ticket: bytes
    built_a: BuiltRows16
    built_b: BuiltRows16


def _rebuild_operand(rows: torch.Tensor, e: torch.Tensor, f: torch.Tensor, r: int) -> BuiltRows16:
    """``noisy_quantize`` with the A100 GEMM's ``m % 16 == 0`` requirement handled
    by zero-padding the rows (and their ``E`` factor) up to a multiple of 16 and
    slicing the per-row outputs back. Bit-exact: GEMM rows are independent, so the
    padded zero rows never perturb the real rows' ``E@F^T`` or per-row scales."""
    num_rows = rows.shape[0]
    m16 = (num_rows + 15) & ~15
    if m16 == num_rows:
        return noisy_quantize(rows, e, f, r)
    pad_rows = torch.zeros((m16 - num_rows, rows.shape[1]), dtype=rows.dtype, device=rows.device)
    pad_e = torch.zeros((m16 - num_rows, e.shape[1]), dtype=e.dtype, device=e.device)
    built = noisy_quantize(
        torch.cat([rows, pad_rows], dim=0), torch.cat([e, pad_e], dim=0), f, r
    )
    return BuiltRows16(
        noised_part=built.noised_part[:num_rows].contiguous(),
        alpha=built.alpha[:num_rows].contiguous(),
        beta=built.beta[:num_rows].contiguous(),
        l2=built.l2[:num_rows].contiguous(),
    )


def pipeline(
    a_rows: torch.Tensor,
    b_rows: torch.Tensor,
    e_a: torch.Tensor,
    f_a: torch.Tensor,
    e_b: torch.Tensor,
    f_b: torch.Tensor,
    rows_pattern: AxisPattern,
    cols_pattern: AxisPattern,
    seed_a: bytes,
    r: int,
) -> TileVerify:
    """Run the full FP16 tile pipeline on ``sm_80``, reproducing ``verify_tile``.

    ``a_rows`` is ``(h, k)`` and ``b_rows`` is ``(w, k)`` float16 (the opened,
    committed operands); ``e_a``/``f_a`` (``(h, r)`` / ``(k, r)``) and
    ``e_b``/``f_b`` (``(w, r)`` / ``(k, r)``) are the deterministic FP16 noise
    factors. ``rows_pattern``/``cols_pattern`` are the committed extractor
    patterns (their ``tile_size()`` must equal ``h``/``w``), and ``seed_a`` is the
    32-byte jackpot key.

    Returns a :class:`TileVerify` for *any* tile (admissible or not); unlike the
    Rust ``verify_tile`` it does not raise on rejection -- call :func:`verify_tile`
    for that bail-on-reject contract. Keeps the noised operands on-device between
    the rebuild and the replay.
    """
    if a_rows.ndim != 2 or b_rows.ndim != 2:
        raise ValueError("a_rows and b_rows must be 2D")
    if a_rows.dtype != torch.float16 or b_rows.dtype != torch.float16:
        raise ValueError("a_rows and b_rows must be float16")
    h, k = a_rows.shape
    w, kb = b_rows.shape
    if kb != k:
        raise ValueError(f"a_rows and b_rows must share k ({k} != {kb})")
    if rows_pattern.tile_size() != h:
        raise ValueError(f"rows_pattern tile_size {rows_pattern.tile_size()} != h {h}")
    if cols_pattern.tile_size() != w:
        raise ValueError(f"cols_pattern tile_size {cols_pattern.tile_size()} != w {w}")
    require_arch("fp16_pipeline", a_rows.device, *_SUPPORTED_ARCHS)

    # 1. Rebuild the noised FP16 operands (noise on the committed A100 datapath).
    built_a = _rebuild_operand(a_rows.contiguous(), e_a.contiguous(), f_a.contiguous(), r)
    built_b = _rebuild_operand(b_rows.contiguous(), e_b.contiguous(), f_b.contiguous(), r)

    # 2. Replay the A100 tile on-device and score the accumulation policy.
    tile, report = replay_and_evaluate(built_a.noised_part, built_b.noised_part)

    # 3. Fold the tile into the jackpot ticket over the committed lane layout.
    tile_bits = tile.view(torch.int32).reshape(-1).cpu().numpy().view("uint32")
    lanes = lane_assignment(rows_pattern, cols_pattern)
    message = xor_fold_extract(tile_bits, lanes)
    ticket = compute_jackpot_ticket(seed_a, message)
    return TileVerify(
        tile=tile, report=report, message=message, ticket=ticket, built_a=built_a, built_b=built_b
    )


def verify_tile(
    a_rows: torch.Tensor,
    b_rows: torch.Tensor,
    e_a: torch.Tensor,
    f_a: torch.Tensor,
    e_b: torch.Tensor,
    f_b: torch.Tensor,
    rows_pattern: AxisPattern,
    cols_pattern: AxisPattern,
    seed_a: bytes,
    r: int,
) -> TileVerify:
    """:func:`pipeline` with the Rust ``verify_tile`` bail-on-reject contract:
    raises :class:`ValueError` when the policy rejects the tile, otherwise returns
    the :class:`TileVerify`."""
    v = pipeline(a_rows, b_rows, e_a, f_a, e_b, f_b, rows_pattern, cols_pattern, seed_a, r)
    if not v.report.accept:
        raise ValueError(
            f"the jackpot is not admissible (f_bp={v.report.f_bp:.4f}, rho={v.report.rho:.4f})"
        )
    return v


def verify_tile_proof(
    a_rows: torch.Tensor,
    b_rows: torch.Tensor,
    e_a: torch.Tensor,
    f_a: torch.Tensor,
    e_b: torch.Tensor,
    f_b: torch.Tensor,
    rows_pattern: AxisPattern,
    cols_pattern: AxisPattern,
    seed_a: bytes,
    r: int,
    nbits: int,
) -> TileVerify:
    """:func:`verify_tile` plus the difficulty-target check (Rust
    ``verify_tile_proof``): raises if the tile is inadmissible or the jackpot does
    not clear ``nbits`` scaled by ``h*w*k``."""
    v = verify_tile(a_rows, b_rows, e_a, f_a, e_b, f_b, rows_pattern, cols_pattern, seed_a, r)
    h, k = a_rows.shape
    w = b_rows.shape[0]
    if not check_jackpot_difficulty(v.ticket, nbits, int(h), int(w), int(k)):
        raise ValueError("Jackpot condition not satisfied: hash does not meet difficulty target")
    return v
