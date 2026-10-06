"""Integer-exact Python port of the FP16 "unpredictable accumulation steps"
jackpot policy (``zk-pow/src/api/fp16/policy.rs``).

Extends the trusted A100 accumulation oracle (:mod:`tests.helpers.a100_fp16_reference`)
with the per-group census ``a100_dot`` records when ``census = Some`` and the
``evaluate`` reduction over a tile. It is the oracle the on-GPU sm_80
``fp16_policy`` kernel is validated against.

Per group of ``GROUP = 8`` products the census records a ``PolicyStep``:

* ``breakpoint`` -- the accumulator alignment OR the final FP32 round-toward-zero
  discarded a nonzero bit (``acc_truncated || rz_dropped``);
* ``products_truncated`` -- how many of the group's products lost nonzero bits in
  their right-shift alignment.

Empty no-op groups push a default step (``breakpoint = False``,
``products_truncated = 0``). ``evaluate`` then, per cell, counts breakpoint steps
(``n_bp``), maximal runs of non-breakpoint steps (``n_runs``; empty no-ops extend
the surrounding run), and sums ``products_truncated`` over non-breakpoint steps
(``n_pt``), and aggregates ``f_bp`` and ``rho`` in f64.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from tests.helpers.a100_fp16_reference import (
    GROUP,
    W,
    _NEG,
    _acc_parts,
    _rz_to_f32_bits,
    _shift,
    decompose_fp16,
)

NOISE_RANK = 32
MIN_FBP = 0.30
MIN_RHO = 1.2


@dataclass
class PolicyStep:
    breakpoint: bool
    products_truncated: int


@dataclass
class PolicyReport:
    f_bp: float
    rho: float
    accept: bool
    breakpoints: int
    numerator: int


def a100_dot_census(a: list[int], b: list[int], c_bits: int = 0) -> tuple[int, list[PolicyStep]]:
    """A100 FP16 dot product with per-group census.

    Returns ``(d_bits, steps)``: the FP32 result u32 bit pattern and one
    :class:`PolicyStep` per group of 8 (empty groups included). Mirrors
    ``accumulate::a100_dot`` with ``census = Some`` exactly.
    """
    assert len(a) == len(b), "operand length mismatch"
    cur = c_bits & 0xFFFFFFFF
    k = len(a)
    steps: list[PolicyStep] = []
    g0 = 0
    while g0 < k:
        g1 = min(g0 + GROUP, k)
        cs, cm, cel, culp = _acc_parts(cur)
        eta = cel
        for u in range(g0, g1):
            _, ma, ea = decompose_fp16(a[u])
            _, mb, eb = decompose_fp16(b[u])
            if ma != 0 and mb != 0:
                eta = max(eta, ea + eb)
        if eta == _NEG:
            steps.append(PolicyStep(breakpoint=False, products_truncated=0))
            g0 = g1
            continue
        unit = eta - W
        total = 0
        products_truncated = 0
        for u in range(g0, g1):
            sa, ma, ea = decompose_fp16(a[u])
            sb, mb, eb = decompose_fp16(b[u])
            if ma == 0 or mb == 0:
                continue
            prod = ma * mb
            sh = (ea + eb) - 20 - unit
            aligned = _shift(prod, sh)
            if sh < 0 and (prod & ((1 << min(-sh, 126)) - 1)) != 0:
                products_truncated += 1
            total += (sa * sb) * aligned
        csh = culp - unit
        acc_aligned = _shift(cm, csh)
        acc_truncated = csh < 0 and cm != 0 and (cm & ((1 << min(-csh, 126)) - 1)) != 0
        total += cs * acc_aligned
        new_bits = _rz_to_f32_bits(total, unit)
        # rz dropped a nonzero bit iff the result differs from the exact sum.
        new_val = struct.unpack("<f", struct.pack("<I", new_bits))[0]
        rz_dropped = float(new_val) != float(total) * (2.0**unit)
        steps.append(PolicyStep(breakpoint=bool(acc_truncated or rz_dropped),
                                products_truncated=products_truncated))
        cur = new_bits
        g0 = g1
    return cur & 0xFFFFFFFF, steps


def cell_counts(steps: list[PolicyStep]) -> tuple[int, int, int]:
    """``(n_bp, n_runs, n_pt)`` for one cell, mirroring ``evaluate``'s inner loop."""
    n_bp = n_pt = n_runs = 0
    in_run = False
    for s in steps:
        if s.breakpoint:
            n_bp += 1
            in_run = False
        else:
            n_pt += s.products_truncated
            if not in_run:
                n_runs += 1
                in_run = True
    return n_bp, n_runs, n_pt


def evaluate(percell: list[tuple[int, int, int]], cells: int, k: int) -> PolicyReport:
    """Fold per-cell ``(n_bp, n_runs, n_pt)`` into the tile report (``evaluate``)."""
    assert cells > 0 and k > 0, "empty tile"
    steps_per_cell = -(-k // GROUP)  # ceil
    breakpoints = sum(nb for nb, _, _ in percell)
    numerator = sum(GROUP * nb + NOISE_RANK * nr + npt for nb, nr, npt in percell)
    f_bp = breakpoints / float(cells * steps_per_cell)
    rho = numerator / float(cells * k)
    return PolicyReport(
        f_bp=f_bp,
        rho=rho,
        accept=f_bp >= MIN_FBP and rho >= MIN_RHO,
        breakpoints=breakpoints,
        numerator=numerator,
    )


def replay_and_evaluate(
    a: np.ndarray, b: np.ndarray, m: int, n: int, k: int
) -> tuple[np.ndarray, np.ndarray, PolicyReport]:
    """Device-order replay + policy evaluation of the ``m x n`` tile ``a @ b.T``.

    ``a`` is ``(m, k)`` and ``b`` is ``(n, k)`` FP16 bit patterns (``uint16``).
    Returns ``(tile_bits, percell, report)``: the ``(m, n)`` u32 tile bits, the
    ``(m*n, 3)`` int per-cell ``(n_bp, n_runs, n_pt)`` census, and the report.
    """
    a = np.asarray(a, dtype=np.uint16).reshape(m, k)
    b = np.asarray(b, dtype=np.uint16).reshape(n, k)
    tile = np.zeros((m, n), dtype=np.uint32)
    percell: list[tuple[int, int, int]] = []
    for i in range(m):
        ai = a[i].tolist()
        for j in range(n):
            d_bits, steps = a100_dot_census(ai, b[j].tolist(), 0)
            tile[i, j] = d_bits
            percell.append(cell_counts(steps))
    report = evaluate(percell, m * n, k)
    return tile, np.asarray(percell, dtype=np.int64), report
