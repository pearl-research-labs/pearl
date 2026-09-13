"""Lottery extractor, work normalization, winning condition, and policies.

The lottery runs on the NON-PEELED first-stage product ``acc = A' @ B'.T``
(``matmul_dtype`` stored as FP32). ``XorFoldExtractor`` compresses the tile,
``is_winning`` gates the strong hash against ``target * work``.

``ReferenceJackpotPolicy`` is the consensus knob: it mirrors the plain
verifier's ``JackpotPolicy`` (``zk-pow/src/api/fp8/jackpot_policy.rs``) --
the tile-level checks on an opened tile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .hardware import DType, Hardware
from .layout import LANES as LAYOUT_LANES
from .quantization import DELTA
from .scheme import StackedRows
from .transcript import jackpot_digest

_MAX_256 = (1 << 256) - 1


def _rotl32(x: int, n: int) -> int:
    x &= 0xFFFFFFFF
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


class XorFoldExtractor:
    """Reference epilogue: fold each of the tile's 16 committed subtiles into one lane.

    The lane layout (which tile element feeds which lane, in what order) is
    committed via the operand patterns (``layout.py``);
    ``lane_indices[j]`` = lane ``j``'s flat tile indices in fold order. The
    16 x u32 output is 64 bytes -- one BLAKE3 block for the lottery hash.
    (Consensus-whitelisted in the real protocol; a placeholder here.)
    """

    LANES = LAYOUT_LANES

    def __init__(self, lane_indices: list[list[int]]):
        assert len(lane_indices) == self.LANES, f"expected {self.LANES} lanes"
        self.lane_indices = lane_indices
        self._tile_elems = sum(len(idxs) for idxs in lane_indices)

    def extract(self, c_tile: torch.Tensor) -> bytes:
        assert c_tile.dtype == DType.MATMUL.value, "extractor runs on matmul_dtype"
        # Reinterpret each matmul_dtype word as its same-width unsigned integer.
        words = c_tile.reshape(-1).contiguous().view(torch.int32).numpy().astype(np.uint32)
        assert words.size == self._tile_elems, "tile shape does not match the committed layout"
        lanes = [0] * self.LANES
        for j, idxs in enumerate(self.lane_indices):
            for i in idxs:
                lanes[j] = _rotl32((lanes[j] * 0x9E3779B1 + int(words[i])) & 0xFFFFFFFF, 13)
        return b"".join(int(x).to_bytes(4, "little") for x in lanes)  # 64 bytes


def effective_work(tile_rows: int, tile_cols: int, k: int, rank: int) -> int:
    """``tile_elems * (k - k % r)``: the v4 difficulty adjustment
    (``zk-pow/src/api/verify.rs``) — ``|IA| * |IB| * k`` with ``k`` floored
    to the rank multiple the accumulation consumes."""
    return tile_rows * tile_cols * (k - k % rank)


def is_winning(extracted: bytes, noise_seed_a: bytes, target: int, work: int) -> bytes | None:
    """The labelled jackpot digest when it meets ``target * work`` (capped)."""

    threshold = min(int(target * work), _MAX_256)
    digest = jackpot_digest(extracted, noise_seed_a)
    return digest if int.from_bytes(digest, "little") <= threshold else None


@dataclass
class Verdict:
    admissible: bool
    credit: float = 1.0


@dataclass
class ReferenceJackpotPolicy:
    """The tile-level jackpot policy, mirroring the plain verifier's
    ``JackpotPolicy`` (``zk-pow/src/api/fp8/jackpot_policy.rs``).
    Checks 3 and 4 are decided exactly in integers (identical verdicts to the
    verifier); check 1 is float-based and may differ in edge
    cases (this reference runs in double precision, the verifier in mixed
    f32/f64 -- the miner-side check is a local admissibility filter, not
    consensus-critical; the verifier's replay is the ground truth).

    The verifier replays the winning tile bit-exactly on the committed
    device's FP8 MMA semantics; on top of it this policy certifies that
    producing the tile cost ``~tile_elems * k`` fresh multiply-adds. Design
    rules: (i) prefer checks that depend only on clean
    data + public constants ("x-only"), so a committed matrix passes or fails
    identically for every noise draw and grinding gains nothing; (ii) count
    density-type violations over the WHOLE lottery tile, so honest counts
    concentrate around their mean and never fail on tail fluctuations.

    Notation. Per-row noise scales (quantized units):
    ``sigma_i = DELTA * alpha_i * l2_i``. Scaled clean entries:
    ``A_bar_iu = alpha_i * X_iu``, ``B_bar_ju = alpha_j' * X_ju``.

    The checks, per opened tile (each independent):

    1. **Entry liveness.** ``|D_X| <= eps_idle * |I_X| * k`` per side, where
       ``D_X = {(i,u): |X_bar_iu| >= tau_idle * sigma_i^X}``. Since
       ``X_bar_iu = alpha_i * X_iu`` and ``sigma_i = DELTA * alpha_i * l2_i``,
       this is ``alpha``-free: ``|X_iu| >= tau_idle * DELTA * l2_i``.
    2. **Noise floor.** ``sigma_i^X >= sigma_min`` for every row of both sides.
    3. **Tamed products.** ``|U| <= eps_tame * |I_A| * |I_B|``, where
       ``U = {(i,j): 2^(e_ij) > tau_tame * sqrt(k) * sigma_i^A * sigma_j^B}``,
       ``e_ij = floor(log2 M_ij)`` the binade of the replay magnitude
       ``M_ij = max{max_u |A'_iu * B'_ju|, max_t |c_ij,t|}``. The predicate is
       decided exactly in integers.
    4. **Unpredictable summands.** ``|S| <= eps_pred * k * |I_A| * |I_B|``,
       where ``S`` collects the summands ``(i,j,u)`` with
       ``v_iju < ulp_Device(M_ij)^2``, for
       ``v_iju = (A_bar_iu^2 + (sigma_i^A)^2)(B_bar_ju^2 + (sigma_j^B)^2) / 2^21``
       and ``ulp_Device(x) = 2^(floor(log2 x) - W)``, ``W = 25`` for Blackwell. Decided exactly in
       integers on a base-2 log lattice with steps of ``1/64``: each factor
       gets a score ``lambda ~= 64 * log2(X_bar_iu^2 + (sigma_i^X)^2)``
       (``_lambda_scores``; never above the true value, at most ``1.1``
       steps below), the threshold ``64 * log2(2^21 * ulp_Device(M_ij)^2)``
       is an exact integer, and ``(i,j,u)`` is skippable iff
       ``lambda_A + lambda_B`` falls below it. Zero cells never skip.

    There is no policy-internal ``k`` cap: the public-params envelope bounds
    ``k <= 2^16`` upstream; check 4's window ``W = 25`` is slack at any
    practical ``k``.

    The verdict is a BINARY gate (``credit`` stays ``1.0``): a ``D``-dependent
    credit would incentivize shaping data toward incoherence.
    """

    tau_idle: float = 8.0
    eps_idle: float = 0.015625  # 1/64
    sigma_min: float = 1.0
    tau_tame: float = 256.0
    eps_tame: float = 0.015625  # 1/64
    eps_pred: float = 0.0625  # 1/16

    def evaluate(
        self,
        a_stacked: StackedRows,
        b_stacked: StackedRows,
        a_clean: torch.Tensor,
        b_clean: torch.Tensor,
        hardware: Hardware,
        k: int,
    ) -> Verdict:
        assert k > 0, "k must be positive"
        assert a_clean.shape[1] == k and b_clean.shape[1] == k, "clean rows must be m/n x k"
        assert a_stacked.quant_part.shape[0] == a_clean.shape[0], (
            "one A-side noised row per clean row"
        )
        assert b_stacked.quant_part.shape[0] == b_clean.shape[0], (
            "one B-side noised row per clean row"
        )

        sigma_a = self._sigmas(a_stacked)
        sigma_b = self._sigmas(b_stacked)

        # Check 2: noise floor, sigma_i >= sigma_min on both sides.
        if bool((sigma_a < self.sigma_min).any()) or bool((sigma_b < self.sigma_min).any()):
            return Verdict(False, 0.0)

        # Check 1: entry liveness, per side.
        if not self._liveness_ok(a_clean, a_stacked.l2) or not self._liveness_ok(
            b_clean, b_stacked.l2
        ):
            return Verdict(False, 0.0)

        # Checks 3 & 4 share the replay; both run in one pass over the tile.
        if not self._tamed_and_unpredictable_ok(
            a_clean,
            b_clean,
            k,
            a_stacked,
            b_stacked,
            sigma_a,
            sigma_b,
            hardware,
        ):
            return Verdict(False, 0.0)

        return Verdict(True, 1.0)

    def _sigmas(self, stacked: StackedRows) -> torch.Tensor:
        """Per-row noise stds ``sigma_i = DELTA * alpha_i * l2_i`` (double
        precision), from the floored norms and scales recorded at
        quantization time."""
        return DELTA * stacked.alpha.to(torch.float64) * stacked.l2.to(torch.float64)

    def _liveness_ok(self, clean: torch.Tensor, l2: torch.Tensor) -> bool:
        """Check 1: the fraction of dead entries over all of the side's tile
        entries is at most ``eps_idle``. Dead iff ``|X_iu| >= tau_idle * DELTA *
        l2_i`` (the ``alpha``-free form of ``|X_bar_iu| >= tau_idle * sigma_i``)."""
        dead_bound = (self.tau_idle * DELTA) * l2.to(torch.float64)  # (rows, 1)
        dead = int((clean.abs().to(torch.float64) >= dead_bound).sum().item())
        return dead <= self.eps_idle * clean.numel()

    def _tamed_and_unpredictable_ok(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        k: int,
        a_stacked: StackedRows,
        b_stacked: StackedRows,
        sigma_a: torch.Tensor,
        sigma_b: torch.Tensor,
        hardware: Hardware,
    ) -> bool:
        """Checks 3 (tamed products) and 4 (unpredictable summands). Both
        consume the replay magnitude ``M_ij``, so they share one pass."""
        h, w = a.shape[0], b.shape[0]

        # The k/32 partial sums each cell's bit-exact replay encounters.
        partials = hardware.matmul_fp8_partials(
            a_stacked.quant_part, b_stacked.quant_part
        )  # (h, w, g)

        # Exact decodes of the noised FP8 codes (fp8 -> fp32 is exact).
        a_prime = a_stacked.quant_part.to(torch.float32)  # (h, k)
        b_prime = b_stacked.quant_part.to(torch.float32)  # (w, k)
        products = (a_prime[:, None, :] * b_prime[None, :, :]).abs()  # (h, w, k)

        # Replay magnitude M_ij = max{max_u |A'_iu * B'_ju|, max_t |c_ij,t|}.
        # Check 3 consumes only its binade floor(log2 M_ij).
        anchor_f32 = torch.maximum(
            products.amax(dim=-1), partials.abs().amax(dim=-1)
        )  # (h, w), f32

        # Check 3: tamed products, U = {(i,j): 2^(e_ij) > tau_tame *
        # sqrt(k) * si * sj} with e_ij the binade of M_ij, decided exactly in
        # integers. tau_tame^2 must be an exact integer in [0, 2^20] and
        # k <= 2^16 (the verifier's `tau_sq_k` domain); anything else is a
        # consensus misconfiguration, not a tile fault.
        tau_sq = self.tau_tame * self.tau_tame
        assert tau_sq == int(tau_sq) and 0 <= tau_sq <= 1 << 20, (
            f"tau_tame^2 must be an exact integer in [0, 2^20], got tau_tame = {self.tau_tame}"
        )
        assert k <= 1 << 16, f"k must be at most 2^16, got {k}"
        tau_sq_k = int(tau_sq) * k
        sig_a = [float(s) for s in sigma_a.reshape(-1)]
        sig_b = [float(s) for s in sigma_b.reshape(-1)]
        untamed = sum(
            _untamed_exact(float(anchor_f32[i, j]), tau_sq_k, sig_a[i], sig_b[j])
            for i in range(h)
            for j in range(w)
        )
        # Same float association as the Rust verifier: eps * (h*w), not (eps*h)*w.
        if untamed > self.eps_tame * (h * w):
            return False

        # Check 4: skippable summands
        lambda_a = _lambda_scores(a, a_stacked)  # (h, k), int64
        lambda_b = _lambda_scores(b, b_stacked)  # (w, k), int64
        # e_m = floor(log2 M_ij). M_ij is 0 or a normal f32 (a nonzero fp8
        # product is at least 2^-18), so floor(log2) is the f32 exponent field
        # minus its bias 127.
        bits = anchor_f32.view(torch.int32).to(torch.int64)
        e_m = ((bits >> 23) & 0xFF) - 127  # (h, w), meaningful where M_ij > 0
        # The defining inequality v_iju < ulp^2, through log2 and scaled by 64:
        # skip iff lambda_A + lambda_B < 64*(21 + 2*(e_m - W)). Zero cells
        # (ulp(0) = 0) never skip.
        threshold = 1344 + 128 * (e_m - hardware.fp8_window_bits)  # (h, w)
        cell_live = (anchor_f32 > 0.0)[:, :, None]  # (h, w, 1)
        sums = lambda_a[:, None, :] + lambda_b[None, :, :]  # (h, w, k)
        skippable = int(((sums < threshold[:, :, None]) & cell_live).sum().item())
        return skippable <= self.eps_pred * k * (h * w)


def _untamed_exact(m: float, tau_sq_k: int, si: float, sj: float) -> bool:
    """Check 3's untamed predicate ``ufp(m) > tau_tame * sqrt(k) * si * sj``
    (``ufp(m) = 2^floor(log2 m)``, the largest power of two not exceeding the
    replay magnitude), decided exactly by comparing squares
    (``tau_sq_k = tau_tame^2 * k``) -- identical verdict to the verifier's
    ``untamed_exact`` (``jackpot_policy.rs``). Exact because every operand is a
    dyadic rational: ``float.as_integer_ratio`` is exact, and the sigmas' f64
    values are themselves exact (``sigma = DELTA * alpha * l2`` carries at
    most 16 significand bits). ``m`` must be finite and nonnegative."""
    assert m >= 0.0, "m is a max of magnitudes"
    sin, sid = si.as_integer_ratio()
    sjn, sjd = sj.as_integer_ratio()
    if tau_sq_k == 0 or sin == 0 or sjn == 0:
        return m != 0.0  # bound = 0: untamed iff m > 0
    if (sin < 0) != (sjn < 0):
        return True  # bound < 0 <= m
    if m == 0.0:
        return False  # m = 0, bound > 0
    # 2^(2*e_m) > tau_sq_k * si^2 * sj^2, cleared of the (power-of-two)
    # denominators. floor(log2 m) = frexp exponent - 1, exact for every float.
    shift = 2 * (math.frexp(m)[1] - 1)  # = 2 * e_m
    lhs = (sid * sjd) ** 2
    rhs = tau_sq_k * (sin * sjn) ** 2
    return (lhs << shift) > rhs if shift >= 0 else lhs > (rhs << -shift)


def _bf16_e_star_m(code: int) -> tuple[int, int]:
    """Bf16 M-form split of a finite code: ``M = 128*(1 - [exp = 0]) + mantissa``,
    ``E* = max(exp, 1)``, ``value = ±M * 2^(E* - 134)``."""
    exp = (code >> 7) & 0xFF
    assert exp != 255, "non-finite bf16 code"
    mantissa = code & 0x7F
    return (1, mantissa) if exp == 0 else (exp, 128 + mantissa)


def _normalize16(sig: int, exp: int) -> tuple[int, int]:
    """``sig * 2^exp`` (``0 < sig < 2^16``) rewritten as ``n * 2^(e - 15)``
    with a normalized significand ``n`` in ``[2^15, 2^16)`` and the true
    binade ``e = floor(log2(sig * 2^exp))``. Exact: the shift only
    re-indexes the same bits."""
    width = sig.bit_length()
    return sig << (16 - width), exp + width - 1


def _lambda_scores(clean: torch.Tensor, stacked: StackedRows) -> torch.Tensor:
    """Check 4's per-element integer summand scores: a one-sided fixed-point
    log2, 64 steps per factor of 2, of
    ``S = (alpha_i * X_iu)^2 + sigma_i^2`` (``sigma_i = DELTA * alpha_i * l2_i``):

        lambda <= 64 * log2(S) < lambda + 1.1

    Every step is exact integer arithmetic on the bf16 codes:

    * each addend's root is put on a 16-bit significand (``_normalize16``;
      exact, since ``alpha*X`` and ``DELTA*alpha*l2`` carry at most 16
      significand bits): ``|alpha*X| = n_x * 2^(e_x - 15)`` and
      ``sigma = n_s * 2^(e_s - 15)``;
    * the squares are summed in the larger binade's frame, flooring the
      smaller addend's bits shifted below it (dropped whole once the binade
      gap reaches 16 — exact, the shift clears ``n_small^2 < 2^32``):
      ``V = n_big^2 + floor(n_small^2 / 4^gap)``;
    * ``V``'s top 16 bits classify the sum: ``kappa = floor(V / 2^17)`` is
      in ``[2^13, 2^16)`` and

          lambda = 128*e_big + floor(64 * log2 kappa) - 64*13,

      where ``-64*13`` removes kappa's residual ``2^13`` scale and
      ``floor(64 * log2 kappa)`` is ``bit_length(kappa^64) - 1``, exactly.

    The verifier applies the same rule; verdicts match bit for bit."""
    rows, k = clean.shape
    alpha_codes = stacked.alpha.reshape(-1).contiguous().view(torch.int16).tolist()
    l2_codes = stacked.l2.reshape(-1).contiguous().view(torch.int16).tolist()
    clean_codes = clean.contiguous().view(torch.int16).tolist()
    scores = torch.zeros((rows, k), dtype=torch.int64)
    for i in range(rows):
        ae, am = _bf16_e_star_m(alpha_codes[i] & 0xFFFF)
        assert am >= 128, "alpha is positive normal"
        le, lm = _bf16_e_star_m(l2_codes[i] & 0xFFFF)
        assert lm >= 128, "l2 is floored to a positive normal"
        # sigma = DELTA * alpha * l2 = (am * lm) * 2^(ae + le - 269).
        n_sigma, e_sigma = _normalize16(am * lm, ae + le - 269)
        row = scores[i]
        for u in range(k):
            xe, xm = _bf16_e_star_m(clean_codes[i][u] & 0xFFFF)
            # |alpha * X| = (am * xm) * 2^(ae + xe - 268).
            product = am * xm
            if product == 0:
                n_big, e_big, aligned_small_sq = n_sigma, e_sigma, 0
            else:
                n_x, e_x = _normalize16(product, ae + xe - 268)
                if e_x >= e_sigma:
                    n_big, e_big, n_small, gap = n_x, e_x, n_sigma, e_x - e_sigma
                else:
                    n_big, e_big, n_small, gap = n_sigma, e_sigma, n_x, e_sigma - e_x
                aligned_small_sq = (n_small * n_small) >> (2 * gap) if gap < 16 else 0
            sum_of_squares = n_big * n_big + aligned_small_sq
            kappa = sum_of_squares >> 17  # in [2^13, 2^16)
            # floor(64 * log2 kappa) = bit_length(kappa^64) - 1, exactly.
            row[u] = 128 * e_big + (kappa**64).bit_length() - 1 - 832
    return scores
