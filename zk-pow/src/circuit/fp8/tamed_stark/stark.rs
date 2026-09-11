//! Per-cell tamed certificates and the tile's untamed and skip budgets.
//!
//! # Imported values and the tamed condition
//!
//! Matmul supplies each cell's skip count and an upper bound on its replay-magnitude
//! exponent. For nonzero magnitude `M`, the true exponent is `floor(log2(M))`.
//! Write the claimed bound as `e_M = cell_magnitude_exponent - 139`; zero cells
//! use the sentinel `cell_magnitude_exponent = 0`.
//!
//! Scale supplies the row and column noise scales in the form:
//!
//! ```text
//! sigma_A = SIG_A * 2^(E_A - 2048)
//! sigma_B = SIG_B * 2^(E_B - 2048)
//! 0 <= SIG_A, SIG_B < 2^16
//! ```
//!
//! A nonzero cell claimed tamed must prove the squared policy condition:
//!
//! ```text
//! 2^(2*e_M) <= tau_tame^2 * k * (sigma_A*sigma_B)^2
//! ```
//!
//! Equality is tamed. The circuit fixes `tau_tame = 256`, so `tau_tame^2 = 2^16`
//! can be moved into the exponent on the left. With `E_M = cell_magnitude_exponent`:
//!
//! ```text
//! FRAME_GAP_OFFSET = 2*2048 - 139 - 16/2 = 3949
//! D = 2*(E_M - E_A - E_B + FRAME_GAP_OFFSET)
//! Y = k*(SIG_A*SIG_B)^2
//!
//! certificate: 2^D <= Y
//! ```
//!
//! Since `k <= 2^16`, the integer bound is `Y < 2^16*(2^32)^2 = 2^80`.
//! The native policy helper retains `tau_tame^2` in its integer right-hand side
//! to support custom thresholds; its larger bound applies to that different expression.
//!
//! # Bounded integer certificate (J2–J5)
//!
//! XFPOW2 supplies `2^A`, where `A = min(max(D, 0), 80)`. Clamping the exponent
//! preserves the comparison against the nonnegative integer `Y`:
//!
//! - If `D <= 0`, every positive `Y` passes both comparisons.
//! - If `0 < D < 80`, the exponent is unchanged.
//! - If `D >= 80`, every `Y < 2^80` fails both comparisons.
//! - `Y = 0` fails for every exponent.
//!
//! The certificate splits `SIG_A*SIG_B` into 16-bit limbs and computes `Y` by
//! schoolbook multiplication. A borrow chain then proves `2^A <= Y`.
//! Range checks keep each multiplication-position sum below `2^34`, so the
//! field equations also hold over the integers.
//!
//! # Flags and tile budgets (J1, J6–J7)
//!
//! J1 makes the untamed and zero-cell flags boolean and mutually exclusive.
//! A zero-cell claim requires the imported magnitude exponent to be zero.
//! Every other cell either supplies a tamed certificate or increments the untamed count.
//!
//! J6 and J7 accumulate the counts and enforce the public budgets on the last row:
//!
//! ```text
//! untamed_total <= floor(eps_tame * h * w)
//! skip_total   <= floor(eps_pred * k * h * w)
//! ```
//!
//! A prover may overcount untamed cells or skips, but cannot undercount them.
//! Larger claimed magnitude bounds only make acceptance harder.
//!
//! Padding imports no data and adds zero to both counts. Its certificate limbs
//! are zero, and the power lookup is disabled. [`super::ctl`] declares the import
//! channels, limb range checks and final budget checks.

use core::borrow::{Borrow, BorrowMut};
use core::marker::PhantomData;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use starky::evaluation_frame::{StarkEvaluationFrame, StarkFrame};
use starky::stark::Stark;

use super::columns::{
    COMPARISON_DIGITS, NUM_TAMED_COLUMNS, NUM_TAMED_KNOWN_COLUMNS, NUM_TAMED_PUBLIC_INPUTS, SKIP_LIMIT_PUBLIC_INPUT,
    TAME_K_PUBLIC_INPUT, TAME_LIMIT_PUBLIC_INPUT, TamedColumnsView,
};
use crate::api::fp8::jackpot_policy::JackpotPolicy;
use crate::circuit::fp8::luts::{XFPOW2_CAP, XFPOW2_LIMBS};
use crate::circuit::fp8::matmul_b200_stark::stark::BINADE_BIAS;
use crate::circuit::fp8::unpredictability::budget;
use crate::circuit::utils::evaluator::Evaluator;
use crate::circuit::utils::native_evaluator::NativeEvaluator;
use crate::circuit::utils::symbolic_evaluator::SymbolicEvaluator;

const LIMB: u64 = 1 << 16;
const MASK: u64 = LIMB - 1;

/// Fixed log2(tau_tame^2), absorbed into the comparison exponent.
/// `TamedProgram::k` checks that the policy matches this constant.
pub(crate) const TAU_TAME_SQ_LOG2: i64 = 16;

/// Bias/tau adjustment for half the comparison exponent:
/// `D/2 = cell_magnitude_exponent - sigma_a_exp - sigma_b_exp + FRAME_GAP_OFFSET`.
/// The XFPOW2 lookup adds its own key zero point.
pub(crate) const FRAME_GAP_OFFSET: i64 = 2 * 2048 - (BINADE_BIAS as i64) - TAU_TAME_SQ_LOG2 / 2;

// Program and trace generation

/// The public geometry of one tamed-products check: `h * w` cells over inner dimension `k`.
/// The structural columns (cell id, the two sigma group keys, the last-row and pad flags) are
/// recomputable from this program alone; the binade and sigma tuples are witness data
/// (bound to Matmul's and Scale's committed traces by the import channels).
#[derive(Clone, Debug)]
pub struct TamedProgram {
    /// Tile rows (rows of A).
    pub h: usize,
    /// Tile columns (rows of B).
    pub w: usize,
    /// Inner (reduction) dimension.
    pub k: usize,
}

/// One imported sigma frame `SIG * 2^(EXP - 2048)` — the sigma channel tuple values.
pub type Frame = (u64, u64);

impl TamedProgram {
    /// Trace height: one row per cell, padded to the next power of two.
    pub fn num_rows(&self) -> usize {
        (self.h * self.w).next_power_of_two()
    }

    /// Checks k <= 2^16 and the fixed tau assumption used by the limb bounds and exponent shift.
    fn k(&self) -> u64 {
        let policy = JackpotPolicy::default();
        assert!(
            policy.tau_tame * policy.tau_tame == (1u64 << TAU_TAME_SQ_LOG2) as f64,
            "tau_tame^2 must match the fixed XFPOW2 exponent offset"
        );
        assert!(self.k <= 1 << 16, "k must be at most 2^16");
        self.k as u64
    }

    /// The untamed-cell allowance `floor(eps_tame * h * w)`, in f64 exactly like the
    /// plaintext gate `untamed as f64 <= eps_tame * cells as f64` (an integer count passes
    /// that gate iff it is at most this floor). `< 2^16` for every sanctioned geometry, so
    /// the J6 gate is a single RC16.
    fn tame_limit(&self) -> u64 {
        let limit = (JackpotPolicy::default().eps_tame * (self.h * self.w) as f64).floor() as u64;
        assert!(limit < 1 << 16, "TAME_LIMIT exceeds the single-limb RC16 gate");
        limit
    }

    /// The skippable-summand budget `floor(eps_pred * k * h * w)`
    /// (`circuit::fp8::unpredictability::budget`, mirroring the plaintext f64 gate exactly).
    /// `< 2^32` for every sanctioned geometry, so the J7 gate is a two-limb split.
    fn skip_limit(&self) -> u64 {
        let limit = budget(self.k, self.h, self.w);
        assert!(limit < 1 << 32, "SKIP_LIMIT exceeds the two-limb gate");
        limit
    }

    /// The expected public inputs of a proof under this program — pure functions of the
    /// program: the verifier assembles its expectation from the statement alone.
    pub fn public_inputs<F: RichField>(&self) -> [F; NUM_TAMED_PUBLIC_INPUTS] {
        let mut pis = [F::ZERO; NUM_TAMED_PUBLIC_INPUTS];
        pis[TAME_K_PUBLIC_INPUT] = F::from_canonical_u64(self.k());
        pis[TAME_LIMIT_PUBLIC_INPUT] = F::from_canonical_u64(self.tame_limit());
        pis[SKIP_LIMIT_PUBLIC_INPUT] = F::from_canonical_u64(self.skip_limit());
        pis
    }

    /// Generates the trace and public inputs.
    /// - `cells`: row-major `(magnitude_exponent, skip_count)` tuples from Matmul;
    /// - `sigma_a`, `sigma_b`: row/column `(significand, exponent)` frames from Scale,
    ///   with exponents biased by 2048.
    /// Panics if the untamed or skip count exceeds its budget.
    pub fn generate_trace<F: RichField>(
        &self,
        cells: &[(u64, u64)],
        sigma_a: &[Frame],
        sigma_b: &[Frame],
    ) -> (Vec<[F; NUM_TAMED_COLUMNS]>, [F; NUM_TAMED_PUBLIC_INPUTS]) {
        let (h, w) = (self.h, self.w);
        assert_eq!(cells.len(), h * w, "one (E_CELL, CELL_SKIPS) tuple per cell");
        assert_eq!(sigma_a.len(), h, "one sigma frame per tile row");
        assert_eq!(sigma_b.len(), w, "one sigma frame per tile column");
        let k = self.k();
        let num_rows = self.num_rows();

        let mut rows: Vec<[F; NUM_TAMED_COLUMNS]> = Vec::with_capacity(num_rows);
        let mut running = 0u64;
        let mut running_skips = 0u64;
        for (cell, &(cell_magnitude_exponent, skips)) in cells.iter().enumerate() {
            let (i, j) = (cell / w, cell % w);
            let (sa, sb) = (sigma_a[i], sigma_b[j]);
            running_skips += skips;
            let mut v = TamedColumnsView {
                cell_id: F::from_canonical_usize(cell),
                a_group_key: F::from_canonical_usize((i + 1) * self.k - 1),
                b_group_key: F::from_canonical_usize(h * self.k + (j + 1) * self.k - 1),
                is_last_row: F::from_bool(cell == num_rows - 1),
                cell_magnitude_exponent: F::from_canonical_u64(cell_magnitude_exponent),
                sigma_a_significand: F::from_canonical_u64(sa.0),
                sigma_a_exp: F::from_canonical_u64(sa.1),
                sigma_b_significand: F::from_canonical_u64(sb.0),
                sigma_b_exp: F::from_canonical_u64(sb.1),
                cell_skips: F::from_canonical_u64(skips),
                running_skips: F::from_canonical_u64(running_skips),
                ..TamedColumnsView::default()
            };
            if untamed(k, cell_magnitude_exponent, sa, sb) {
                running += 1;
                v.untamed = F::ONE;
            } else if cell_magnitude_exponent == 0 {
                v.cell_is_zero = F::ONE; // M = 0: tamed by definition, certificate skipped.
            } else {
                fill_certificate(&mut v, k, cell_magnitude_exponent, sa, sb);
            }
            v.running_untamed = F::from_canonical_u64(running);
            rows.push(v.into());
        }
        assert!(
            running <= self.tame_limit(),
            "policy-rejected witness: untamed cells exceed the eps_tame allowance"
        );
        assert!(
            running_skips <= self.skip_limit(),
            "policy-rejected witness: skippable summands exceed the eps_pred budget"
        );

        // Pad rows: all-zero except the schedule columns — the active certificate gates hold
        // vacuously (0 <= 0) and the frozen running counts carry to the J6/J7 gate row.
        for t in h * w..num_rows {
            let v = TamedColumnsView {
                cell_id: F::from_canonical_usize(t),
                is_last_row: F::from_bool(t == num_rows - 1),
                is_pad: F::ONE,
                running_untamed: F::from_canonical_u64(running),
                running_skips: F::from_canonical_u64(running_skips),
                ..TamedColumnsView::default()
            };
            rows.push(v.into());
        }

        // The J7 budget-gate witness on the last row: the two-limb split of the slack.
        let slack = self.skip_limit() - running_skips;
        let last: &mut TamedColumnsView<F> = rows.last_mut().expect("at least one row").borrow_mut();
        last.skip_gate_slack_lo = F::from_canonical_u64(slack & MASK);
        last.skip_gate_slack_hi = F::from_canonical_u64(slack >> 16);

        (rows, self.public_inputs())
    }

    /// Recomputes the leading schedule columns in trace order from public geometry.
    pub fn known_values<F: RichField>(&self) -> Vec<PolynomialValues<F>> {
        let num_rows = self.num_rows();
        let live = self.h * self.w;
        let mut cols = (0..NUM_TAMED_KNOWN_COLUMNS)
            .map(|_| Vec::with_capacity(num_rows))
            .collect::<Vec<_>>();
        for r in 0..num_rows {
            let (i, j) = (r / self.w, r % self.w);
            cols[0].push(F::from_canonical_usize(r));
            cols[1].push(F::from_canonical_usize(if r < live { (i + 1) * self.k - 1 } else { 0 }));
            cols[2].push(F::from_canonical_usize(if r < live {
                self.h * self.k + (j + 1) * self.k - 1
            } else {
                0
            }));
            cols[3].push(F::from_bool(r == num_rows - 1));
            cols[4].push(F::from_bool(r >= live));
        }
        cols.into_iter().map(PolynomialValues::new).collect()
    }
}

/// Exact comparison `2^D > k*(SIG_A*SIG_B)^2`, with tau folded into D.
/// The right side fits u128; a bit-length test compares it against the power of two.
fn untamed(k: u64, cell_magnitude_exponent: u64, sa: Frame, sb: Frame) -> bool {
    let pp = sa.0 * sb.0;
    if k == 0 || pp == 0 {
        return cell_magnitude_exponent != 0; // bound = 0: untamed iff M > 0
    }
    if cell_magnitude_exponent == 0 {
        return false; // M = 0, bound > 0
    }
    let y = u128::from(k) * u128::from(pp) * u128::from(pp);
    // For Y > 0, `2^D > Y` iff D >= bit_length(Y).
    let d = 2 * (cell_magnitude_exponent as i64 - sa.1 as i64 - sb.1 as i64 + FRAME_GAP_OFFSET);
    d >= i64::from(128 - y.leading_zeros())
}

/// Fills the tamed-certificate witness block (J2-J5) of one nonzero tamed row: the exact
/// schoolbook digits of `Y = K * PP^2`, the one-hot limbs of `2^A`, and the borrow chain
/// proving `2^A <= Y`. Panics if the cell is in fact untamed (the caller decides the
/// verdict first).
fn fill_certificate<F: RichField>(v: &mut TamedColumnsView<F>, k: u64, cell_magnitude_exponent: u64, sa: Frame, sb: Frame) {
    let f = F::from_canonical_u64;

    // J2: the sigma product's limb split.
    let pp = sa.0 * sb.0;
    let (pp_lo, pp_hi) = (pp & MASK, pp >> 16);
    v.sigma_product_limbs = [f(pp_lo), f(pp_hi)];

    // J3: W = K * PP (digits e), then Y = W * PP (digits y).
    let limb2 = |x: u64| [x & MASK, x >> 16];
    let (kl, kh) = (limb2(k * pp_lo), limb2(k * pp_hi));
    v.k_sigma_product_lo = kl.map(f);
    v.k_sigma_product_hi = kh.map(f);
    let s = kl[1] + kh[0];
    let (w1, carry) = (s & MASK, s >> 16);
    v.k_sigma_product_middle_limbs[0] = f(w1);
    v.k_sigma_product_carries[0] = f(carry);
    let e = [kl[0], w1, kh[1] + carry];

    let mut y = [0u64; COMPARISON_DIGITS];
    let mut carry = 0u64;
    for pos in 0..COMPARISON_DIGITS - 2 {
        let lo = if pos < 3 { e[pos] * pp_lo } else { 0 };
        let hi = if pos >= 1 { e[pos - 1] * pp_hi } else { 0 };
        let s = lo + hi + carry;
        (y[pos], carry) = (s & MASK, s >> 16);
        match pos {
            0 => v.bound_carry_1 = f(carry),
            1 | 2 => {
                v.bound_carries_lo[pos - 1] = f(carry & MASK);
                v.bound_carries_bit[pos - 1] = f(carry >> 16);
            }
            _ => {}
        }
    }
    y[COMPARISON_DIGITS - 2] = carry;
    for (limb, &digit) in v.bound_limbs.iter_mut().zip(&y) {
        *limb = f(digit);
    }
    v.bound_top = f(y[COMPARISON_DIGITS - 2]);

    // J4: the saturated one-hot power 2^A.
    let d = 2 * (cell_magnitude_exponent as i64 - sa.1 as i64 - sb.1 as i64 + FRAME_GAP_OFFSET);
    let a = (d.max(0) as u64).min(XFPOW2_CAP);
    let mut a_limbs = [0u64; XFPOW2_LIMBS];
    a_limbs[(a / 16) as usize] = 1 << (a % 16);
    v.comparison_power_limbs = a_limbs.map(f);

    // J5: the borrow chain of Y - 2^A over the six digits.
    let mut borrow = 0u64;
    for pos in 0..COMPARISON_DIGITS - 1 {
        let need = a_limbs[pos] + borrow;
        borrow = u64::from(y[pos] < need);
        v.comparison_borrows[pos] = f(borrow);
    }
    assert!(
        y[COMPARISON_DIGITS - 1] >= a_limbs[COMPARISON_DIGITS - 1] + borrow,
        "fill_certificate on an untamed cell"
    );
}

// Constraints, written once against the generic `Evaluator`

/// Evaluates every arithmetic constraint of TamedStark. Lookup-borne facts (LUT oracle,
/// import channels) are *not* emitted here — see the module docs and
/// `ctl::tamed_lut_lookups`.
pub(crate) fn eval_tamed_constraints<V, S, E>(vars: &StarkFrame<V, S, NUM_TAMED_COLUMNS, NUM_TAMED_PUBLIC_INPUTS>, eval: &mut E)
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let lv: &[V; NUM_TAMED_COLUMNS] = vars.get_local_values().try_into().unwrap();
    let lv: &TamedColumnsView<V> = lv.borrow();
    let nv: &[V; NUM_TAMED_COLUMNS] = vars.get_next_values().try_into().unwrap();
    let nv: &TamedColumnsView<V> = nv.borrow();
    let k = eval.scalar(vars.get_public_inputs()[TAME_K_PUBLIC_INPUT]);

    let one = eval.i32(1);
    let limb = eval.u64(LIMB);
    // The certificate gate: every J2-J5 equation is `TAMED * (...)` with
    // TAMED = 1 - UNTAMED - CELL_IS_ZERO — boolean by J1 — active exactly on the rows
    // claiming a tamed nonzero cell (pads included: their all-zero fill satisfies
    // everything).
    let claimed_off = eval.add(lv.untamed, lv.cell_is_zero);
    let tamed = eval.sub(one, claimed_off);
    let gate = |eval: &mut E, c: V| {
        let gated = eval.mul(tamed, c);
        eval.constraint(gated);
    };
    // `sum_limbs - lo - 2^16*hi`, the recurring split shape.
    let off_split = |eval: &mut E, sum: V, lo: V, hi: V| {
        let split = eval.mad(hi, limb, lo);
        eval.sub(sum, split)
    };

    // J1 — the verdict bits and the running count.
    eval.constraint_bool(lv.untamed);
    eval.constraint_bool(lv.cell_is_zero);
    // Mutual exclusion keeps the gate (and the XFPOW2 filter) boolean-valued.
    let both = eval.mul(lv.untamed, lv.cell_is_zero);
    eval.constraint(both);
    // CELL_IS_ZERO * CELL_MAGNITUDE_EXPONENT = 0: only an all-zero cell (CELL_MAGNITUDE_EXPONENT = 0, by Matmul's M13) may
    // set the flag.
    let zero_pin = eval.mul(lv.cell_is_zero, lv.cell_magnitude_exponent);
    eval.constraint(zero_pin);
    // Pads are tamed: the count is exactly the live untamed count when the J6 gate reads it.
    let pad_pin = eval.mul(lv.is_pad, lv.untamed);
    eval.constraint(pad_pin);
    let anchor = eval.sub(lv.running_untamed, lv.untamed);
    eval.constraint_first_row(anchor);
    let step = eval.sub(nv.running_untamed, lv.running_untamed);
    let step = eval.sub(step, nv.untamed);
    eval.constraint_transition(step);

    // J7 — the skip census and its budget gate (jackpot check 4).
    // Pads import no skips (they are outside the E-cell channel, so the pin is load-bearing:
    // an unconstrained pad count could offset the census).
    let pad_skips = eval.mul(lv.is_pad, lv.cell_skips);
    eval.constraint(pad_skips);
    let anchor = eval.sub(lv.running_skips, lv.cell_skips);
    eval.constraint_first_row(anchor);
    let step = eval.sub(nv.running_skips, lv.running_skips);
    let step = eval.sub(step, nv.cell_skips);
    eval.constraint_transition(step);
    // The budget gate: on the last row `SKIP_LIMIT - RUNNING_SKIPS` recomposes from the two
    // range-checked slack limbs, i.e. lies in [0, 2^32) — a census above the budget wraps to
    // p - x > 2^32 and has no such representation.
    let skip_limit = eval.scalar(vars.get_public_inputs()[SKIP_LIMIT_PUBLIC_INPUT]);
    let slack = eval.mad(lv.skip_gate_slack_hi, limb, lv.skip_gate_slack_lo);
    let diff = eval.sub(skip_limit, lv.running_skips);
    let c = eval.sub(diff, slack);
    eval.constraint_last_row(c);

    // J2 — PP = SIG_A*SIG_B, split into range-checked limbs (exact: both factors are
    // import-bound < 2^16, so the product is < 2^32 << p).
    let pp = eval.mul(lv.sigma_a_significand, lv.sigma_b_significand);
    let c = off_split(eval, pp, lv.sigma_product_limbs[0], lv.sigma_product_limbs[1]);
    gate(eval, c);

    // J3 — Y = K * PP^2, via W = K*PP (K = k <= 2^16, the public input).
    // The two partial products K * PP_LO/HI as two-limb recompositions (< 2^32 by the limb
    // RCs, so the field equation is the integer equation).
    for (partial, pp_limb) in [
        (&lv.k_sigma_product_lo, lv.sigma_product_limbs[0]),
        (&lv.k_sigma_product_hi, lv.sigma_product_limbs[1]),
    ] {
        let mut c = eval.mul(k, pp_limb);
        for (pos, &l) in partial.iter().enumerate() {
            let scale = eval.u64(1 << (16 * pos));
            let term = eval.mul(scale, l);
            c = eval.sub(c, term);
        }
        gate(eval, c);
    }

    // W's digit-aligned add: W = KL + 2^16*KH, digits e_0 = KL[0], e_1 committed,
    // e_2 = KH[1] + carry (an expression: its width is forced by KH[1]'s RC16 and the
    // carry bit, so no committed limb is needed).
    let kl = &lv.k_sigma_product_lo;
    let kh = &lv.k_sigma_product_hi;
    let s = eval.add(kl[1], kh[0]);
    let c = off_split(eval, s, lv.k_sigma_product_middle_limbs[0], lv.k_sigma_product_carries[0]);
    gate(eval, c);
    eval.constraint_bool(lv.k_sigma_product_carries[0]);
    let e = [kl[0], lv.k_sigma_product_middle_limbs[0], eval.add(kh[1], lv.k_sigma_product_carries[0])];

    // Y = W * PP schoolbook positions 0..=3 (position 4 is the carry itself: BOUND_TOP;
    // digit 5 is structurally zero — Y < 2^80). Per-position raw sums are < 2^33 + 2^17
    // << p: exact over Z; carries kappa_2..kappa_3 are 17-bit (lo + bit), kappa_1 and the
    // top are 16-bit.
    let bound_carry = |eval: &mut E, pos: usize, lv: &TamedColumnsView<V>| -> V {
        match pos {
            0 => eval.i32(0),
            1 => lv.bound_carry_1,
            2 | 3 => eval.mad(lv.bound_carries_bit[pos - 2], limb, lv.bound_carries_lo[pos - 2]),
            4 => lv.bound_top,
            _ => unreachable!(),
        }
    };
    for &b in &lv.bound_carries_bit {
        eval.constraint_bool(b);
    }
    for pos in 0..COMPARISON_DIGITS - 2 {
        let mut s = bound_carry(eval, pos, lv);
        if pos < 3 {
            s = eval.mad(e[pos], lv.sigma_product_limbs[0], s);
        }
        if pos >= 1 {
            s = eval.mad(e[pos - 1], lv.sigma_product_limbs[1], s);
        }
        let carry_out = bound_carry(eval, pos + 1, lv);
        let c = off_split(eval, s, lv.bound_limbs[pos], carry_out);
        gate(eval, c);
    }

    // J4 — nothing in-AIR: the left side 2^A *is* the XFPOW2-bound one-hot limb vector
    // (`COMPARISON_POWER_LIMBS`), bound on live tamed nonzero rows by the lookup in `super::ctl`.

    // J5 — the comparison borrows (the range-checked digit keys live in the LUT inventory).
    for &b in &lv.comparison_borrows {
        eval.constraint_bool(b);
    }
}

// Stark impl

/// TamedStark. A CTL party of the fp8 batch (`requires_ctls()`): its proofs carry the
/// cross-table openings of the E-cell, sigma and LUT channels, so the batch driver is the
/// only supported proving path.
#[derive(Clone, Debug)]
pub struct TamedStark<F: RichField + Extendable<D>, const D: usize> {
    pub program: TamedProgram,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> TamedStark<F, D> {
    pub fn new(program: TamedProgram) -> Self {
        Self {
            program,
            _phantom: PhantomData,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for TamedStark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, NUM_TAMED_COLUMNS, NUM_TAMED_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget = StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, NUM_TAMED_COLUMNS, NUM_TAMED_PUBLIC_INPUTS>;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let mut evaluator = NativeEvaluator::new(yield_constr);
        eval_tamed_constraints(vars, &mut evaluator);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let mut evaluator = SymbolicEvaluator::new(builder, yield_constr);
        eval_tamed_constraints(vars, &mut evaluator);
    }

    fn constraint_degree(&self) -> usize {
        3
    }

    // Party to the E-cell and sigma channels plus the committed LUT channels (declared in
    // `super::super::ctl` / `super::ctl`).
    fn requires_ctls(&self) -> bool {
        true
    }
}

// Tests

#[cfg(test)]
mod tests {
    use core::borrow::BorrowMut;

    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::{Field, PrimeField64};
    use plonky2::plonk::config::PoseidonGoldilocksConfig;
    use starky::stark_testing::{test_stark_circuit_constraints, test_stark_low_degree};

    use super::super::ctl::tamed_lut_lookups;
    use super::*;
    use crate::circuit::fp8::ctl::LutTable;
    use crate::circuit::fp8::luts;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = GoldilocksField;
    type S = TamedStark<F, D>;

    /// A 10x10 tile with k = 64: one untamed cell, one all-zero cell, and 197 skips.
    /// Sigma frames and the remaining cell exponents keep all other cells tamed.
    /// Padding must preserve both counts through the final budget checks.
    #[allow(clippy::type_complexity)]
    fn test_program_and_inputs() -> (TamedProgram, Vec<(u64, u64)>, Vec<Frame>, Vec<Frame>) {
        let program = TamedProgram { h: 10, w: 10, k: 64 };
        let sigma_a: Vec<Frame> = (0..program.h as u64)
            .map(|i| (16384 + 1234 * i % 49152, 2040 + 3 * i % 16))
            .collect();
        let sigma_b: Vec<Frame> = (0..program.w as u64)
            .map(|j| (16384 + 4321 * j % 49152, 2038 + 5 * j % 18))
            .collect();
        let mut cells = Vec::new();
        for cell in 0..program.h * program.w {
            cells.push(match cell {
                0 => (400, 0),                                        // far above the bound: untamed
                3 => (0, 0),                                          // all-zero cell: the zero short-circuit
                _ => (121 + (cell as u64 * 7) % 37, cell as u64 % 5), // honest binade range
            });
        }
        (program, cells, sigma_a, sigma_b)
    }

    fn test_trace() -> (TamedProgram, Vec<[F; NUM_TAMED_COLUMNS]>, [F; NUM_TAMED_PUBLIC_INPUTS]) {
        let (program, cells, sigma_a, sigma_b) = test_program_and_inputs();
        let (rows, pis) = program.generate_trace(&cells, &sigma_a, &sigma_b);
        (program, rows, pis)
    }

    fn check_constraints(rows: &[[F; NUM_TAMED_COLUMNS]], pis: &[F; NUM_TAMED_PUBLIC_INPUTS]) -> Result<(), String> {
        let stark = S::new(TamedProgram { h: 10, w: 10, k: 64 });
        let n = rows.len();
        for i in 0..n {
            let frame = StarkFrame::from_values(&rows[i], &rows[(i + 1) % n], pis);
            // starky's z_last semantics: transitions are excluded on the last -> first wrap.
            let mut consumer = ConstraintConsumer::new(
                vec![F::from_canonical_u64(2), F::from_canonical_u64(0x876543210)],
                F::from_bool(i != n - 1),
                F::from_bool(i == 0),
                F::from_bool(i == n - 1),
            );
            stark.eval_packed_generic(&frame, &mut consumer);
            for acc in consumer.accumulators() {
                if acc != F::ZERO {
                    return Err(format!("constraints do not vanish on row {i}"));
                }
            }
        }
        Ok(())
    }

    /// Replays every LUT inventory instance against the committed tables' definitions.
    fn check_lut_lookups(rows: &[[F; NUM_TAMED_COLUMNS]], pis: &[F; NUM_TAMED_PUBLIC_INPUTS]) -> Result<(), String> {
        use starky::util::trace_rows_to_poly_values;
        let polys = trace_rows_to_poly_values(rows.to_vec());
        for (li, lookup) in tamed_lut_lookups::<F>().iter().enumerate() {
            for row in 0..rows.len() {
                let f = lookup.filter.eval_table(&polys, row, pis);
                if f == F::ZERO {
                    continue;
                }
                if f != F::ONE {
                    return Err(format!("lookup {li}: non-boolean filter on row {row}"));
                }
                let key = lookup.keys[0].eval_table(&polys, row, pis).to_canonical_u64();
                match lookup.table {
                    LutTable::Range16 => {
                        if key >= 1 << 16 {
                            return Err(format!("lookup {li} (RC16) row {row}: key {key} out of range"));
                        }
                    }
                    LutTable::XfPow2 => {
                        if key >= 1 << 11 {
                            return Err(format!("lookup {li} (XFPOW2) row {row}: key {key} out of domain"));
                        }
                        let table = luts::generate::<F>(LutTable::XfPow2, 0);
                        for (vi, col) in lookup.values.iter().enumerate() {
                            let got = col.eval_table(&polys, row, pis);
                            if got != table[vi][key as usize] {
                                return Err(format!("lookup {li} (XFPOW2) row {row}: value {vi} mismatch"));
                            }
                        }
                    }
                    other => return Err(format!("unexpected table {other:?} in the Tamed inventory")),
                }
            }
        }
        Ok(())
    }

    #[test]
    fn honest_trace_satisfies_constraints_and_lookups() {
        let (program, rows, pis) = test_trace();
        assert_eq!(rows.len(), program.num_rows());
        check_constraints(&rows, &pis).unwrap();
        check_lut_lookups(&rows, &pis).unwrap();
        // The verdict bits match the exact predicate (cell 0 untamed by construction);
        // the all-zero cell 3 takes the zero short-circuit.
        for (cell, row) in rows[..program.h * program.w].iter().enumerate() {
            let v: &TamedColumnsView<F> = row.borrow();
            assert_eq!(v.untamed == F::ONE, cell == 0, "cell {cell}");
            assert_eq!(v.cell_is_zero == F::ONE, cell == 3, "cell {cell}");
        }
        // The skip census accumulates to the fixture total and freezes through the pads.
        let last: &TamedColumnsView<F> = rows.last().unwrap().borrow();
        assert_eq!(last.running_skips, F::from_canonical_u64(197));
    }

    #[test]
    fn known_values_are_bit_exact_with_the_trace() {
        let (program, rows, _) = test_trace();
        let known = program.known_values::<F>();
        assert_eq!(known.len(), NUM_TAMED_KNOWN_COLUMNS);
        for (c, col) in known.iter().enumerate() {
            for (r, &v) in col.values.iter().enumerate() {
                assert_eq!(v, rows[r][c], "known column {c} row {r}");
            }
        }
    }

    #[test]
    fn understating_the_verdict_is_unsatisfiable() {
        // Cell 0 is untamed; flipping its bit to 0 activates the certificate, which the
        // all-zero witness block cannot satisfy (PP != 0 breaks the J2 split).
        let (_, mut rows, pis) = test_trace();
        {
            let v: &mut TamedColumnsView<F> = rows[0].borrow_mut();
            assert_eq!(v.untamed, F::ONE, "test premise: cell 0 is untamed");
            v.untamed = F::ZERO;
        }
        assert!(check_constraints(&rows, &pis).is_err(), "J2 must reject the zero certificate");
    }

    #[test]
    fn zero_claim_on_a_nonzero_cell_is_unsatisfiable() {
        // Cell 1 is a tamed nonzero cell; claiming CELL_IS_ZERO both violates the J1 zero pin
        // (CELL_MAGNITUDE_EXPONENT != 0) and, were CELL_MAGNITUDE_EXPONENT also forged to 0, would desync the import channel.
        let (_, mut rows, pis) = test_trace();
        {
            let v: &mut TamedColumnsView<F> = rows[1].borrow_mut();
            assert_eq!(v.untamed, F::ZERO, "test premise: cell 1 is tamed");
            assert!(v.cell_magnitude_exponent != F::ZERO, "test premise: cell 1 is nonzero");
            v.cell_is_zero = F::ONE;
        }
        assert!(check_constraints(&rows, &pis).is_err(), "the J1 zero pin must reject");
    }

    #[test]
    fn tame_limit_public_input_is_load_bearing() {
        // Lowering TAME_LIMIT below the honest count makes the J6 gate key negative: the
        // policy threshold really is the public input, not the trace.
        let (_, rows, mut pis) = test_trace();
        pis[TAME_LIMIT_PUBLIC_INPUT] = F::ZERO; // honest count is 1
        assert!(check_lut_lookups(&rows, &pis).is_err(), "the J6 RC16 must reject");
    }

    #[test]
    fn skip_limit_public_input_is_load_bearing() {
        // Lowering SKIP_LIMIT below the honest census (197) breaks the last-row slack
        // recomposition; no in-range slack limbs exist for a negative difference.
        let (_, rows, mut pis) = test_trace();
        pis[SKIP_LIMIT_PUBLIC_INPUT] = F::from_canonical_u64(196);
        assert!(check_constraints(&rows, &pis).is_err(), "the J7 gate must reject");
    }

    #[test]
    fn understating_the_census_is_unsatisfiable() {
        // Shrinking one cell's imported census desynchronizes the running sum (and the
        // import channel, were the tuple also forged at the Matmul side).
        let (_, mut rows, pis) = test_trace();
        {
            let v: &mut TamedColumnsView<F> = rows[4].borrow_mut();
            assert!(v.cell_skips != F::ZERO, "test premise: cell 4 has skips");
            v.cell_skips -= F::ONE;
        }
        assert!(check_constraints(&rows, &pis).is_err(), "the J7 recurrence must reject");
    }

    #[test]
    fn untamed_predicate_matches_the_plaintext_shape() {
        // Boundary sweep: with K = k = 64 and sigma sigs 2^15 at exps 2048 (PP = 2^30,
        // Y = 64 * 2^60 = 2^66, bitlen 67), D = 2*(cell_magnitude_exponent - 147): untamed iff D >= 67 iff
        // cell_magnitude_exponent >= 181 (i.e. e_M >= 42 — the bound is tau_tame * sqrt(k) * sigma_A *
        // sigma_B = 256 * 8 * 2^30 = 2^41, and the predicate flags exactly the binades
        // starting above it: ufp(M) = 2^e_M > 2^41).
        assert!(!untamed(64, 180, (1 << 15, 2048), (1 << 15, 2048)));
        assert!(untamed(64, 181, (1 << 15, 2048), (1 << 15, 2048)));
        // Zero threshold: untamed iff M > 0.
        assert!(untamed(0, 1, (1 << 15, 2048), (1 << 15, 2048)));
        assert!(!untamed(0, 0, (1 << 15, 2048), (1 << 15, 2048)));
        // The all-zero cell is tamed against any positive bound.
        assert!(!untamed(64, 0, (1 << 15, 2048), (1 << 15, 2048)));
        // A zero sigma side zeroes the bound: any nonzero binade is untamed.
        assert!(untamed(64, 121, (0, 2048), (1 << 15, 2048)));
    }

    #[test]
    fn certificate_is_tight_at_the_boundary() {
        // At the last tamed binade (cell_magnitude_exponent = 180 in the sweep above) the certificate must be
        // satisfiable with A = D = 66 and Y = 2^66 — equality, zero borrows.
        let (sa, sb): (Frame, Frame) = ((1 << 15, 2048), (1 << 15, 2048));
        let mut v = TamedColumnsView::<F>::default();
        fill_certificate(&mut v, 64, 180, sa, sb);
        // A = 66: limb 4 holds 2^2; Y's digit 4 (BOUND_TOP) holds 2^2 too (Y = 2^66).
        assert_eq!(v.comparison_power_limbs[4], F::from_canonical_u64(4));
        assert_eq!(v.bound_top, F::from_canonical_u64(4));
        assert!(
            v.comparison_borrows.iter().all(|&b| b == F::ZERO),
            "equality needs no borrows"
        );
    }

    #[test]
    fn stark_degree_and_circuit() {
        let stark = S::new(TamedProgram { h: 10, w: 10, k: 64 });
        test_stark_low_degree::<F, S, D>(stark.clone()).unwrap();
        test_stark_circuit_constraints::<F, C, S, D>(stark).unwrap();
    }
}
