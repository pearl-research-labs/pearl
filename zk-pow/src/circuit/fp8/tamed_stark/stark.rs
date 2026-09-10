//! TamedStark: jackpot check 3 — the tamed-products count — and the check 4 budget gate.
//!
//! One row per tile cell `(i, j)`. The row imports the cell's replay-magnitude binade
//! `E_CELL = floor(log2 M_ij) + 139` and its skip census `CELL_SKIPS` from Matmul (E_CELL is
//! 0 only for an all-zero cell; M13/MB13 prove E_CELL bounds every product's and partial
//! sum's binade) and the two exact noise stds `sigma = SIG * 2^(EXP - 2048)` from Scale,
//! and decides the untamed predicate
//!
//! ```text
//! UNTAMED(i,j)  <=>  2^(2*e_M) > tau_tame^2 * k * (sigma_A * sigma_B)^2
//! ```
//!
//! exactly over the reals, mirroring the plaintext `jackpot_policy::untamed_exact`. The
//! policy freezes `tau_tame = 128`, so `tau_tame^2 = 2^14` moves to the exponent side of
//! the comparison (`FRAME_GAP_OFFSET`) and only `K = k <= 2^16` rides the proof as a public
//! input. The AIR is one-sided: a committed `UNTAMED = 0` activates the *tamed certificate*
//!
//! ```text
//! 2^A <= Y,   Y = K * PP^2,   PP = SIG_A * SIG_B,
//! ```
//!
//! with `2^A` the XFPOW2 saturating power of the doubled frame gap with the tau fold,
//! `D = 2*(e_M - e_A - e_B - 7)` (`A = min(max(D, 0), 80)`; since `Y < 2^80`, saturation
//! never changes the verdict). `UNTAMED = 1` rows prove nothing, so the count can only be
//! *overstated* — and the J6 gate `RUNNING_UNTAMED <= TAME_LIMIT` on the last row makes
//! overstating useless. An all-zero cell (`M = 0`) is tamed by definition and skips the
//! certificate through `CELL_IS_ZERO` (J1 pins `CELL_IS_ZERO * E_CELL = 0`), since the
//! certificate would demand `Y >= 2^A > 0`. Completeness: the imported `E_CELL` is the
//! binade of the plaintext `M_ij`, so the certificate is satisfiable exactly on the
//! plaintext-tamed nonzero cells.
//!
//! Constraint groups (lookups live in `super::ctl`):
//!
//! - **J1 (verdict):** `UNTAMED` and `CELL_IS_ZERO` boolean and mutually exclusive (the gate
//!   `1 - UNTAMED - CELL_IS_ZERO` stays boolean), `UNTAMED` pinned 0 on pads,
//!   `CELL_IS_ZERO * E_CELL = 0`; `RUNNING_UNTAMED` anchored on the first row and
//!   accumulated by transition.
//! - **J2 (product):** `PP = SIG_A*SIG_B` split into an RC16'd limb pair (both factors are
//!   import-bound below 2^16, so the split is exact over Z).
//! - **J3 (bound build):** `W = K*PP` via the two partial products `K*PP_LO/HI` (each
//!   < 2^32: two RC16'd limbs) and a digit-aligned add; then `Y = W*PP` by base-2^16
//!   schoolbook positions with 17-bit carries. Every position equation is exact over Z
//!   (all terms < 2^34 << p), so the digits are the unique base-2^16 representation.
//! - **J4 (left side):** none in-AIR — `2^A`'s digits are the XFPOW2-bound one-hot limbs.
//! - **J5 (comparison):** a digit-wise borrow chain over the six positions, RC16'd keys,
//!   no borrow out of the top digit — `2^A <= Y` exactly.
//! - **J7 (skip budget, jackpot check 4):** `CELL_SKIPS` pinned 0 on pads; `RUNNING_SKIPS`
//!   anchored and accumulated like `RUNNING_UNTAMED`; on the last row
//!   `SKIP_LIMIT - RUNNING_SKIPS` recomposes from its two RC16'd slack limbs, i.e. lies in
//!   `[0, 2^32)`. The imported census can only overstate the true skippable count (Matmul's
//!   M15/MB15), so an accepted proof implies the honest census is within budget.
//!
//! All J2-J5 equations are gated by `1 - UNTAMED - CELL_IS_ZERO`. Pad rows keep the gate
//! active with all-zero imports and witnesses (`2^0` has zero limbs there — the XFPOW2
//! lookup is filtered off, and `0 <= 0` digit-wise), so padding needs no extra machinery.

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

/// `log2(tau_tame^2) = 14`: the policy's `tau_tame = 128` makes the tamed threshold's tau
/// factor a power of two, folded into the exponent side of the comparison instead of the
/// `Y` build (`TamedProgram::k` asserts the policy still matches).
pub(crate) const TAU_TAME_SQ_LOG2: i64 = 16;

/// The halved shift of the squared comparison, folding every bias and the tau fold into one
/// constant:
/// `D/2 = e_M - e_A - e_B - 7 = (E_CELL - 139) - (SA - 2048) - (SB - 2048) - 14/2
///      = E_CELL - SA - SB + 3950`.
/// The XFPOW2 key adds the table's zero point on top (`super::ctl`).
pub(crate) const FRAME_GAP_OFFSET: i64 = 2 * 2048 - (BINADE_BIAS as i64) - TAU_TAME_SQ_LOG2 / 2;

// ==================================================================================================
// Program and trace generation
// ==================================================================================================

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

    /// The inner dimension `k` — the `K` public input. The untamed threshold is
    /// `tau_tame^2 * k * PP^2` with `tau_tame^2 = 128^2 = 2^14` folded into the XFPOW2
    /// shift (`FRAME_GAP_OFFSET`), so only `k` rides the proof; the assert pins the frozen
    /// policy the fold assumes (`2^16` bounds `k`, the width every J3 cap assumes).
    fn k(&self) -> u64 {
        let policy = JackpotPolicy::default();
        assert!(
            policy.tau_tame * policy.tau_tame == (1u64 << TAU_TAME_SQ_LOG2) as f64,
            "the circuit freezes tau_tame^2 = 2^14 (folded into the XFPOW2 shift)"
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

    /// Generates the TamedStark trace and public inputs. `cells` is one
    /// `(E_CELL, CELL_SKIPS)` tuple per cell in row-major cell order — the exact tuples
    /// Matmul's E-cell channel exports (`E_CELL = floor(log2 M) + 139`, sentinel 0 for
    /// all-zero cells; `CELL_SKIPS` the cell's skip census); `sigma_a` / `sigma_b` are the
    /// per-tile-row/-column `(SIG, EXP)` frames of Scale's sigma channel, exponents biased
    /// by 2048.
    ///
    /// Panics on a policy-rejected witness (untamed count above the allowance, or skip
    /// census above the budget) — the J6/J7 gates have no satisfying row, so a prover has
    /// no business tracing it.
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
        for (cell, &(e_cell, skips)) in cells.iter().enumerate() {
            let (i, j) = (cell / w, cell % w);
            let (sa, sb) = (sigma_a[i], sigma_b[j]);
            running_skips += skips;
            let mut v = TamedColumnsView {
                cell_id: F::from_canonical_usize(cell),
                a_group_key: F::from_canonical_usize((i + 1) * self.k - 1),
                b_group_key: F::from_canonical_usize(h * self.k + (j + 1) * self.k - 1),
                is_last_row: F::from_bool(cell == num_rows - 1),
                e_cell: F::from_canonical_u64(e_cell),
                sigma_a_significand: F::from_canonical_u64(sa.0),
                sigma_a_exp: F::from_canonical_u64(sa.1),
                sigma_b_significand: F::from_canonical_u64(sb.0),
                sigma_b_exp: F::from_canonical_u64(sb.1),
                cell_skips: F::from_canonical_u64(skips),
                running_skips: F::from_canonical_u64(running_skips),
                ..TamedColumnsView::default()
            };
            if untamed(k, e_cell, sa, sb) {
                running += 1;
                v.untamed = F::ONE;
            } else if e_cell == 0 {
                v.cell_is_zero = F::ONE; // M = 0: tamed by definition, certificate skipped.
            } else {
                fill_certificate(&mut v, k, e_cell, sa, sb);
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

    /// The class (a) ("known") column values — the leading [`NUM_TAMED_KNOWN_COLUMNS`] trace
    /// columns in their `columns.rs` order (`cell_id`, `a_group_key`, `b_group_key`,
    /// `is_last_row`, `is_pad`), pure functions of the program geometry. Bit-exact with
    /// [`Self::generate_trace`]'s fill; the batch verifier recomputes exactly this and checks
    /// the trace openings against it (`starky`'s `BatchKnownColumns`).
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

/// The untamed predicate on the exact imported values —
/// `2^(2*e_M) > tau_tame^2 * k * (sigma_A*sigma_B)^2` over the reals — mirroring
/// `jackpot_policy::untamed_exact` at the frozen `tau_tame^2 = 2^14`, which moves to the
/// exponent side: `2^D > Y` with `D` the tau-folded doubled frame gap and `Y = k * PP^2`
/// (`< 2^80`, u128-exact; the left side is a pure power of two, so the comparison is one
/// bit-length test).
fn untamed(k: u64, e_cell: u64, sa: Frame, sb: Frame) -> bool {
    let pp = sa.0 * sb.0;
    if k == 0 || pp == 0 {
        return e_cell != 0; // bound = 0: untamed iff M > 0
    }
    if e_cell == 0 {
        return false; // M = 0, bound > 0
    }
    let y = u128::from(k) * u128::from(pp) * u128::from(pp);
    // 2^D > Y at the tau-folded doubled frame gap (the biases and the 2^14 collapse to
    // FRAME_GAP_OFFSET); 2^D > Y <=> D >= bitlen(Y) for Y > 0.
    let d = 2 * (e_cell as i64 - sa.1 as i64 - sb.1 as i64 + FRAME_GAP_OFFSET);
    d >= i64::from(128 - y.leading_zeros())
}

/// Fills the tamed-certificate witness block (J2-J5) of one nonzero tamed row: the exact
/// schoolbook digits of `Y = K * PP^2`, the one-hot limbs of `2^A`, and the borrow chain
/// proving `2^A <= Y`. Panics if the cell is in fact untamed (the caller decides the
/// verdict first).
fn fill_certificate<F: RichField>(v: &mut TamedColumnsView<F>, k: u64, e_cell: u64, sa: Frame, sb: Frame) {
    let f = F::from_canonical_u64;

    // ---- J2: the sigma product's limb split. ----
    let pp = sa.0 * sb.0;
    let (pp_lo, pp_hi) = (pp & MASK, pp >> 16);
    v.sigma_product_limbs = [f(pp_lo), f(pp_hi)];

    // ---- J3: W = K * PP (digits e), then Y = W * PP (digits y). ----
    let limb2 = |x: u64| [x & MASK, x >> 16];
    let (kl, kh) = (limb2(k * pp_lo), limb2(k * pp_hi));
    v.k_pp_lo_partial = kl.map(f);
    v.k_pp_hi_partial = kh.map(f);
    let s = kl[1] + kh[0];
    let (w1, carry) = (s & MASK, s >> 16);
    v.k_pp_mid_limbs[0] = f(w1);
    v.k_pp_carries[0] = f(carry);
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

    // ---- J4: the saturated one-hot power 2^A. ----
    let d = 2 * (e_cell as i64 - sa.1 as i64 - sb.1 as i64 + FRAME_GAP_OFFSET);
    let a = (d.max(0) as u64).min(XFPOW2_CAP);
    let mut a_limbs = [0u64; XFPOW2_LIMBS];
    a_limbs[(a / 16) as usize] = 1 << (a % 16);
    v.shift_a_limbs = a_limbs.map(f);

    // ---- J5: the borrow chain of Y - 2^A over the six digits. ----
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

// ==================================================================================================
// Constraints, written once against the generic `Evaluator`
// ==================================================================================================

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

    // ---- J1 — the verdict bits and the running count. ----
    eval.constraint_bool(lv.untamed);
    eval.constraint_bool(lv.cell_is_zero);
    // Mutual exclusion keeps the gate (and the XFPOW2 filter) boolean-valued.
    let both = eval.mul(lv.untamed, lv.cell_is_zero);
    eval.constraint(both);
    // CELL_IS_ZERO * E_CELL = 0: only an all-zero cell (E_CELL = 0, by Matmul's M13) may
    // set the flag.
    let zero_pin = eval.mul(lv.cell_is_zero, lv.e_cell);
    eval.constraint(zero_pin);
    // Pads are tamed: the count is exactly the live untamed count when the J6 gate reads it.
    let pad_pin = eval.mul(lv.is_pad, lv.untamed);
    eval.constraint(pad_pin);
    let anchor = eval.sub(lv.running_untamed, lv.untamed);
    eval.constraint_first_row(anchor);
    let step = eval.sub(nv.running_untamed, lv.running_untamed);
    let step = eval.sub(step, nv.untamed);
    eval.constraint_transition(step);

    // ---- J7 — the skip census and its budget gate (jackpot check 4). ----
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
    // RC16'd slack limbs, i.e. lies in [0, 2^32) — a census above the budget wraps to
    // p - x > 2^32 and has no such representation.
    let skip_limit = eval.scalar(vars.get_public_inputs()[SKIP_LIMIT_PUBLIC_INPUT]);
    let slack = eval.mad(lv.skip_gate_slack_hi, limb, lv.skip_gate_slack_lo);
    let diff = eval.sub(skip_limit, lv.running_skips);
    let c = eval.sub(diff, slack);
    eval.constraint_last_row(c);

    // ---- J2 — PP = SIG_A*SIG_B, split into RC16'd limbs (exact: both factors are
    // import-bound < 2^16, so the product is < 2^32 << p). ----
    let pp = eval.mul(lv.sigma_a_significand, lv.sigma_b_significand);
    let c = off_split(eval, pp, lv.sigma_product_limbs[0], lv.sigma_product_limbs[1]);
    gate(eval, c);

    // ---- J3 — Y = K * PP^2, via W = K*PP (K = k <= 2^16, the public input). ----
    // The two partial products K * PP_LO/HI as two-limb recompositions (< 2^32 by the limb
    // RCs, so the field equation is the integer equation).
    for (partial, pp_limb) in [
        (&lv.k_pp_lo_partial, lv.sigma_product_limbs[0]),
        (&lv.k_pp_hi_partial, lv.sigma_product_limbs[1]),
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
    let kl = &lv.k_pp_lo_partial;
    let kh = &lv.k_pp_hi_partial;
    let s = eval.add(kl[1], kh[0]);
    let c = off_split(eval, s, lv.k_pp_mid_limbs[0], lv.k_pp_carries[0]);
    gate(eval, c);
    eval.constraint_bool(lv.k_pp_carries[0]);
    let e = [kl[0], lv.k_pp_mid_limbs[0], eval.add(kh[1], lv.k_pp_carries[0])];

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

    // ---- J4 — nothing in-AIR: the left side 2^A *is* the XFPOW2-bound one-hot limb vector
    // (`SHIFT_A_LIMBS`), bound on live tamed nonzero rows by the lookup in `super::ctl`. ----

    // ---- J5 — the comparison borrows (the RC16'd digit keys live in the LUT inventory). ----
    for &b in &lv.comparison_borrows {
        eval.constraint_bool(b);
    }
}

// ==================================================================================================
// Stark impl
// ==================================================================================================

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

// ==================================================================================================
// Tests
// ==================================================================================================

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

    /// A 10x10 tile over k = 64 (untamed allowance `floor(100 / 64) = 1`, skip budget
    /// `floor(64 * 100 / 16) = 400`, and `K = k = 64`) with realistic frame magnitudes:
    /// sigma significands are products of bf16 significand pairs (in `[2^14, 2^16)`, so
    /// `Y = 64 * PP^2 >= 2^62`) at exponents near 2048 (`SA + SB in [4078, 4110]`); live
    /// binades sit in the honest [121, 157] range, keeping
    /// `D <= 2*(157 + 3950 - 4078) = 58 < 63 <= bitlen(Y)`: tamed. Cell 0 is untamed by
    /// construction (`D >= 2*(400 + 3950 - 4110) = 480 > bitlen(Y)`); cell 3 is the
    /// all-zero cell (zero skips — a zero cell never counts as skipped). The censuses sum
    /// to 197, within the budget.
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
        // (E_CELL != 0) and, were E_CELL also forged to 0, would desync the import channel.
        let (_, mut rows, pis) = test_trace();
        {
            let v: &mut TamedColumnsView<F> = rows[1].borrow_mut();
            assert_eq!(v.untamed, F::ZERO, "test premise: cell 1 is tamed");
            assert!(v.e_cell != F::ZERO, "test premise: cell 1 is nonzero");
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
        // Y = 64 * 2^60 = 2^66, bitlen 67), D = 2*(e_cell - 147): untamed iff D >= 67 iff
        // e_cell >= 181 (i.e. e_M >= 42 — the bound is tau_tame * sqrt(k) * sigma_A *
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
        // At the last tamed binade (e_cell = 180 in the sweep above) the certificate must be
        // satisfiable with A = D = 66 and Y = 2^66 — equality, zero borrows.
        let (sa, sb): (Frame, Frame) = ((1 << 15, 2048), (1 << 15, 2048));
        let mut v = TamedColumnsView::<F>::default();
        fill_certificate(&mut v, 64, 180, sa, sb);
        // A = 66: limb 4 holds 2^2; Y's digit 4 (BOUND_TOP) holds 2^2 too (Y = 2^66).
        assert_eq!(v.shift_a_limbs[4], F::from_canonical_u64(4));
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
