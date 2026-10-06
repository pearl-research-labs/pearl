//! The FP16 jackpot-policy AIR: proves an opened tile clears the "unpredictable accumulation
//! steps" gate of [`crate::v5::api::policy`] — breakpoint density `f_bp >= 0.30` AND
//! certified-work ratio `rho >= 1.2` — as exact integer inequalities over the per-group census.
//!
//! The AIR forks the tile-wide integer-budget pattern of `circuit::fp8::unpredictability`: it
//! walks the same one-row-per-group-step grid as the A100 matmul AIR, accumulates the three
//! policy quantities per cell and tile-wide, and gates on the last row. The gate avoids division
//! by clearing denominators:
//!
//! * `rho = numerator / (cells*k) >= 6/5  <=>  5*numerator >= 6*cells*k`,
//! * `f_bp = breakpoints / total_steps >= 3/10  <=>  10*breakpoints >= 3*total_steps`,
//!
//! with `numerator = sum_cells[ 8*N_bp + 32*N_runs + N_pt ]` (`G = 8`, `NOISE_RANK = 32`),
//! `N_runs` counted by a run-start detector (a step starts a run iff it is non-breakpoint and the
//! previous step was a breakpoint or the cell start), and `N_pt` the products truncated on
//! non-breakpoint steps. Each inequality is witnessed by a nonnegative slack whose 16-bit limbs
//! are RANGE16-checked, so a tile below either threshold has no satisfying (nonnegative) slack.
//!
//! The per-step census columns (`group_breakpoint`, `products_truncated`) are the matmul AIR's
//! tightly-pinned census (MA11 / MA3); in the FP16 batch they are bound to the matmul's via the
//! `matmul -> policy` census-import CTL ([`crate::v5::circuit::ctl::census_import_ctl`]). Here the
//! trace generator fills them from the ground-truth [`crate::v5::api::accumulate::a100_dot`]
//! census, and the tests cross-check the AIR's gate against [`crate::v5::api::policy::evaluate`].

use core::borrow::Borrow;
use std::marker::PhantomData;

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
    NUM_POLICY_A100_COLUMNS, NUM_POLICY_A100_PUBLIC_INPUTS, PolicyA100ColumnsView,
};
use crate::v5::api::accumulate::{GROUP, PolicyStep, a100_dot};
use crate::v5::api::policy::{MIN_FBP, MIN_RHO, NOISE_RANK};
use crate::v4::circuit::utils::evaluator::Evaluator;
use crate::v4::circuit::utils::native_evaluator::NativeEvaluator;
use crate::v4::circuit::utils::symbolic_evaluator::SymbolicEvaluator;

/// `rho >= MIN_RHO` cleared of division: `RHO_NUM * numerator >= RHO_DEN * cells * k`.
/// `MIN_RHO = 1.2 = 6/5`.
const RHO_NUM: u64 = 5;
const RHO_DEN: u64 = 6;
/// `f_bp >= MIN_FBP` cleared of division: `FBP_NUM * breakpoints >= FBP_DEN * total_steps`.
/// `MIN_FBP = 0.30 = 3/10`.
const FBP_NUM: u64 = 10;
const FBP_DEN: u64 = 3;

const _: () = assert!(RHO_DEN as f64 / RHO_NUM as f64 == MIN_RHO);
const _: () = assert!(FBP_DEN as f64 / FBP_NUM as f64 == MIN_FBP);

/// The A100 FP16 policy AIR for one `h x w` tile with inner dimension `k` (a multiple of `G`).
#[derive(Clone, Debug)]
pub struct PolicyStarkA100<F: RichField + Extendable<D>, const D: usize> {
    pub h: usize,
    pub w: usize,
    pub k: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> PolicyStarkA100<F, D> {
    pub fn new(h: usize, w: usize, k: usize) -> Self {
        assert_eq!(k % GROUP, 0, "k must be a multiple of the group size {GROUP}");
        Self { h, w, k, _phantom: PhantomData }
    }

    pub fn rows_per_cell(&self) -> usize {
        self.k / GROUP
    }

    pub fn live_rows(&self) -> usize {
        self.h * self.w * self.rows_per_cell()
    }

    pub fn num_rows(&self) -> usize {
        self.live_rows().next_power_of_two()
    }

    /// `cells * k` — the `rho` denominator (before clearing).
    fn cells_times_k(&self) -> u64 {
        (self.h * self.w * self.k) as u64
    }

    /// `total_steps = live_rows` — the `f_bp` denominator.
    fn total_steps(&self) -> u64 {
        self.live_rows() as u64
    }

    /// The gate's cleared right-hand sides `(6*cells*k, 3*total_steps)`.
    fn gate_constants(&self) -> (u64, u64) {
        (RHO_DEN * self.cells_times_k(), FBP_DEN * self.total_steps())
    }

    /// Class (a) known columns: `cell_id`, `is_cell_final`, `is_padding`, and the two
    /// operand-index bases (`operand_index_base_a/b`) the census-import CTL keys on. The bases
    /// mirror the matmul AIR's columns of the same name bit-for-bit so the `(base_a, base_b)`
    /// key tuple matches across the two tables on every live row.
    pub fn known_values(&self) -> Vec<PolynomialValues<F>> {
        let (h, w, k) = (self.h, self.w, self.k);
        let rpc = self.rows_per_cell();
        let num_rows = self.num_rows();
        let mut cell_id = Vec::with_capacity(num_rows);
        let mut is_cell_final = Vec::with_capacity(num_rows);
        let mut is_padding = Vec::with_capacity(num_rows);
        let mut base_a = Vec::with_capacity(num_rows);
        let mut base_b = Vec::with_capacity(num_rows);
        for r in 0..h {
            for c in 0..w {
                for j in 0..rpc {
                    cell_id.push(F::from_canonical_usize(r * w + c));
                    is_cell_final.push(F::from_bool(j == rpc - 1));
                    is_padding.push(F::ZERO);
                    base_a.push(F::from_canonical_usize(r * k + j * GROUP));
                    base_b.push(F::from_canonical_usize(h * k + c * k + j * GROUP));
                }
            }
        }
        for t in 0..num_rows - self.live_rows() {
            cell_id.push(F::from_canonical_usize(h * w + t));
            is_cell_final.push(F::ONE);
            is_padding.push(F::ONE);
            base_a.push(F::ZERO);
            base_b.push(F::ZERO);
        }
        [cell_id, is_cell_final, is_padding, base_a, base_b]
            .into_iter()
            .map(PolynomialValues::new)
            .collect()
    }

    /// Generates the policy trace for the `h x w x k` tile (operands row-major; `b` transposed),
    /// accumulated from `c = 0` exactly as the scheme does. The per-step census is the
    /// ground-truth `a100_dot` census; the run detector and the tile accumulators are filled to
    /// match [`crate::v5::api::policy::evaluate`] bit-for-bit.
    pub fn generate_trace(&self, a_codes: &[u16], b_codes: &[u16]) -> Vec<[F; NUM_POLICY_A100_COLUMNS]> {
        let (h, w, k) = (self.h, self.w, self.k);
        assert_eq!(a_codes.len(), h * k, "a_codes must be h*k FP16 codes");
        assert_eq!(b_codes.len(), w * k, "b_codes must be w*k FP16 codes");
        let rpc = self.rows_per_cell();
        let num_rows = self.num_rows();
        let mut rows: Vec<[F; NUM_POLICY_A100_COLUMNS]> = Vec::with_capacity(num_rows);

        let mut tile_breakpoints: u64 = 0;
        let mut tile_numerator: u64 = 0;
        for r in 0..h {
            for c in 0..w {
                let mut census: Vec<PolicyStep> = Vec::with_capacity(rpc);
                a100_dot(&a_codes[r * k..r * k + k], &b_codes[c * k..c * k + k], 0.0, Some(&mut census));
                debug_assert_eq!(census.len(), rpc);
                let mut in_run = false;
                for (j, step) in census.iter().enumerate() {
                    let run_start = !step.breakpoint && !in_run;
                    if step.breakpoint {
                        in_run = false;
                    } else if !in_run {
                        in_run = true;
                    }
                    tile_breakpoints += u64::from(step.breakpoint);
                    tile_numerator += GROUP as u64 * u64::from(step.breakpoint)
                        + NOISE_RANK * u64::from(run_start)
                        + if step.breakpoint { 0 } else { u64::from(step.products_truncated) };
                    let row = PolicyA100ColumnsView::<F> {
                        cell_id: F::from_canonical_usize(r * w + c),
                        is_cell_final: F::from_bool(j == rpc - 1),
                        operand_index_base_a: F::from_canonical_usize(r * k + j * GROUP),
                        operand_index_base_b: F::from_canonical_usize(h * k + c * k + j * GROUP),
                        group_breakpoint: F::from_bool(step.breakpoint),
                        products_truncated: F::from_canonical_u32(step.products_truncated),
                        run_start: F::from_bool(run_start),
                        tile_breakpoints: F::from_canonical_u64(tile_breakpoints),
                        tile_numerator: F::from_canonical_u64(tile_numerator),
                        ..Default::default()
                    };
                    rows.push(row.into());
                }
            }
        }
        for t in 0..num_rows - self.live_rows() {
            // Padding rows carry a breakpoint of 0 and (since their predecessor is always a
            // cell-final row) a run_start of 1, matching PP2's transition formula; both are gated
            // out of the accumulators by `is_padding`, which only ever carry the running totals.
            let row = PolicyA100ColumnsView::<F> {
                cell_id: F::from_canonical_usize(h * w + t),
                is_cell_final: F::ONE,
                is_padding: F::ONE,
                run_start: F::ONE,
                tile_breakpoints: F::from_canonical_u64(tile_breakpoints),
                tile_numerator: F::from_canonical_u64(tile_numerator),
                ..Default::default()
            };
            rows.push(row.into());
        }

        // The gate slack lives on the last row (RANGE16-limbed; zero elsewhere).
        let (rho_rhs, fbp_rhs) = self.gate_constants();
        let rho_slack = (RHO_NUM * tile_numerator).wrapping_sub(rho_rhs);
        let fbp_slack = (FBP_NUM * tile_breakpoints).wrapping_sub(fbp_rhs);
        let last = num_rows - 1;
        let m = &super::columns::POLICY_A100_COL_MAP;
        rows[last][m.rho_slack_lo] = F::from_canonical_u64(rho_slack & 0xFFFF);
        rows[last][m.rho_slack_hi] = F::from_canonical_u64((rho_slack >> 16) & 0xFFFF);
        rows[last][m.fbp_slack_lo] = F::from_canonical_u64(fbp_slack & 0xFFFF);
        rows[last][m.fbp_slack_hi] = F::from_canonical_u64((fbp_slack >> 16) & 0xFFFF);
        rows
    }
}

/// Evaluates every policy constraint (degree <= 3). RANGE16 facts (the slack limbs and the
/// per-step `products_truncated`) are LUT-borne; see [`super::ctl`].
pub(crate) fn eval_policy_a100_constraints<V, S, E>(
    vars: &StarkFrame<V, S, NUM_POLICY_A100_COLUMNS, NUM_POLICY_A100_PUBLIC_INPUTS>,
    eval: &mut E,
    rho_rhs: u64,
    fbp_rhs: u64,
) where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let lv: &[V; NUM_POLICY_A100_COLUMNS] = vars.get_local_values().try_into().unwrap();
    let lv: &PolicyA100ColumnsView<V> = lv.borrow();
    let nv: &[V; NUM_POLICY_A100_COLUMNS] = vars.get_next_values().try_into().unwrap();
    let nv: &PolicyA100ColumnsView<V> = nv.borrow();

    let one = eval.i32(1);

    // ---- PP1 — per-step booleans. ----
    eval.constraint_bool(lv.is_padding);
    eval.constraint_bool(lv.group_breakpoint);
    eval.constraint_bool(lv.run_start);

    // The per-step work increment incr(x) = (1-pad)*(8*bp + 32*run_start + (1-bp)*pt).
    let incr = |eval: &mut E, x: &PolicyA100ColumnsView<V>| -> V {
        let g = eval.u64(GROUP as u64);
        let g_bp = eval.mul(g, x.group_breakpoint);
        let r = eval.u64(NOISE_RANK);
        let r_run = eval.mul(r, x.run_start);
        let not_bp = eval.sub(one, x.group_breakpoint);
        let pt_term = eval.mul(not_bp, x.products_truncated);
        let s = eval.add(g_bp, r_run);
        let s = eval.add(s, pt_term);
        let not_pad = eval.sub(one, x.is_padding);
        eval.mul(not_pad, s)
    };

    // ---- PP2 — run-start detector. First step of the trace is a cell start. ----
    let first_run = eval.sub(one, lv.group_breakpoint);
    let first_run = eval.sub(lv.run_start, first_run);
    eval.constraint_first_row(first_run);
    // Transition: a non-breakpoint step starts a run iff the previous row ended a run, i.e. it was
    // a cell-final row or itself a breakpoint. (cell-final + (1-cell-final)*prev_bp is in {0,1}.)
    let not_bp_next = eval.sub(one, nv.group_breakpoint);
    let not_final = eval.sub(one, lv.is_cell_final);
    let carry_run = eval.mul(not_final, lv.group_breakpoint);
    let can_start = eval.add(lv.is_cell_final, carry_run);
    let expect = eval.mul(not_bp_next, can_start);
    let run_diff = eval.sub(nv.run_start, expect);
    eval.constraint_transition(run_diff);

    // ---- PP3 — tile breakpoint count (inclusive running sum over non-padding rows). The first
    // trace row is always a live cell start, so its anchor needs no padding factor (keeping the
    // Lagrange-weighted first-row constraint within the degree-3 budget). ----
    let first_bp = eval.sub(lv.tile_breakpoints, lv.group_breakpoint);
    eval.constraint_first_row(first_bp);
    let not_pad_next = eval.sub(one, nv.is_padding);
    let add_bp = eval.mul(not_pad_next, nv.group_breakpoint);
    let step_bp = eval.add(lv.tile_breakpoints, add_bp);
    let step_bp = eval.sub(nv.tile_breakpoints, step_bp);
    eval.constraint_transition(step_bp);

    // ---- PP4 — tile numerator (inclusive running sum of incr). The first row is live, so its
    // anchor uses the padding-free increment `8*bp + 32*run_start + (1-bp)*pt` (degree 2). ----
    let g = eval.u64(GROUP as u64);
    let g_bp0 = eval.mul(g, lv.group_breakpoint);
    let r = eval.u64(NOISE_RANK);
    let r_run0 = eval.mul(r, lv.run_start);
    let not_bp0 = eval.sub(one, lv.group_breakpoint);
    let pt_term0 = eval.mul(not_bp0, lv.products_truncated);
    let incr0 = eval.add(g_bp0, r_run0);
    let incr0 = eval.add(incr0, pt_term0);
    let first_num = eval.sub(lv.tile_numerator, incr0);
    eval.constraint_first_row(first_num);
    let incr_next = incr(eval, nv);
    let step_num = eval.add(lv.tile_numerator, incr_next);
    let step_num = eval.sub(nv.tile_numerator, step_num);
    eval.constraint_transition(step_num);

    // ---- PP5 — rho gate on the last row: 5*numerator - 6*cells*k = rho_slack >= 0. The slack's
    // RANGE16 limbs (PP-CTL) make >= 0 and < 2^32 the only representation, so a tile with
    // rho < 1.2 has no valid nonnegative slack. ----
    let limb_shift = eval.u64(1 << 16);
    let rho_num = eval.u64(RHO_NUM);
    let lhs = eval.mul(rho_num, lv.tile_numerator);
    let rhs = eval.u64(rho_rhs);
    let lhs = eval.sub(lhs, rhs);
    let slack = eval.mad(lv.rho_slack_hi, limb_shift, lv.rho_slack_lo);
    let gate = eval.sub(lhs, slack);
    eval.constraint_last_row(gate);

    // ---- PP6 — f_bp gate on the last row: 10*breakpoints - 3*total_steps = fbp_slack >= 0. ----
    let fbp_num = eval.u64(FBP_NUM);
    let lhs = eval.mul(fbp_num, lv.tile_breakpoints);
    let rhs = eval.u64(fbp_rhs);
    let lhs = eval.sub(lhs, rhs);
    let slack = eval.mad(lv.fbp_slack_hi, limb_shift, lv.fbp_slack_lo);
    let gate = eval.sub(lhs, slack);
    eval.constraint_last_row(gate);
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for PolicyStarkA100<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, NUM_POLICY_A100_COLUMNS, NUM_POLICY_A100_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget =
        StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, NUM_POLICY_A100_COLUMNS, NUM_POLICY_A100_PUBLIC_INPUTS>;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let (rho_rhs, fbp_rhs) = self.gate_constants();
        let mut evaluator = NativeEvaluator::new(yield_constr);
        eval_policy_a100_constraints(vars, &mut evaluator, rho_rhs, fbp_rhs);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let (rho_rhs, fbp_rhs) = self.gate_constants();
        let mut evaluator = SymbolicEvaluator::new(builder, yield_constr);
        eval_policy_a100_constraints(vars, &mut evaluator, rho_rhs, fbp_rhs);
    }

    fn constraint_degree(&self) -> usize {
        3
    }

    /// The policy AIR is a looking side of its RANGE16 checks and the looked side of the
    /// census-import CTL, so it appears in the FP16 batch's CTL set.
    fn requires_ctls(&self) -> bool {
        true
    }
}

// ==================================================================================================
// Tests
// ==================================================================================================

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::{Field, PrimeField64};
    use plonky2::plonk::config::PoseidonGoldilocksConfig;
    use starky::stark_testing::{test_stark_circuit_constraints, test_stark_low_degree};

    use super::*;
    use crate::v5::api::dtype::f32_to_fp16;
    use crate::v5::api::policy::{evaluate, replay_and_evaluate};
    use crate::v5::circuit::policy_stark::columns::POLICY_A100_COL_MAP;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = GoldilocksField;
    type P = PolicyStarkA100<F, D>;

    const VECTORS: &str = include_str!("../../api/testdata/a100_dot_vectors.txt");

    fn constraints_hold(program: &P, rows: &[[F; NUM_POLICY_A100_COLUMNS]]) -> bool {
        let n = rows.len();
        (0..n).all(|i| {
            let frame = StarkFrame::from_values(&rows[i], &rows[(i + 1) % n], &[]);
            let mut consumer = ConstraintConsumer::new(
                vec![F::from_canonical_u64(2), F::from_canonical_u64(0x876543210)],
                // z_last is zero on the last row, so `constraint_transition`s (and their cyclic
                // wrap) are inactive there — matching the prover's vanishing polynomial.
                F::from_bool(i != n - 1),
                F::from_bool(i == 0),
                F::from_bool(i == n - 1),
            );
            program.eval_packed_generic(&frame, &mut consumer);
            consumer.accumulators().into_iter().all(|acc| acc == F::ZERO)
        })
    }

    /// The oracle numerator and breakpoint count for a tile, straight from `policy`'s definition.
    fn oracle_totals(a: &[u16], b: &[u16], h: usize, w: usize, k: usize) -> (u64, u64) {
        use crate::v5::api::accumulate::{GROUP, PolicyStep, a100_dot};
        let (mut numerator, mut breakpoints) = (0u64, 0u64);
        for r in 0..h {
            for c in 0..w {
                let mut census: Vec<PolicyStep> = Vec::new();
                a100_dot(&a[r * k..r * k + k], &b[c * k..c * k + k], 0.0, Some(&mut census));
                let mut in_run = false;
                for step in &census {
                    if step.breakpoint {
                        breakpoints += 1;
                        numerator += GROUP as u64;
                        in_run = false;
                    } else {
                        numerator += u64::from(step.products_truncated);
                        if !in_run {
                            numerator += NOISE_RANK;
                            in_run = true;
                        }
                    }
                }
            }
        }
        (numerator, breakpoints)
    }

    fn last_totals(program: &P, rows: &[[F; NUM_POLICY_A100_COLUMNS]]) -> (u64, u64) {
        let last = program.num_rows() - 1;
        let m = &POLICY_A100_COL_MAP;
        (rows[last][m.tile_numerator].to_canonical_u64(), rows[last][m.tile_breakpoints].to_canonical_u64())
    }

    #[test]
    fn padded_trace_matches_known_values() {
        let program = P::new(3, 5, 32); // 15 cells x 4 rows = 60 live -> 64.
        let a: Vec<u16> = (0..program.h * program.k).map(|i| f32_to_fp16(((i % 7) as f32) - 3.0).unwrap()).collect();
        let b: Vec<u16> = (0..program.w * program.k).map(|i| f32_to_fp16(((i % 5) as f32) - 2.0).unwrap()).collect();
        let rows = program.generate_trace(&a, &b);
        assert_eq!(program.live_rows(), 60);
        assert_eq!(rows.len(), 64);
        let known = program.known_values();
        for (col, poly) in known.iter().enumerate() {
            for (r, row) in rows.iter().enumerate() {
                assert_eq!(poly.values[r], row[col], "known column {col} row {r}");
            }
        }
        assert!(constraints_hold(&program, &rows) == evaluate_from_codes(&program, &a, &b).accept);
    }

    fn evaluate_from_codes(program: &P, a: &[u16], b: &[u16]) -> crate::v5::api::policy::PolicyReport {
        replay_and_evaluate(a, b, program.h, program.w, program.k).1
    }

    /// The AIR's gate decision equals `policy::evaluate` on every reference-vector tile, and its
    /// running numerator / breakpoint totals equal the oracle's — covering both accept and reject.
    #[test]
    fn gate_matches_policy_oracle_on_reference_vectors() {
        let (mut saw_accept, mut saw_reject) = (false, false);
        for line in VECTORS.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let t: Vec<&str> = line.split_whitespace().collect();
            let k: usize = t[0].parse().unwrap();
            if k % GROUP != 0 {
                continue;
            }
            let a: Vec<u16> = t[1..1 + k].iter().map(|x| x.parse().unwrap()).collect();
            let b: Vec<u16> = t[1 + k..1 + 2 * k].iter().map(|x| x.parse().unwrap()).collect();
            let program = P::new(1, 1, k);
            let rows = program.generate_trace(&a, &b);
            let report = evaluate(&replay_census(&a, &b, k), k);
            // Accumulators match the oracle totals exactly.
            assert_eq!(last_totals(&program, &rows), oracle_totals(&a, &b, 1, 1, k), "totals k={k}");
            // The gate decision matches policy::evaluate.
            assert_eq!(constraints_hold(&program, &rows), report.accept, "gate decision k={k}");
            saw_accept |= report.accept;
            saw_reject |= !report.accept;
        }
        assert!(saw_accept, "reference vectors must include an accepting 1x1 tile");
        assert!(saw_reject, "reference vectors must include a rejecting 1x1 tile");
    }

    fn replay_census(a: &[u16], b: &[u16], _k: usize) -> Vec<Vec<crate::v5::api::accumulate::PolicyStep>> {
        use crate::v5::api::accumulate::a100_dot;
        let mut steps = Vec::new();
        a100_dot(a, b, 0.0, Some(&mut steps));
        vec![steps]
    }

    /// A flat tile (all operands equal, no truncation, no breakpoints) has `f_bp = 0` and
    /// `rho < 1.2`: both gate slacks go negative, so no RANGE16-valid witness exists and the
    /// last-row gate constraints cannot vanish.
    #[test]
    fn flat_tile_is_rejected_by_the_gate() {
        let program = P::new(2, 2, 64);
        let one = f32_to_fp16(1.0).unwrap();
        let a = vec![one; program.h * program.k];
        let b = vec![one; program.w * program.k];
        let report = evaluate_from_codes(&program, &a, &b);
        assert!(!report.accept, "a flat tile must fail the policy");
        let rows = program.generate_trace(&a, &b);
        assert!(!constraints_hold(&program, &rows), "the gate must reject a flat tile");
    }

    /// Overstating the certified work after the fact — bumping the final tile numerator without a
    /// matching slack — breaks the rho gate; likewise for the breakpoint count and f_bp.
    #[test]
    fn tampered_gate_witnesses_break_constraints() {
        // Use an accepting reference-vector tile as the honest baseline.
        let mut chosen: Option<(usize, Vec<u16>, Vec<u16>)> = None;
        for line in VECTORS.lines() {
            let t: Vec<&str> = line.split_whitespace().collect();
            if t.is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let k: usize = t[0].parse().unwrap();
            if k % GROUP != 0 {
                continue;
            }
            let a: Vec<u16> = t[1..1 + k].iter().map(|x| x.parse().unwrap()).collect();
            let b: Vec<u16> = t[1 + k..1 + 2 * k].iter().map(|x| x.parse().unwrap()).collect();
            if evaluate(&replay_census(&a, &b, k), k).accept {
                chosen = Some((k, a, b));
                break;
            }
        }
        let (k, a, b) = chosen.expect("need an accepting reference tile");
        let program = P::new(1, 1, k);
        let rows = program.generate_trace(&a, &b);
        assert!(constraints_hold(&program, &rows), "honest accepting tile must satisfy the gate");
        let m = &POLICY_A100_COL_MAP;
        let last = program.num_rows() - 1;

        // Bumping a slack limb breaks the gate equality (slack no longer equals LHS).
        for col in [m.rho_slack_lo, m.fbp_slack_lo, m.tile_numerator, m.tile_breakpoints] {
            let mut t = rows.clone();
            t[last][col] += F::ONE;
            assert!(!constraints_hold(&program, &t), "tampering column {col} must break the gate/accumulator");
        }
        // Inflating the running numerator mid-trace breaks the inclusive-sum transition.
        let mid = program.live_rows() / 2;
        let mut t = rows.clone();
        t[mid][m.tile_numerator] += F::ONE;
        assert!(!constraints_hold(&program, &t), "a broken numerator running sum must be caught");
    }

    #[test]
    fn degree_is_at_most_three() {
        test_stark_low_degree::<F, P, D>(P::new(2, 2, 64)).unwrap();
    }

    #[test]
    fn circuit_constraints_match_native() {
        test_stark_circuit_constraints::<F, C, P, D>(P::new(2, 2, 64)).unwrap();
    }
}
