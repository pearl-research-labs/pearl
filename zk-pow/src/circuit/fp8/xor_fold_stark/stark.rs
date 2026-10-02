//! Proves the protocol's 32-bit XorFold lottery mixer, one matrix-output cell per live row:
//!
//! ```text
//! cell_word = cell_result_f32_lo + 2^16*cell_result_f32_hi
//! mixed     = fold_state_in*0x9E3779B1 + cell_word
//! fold_out  = rotate_left_32(low_32_bits(mixed), 13)
//! ```
//!
//! `cell_word` is the raw f32 bit pattern emitted by Matmul. `0x9E3779B1` and the 13-bit
//! rotation are fixed protocol mixing constants, not field parameters.
//!
//! # Integer arithmetic
//!
//! The multiply-add fits below `2^63.5`. Constraint group X1 uses the rotation fields for
//! its low word and two 16-bit limbs for its high word:
//!
//! ```text
//! low   = bottom19_limb_0 + 2^16*bottom19_limb_1 + 2^19*top13
//! mixed = low + 2^32*high_0 + 2^48*high_1.
//! ```
//!
//! Range checks bound `bottom19_limb_0`, `bottom19_limb_1`, and `top13` to 16, 3, and 13
//! bits respectively, so `low < 2^32`. Both high limbs are 16-bit, and `high_1` is at most
//! `0xFFFE = 2^16 - 2`. That cap keeps the reconstructed integer below the Goldilocks modulus
//! and rules out adding one modulus while preserving the same field equality.
//!
//! X2 reuses this split: `bottom19 = bottom19_limb_0 + 2^16*bottom19_limb_1`.
//! Therefore rotation by 13 is exactly
//! `fold_out = bottom19*2^13 + top13`; the numbers 13 and 19 sum to the 32-bit word width.
//!
//! # Lane chaining and table boundaries
//!
//! X3 starts each public-layout lane at state zero, carries `fold_out` into the next live row,
//! and resets at `is_lane_final`. Live rows receive
//! `(cell_id, cell_result_f32_lo, cell_result_f32_hi, cell_skips)` from the selected
//! device-specific Matmul table. Each lane-final row
//! sends `(lane_id, fold_out)` to Blake3Stark as one jackpot message word.
//!
//! The verifier recomputes `cell_id`, `lane_id`, `is_lane_final`, and `is_pad` from the public
//! lane assignment and checks their openings. Padding rows set `is_pad = 1`, contribute no
//! cell subtotal, carry the final running skip count, and participate in neither cross-table
//! lookup.
//!
//! # Global unpredictability census
//!
//! For each output cell, Matmul classifies all `k` summands against the device's M/Z
//! unpredictability threshold and exports their subtotal as `cell_skips` through the
//! cell-results CTL. X4 computes the inclusive sum of those subtotals in XorFold trace order.
//! Because the lane assignment contains every output cell exactly once, the last row's
//! `running_skips` is the tile-wide count.
//!
//! The sole public input is the verifier-derived allowance `skip_limit = floor(k*h*w/20)`.
//! On the final trace row, X4 enforces
//!
//! ```text
//! skip_limit - running_skips = skip_gate_slack_lo + 2^16*skip_gate_slack_hi.
//! ```
//!
//! RC16 lookups bound both slack limbs, turning that field equality into the integer
//! inequality `running_skips <= skip_limit`. Padding rows keep the final running count
//! unchanged so the same terminal gate works whether or not `h*w` is a power of two.

use core::borrow::{Borrow, BorrowMut};
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

use super::columns::{XorFoldColumnsView, NUM_XOR_FOLD_COLUMNS, NUM_XOR_FOLD_PUBLIC_INPUTS, SKIP_LIMIT_PUBLIC_INPUT};
use crate::api::layout::JACKPOT_ENTRIES as XOR_FOLD_LANES;
use crate::circuit::utils::evaluator::Evaluator;
use crate::circuit::utils::native_evaluator::NativeEvaluator;
use crate::circuit::utils::symbolic_evaluator::SymbolicEvaluator;

/// The protocol's fixed odd mixing multiplier (the golden-ratio-derived XorFold constant).
const GOLDEN: u64 = 0x9E3779B1;

/// The committed lane layout: `lanes[j]` lists lane `j`'s cell ids in fold order
/// (`crate::api::layout::lane_assignment`).
#[derive(Clone, Debug)]
pub struct XorFoldProgram {
    pub lanes: Vec<Vec<usize>>,
    /// Exact consolidated-policy allowance `floor(k*h*w/20)`.
    pub skip_limit: u64,
}

impl XorFoldProgram {
    /// One row per folded cell (`h*w` live rows).
    pub fn live_rows(&self) -> usize {
        self.lanes.iter().map(Vec::len).sum()
    }

    /// Trace height: the live rows padded to the next power of two with all-zero
    /// `IS_PAD = 1` rows.
    pub fn num_rows(&self) -> usize {
        self.live_rows().next_power_of_two()
    }

    /// `cell_words[c]` is cell `c`'s f32 bit pattern and `cell_skips[c]` is its certified
    /// summand-skip subtotal. Matmul exposes both on its `IS_CELL_FINAL` row.
    pub fn generate_trace<F: RichField>(
        &self,
        cell_words: &[u32],
        cell_skips: &[u64],
    ) -> (Vec<[F; NUM_XOR_FOLD_COLUMNS]>, [F; NUM_XOR_FOLD_PUBLIC_INPUTS]) {
        assert_eq!(self.lanes.len(), XOR_FOLD_LANES, "the lottery block has 16 lanes");
        assert_eq!(cell_skips.len(), cell_words.len(), "one skip count per folded cell");
        // Each cell folded exactly once; no lane empty (an empty lane has no IS_LANE_FINAL row
        // and the Blake3 lottery channel cannot balance).
        let mut seen = vec![false; cell_words.len()];
        for lane in &self.lanes {
            assert!(!lane.is_empty(), "empty lane");
            for &c in lane {
                assert!(!std::mem::replace(&mut seen[c], true), "cell {c} folded twice");
            }
        }
        assert!(seen.iter().all(|&s| s), "not every cell is folded");

        let mut rows: Vec<[F; NUM_XOR_FOLD_COLUMNS]> = Vec::with_capacity(self.num_rows());
        let mut running_skips = 0u64;
        for (j, lane) in self.lanes.iter().enumerate() {
            let mut state = 0u32;
            for (step, &cell) in lane.iter().enumerate() {
                // The leading four columns are class (a) — keep in sync with known_values.
                let w = cell_words[cell];
                running_skips += cell_skips[cell];
                let t = state as u64 * GOLDEN + w as u64; // < 2^63.5: exact in the field too
                let (lo, hi) = (t as u32, (t >> 32) as u32);
                debug_assert!(hi >> 16 <= 0x9E38, "honest top limb under the 0xFFFE cap");
                let row = XorFoldColumnsView::<F> {
                    cell_id: F::from_canonical_usize(cell),
                    lane_id: F::from_canonical_usize(j),
                    is_lane_final: F::from_bool(step == lane.len() - 1),
                    is_pad: F::ZERO,
                    cell_result_f32_lo: F::from_canonical_u32(w & 0xFFFF),
                    cell_result_f32_hi: F::from_canonical_u32(w >> 16),
                    cell_skips: F::from_canonical_u64(cell_skips[cell]),
                    running_skips: F::from_canonical_u64(running_skips),
                    skip_gate_slack_lo: F::ZERO,
                    skip_gate_slack_hi: F::ZERO,
                    fold_state_in: F::from_canonical_u32(state),
                    muladd_high_limb_0: F::from_canonical_u32(hi & 0xFFFF),
                    muladd_high_limb_1: F::from_canonical_u32(hi >> 16),
                    rotation_input_top13: F::from_canonical_u32(lo >> 19),
                    rotation_input_bottom19_limb_0: F::from_canonical_u32(lo & 0xFFFF),
                    rotation_input_bottom19_limb_1: F::from_canonical_u32((lo >> 16) & 0x7),
                };
                rows.push(row.into());
                state = lo.rotate_left(13);
            }
        }
        // Pad to the next power of two: mixer and cell-subtotal columns are zero, while
        // RUNNING_SKIPS carries the final count to the last-row budget gate. X1-X3 hold on
        // zeros (the last live row is lane-final, so the pad chain starts and stays at state 0).
        assert!(
            running_skips <= self.skip_limit,
            "policy-rejected witness: skippable summands exceed the census budget"
        );
        let pad_row = XorFoldColumnsView::<F> {
            is_pad: F::ONE,
            running_skips: F::from_canonical_u64(running_skips),
            ..XorFoldColumnsView::default()
        };
        rows.resize(self.num_rows(), pad_row.into());
        let slack = self.skip_limit - running_skips;
        let last: &mut XorFoldColumnsView<F> = rows.last_mut().expect("at least one row").borrow_mut();
        last.skip_gate_slack_lo = F::from_canonical_u64(slack & 0xFFFF);
        last.skip_gate_slack_hi = F::from_canonical_u64(slack >> 16);
        (rows, [F::from_canonical_u64(self.skip_limit)])
    }

    /// The class (a) ("known") column values — the leading
    /// [`NUM_XOR_FOLD_KNOWN_COLUMNS`](super::columns::NUM_XOR_FOLD_KNOWN_COLUMNS) trace columns
    /// in their `columns.rs` order (`CELL_ID`, `LANE_ID`, `IS_LANE_FINAL`, `IS_PAD`), pure
    /// functions of the committed lane layout. Bit-exact with [`Self::generate_trace`]'s fill;
    /// the batch verifier recomputes exactly this and checks the trace openings against it
    /// (`starky`'s `BatchKnownColumns`).
    pub fn known_values<F: RichField>(&self) -> Vec<PolynomialValues<F>> {
        let num_rows = self.num_rows();
        let mut cell_id = Vec::with_capacity(num_rows);
        let mut lane_id = Vec::with_capacity(num_rows);
        let mut is_lane_final = Vec::with_capacity(num_rows);
        for (j, lane) in self.lanes.iter().enumerate() {
            for (step, &cell) in lane.iter().enumerate() {
                cell_id.push(F::from_canonical_usize(cell));
                lane_id.push(F::from_canonical_usize(j));
                is_lane_final.push(F::from_bool(step == lane.len() - 1));
            }
        }
        let mut is_pad = vec![F::ZERO; cell_id.len()];
        for col in [&mut cell_id, &mut lane_id, &mut is_lane_final] {
            col.resize(num_rows, F::ZERO);
        }
        is_pad.resize(num_rows, F::ONE);
        [cell_id, lane_id, is_lane_final, is_pad]
            .into_iter()
            .map(PolynomialValues::new)
            .collect()
    }
}

/// Evaluates the X1-X4 constraint groups (module docs). The 10 RC16 facts (limbs, split
/// bounds, the `MULADD_HIGH_LIMB_1` canonicity cap, and budget-slack limbs) live in
/// `super::ctl::xor_fold_lut_lookups`.
pub(crate) fn eval_xor_fold_constraints<V, S, E>(
    vars: &StarkFrame<V, S, NUM_XOR_FOLD_COLUMNS, NUM_XOR_FOLD_PUBLIC_INPUTS>,
    eval: &mut E,
) where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let lv: &[V; NUM_XOR_FOLD_COLUMNS] = vars.get_local_values().try_into().unwrap();
    let lv: &XorFoldColumnsView<V> = lv.borrow();
    let nv: &[V; NUM_XOR_FOLD_COLUMNS] = vars.get_next_values().try_into().unwrap();
    let nv: &XorFoldColumnsView<V> = nv.borrow();

    let one = eval.u64(1);
    let two16 = eval.u64(1 << 16);

    // The rotation fields directly represent the low 32 bits. Their 16-, 3-, and 13-bit
    // bounds imply 0 <= low_word < 2^32, without separate low-word limb columns.
    let two19 = eval.u64(1 << 19);
    let bottom19_high = eval.mul(two16, lv.rotation_input_bottom19_limb_1);
    let bottom19 = eval.add(lv.rotation_input_bottom19_limb_0, bottom19_high);
    let low_word = eval.mad(lv.rotation_input_top13, two19, bottom19);

    // X1 — multiply-add split: FOLD_STATE_IN*0x9E3779B1 + W = HI*2^32 + LO over the integers.
    // The honest value is < 2^63.5 (no field wrap); the MULADD_HIGH_LIMB_1 <= 0xFFFE cap
    // (RC16 inventory) keeps the limb side below p too — without it every row with value
    // < 2^32 (all lane starts) would admit the `+p` limb alias, forging the folded word.
    let golden = eval.u64(GOLDEN);
    let two32 = eval.u64(1 << 32);
    let w_hi = eval.mul(two16, lv.cell_result_f32_hi);
    let w = eval.add(lv.cell_result_f32_lo, w_hi);
    let muladd = eval.mad(lv.fold_state_in, golden, w);
    let hi_1 = eval.mul(two16, lv.muladd_high_limb_1);
    let hi = eval.add(lv.muladd_high_limb_0, hi_1);
    let limbs = eval.mad(hi, two32, low_word);
    eval.constraint_eq(muladd, limbs);

    // X2 — rotating the low word left by 13 gives BOTTOM19*2^13 + TOP13 (an expression).
    let two13 = eval.u64(1 << 13);
    let fold_out = eval.mad(bottom19, two13, lv.rotation_input_top13);

    // X3 — chaining: 0 entering the first row; a lane-final row resets the next row's state;
    // otherwise the state chains. The next-row constraints are plain (cyclic): on the wrap pair
    // the last row is lane-final — a class (a) fact carried by the known-column binding
    // (IS_LANE_FINAL is verifier-recomputed and checked against the trace openings,
    // `super::super::known_values`; the same binding gives its booleanness, which X3 and the
    // lottery channel's filter rely on) — so the wrap instance is row 0's reset.
    eval.constraint_first_row(lv.fold_state_in);
    let reset = eval.mul(lv.is_lane_final, nv.fold_state_in);
    eval.constraint(reset);
    let not_final = eval.sub(one, lv.is_lane_final);
    let chain_diff = eval.sub(nv.fold_state_in, fold_out);
    let chain = eval.mul(not_final, chain_diff);
    eval.constraint(chain);

    // X4 — tile-wide unpredictability census. The cell-results CTL binds CELL_SKIPS on every
    // live row to Matmul's per-cell subtotal. IS_PAD rows contribute zero, and the recurrence
    // carries the inclusive sum to the final row.
    let pad_skips = eval.mul(lv.is_pad, lv.cell_skips);
    eval.constraint(pad_skips);
    let first_running = eval.sub(lv.running_skips, lv.cell_skips);
    eval.constraint_first_row(first_running);
    let running_step = eval.sub(nv.running_skips, lv.running_skips);
    let running_step = eval.sub(running_step, nv.cell_skips);
    eval.constraint_transition(running_step);

    // The verifier derives SKIP_LIMIT = floor(k*h*w/20). RC16 bounds both slack limbs, so
    // this final-row field equality proves the corresponding integer inequality.
    let skip_limit = eval.scalar(vars.get_public_inputs()[SKIP_LIMIT_PUBLIC_INPUT]);
    let slack = eval.mad(lv.skip_gate_slack_hi, two16, lv.skip_gate_slack_lo);
    let budget = eval.sub(skip_limit, lv.running_skips);
    let budget = eval.sub(budget, slack);
    eval.constraint_last_row(budget);
}

/// XorFoldStark. A CTL party of the fp8 batch (`requires_ctls()`): its proofs carry the
/// cross-table openings of the channels declared in `super::ctl`, so the batch driver is the
/// only supported proving path — there is no standalone uni-STARK proof object.
#[derive(Clone, Debug)]
pub struct XorFoldStark<F: RichField + Extendable<D>, const D: usize> {
    pub program: XorFoldProgram,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> XorFoldStark<F, D> {
    pub fn new(program: XorFoldProgram) -> Self {
        Self {
            program,
            _phantom: PhantomData,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for XorFoldStark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, NUM_XOR_FOLD_COLUMNS, NUM_XOR_FOLD_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget =
        StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, NUM_XOR_FOLD_COLUMNS, NUM_XOR_FOLD_PUBLIC_INPUTS>;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let mut evaluator = NativeEvaluator::new(yield_constr);
        eval_xor_fold_constraints(vars, &mut evaluator);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let mut evaluator = SymbolicEvaluator::new(builder, yield_constr);
        eval_xor_fold_constraints(vars, &mut evaluator);
    }

    fn constraint_degree(&self) -> usize {
        3
    }

    // Party to the cell-results and lottery-words channels plus the committed LUT channels
    // (declared in `super::ctl`).
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
    use starky::util::trace_rows_to_poly_values;

    use super::super::columns::XOR_FOLD_COL_MAP;
    use super::super::ctl::xor_fold_lut_lookups;
    use super::*;
    use crate::api::fp8::utils::xor_fold_extract;
    use crate::api::layout::{lane_assignment, AxisPattern, DimType};
    use crate::circuit::fp8::matmul_b200_stark::columns::MATMUL_B200_COL_MAP;
    use crate::circuit::fp8::matmul_b200_stark::MatmulStarkB200;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = GoldilocksField;
    type S = XorFoldStark<F, D>;

    fn to_u64(x: F) -> u64 {
        x.to_canonical_u64()
    }

    /// The committed layout of `utils.rs`'s reference-vector test: an 8x8 merged tile, 16 lanes
    /// of 4 cells each (per axis: fold offsets `{0, 1}`, blake offsets `{0, 2, 4, 6}`).
    fn test_layout() -> (Vec<f32>, Vec<Vec<usize>>) {
        let axis = AxisPattern::new(&[(2, DimType::Fold), (4, DimType::Blake)]).unwrap();
        let base = [1.0f32, -2.5, 0.0, 3.171875, 1e-3, -0.0, 448.0, 2.0, 0.5, -1.0];
        let tile: Vec<f32> = (0..64).map(|i| base[i % base.len()]).collect();
        (tile, lane_assignment(&axis, &axis))
    }

    fn test_trace() -> (
        XorFoldProgram,
        Vec<[F; NUM_XOR_FOLD_COLUMNS]>,
        [F; NUM_XOR_FOLD_PUBLIC_INPUTS],
    ) {
        let (tile, lanes) = test_layout();
        let skips = vec![1u64; tile.len()];
        let program = XorFoldProgram {
            lanes,
            skip_limit: skips.iter().sum(),
        };
        let words: Vec<u32> = tile.iter().map(|x| x.to_bits()).collect();
        let (rows, pis) = program.generate_trace::<F>(&words, &skips);
        (program, rows, pis)
    }

    fn constraints_violated(stark: &S, rows: &[[F; NUM_XOR_FOLD_COLUMNS]], pis: &[F; NUM_XOR_FOLD_PUBLIC_INPUTS]) -> bool {
        let n = rows.len();
        (0..n).any(|i| {
            let frame = StarkFrame::from_values(&rows[i], &rows[(i + 1) % n], pis);
            let mut consumer = ConstraintConsumer::new(
                vec![F::from_canonical_u64(2), F::from_canonical_u64(0x876543210)],
                // `constraint_transition` is active on every row except the last. Plain
                // constraints remain cyclic and still inspect the last -> first frame.
                F::from_bool(i != n - 1),
                F::from_bool(i == 0),
                F::from_bool(i == n - 1),
            );
            stark.eval_packed_generic(&frame, &mut consumer);
            consumer.accumulators().iter().any(|&acc| acc != F::ZERO)
        })
    }

    fn lut_keys_are_in_domain(rows: &[[F; NUM_XOR_FOLD_COLUMNS]]) -> bool {
        let polys = trace_rows_to_poly_values(rows.to_vec());
        xor_fold_lut_lookups::<F>()
            .iter()
            .all(|lookup| (0..rows.len()).all(|r| lookup.keys[0].eval_table(&polys, r, &[]).to_canonical_u64() < 1 << 16))
    }

    #[test]
    fn honest_trace_satisfies_all_constraints() {
        // Plain X3 constraints inspect the last -> first wrap as row 0's reset instance; X4's
        // running-sum transition correctly stops at the last row.
        let (program, rows, pis) = test_trace();
        let last: &XorFoldColumnsView<F> = rows.last().unwrap().borrow();
        assert_eq!(last.skip_gate_slack_lo, F::ZERO);
        assert_eq!(last.skip_gate_slack_hi, F::ZERO);
        assert!(!constraints_violated(&S::new(program), &rows, &pis));
    }

    #[test]
    fn padded_trace_satisfies_all_constraints() {
        // A non-power-of-two live count: 16 lanes x 3 cells = 48 live rows, padded to 64. The
        // wrap pair is (pad, row 0), exercising the pad chain closing on the FOLD_STATE_IN = 0
        // anchor and the padded last-row anchor.
        let program = XorFoldProgram {
            lanes: (0..16).map(|j| vec![3 * j, 3 * j + 1, 3 * j + 2]).collect(),
            skip_limit: 48,
        };
        let words: Vec<u32> = (0..48u32).map(|i| f32::to_bits(i as f32 - 20.5)).collect();
        let (rows, pis) = program.generate_trace::<F>(&words, &vec![1; words.len()]);
        assert_eq!(rows.len(), 64, "48 live rows pad to 64");
        let known = program.known_values::<F>();
        assert!(known.iter().all(|c| c.len() == 64));
        // The known columns are bit-exact with the trace fill (the batch verifier's check).
        for (c, col) in known.iter().enumerate() {
            for (r, row) in rows.iter().enumerate() {
                assert_eq!(col.values[r], row[c], "known column {c} row {r}");
            }
        }
        for row in &rows[48..] {
            let v: &XorFoldColumnsView<F> = row.borrow();
            assert_eq!(v.cell_skips, F::ZERO, "padding must not add skips");
            assert_eq!(
                v.running_skips,
                F::from_canonical_u64(48),
                "padding must carry the final total"
            );
        }
        assert!(!constraints_violated(&S::new(program), &rows, &pis));
    }

    #[test]
    fn census_gate_accepts_two_limb_slack() {
        let (tile, lanes) = test_layout();
        let skips = vec![1u64; tile.len()];
        let total: u64 = skips.iter().sum();
        let slack = (1 << 16) + 2;
        let program = XorFoldProgram {
            lanes,
            skip_limit: total + slack,
        };
        let words: Vec<u32> = tile.iter().map(|x| x.to_bits()).collect();
        let (rows, pis) = program.generate_trace::<F>(&words, &skips);
        let last: &XorFoldColumnsView<F> = rows.last().unwrap().borrow();
        assert_eq!(last.running_skips, F::from_canonical_u64(total));
        assert_eq!(last.skip_gate_slack_lo, F::from_canonical_u64(2));
        assert_eq!(last.skip_gate_slack_hi, F::ONE);
        assert!(!constraints_violated(&S::new(program), &rows, &pis));
        assert!(lut_keys_are_in_domain(&rows));
    }

    #[test]
    fn census_gate_rejects_bad_limit_running_sum_and_slack() {
        let (program, rows, pis) = test_trace();
        let stark = S::new(program);

        let mut too_small = pis;
        too_small[SKIP_LIMIT_PUBLIC_INPUT] -= F::ONE;
        assert!(constraints_violated(&stark, &rows, &too_small), "lowered public limit passed");

        let mut bad_running = rows.clone();
        bad_running[1][XOR_FOLD_COL_MAP.running_skips] += F::ONE;
        assert!(
            constraints_violated(&stark, &bad_running, &pis),
            "tampered running sum passed"
        );

        let mut bad_slack = rows.clone();
        bad_slack.last_mut().unwrap()[XOR_FOLD_COL_MAP.skip_gate_slack_lo] += F::ONE;
        assert!(constraints_violated(&stark, &bad_slack, &pis), "tampered budget slack passed");

        // A field element of -1 can satisfy the final equality for the lowered limit, but it
        // is not a 16-bit nonnegative integer and the RC16 lookup rejects it.
        let mut wrapped_slack = rows.clone();
        wrapped_slack.last_mut().unwrap()[XOR_FOLD_COL_MAP.skip_gate_slack_lo] = -F::ONE;
        assert!(
            !constraints_violated(&stark, &wrapped_slack, &too_small),
            "the polynomial equality alone should admit the wrapped slack"
        );
        assert!(
            !lut_keys_are_in_domain(&wrapped_slack),
            "RC16 must reject a field-wrapped negative slack"
        );
    }

    #[test]
    fn trace_is_bit_exact_vs_native_xor_fold_extract() {
        let (tile, lanes) = test_layout();
        let (program, rows, _) = test_trace();
        let extracted = xor_fold_extract(&tile, &lanes);
        let mut finals = 0;
        for row in &rows {
            let v: &XorFoldColumnsView<F> = row.borrow();
            if v.is_lane_final != F::ONE {
                continue;
            }
            finals += 1;
            // FOLD_OUT recomposed from the rotation split = the extractor's LE lane word.
            let bot19 = to_u64(v.rotation_input_bottom19_limb_0) + (to_u64(v.rotation_input_bottom19_limb_1) << 16);
            let fold_out = (bot19 << 13) + to_u64(v.rotation_input_top13);
            let lane = to_u64(v.lane_id) as usize;
            let expected = u32::from_le_bytes(extracted[lane * 4..lane * 4 + 4].try_into().unwrap());
            assert_eq!(fold_out as u32, expected, "lane {lane}");
        }
        assert_eq!(finals, XOR_FOLD_LANES);
        drop(program);
    }

    #[test]
    fn cell_results_multiset_balances_with_matmul() {
        // The cell-results channel end to end: Matmul's looked tuples (CELL_ID, LO, HI on
        // IS_CELL_FINAL rows) must equal XorFold's looking tuples (every row) as multisets when
        // XorFold folds exactly Matmul's outputs.
        let matmul = MatmulStarkB200::<F, D>::new(4, 4, 128);
        // Same code recipe as the Matmul tests: pool codes with a zero sprinkled in.
        const POOL: [u8; 8] = [0x38, 0x40, 0xB9, 0x3A, 0xC1, 0x3B, 0xBA, 0x42];
        let codes = |len: usize, salt: u64, zero_every: usize| -> Vec<u8> {
            (0..len)
                .map(|i| {
                    if i % zero_every == 0 {
                        0
                    } else {
                        POOL[((i as u64).wrapping_mul(salt) ^ (i as u64 >> 3)) as usize % POOL.len()]
                    }
                })
                .collect()
        };
        let a = codes(matmul.h * matmul.k, 0x9E3779B97F4A7C15, 8);
        let b = codes(matmul.w * matmul.k, 0xC2B2AE3D27D4EB4F, 11);
        let lambdas = vec![300u64; matmul.h * matmul.k];
        let (matmul_rows, _) = matmul.generate_trace(&a, &b, &lambdas, &lambdas);

        // The looked filter is IS_CELL_FINAL * (1 - IS_PADDING): the phantom padding cells'
        // finals emit no word.
        let mut looked: Vec<(u64, u64, u64, u64)> = matmul_rows
            .iter()
            .filter(|row| row[MATMUL_B200_COL_MAP.is_cell_final] == F::ONE && row[MATMUL_B200_COL_MAP.is_padding] == F::ZERO)
            .map(|row| {
                (
                    to_u64(row[MATMUL_B200_COL_MAP.cell_id]),
                    to_u64(row[MATMUL_B200_COL_MAP.cell_result_f32_lo]),
                    to_u64(row[MATMUL_B200_COL_MAP.cell_result_f32_hi]),
                    to_u64(row[MATMUL_B200_COL_MAP.cell_skips]),
                )
            })
            .collect();

        // Fold those 16 cells, one per lane (16 = h*w keeps every lane nonempty).
        let mut words = vec![0u32; 16];
        let mut skips = vec![0u64; 16];
        for &(cell, lo, hi, cell_skips) in &looked {
            words[cell as usize] = (lo | (hi << 16)) as u32;
            skips[cell as usize] = cell_skips;
        }
        let program = XorFoldProgram {
            lanes: (0..16).map(|j| vec![j]).collect(),
            skip_limit: skips.iter().sum(),
        };
        let (rows, pis) = program.generate_trace::<F>(&words, &skips);
        assert!(!constraints_violated(&S::new(program), &rows, &pis));

        let mut looking: Vec<(u64, u64, u64, u64)> = rows
            .iter()
            .filter(|row| {
                let v: &XorFoldColumnsView<F> = (*row).borrow();
                v.is_pad == F::ZERO
            })
            .map(|row| {
                let v: &XorFoldColumnsView<F> = row.borrow();
                (
                    to_u64(v.cell_id),
                    to_u64(v.cell_result_f32_lo),
                    to_u64(v.cell_result_f32_hi),
                    to_u64(v.cell_skips),
                )
            })
            .collect();
        looked.sort_unstable();
        looking.sort_unstable();
        assert_eq!(looked, looking, "cell-results channel out of balance");
    }

    #[test]
    fn lut_inventory_holds_on_honest_trace_and_catches_the_alias() {
        let (_, rows, _) = test_trace();
        assert!(lut_keys_are_in_domain(&rows), "honest trace has an out-of-range RC16 key");

        // The `+p` limb alias: at a lane start (state 0, t = W < 2^32) rewrite the limbs as
        // t + p = (t + 1) + 0xFFFFFFFF*2^32 and re-split consistently. X1/X2 still vanish in the
        // field — only the RC16(MULADD_HIGH_LIMB_1 + 1) canonicity cap catches the forgery.
        let mut forged = rows.clone();
        {
            let v: &mut XorFoldColumnsView<F> = forged[0].borrow_mut();
            assert_eq!(v.fold_state_in, F::ZERO);
            let t = to_u64(v.cell_result_f32_lo) | (to_u64(v.cell_result_f32_hi) << 16);
            let lo = (t + 1) as u32;
            v.muladd_high_limb_0 = F::from_canonical_u32(0xFFFF);
            v.muladd_high_limb_1 = F::from_canonical_u32(0xFFFF);
            v.rotation_input_top13 = F::from_canonical_u32(lo >> 19);
            v.rotation_input_bottom19_limb_0 = F::from_canonical_u32(lo & 0xFFFF);
            v.rotation_input_bottom19_limb_1 = F::from_canonical_u32((lo >> 16) & 0x7);
        }
        // Not caught in-AIR on the forged row itself (the chain then breaks downstream unless
        // the whole lane is rewritten — do that to isolate the alias)...
        let (tile, lanes) = test_layout();
        let words: Vec<u32> = tile.iter().map(|x| x.to_bits()).collect();
        let lane0 = &lanes[0];
        let mut state = ((words[lane0[0]] as u64 + 1) as u32).rotate_left(13);
        for (step, &cell) in lane0.iter().enumerate().skip(1) {
            let t = state as u64 * 0x9E3779B1 + words[cell] as u64;
            let (lo, hi) = (t as u32, (t >> 32) as u32);
            let v: &mut XorFoldColumnsView<F> = forged[step].borrow_mut();
            v.fold_state_in = F::from_canonical_u32(state);
            v.muladd_high_limb_0 = F::from_canonical_u32(hi & 0xFFFF);
            v.muladd_high_limb_1 = F::from_canonical_u32(hi >> 16);
            v.rotation_input_top13 = F::from_canonical_u32(lo >> 19);
            v.rotation_input_bottom19_limb_0 = F::from_canonical_u32(lo & 0xFFFF);
            v.rotation_input_bottom19_limb_1 = F::from_canonical_u32((lo >> 16) & 0x7);
            state = lo.rotate_left(13);
        }
        let (program, _, pis) = test_trace();
        assert!(
            !constraints_violated(&S::new(program), &forged, &pis),
            "the alias must satisfy X1-X3 — it is the RC16 cap's job"
        );
        assert!(
            !lut_keys_are_in_domain(&forged),
            "the RC16(MULADD_HIGH_LIMB_1 + 1) cap must catch the +p alias"
        );
    }

    #[test]
    fn rotation_fields_reject_modular_aliases_of_the_low_word() {
        // Each lane has one row, so a forged rotation cannot be rejected merely by a
        // downstream state mismatch. These aliases preserve X1 and the scaled range check;
        // only the unscaled range check rejects the forged rotation field.
        let cases = [
            (
                XOR_FOLD_COL_MAP.rotation_input_top13,
                7u32 << 16,
                F::from_canonical_u64(7) / F::from_canonical_u64(8),
            ),
            (
                XOR_FOLD_COL_MAP.rotation_input_bottom19_limb_1,
                8191u32 << 3,
                F::from_canonical_u64(8191) / F::from_canonical_u64(1 << 13),
            ),
        ];
        for (column, word, alias) in cases {
            let program = XorFoldProgram {
                lanes: (0..16).map(|j| vec![j]).collect(),
                skip_limit: 0,
            };
            let (mut rows, pis) = program.generate_trace::<F>(&[word; 16], &[0; 16]);
            let row: &mut XorFoldColumnsView<F> = rows[0].borrow_mut();
            row.rotation_input_top13 = F::ZERO;
            row.rotation_input_bottom19_limb_0 = F::ZERO;
            row.rotation_input_bottom19_limb_1 = F::ZERO;
            rows[0][column] = alias;
            assert!(alias.to_canonical_u64() >= 1 << 16);
            assert!(!constraints_violated(&S::new(program), &rows, &pis));

            let polys = trace_rows_to_poly_values(rows);
            let failing_checks: Vec<_> = xor_fold_lut_lookups::<F>()
                .iter()
                .filter(|lookup| lookup.keys[0].eval_table(&polys, 0, &[]).to_canonical_u64() >= 1 << 16)
                .map(|lookup| lookup.keys[0].eval_table(&polys, 0, &[]))
                .collect();
            assert_eq!(failing_checks, vec![alias], "the unscaled check must reject the alias");
        }
    }

    #[test]
    fn tampered_traces_fail() {
        let (program, rows, pis) = test_trace();
        let stark = S::new(program);
        let cases = [
            // A lane's chain: the next row's state no longer matches FOLD_OUT.
            ("fold_state_in", XOR_FOLD_COL_MAP.fold_state_in),
            // The folded word itself (X1 breaks).
            ("cell_result_f32_lo", XOR_FOLD_COL_MAP.cell_result_f32_lo),
            // Each part of the low word is bound by X1.
            ("rotation_input_top13", XOR_FOLD_COL_MAP.rotation_input_top13),
            (
                "rotation_input_bottom19_limb_0",
                XOR_FOLD_COL_MAP.rotation_input_bottom19_limb_0,
            ),
            (
                "rotation_input_bottom19_limb_1",
                XOR_FOLD_COL_MAP.rotation_input_bottom19_limb_1,
            ),
            // A mid-lane final flag (0 -> 1): the reset forces the next state to 0 (false here).
            ("is_lane_final", XOR_FOLD_COL_MAP.is_lane_final),
        ];
        for (name, col) in cases {
            let mut forged = rows.clone();
            // Row 1 is mid-lane (lanes have 4 cells) and has a nonzero next state.
            forged[1][col] += F::ONE;
            assert!(constraints_violated(&stark, &forged, &pis), "{name} tamper undetected");
        }
        // Clearing the trace-final flag: X3's wrap instance becomes a chain pair, forcing
        // FOLD_STATE_IN(0) = FOLD_OUT(last) — the anchored 0 vs. the lane-15 word — so the AIR
        // still rejects this trace. (The flag is class (a) regardless: any divergence, even
        // one engineered so FOLD_OUT(last) = 0, is caught by the batch verifier's
        // known-column recompute.)
        let mut forged = rows.clone();
        let last: &mut XorFoldColumnsView<F> = forged.last_mut().unwrap().borrow_mut();
        last.is_lane_final = F::ZERO;
        assert!(
            constraints_violated(&stark, &forged, &pis),
            "cleared trace-final flag undetected"
        );
        let known = stark.program.known_values::<F>();
        assert_ne!(
            known[XOR_FOLD_COL_MAP.is_lane_final].values[rows.len() - 1],
            forged[rows.len() - 1][XOR_FOLD_COL_MAP.is_lane_final],
            "IS_LANE_FINAL must diverge from the verifier's recompute"
        );
    }

    #[test]
    fn degree_is_at_most_three() {
        let (program, _, _) = test_trace();
        test_stark_low_degree::<F, S, D>(S::new(program)).unwrap();
    }

    #[test]
    fn circuit_constraints_match_native() {
        let (program, _, _) = test_trace();
        test_stark_circuit_constraints::<F, C, S, D>(S::new(program)).unwrap();
    }

    // No standalone prove/verify smoke test: this table is a CTL party (`requires_ctls`), so a
    // proof without the cross-table argument is not a supported object. The end-to-end proving
    // path is covered by `fp8::driver::tests::batch_proof_roundtrips_and_rejects_tampering`.
}
