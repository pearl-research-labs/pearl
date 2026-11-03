//! Proves the protocol's 32-bit XorFold lottery mixer for FP16, one matmul-output cell per live
//! row:
//!
//! ```text
//! cell_word = cell_result_f32_lo + 2^16*cell_result_f32_hi
//! mixed     = fold_state_in*0x9E3779B1 + cell_word
//! fold_out  = rotate_left_32(low_32_bits(mixed), 13)
//! ```
//!
//! `cell_word` is the raw f32 bit pattern emitted by the matmul (`cell_result_f32_lo/hi`).
//! `0x9E3779B1` and the 13-bit rotation are fixed protocol mixing constants, shared with the
//! plaintext extractor [`crate::v4::api::utils::xor_fold_extract`].
//!
//! This is the FP16 analogue of [`crate::v4::circuit::xor_fold_stark`] with the FP8
//! consolidated-policy skip census removed (FP16 scores its jackpot in
//! [`crate::v5::circuit::policy_stark`]). Only three constraint groups remain:
//!
//! * **X1** — multiply-add split: `FOLD_STATE_IN*0x9E3779B1 + W = HI*2^32 + LO` over the integers.
//!   The honest value is `< 2^63.5` (no field wrap); the `MULADD_HIGH_LIMB_1 <= 0xFFFE` cap
//!   (RC16 inventory) keeps the limb side below `p` too, ruling out the `+p` limb alias.
//! * **X2** — `rotl32` by 13 is a pure re-split of the low word: `LO = TOP13*2^19 + BOT19`, so
//!   `FOLD_OUT = BOT19*2^13 + TOP13`.
//! * **X3** — lane chaining: state 0 entering the first row, a lane-final row resets the next
//!   row's state, otherwise the state chains `FOLD_OUT`.
//!
//! The verifier recomputes `cell_id`, `lane_id`, `is_lane_final`, and `is_pad` from the public
//! lane assignment and checks their openings. The 10 RC16 facts (limbs, split bounds, the
//! `MULADD_HIGH_LIMB_1` canonicity cap) live in [`super::ctl::xor_fold_lut_lookups`].

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

use super::columns::{NUM_XOR_FOLD_COLUMNS, NUM_XOR_FOLD_PUBLIC_INPUTS, XorFoldColumnsView};
use crate::v4::api::layout::JACKPOT_ENTRIES as XOR_FOLD_LANES;
use crate::v4::circuit::utils::evaluator::Evaluator;
use crate::v4::circuit::utils::native_evaluator::NativeEvaluator;
use crate::v4::circuit::utils::symbolic_evaluator::SymbolicEvaluator;

/// The protocol's fixed odd mixing multiplier (the golden-ratio-derived XorFold constant).
const GOLDEN: u64 = 0x9E3779B1;

/// The committed lane layout: `lanes[j]` lists lane `j`'s cell ids in fold order
/// (`crate::v4::api::layout::lane_assignment`).
#[derive(Clone, Debug)]
pub struct XorFoldProgram {
    pub lanes: Vec<Vec<usize>>,
}

impl XorFoldProgram {
    /// One row per folded cell (`h*w` live rows).
    pub fn live_rows(&self) -> usize {
        self.lanes.iter().map(Vec::len).sum()
    }

    /// Trace height: the live rows padded to the next power of two with all-zero `IS_PAD = 1`
    /// rows.
    pub fn num_rows(&self) -> usize {
        self.live_rows().next_power_of_two()
    }

    /// `cell_words[c]` is cell `c`'s f32 bit pattern. The matmul exposes it on its
    /// `IS_CELL_FINAL` row via the cell-results CTL.
    pub fn generate_trace<F: RichField>(&self, cell_words: &[u32]) -> Vec<[F; NUM_XOR_FOLD_COLUMNS]> {
        assert_eq!(self.lanes.len(), XOR_FOLD_LANES, "the lottery block has 16 lanes");
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
        for (j, lane) in self.lanes.iter().enumerate() {
            let mut state = 0u32;
            for (step, &cell) in lane.iter().enumerate() {
                // The leading four columns are class (a) — keep in sync with known_values.
                let w = cell_words[cell];
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
                    fold_state_in: F::from_canonical_u32(state),
                    muladd_low_limb_0: F::from_canonical_u32(lo & 0xFFFF),
                    muladd_low_limb_1: F::from_canonical_u32(lo >> 16),
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
        // Pad to the next power of two: all-zero IS_PAD rows. X1-X3 hold on zeros (the last live
        // row is lane-final, so the pad chain starts and stays at state 0).
        let pad_row = XorFoldColumnsView::<F> {
            is_pad: F::ONE,
            ..XorFoldColumnsView::default()
        };
        rows.resize(self.num_rows(), pad_row.into());
        rows
    }

    /// The class (a) ("known") column values — the leading
    /// [`NUM_XOR_FOLD_KNOWN_COLUMNS`](super::columns::NUM_XOR_FOLD_KNOWN_COLUMNS) trace columns
    /// in their `columns.rs` order (`CELL_ID`, `LANE_ID`, `IS_LANE_FINAL`, `IS_PAD`), pure
    /// functions of the committed lane layout. Bit-exact with [`Self::generate_trace`]'s fill.
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

/// Evaluates the X1-X3 constraint groups (module docs). The 10 RC16 facts (limbs, split bounds,
/// the `MULADD_HIGH_LIMB_1` canonicity cap) live in `super::ctl::xor_fold_lut_lookups`.
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

    // X1 — multiply-add split: FOLD_STATE_IN*0x9E3779B1 + W = HI*2^32 + LO over the integers.
    // The honest value is < 2^63.5 (no field wrap); the MULADD_HIGH_LIMB_1 <= 0xFFFE cap
    // (RC16 inventory) keeps the limb side below p too — without it every row with value
    // < 2^32 (all lane starts) would admit the `+p` limb alias, forging the folded word.
    let golden = eval.u64(GOLDEN);
    let two32 = eval.u64(1 << 32);
    let w_hi = eval.mul(two16, lv.cell_result_f32_hi);
    let w = eval.add(lv.cell_result_f32_lo, w_hi);
    let muladd = eval.mad(lv.fold_state_in, golden, w);
    let lo_1 = eval.mul(two16, lv.muladd_low_limb_1);
    let lo = eval.add(lv.muladd_low_limb_0, lo_1);
    let hi_1 = eval.mul(two16, lv.muladd_high_limb_1);
    let hi = eval.add(lv.muladd_high_limb_0, hi_1);
    let limbs = eval.mad(hi, two32, lo);
    eval.constraint_eq(muladd, limbs);

    // X2 — rotl32 by 13 is a pure re-split of the low word: LO = TOP13*2^19 + BOT19.
    let two19 = eval.u64(1 << 19);
    let bot19_1 = eval.mul(two16, lv.rotation_input_bottom19_limb_1);
    let bot19 = eval.add(lv.rotation_input_bottom19_limb_0, bot19_1);
    let split = eval.mad(lv.rotation_input_top13, two19, bot19);
    eval.constraint_eq(lo, split);

    // The rotated word: FOLD_OUT = BOT19*2^13 + TOP13 (affine, not a column).
    let two13 = eval.u64(1 << 13);
    let fold_out = eval.mad(bot19, two13, lv.rotation_input_top13);

    // X3 — chaining: 0 entering the first row; a lane-final row resets the next row's state;
    // otherwise the state chains. The next-row constraints are plain (cyclic): on the wrap pair
    // the last row is lane-final (a class (a) fact carried by the known-column binding), so the
    // wrap instance is row 0's reset.
    eval.constraint_first_row(lv.fold_state_in);
    let reset = eval.mul(lv.is_lane_final, nv.fold_state_in);
    eval.constraint(reset);
    let not_final = eval.sub(one, lv.is_lane_final);
    let chain_diff = eval.sub(nv.fold_state_in, fold_out);
    let chain = eval.mul(not_final, chain_diff);
    eval.constraint(chain);
}

/// FP16 XorFoldStark. A CTL party of the FP16 batch (`requires_ctls()`): its proofs carry the
/// cross-table openings of the channels declared in `super::ctl`, so the batch driver is the only
/// supported proving path — there is no standalone uni-STARK proof object.
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
    use crate::v4::api::utils::xor_fold_extract;
    use crate::v4::api::layout::{AxisPattern, DimType, lane_assignment};

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

    fn test_trace() -> (XorFoldProgram, Vec<[F; NUM_XOR_FOLD_COLUMNS]>) {
        let (tile, lanes) = test_layout();
        let program = XorFoldProgram { lanes };
        let words: Vec<u32> = tile.iter().map(|x| x.to_bits()).collect();
        let rows = program.generate_trace::<F>(&words);
        (program, rows)
    }

    fn constraints_violated(stark: &S, rows: &[[F; NUM_XOR_FOLD_COLUMNS]]) -> bool {
        let n = rows.len();
        (0..n).any(|i| {
            let frame = StarkFrame::from_values(&rows[i], &rows[(i + 1) % n], &[]);
            let mut consumer = ConstraintConsumer::new(
                vec![F::from_canonical_u64(2), F::from_canonical_u64(0x876543210)],
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
        let (program, rows) = test_trace();
        assert!(!constraints_violated(&S::new(program), &rows));
    }

    #[test]
    fn padded_trace_satisfies_all_constraints() {
        // A non-power-of-two live count: 16 lanes x 3 cells = 48 live rows, padded to 64.
        let program = XorFoldProgram {
            lanes: (0..16).map(|j| vec![3 * j, 3 * j + 1, 3 * j + 2]).collect(),
        };
        let words: Vec<u32> = (0..48u32).map(|i| f32::to_bits(i as f32 - 20.5)).collect();
        let rows = program.generate_trace::<F>(&words);
        assert_eq!(rows.len(), 64, "48 live rows pad to 64");
        let known = program.known_values::<F>();
        assert!(known.iter().all(|c| c.len() == 64));
        // The known columns are bit-exact with the trace fill (the batch verifier's check).
        for (c, col) in known.iter().enumerate() {
            for (r, row) in rows.iter().enumerate() {
                assert_eq!(col.values[r], row[c], "known column {c} row {r}");
            }
        }
        assert!(!constraints_violated(&S::new(program), &rows));
    }

    #[test]
    fn trace_is_bit_exact_vs_native_xor_fold_extract() {
        let (tile, lanes) = test_layout();
        let (program, rows) = test_trace();
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
    fn lut_inventory_holds_on_honest_trace_and_catches_the_alias() {
        let (_, rows) = test_trace();
        assert!(lut_keys_are_in_domain(&rows), "honest trace has an out-of-range RC16 key");

        // The `+p` limb alias: at a lane start (state 0, t = W < 2^32) rewrite the limbs as
        // t + p = (t + 1) + 0xFFFFFFFF*2^32 and re-split consistently. X1/X2 still vanish in the
        // field — only the RC16(MULADD_HIGH_LIMB_1 + 1) canonicity cap catches the forgery.
        let (tile, lanes) = test_layout();
        let words: Vec<u32> = tile.iter().map(|x| x.to_bits()).collect();
        let mut forged = rows.clone();
        {
            let v: &mut XorFoldColumnsView<F> = forged[0].borrow_mut();
            assert_eq!(v.fold_state_in, F::ZERO);
            let t = to_u64(v.cell_result_f32_lo) | (to_u64(v.cell_result_f32_hi) << 16);
            let lo = (t + 1) as u32;
            v.muladd_low_limb_0 = F::from_canonical_u32(lo & 0xFFFF);
            v.muladd_low_limb_1 = F::from_canonical_u32(lo >> 16);
            v.muladd_high_limb_0 = F::from_canonical_u32(0xFFFF);
            v.muladd_high_limb_1 = F::from_canonical_u32(0xFFFF);
            v.rotation_input_top13 = F::from_canonical_u32(lo >> 19);
            v.rotation_input_bottom19_limb_0 = F::from_canonical_u32(lo & 0xFFFF);
            v.rotation_input_bottom19_limb_1 = F::from_canonical_u32((lo >> 16) & 0x7);
        }
        // Rewrite the rest of lane 0 so the chain stays consistent and only the alias row differs.
        let lane0 = &lanes[0];
        let mut state = ((words[lane0[0]] as u64 + 1) as u32).rotate_left(13);
        for (step, &cell) in lane0.iter().enumerate().skip(1) {
            let t = state as u64 * GOLDEN + words[cell] as u64;
            let (lo, hi) = (t as u32, (t >> 32) as u32);
            let v: &mut XorFoldColumnsView<F> = forged[step].borrow_mut();
            v.fold_state_in = F::from_canonical_u32(state);
            v.muladd_low_limb_0 = F::from_canonical_u32(lo & 0xFFFF);
            v.muladd_low_limb_1 = F::from_canonical_u32(lo >> 16);
            v.muladd_high_limb_0 = F::from_canonical_u32(hi & 0xFFFF);
            v.muladd_high_limb_1 = F::from_canonical_u32(hi >> 16);
            v.rotation_input_top13 = F::from_canonical_u32(lo >> 19);
            v.rotation_input_bottom19_limb_0 = F::from_canonical_u32(lo & 0xFFFF);
            v.rotation_input_bottom19_limb_1 = F::from_canonical_u32((lo >> 16) & 0x7);
            state = lo.rotate_left(13);
        }
        let (program, _) = test_trace();
        assert!(
            !constraints_violated(&S::new(program), &forged),
            "the alias must satisfy X1-X3 — it is the RC16 cap's job"
        );
        assert!(
            !lut_keys_are_in_domain(&forged),
            "the RC16(MULADD_HIGH_LIMB_1 + 1) cap must catch the +p alias"
        );
    }

    #[test]
    fn tampered_traces_fail() {
        let (program, rows) = test_trace();
        let stark = S::new(program);
        let cases = [
            // A lane's chain: the next row's state no longer matches FOLD_OUT.
            ("fold_state_in", XOR_FOLD_COL_MAP.fold_state_in),
            // The folded word itself (X1 breaks).
            ("cell_result_f32_lo", XOR_FOLD_COL_MAP.cell_result_f32_lo),
            // The rotation split (X2 breaks).
            ("rotation_input_top13", XOR_FOLD_COL_MAP.rotation_input_top13),
            // A mid-lane final flag (0 -> 1): the reset forces the next state to 0 (false here).
            ("is_lane_final", XOR_FOLD_COL_MAP.is_lane_final),
        ];
        for (name, col) in cases {
            let mut forged = rows.clone();
            // Row 1 is mid-lane (lanes have 4 cells) and has a nonzero next state.
            forged[1][col] += F::ONE;
            assert!(constraints_violated(&stark, &forged), "{name} tamper undetected");
        }
    }

    #[test]
    fn degree_is_at_most_three() {
        let (program, _) = test_trace();
        test_stark_low_degree::<F, S, D>(S::new(program)).unwrap();
    }

    #[test]
    fn circuit_constraints_match_native() {
        let (program, _) = test_trace();
        test_stark_circuit_constraints::<F, C, S, D>(S::new(program)).unwrap();
    }

    // No standalone prove/verify smoke test: this table is a CTL party (`requires_ctls`), so a
    // proof without the cross-table argument is not a supported object. The end-to-end proving
    // path is covered once the table is wired into `fp16::driver`.
}
