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
//! The multiply-add fits below `2^63.5`. Constraint group X1 recomposes it as four 16-bit
//! limbs:
//!
//! ```text
//! mixed = low_0 + 2^16*low_1 + 2^32*high_0 + 2^48*high_1.
//! ```
//!
//! RC16 bounds every limb, and the top limb is additionally at most
//! `0xFFFE = 2^16 - 2`. That cap keeps the reconstructed integer below the Goldilocks modulus
//! and rules out adding one modulus while preserving the same field equality.
//!
//! X2 splits the low 32-bit word into a 13-bit top part and 19-bit bottom part:
//!
//! ```text
//! low      = top13*2^19 + bottom19
//! fold_out = bottom19*2^13 + top13
//! ```
//!
//! The second equation rotates the word left by 13 bits: the top 13 bits wrap
//! into the low positions, and the remaining 19 bits move up.
//!
//! # Lane chaining and table boundaries
//!
//! X3 starts each public-layout lane at state zero, carries `fold_out` into the next live row,
//! and resets at `is_lane_final`. Live rows receive
//! `(cell_id, cell_result_f32_lo, cell_result_f32_hi)` from MatmulB200Stark. Each lane-final row
//! sends `(lane_id, fold_out)` to Blake3Stark as one jackpot message word.
//!
//! The verifier recomputes `cell_id`, `lane_id`, `is_lane_final`, and `is_pad` from the public
//! lane assignment and checks their openings. Padding rows set `is_pad = 1`, zero all data
//! columns, and participate in neither cross-table lookup.

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
use crate::api::layout::JACKPOT_ENTRIES as XOR_FOLD_LANES;
use crate::circuit::utils::evaluator::Evaluator;
use crate::circuit::utils::native_evaluator::NativeEvaluator;
use crate::circuit::utils::symbolic_evaluator::SymbolicEvaluator;

/// The protocol's fixed odd mixing multiplier (the golden-ratio-derived XorFold constant).
const FOLD_MULTIPLIER: u64 = 0x9E3779B1;

/// The committed lane layout: `lanes[j]` lists lane `j`'s cell ids in fold order
/// (`crate::api::layout::lane_assignment`).
#[derive(Clone, Debug)]
pub struct XorFoldProgram {
    pub lanes: Vec<Vec<usize>>,
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

    /// `cell_words[c]` = cell `c`'s f32 bit pattern (what Matmul's `IS_CELL_FINAL` rows expose
    /// as `CELL_RESULT_F32_LO/HI`).
    pub fn generate_trace<F: RichField>(
        &self,
        cell_words: &[u32],
    ) -> (Vec<[F; NUM_XOR_FOLD_COLUMNS]>, [F; NUM_XOR_FOLD_PUBLIC_INPUTS]) {
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

        let mut rows = Vec::with_capacity(self.num_rows());
        for (j, lane) in self.lanes.iter().enumerate() {
            let mut state = 0u32;
            for (step, &cell) in lane.iter().enumerate() {
                // The leading four columns are verifier-known — keep in sync with known_values.
                let w = cell_words[cell];
                let t = state as u64 * FOLD_MULTIPLIER + w as u64; // < 2^63.5: exact in the field too
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
        // Pad to the next power of two with all-zero IS_PAD rows: X1-X3 hold on zeros (the
        // last live row is lane-final, so the pad chain starts and stays at state 0).
        let pad_row = XorFoldColumnsView::<F> {
            is_pad: F::ONE,
            ..XorFoldColumnsView::default()
        };
        rows.resize(self.num_rows(), pad_row.into());
        (rows, [])
    }

    /// Recomputes the leading known columns in trace order from public geometry.
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

/// Evaluates the X1-X3 constraint groups (module docs). The 10 RC16 facts (limbs, split
/// bounds, the `MULADD_HIGH_LIMB_1` canonicity cap) live in
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

    // X1 — multiply-add split: FOLD_STATE_IN*0x9E3779B1 + W = HI*2^32 + LO over the integers.
    // The honest value is < 2^63.5 (no field wrap); the MULADD_HIGH_LIMB_1 <= 0xFFFE cap
    // (RC16 inventory) keeps the limb side below p too — without it every row with value
    // < 2^32 (all lane starts) would admit the `+p` limb alias, forging the folded word.
    let fold_multiplier = eval.u64(FOLD_MULTIPLIER);
    let two32 = eval.u64(1 << 32);
    let w_hi = eval.mul(two16, lv.cell_result_f32_hi);
    let w = eval.add(lv.cell_result_f32_lo, w_hi);
    let muladd = eval.mad(lv.fold_state_in, fold_multiplier, w);
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

    // X3: start row 0 at zero, reset after lane-final rows, otherwise carry fold_out.
    // These constraints also apply to the cyclic wrap. The known schedule makes the
    // last row lane-final, so the wrap resets row 0; padding states remain zero.
    eval.constraint_first_row(lv.fold_state_in);
    let reset = eval.mul(lv.is_lane_final, nv.fold_state_in);
    eval.constraint(reset);
    let not_final = eval.sub(one, lv.is_lane_final);
    let chain_diff = eval.sub(nv.fold_state_in, fold_out);
    let chain = eval.mul(not_final, chain_diff);
    eval.constraint(chain);
}

/// Lottery-fold AIR, proved through the batch driver with the channels in `super::ctl`.
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
    use crate::api::layout::{AxisPattern, DimType, lane_assignment};
    use crate::circuit::fp8::matmul_b200_stark::columns::MATMUL_B200_COL_MAP;
    use crate::circuit::fp8::matmul_b200_stark::stark::{MatmulProgram, generate_b200_trace};

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

    fn test_trace() -> (XorFoldProgram, Vec<[F; NUM_XOR_FOLD_COLUMNS]>, [F; 0]) {
        let (tile, lanes) = test_layout();
        let program = XorFoldProgram { lanes };
        let words: Vec<u32> = tile.iter().map(|x| x.to_bits()).collect();
        let (rows, pis) = program.generate_trace::<F>(&words);
        (program, rows, pis)
    }

    fn constraints_violated(stark: &S, rows: &[[F; NUM_XOR_FOLD_COLUMNS]], pis: &[F; 0]) -> bool {
        let n = rows.len();
        (0..n).any(|i| {
            let frame = StarkFrame::from_values(&rows[i], &rows[(i + 1) % n], pis);
            let mut consumer = ConstraintConsumer::new(
                vec![F::from_canonical_u64(2), F::from_canonical_u64(0x876543210)],
                F::ONE,
                F::from_bool(i == 0),
                F::from_bool(i == n - 1),
            );
            stark.eval_packed_generic(&frame, &mut consumer);
            consumer.accumulators().iter().any(|&acc| acc != F::ZERO)
        })
    }

    #[test]
    fn honest_trace_satisfies_all_constraints() {
        // Includes the last -> first wrap, which the AIR uses as row 0's reset instance.
        let (program, rows, pis) = test_trace();
        assert!(!constraints_violated(&S::new(program), &rows, &pis));
    }

    #[test]
    fn padded_trace_satisfies_all_constraints() {
        // A non-power-of-two live count: 16 lanes x 3 cells = 48 live rows, padded to 64. The
        // wrap pair is (pad, row 0), exercising the pad chain closing on the FOLD_STATE_IN = 0
        // anchor and the padded last-row anchor.
        let program = XorFoldProgram {
            lanes: (0..16).map(|j| vec![3 * j, 3 * j + 1, 3 * j + 2]).collect(),
        };
        let words: Vec<u32> = (0..48u32).map(|i| f32::to_bits(i as f32 - 20.5)).collect();
        let (rows, pis) = program.generate_trace::<F>(&words);
        assert_eq!(rows.len(), 64, "48 live rows pad to 64");
        let known = program.known_values::<F>();
        assert!(known.iter().all(|c| c.len() == 64));
        // The known columns are bit-exact with the trace fill (the batch verifier's check).
        for (c, col) in known.iter().enumerate() {
            for (r, row) in rows.iter().enumerate() {
                assert_eq!(col.values[r], row[c], "known column {c} row {r}");
            }
        }
        assert!(!constraints_violated(&S::new(program), &rows, &pis));
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
        let matmul = MatmulProgram { h: 4, w: 4, k: 128 };
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
        let (matmul_rows, _) = generate_b200_trace::<F>(&matmul, &a, &b, &lambdas, &lambdas);

        // The looked filter is IS_CELL_FINAL * (1 - IS_PADDING): the phantom padding cells'
        // finals emit no word.
        let mut looked: Vec<(u64, u64, u64)> = matmul_rows
            .iter()
            .filter(|row| row[MATMUL_B200_COL_MAP.is_cell_final] == F::ONE && row[MATMUL_B200_COL_MAP.is_padding] == F::ZERO)
            .map(|row| {
                (
                    to_u64(row[MATMUL_B200_COL_MAP.cell_id]),
                    to_u64(row[MATMUL_B200_COL_MAP.cell_result_f32_lo]),
                    to_u64(row[MATMUL_B200_COL_MAP.cell_result_f32_hi]),
                )
            })
            .collect();

        // Fold those 16 cells, one per lane (16 = h*w keeps every lane nonempty).
        let mut words = vec![0u32; 16];
        for &(cell, lo, hi) in &looked {
            words[cell as usize] = (lo | (hi << 16)) as u32;
        }
        let program = XorFoldProgram {
            lanes: (0..16).map(|j| vec![j]).collect(),
        };
        let (rows, pis) = program.generate_trace::<F>(&words);
        assert!(!constraints_violated(&S::new(program), &rows, &pis));

        let mut looking: Vec<(u64, u64, u64)> = rows
            .iter()
            .filter(|row| {
                let v: &XorFoldColumnsView<F> = (*row).borrow();
                v.is_pad == F::ZERO
            })
            .map(|row| {
                let v: &XorFoldColumnsView<F> = row.borrow();
                (to_u64(v.cell_id), to_u64(v.cell_result_f32_lo), to_u64(v.cell_result_f32_hi))
            })
            .collect();
        looked.sort_unstable();
        looking.sort_unstable();
        assert_eq!(looked, looking, "cell-results channel out of balance");
    }

    #[test]
    fn lut_inventory_holds_on_honest_trace_and_catches_the_alias() {
        let (_, rows, _) = test_trace();
        let polys = trace_rows_to_poly_values(rows.clone());
        let in_domain = |polys: &[plonky2::field::polynomial::PolynomialValues<F>]| {
            xor_fold_lut_lookups::<F>()
                .iter()
                .all(|l| (0..rows.len()).all(|r| l.keys[0].eval_table(polys, r, &[]).to_canonical_u64() < 1 << 16))
        };
        assert!(in_domain(&polys), "honest trace has an out-of-range RC16 key");

        // The `+p` limb alias: at a lane start (state 0, t = W < 2^32) rewrite the limbs as
        // t + p = (t + 1) + 0xFFFFFFFF*2^32 and re-split consistently. X1/X2 still vanish in the
        // field — only the RC16(MULADD_HIGH_LIMB_1 + 1) canonicity cap catches the forgery.
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
            v.muladd_low_limb_0 = F::from_canonical_u32(lo & 0xFFFF);
            v.muladd_low_limb_1 = F::from_canonical_u32(lo >> 16);
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
            !in_domain(&trace_rows_to_poly_values(forged)),
            "the RC16(MULADD_HIGH_LIMB_1 + 1) cap must catch the +p alias"
        );
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
            // The rotation split (X2 breaks).
            ("rotation_input_top13", XOR_FOLD_COL_MAP.rotation_input_top13),
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
        // still rejects this trace. (The flag is verifier-known regardless: any divergence, even
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
}
