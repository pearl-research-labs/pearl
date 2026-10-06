//! Connects each XorFold step's `cell_word` input and `fold_out` output.
//!
//! Live rows receive `(cell_id, cell_result_f32_lo, cell_result_f32_hi, cell_skips)` from
//! Matmul, binding each folded result to its certified unpredictability-census subtotal.
//! Lane-final rows send `(lane_id, fold_out)` to Blake3. RC16 lookups bound both the mixer
//! arithmetic and the two limbs of the final nonnegative skip-budget slack.

use plonky2::field::types::Field;
use starky::cross_table_lookup::TableWithColumns;
use starky::lookup::{Column, Filter};

use super::super::ctl::Table;
use super::super::luts::ctl::LutLookup;
use super::columns::XOR_FOLD_COL_MAP;

/// XorFold's looking side of the **cell results** channel: `(CELL_ID,
/// CELL_RESULT_F32_LO, CELL_RESULT_F32_HI, CELL_SKIPS)`, filter `1 - IS_PAD` — every live
/// row folds one finished cell and imports the number of that cell's summands below the
/// Matmul table's M/Z unpredictability threshold.
/// Matmul's looked side is `super::super::ctl::ctl_cell_results_looked_matmul`.
pub fn ctl_cell_results_looking_xor_fold<F: Field>() -> TableWithColumns<F> {
    let m = &XOR_FOLD_COL_MAP;
    TableWithColumns::new(
        Table::XorFold.into(),
        Column::singles([m.cell_id, m.cell_result_f32_lo, m.cell_result_f32_hi, m.cell_skips]).collect(),
        Filter::from_column(Column::linear_combination_with_constant([(m.is_pad, -F::ONE)], F::ONE)),
    )
}

/// Exports `(lane_id, fold_out)` on lane-final rows, binding Blake3's 16 lottery message words.
pub fn ctl_lottery_words_looked_xor_fold<F: Field>() -> TableWithColumns<F> {
    let m = &XOR_FOLD_COL_MAP;
    TableWithColumns::new(
        Table::XorFold.into(),
        vec![
            Column::single(m.lane_id),
            Column::linear_combination([
                (m.rotation_input_top13, F::ONE),
                (m.rotation_input_bottom19_limb_0, F::from_canonical_u64(1 << 13)),
                (m.rotation_input_bottom19_limb_1, F::from_canonical_u64(1 << 29)),
            ]),
        ],
        Filter::from_column(Column::single(m.is_lane_final)),
    )
}

/// XorFoldStark's per-row LUT inventory: RC16 x10 — the two high mul-add limbs,
/// the rotation-split bounds and the canonicity cap `MULADD_HIGH_LIMB_1 + 1` that kills X1's
/// `+p` limb alias, plus the two skip-budget slack limbs. The slack lookups apply on every row;
/// only the final row enters the budget equality, and honest traces set both limbs to zero
/// elsewhere.
///
/// Both sub-16-bit splits use the unshifted + shifted RC16 pair, because a scaled RC16 alone
/// never bounds a Goldilocks column (`2^k` divides `v + j*p` for suitable `j`, producing huge
/// canonical aliases whose scaled key is still `< 2^16`):
/// - `ROTATION_INPUT_TOP13`: the unshifted check prevents wrap, then the `2^3`-scaled check
///   gives the 13-bit bound.
/// - `ROTATION_INPUT_BOTTOM19_LIMB_1`: the unshifted check plus the `2^13`-scaled check gives
///   the 3-bit bound. Without the unshifted half, an alias can satisfy X1 while moving
///   `FOLD_OUT`, enabling free lottery grinding.
pub fn xor_fold_lut_lookups<F: Field>() -> Vec<LutLookup<F>> {
    let m = &XOR_FOLD_COL_MAP;
    vec![
        LutLookup::rc16(Column::single(m.muladd_high_limb_0)),
        LutLookup::rc16(Column::single(m.muladd_high_limb_1)),
        LutLookup::rc16(Column::single(m.rotation_input_bottom19_limb_0)),
        // Bound to 16 bits before scaling by 2^3, then require the scaled value to fit too.
        // Together these give a 13-bit bound. A scaled check alone can wrap modulo p,
        // admitting a large field value that changes FOLD_OUT while satisfying X2.
        LutLookup::rc16(Column::single(m.rotation_input_top13)),
        LutLookup::rc16(Column::linear_combination([(
            m.rotation_input_top13,
            F::from_canonical_u64(1 << 3),
        )])),
        // The same pair with scale 2^13 bounds the upper part of the 19-bit split to 3 bits.
        LutLookup::rc16(Column::single(m.rotation_input_bottom19_limb_1)),
        LutLookup::rc16(Column::linear_combination([(
            m.rotation_input_bottom19_limb_1,
            F::from_canonical_u64(1 << 13),
        )])),
        // Cap the highest limb at 65534, keeping the 64-bit recomposition below p.
        // Otherwise X1 could accept the intended integer plus the field modulus.
        LutLookup::rc16(Column::linear_combination_with_constant(
            [(m.muladd_high_limb_1, F::ONE)],
            F::ONE,
        )),
        LutLookup::rc16(Column::single(m.skip_gate_slack_lo)),
        LutLookup::rc16(Column::single(m.skip_gate_slack_hi)),
    ]
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;

    use super::super::super::ctl::LutTable;
    use super::*;

    type F = GoldilocksField;

    #[test]
    fn xor_fold_ctl_halves_are_well_formed() {
        ctl_cell_results_looking_xor_fold::<F>();
        ctl_lottery_words_looked_xor_fold::<F>();
    }

    #[test]
    fn xor_fold_lut_inventory_matches_documented_counts() {
        // Documented inventory: 10 RC16 instances, nothing else.
        let lookups = xor_fold_lut_lookups::<F>();
        assert_eq!(lookups.len(), 10);
        assert!(lookups.iter().all(|l| l.table == LutTable::Range16));
    }
}
