//! Connects each XorFold step's `cell_word` input and `fold_out` output.
//!
//! Live rows receive `(cell_id, cell_result_f32_lo, cell_result_f32_hi)` from Matmul.
//! Lane-final rows send `(lane_id, fold_out)` to Blake3; RC16 lookups bound arithmetic limbs.

use plonky2::field::types::Field;
use starky::cross_table_lookup::TableWithColumns;
use starky::lookup::{Column, Filter};

use super::super::ctl::{LutLookup, Table};
use super::columns::XOR_FOLD_COL_MAP;

/// Imports `(cell_id, f32_lo, f32_hi)` from Matmul on every live row; padding folds no cell.
pub fn ctl_cell_results_looking_xor_fold<F: Field>() -> TableWithColumns<F> {
    let m = &XOR_FOLD_COL_MAP;
    TableWithColumns::new(
        Table::XorFold.into(),
        Column::singles([m.cell_id, m.cell_result_f32_lo, m.cell_result_f32_hi]).collect(),
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

/// Ten RC16 lookups per row bound the multiply-add limbs and rotation splits,
/// and keep the recomposed multiply-add result below the field modulus.
pub fn xor_fold_lut_lookups<F: Field>() -> Vec<LutLookup<F>> {
    let m = &XOR_FOLD_COL_MAP;
    vec![
        LutLookup::rc16(Column::single(m.muladd_low_limb_0)),
        LutLookup::rc16(Column::single(m.muladd_low_limb_1)),
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
