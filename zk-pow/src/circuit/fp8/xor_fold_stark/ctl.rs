//! Connects each XorFold step's `cell_word` input and `fold_out` output.
//!
//! Live rows receive `(cell_id, cell_result_f32_lo, cell_result_f32_hi)` from Matmul.
//! Lane-final rows send `(lane_id, fold_out)` to Blake3; RC16 lookups bound arithmetic limbs.

use plonky2::field::types::Field;
use starky::cross_table_lookup::TableWithColumns;
use starky::lookup::{Column, Filter};

use super::super::ctl::Table;
use super::super::luts::ctl::LutLookup;
use super::columns::XOR_FOLD_COL_MAP;

/// XorFold's looking side of the **cell results** channel: `(CELL_ID,
/// CELL_RESULT_F32_LO, CELL_RESULT_F32_HI)`, filter `1 - IS_PAD` — every live row folds
/// exactly one finished cell; the power-of-two padding rows fold nothing.
/// Matmul's looked side is `super::super::ctl::ctl_cell_results_looked_matmul`.
pub fn ctl_cell_results_looking_xor_fold<F: Field>() -> TableWithColumns<F> {
    let m = &XOR_FOLD_COL_MAP;
    TableWithColumns::new(
        Table::XorFold.into(),
        Column::singles([m.cell_id, m.cell_result_f32_lo, m.cell_result_f32_hi]).collect(),
        Filter::from_column(Column::linear_combination_with_constant([(m.is_pad, -F::ONE)], F::ONE)),
    )
}

/// XorFold's looked side of the **lottery words** channel: `(LANE_ID,
/// FOLD_OUT)` with the affine `FOLD_OUT = (ROTATION_INPUT_BOTTOM19_LIMB_0
/// + 2^16*ROTATION_INPUT_BOTTOM19_LIMB_1)*2^13 + ROTATION_INPUT_TOP13`, filter
/// `IS_LANE_FINAL`. Blake3's looking side sends 16 tuples pairing each word position with the
/// corresponding `BLAKE3_MSG` word on the lottery message-load row
/// (`blake3_stark::ctl::ctl_lottery_words_looking_blake3`).
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

/// XorFoldStark's per-row LUT inventory: RC16 x10 — the four mul-add limbs,
/// the rotation-split bounds and the canonicity cap `MULADD_HIGH_LIMB_1 + 1` that kills X1's
/// `+p` limb alias.
///
/// Both sub-16-bit splits use the unshifted + shifted RC16 pair, because a scaled RC16 alone
/// never bounds a Goldilocks column (`2^k` divides `v + j*p` for suitable `j`, producing huge
/// canonical aliases whose scaled key is still `< 2^16`):
/// - `ROTATION_INPUT_TOP13`: the unshifted check prevents wrap, then the `2^3`-scaled check
///   gives the 13-bit bound.
/// - `ROTATION_INPUT_BOTTOM19_LIMB_1`: the unshifted check plus the `2^13`-scaled check gives
///   the 3-bit bound. Without the unshifted half, an alias can satisfy X2's split while moving
///   `FOLD_OUT`, enabling free lottery grinding.
pub fn xor_fold_lut_lookups<F: Field>() -> Vec<LutLookup<F>> {
    let m = &XOR_FOLD_COL_MAP;
    vec![
        LutLookup::rc16(Column::single(m.muladd_low_limb_0)),
        LutLookup::rc16(Column::single(m.muladd_low_limb_1)),
        LutLookup::rc16(Column::single(m.muladd_high_limb_0)),
        LutLookup::rc16(Column::single(m.muladd_high_limb_1)),
        LutLookup::rc16(Column::single(m.rotation_input_bottom19_limb_0)),
        LutLookup::rc16(Column::single(m.rotation_input_top13)),
        LutLookup::rc16(Column::linear_combination([(
            m.rotation_input_top13,
            F::from_canonical_u64(1 << 3),
        )])),
        LutLookup::rc16(Column::single(m.rotation_input_bottom19_limb_1)),
        LutLookup::rc16(Column::linear_combination([(
            m.rotation_input_bottom19_limb_1,
            F::from_canonical_u64(1 << 13),
        )])),
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
