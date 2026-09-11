//! Lookup and cross-table channels for B200 matrix multiplication.
//!
//! Operand channels import pairs of FP8 codes and their summand scores.
//! B200ALIGN supplies each lane's aligned product. POW2GB supplies the carry's
//! shift power, and WIDTH32 supplies the group sum's bit width for truncation.
//!
//! Output channels export the final f32 cell word to XorFold and the cell's
//! magnitude bound and skip count to Tamed. [`super::stark`] defines the
//! accumulation arithmetic and the MB1–MB16 constraint groups.

use plonky2::field::types::Field;
use starky::cross_table_lookup::TableWithColumns;
use starky::lookup::{Column, Filter};

use super::columns::{GROUP_WIDTH, MATMUL_B200_COL_MAP};
use super::stark::PARTIAL_BINADE_OFFSET;
use crate::circuit::fp8::ctl::{LutLookup, LutTable, Table};
use crate::circuit::fp8::unpredictability::SKIP_THRESHOLD_OFFSET;

/// Imports pairs of FP8 codes and their two separate summand scores, A then B.
/// Keys identify the first element; B keys include h*k. Padding imports nothing.
/// Both sides bound each code to a byte, making packing unique. Scores remain separate
/// so each inherits InputQuant's value and range. InputQuant's multiplicities account
/// for reuse across w cells per A element and h cells per B element.
pub fn ctl_operand_codes_looking_matmul_b200<F: Field>() -> Vec<TableWithColumns<F>> {
    let m = &MATMUL_B200_COL_MAP;
    let byte_shift = F::from_canonical_u64(1 << 8);
    let not_padding = || Column::linear_combination_with_constant([(m.is_padding, -F::ONE)], F::ONE);
    let sides = [
        (m.operand_index_base_a, &m.operand_codes_a, &m.summand_score_a),
        (m.operand_index_base_b, &m.operand_codes_b, &m.summand_score_b),
    ];
    sides
        .into_iter()
        .flat_map(|(base, codes, lambdas)| {
            (0..GROUP_WIDTH / 2).map(move |i| {
                TableWithColumns::new(
                    Table::Matmul.into(),
                    vec![
                        Column::linear_combination_with_constant([(base, F::ONE)], F::from_canonical_usize(2 * i)),
                        Column::linear_combination([(codes[2 * i], F::ONE), (codes[2 * i + 1], byte_shift)]),
                        Column::single(lambdas[2 * i]),
                        Column::single(lambdas[2 * i + 1]),
                    ],
                    Filter::from_column(not_padding()),
                )
            })
        })
        .collect()
}

/// Export each completed cell's f32 word to XorFold as `(cell_id, low_limb, high_limb)`.
/// The two limbs are range-checked. Only a real cell's final row emits a tuple.
pub fn ctl_cell_results_looked_matmul_b200<F: Field>() -> TableWithColumns<F> {
    let m = &MATMUL_B200_COL_MAP;
    TableWithColumns::new(
        Table::Matmul.into(),
        Column::singles([m.cell_id, m.cell_result_f32_lo, m.cell_result_f32_hi]).collect(),
        Filter::new(
            vec![(
                Column::single(m.is_cell_final),
                Column::linear_combination_with_constant([(m.is_padding, -F::ONE)], F::ONE),
            )],
            vec![],
        ),
    )
}

/// Exports `(cell_id, magnitude_exponent_bound, skip_count)` once per live cell, on its final row.
pub fn ctl_e_cell_looked_matmul_b200<F: Field>() -> TableWithColumns<F> {
    let m = &MATMUL_B200_COL_MAP;
    TableWithColumns::new(
        Table::Matmul.into(),
        Column::singles([m.cell_id, m.cell_magnitude_exponent, m.cell_skips]).collect(),
        Filter::new(
            vec![(
                Column::single(m.is_cell_final),
                Column::linear_combination_with_constant([(m.is_padding, -F::ONE)], F::ONE),
            )],
            vec![],
        ),
    )
}

/// The 112 per-row lookup instances, independent of job geometry:
///
/// - 78 RC16 checks: 13 for window arithmetic, 33 for magnitude bounds
///   (32 lanes and one partial accumulator), and 32 for non-skip certificates;
/// - 32 B200ALIGN lookups, one per product lane;
/// - one filtered POW2GB lookup for the carry shift;
/// - one filtered WIDTH32 lookup for the nonzero sum's bit width.
///
/// Each instance becomes a cross-table lookup into its committed `LutStark`.
pub fn matmul_b200_lut_lookups<F: Field>() -> Vec<LutLookup<F>> {
    let m = &MATMUL_B200_COL_MAP;
    let one = F::ONE;
    let neg = -F::ONE;
    let hi10 = F::from_canonical_u64(1 << 6);
    let group_sum_is_nonzero = Filter::from_column(Column::linear_combination_with_constant([(m.group_sum_is_zero, neg)], one));

    let mut lookups = vec![
        // MB5: bound the carry quotient, remainder and remainder bound as integers.
        // RC16(HI*2^6) bounds the lookup key, not necessarily HI itself. This is sufficient:
        // HI*2^16 + LO = key*2^10 + LO < 2^26 + 2^16, and MB5/MB6 use only this recomposition.
        // Any alternative HI therefore represents the same bounded integer. The quotient
        // bound is essential: without it, field division could satisfy a false floor equation.
        LutLookup::rc16(Column::single(m.aligned_incoming_carry_lo)),
        LutLookup::rc16(Column::linear_combination([(m.aligned_incoming_carry_hi, hi10)])),
        LutLookup::rc16(Column::single(m.incoming_carry_remainder_lo)),
        LutLookup::rc16(Column::linear_combination([(m.incoming_carry_remainder_hi, hi10)])),
        LutLookup::rc16(Column::single(m.incoming_carry_remainder_bound_lo)),
        LutLookup::rc16(Column::linear_combination([(m.incoming_carry_remainder_bound_hi, hi10)])),
        // MB7: RC16(HI) and RC16((HI - 128)*2^9) together require 128 <= HI < 256.
        // With RC16(LO), this puts a nonzero normalized significand in [2^23, 2^24).
        // The scaled check alone allows recompositions up to 2^24 + 2^16 - 1: an alias
        // could claim width W-1 with a 25-bit significand and change the output f32 word.
        LutLookup::rc16(Column::single(m.normalized_group_sum_significand_lo)),
        LutLookup::rc16(Column::single(m.normalized_group_sum_significand_hi)),
        LutLookup::rc16_filtered(
            Column::linear_combination_with_constant(
                [(m.normalized_group_sum_significand_hi, F::from_canonical_u64(1 << 9))],
                -F::from_canonical_u64(128 << 9),
            ),
            group_sum_is_nonzero.clone(),
        ),
        // MB7: the Euclidean remainder and its bound (< TRUNCATION_POWER <= 2^8).
        LutLookup::rc16(Column::single(m.truncation_remainder)),
        LutLookup::rc16(Column::single(m.truncation_remainder_bound)),
        // MB12: f32 limb splits.
        LutLookup::rc16(Column::single(m.cell_result_f32_lo)),
        LutLookup::rc16(Column::single(m.cell_result_f32_hi)),
        // MB13 (jackpot check 3):
        // CELL_MAGNITUDE_EXPONENT >= GROUP_OUTPUT_BIASED_EXPONENT + 101
        // (see `PARTIAL_BINADE_OFFSET`). Zero partials have no binade:
        // the filter skips them.
        LutLookup::rc16_filtered(
            Column::linear_combination_with_constant(
                [(m.cell_magnitude_exponent, one), (m.group_output_biased_exponent, neg)],
                -F::from_canonical_u64(PARTIAL_BINADE_OFFSET),
            ),
            group_sum_is_nonzero.clone(),
        ),
        // MB5: bind the next row's carry shift to this row's output exponent.
        // POW2GB rejects negative gaps, requiring GROUP_MAX_BIASED_EXPONENT' >= GROUP_OUTPUT_BIASED_EXPONENT.
        // The next-row read wraps at the trace end; MB4 disables that lookup by fixing
        // row 0's INCOMING_CARRY_IS_ZERO to 1. Enabling it would only add an obligation.
        LutLookup {
            table: LutTable::Pow2Gb,
            keys: vec![Column::linear_combination_and_next_row_with_constant(
                vec![(m.group_output_biased_exponent, neg)],
                vec![(m.group_max_biased_exponent, one)],
                F::ZERO,
            )],
            values: vec![Column::single_next_row(m.incoming_carry_shift_power)],
            filter: Filter::from_column(Column::linear_combination_and_next_row_with_constant(
                vec![],
                vec![(m.incoming_carry_is_zero, neg)],
                one,
            )),
        },
        // MB7: WIDTH32 on the claimed width — the [1, 32] key domain is the width range
        // proof (there is no key 0 row: a zero width cannot be served).
        LutLookup {
            table: LutTable::Width32,
            keys: vec![Column::single(m.group_sum_width)],
            values: Column::singles([m.truncation_power, m.lifting_power]).collect(),
            filter: group_sum_is_nonzero,
        },
    ];

    // MB1: B200ALIGN x32 — the whole per-lane product, pre-truncated to the window,
    // plus its binade for check 3.
    let byte_shift = F::from_canonical_u64(1 << 8);
    let rel_shift = F::from_canonical_u64(1 << 16);
    for i in 0..GROUP_WIDTH {
        lookups.push(LutLookup {
            table: LutTable::B200Align,
            keys: vec![Column::linear_combination([
                (m.operand_codes_a[i], one),
                (m.operand_codes_b[i], byte_shift),
                (m.group_max_biased_exponent, rel_shift),
                (m.product_biased_exponents[i], -rel_shift),
            ])],
            values: Column::singles([
                m.aligned_lane_terms[i],
                m.product_biased_exponents[i],
                m.operand_codes_a[i],
                m.operand_codes_b[i],
                m.lane_binades[i],
            ])
            .collect(),
            filter: Filter::default(),
        });
    }

    // MB13: CELL_MAGNITUDE_EXPONENT >= LANE_BINADES_i on every lane. Zero products (binade 0) always
    // pass; nonzero binades are >= 121, so CELL_MAGNITUDE_EXPONENT = 0 implies an all-zero cell.
    for i in 0..GROUP_WIDTH {
        lookups.push(LutLookup::rc16(Column::linear_combination([
            (m.cell_magnitude_exponent, one),
            (m.lane_binades[i], neg),
        ])));
    }

    // MB15: the per-lane non-skip certificates (jackpot check 4) — the affine key
    // LAMBDA_A_i + LAMBDA_B_i - 128*CELL_MAGNITUDE_EXPONENT - 45376 under the filter
    // CELL_NONZERO * (1 - SKIP_FLAG_i). A true skip's key is negative and wraps far outside
    // [0, 2^16), forcing SKIP_FLAG_i = 1; zero cells and padding pin the flag to 0 and the
    // filter is off, so they pay no certificate.
    let offset = -F::from_canonical_u64(SKIP_THRESHOLD_OFFSET);
    let neg128 = -F::from_canonical_u64(128);
    for i in 0..GROUP_WIDTH {
        lookups.push(LutLookup::rc16_filtered(
            Column::linear_combination_with_constant([(m.summand_score_a[i], one), (m.summand_score_b[i], one), (m.cell_magnitude_exponent, neg128)], offset),
            Filter::new(
                vec![(
                    Column::single(m.cell_nonzero),
                    Column::linear_combination_with_constant([(m.skip_flag[i], neg)], one),
                )],
                vec![],
            ),
        ));
    }

    lookups
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;

    use super::*;

    type F = GoldilocksField;

    #[test]
    fn ctl_halves_are_well_formed() {
        let looking = ctl_operand_codes_looking_matmul_b200::<F>();
        assert_eq!(looking.len(), GROUP_WIDTH, "16 pair instances per side");
        ctl_cell_results_looked_matmul_b200::<F>();
    }

    #[test]
    fn lut_inventory_matches_documented_counts() {
        let lookups = matmul_b200_lut_lookups::<F>();
        let count = |t: LutTable| lookups.iter().filter(|l| l.table == t).count();
        // Documented inventory: RC16 x78, B200ALIGN x32, POW2GB x1, WIDTH32 x1.
        assert_eq!(count(LutTable::Range16), 78);
        assert_eq!(count(LutTable::Bytes2), 0);
        assert_eq!(count(LutTable::B200Align), 32);
        assert_eq!(count(LutTable::Pow2Gb), 1);
        assert_eq!(count(LutTable::Width32), 1);
        assert_eq!(lookups.len(), 112);
    }
}
