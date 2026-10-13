//! Lookup and cross-table channels for H100 matrix multiplication.

use plonky2::field::types::Field;
use starky::cross_table_lookup::TableWithColumns;
use starky::lookup::{Column, Filter};

use super::columns::{GROUP_WIDTH, MATMUL_COL_MAP};
use crate::v4::circuit::ctl::Table;
use crate::v4::circuit::luts::LutTable;
use crate::v4::circuit::luts::ctl::LutLookup;
use crate::v4::circuit::unpredictability::SKIP_THRESHOLD_OFFSET;

/// H100 Matmul's looking side of the pair-packed operand-code channel.
pub fn ctl_operand_codes_looking_matmul<F: Field>() -> Vec<TableWithColumns<F>> {
    let m = &MATMUL_COL_MAP;
    let byte_shift = F::from_canonical_u64(1 << 8);
    let not_padding = || Column::linear_combination_with_constant([(m.is_padding, -F::ONE)], F::ONE);
    let sides = [
        (m.operand_index_base_a, &m.operand_codes_a, &m.lambda_a),
        (m.operand_index_base_b, &m.operand_codes_b, &m.lambda_b),
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

/// H100 Matmul's looked side of the result-and-census channel.
pub fn ctl_cell_results_looked_matmul<F: Field>() -> TableWithColumns<F> {
    let m = &MATMUL_COL_MAP;
    TableWithColumns::new(
        Table::Matmul.into(),
        Column::singles([m.cell_id, m.cell_result_f32_lo, m.cell_result_f32_hi, m.cell_skips]).collect(),
        Filter::new(
            vec![(
                Column::single(m.is_cell_final),
                Column::linear_combination_with_constant([(m.is_padding, -F::ONE)], F::ONE),
            )],
            vec![],
        ),
    )
}

/// H100 Matmul's committed-LUT inventory.
pub fn matmul_lut_lookups<F: Field>() -> Vec<LutLookup<F>> {
    let m = &MATMUL_COL_MAP;
    let one = F::ONE;
    let neg = -F::ONE;
    let nonzero_local = Filter::from_column(Column::linear_combination_with_constant([(m.group_sum_is_zero, neg)], one));

    let mut lookups = vec![
        LutLookup::rc16(Column::single(m.incoming_carry_remainder)),
        LutLookup::rc16(Column::linear_combination_with_constant(
            [(m.incoming_carry_shift_power, one), (m.incoming_carry_remainder, neg)],
            neg,
        )),
        LutLookup::rc16(Column::single(m.aligned_incoming_carry)),
        LutLookup::rc16_filtered(
            Column::linear_combination_with_constant([(m.group_output_significand, one)], -F::from_canonical_u64(1 << 13)),
            nonzero_local.clone(),
        ),
        LutLookup::rc16(Column::single(m.cell_result_f32_lo)),
        LutLookup::rc16(Column::single(m.cell_result_f32_hi)),
        // E_CELL bounds the exact, unrounded C+c binade. The projected FP32 exponent equals
        // that binade except for an RNE carry and the far power-of-two subtraction case.
        LutLookup::rc16(Column::linear_combination([
            (m.e_cell, one),
            (m.promotion.global_out_exponent, neg),
            (m.promotion.binade_correction, one),
        ])),
        // E_GRID includes the M branch and the Hopper Z branch for every local c.
        LutLookup::rc16(Column::linear_combination([(m.e_grid, one), (m.e_cell, neg)])),
        // The shared grid uses 18 fractional bits: Hopper's 13-bit ULP contributes +5.
        LutLookup::rc16_filtered(
            Column::linear_combination_with_constant(
                [(m.e_grid, one), (m.group_output_exponent, neg)],
                -F::from_canonical_u64(5),
            ),
            nonzero_local.clone(),
        ),
        LutLookup {
            table: LutTable::Pow2G,
            keys: vec![Column::linear_combination_and_next_row_with_constant(
                vec![(m.group_output_exponent, neg)],
                vec![(m.group_max_product_exponent, one)],
                F::ZERO,
            )],
            values: vec![
                Column::single_next_row(m.incoming_carry_shift_power),
                Column::constant(F::ZERO),
            ],
            filter: Filter::from_column(Column::linear_combination_and_next_row_with_constant(
                vec![],
                vec![(m.incoming_carry_is_zero, neg)],
                one,
            )),
        },
        // Tagged flags prevent an out-of-range key from matching a different operation.
        LutLookup {
            table: LutTable::Pow2G,
            keys: vec![Column::linear_combination_with_constant(
                [(m.promotion.exponent_gap, one)],
                F::from_canonical_u64(128),
            )],
            values: vec![
                Column::single(m.promotion.gap_power),
                Column::linear_combination_with_constant([(m.promotion.far_active, one)], F::TWO),
            ],
            filter: Filter::from_column(Column::linear_combination([
                (m.promotion.near_active, one),
                (m.promotion.far_active, one),
            ])),
        },
        LutLookup {
            table: LutTable::Pow2G,
            keys: vec![Column::linear_combination_with_constant(
                [(m.promotion.exact_width, one)],
                F::from_canonical_u64(256),
            )],
            values: vec![
                Column::single(m.promotion.shift_power),
                Column::linear_combination_with_constant([(m.promotion.near_low, one)], F::from_canonical_u64(4)),
            ],
            filter: Filter::from_column(Column::linear_combination([
                (m.promotion.near_low, one),
                (m.promotion.near_high, one),
            ])),
        },
        LutLookup {
            table: LutTable::WidthNorm,
            keys: vec![Column::single(m.group_sum_abs)],
            values: Column::singles([m.group_sum_width, m.group_output_significand]).collect(),
            filter: nonzero_local,
        },
    ];

    // Sixteen-bit limbs bound normalization, remainder comparisons, and all field products.
    let p = &m.promotion;
    for column in p
        .quotient_limbs
        .into_iter()
        .chain(p.magnitude_or_remainder_limbs)
        .chain(p.remainder_bound_limbs)
        .chain(p.remainder_cmp_limbs)
    {
        lookups.push(LutLookup::rc16(Column::single(column)));
    }
    // With a 16-bit low limb, 128 <= Q_hi <= 255 proves 2^23 <= Q < 2^24.
    lookups.push(LutLookup::rc16_filtered(
        Column::linear_combination_with_constant(
            [(p.quotient_limbs[1], F::from_canonical_u64(512))],
            -F::from_canonical_u64(128 * 512),
        ),
        Filter::from_column(Column::linear_combination([(p.near_low, one), (p.near_high, one)])),
    ));
    // The low limb and Boolean LSB determine integer parity; the high limb is even.
    let half = F::TWO.inverse();
    lookups.push(LutLookup::rc16_filtered(
        Column::linear_combination([(p.quotient_limbs[0], half), (p.quotient_lsb, -half)]),
        Filter::from_column(Column::single(p.near_high)),
    ));

    let byte_shift = F::from_canonical_u64(1 << 8);
    let shift_shift = F::from_canonical_u64(1 << 16);
    for i in 0..GROUP_WIDTH {
        lookups.push(LutLookup {
            table: LutTable::ProdAlign15,
            keys: vec![Column::linear_combination([
                (m.operand_codes_a[i], one),
                (m.operand_codes_b[i], byte_shift),
                (m.group_max_product_exponent, shift_shift),
                (m.prod_fp22_exp[i], -shift_shift),
            ])],
            values: Column::singles([
                m.aligned_lane_terms[i],
                m.prod_fp22_exp[i],
                m.operand_codes_a[i],
                m.operand_codes_b[i],
                m.lane_binades[i],
            ])
            .collect(),
            filter: Filter::default(),
        });
    }

    for i in 0..GROUP_WIDTH {
        lookups.push(LutLookup::rc16(Column::linear_combination([
            (m.e_cell, one),
            (m.lane_binades[i], neg),
        ])));
        lookups.push(LutLookup::rc16_filtered(
            Column::linear_combination_with_constant([(m.e_grid, one), (m.lane_binades[i], neg)], -F::from_canonical_u64(5)),
            Filter::from_column(Column::single(m.cell_nonzero)),
        ));
    }

    let offset = -F::from_canonical_u64(SKIP_THRESHOLD_OFFSET);
    let neg128 = -F::from_canonical_u64(128);
    for i in 0..GROUP_WIDTH {
        lookups.push(LutLookup::rc16_filtered(
            Column::linear_combination_with_constant([(m.lambda_a[i], one), (m.lambda_b[i], one), (m.e_grid, neg128)], offset),
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
        assert_eq!(ctl_operand_codes_looking_matmul::<F>().len(), GROUP_WIDTH);
        ctl_cell_results_looked_matmul::<F>();
    }

    #[test]
    fn lut_inventory_matches_documented_counts() {
        let lookups = matmul_lut_lookups::<F>();
        let count = |table| lookups.iter().filter(|lookup| lookup.table == table).count();
        assert_eq!(count(LutTable::Range16), 115);
        assert_eq!(count(LutTable::ProdAlign15), 32);
        assert_eq!(count(LutTable::Pow2G), 3);
        assert_eq!(count(LutTable::WidthNorm), 1);
        assert_eq!(lookups.len(), 151);
    }
}
