//! Lookup and cross-table channels for B200 matrix multiplication.
//!
//! Operand CTLs import pair-packed FP8 codes, B200ALIGN aligns each lane, POW2GB aligns the
//! incoming carry, WIDTH32 truncates the group sum, and the result CTL exports each final FP32
//! cell.

use plonky2::field::types::Field;
use starky::cross_table_lookup::TableWithColumns;
use starky::lookup::{Column, Filter};

use super::columns::{GROUP_WIDTH, MATMUL_B200_COL_MAP};
use super::stark::PARTIAL_BINADE_OFFSET;
use crate::circuit::fp8::ctl::{LutLookup, LutTable, Table};
use crate::circuit::fp8::unpredictability::SKIP_THRESHOLD_OFFSET;

/// Matmul's looking side of the operand-code channel: 16 pair-packed instances per side
/// (A first, then B — 32 in total), filter `1 - IS_PADDING`, each element paired with its
/// summand score `lambda`; the padding rows (trailing phantom cells) read no operands.
/// Instance `i` of a side sends
/// `(OPERAND_INDEX_BASE + 2i, OPERAND_CODES_{2i} + 2^8*OPERAND_CODES_{2i+1}, LAMBDA_{2i},
/// LAMBDA_{2i+1})`; `OPERAND_INDEX_BASE_B` already carries the `h*k` B-plane key offset, so
/// the A and B tuples occupy disjoint key spaces.
///
/// Sound as a pair-packing because each code column is individually pinned into
/// `[0, 256)`: here by MB1's `OPERAND_CODES_A/OPERAND_CODES_B` binding, on the InputQuant
/// side by the tuple-valued QCAST binding. The two lambda scores ride as separate slots
/// (never packed), so each lane's lambda inherits InputQuant's exact value and range by
/// slot-wise equality.
///
/// Looked side: `input_quant_stark::ctl::ctl_operand_codes_looked_input_quant` (slots A, B),
/// whose filter value `w * IS_EVEN_ROW` (A) resp. `h * IS_EVEN_ROW` (B) supplies the
/// looked-side multiplicity — each element is read once per opposite-side cell.
pub fn ctl_operand_codes_looking_matmul_b200<F: Field>() -> Vec<TableWithColumns<F>> {
    let m = &MATMUL_B200_COL_MAP;
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

/// Matmul's looked side of the cell-result channel: `(CELL_ID, CELL_RESULT_F32_LO,
/// CELL_RESULT_F32_HI)`, filter `IS_CELL_FINAL * (1 - IS_PADDING)` — phantom padding cells
/// emit no word. XorFold receives each finished real cell's f32 word (as two RC16'd limbs)
/// exactly once, keyed by the cell it folds.
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

/// Matmul's looked side of the E-cell channel (jackpot checks 3 + 4):
/// `(CELL_ID, E_CELL, CELL_SKIPS)` on the cell-final row of every live cell, with
/// `E_CELL = floor(log2 M) + 139` and `CELL_SKIPS` the cell's skip census (MB16).
pub fn ctl_e_cell_looked_matmul_b200<F: Field>() -> TableWithColumns<F> {
    let m = &MATMUL_B200_COL_MAP;
    TableWithColumns::new(
        Table::Matmul.into(),
        Column::singles([m.cell_id, m.e_cell, m.cell_skips]).collect(),
        Filter::new(
            vec![(
                Column::single(m.is_cell_final),
                Column::linear_combination_with_constant([(m.is_padding, -F::ONE)], F::ONE),
            )],
            vec![],
        ),
    )
}

/// MatmulB200Stark's per-row LUT instance inventory: RC16 x78 (13 window-sum + 33 check 3:
/// `E_CELL - LANE_BINADES_i >= 0` on the 32 lanes and the filtered partial-sum bound + 32
/// check 4: the per-lane non-skip certificates), B200ALIGN x32, POW2GB x1 (filtered),
/// WIDTH32 x1 (filtered) — 112 instances.
/// Each becomes a CTL into the matching committed `LutStark` (`lut_cross_table_lookups`).
/// The inventory is program-independent.
///
/// The high limbs of the 26-bit carry quotient/remainder/bound are pinned by scaled RC16s
/// (`HI * 2^6 < 2^16`). These need no unshifted companion: the scale divides the `2^16`
/// recomposition weight exactly (`HI * 2^16 = (HI * 2^6) * 2^10`), so any alias of `HI`
/// re-splits the *same* bounded integer `(HI*2^6 mod p)*2^10 + LO < 2^26 + 2^16` — the MB5/MB6
/// equations only ever see the recomposition, which stays integer-exact.
///
/// The normalized-significand high limb is different: its value feeds the output-significand
/// mux bound. The scaled floor check `(HI - 128) * 2^9 < 2^16` alone leaves an overshoot window
/// `[2^24, 2^24 + 2^16)` (`key*2^7 + 2^23 + LO` with `key, LO < 2^16`).
/// A sum barely above a power of two could then claim width `W - 1` with a 25-bit
/// "normalized" significand, exiting MB10/MB12 as a different `CELL_RESULT_F32` bit pattern for
/// the same cell. The unshifted `RC16(HI)` companion closes the window: together they pin
/// `HI` in `[128, 256)` exactly, i.e. the normalized significand in `[2^23, 2^24)`.
///
/// The MB5 carry constraints are anchored at the producing row, so the POW2GB instance keys
/// next-row columns against the local `GROUP_OUTPUT_BIASED_EXPONENT` — expressible because
/// `Column` supports next-row linear combinations. Next-row reads are cyclic (row n-1's
/// "next" is row 0), but the wrap instance is filtered out by row 0's
/// `INCOMING_CARRY_IS_ZERO = 1` (the MB4 first-row anchor); dishonestly turning it *on* only
/// adds a key-domain obligation on the forged trace.
pub fn matmul_b200_lut_lookups<F: Field>() -> Vec<LutLookup<F>> {
    let m = &MATMUL_B200_COL_MAP;
    let one = F::ONE;
    let neg = -F::ONE;
    let hi10 = F::from_canonical_u64(1 << 6);
    let group_sum_is_nonzero = Filter::from_column(Column::linear_combination_with_constant([(m.group_sum_is_zero, neg)], one));

    let mut lookups = vec![
        // ---- MB5: carry alignment witnesses — 16 + 10-bit limb pairs of the quotient, the
        // remainder and its bound (the quotient range is what makes the floor equation exact
        // over the integers — without it the field admits a fractional-quotient alias). ----
        LutLookup::rc16(Column::single(m.aligned_incoming_carry_lo)),
        LutLookup::rc16(Column::linear_combination([(m.aligned_incoming_carry_hi, hi10)])),
        LutLookup::rc16(Column::single(m.incoming_carry_remainder_lo)),
        LutLookup::rc16(Column::linear_combination([(m.incoming_carry_remainder_hi, hi10)])),
        LutLookup::rc16(Column::single(m.incoming_carry_remainder_bound_lo)),
        LutLookup::rc16(Column::linear_combination([(m.incoming_carry_remainder_bound_hi, hi10)])),
        // ---- MB7: normalized-significand limbs. The LO check, unshifted HI bound, and
        // normalization floor pin the value into [2^23, 2^24) on live rows. ----
        LutLookup::rc16(Column::single(m.normalized_group_sum_significand_lo)),
        LutLookup::rc16(Column::single(m.normalized_group_sum_significand_hi)),
        LutLookup::rc16_filtered(
            Column::linear_combination_with_constant(
                [(m.normalized_group_sum_significand_hi, F::from_canonical_u64(1 << 9))],
                -F::from_canonical_u64(128 << 9),
            ),
            group_sum_is_nonzero.clone(),
        ),
        // ---- MB7: the Euclidean remainder and its bound (< TRUNCATION_POWER <= 2^8). ----
        LutLookup::rc16(Column::single(m.truncation_remainder)),
        LutLookup::rc16(Column::single(m.truncation_remainder_bound)),
        // ---- MB12: f32 limb splits. ----
        LutLookup::rc16(Column::single(m.cell_result_f32_lo)),
        LutLookup::rc16(Column::single(m.cell_result_f32_hi)),
        // ---- MB13 (jackpot check 3):
        // E_CELL >= GROUP_OUTPUT_BIASED_EXPONENT + 101
        // (see `PARTIAL_BINADE_OFFSET`). Zero partials have no binade:
        // the filter skips them. ----
        LutLookup::rc16_filtered(
            Column::linear_combination_with_constant(
                [(m.e_cell, one), (m.group_output_biased_exponent, neg)],
                -F::from_canonical_u64(PARTIAL_BINADE_OFFSET),
            ),
            group_sum_is_nonzero.clone(),
        ),
        // ---- MB5: POW2GB, keyed at the producing row against the consuming (next) row's
        // shift witness: POW2GB(GROUP_MAX_BIASED_EXPONENT' - GROUP_OUTPUT_BIASED_EXPONENT;
        // INCOMING_CARRY_SHIFT_POWER'), filter 1 - INCOMING_CARRY_IS_ZERO' (out-of-domain
        // negative keys enforce GROUP_MAX_BIASED_EXPONENT' >= GROUP_OUTPUT_BIASED_EXPONENT; the
        // wrap instance is inert because row 0's flag is anchored 1). ----
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
        // ---- MB7: WIDTH32 on the claimed width — the [1, 32] key domain is the width range
        // proof (there is no key 0 row: a zero width cannot be served). ----
        LutLookup {
            table: LutTable::Width32,
            keys: vec![Column::single(m.group_sum_width)],
            values: Column::singles([m.truncation_power, m.lifting_power]).collect(),
            filter: group_sum_is_nonzero,
        },
    ];

    // ---- MB1: B200ALIGN x32 — the whole per-lane product, pre-truncated to the window,
    // plus its binade for check 3. ----
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

    // ---- MB13: E_CELL >= LANE_BINADES_i on every lane. Zero products (binade 0) always
    // pass; nonzero binades are >= 121, so E_CELL = 0 implies an all-zero cell. ----
    for i in 0..GROUP_WIDTH {
        lookups.push(LutLookup::rc16(Column::linear_combination([
            (m.e_cell, one),
            (m.lane_binades[i], neg),
        ])));
    }

    // ---- MB15: the per-lane non-skip certificates (jackpot check 4) — the affine key
    // LAMBDA_A_i + LAMBDA_B_i - 128*E_CELL - 45376 under the filter
    // CELL_NONZERO * (1 - SKIP_FLAG_i). A true skip's key is negative and wraps far outside
    // [0, 2^16), forcing SKIP_FLAG_i = 1; zero cells and padding pin the flag to 0 and the
    // filter is off, so they pay no certificate. ----
    let offset = -F::from_canonical_u64(SKIP_THRESHOLD_OFFSET);
    let neg128 = -F::from_canonical_u64(128);
    for i in 0..GROUP_WIDTH {
        lookups.push(LutLookup::rc16_filtered(
            Column::linear_combination_with_constant([(m.lambda_a[i], one), (m.lambda_b[i], one), (m.e_cell, neg128)], offset),
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
