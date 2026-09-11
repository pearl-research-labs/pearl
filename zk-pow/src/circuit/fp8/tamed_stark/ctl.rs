//! Tamed's import channels and lookup tables.
//!
//! Each live row imports one cell's magnitude bound and skip count from Matmul.
//! It also imports the cell's A-row and B-column noise scales from Scale. A Scale
//! tuple is used once per cell in its tile row or column; Scale's looked-side
//! multiplicities are therefore `w` for A and `h` for B.
//!
//! RC16 lookups constrain integers to `[0, 2^16)`. They bound the certificate's
//! limbs and carries, enforce its subtraction digits, and check the final budgets.
//! XFPOW2 supplies the clamped comparison power. [`super::stark`] derives the
//! certificate `2^D <= Y` and explains the J1–J7 constraint groups.

use plonky2::field::types::Field;
use starky::cross_table_lookup::TableWithColumns;
use starky::lookup::{Column, Filter};

use super::super::ctl::{LutLookup, LutTable, Table};
use super::columns::{COMPARISON_DIGITS, TAME_LIMIT_PUBLIC_INPUT, TAMED_COL_MAP};
use super::stark::FRAME_GAP_OFFSET;
use crate::circuit::fp8::luts::XFPOW2_ZERO_POINT;

// Channel: E-cell binades — Tamed (looking) -> Matmul (looked)

/// Imports `(cell_id, magnitude_exponent_bound, skip_count)` once per live cell.
pub fn ctl_e_cell_looking_tamed<F: Field>() -> TableWithColumns<F> {
    let m = &TAMED_COL_MAP;
    TableWithColumns::new(
        Table::Tamed.into(),
        Column::singles([m.cell_id, m.cell_magnitude_exponent, m.cell_skips]).collect(),
        Filter::from_column(Column::linear_combination_with_constant([(m.is_pad, -F::ONE)], F::ONE)),
    )
}

// Channel: sigma frames — Tamed (looking) -> Scale (looked)

/// Tamed's looking half of the sigma channel: two slots per live row —
/// `(A_GROUP_KEY, SIGMA_A_SIGNIFICAND, SIGMA_A_EXP)` and the B counterpart — filter
/// `1 - IS_PAD`. The A and B group keys occupy disjoint spaces (`(i+1)*k - 1` vs
/// `h*k + (j+1)*k - 1`), and each Scale row's tuple is read once per cell of its tile
/// row/column, matching Scale's `w`/`h` public-input multiplicities
/// (`ctl_sigma_looked_scale`).
pub fn ctl_sigma_looking_tamed<F: Field>() -> Vec<TableWithColumns<F>> {
    let m = &TAMED_COL_MAP;
    let not_pad = || Filter::from_column(Column::linear_combination_with_constant([(m.is_pad, -F::ONE)], F::ONE));
    vec![
        TableWithColumns::new(
            Table::Tamed.into(),
            Column::singles([m.a_group_key, m.sigma_a_significand, m.sigma_a_exp]).collect(),
            not_pad(),
        ),
        TableWithColumns::new(
            Table::Tamed.into(),
            Column::singles([m.b_group_key, m.sigma_b_significand, m.sigma_b_exp]).collect(),
            not_pad(),
        ),
    ]
}

// Committed LUT oracle instances

/// TamedStark's per-row LUT instance inventory: RC16 x24, XFPOW2 x1 — 25 instances. Order:
/// the J2-J3 limb/carry widths, the J5 borrow-chain digit keys, the J4 shift binding, the
/// J6 count gate, then the J7 budget-gate slack limbs.
pub fn tamed_lut_lookups<F: Field>() -> Vec<LutLookup<F>> {
    let m = &TAMED_COL_MAP;
    let one = F::ONE;
    let neg = -F::ONE;
    let limb_shift = F::from_canonical_u64(1 << 16);
    let mut lookups: Vec<LutLookup<F>> = Vec::new();

    // J2: the sigma product split.
    for c in &m.sigma_product_limbs {
        lookups.push(LutLookup::rc16(Column::single(*c)));
    }

    // J3: the two K partial products (< 2^32: two full limbs each), the W mid digit,
    // and the Y digits/carries (kappa_2..kappa_3 are 17-bit: lo + bit).
    for partial in [&m.k_sigma_product_lo, &m.k_sigma_product_hi] {
        for c in partial {
            lookups.push(LutLookup::rc16(Column::single(*c)));
        }
    }
    for c in m.k_sigma_product_middle_limbs.iter().chain(&m.bound_limbs) {
        lookups.push(LutLookup::rc16(Column::single(*c)));
    }
    lookups.push(LutLookup::rc16(Column::single(m.bound_carry_1)));
    for c in &m.bound_carries_lo {
        lookups.push(LutLookup::rc16(Column::single(*c)));
    }
    lookups.push(LutLookup::rc16(Column::single(m.bound_top)));

    // J5: with L the base-2^16 digits of 2^A, range-check each subtraction digit:
    //
    // d_i = Y_i - L_i - b_i + 2^16*b_{i+1} in [0, 2^16), with b_0 = b_6 = 0.
    //
    // Summing d_i*2^(16*i) cancels the internal borrows and gives Y - 2^A >= 0.
    // Bounded digits and boolean borrows prevent field wraparound.
    //
    // Since Y < 2^80, digit 5 has no Y term. A clamped 2^80 power or a final
    // borrow makes that key negative and fails RC16.
    // The all-zero fill satisfies these unfiltered checks on inactive rows.
    for w in 0..COMPARISON_DIGITS {
        let mut terms = vec![(m.comparison_power_limbs[w], neg)];
        if w < COMPARISON_DIGITS - 2 {
            terms.push((m.bound_limbs[w], one));
        } else if w == COMPARISON_DIGITS - 2 {
            terms.push((m.bound_top, one));
        }
        if w > 0 {
            terms.push((m.comparison_borrows[w - 1], neg));
        }
        if w < COMPARISON_DIGITS - 1 {
            terms.push((m.comparison_borrows[w], limb_shift));
        }
        lookups.push(LutLookup::rc16(Column::linear_combination(terms)));
    }

    // J4: fetch the comparison power at the half-gap plus XFPOW2_ZERO_POINT.
    //
    // key = cell_magnitude_exponent - sigma_a_exp - sigma_b_exp
    //       + FRAME_GAP_OFFSET + XFPOW2_ZERO_POINT
    // D   = 2*(key - XFPOW2_ZERO_POINT).
    //
    // FRAME_GAP_OFFSET includes both exponent biases and tau. Only live, nonzero
    // cells claimed tamed use this lookup; padding supplies zero comparison limbs.
    lookups.push(LutLookup {
        table: LutTable::XfPow2,
        keys: vec![Column::linear_combination_with_constant(
            [(m.cell_magnitude_exponent, one), (m.sigma_a_exp, neg), (m.sigma_b_exp, neg)],
            F::from_canonical_u64(FRAME_GAP_OFFSET as u64 + XFPOW2_ZERO_POINT),
        )],
        values: m.comparison_power_limbs.iter().map(|&c| Column::single(c)).collect(),
        filter: Filter::new(
            vec![(
                Column::linear_combination_with_constant([(m.untamed, neg), (m.cell_is_zero, neg)], one),
                Column::linear_combination_with_constant([(m.is_pad, neg)], one),
            )],
            vec![],
        ),
    });

    // J6: the jackpot count gate, on the last row only (filter IS_LAST_ROW, a known
    // column): RC16(TAME_LIMIT - RUNNING_UNTAMED) proves the untamed count is within the
    // allowance. One limb decides: the count is at most `h*w < 2^16` (pads pinned tamed) and
    // the limit is `floor(h*w/64) < 2^16`, so an over-limit count can never wrap into
    // range.
    lookups.push(LutLookup::rc16_filtered(
        Column::public_input_minus_single(TAME_LIMIT_PUBLIC_INPUT, m.running_untamed),
        Filter::from_column(Column::single(m.is_last_row)),
    ));

    // J7: the skip budget gate's slack limbs (jackpot check 4). The last-row
    // recomposition `SKIP_LIMIT - RUNNING_SKIPS = LO + 2^16*HI` lives in `super::stark`;
    // these RC16s bound the limbs, so the difference lies in [0, 2^32) — the census is
    // within the budget. Unfiltered: non-last rows hold the all-zero fill.
    lookups.push(LutLookup::rc16(Column::single(m.skip_gate_slack_lo)));
    lookups.push(LutLookup::rc16(Column::single(m.skip_gate_slack_hi)));

    lookups
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;

    use super::*;

    type F = GoldilocksField;

    #[test]
    fn tamed_ctl_halves_are_well_formed() {
        ctl_e_cell_looking_tamed::<F>();
        assert_eq!(ctl_sigma_looking_tamed::<F>().len(), 2, "A and B sigma slots");
    }

    #[test]
    fn tamed_lut_inventory_matches_documented_counts() {
        let lookups = tamed_lut_lookups::<F>();
        let count = |t: LutTable| lookups.iter().filter(|l| l.table == t).count();
        // Documented inventory: RC16 x24, XFPOW2 x1.
        assert_eq!(count(LutTable::Range16), 24);
        assert_eq!(count(LutTable::XfPow2), 1);
        assert_eq!(lookups.len(), 25);
    }
}
