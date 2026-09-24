//! Lookup descriptors for the TamedStark computation.
//!
//! The two import channels bind every live row's values to the committed sources: the E-cell
//! channel consumes each Matmul cell's replay-magnitude binade and skip census exactly once,
//! and the sigma channel consumes each Scale row's noise-std tuple once per cell of its tile
//! row/column (Scale's looked filter carries the `w`/`h` multiplicities). Committed lookup
//! tables then range-check the certificate's digit and carry columns, bind the XFPOW2
//! saturating power `2^A`, gate the running untamed count against the `TAME_LIMIT` public
//! input on the last row (J6), and range-check the skip budget gate's slack limbs (J7).
//!
//! The inventory is RC16 x24 and XFPOW2 x1 — 25 instances. The RC16 digit keys of the J5
//! borrow chain (six affine expressions) make the six-digit comparison `2^A <= Y` exact;
//! the remaining RC16s pin every committed limb and carry to the widths the J2-J3 position
//! equations assume.

use plonky2::field::types::Field;
use starky::cross_table_lookup::TableWithColumns;
use starky::lookup::{Column, Filter};

use super::super::ctl::Table;
use super::super::luts::LutTable;
use super::super::luts::ctl::LutLookup;
use super::columns::{COMPARISON_DIGITS, TAME_LIMIT_PUBLIC_INPUT, TAMED_COL_MAP};
use super::stark::FRAME_GAP_OFFSET;
use crate::circuit::fp8::luts::XFPOW2_ZERO_POINT;

// ==================================================================================================
// Channel: E-cell binades — Tamed (looking) -> Matmul (looked)
// ==================================================================================================

/// Tamed's looking half of the E-cell channel: `(CELL_ID, E_CELL, CELL_SKIPS)` per live row,
/// filter `1 - IS_PAD` — each cell's binade and skip census is read exactly once, matching
/// the Matmul side's one-tuple-per-live-cell filter (`ctl_e_cell_looked_matmul[_b200]`).
pub fn ctl_e_cell_looking_tamed<F: Field>() -> TableWithColumns<F> {
    let m = &TAMED_COL_MAP;
    TableWithColumns::new(
        Table::Tamed.into(),
        Column::singles([m.cell_id, m.e_cell, m.cell_skips]).collect(),
        Filter::from_column(Column::linear_combination_with_constant([(m.is_pad, -F::ONE)], F::ONE)),
    )
}

// ==================================================================================================
// Channel: sigma frames — Tamed (looking) -> Scale (looked)
// ==================================================================================================

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

// ==================================================================================================
// Committed LUT oracle instances
// ==================================================================================================

/// TamedStark's per-row LUT instance inventory: RC16 x24, XFPOW2 x1 — 25 instances. Order:
/// the J2-J3 limb/carry widths, the J5 borrow-chain digit keys, the J4 shift binding, the
/// J6 count gate, then the J7 budget-gate slack limbs.
pub fn tamed_lut_lookups<F: Field>() -> Vec<LutLookup<F>> {
    let m = &TAMED_COL_MAP;
    let one = F::ONE;
    let neg = -F::ONE;
    let limb_shift = F::from_canonical_u64(1 << 16);
    let mut lookups: Vec<LutLookup<F>> = Vec::new();

    // ---- J2: the sigma product split. ----
    for c in &m.sigma_product_limbs {
        lookups.push(LutLookup::rc16(Column::single(*c)));
    }

    // ---- J3: the two K partial products (< 2^32: two full limbs each), the W mid digit,
    // and the Y digits/carries (kappa_2..kappa_3 are 17-bit: lo + bit). ----
    for partial in [&m.k_pp_lo_partial, &m.k_pp_hi_partial] {
        for c in partial {
            lookups.push(LutLookup::rc16(Column::single(*c)));
        }
    }
    for c in m.k_pp_mid_limbs.iter().chain(&m.bound_limbs) {
        lookups.push(LutLookup::rc16(Column::single(*c)));
    }
    lookups.push(LutLookup::rc16(Column::single(m.bound_carry_1)));
    for c in &m.bound_carries_lo {
        lookups.push(LutLookup::rc16(Column::single(*c)));
    }
    lookups.push(LutLookup::rc16(Column::single(m.bound_top)));

    // ---- J5: the borrow-chain digit keys — RC16(Y_w - L_w - BORROW_w + 2^16*BORROW_{w+1})
    // over the six positions, L the one-hot limbs of 2^A (no borrow into position 0, none
    // out of position 5). Y's digits are BOUND_LIMBS then BOUND_TOP; its top position is
    // structurally zero (Y = K*PP^2 < 2^80 by the J3 recomposition), so the last key has no
    // Y term — a saturated 2^80 claim or a top borrow makes it negative, and RC16 rejects.
    // Every operand is 16-bit-bound on active rows, so the keys are in range iff the
    // digit-wise subtraction never goes negative — 2^A <= Y over Z. Unfiltered: gate-off
    // rows hold the all-zero fill (key 0). ----
    for w in 0..COMPARISON_DIGITS {
        let mut terms = vec![(m.shift_a_limbs[w], neg)];
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

    // ---- J4: the XFPOW2 saturating power at the tau-folded doubled frame gap, keyed
    // `E_CELL - SIGMA_A_EXP - SIGMA_B_EXP + 4974` (the table decodes `D = 2*(key - 1024)`
    // and 4974 = FRAME_GAP_OFFSET + the zero point; the key domain doubles as the frame-gap
    // window proof). Active exactly on live tamed nonzero rows: untamed and zero-cell rows
    // prove nothing, pad rows' zero limbs feed the vacuous J5 keys. ----
    lookups.push(LutLookup {
        table: LutTable::XfPow2,
        keys: vec![Column::linear_combination_with_constant(
            [(m.e_cell, one), (m.sigma_a_exp, neg), (m.sigma_b_exp, neg)],
            F::from_canonical_u64(FRAME_GAP_OFFSET as u64 + XFPOW2_ZERO_POINT),
        )],
        values: m.shift_a_limbs.iter().map(|&c| Column::single(c)).collect(),
        filter: Filter::new(
            vec![(
                Column::linear_combination_with_constant([(m.untamed, neg), (m.cell_is_zero, neg)], one),
                Column::linear_combination_with_constant([(m.is_pad, neg)], one),
            )],
            vec![],
        ),
    });

    // ---- J6: the jackpot count gate, on the last row only (filter IS_LAST_ROW, a known
    // column): RC16(TAME_LIMIT - RUNNING_UNTAMED) proves the untamed count is within the
    // allowance. One limb decides: the count is at most `h*w < 2^16` (pads pinned tamed) and
    // the limit is `floor(h*w/64) < 2^16`, so an over-limit count can never wrap into
    // range. ----
    lookups.push(LutLookup::rc16_filtered(
        Column::public_input_minus_single(TAME_LIMIT_PUBLIC_INPUT, m.running_untamed),
        Filter::from_column(Column::single(m.is_last_row)),
    ));

    // ---- J7: the skip budget gate's slack limbs (jackpot check 4). The last-row
    // recomposition `SKIP_LIMIT - RUNNING_SKIPS = LO + 2^16*HI` lives in `super::stark`;
    // these RC16s bound the limbs, so the difference lies in [0, 2^32) — the census is
    // within the budget. Unfiltered: non-last rows hold the all-zero fill. ----
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
