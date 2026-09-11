//! Scale's cross-table channels and lookup tables.
//!
//! The group channel binds InputQuant's completed row statistics, scale claims
//! and normalized noise scales. The sigma channel supplies exact noise scales
//! to Tamed, with multiplicities matching their reuse across output cells.
//!
//! Lookup tables check square-root comparisons, BF16 rounding, norm floors,
//! scale derivation and policy budgets. Constraint labels and mathematical
//! definitions are in [`super::stark`].

use plonky2::field::types::Field;
use starky::cross_table_lookup::TableWithColumns;
use starky::lookup::{Column, Filter};

use super::super::ctl::{LutLookup, LutTable, Table};
use super::columns::{
    ALIGNED_LIMBS, CLAIM_LIMBS, DEAD_LIMIT_A_PUBLIC_INPUT, DEAD_LIMIT_B_PUBLIC_INPUT, H_MULT_PUBLIC_INPUT, L2_SUM_LIMBS,
    SCALE_COL_MAP, W_MULT_PUBLIC_INPUT,
};
use super::stark::ScaleProgram;

// Channel: group tuples — InputQuant (looking) -> Scale (looked)

/// Binds each live InputQuant group to one Scale row; padding consumes no tuple.
/// The tuple contains its key, framed sum/exponent, maximum, alpha/beta fields,
/// liveness bound/count and normalized sigma.
///
/// Two values are affine expressions rather than columns:
///
/// - dead bound: `code(l2f) + 256`, encoding 4*l2f;
/// - biased sigma exponent: `alpha_exp + l2_floored_exponent + sigma_sig_is_wide + 13`.
///
/// InputQuant exports the same tuple order on live group-final rows.
pub fn ctl_looked_scale_group_tuple<F: Field>() -> TableWithColumns<F> {
    let m = &SCALE_COL_MAP;
    let mut columns: Vec<Column<F>> = Column::singles([
        m.group_key,
        m.l2_frame_sum,
        m.frame_doubled_scale_exponent,
        m.max_abs,
        m.alpha_exp,
        m.alpha_mantissa,
        m.beta_exp,
        m.beta_mantissa,
        m.beta_exp_is_zero,
    ])
    .collect();
    columns.push(Column::linear_combination_with_constant(
        [
            (m.l2_floored_exponent, F::from_canonical_u64(128)),
            (m.l2_floored_significand, F::ONE),
        ],
        F::from_canonical_u64(128),
    ));
    columns.push(Column::single(m.dead_count));
    columns.push(Column::linear_combination_with_constant(
        [
            (m.alpha_exp, F::ONE),
            (m.l2_floored_exponent, F::ONE),
            (m.sigma_sig_is_wide, F::ONE),
        ],
        F::from_canonical_u64(13),
    ));
    columns.push(Column::single(m.normalized_sigma_significand));
    TableWithColumns::new(
        Table::Scale.into(),
        columns,
        Filter::from_column(Column::linear_combination_with_constant([(m.is_pad, -F::ONE)], F::ONE)),
    )
}

// Channel: sigma frames — Tamed (looking) -> Scale (looked)

/// `SIGMA_EXP = ALPHA_EXP + L2_FLOORED_EXPONENT + SIGMA_EXP_OFFSET`: folds the two bf16 units
/// (`M * 2^(E - 134)` each, both factors structurally normal), `DELTA = 2^-1`, and the +2048
/// sigma frame bias TamedStark decodes (`2048 - 2*134 - 1 = 1779`).
pub const SIGMA_EXP_OFFSET: u64 = 1779;

/// Export each row's exact noise scale to Tamed as `(GROUP_KEY, SIGMA_SIGNIFICAND, SIGMA_EXP)`:
///
/// `sigma = SIGMA_SIGNIFICAND * 2^(SIGMA_EXP - 2048)`.
///
/// Tamed reads each A row's scale once per tile column, and each B row's once per
/// tile row. The filter therefore counts `w` uses for A, `h` for B, and zero for
/// padding. The verifier binds `w` and `h` to the statement's tile dimensions.
pub fn ctl_sigma_looked_scale<F: Field>() -> TableWithColumns<F> {
    let m = &SCALE_COL_MAP;
    TableWithColumns::new(
        Table::Scale.into(),
        vec![
            Column::single(m.group_key),
            Column::single(m.sigma_significand),
            Column::linear_combination_with_constant(
                [(m.alpha_exp, F::ONE), (m.l2_floored_exponent, F::ONE)],
                F::from_canonical_u64(SIGMA_EXP_OFFSET),
            ),
        ],
        Filter::new(
            vec![
                (Column::single(m.is_a_row), Column::public_input(W_MULT_PUBLIC_INPUT)),
                (
                    Column::linear_combination_with_constant([(m.is_a_row, -F::ONE), (m.is_pad, -F::ONE)], F::ONE),
                    Column::public_input(H_MULT_PUBLIC_INPUT),
                ),
            ],
            vec![],
        ),
    )
}

// Committed LUT oracle instances

/// ScaleStark's per-row LUT instance inventory: RC16 x63, PAIR128 x3, EXPINFO x3, CLAMP22 x3,
/// POW2D x3, RNERND x3, DIV448 x1 — 79 instances. Order: square root (Q), grid snap (G),
/// `linf` (N), the scale chain (H: norm floors, FMA, alpha, beta steps B1/B2), the liveness
/// gates (T3), the noise-floor gate (F3), then the sigma width bit (S2).
pub fn scale_lut_lookups<F: Field>(program: &ScaleProgram) -> Vec<LutLookup<F>> {
    let m = &SCALE_COL_MAP;
    let one = F::ONE;
    let neg = -F::ONE;
    let limb_shift = F::from_canonical_u64(1 << 16);
    let neg_limb_shift = -limb_shift;
    let half = F::TWO.inverse();
    let mut lookups: Vec<LutLookup<F>> = Vec::new();

    // Q1: frame-sum limbs + the canonicity cap (top limb < 2^14 => S < 2^62 < p).
    for i in 0..L2_SUM_LIMBS {
        lookups.push(LutLookup::rc16(Column::single(m.frame_sum_limbs[i])));
    }
    lookups.push(LutLookup::rc16(Column::linear_combination([(
        m.frame_sum_limbs[L2_SUM_LIMBS - 1],
        F::from_canonical_u64(4),
    )])));

    // Q3: EXPINFO bounds the exponent; PAIR128 bounds both mantissa and its half for exact parity.
    lookups.push(LutLookup {
        table: LutTable::ExpInfo,
        keys: vec![Column::single(m.sqrt_exp)],
        values: vec![Column::single(m.sqrt_exp_is_zero)],
        filter: Filter::default(),
    });
    lookups.push(LutLookup {
        table: LutTable::Pair128,
        keys: vec![Column::single(m.sqrt_mantissa), Column::single(m.sqrt_mantissa_half)],
        values: vec![],
        filter: Filter::default(),
    });

    // Q7: midpoint-square limbs; MSQ = B_LO^2 < 2^20, so limb 1 < 2^4.
    lookups.push(LutLookup::rc16(Column::single(m.lower_boundary_squared_limbs[0])));
    lookups.push(LutLookup::rc16(Column::single(m.lower_boundary_squared_limbs[1])));
    lookups.push(LutLookup::rc16(Column::linear_combination([(
        m.lower_boundary_squared_limbs[1],
        F::from_canonical_u64(1 << 12),
    )])));

    // Q7: claim-side products B^2 * k * 2^15 < 2^53 — limbs 1..=3 committed (limb 0 is
    // provably zero: 32 | k), top limb < 2^5, capped at < 2^4 via the 2^12 scale (the honest
    // bound is 2^52; the cap keeps the recomposition < 2^53 < p, alias-free).
    for limbs in [&m.lower_boundary_product_limbs, &m.upper_boundary_product_limbs] {
        for i in 0..CLAIM_LIMBS {
            lookups.push(LutLookup::rc16(Column::single(limbs[i])));
        }
        lookups.push(LutLookup::rc16(Column::linear_combination([(
            limbs[CLAIM_LIMBS - 1],
            F::from_canonical_u64(1 << 12),
        )])));
    }

    // Q5: the mod-16 shift residue r2 in [0, 15]: POW2D pins r2 >= 0 (key domain) and the
    // power; RC16(15 - r2) the cap.
    lookups.push(LutLookup {
        table: LutTable::Pow2D,
        keys: vec![Column::single(m.sum_shift_remainder)],
        values: vec![Column::single(m.sum_shift_power)],
        filter: Filter::default(),
    });
    lookups.push(LutLookup::rc16(Column::linear_combination_with_constant(
        [(m.sum_shift_remainder, neg)],
        F::from_canonical_u64(15),
    )));

    // Q6: shifted-sum limbs and their carries (each per-limb product equation is < 2^32 on
    // both sides given these ranges, hence exact over Z).
    for i in 0..L2_SUM_LIMBS {
        lookups.push(LutLookup::rc16(Column::single(m.shifted_sum_limbs[i])));
    }
    for i in 0..L2_SUM_LIMBS {
        lookups.push(LutLookup::rc16(Column::single(m.shifted_sum_carries[i])));
    }

    // Q8: the two borrow chains as per-digit affine RC16 keys. Digit ranges force each
    // borrow bit and make the telescoped comparison exact over Z (no negative-difference alias:
    // every digit is small, so the field equation is the integer equation).
    //
    // Layout (with the +32 rebias): 8 positions. Shifted sum SS8 = [0, 0, SS_0..SS_3,
    // SHIFTED_SUM_CARRIES_3, 0] (i.e. SS * 2^32, positions 2..=6), aligned boundaries
    // ACL8/ACU8 = [0, AC_0..AC_6] (positions 1..=7, limb offset q in {0..4}).
    //
    // LEFT (aligned lower boundary + ODD <= SS * 2^32): minuend SS8, subtrahend ACL8, ODD
    // subtracted at digit 0:
    //   digit_0 = -ODD + 2^16*BL_0                                  (both arrays 0 at position 0)
    //   digit_i = SS8_i - ACL_{i-1} - BL_{i-1} + 2^16*BL_i          (i = 1..=6)
    //   digit_7 = -ACL_6 - BL_6                                     (top: no borrow out; SS8_7 = 0)
    // (digit_7 forces ACL_6 = BL_6 = 0 — an accepted lower arm never carries a borrow past the
    // shifted sum's span, and a nonzero top aligned limb means the claim side exceeds
    // SS * 2^32 < 2^109 outright; see stark.rs.)
    let ss8 = |i: usize| -> Option<usize> {
        match i {
            0 | 1 | 7 => None, // structural zeros of SS * 2^32
            2..=5 => Some(m.shifted_sum_limbs[i - 2]),
            6 => Some(m.shifted_sum_carries[3]),
            _ => unreachable!(),
        }
    };
    lookups.push(LutLookup::rc16(Column::linear_combination([
        (m.sqrt_mantissa_parity, neg),
        (m.lower_comparison_borrows[0], limb_shift),
    ])));
    for i in 1..=6 {
        let mut terms = vec![
            (m.aligned_lower_boundary_limbs[i - 1], neg),
            (m.lower_comparison_borrows[i - 1], neg),
            (m.lower_comparison_borrows[i], limb_shift),
        ];
        if let Some(ss) = ss8(i) {
            terms.push((ss, one));
        }
        lookups.push(LutLookup::rc16(Column::linear_combination(terms)));
    }
    lookups.push(LutLookup::rc16(Column::linear_combination([
        (m.aligned_lower_boundary_limbs[ALIGNED_LIMBS - 1], neg),
        (m.lower_comparison_borrows[6], neg),
    ])));
    // RIGHT (SS * 2^32 + ODD <= aligned upper boundary): minuend ACU8, subtrahend SS8, ODD at
    // digit 0:
    //   digit_0 = -ODD + 2^16*BR_0
    //   digit_i = ACU_{i-1} - SS8_i - BR_{i-1} + 2^16*BR_i          (i = 1..=6)
    //   digit_7 = ACU_6 - BR_6
    lookups.push(LutLookup::rc16(Column::linear_combination([
        (m.sqrt_mantissa_parity, neg),
        (m.upper_comparison_borrows[0], limb_shift),
    ])));
    for i in 1..=6 {
        let mut terms = vec![
            (m.aligned_upper_boundary_limbs[i - 1], one),
            (m.upper_comparison_borrows[i - 1], neg),
            (m.upper_comparison_borrows[i], limb_shift),
        ];
        if let Some(ss) = ss8(i) {
            terms.push((ss, neg));
        }
        lookups.push(LutLookup::rc16(Column::linear_combination(terms)));
    }
    lookups.push(LutLookup::rc16(Column::linear_combination([
        (m.aligned_upper_boundary_limbs[ALIGNED_LIMBS - 1], one),
        (m.upper_comparison_borrows[6], neg),
    ])));

    // G2: the snap remainder window, two-sided.
    lookups.push(LutLookup::rc16(Column::single(m.grid_snap_remainder)));
    lookups.push(LutLookup::rc16(Column::linear_combination_with_constant(
        [(m.grid_snap_remainder, neg)],
        F::from_canonical_u64(3),
    )));

    // G3: snapped-l2 decode — EXPINFO (also rejects a snap into the inf field: exponent
    // 255 has no row) and the shared (L2_MANTISSA, LINF_MANTISSA) pair.
    lookups.push(LutLookup {
        table: LutTable::ExpInfo,
        keys: vec![Column::single(m.l2_exp)],
        values: vec![Column::single(m.l2_exp_is_zero)],
        filter: Filter::default(),
    });
    lookups.push(LutLookup {
        table: LutTable::Pair128,
        keys: vec![Column::single(m.l2_mantissa), Column::single(m.linf_mantissa)],
        values: vec![],
        filter: Filter::default(),
    });

    // N1: linf decode — EXPINFO (mantissa rides the G3 pair).
    lookups.push(LutLookup {
        table: LutTable::ExpInfo,
        keys: vec![Column::single(m.linf_exp)],
        values: vec![Column::single(m.linf_exp_is_zero)],
        filter: Filter::default(),
    });

    // H0: the two norm-floor MAXes' order slacks (l2 and linf vs 2^-32; the muxes are
    // arithmetic constraints).
    lookups.push(LutLookup::rc16(Column::single(m.l2_floor_order_slack)));
    lookups.push(LutLookup::rc16(Column::single(m.linf_floor_order_slack)));

    // H1 (FMA), W2/W3: the order and far-gap slacks (0 on the inactive side, so
    // unfiltered is complete).
    lookups.push(LutLookup::rc16(Column::single(m.noised_bound_fma.scale_gap_slack)));
    lookups.push(LutLookup::rc16(Column::single(m.noised_bound_fma.far_gap_slack)));

    // H1, W4/W8: the two POW2D powers (key domains are the range proofs).
    lookups.push(LutLookup {
        table: LutTable::Pow2D,
        keys: vec![Column::single(m.noised_bound_fma.exp_gap_capped)],
        values: vec![Column::single(m.noised_bound_fma.exp_gap_pow2)],
        filter: Filter::default(),
    });
    lookups.push(LutLookup {
        table: LutTable::Pow2D,
        keys: vec![Column::single(m.noised_bound_fma.compression_shift)],
        values: vec![Column::single(m.noised_bound_fma.shift_pow2)],
        filter: Filter::default(),
    });

    // H1, W8: wide split floor sandwich. K >= 2^16 here; the rounding-key bound proves
    // K < 2^17. The remainder checks below prove R < 2^SHIFT.
    lookups.push(LutLookup::rc16_filtered(
        Column::linear_combination_with_constant(
            [(m.noised_bound_fma.compression_quotient, one)],
            -F::from_canonical_u64(1 << 16),
        ),
        Filter::from_column(Column::single(m.noised_bound_fma.is_wide)),
    ));
    lookups.push(LutLookup::rc16(Column::single(m.noised_bound_fma.compression_remainder)));
    lookups.push(LutLookup::rc16(Column::linear_combination_with_constant(
        [
            (m.noised_bound_fma.shift_pow2, one),
            (m.noised_bound_fma.compression_remainder, neg),
        ],
        neg,
    )));

    // H1, W9: K's parity, two-sided: (K - K0)/2 in [0, 2^16).
    lookups.push(LutLookup::rc16(Column::linear_combination([
        (m.noised_bound_fma.compression_quotient, half),
        (m.noised_bound_fma.compression_quotient_lsb, -half),
    ])));

    // H1, W10: bound the RNERND significand below 2^17 to prevent cut-slot aliases.
    lookups.push(LutLookup::rc16(Column::linear_combination([
        (m.noised_bound_fma.rounding_significand_key, one),
        (m.noised_bound_fma.rounding_significand_key_high_bit, neg_limb_shift),
    ])));

    // H1, W11: the cut depth and the shared RNE back-end.
    lookups.push(LutLookup {
        table: LutTable::Clamp22,
        keys: vec![Column::linear_combination_with_constant(
            [(m.noised_bound_fma.key_scale, neg)],
            F::from_canonical_u64(267),
        )],
        values: vec![Column::single(m.noised_bound_fma.cut_used)],
        filter: Filter::default(),
    });
    lookups.push(LutLookup {
        table: LutTable::RneRnd,
        keys: vec![Column::linear_combination([
            (m.noised_bound_fma.rounding_significand_key, one),
            (m.noised_bound_fma.cut_used, F::from_canonical_u64(1 << 17)),
        ])],
        values: Column::singles([
            m.noised_bound_fma.out_mantissa,
            m.noised_bound_fma.width_adjust,
            m.noised_bound_fma.out_is_zero,
            m.noised_bound_fma.out_exp_is_zero,
        ])
        .collect(),
        filter: Filter::default(),
    });

    // H1, W12: exclude exponent 255 before constructing the BF16 code used by DIV448.
    lookups.push(LutLookup::rc16(Column::linear_combination_with_constant(
        [(m.noised_bound_fma.out_exp, neg)],
        F::from_canonical_u64(254),
    )));

    // H3: DIV448 binds alpha to the rounded 448/noised_bound quotient.
    // Exponent-255 sentinels fail the validity checks. The alpha-mantissa range check
    // makes the output code's field decomposition unique.
    lookups.push(LutLookup {
        table: LutTable::Div448,
        keys: vec![Column::linear_combination([
            (m.noised_bound_fma.out_exp, F::from_canonical_u64(128)),
            (m.noised_bound_fma.out_mantissa, one),
        ])],
        values: vec![Column::linear_combination([
            (m.alpha_exp, F::from_canonical_u64(128)),
            (m.alpha_mantissa, one),
        ])],
        filter: Filter::default(),
    });
    lookups.push(LutLookup::rc16(Column::linear_combination_with_constant(
        [(m.alpha_exp, one)],
        neg,
    )));
    lookups.push(LutLookup::rc16(Column::linear_combination_with_constant(
        [(m.alpha_exp, neg)],
        F::from_canonical_u64(254),
    )));
    lookups.push(LutLookup {
        table: LutTable::Pair128,
        keys: vec![Column::single(m.alpha_mantissa), Column::zero()],
        values: vec![],
        filter: Filter::default(),
    });

    // H4 (B1 = alpha * l2f, the FLOORED l2): CLAMP22 cut (key 535 - E*(alpha) - E*(l2f),
    // affine in the committed floored exponent) + RNERND + the exponent ban.
    lookups.push(LutLookup {
        table: LutTable::Clamp22,
        keys: vec![Column::linear_combination_with_constant(
            [(m.alpha_exp, neg), (m.l2_floored_exponent, neg)],
            F::from_canonical_u64(535),
        )],
        values: vec![Column::single(m.alpha_l2_multiply.cut_depth)],
        filter: Filter::default(),
    });
    lookups.push(LutLookup {
        table: LutTable::RneRnd,
        keys: vec![Column::linear_combination([
            (m.alpha_l2_multiply.sig_product, one),
            (m.alpha_l2_multiply.cut_depth, F::from_canonical_u64(1 << 17)),
        ])],
        values: Column::singles([
            m.alpha_l2_multiply.out_mantissa,
            m.alpha_l2_multiply.width_adjust,
            m.alpha_l2_multiply.out_is_zero,
            m.alpha_l2_multiply.out_exp_is_zero,
        ])
        .collect(),
        filter: Filter::default(),
    });
    lookups.push(LutLookup::rc16(Column::linear_combination_with_constant(
        [(m.alpha_l2_multiply.out_exp, neg)],
        F::from_canonical_u64(254),
    )));

    // H5: multiply beta_1 by dos. Its effective exponent is folded into the CLAMP22 key.
    // The production noise rank is fixed, so this offset is independent of job geometry.
    let e_star_dos = u64::from(program.dos_code() >> 7);
    lookups.push(LutLookup {
        table: LutTable::Clamp22,
        keys: vec![Column::linear_combination_with_constant(
            [(m.alpha_l2_multiply.out_exp, neg), (m.alpha_l2_multiply.out_exp_is_zero, neg)],
            F::from_canonical_u64(535 - e_star_dos),
        )],
        values: vec![Column::single(m.beta_scale_multiply.cut_depth)],
        filter: Filter::default(),
    });
    lookups.push(LutLookup {
        table: LutTable::RneRnd,
        keys: vec![Column::linear_combination([
            (m.beta_scale_multiply.sig_product, one),
            (m.beta_scale_multiply.cut_depth, F::from_canonical_u64(1 << 17)),
        ])],
        values: Column::singles([
            m.beta_scale_multiply.out_mantissa,
            m.beta_scale_multiply.width_adjust,
            m.beta_scale_multiply.out_is_zero,
            m.beta_scale_multiply.out_exp_is_zero,
        ])
        .collect(),
        filter: Filter::default(),
    });
    lookups.push(LutLookup::rc16(Column::linear_combination_with_constant(
        [(m.beta_scale_multiply.out_exp, neg)],
        F::from_canonical_u64(254),
    )));

    // T3: on the last row, prove DEAD_LIMIT - RUNNING_DEAD = low16 + 2^16*high_bit.
    // The high bit is boolean. Counts stay below 2^22 and limits below 2^16, so a
    // negative slack wraps far outside the accepted range for either high-bit value.
    lookups.push(LutLookup::rc16_filtered(
        Column::public_input_minus_linear_combination(
            DEAD_LIMIT_A_PUBLIC_INPUT,
            [
                (m.running_dead_a, F::ONE),
                (m.dead_slack_hi_a, F::from_canonical_u64(1 << 16)),
            ],
        ),
        Filter::from_column(Column::single(m.is_last_row)),
    ));
    lookups.push(LutLookup::rc16_filtered(
        Column::public_input_minus_linear_combination(
            DEAD_LIMIT_B_PUBLIC_INPUT,
            [
                (m.running_dead_b, F::ONE),
                (m.dead_slack_hi_b, F::from_canonical_u64(1 << 16)),
            ],
        ),
        Filter::from_column(Column::single(m.is_last_row)),
    ));

    // F3: the jackpot noise-floor gate (check 2), every row:
    // sigma = DELTA*alpha*l2f >= sigma_min <=> alpha*l2f >= 2 <=> E_SUM >= 255, or
    // E_SUM = 254 and SIG >= 2^15, with E_SUM = ALPHA_EXP + L2_FLOORED_EXPONENT and
    // SIG = ALPHA_L2_MULTIPLY.SIG_PRODUCT (see `columns.rs` for details). Wrap-safety: E_SUM <= 508 and SIG <= 255^2 < 2^16 on the pinned domains, so
    // each key is in range exactly on its branch's pass region and wraps far outside
    // [0, 2^16) below it.
    lookups.push(LutLookup::rc16_filtered(
        Column::linear_combination_with_constant(
            [(m.alpha_exp, one), (m.l2_floored_exponent, one)],
            -F::from_canonical_u64(255),
        ),
        Filter::from_column(Column::single(m.sigma_exp_clears_floor)),
    ));
    lookups.push(LutLookup::rc16_filtered(
        Column::linear_combination_with_constant([(m.alpha_l2_multiply.sig_product, one)], -F::from_canonical_u64(1 << 15)),
        Filter::from_column(Column::linear_combination_with_constant(
            [(m.sigma_exp_clears_floor, neg)],
            F::ONE,
        )),
    ));

    // S2: the two filtered range checks select the sigma product's binade at 2^15.
    // The product is already below 2^16, so a wrong branch produces an out-of-range key.
    lookups.push(LutLookup::rc16_filtered(
        Column::linear_combination_with_constant([(m.sigma_significand, one)], -F::from_canonical_u64(1 << 15)),
        Filter::from_column(Column::single(m.sigma_sig_is_wide)),
    ));
    lookups.push(LutLookup::rc16_filtered(
        Column::linear_combination_with_constant([(m.sigma_significand, neg)], F::from_canonical_u64((1 << 15) - 1)),
        Filter::from_column(Column::linear_combination_with_constant([(m.sigma_sig_is_wide, neg)], F::ONE)),
    ));

    lookups
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;

    use super::*;

    type F = GoldilocksField;

    fn test_program() -> ScaleProgram {
        ScaleProgram::new(4, 4, 2048, 4)
    }

    #[test]
    fn scale_ctl_looked_half_is_well_formed() {
        let looked = ctl_looked_scale_group_tuple::<F>();
        // TableWithColumns exposes no accessors; construction itself checks the column algebra.
        let _ = looked;
    }

    #[test]
    fn scale_lut_inventory_matches_documented_counts() {
        let lookups = scale_lut_lookups::<F>(&test_program());
        let count = |t: LutTable| lookups.iter().filter(|l| l.table == t).count();
        // Inventory documented by `scale_lut_lookups`:
        // RC16 x63, PAIR128 x3, EXPINFO x3, CLAMP22 x3, POW2D x3, RNERND x3, DIV448 x1.
        assert_eq!(count(LutTable::Range16), 63);
        assert_eq!(count(LutTable::Pair128), 3);
        assert_eq!(count(LutTable::ExpInfo), 3);
        assert_eq!(count(LutTable::Clamp22), 3);
        assert_eq!(count(LutTable::Pow2D), 3);
        assert_eq!(count(LutTable::RneRnd), 3);
        assert_eq!(count(LutTable::Div448), 1);
        assert_eq!(lookups.len(), 79);
    }
}
