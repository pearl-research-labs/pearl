//! Lookup descriptors for the ScaleStark computation.
//!
//! The group cross-table lookup consumes each live InputQuant aggregate exactly once.
//! Committed lookup tables then range-check the frame-sum and square-root-comparison limbs,
//! decode bf16 fields, floor both norms at `2^-32` (the scheme's `row_norms` floor), evaluate
//! the noised-bound FMA and two multiplies, divide the largest finite E4M3 magnitude (448) by
//! the noised bound, gate the per-side running dead totals against the `DEAD_LIMIT`
//! public inputs on the last row (jackpot check 1), and hold every row's noise scale
//! `sigma = DELTA * alpha * l2f` at or above the sigma floor (jackpot check 2).
//!
//! The inventory is RC16 x62, PAIR128 x3, EXPINFO x3, CLAMP22 x3, POW2D x3, RNERND x3, and
//! DIV448 x1. The additional RC16 and PAIR128 instances make the integer borrow comparisons,
//! FMA key bound, sqrt parity split, and alpha field split sound. The verifier derives the
//! normal bf16 constants `dr = RNE_bf16(delta*sqrt(noise_rank))` and
//! `dos = RNE_bf16(delta*sqrt(noise_rank)/256^2)`.

use plonky2::field::types::Field;
use starky::cross_table_lookup::TableWithColumns;
use starky::lookup::{Column, Filter};

use super::super::ctl::Table;
use super::super::luts::ctl::LutLookup;
use super::super::luts::LutTable;
use super::columns::{
    ALIGNED_LOWER_LIMBS, ALIGNED_UPPER_LIMBS, CLAIM_LIMBS, DEAD_LIMIT_A_PUBLIC_INPUT, DEAD_LIMIT_B_PUBLIC_INPUT, L2_SUM_LIMBS,
    SCALE_COL_MAP,
};
use super::stark::ScaleProgram;

// ==================================================================================================
// Channel: group tuples — InputQuant (looking) -> Scale (looked)
// ==================================================================================================

/// Scale's looked half of the group-tuple channel: the 13-component tuple
/// `(GROUP_KEY, L2_FRAME_SUM, FRAME_DOUBLED_SCALE_EXPONENT, MAX_ABS, ALPHA_EXP,
/// ALPHA_MANTISSA, BETA_EXP, BETA_MANTISSA, BETA_EXP_IS_ZERO, DEAD_BOUND, DEAD_COUNT,
/// SIGMA_ENC, SIGMA_NORM)`, filter `1 - IS_PAD` — every live Scale row consumes exactly one
/// InputQuant group; the power-of-two padding rows consume none (their tuple columns carry
/// the canonical zero-row fill, which T2 ignores).
/// The three beta components come directly from `beta_scale_multiply`'s output columns.
///
/// Three components are not Scale columns but affine expressions the channel itself binds:
///
/// - `MAX_ABS = 128*LINF_EXP + LINF_MANTISSA`, the decoded maximum's bf16 encoding;
/// - `DEAD_BOUND` is the device-specific BF16 code of `tau_idle * delta * l2f`;
/// - `SIGMA_ENC = ALPHA_EXP + L2_FLOORED_EXPONENT + SIGMA_SIG_IS_WIDE + offset(device) = e(sigma) + 268`
///   (S2, jackpot check 4) — the row's exact sigma encoding
///   (`sigma = SIGMA_SIGNIFICAND * 2^(ALPHA_EXP + L2_FLOORED_EXPONENT - 269)` with
///   `SIGMA_SIGNIFICAND in [2^14, 2^16)`, so `e(sigma)` adds `14 + SIGMA_SIG_IS_WIDE` to the
///   frame exponent; sigma is never zero in-scheme — alpha is structurally normal and the
///   floored l2 is at least `2^-32`).
///
/// InputQuant's looking half is `input_quant_stark::ctl::ctl_group_tuples_looking_input_quant`
/// (2 instances on live `IS_GROUP_FINAL` rows, A and B planes with the `+ h*k` key offset, the
/// same 13-component order).
pub fn ctl_looked_scale_group_tuple<F: Field>(program: &ScaleProgram) -> TableWithColumns<F> {
    let m = &SCALE_COL_MAP;
    let mut columns = vec![
        Column::single(m.group_key),
        Column::single(m.l2_frame_sum),
        Column::single(m.frame_doubled_scale_exponent),
        Column::linear_combination([(m.linf_exp, F::from_canonical_u64(128)), (m.linf_mantissa, F::ONE)]),
        Column::single(m.alpha_exp),
        Column::single(m.alpha_mantissa),
        Column::single(m.beta_scale_multiply.out_exp),
        Column::single(m.beta_scale_multiply.out_mantissa),
        Column::single(m.beta_scale_multiply.out_exp_is_zero),
    ];
    columns.push(Column::linear_combination_with_constant(
        [
            (m.l2_floored_exponent, F::from_canonical_u64(128)),
            (m.l2_floored_significand, F::ONE),
        ],
        F::from_canonical_u64(program.device.liveness_code_shift() - 128),
    ));
    columns.push(Column::single(m.dead_count));
    columns.push(Column::linear_combination_with_constant(
        [
            (m.alpha_exp, F::ONE),
            (m.l2_floored_exponent, F::ONE),
            (m.sigma_sig_is_wide, F::ONE),
        ],
        F::from_canonical_u64(program.device.sigma_encoding_offset()),
    ));
    columns.push(Column::single(m.sigma_norm));
    TableWithColumns::new(
        Table::Scale.into(),
        columns,
        Filter::from_column(Column::linear_combination_with_constant([(m.is_pad, -F::ONE)], F::ONE)),
    )
}

// Committed LUT oracle instances
// ==================================================================================================

/// ScaleStark's per-row LUT instance inventory: RC16 x62, PAIR128 x3, EXPINFO x3, CLAMP22 x3,
/// POW2D x3, RNERND x3, DIV448 x1 — 78 instances. Order: square root (Q), grid snap (G),
/// `linf` (N), the scale chain (H: norm floors, FMA, alpha, beta steps B1/B2), the liveness
/// gates (T3), the noise-floor gate (F3), then the sigma width bit (S2).
pub fn scale_lut_lookups<F: Field>(program: &ScaleProgram) -> Vec<LutLookup<F>> {
    let m = &SCALE_COL_MAP;
    let one = F::ONE;
    let neg = -F::ONE;
    let limb_base = F::from_canonical_u64(1 << 16);
    let neg_limb_base = -limb_base;
    let half = F::TWO.inverse();
    let mut lookups: Vec<LutLookup<F>> = Vec::new();

    // ---- Q1: frame-sum limbs + the canonicity cap (top limb < 2^14 => S < 2^62 < p). ----
    for i in 0..L2_SUM_LIMBS {
        lookups.push(LutLookup::rc16(Column::single(m.frame_sum_limbs[i])));
    }
    lookups.push(LutLookup::rc16(Column::linear_combination([(
        m.frame_sum_limbs[L2_SUM_LIMBS - 1],
        F::from_canonical_u64(4),
    )])));

    // ---- Q3: range-check the exponent, mantissa, and mantissa-half witness.
    // Checking HALF as well as MANTISSA makes Q4's equation
    // MANTISSA = 2*HALF + PARITY exact over integers, preventing field wraparound. ----
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

    // ---- Q7: midpoint-square limbs; MSQ = B_LO^2 < 2^20, so limb 1 < 2^4. ----
    lookups.push(LutLookup::rc16(Column::single(m.lower_boundary_squared_limbs[0])));
    lookups.push(LutLookup::rc16(Column::single(m.lower_boundary_squared_limbs[1])));
    lookups.push(LutLookup::rc16(Column::linear_combination([(
        m.lower_boundary_squared_limbs[1],
        F::from_canonical_u64(1 << 12),
    )])));

    // ---- Q7: claim-side products B^2 * k * 2^15 < 2^53 — limbs 1..=3 committed (limb 0 is
    // provably zero: 32 | k), top limb < 2^5, capped at < 2^4 via the 2^12 scale (the honest
    // bound is 2^52; the cap keeps the recomposition < 2^53 < p, alias-free). ----
    for limbs in [&m.lower_boundary_product_limbs, &m.upper_boundary_product_limbs] {
        for i in 0..CLAIM_LIMBS {
            lookups.push(LutLookup::rc16(Column::single(limbs[i])));
        }
        lookups.push(LutLookup::rc16(Column::linear_combination([(
            limbs[CLAIM_LIMBS - 1],
            F::from_canonical_u64(1 << 12),
        )])));
    }

    // ---- Q5: bound the extra left shift of l2_frame_sum to 0..15 bits.
    // With r2 = sum_shift_remainder, POW2D permits r2 in 0..19 and fixes sum_shift_power
    // to 2^r2. Range-checking 15-r2 excludes r2 > 15, whose negative difference wraps
    // outside the 16-bit range. ----
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

    // ---- Q6: shifted-sum limbs and their carries (each per-limb product equation is < 2^32 on
    // both sides given these ranges, hence exact over Z). ----
    for i in 0..L2_SUM_LIMBS {
        lookups.push(LutLookup::rc16(Column::single(m.shifted_sum_limbs[i])));
    }
    for i in 0..L2_SUM_LIMBS {
        lookups.push(LutLookup::rc16(Column::single(m.shifted_sum_carries[i])));
    }

    // ---- Q8: prove lower_bound + parity <= scaled_sum <= upper_bound - parity.
    // scaled_sum = l2_frame_sum * 2^(32 + sum_shift_remainder); parity = sqrt_mantissa_parity.
    // lower_bound and upper_bound are encoded by aligned_lower/upper_boundary_limbs,
    // whose array entry i occupies position i+1; their position 0 is zero.
    let scaled_sum_limb_column = |limb_position: usize| -> Option<usize> {
        match limb_position {
            0 | 1 | 7 => None, // two low zeros from the 2^32 shift; the top digit is also zero
            2..=5 => Some(m.shifted_sum_limbs[limb_position - 2]),
            6 => Some(m.shifted_sum_carries[3]),
            _ => unreachable!(),
        }
    };
    // Lower difference = scaled_sum - lower_bound - sqrt_mantissa_parity.
    // scaled_sum_digit[i] reads the column selected by scaled_sum_limb_column(i), or 0 if absent:
    //   digit[0] = -sqrt_mantissa_parity + 2^16*lower_comparison_borrows[0]
    //   digit[i] = scaled_sum_digit[i] - aligned_lower_boundary_limbs[i-1]
    //              - lower_comparison_borrows[i-1] + 2^16*lower_comparison_borrows[i]  (i=1..5)
    //   digit[6] = scaled_sum_digit[6] - aligned_lower_boundary_limbs[5] - lower_comparison_borrows[5]
    // Position 6 has no outgoing borrow: a negative final digit must be rejected.
    // Both operands' position-7 digits are zero; stark.rs enforces this for the lower boundary.
    lookups.push(LutLookup::rc16(Column::linear_combination([
        (m.sqrt_mantissa_parity, neg),
        (m.lower_comparison_borrows[0], limb_base),
    ])));
    for limb_position in 1..=ALIGNED_LOWER_LIMBS {
        let mut lower_difference_terms = vec![
            (m.aligned_lower_boundary_limbs[limb_position - 1], neg),
            (m.lower_comparison_borrows[limb_position - 1], neg),
        ];
        if let Some(&outgoing_borrow_column) = m.lower_comparison_borrows.get(limb_position) {
            lower_difference_terms.push((outgoing_borrow_column, limb_base));
        }
        if let Some(scaled_sum_column) = scaled_sum_limb_column(limb_position) {
            lower_difference_terms.push((scaled_sum_column, one));
        }
        lookups.push(LutLookup::rc16(Column::linear_combination(lower_difference_terms)));
    }
    // Upper difference = upper_bound - scaled_sum - sqrt_mantissa_parity.
    //   digit[0] = -sqrt_mantissa_parity + 2^16*upper_comparison_borrows[0]
    //   digit[i] = aligned_upper_boundary_limbs[i-1] - scaled_sum_digit[i]
    //              - upper_comparison_borrows[i-1] + 2^16*upper_comparison_borrows[i]  (i=1..6)
    //   digit[7] = aligned_upper_boundary_limbs[6] - upper_comparison_borrows[6]
    // Again there is no outgoing borrow at position 7.
    lookups.push(LutLookup::rc16(Column::linear_combination([
        (m.sqrt_mantissa_parity, neg),
        (m.upper_comparison_borrows[0], limb_base),
    ])));
    for limb_position in 1..=6 {
        let mut upper_difference_terms = vec![
            (m.aligned_upper_boundary_limbs[limb_position - 1], one),
            (m.upper_comparison_borrows[limb_position - 1], neg),
            (m.upper_comparison_borrows[limb_position], limb_base),
        ];
        if let Some(scaled_sum_column) = scaled_sum_limb_column(limb_position) {
            upper_difference_terms.push((scaled_sum_column, neg));
        }
        lookups.push(LutLookup::rc16(Column::linear_combination(upper_difference_terms)));
    }
    lookups.push(LutLookup::rc16(Column::linear_combination([
        (m.aligned_upper_boundary_limbs[ALIGNED_UPPER_LIMBS - 1], one),
        (m.upper_comparison_borrows[6], neg),
    ])));

    // ---- G1: the snap quotient, RC16'd so the snap equations are integer equations. ----
    lookups.push(LutLookup::rc16(Column::single(m.grid_snap_quotient)));

    // ---- G2: the snap remainder window, two-sided. ----
    lookups.push(LutLookup::rc16(Column::single(m.grid_snap_remainder)));
    lookups.push(LutLookup::rc16(Column::linear_combination_with_constant(
        [(m.grid_snap_remainder, neg)],
        F::from_canonical_u64(3),
    )));

    // ---- G3: snapped-l2 decode — EXPINFO (also rejects a snap into the inf field: exponent
    // 255 has no row) and the shared (L2_MANTISSA, LINF_MANTISSA) pair. ----
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

    // ---- N1: linf decode — EXPINFO (mantissa rides the G3 pair). ----
    lookups.push(LutLookup {
        table: LutTable::ExpInfo,
        keys: vec![Column::single(m.linf_exp)],
        values: vec![Column::single(m.linf_exp_is_zero)],
        filter: Filter::default(),
    });

    // ---- H0: the two norm-floor MAXes' order slacks (l2 and linf vs 2^-32; the muxes are
    // arithmetic constraints). ----
    lookups.push(LutLookup::rc16(Column::single(m.l2_floor_order_slack)));
    lookups.push(LutLookup::rc16(Column::single(m.linf_floor_order_slack)));

    // ---- H1 (FMA), W2/W3: the order and far-gap slacks (0 on the inactive side, so
    // unfiltered is complete). ----
    lookups.push(LutLookup::rc16(Column::single(m.noised_bound_fma.scale_gap_slack)));
    lookups.push(LutLookup::rc16(Column::single(m.noised_bound_fma.far_gap_slack)));

    // ---- H1, W4/W8: the two POW2D powers (key domains are the range proofs). ----
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

    // ---- H1, W8: wide split floor sandwich. K >= 2^16 here; the rounding-key bound proves
    // K < 2^17. The remainder checks below prove R < 2^SHIFT. ----
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

    // ---- H1, W9: K's parity, two-sided: (K - K0)/2 in [0, 2^16). ----
    lookups.push(LutLookup::rc16(Column::linear_combination([
        (m.noised_bound_fma.compression_quotient, half),
        (m.noised_bound_fma.compression_quotient_lsb, -half),
    ])));

    // ---- H1, W10: ROUNDING_SIGNIFICAND_KEY < 2^17 via the committed high bit — the RNERND key's
    // cut-slot aliasing fix (module docs). ----
    lookups.push(LutLookup::rc16(Column::linear_combination([
        (m.noised_bound_fma.rounding_significand_key, one),
        (m.noised_bound_fma.rounding_significand_key_high_bit, neg_limb_base),
    ])));

    // ---- H1, W11: the cut depth and the shared RNE back-end. ----
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

    // ---- H1, W12: the FMA exponent ban (out_exp <= 254 — makes the H2 code comparison and
    // the downstream DIV448 key honest bf16 codes; 0 on zero/subnormal rows, so unfiltered). ----
    lookups.push(LutLookup::rc16(Column::linear_combination_with_constant(
        [(m.noised_bound_fma.out_exp, neg)],
        F::from_canonical_u64(254),
    )));

    // ---- H3: alpha = RNE(448 / noised_bound) *is* the DIV448 row, keyed by the FMA output's
    // affine code (no denominator floor — H0 floors the norms instead, and the key domain
    // covers every constrained output; sentinel outputs carry exponent field 255, rejected by
    // the validity RCs below), plus the alpha-mantissa 7-bit check (deviation: without it the
    // value split is ambiguous). ----
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

    // ---- H4 (B1 = alpha * l2f, the FLOORED l2): CLAMP22 cut (key 535 - E*(alpha) - E*(l2f),
    // affine in the committed floored exponent) + RNERND + the exponent ban. ----
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

    // ---- H5 (B2 = B1 * dos): as H4, with the DOS effective exponent folded into the CLAMP22
    // key as a program constant (the verifier-side constant-offset mechanism; DOS is
    // structurally normal, so E*(dos) is its exponent field). `r` is the wire
    // constant 16, so this offset is a consensus constant, not cache-key material.
    // D1: derive it from the `DOS_EXP` public input once lookup keys can name PIs. ----
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

    // ---- T3: the jackpot liveness gates, one per side, on the last live row only (filter
    // IS_LAST_ROW, a known column): RC16(DEAD_LIMIT - RUNNING_DEAD - 2^16·HI) with the
    // boolean high-bit witness HI (constrained in `stark.rs`) proves
    // `RUNNING_DEAD <= DEAD_LIMIT`. The honest slack is at most
    // `DEAD_LIMIT = floor(side*k/64) < 2^16` (envelope `side*k < 2^22`), so the high
    // bit plus one RC16 limb cover it; an over-limit total wraps the slack to `~p`, far
    // outside RC16's window for either high-bit value. ----
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

    // ---- F: the jackpot noise-floor gate (check 2), every row:
    // SIGMA_ENC = E_SUM + SIGMA_SIG_IS_WIDE + offset(device) = e(sigma) + 268, so
    // sigma >= sigma_min = 1 iff SIGMA_ENC - 268 is nonnegative. Its maximum is 255 on
    // the pinned exponent domain; a negative key wraps far outside RC16's range. ----
    lookups.push(LutLookup::rc16(Column::linear_combination_with_constant(
        [(m.alpha_exp, one), (m.l2_floored_exponent, one), (m.sigma_sig_is_wide, one)],
        F::from_canonical_u64(program.device.sigma_encoding_offset()) - F::from_canonical_u64(268),
    )));

    // ---- S2: the sigma significand's width bit (jackpot check 4), two-sided:
    // SIGMA_SIGNIFICAND - 2^15 under the bit, 2^15 - 1 - SIGMA_SIGNIFICAND under its
    // complement. S1 pins the significand to H4's product (< 2^16), so each branch's key
    // wraps far outside [0, 2^16) exactly when the bit is wrong; pad rows (significand 0,
    // bit 0) pass the complement branch. ----
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
    use starky::cross_table_lookup::{debug_utils::check_ctls, CrossTableLookup};
    use starky::util::trace_rows_to_poly_values;

    use super::*;
    use crate::v4::api::public_params::Device;
    use crate::v4::circuit::consistency::build_fixture;
    use crate::v4::circuit::ctl::NUM_TABLES;
    use crate::v4::circuit::input_quant_stark::ctl::ctl_group_tuples_looking_input_quant;

    type F = GoldilocksField;

    fn test_program() -> ScaleProgram {
        ScaleProgram::new_for_device(4, 4, 2048, 4, Device::B200)
    }

    #[test]
    fn scale_ctl_looked_half_is_well_formed() {
        // 13 tuple components, filter 1 - IS_PAD (padding rows consume no tuple).
        let looked = ctl_looked_scale_group_tuple::<F>(&test_program());
        // TableWithColumns exposes no accessors; construction itself checks the column algebra.
        let _ = looked;
    }

    #[test]
    fn group_lookup_binds_linf_fields_to_input_quant_maximum() {
        let fixture = build_fixture(false);
        let ctl = CrossTableLookup::new(
            ctl_group_tuples_looking_input_quant(),
            vec![ctl_looked_scale_group_tuple(&fixture.scale)],
        );
        let mut traces = vec![Vec::new(); NUM_TABLES];
        traces[Table::InputQuant as usize] = trace_rows_to_poly_values(fixture.iq_rows);
        traces[Table::Scale as usize] = trace_rows_to_poly_values(fixture.scale_rows);
        let ctls = [ctl];
        check_ctls(&traces, &fixture.public_inputs, &ctls, &Default::default());

        // InputQuant's maximum stays fixed. Changing either decoded field must
        // break the group lookup, even though Scale has no separate max_abs column.
        for column in [SCALE_COL_MAP.linf_exp, SCALE_COL_MAP.linf_mantissa] {
            let mut tampered = traces.clone();
            let value = &mut tampered[Table::Scale as usize][column].values[0];
            *value = if *value == F::ZERO { F::ONE } else { *value - F::ONE };
            assert!(
                std::panic::catch_unwind(|| {
                    check_ctls(&tampered, &fixture.public_inputs, &ctls, &Default::default());
                })
                .is_err(),
                "a different linf encoding must not match InputQuant's maximum"
            );
        }
    }

    #[test]
    fn group_lookup_binds_beta_multiply_outputs_to_input_quant() {
        let fixture = build_fixture(false);
        let ctl = CrossTableLookup::new(
            ctl_group_tuples_looking_input_quant(),
            vec![ctl_looked_scale_group_tuple(&fixture.scale)],
        );
        let mut traces = vec![Vec::new(); NUM_TABLES];
        traces[Table::InputQuant as usize] = trace_rows_to_poly_values(fixture.iq_rows);
        traces[Table::Scale as usize] = trace_rows_to_poly_values(fixture.scale_rows);
        let ctls = [ctl];
        check_ctls(&traces, &fixture.public_inputs, &ctls, &Default::default());

        // Hold InputQuant's beta claim fixed. Each multiplication output must match it
        // through the group lookup, including the exponent-zero classification.
        let beta = &SCALE_COL_MAP.beta_scale_multiply;
        for column in [beta.out_exp, beta.out_mantissa, beta.out_exp_is_zero] {
            let mut tampered = traces.clone();
            let value = &mut tampered[Table::Scale as usize][column].values[0];
            *value = if *value == F::ZERO { F::ONE } else { *value - F::ONE };
            assert!(
                std::panic::catch_unwind(|| {
                    check_ctls(&tampered, &fixture.public_inputs, &ctls, &Default::default());
                })
                .is_err(),
                "changing beta output column {column} must break the group lookup"
            );
        }
    }

    #[test]
    fn scale_lut_inventory_matches_documented_counts() {
        let lookups = scale_lut_lookups::<F>(&test_program());
        let count = |t: LutTable| lookups.iter().filter(|l| l.table == t).count();
        // Inventory documented by `scale_lut_lookups`:
        // RC16 x62, PAIR128 x3, EXPINFO x3, CLAMP22 x3, POW2D x3, RNERND x3, DIV448 x1.
        assert_eq!(count(LutTable::Range16), 62);
        assert_eq!(count(LutTable::Pair128), 3);
        assert_eq!(count(LutTable::ExpInfo), 3);
        assert_eq!(count(LutTable::Clamp22), 3);
        assert_eq!(count(LutTable::Pow2D), 3);
        assert_eq!(count(LutTable::RneRnd), 3);
        assert_eq!(count(LutTable::Div448), 1);
        assert_eq!(lookups.len(), 78);
    }
}
