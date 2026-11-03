//! Cross-table-lookup declarations for the FP16 RowScaleStark.
//!
//! Two kinds of channel:
//!
//! * **Parameterized hooks** — this AIR proves the per-row norm + scale arithmetic from the raw
//!   FP16 operand codes (witness inputs). Binding those codes to the operand commitment and the
//!   `(alpha, beta)` outputs to `noisy_quant_stark` is left to later batch integration, so each
//!   builder takes the counterparty table's batch index explicitly, exactly like
//!   [`crate::v5::circuit::noisy_quant_stark::ctl`] and [`crate::v5::circuit::noise_stark::ctl`]
//!   (RowScaleStark is not registered in [`crate::v5::circuit::ctl`]).
//! * The committed-LUT inventory ([`row_scale_lut_lookups`]): the `FP16DECODE` operand decodes, the
//!   `WIDTH32` 24-bit normalizations, the `FP16POW2` alignment/rounding shifts, and the `RANGE16`
//!   limb/slack range checks that make every ties-to-even bracket sound. No new table is introduced —
//!   every fact targets the shared `FP16DECODE`/`RANGE16`/`FP16POW2`/`WIDTH32` tables of the FP16 batch.

use plonky2::field::types::Field;
use starky::cross_table_lookup::{TableIdx, TableWithColumns};
use starky::lookup::{Column, Filter};

use super::columns::ROW_SCALE_COL_MAP;
use crate::v4::circuit::luts::LutTable;
use crate::v4::circuit::luts::ctl::LutLookup;

/// The live-row filter `1 - IS_PAD`.
fn live_filter<F: Field>() -> Filter<F> {
    Filter::from_column(Column::linear_combination_with_constant([(ROW_SCALE_COL_MAP.is_pad, -F::ONE)], F::ONE))
}

/// `(1 - IS_PAD) * (1 - flag_col)` as a product filter.
fn live_and_not<F: Field>(flag_col: usize) -> Filter<F> {
    Filter::new(
        vec![(
            Column::linear_combination_with_constant([(ROW_SCALE_COL_MAP.is_pad, -F::ONE)], F::ONE),
            Column::linear_combination_with_constant([(flag_col, -F::ONE)], F::ONE),
        )],
        vec![],
    )
}

/// Hook: the clean FP16 operand code `CODE`, filter `1 - IS_PAD`. Counterparty: the operand
/// commitment (index supplied at integration time).
pub fn ctl_operand_looking<F: Field>(table: usize) -> TableWithColumns<F> {
    TableWithColumns::new(TableIdx::from(table), vec![Column::single(ROW_SCALE_COL_MAP.code)], live_filter())
}

/// Hook: the per-row `(L2_CODE, LINF_CODE, ALPHA_CODE, BETA_CODE)` outputs (scalar columns,
/// identical on every row). The counterparty (`noisy_quant_stark` for alpha/beta) is wired at batch
/// integration. Filter `1 - IS_PAD`.
pub fn ctl_scales_looking<F: Field>(table: usize) -> TableWithColumns<F> {
    let m = &ROW_SCALE_COL_MAP;
    TableWithColumns::new(
        TableIdx::from(table),
        Column::singles([m.l2_code, m.linf_code, m.alpha_code, m.beta_code]).collect(),
        live_filter(),
    )
}

/// RowScaleStark's per-row committed-LUT inventory (FP16DECODE + WIDTH32 + FP16POW2 + RANGE16).
pub fn row_scale_lut_lookups<F: Field>() -> Vec<LutLookup<F>> {
    let m = &ROW_SCALE_COL_MAP;
    let live = live_filter::<F>();
    let nz_sq = live_and_not::<F>(m.x_is_zero);
    let nz_w = live_and_not::<F>(m.w_is_zero);
    let nz_q = live_and_not::<F>(m.q_zero);
    let nz_sumsq = live_and_not::<F>(m.sumsq_zero);
    let nz_max = live_and_not::<F>(m.max_is_zero);
    let one_c = Column::constant(F::ONE);
    let mut lu: Vec<LutLookup<F>> = Vec::new();

    let rc = LutLookup::rc16;
    let scaled = |col: usize, sh: u64| Column::linear_combination([(col, F::from_canonical_u64(1 << sh))]);
    // `(hi - 128) * 2^8` — pins a 24-bit significand's high limb into `[128, 256)`.
    let hi_norm = |col: usize| {
        Column::linear_combination_with_constant([(col, F::from_canonical_u64(1 << 8))], -F::from_canonical_u64(128 << 8))
    };

    // ---- FP16DECODE. ----
    lu.push(LutLookup {
        table: LutTable::Fp16Decode,
        keys: vec![Column::single(m.code)],
        values: Column::singles([m.x_sig, m.x_sign, m.x_eps_biased, m.x_is_zero]).collect(),
        filter: live.clone(),
    });
    lu.push(LutLookup {
        table: LutTable::Fp16Decode,
        keys: vec![Column::single(m.max_abs_code)],
        values: Column::singles([m.max_sig, m.max_sign, m.max_eps_biased, m.max_is_zero]).collect(),
        filter: live.clone(),
    });

    // ---- WIDTH32: 24-bit normalizations (trunc is always 1 since each width <= 23). ----
    lu.push(LutLookup {
        table: LutTable::Width32,
        keys: vec![Column::single(m.sq_w)],
        values: vec![one_c.clone(), Column::single(m.sq_lift)],
        filter: nz_sq.clone(),
    });
    lu.push(LutLookup {
        table: LutTable::Width32,
        keys: vec![Column::single(m.ww)],
        values: Column::singles([m.trunc_w, m.lift_w]).collect(),
        filter: nz_w.clone(),
    });
    lu.push(LutLookup {
        table: LutTable::Width32,
        keys: vec![Column::single(m.max_w)],
        values: vec![one_c.clone(), Column::single(m.max_lift)],
        filter: nz_max.clone(),
    });

    // ---- FP16POW2: alignment + rounding shifts. ----
    let pow = |key: Column<F>, val: usize, filter: Filter<F>| LutLookup {
        table: LutTable::Fp16Pow2,
        keys: vec![key],
        values: vec![Column::single(val)],
        filter,
    };
    lu.push(pow(Column::single(m.rel_sq), m.pow_sq, Filter::from_column(Column::single(m.active_sq))));
    lu.push(pow(Column::single(m.rel_acc), m.pow_acc, Filter::from_column(Column::single(m.active_acc))));
    lu.push(pow(Column::single(m.q_dexp), m.q_pow, nz_q.clone()));
    lu.push(pow(Column::single(m.s_dexp), m.s_pow, nz_q.clone()));
    lu.push(pow(Column::single(m.nb_shift), m.nb_shiftpow, live.clone()));
    lu.push(pow(Column::single(m.nb_gm), m.nb_pow, live.clone()));
    // alpha: key A = 286 - nb_exp - alpha_exp.
    lu.push(pow(
        Column::linear_combination_with_constant(
            [(m.nb_exp, -F::ONE), (m.alpha_exp, -F::ONE)],
            F::from_canonical_u64(super::stark::DIV_KEY_BASE),
        ),
        m.alpha_pow,
        live.clone(),
    ));
    lu.push(pow(Column::single(m.m1_gm), m.m1_pow, live.clone()));
    lu.push(pow(Column::single(m.beta_gm), m.beta_pow, live.clone()));

    // ---- RANGE16: per-element. ----
    lu.push(rc(Column::single(m.max_slack)));
    lu.push(rc(Column::single(m.sq_norm_lo)));
    lu.push(LutLookup::rc16_filtered(hi_norm(m.sq_norm_hi), nz_sq.clone()));
    lu.push(rc(Column::single(m.rel_sq)));
    lu.push(rc(Column::single(m.rel_acc)));
    lu.push(rc(Column::single(m.far_sq_slack)));
    lu.push(rc(Column::single(m.far_acc_slack)));
    lu.push(rc(Column::single(m.aligned_sq_lo)));
    lu.push(rc(scaled(m.aligned_sq_hi, 5)));
    lu.push(rc(Column::single(m.aligned_acc_lo)));
    lu.push(rc(scaled(m.aligned_acc_hi, 5)));
    for (lo, hi) in [
        (m.rem_sq_lo, m.rem_sq_hi),
        (m.rem_sq_bound_lo, m.rem_sq_bound_hi),
        (m.rem_acc_lo, m.rem_acc_hi),
        (m.rem_acc_bound_lo, m.rem_acc_bound_hi),
    ] {
        lu.push(rc(Column::single(lo)));
        lu.push(rc(scaled(hi, 6)));
    }
    lu.push(rc(Column::single(m.w_abs_lo)));
    lu.push(rc(scaled(m.w_abs_hi, 4)));
    lu.push(rc(Column::single(m.m_rz_lo)));
    lu.push(LutLookup::rc16_filtered(hi_norm(m.m_rz_hi), nz_w.clone()));
    lu.push(rc(Column::single(m.rz_rem)));
    lu.push(rc(Column::single(m.rz_rem_bound)));
    lu.push(rc(Column::single(m.rz_half)));
    lu.push(rc(Column::single(m.gt_slack)));

    // ---- RANGE16: scalars (sumsq / q / s). ----
    lu.push(rc(Column::single(m.sumsq_sig_lo)));
    lu.push(LutLookup::rc16_filtered(hi_norm(m.sumsq_sig_hi), nz_sumsq.clone()));
    lu.push(rc(Column::single(m.q_sig_lo)));
    lu.push(LutLookup::rc16_filtered(hi_norm(m.q_sig_hi), nz_q.clone()));
    lu.push(rc(Column::single(m.q_half)));
    for c in [m.q_sl_lo, m.q_sl_mid, m.q_sl_hi, m.q_su_lo, m.q_su_mid, m.q_su_hi] {
        lu.push(rc(Column::single(c)));
    }
    lu.push(rc(Column::single(m.s_sig_lo)));
    lu.push(LutLookup::rc16_filtered(hi_norm(m.s_sig_hi), nz_q.clone()));
    lu.push(rc(Column::single(m.s_half)));
    for c in [m.s_sl_lo, m.s_sl_mid, m.s_sl_hi, m.s_su_lo, m.s_su_mid, m.s_su_hi] {
        lu.push(rc(Column::single(c)));
    }

    // ---- RANGE16: l2raw / grid / l2 floor. ----
    lu.push(rc(scaled(m.l2raw_mant, 9)));
    lu.push(rc(Column::single(m.l2raw_half)));
    for c in [m.l2raw_sl_lo, m.l2raw_sl_hi, m.l2raw_su_lo, m.l2raw_su_hi] {
        lu.push(rc(Column::single(c)));
    }
    lu.push(rc(Column::single(m.grid_q)));
    lu.push(rc(Column::single(m.l2_floor_slack)));
    lu.push(rc(scaled(m.l2_e, 8)));
    lu.push(rc(scaled(m.l2_m, 9)));

    // ---- RANGE16: linf. ----
    lu.push(rc(Column::single(m.max_norm_lo)));
    lu.push(LutLookup::rc16_filtered(hi_norm(m.max_norm_hi), nz_max.clone()));
    lu.push(rc(scaled(m.linf_raw_mant, 9)));
    lu.push(rc(Column::single(m.linf_raw_half)));
    for c in [m.linf_raw_sl_lo, m.linf_raw_sl_hi, m.linf_raw_su_lo, m.linf_raw_su_hi] {
        lu.push(rc(Column::single(c)));
    }
    lu.push(rc(Column::single(m.linf_floor_slack)));
    lu.push(rc(scaled(m.linf_e, 8)));
    lu.push(rc(scaled(m.linf_m, 9)));

    // ---- RANGE16: noised_bound / alpha / m1 / beta. ----
    lu.push(rc(Column::single(m.nb_w_lo)));
    lu.push(rc(Column::single(m.nb_w_hi)));
    lu.push(rc(scaled(m.nb_mant, 9)));
    lu.push(rc(Column::single(m.nb_half)));
    for c in [m.nb_sl_lo, m.nb_sl_hi, m.nb_su_lo, m.nb_su_hi] {
        lu.push(rc(Column::single(c)));
    }
    lu.push(rc(scaled(m.alpha_mant, 9)));
    lu.push(rc(scaled(m.alpha_exp, 8)));
    lu.push(rc(Column::single(m.alpha_half)));
    lu.push(rc(Column::single(m.alpha_sl)));
    lu.push(rc(Column::single(m.alpha_su)));
    lu.push(rc(scaled(m.m1_mant, 9)));
    lu.push(rc(Column::single(m.m1_half)));
    lu.push(rc(Column::single(m.m1_sl)));
    lu.push(rc(Column::single(m.m1_su)));
    lu.push(rc(scaled(m.beta_mant, 9)));
    lu.push(rc(Column::single(m.beta_half)));
    lu.push(rc(Column::single(m.beta_sl)));
    lu.push(rc(Column::single(m.beta_su)));

    // ---- Entry-liveness gate (group L). ----
    // FP16POW2: the aligned-compare shift `dead_key -> dead_pow = 2^min(dead_key,26)` (live rows;
    // membership also bounds `dead_key` to the nonneg key domain, pinning `dead_sign`).
    lu.push(pow(Column::single(m.dead_key), m.dead_pow, live.clone()));
    // RANGE16: the two-sided compare slack (3 limbs, `< 2^37`) and the side-mask pin slack, live.
    lu.push(LutLookup::rc16_filtered(Column::single(m.dead_slack_lo), live.clone()));
    lu.push(LutLookup::rc16_filtered(Column::single(m.dead_slack_mid), live.clone()));
    lu.push(LutLookup::rc16_filtered(scaled(m.dead_slack_hi, 11), live.clone()));
    lu.push(LutLookup::rc16_filtered(Column::single(m.b_side_slack), live.clone()));
    // RANGE16: the per-side gate slacks (meaningful on the last row, zero elsewhere; `< 2^23`).
    lu.push(rc(Column::single(m.a_gate_slack_lo)));
    lu.push(rc(scaled(m.a_gate_slack_hi, 9)));
    lu.push(rc(Column::single(m.b_gate_slack_lo)));
    lu.push(rc(scaled(m.b_gate_slack_hi, 9)));

    lu
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::PrimeField64;
    use starky::util::trace_rows_to_poly_values;

    use super::super::stark::RowScaleProgram;
    use super::*;

    type F = GoldilocksField;

    #[test]
    fn ctl_halves_are_well_formed() {
        ctl_operand_looking::<F>(3);
        ctl_scales_looking::<F>(3);
        let _ = row_scale_lut_lookups::<F>();
    }

    #[test]
    fn lut_inventory_counts() {
        let lu = row_scale_lut_lookups::<F>();
        let count = |t: LutTable| lu.iter().filter(|l| l.table == t).count();
        assert_eq!(count(LutTable::Fp16Decode), 2);
        assert_eq!(count(LutTable::Width32), 3);
        assert_eq!(count(LutTable::Fp16Pow2), 10); // +1: the liveness aligned-compare shift
        assert_eq!(count(LutTable::Range16), lu.len() - 15);
    }

    /// Every committed-LUT key the honest trace presents lands in its table's domain (RANGE16 keys
    /// `< 2^16`, FP16DECODE keys `< 2^16`, FP16POW2 keys `<= 255` with `value = 2^min(key, 26)`,
    /// WIDTH32 keys in `[1, 32]`). Checked on the live rows (filtered lookups are inactive elsewhere).
    #[test]
    fn honest_lut_keys_are_in_domain() {
        use crate::v5::api::dtype::f32_to_fp16;
        for (k, seed, scale) in [(64usize, 1u64, 4.0f32), (48, 7, 0.5), (96, 11, 20.0), (32, 3, 1.0)] {
            let program = RowScaleProgram::new(k, 32);
            let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
            let row: Vec<u16> = (0..k)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    f32_to_fp16(((s >> 40) as i32 % 2000 - 1000) as f32 / 1000.0 * scale).unwrap()
                })
                .collect();
            let rows = program.generate_trace::<F>(&row);
            let polys = trace_rows_to_poly_values(rows);
            let lu = row_scale_lut_lookups::<F>();
            for (li, lookup) in lu.iter().enumerate() {
                for r in 0..k {
                    let active = lookup.filter.eval_table(&polys, r, &[]).to_canonical_u64();
                    if active == 0 {
                        continue;
                    }
                    match lookup.table {
                        LutTable::Range16 => {
                            let key = lookup.keys[0].eval_table(&polys, r, &[]).to_canonical_u64();
                            assert!(key < 1 << 16, "RANGE16 lookup {li} key {key} out of range at row {r}");
                        }
                        LutTable::Fp16Decode => {
                            let key = lookup.keys[0].eval_table(&polys, r, &[]).to_canonical_u64();
                            assert!(key < 1 << 16, "FP16DECODE lookup {li} key {key} at row {r}");
                        }
                        LutTable::Fp16Pow2 => {
                            let key = lookup.keys[0].eval_table(&polys, r, &[]).to_canonical_u64();
                            let val = lookup.values[0].eval_table(&polys, r, &[]).to_canonical_u64();
                            assert!(key <= 255, "FP16POW2 lookup {li} key {key} at row {r}");
                            assert_eq!(val, 1 << key.min(26), "FP16POW2 lookup {li} value {val} != 2^min({key},26) at row {r}");
                        }
                        LutTable::Width32 => {
                            let key = lookup.keys[0].eval_table(&polys, r, &[]).to_canonical_u64();
                            assert!((1..=32).contains(&key), "WIDTH32 lookup {li} key {key} at row {r}");
                        }
                        other => panic!("unexpected table {other:?}"),
                    }
                }
            }
        }
    }
}
