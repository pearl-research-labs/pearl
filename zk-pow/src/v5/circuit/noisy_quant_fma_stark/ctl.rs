//! Cross-table-lookup declarations for the FP16 single-rounding FMA AIR (group G2).
//!
//! * **Pairing hook** ([`ctl_fma_pairing_looking`]): exposes the tuple
//!   `(alpha, raw, t_sign, t_mant, t_exp, noised_lo, noised_hi)`, filter `1 - IS_PAD`. This is the
//!   EXACT tuple [`crate::v5::circuit::noisy_quant_stark::ctl::ctl_fma_hook_looking`] exposes, so
//!   pairing the two in the batch binds this AIR's inputs `(alpha, raw, t)` and its proven output
//!   `noised` to group G1's `t` and group G3's `noised` input in one channel. The table index is
//!   supplied at integration time (this AIR is not registered in [`crate::v5::circuit::ctl`]).
//! * The committed-LUT inventory ([`fma_lut_lookups`]): FP16DECODE (raw), WIDTH32 (the two bit-width
//!   normalizations), FP16POW2 (the two alignment shifts), and RANGE16 (every limb/slack). No new
//!   table is introduced — all facts target the FP16 batch's `Fp16Decode`/`Width32`/`Fp16Pow2`/
//!   `Range16` tables.

use plonky2::field::types::Field;
use starky::cross_table_lookup::{TableIdx, TableWithColumns};
use starky::lookup::{Column, Filter};

use super::columns::FMA_COL_MAP;
use crate::v4::circuit::luts::LutTable;
use crate::v4::circuit::luts::ctl::LutLookup;

fn live_filter<F: Field>() -> Filter<F> {
    Filter::from_column(Column::linear_combination_with_constant([(FMA_COL_MAP.is_pad, -F::ONE)], F::ONE))
}

/// `(1 - IS_PAD) * (1 - flag_col)` as a product filter.
fn live_and_not<F: Field>(flag_col: usize) -> Filter<F> {
    Filter::new(
        vec![(
            Column::linear_combination_with_constant([(FMA_COL_MAP.is_pad, -F::ONE)], F::ONE),
            Column::linear_combination_with_constant([(flag_col, -F::ONE)], F::ONE),
        )],
        vec![],
    )
}

/// The pairing hook: `(alpha, raw, t_sign, t_mant, t_exp, noised_lo, noised_hi)`, filter `1-IS_PAD`.
pub fn ctl_fma_pairing_looking<F: Field>(table: usize) -> TableWithColumns<F> {
    let m = &FMA_COL_MAP;
    TableWithColumns::new(
        TableIdx::from(table),
        Column::singles([m.alpha, m.raw, m.t_sign, m.t_mant, m.t_exp, m.noised_lo, m.noised_hi]).collect(),
        live_filter(),
    )
}

/// Committed-LUT inventory (FP16DECODE + WIDTH32 + FP16POW2 + RANGE16).
pub fn fma_lut_lookups<F: Field>() -> Vec<LutLookup<F>> {
    let m = &FMA_COL_MAP;
    let nz_p = live_and_not::<F>(m.p_is_zero);
    let nz_w = live_and_not::<F>(m.w_is_zero);
    let mut lu: Vec<LutLookup<F>> = Vec::new();

    // ---- FP16DECODE: raw -> (sig, sign, eps_biased, is_zero). Filtered to live rows: the all-zero
    // padding rows carry a default value tuple that does not match FP16DECODE's slot for code 0, so
    // an unfiltered lookup would be an unservable (value-mismatch) instance in the batch. Padding
    // rows are unconstrained; every live row is still decoded. ----
    lu.push(LutLookup {
        table: LutTable::Fp16Decode,
        keys: vec![Column::single(m.raw)],
        values: Column::singles([m.x_sig, m.x_sign, m.x_eps_biased, m.x_is_zero]).collect(),
        filter: live_filter(),
    });

    // ---- WIDTH32: b_p -> (trunc_p, lift_p) (nonzero-P rows), w -> (trunc_w, lift_w) (nonzero-W). ----
    lu.push(LutLookup {
        table: LutTable::Width32,
        keys: vec![Column::single(m.bp)],
        values: Column::singles([m.trunc_p, m.lift_p]).collect(),
        filter: nz_p.clone(),
    });
    lu.push(LutLookup {
        table: LutTable::Width32,
        keys: vec![Column::single(m.ww)],
        values: Column::singles([m.trunc_w, m.lift_w]).collect(),
        filter: nz_w.clone(),
    });

    // ---- FP16POW2: alignment shifts, keyed on rel, filtered to active rows. ----
    lu.push(LutLookup {
        table: LutTable::Fp16Pow2,
        keys: vec![Column::single(m.rel_p)],
        values: vec![Column::single(m.pow_p)],
        filter: Filter::from_column(Column::single(m.active_p)),
    });
    lu.push(LutLookup {
        table: LutTable::Fp16Pow2,
        keys: vec![Column::single(m.rel_t)],
        values: vec![Column::single(m.pow_t)],
        filter: Filter::from_column(Column::single(m.active_t)),
    });

    // ---- RANGE16. ----
    let rc = |c: Column<F>| LutLookup::rc16(c);
    let scaled = |col: usize, sh: u64| Column::linear_combination([(col, F::from_canonical_u64(1 << sh))]);
    // 7-bit fields.
    lu.push(rc(scaled(m.alpha_mant, 9)));
    // Mp_norm limbs (hi in [128,256), filtered nonzero-P).
    lu.push(rc(Column::single(m.mp_norm_lo)));
    lu.push(LutLookup::rc16_filtered(
        Column::linear_combination_with_constant([(m.mp_norm_hi, F::from_canonical_u64(1 << 8))], -F::from_canonical_u64(128 << 8)),
        nz_p,
    ));
    // rel, far slacks.
    lu.push(rc(Column::single(m.rel_p)));
    lu.push(rc(Column::single(m.rel_t)));
    lu.push(rc(Column::single(m.far_p_slack)));
    lu.push(rc(Column::single(m.far_t_slack)));
    // aligned limbs (< 2^27: lo < 2^16, hi*2^5 < 2^16 -> hi < 2^11).
    lu.push(rc(Column::single(m.aligned_p_lo)));
    lu.push(rc(scaled(m.aligned_p_hi, 5)));
    lu.push(rc(Column::single(m.aligned_t_lo)));
    lu.push(rc(scaled(m.aligned_t_hi, 5)));
    // rem / rem_bound limbs (< 2^26: lo < 2^16, hi*2^6 < 2^16 -> hi < 2^10).
    for (lo, hi) in [
        (m.rem_p_lo, m.rem_p_hi),
        (m.rem_p_bound_lo, m.rem_p_bound_hi),
        (m.rem_t_lo, m.rem_t_hi),
        (m.rem_t_bound_lo, m.rem_t_bound_hi),
    ] {
        lu.push(rc(Column::single(lo)));
        lu.push(rc(scaled(hi, 6)));
    }
    // |W| limbs (< 2^28: lo < 2^16, hi*2^4 < 2^16 -> hi < 2^12).
    lu.push(rc(Column::single(m.w_abs_lo)));
    lu.push(rc(scaled(m.w_abs_hi, 4)));
    // M_rz limbs (hi in [128,256), filtered nonzero-W).
    lu.push(rc(Column::single(m.m_rz_lo)));
    lu.push(LutLookup::rc16_filtered(
        Column::linear_combination_with_constant([(m.m_rz_hi, F::from_canonical_u64(1 << 8))], -F::from_canonical_u64(128 << 8)),
        nz_w,
    ));
    // RNE small slacks.
    lu.push(rc(Column::single(m.rz_rem)));
    lu.push(rc(Column::single(m.rz_rem_bound)));
    lu.push(rc(Column::single(m.rz_half)));
    lu.push(rc(Column::single(m.gt_slack)));

    lu
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;

    use super::*;

    type F = GoldilocksField;

    #[test]
    fn ctl_halves_are_well_formed() {
        ctl_fma_pairing_looking::<F>(5);
    }

    #[test]
    fn lut_inventory_counts() {
        let lu = fma_lut_lookups::<F>();
        let count = |t: LutTable| lu.iter().filter(|l| l.table == t).count();
        assert_eq!(count(LutTable::Fp16Decode), 1);
        assert_eq!(count(LutTable::Width32), 2);
        assert_eq!(count(LutTable::Fp16Pow2), 2);
        assert_eq!(count(LutTable::Range16), lu.len() - 5);
    }
}
