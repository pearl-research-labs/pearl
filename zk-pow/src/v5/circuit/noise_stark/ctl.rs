//! Cross-table-lookup declarations for the FP16 NoiseStark.
//!
//! Two kinds of channel:
//!
//! * The **raw-byte** channel ([`ctl_noise_bytes_looking`]) exposes each live row's `BYTE` so a
//!   later stage can CTL-bind it to the keyed-BLAKE3 XOF table (that binding is NOT implemented
//!   here — this AIR takes the bytes as witness inputs and proves everything downstream). The
//!   builder takes the table's batch index explicitly, exactly like
//!   [`crate::v5::circuit::xor_fold_stark::ctl`], because NoiseStark is not registered in
//!   [`crate::v5::circuit::ctl`].
//! * The committed-LUT inventory ([`noise_lut_lookups`]): the RC16 limb/slack range checks, the
//!   PAIR128 7-bit mantissa checks, and the POW2D shift lookups that make the integer isqrt and
//!   the three ties-to-even BF16/FP16 rounding brackets sound. No new table is introduced — all
//!   facts target the shared `RANGE16`, `PAIR128` and `POW2D` tables.

use plonky2::field::types::Field;
use starky::cross_table_lookup::{TableIdx, TableWithColumns};
use starky::lookup::{Column, Filter};

use super::columns::NOISE_COL_MAP;
use crate::v4::circuit::luts::LutTable;
use crate::v4::circuit::luts::ctl::LutLookup;

/// The live-row filter `1 - IS_PAD`.
fn live_filter<F: Field>() -> Filter<F> {
    Filter::from_column(Column::linear_combination_with_constant(
        [(NOISE_COL_MAP.is_pad, -F::ONE)],
        F::ONE,
    ))
}

/// NoiseStark's looking half of the **raw-byte** channel: `(BYTE)` on every live row, filter
/// `1 - IS_PAD`. The counterparty (the keyed-BLAKE3 XOF commitment) is a later integration step;
/// the `noise_bytes_table` index is supplied then.
pub fn ctl_noise_bytes_looking<F: Field>(noise_bytes_table: usize) -> TableWithColumns<F> {
    TableWithColumns::new(
        TableIdx::from(noise_bytes_table),
        vec![Column::single(NOISE_COL_MAP.byte)],
        live_filter(),
    )
}

/// NoiseStark's **looked** half of the E/F-operand binding channel (increment 6e-2): per live row,
/// the tuple `(GLOBAL_INDEX, ENTRY_FP16)` — the global entry index `line * rank + entry` and the
/// proven normalized FP16 entry. The filter `OPERAND_MULT * (1 - IS_PAD)` supplies each entry with
/// the noise matmul's per-cell reuse multiplicity (an `E` entry is reused in `k` cells, an `F` entry
/// in `h`/`w`), so the looking side (the noise matmuls' operand codes, keyed `base + lane (+ offset)`
/// = `GLOBAL_INDEX`) multiset-balances. This makes the noise matmuls' `E`/`F` operands provably the
/// normalized noise lines this AIR derives — no longer free witness.
pub fn ctl_noise_operands_looked<F: Field>(table: usize) -> TableWithColumns<F> {
    let m = &NOISE_COL_MAP;
    TableWithColumns::new(
        TableIdx::from(table),
        vec![Column::single(m.global_index), Column::single(m.entry_fp16)],
        Filter::new(
            vec![(
                Column::single(m.operand_mult),
                Column::linear_combination_with_constant([(m.is_pad, -F::ONE)], F::ONE),
            )],
            vec![],
        ),
    )
}

/// NoiseStark's **looked** half of the egress-pair channel (increment 6e-3c): per `IS_EGRESS_PAIR`
/// row, the tuple `(EGRESS_KEY, BYTE_PAIR)` — the little-endian 16-bit XOF limb keyed by
/// `line*16 + entry/2`. The looking side is the noise-BLAKE3 engine's output egress
/// ([`crate::v5::circuit::blake3_fp16_stark::ctl::ctl_cv_egress_looking_blake3`]), whose line
/// compression `L` exports `cv_out` as 16 limbs on keys `L*16 .. L*16 + 16`. The multiset balance
/// forces NoiseStark's normalized line's raw XOF bytes to equal the seed-derived keyed-BLAKE3 XOF, so
/// the noise is no longer grindable through those bytes.
pub fn ctl_noise_egress_pair_looked<F: Field>(table: usize) -> TableWithColumns<F> {
    let m = &NOISE_COL_MAP;
    TableWithColumns::new(
        TableIdx::from(table),
        vec![Column::single(m.egress_key), Column::single(m.byte_pair)],
        Filter::from_column(Column::single(m.is_egress_pair)),
    )
}

/// `x - 2^16 * hi` — the low 16-bit limb of a two-limb value (RC16 key).
fn low_limb<F: Field>(x: usize, hi: usize) -> Column<F> {
    Column::linear_combination([(x, F::ONE), (hi, -F::from_canonical_u64(1 << 16))])
}

/// NoiseStark's per-row committed-LUT inventory.
///
/// * **RANGE16**: the sum/isqrt limbs and their canonicity caps, the three brackets' lower/upper
///   slacks (each proved `>= 0` by a two-limb RC16 split), the parity halves, and the FP16
///   exponent window.
/// * **PAIR128**: the four 7-bit mantissa fields, paired two-per-lookup.
/// * **POW2D**: the `denom` shift `2^(DENOM_EXP - 134)`, the division shift
///   `2^(283 - DENOM_EXP - SCALE_EXP)`, and the per-entry multiply shift `2^gm`; each key domain
///   `[0, 19]` also proves the shift is in range.
pub fn noise_lut_lookups<F: Field>() -> Vec<LutLookup<F>> {
    let m = &NOISE_COL_MAP;
    let one = F::ONE;
    let neg = -F::ONE;
    let two16 = F::from_canonical_u64(1 << 16);
    let neg16 = -two16;
    let live = live_filter::<F>();
    let mut lu: Vec<LutLookup<F>> = Vec::new();

    // ---- S: sum-of-squares and isqrt limbs. ----
    // TOTAL_SUMSQ < 2^32.
    lu.push(LutLookup::rc16(low_limb(m.total_sumsq, m.total_hi)));
    lu.push(LutLookup::rc16(Column::single(m.total_hi)));
    // NORM_SCALED (q) < 2^21: low limb, high limb, and the 2^11-scaled cap (high limb < 2^5).
    lu.push(LutLookup::rc16(low_limb(m.norm_scaled, m.norm_scaled_hi)));
    lu.push(LutLookup::rc16(Column::single(m.norm_scaled_hi)));
    lu.push(LutLookup::rc16(Column::linear_combination([(
        m.norm_scaled_hi,
        F::from_canonical_u64(1 << 11),
    )])));
    // ISQRT_REM, ISQRT_S2 < 2^22 (high limb < 2^6 via the 2^10-scaled cap).
    for (val, hi) in [(m.isqrt_rem, m.isqrt_rem_hi), (m.isqrt_s2, m.isqrt_s2_hi)] {
        lu.push(LutLookup::rc16(low_limb(val, hi)));
        lu.push(LutLookup::rc16(Column::single(hi)));
        lu.push(LutLookup::rc16(Column::linear_combination([(hi, F::from_canonical_u64(1 << 10))])));
    }

    // ---- D: denom bracket. Lower slack 4q - DENOM_BLO - DENOM_PARITY >= 0; upper slack
    // DENOM_BHI - 4q - DENOM_PARITY >= 0 (each a two-limb RC16 split). ----
    lu.push(LutLookup::rc16(Column::linear_combination([
        (m.norm_scaled, F::from_canonical_u64(4)),
        (m.denom_blo, neg),
        (m.denom_parity, neg),
        (m.denom_sl_hi, neg16),
    ])));
    lu.push(LutLookup::rc16(Column::single(m.denom_sl_hi)));
    lu.push(LutLookup::rc16(Column::linear_combination([
        (m.denom_bhi, one),
        (m.norm_scaled, -F::from_canonical_u64(4)),
        (m.denom_parity, neg),
        (m.denom_su_hi, neg16),
    ])));
    lu.push(LutLookup::rc16(Column::single(m.denom_su_hi)));

    // ---- V: division bracket. Lower slack DIV_POW - DIV_BLO - SCALE_PARITY >= 0; upper slack
    // DIV_BHI - DIV_POW - SCALE_PARITY >= 0. ----
    lu.push(LutLookup::rc16(Column::linear_combination([
        (m.div_pow, one),
        (m.div_blo, neg),
        (m.scale_parity, neg),
        (m.div_sl_hi, neg16),
    ])));
    lu.push(LutLookup::rc16(Column::single(m.div_sl_hi)));
    lu.push(LutLookup::rc16(Column::linear_combination([
        (m.div_bhi, one),
        (m.div_pow, neg),
        (m.scale_parity, neg),
        (m.div_su_hi, neg16),
    ])));
    lu.push(LutLookup::rc16(Column::single(m.div_su_hi)));

    // ---- E: multiply bracket. Lower slack 4P - ENTRY_BLO - ENTRY_PARITY >= 0; upper slack
    // ENTRY_BHI - 4P - ENTRY_PARITY >= 0 (0 on pad rows -> in domain unfiltered). ----
    lu.push(LutLookup::rc16(Column::linear_combination([
        (m.entry_prod, F::from_canonical_u64(4)),
        (m.entry_blo, neg),
        (m.entry_parity, neg),
        (m.entry_sl_hi, neg16),
    ])));
    lu.push(LutLookup::rc16(Column::single(m.entry_sl_hi)));
    lu.push(LutLookup::rc16(Column::linear_combination([
        (m.entry_bhi, one),
        (m.entry_prod, -F::from_canonical_u64(4)),
        (m.entry_parity, neg),
        (m.entry_su_hi, neg16),
    ])));
    lu.push(LutLookup::rc16(Column::single(m.entry_su_hi)));

    // ---- Parity halves (two-sided parity splits). ----
    lu.push(LutLookup::rc16(Column::single(m.denom_half)));
    lu.push(LutLookup::rc16(Column::single(m.scale_half)));
    lu.push(LutLookup::rc16(Column::single(m.entry_half)));

    // ---- E: FP16 exponent window [1, 30] (filtered live; pad rows carry 0). ----
    lu.push(LutLookup::rc16_filtered(
        Column::linear_combination_with_constant([(m.entry_fp16_exp, one)], neg),
        live.clone(),
    ));
    lu.push(LutLookup::rc16_filtered(
        Column::linear_combination_with_constant([(m.entry_fp16_exp, neg)], F::from_canonical_u64(30)),
        live.clone(),
    ));

    // ---- RANGE16 (7-bit mantissa pins): the four BF16 mantissa fields `< 128`, proved by the
    // `col * 2^9 < 2^16` scaled range check (the FP16 batch commits no PAIR128 table; RANGE16 scaled
    // by 9 is the exact 7-bit pin `circuit::fp16::row_scale_stark` uses). 0 on padding -> in domain. ----
    let scaled9 = |col: usize| Column::linear_combination([(col, F::from_canonical_u64(1 << 9))]);
    lu.push(LutLookup::rc16(scaled9(m.mag_minus_1)));
    lu.push(LutLookup::rc16(scaled9(m.denom_mant)));
    lu.push(LutLookup::rc16(scaled9(m.scale_mant)));
    lu.push(LutLookup::rc16(scaled9(m.entry_mant)));

    // ---- FP16POW2: the three shift lookups (the FP16 batch commits FP16POW2, value `2^min(key,26)`;
    // every noise shift key is in `[0, 19] < 26`, so the value is exactly `2^key`). All live-filtered:
    // on the trailing all-zero padding block the keys would be out of range (e.g. denom key `0-134`). ----
    // denom: key gd = DENOM_EXP - 134, value DENOM_POW.
    lu.push(LutLookup {
        table: LutTable::Fp16Pow2,
        keys: vec![Column::linear_combination_with_constant(
            [(m.denom_exp, one)],
            -F::from_canonical_u64(134),
        )],
        values: vec![Column::single(m.denom_pow)],
        filter: live.clone(),
    });
    // division: key A = 283 - DENOM_EXP - SCALE_EXP, value DIV_POW.
    lu.push(LutLookup {
        table: LutTable::Fp16Pow2,
        keys: vec![Column::linear_combination_with_constant(
            [(m.denom_exp, neg), (m.scale_exp, neg)],
            F::from_canonical_u64(283),
        )],
        values: vec![Column::single(m.div_pow)],
        filter: live.clone(),
    });
    // multiply: key gm = ENTRY_GM, value ENTRY_POW (filtered live).
    lu.push(LutLookup {
        table: LutTable::Fp16Pow2,
        keys: vec![Column::single(m.entry_gm)],
        values: vec![Column::single(m.entry_pow)],
        filter: live,
    });

    lu
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::PrimeField64;
    use starky::util::trace_rows_to_poly_values;

    use super::super::stark::NoiseProgram;
    use super::*;

    type F = GoldilocksField;

    #[test]
    fn noise_ctl_halves_are_well_formed() {
        // The table index is supplied at batch-integration time; any placeholder is fine here.
        ctl_noise_bytes_looking::<F>(9);
    }

    #[test]
    fn noise_lut_inventory_matches_documented_counts() {
        let lu = noise_lut_lookups::<F>();
        let count = |t: LutTable| lu.iter().filter(|l| l.table == t).count();
        // The FP16 batch commits RANGE16 + FP16POW2 (no PAIR128/POW2D): RC16 x32 (incl. 4 scaled 7-bit
        // mantissa pins), FP16POW2 x3.
        assert_eq!(count(LutTable::Range16), 32);
        assert_eq!(count(LutTable::Fp16Pow2), 3);
        assert_eq!(count(LutTable::Pair128), 0);
        assert_eq!(count(LutTable::Pow2D), 0);
        assert_eq!(lu.len(), 35);
    }

    /// Every committed-LUT key the honest trace presents lands in its table's domain (RANGE16 keys
    /// `< 2^16`, FP16POW2 keys `in [0, 19]` with `value = 2^key`). Checked on the live rows (filtered
    /// lookups are inactive on the padding rows).
    #[test]
    fn honest_lut_keys_are_in_domain() {
        for (rank, _seed) in [(16usize, 2u64), (32, 8), (48, 44), (64, 70)] {
            let program = NoiseProgram::new(rank);
            let bytes: Vec<u8> = (0..rank).map(|i| ((i as u64 * 2654435761) >> 29) as u8 | 1).collect();
            let rows = program.generate_trace::<F>(&bytes);
            let polys = trace_rows_to_poly_values(rows.clone());
            let lu = noise_lut_lookups::<F>();
            for lookup in &lu {
                for r in 0..rank {
                    match lookup.table {
                        LutTable::Range16 => {
                            let k = lookup.keys[0].eval_table(&polys, r, &[]).to_canonical_u64();
                            assert!(k < 1 << 16, "RC16 key {k} out of range at row {r}");
                        }
                        LutTable::Fp16Pow2 => {
                            let k = lookup.keys[0].eval_table(&polys, r, &[]).to_canonical_u64();
                            let v = lookup.values[0].eval_table(&polys, r, &[]).to_canonical_u64();
                            assert!(k <= 19, "FP16POW2 key {k} out of range at row {r}");
                            assert_eq!(v, 1 << k, "FP16POW2 value {v} != 2^{k} at row {r}");
                        }
                        other => panic!("unexpected table {other:?}"),
                    }
                }
            }
        }
    }
}
