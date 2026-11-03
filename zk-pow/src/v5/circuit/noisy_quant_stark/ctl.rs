//! Cross-table-lookup declarations for the FP16 NoisyQuantStark.
//!
//! Two kinds of channel:
//!
//! * **Parameterized hooks** — this AIR proves only the two roundings G1/G3 (see [`super::stark`]);
//!   binding its witness inputs to their producing/consuming stages is left to later integration,
//!   so each builder takes the counterparty table's batch index explicitly, exactly like
//!   [`crate::v5::circuit::xor_fold_stark::ctl`] and [`crate::v5::circuit::noise_stark::ctl`]
//!   (NoisyQuantStark is not registered in [`crate::v5::circuit::ctl`]):
//!   - [`ctl_raw_operand_looking`] exposes `raw` (the clean FP16 operand) for the operand commitment.
//!   - [`ctl_noise_word_looking`] exposes the noise word `N_ij` (two f32 limbs) for the `N = E@F^T`
//!     matmul.
//!   - [`ctl_fma_hook_looking`] exposes `(alpha, raw, t_sign, t_mant, t_exp, noised_lo, noised_hi)`
//!     so the **deferred G2** single-rounding FMA stage can bind `noised = fma(af, X, t)`.
//!   - [`ctl_output_looking`] exposes `out` for the output-tile commitment.
//! * The committed-LUT inventory ([`noisy_quant_lut_lookups`]): the RANGE16 limb/slack range checks
//!   and the FP16POW2 shift that make G1's and G3's ties-to-even brackets sound. No new table is
//!   introduced — every fact targets the shared `RANGE16` and `FP16POW2` tables of the FP16 batch.

use plonky2::field::types::Field;
use starky::cross_table_lookup::{TableIdx, TableWithColumns};
use starky::lookup::{Column, Filter};

use super::columns::NOISY_QUANT_COL_MAP;
use crate::v4::circuit::luts::LutTable;
use crate::v4::circuit::luts::ctl::LutLookup;

/// The live-row filter `1 - IS_PAD`.
fn live_filter<F: Field>() -> Filter<F> {
    Filter::from_column(Column::linear_combination_with_constant([(NOISY_QUANT_COL_MAP.is_pad, -F::ONE)], F::ONE))
}

/// The live-and-nonzero-noise filter `NZ_LIVE` (a committed boolean = `(1-IS_PAD)*(1-NOISE_IS_ZERO)`).
fn nz_live_filter<F: Field>() -> Filter<F> {
    Filter::from_column(Column::single(NOISY_QUANT_COL_MAP.nz_live))
}

/// Hook: the clean FP16 operand code `raw`, filter `1 - IS_PAD`. Counterparty: the operand
/// commitment (index supplied at integration time).
pub fn ctl_raw_operand_looking<F: Field>(table: usize) -> TableWithColumns<F> {
    TableWithColumns::new(TableIdx::from(table), vec![Column::single(NOISY_QUANT_COL_MAP.raw)], live_filter())
}

/// Hook: the noise word `N_ij` (low, high f32 limbs), filter `1 - IS_PAD`. Counterparty: the
/// `N = E@F^T` noise matmul.
pub fn ctl_noise_word_looking<F: Field>(table: usize) -> TableWithColumns<F> {
    TableWithColumns::new(
        TableIdx::from(table),
        Column::singles([NOISY_QUANT_COL_MAP.noise_lo, NOISY_QUANT_COL_MAP.noise_hi]).collect(),
        live_filter(),
    )
}

/// 6e noise-word **looked** side: `(ELEMENT_INDEX, NOISE_LO, NOISE_HI)` per live element, filter
/// `1 - IS_PAD`. The key `ELEMENT_INDEX` is the global operand element index (A elements `0..h*k`,
/// B elements past them), matching the `N = E@F^T` noise matmul's `CELL_ID = i*k + j` (its output is
/// `(h+w) x k`, row-major). `(NOISE_LO, NOISE_HI)` is the f32 noise word this element's G1 rounding
/// consumes (low/high 16-bit limbs). The noise matmul's looking side
/// ([`crate::v5::circuit::matmul_a100_stark::ctl::ctl_cell_results_looked_matmul`]) emits each
/// finished cell's `(CELL_ID, CELL_RESULT_F32_LO, CELL_RESULT_F32_HI)` once, so the multiset balance
/// forces every consumed noise word to equal the proven `E@F^T` product — closing the noise's
/// free-witness gap (increment 6e-1). `E`/`F` remain free witness (binding them to the seed-derived
/// keyed-XOF lines is increment 6e-2/6e-3), so the noise is `E@F` but not yet seed-bound.
pub fn ctl_noise_word_looked_noisy_quant<F: Field>(table: usize) -> TableWithColumns<F> {
    let m = &NOISY_QUANT_COL_MAP;
    TableWithColumns::new(
        TableIdx::from(table),
        Column::singles([m.element_index, m.noise_lo, m.noise_hi]).collect(),
        live_filter(),
    )
}

/// Hook: the deferred-G2 FMA tuple `(alpha, raw, t_sign, t_mant, t_exp, noised_lo, noised_hi)`,
/// filter `1 - IS_PAD`. The G2 stage proves `noised = fma(bf16_to_f32(alpha), fp16_to_f32(raw), t)`
/// where `t` is reconstructed from `(t_sign, t_mant, t_exp)`; `t_sign = NOISE_SIGN`.
pub fn ctl_fma_hook_looking<F: Field>(table: usize) -> TableWithColumns<F> {
    let m = &NOISY_QUANT_COL_MAP;
    TableWithColumns::new(
        TableIdx::from(table),
        Column::singles([m.alpha, m.raw, m.noise_sign, m.t_mant, m.t_exp, m.noised_lo, m.noised_hi]).collect(),
        live_filter(),
    )
}

/// Hook: the output FP16 code `out`, filter `1 - IS_PAD`. Counterparty: the output-tile commitment.
pub fn ctl_output_looking<F: Field>(table: usize) -> TableWithColumns<F> {
    TableWithColumns::new(TableIdx::from(table), vec![Column::single(NOISY_QUANT_COL_MAP.out)], live_filter())
}

/// 6c operand-bytes **looked** side: `(2*ELEMENT_INDEX, RAW)` per live element, filter `1 - IS_PAD`.
/// The key `2*ELEMENT_INDEX` is the element's low-byte offset in the Blake3 operand stream (A
/// elements `0..h*k` at byte `2*e`, B elements at byte `2*h*k + 2*e_B = 2*e`), matching Blake3's
/// looking side ([`crate::v5::circuit::blake3_commit::ctl_operand_bytes_looking_blake3`], key
/// `CTL_KEY_BASE + 2j`, value the committed LE `u16`). Each committed operand element crosses exactly
/// once on each side, so the multiset balance forces `RAW` to equal the committed byte pair — the
/// clean operand the matmul's noised codes derive from (Blake3's bytes are BYTES2-checked, making the
/// pair packing sound). Closes audit Finding 3's operand-bytes leg.
pub fn ctl_operand_bytes_looked_noisy_quant<F: Field>(table: usize) -> TableWithColumns<F> {
    let m = &NOISY_QUANT_COL_MAP;
    TableWithColumns::new(
        TableIdx::from(table),
        vec![
            Column::linear_combination([(m.element_index, F::from_canonical_u64(2))]),
            Column::single(m.raw),
        ],
        live_filter(),
    )
}

/// 6d operand-codes **looked** side: `(ELEMENT_INDEX, OUT)` per live element, filter
/// `OPERAND_MULT * (1 - IS_PAD)` (a degree-2 multiplicity filter). The key `ELEMENT_INDEX` is the
/// matmul's `operand_index_base_{a,b} + lane` global element index; `OUT` is the proven noised FP16
/// code. The matmul reuses each A element in `w` output cells and each B element in `h` cells, so the
/// multiplicity `OPERAND_MULT` (`w` on A rows, `h` on B rows) lets this single keyed tuple balance
/// the matmul's repeated lookups ([`crate::v5::circuit::matmul_a100_stark::ctl::ctl_operand_codes_looking_matmul`]).
/// The multiset equality binds BOTH operand provenance (matmul code == noised `OUT`) AND cross-cell
/// row/column sharing (every cell of a row reuses the same A element because they look up the same
/// key). Closes audit Finding 3's noised->matmul leg.
pub fn ctl_operand_codes_looked_noisy_quant<F: Field>(table: usize) -> TableWithColumns<F> {
    let m = &NOISY_QUANT_COL_MAP;
    TableWithColumns::new(
        TableIdx::from(table),
        vec![Column::single(m.element_index), Column::single(m.out)],
        Filter::new(
            vec![(
                Column::single(m.operand_mult),
                Column::linear_combination_with_constant([(m.is_pad, -F::ONE)], F::ONE),
            )],
            vec![],
        ),
    )
}

/// NoisyQuantStark's per-row committed-LUT inventory (RANGE16 + FP16POW2 only).
pub fn noisy_quant_lut_lookups<F: Field>() -> Vec<LutLookup<F>> {
    let m = &NOISY_QUANT_COL_MAP;
    let one = F::ONE;
    let neg = -F::ONE;
    let two16 = F::from_canonical_u64(1 << 16);
    let two32 = F::from_canonical_u64(1 << 32);
    let live = live_filter::<F>();
    let mut lu: Vec<LutLookup<F>> = Vec::new();

    // ---- FP16POW2: the G1 rounding shift 2^d1, keyed on T_SHIFT (filtered live; zero-noise rows
    // present key 0 -> value 1, a valid in-domain entry). ----
    lu.push(LutLookup {
        table: LutTable::Fp16Pow2,
        keys: vec![Column::single(m.t_shift)],
        values: vec![Column::single(m.t_pow)],
        filter: live.clone(),
    });

    // ---- G1 field range checks. ----
    // beta_mant < 128 (the 2^9-scaled check), noise_exp < 256 (2^8-scaled), noise_mant_hi < 2^7.
    lu.push(LutLookup::rc16(Column::linear_combination([(m.beta_mant, F::from_canonical_u64(1 << 9))])));
    lu.push(LutLookup::rc16(Column::linear_combination([(m.noise_exp, F::from_canonical_u64(1 << 8))])));
    lu.push(LutLookup::rc16(Column::single(m.noise_mant_lo)));
    lu.push(LutLookup::rc16(Column::linear_combination([(m.noise_mant_hi, F::from_canonical_u64(1 << 9))])));
    // P limbs (P < 2^32).
    lu.push(LutLookup::rc16(Column::single(m.pm_lo)));
    lu.push(LutLookup::rc16(Column::single(m.pm_hi)));
    // M_t limbs: lo < 2^16 and hi in [128, 256) (the (hi - 128)*2^8 check) -> M_t in [2^23, 2^24).
    // The [128, 256) check is filtered to live nonzero-noise rows: on zero-noise/padding rows
    // M_t = 0 (hi = 0), where `(0 - 128)*2^8` would be out of RANGE16's domain.
    lu.push(LutLookup::rc16(Column::single(m.t_mant_lo)));
    lu.push(LutLookup::rc16_filtered(
        Column::linear_combination_with_constant(
            [(m.t_mant_hi, F::from_canonical_u64(1 << 8))],
            -F::from_canonical_u64(128 << 8),
        ),
        nz_live_filter(),
    ));
    lu.push(LutLookup::rc16(Column::single(m.t_half)));

    // ---- G1 bracket slacks (each < 2^34, split lo/mid/hi). ----
    // Lower slack SL = 4*P - T_BLO - T_PARITY = low + 2^16*T_SL_MID + 2^32*T_SL_HI.
    lu.push(LutLookup::rc16(Column::linear_combination([
        (m.pm, F::from_canonical_u64(4)),
        (m.t_blo, neg),
        (m.t_parity, neg),
        (m.t_sl_mid, -two16),
        (m.t_sl_hi, -two32),
    ])));
    lu.push(LutLookup::rc16(Column::single(m.t_sl_mid)));
    lu.push(LutLookup::rc16(Column::linear_combination([(m.t_sl_hi, F::from_canonical_u64(1 << 14))])));
    // Upper slack SU = T_BHI - 4*P - T_PARITY = low + 2^16*T_SU_MID + 2^32*T_SU_HI.
    lu.push(LutLookup::rc16(Column::linear_combination([
        (m.t_bhi, one),
        (m.pm, -F::from_canonical_u64(4)),
        (m.t_parity, neg),
        (m.t_su_mid, -two16),
        (m.t_su_hi, -two32),
    ])));
    lu.push(LutLookup::rc16(Column::single(m.t_su_mid)));
    lu.push(LutLookup::rc16(Column::linear_combination([(m.t_su_hi, F::from_canonical_u64(1 << 14))])));

    // ---- G3 field range checks. ----
    lu.push(LutLookup::rc16(Column::single(m.noised_mant_lo)));
    lu.push(LutLookup::rc16(Column::linear_combination([(m.noised_mant_hi, F::from_canonical_u64(1 << 9))])));
    lu.push(LutLookup::rc16(Column::linear_combination([(m.noised_exp, F::from_canonical_u64(1 << 8))])));
    lu.push(LutLookup::rc16(Column::single(m.sat_slack)));
    lu.push(LutLookup::rc16(Column::single(m.ge113_slack)));
    lu.push(LutLookup::rc16(Column::single(m.ge102_slack)));
    lu.push(LutLookup::rc16(Column::single(m.q_lo_slack)));
    lu.push(LutLookup::rc16(Column::single(m.q_half)));

    // ---- FP16POW2: the cast rounding shift 2^CAST_SHIFT, keyed on CAST_SHIFT, filtered to the
    // rounding branches (normal/subnormal); 0 off them (CAST_SHIFT in [13, 24] on those rows). ----
    lu.push(LutLookup {
        table: LutTable::Fp16Pow2,
        keys: vec![Column::single(m.cast_shift)],
        values: vec![Column::single(m.cast_pow)],
        filter: Filter::from_column(Column::linear_combination([(m.f_norm, F::ONE), (m.f_sub, F::ONE)])),
    });

    // ---- G3 cast bracket slacks (each < 2^26, split low + 2^16*hi with hi < 2^10). ----
    // Both carry the constant `4*2^23`, so on the all-zero padding rows the raw key would be `+/-2^25`
    // (out of RANGE16's domain). They are therefore filtered to live rows (padding rows are
    // unconstrained; the standalone soundness is unchanged — every live row is still checked).
    // SL = 4*(2^23 + NOISED_MANT) - CAST_BLO - Q_PARITY.
    lu.push(LutLookup::rc16_filtered(
        Column::linear_combination_with_constant(
            [
                (m.noised_mant, F::from_canonical_u64(4)),
                (m.cast_blo, neg),
                (m.q_parity, neg),
                (m.cast_sl_hi, -two16),
            ],
            F::from_canonical_u64(4 * (1 << 23)),
        ),
        live.clone(),
    ));
    lu.push(LutLookup::rc16(Column::linear_combination([(m.cast_sl_hi, F::from_canonical_u64(1 << 6))])));
    // SU = CAST_BHI - 4*(2^23 + NOISED_MANT) - Q_PARITY.
    lu.push(LutLookup::rc16_filtered(
        Column::linear_combination_with_constant(
            [
                (m.cast_bhi, one),
                (m.noised_mant, -F::from_canonical_u64(4)),
                (m.q_parity, neg),
                (m.cast_su_hi, -two16),
            ],
            -F::from_canonical_u64(4 * (1 << 23)),
        ),
        live.clone(),
    ));
    lu.push(LutLookup::rc16(Column::linear_combination([(m.cast_su_hi, F::from_canonical_u64(1 << 6))])));

    lu
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;

    use super::*;

    type F = GoldilocksField;

    #[test]
    fn ctl_halves_are_well_formed() {
        ctl_raw_operand_looking::<F>(3);
        ctl_noise_word_looking::<F>(3);
        ctl_noise_word_looked_noisy_quant::<F>(5);
        ctl_fma_hook_looking::<F>(3);
        ctl_output_looking::<F>(3);
        ctl_operand_bytes_looked_noisy_quant::<F>(5);
        ctl_operand_codes_looked_noisy_quant::<F>(5);
    }

    #[test]
    fn lut_inventory_matches_documented_counts() {
        let lu = noisy_quant_lut_lookups::<F>();
        let count = |t: LutTable| lu.iter().filter(|l| l.table == t).count();
        assert_eq!(count(LutTable::Fp16Pow2), 2); // G1 t_shift + G3 cast_shift
        assert_eq!(count(LutTable::Range16), lu.len() - 2);
    }
}
