//! Committed-LUT inventory for the A100 FP16 matmul AIR.
//!
//! The matmul AIR trusts a family of auxiliary columns — the per-operand decode fields, the
//! per-lane and carry alignment powers, the RZ truncation/lifting powers, and the FP32 result
//! limbs. This module declares, as [`LutLookup`]s, how each is served by a committed LUT whose
//! own AIR constrains it, so that the matmul STARK's *own* verification (the CTL multiset
//! balance of [`super::super::ctl`]) enforces them:
//!
//! * **FP16DECODE** (`2^16` keys), two lookups per lane: `code -> (sig, sign, eps_biased,
//!   is_zero)`. MA1 derives each lane's product significand, sign, and biased stored exponent
//!   from these, so a forged decode breaks either the lookup or MA1.
//! * **FP16POW2** (`2^min(d, 26)`, `d in [0, 255]`), one lookup per lane, one for the
//!   incoming carry, and one (subnormal-filtered) for the MA13 subnormal RZ divisor
//!   `SUB_SHIFT_POWER = 2^k`: the alignment divisor keyed on the relative shift
//!   `GROUP_MAX_BIASED_EXPONENT - PRODUCT_BIASED_EXPONENT`. A negative shift wraps out of the
//!   key domain, so the lookup also proves the eta ">=" side
//!   (`GROUP_MAX_BIASED_EXPONENT >= PRODUCT_BIASED_EXPONENT`). The carry instance is anchored
//!   at the producing row and reads the consuming (next) row's witness, filtered off when the
//!   carry is zero — mirroring FP8 B200's `Pow2Gb` carry lookup.
//! * **WIDTH32** (keys `[1, 32]`), one filtered lookup: `W -> (2^max(W-24,0), 2^max(24-W,0))`.
//!   The shifted-ramp domain also proves `1 <= GROUP_SUM_WIDTH <= 32`.
//! * **RANGE16**, the FP32 result limbs `CELL_RESULT_F32_LO/HI < 2^16` (MA9's limb split), the
//!   MA12 limb-range splits of the Euclidean-floor witnesses (`aligned_mag`, `lane_rem`,
//!   `lane_rem_bound` per lane; `aligned_carry`, `carry_rem`, `carry_rem_bound`; `norm_sig` — each
//!   as `lo < 2^16` and `2^6*hi < 2^16`, bounding the magnitude by `~2^26`), `norm_sig`
//!   additionally pinned to the normalized range `[2^23, 2^24)` on nonzero-sum rows (so the MA7
//!   width identity cannot admit a false `GROUP_SUM_WIDTH` — the B200 MB7 check), and the RZ
//!   remainder `trunc_rem`/`trunc_rem_bound` (`< TRUNCATION_POWER <= 2^6`).
//!
//! With MA12's reconstruction constraints (`value = lo + 2^16*hi`), the limb ranges make the
//! per-lane and carry `floor` identities integer-exact against Goldilocks field-fraction aliasing
//! in a real FRI proof (a wrapped quotient/remainder has no valid 16/10-bit limb witness), closing
//! the gap-1 residual the feasibility doc flagged. The decode/shift/width facts are the ones it
//! enumerated as "every decoded sig/eps, every shift power, every width".

use plonky2::field::types::Field;
use starky::cross_table_lookup::{TableIdx, TableWithColumns};
use starky::lookup::{Column, Filter};

use super::columns::{GROUP, MATMUL_A100_COL_MAP};
use crate::v4::circuit::luts::LutTable;
use crate::v4::circuit::luts::ctl::LutLookup;

/// The A100 matmul's **looked** side of the cell-results channel: the finished output tile cell
/// `(CELL_ID, CELL_RESULT_F32_LO, CELL_RESULT_F32_HI)`, filtered to each cell's final, live row
/// (`IS_CELL_FINAL * (1 - IS_PADDING)`). The XorFold AIR's looking side
/// ([`super::super::xor_fold_stark::ctl::ctl_cell_results_looking_xor_fold`]) folds exactly these
/// cell words into the lottery lanes, so the multiset equality binds the folded tile to the matmul's
/// proven output.
///
/// This is the FP16 analogue of
/// [`crate::v4::circuit::matmul_h100::ctl::ctl_cell_results_looked_matmul`], dropping the FP8
/// `CELL_SKIPS` census field (FP16 scores its jackpot in the policy AIR, and the matmul -> policy
/// census-import CTL carries the per-step census). The matmul table's batch index is supplied by the
/// batch-integration step.
pub fn ctl_cell_results_looked_matmul<F: Field>(matmul_table: usize) -> TableWithColumns<F> {
    ctl_cell_results_looked_matmul_offset(matmul_table, 0)
}

/// Like [`ctl_cell_results_looked_matmul`] but adds a constant `cell_offset` to the `CELL_ID` key.
/// Used by the noise-word channel (6e-2) to shift the B-side noise matmul's cell ids (`0..w*k`) into
/// the shared noisy-quant element-index space (`h*k..(h+w)*k`), so both noise matmuls' cell results
/// balance one noisy-quant looked side keyed by the global element index.
pub fn ctl_cell_results_looked_matmul_offset<F: Field>(matmul_table: usize, cell_offset: usize) -> TableWithColumns<F> {
    let m = &MATMUL_A100_COL_MAP;
    TableWithColumns::new(
        TableIdx::from(matmul_table),
        vec![
            Column::linear_combination_with_constant([(m.cell_id, F::ONE)], F::from_canonical_usize(cell_offset)),
            Column::single(m.cell_result_f32_lo),
            Column::single(m.cell_result_f32_hi),
        ],
        Filter::new(
            vec![(
                Column::single(m.is_cell_final),
                Column::linear_combination_with_constant([(m.is_padding, -F::ONE)], F::ONE),
            )],
            vec![],
        ),
    )
}

/// The A100 matmul's **looking** side of the operand-codes channel (6d): per live row, for each of
/// the 8 lanes of each operand, the tuple `(operand_index_base_{a,b} + lane, operand_codes_{a,b}[lane])`
/// — the global operand element index and the FP16 code the matmul multiplies. Filter
/// `1 - IS_PADDING`. `operand_index_base_a/b` are class (a) known columns (verifier-recomputed from
/// geometry), so the keys are fixed and only the `operand_codes` values are witness. The counterpart
/// looked side ([`crate::v5::circuit::noisy_quant_stark::ctl::ctl_operand_codes_looked_noisy_quant`])
/// emits each element's noised `OUT` once, with the reuse multiplicity, so the multiset balance binds
/// every matmul operand code to the noised quantization of the committed operand — and forces cells
/// of one output row (resp. column) to reuse the same A (resp. B) element, since they key on the same
/// `operand_index_base + lane`. The FP16 analogue of
/// [`crate::v4::circuit::matmul_h100::ctl::ctl_operand_codes_looking_matmul`] (one u16 code per lane,
/// not a packed int8 pair; no summand-score lambdas — FP16 scores its jackpot in the policy AIR). The
/// matmul table's batch index is supplied by the batch-integration step.
pub fn ctl_operand_codes_looking_matmul<F: Field>(matmul_table: usize) -> Vec<TableWithColumns<F>> {
    ctl_operand_codes_looking_matmul_offset(matmul_table, 0)
}

/// Like [`ctl_operand_codes_looking_matmul`] but adds a constant `index_offset` to every operand
/// element key (`operand_index_base + lane + index_offset`). Used by the E/F-operand binding channel
/// (6e-2): a noise matmul's operand codes (`E`/`F`) are bound to NoiseStark's proven normalized line
/// entries, keyed by the global entry index `line * rank + entry`. The A-side noise matmul needs no
/// offset (its own `[0, (h+k)*r)` index space already matches NoiseStark's leading `E_A`,`F_A`
/// blocks); the B-side matmul's entire `[0, (w+k)*r)` space is shifted by `(h+k)*r` onto the trailing
/// `E_B`,`F_B` blocks.
pub fn ctl_operand_codes_looking_matmul_offset<F: Field>(
    matmul_table: usize,
    index_offset: usize,
) -> Vec<TableWithColumns<F>> {
    let m = &MATMUL_A100_COL_MAP;
    let not_padding = || {
        Filter::from_column(Column::linear_combination_with_constant([(m.is_padding, -F::ONE)], F::ONE))
    };
    let mut lookups = Vec::with_capacity(2 * GROUP);
    for i in 0..GROUP {
        for (base, codes) in [(m.operand_index_base_a, &m.operand_codes_a), (m.operand_index_base_b, &m.operand_codes_b)] {
            lookups.push(TableWithColumns::new(
                TableIdx::from(matmul_table),
                vec![
                    Column::linear_combination_with_constant(
                        [(base, F::ONE)],
                        F::from_canonical_usize(i + index_offset),
                    ),
                    Column::single(codes[i]),
                ],
                not_padding(),
            ));
        }
    }
    lookups
}

/// Every committed-LUT instance the A100 matmul AIR consumes on a row: FP16DECODE x16 (8 lanes
/// x 2 operands), FP16POW2 x10 (8 lanes + 1 carry + 1 MA13 subnormal RZ divisor), WIDTH32 x1,
/// RANGE16 x64 (2 result limbs + 58 MA12/MA13 floor-witness limb checks + 2 RZ-remainder checks
/// + 1 MA13 exp_slack + 1 NORM_SIG [2^23,2^24) range pin) — 91 instances. The inventory is
/// program-independent (no geometry
/// constants bake into any key/value/filter).
pub fn matmul_a100_lut_lookups<F: Field>() -> Vec<LutLookup<F>> {
    let m = &MATMUL_A100_COL_MAP;
    let one = F::ONE;
    let neg = -F::ONE;
    let mut lookups = Vec::with_capacity(2 * GROUP + (GROUP + 1) + 1 + 60);

    // ---- FP16DECODE: the per-operand decode of both operands of every lane. ----
    for i in 0..GROUP {
        lookups.push(LutLookup {
            table: LutTable::Fp16Decode,
            keys: vec![Column::single(m.operand_codes_a[i])],
            values: Column::singles([m.sig_a[i], m.sign_a[i], m.eps_a[i], m.is_zero_a[i]]).collect(),
            filter: Filter::default(),
        });
        lookups.push(LutLookup {
            table: LutTable::Fp16Decode,
            keys: vec![Column::single(m.operand_codes_b[i])],
            values: Column::singles([m.sig_b[i], m.sign_b[i], m.eps_b[i], m.is_zero_b[i]]).collect(),
            filter: Filter::default(),
        });
    }

    // ---- FP16POW2: per-lane alignment divisor, keyed on the relative shift. ----
    for i in 0..GROUP {
        lookups.push(LutLookup {
            table: LutTable::Fp16Pow2,
            keys: vec![Column::linear_combination([
                (m.group_max_biased_exponent, one),
                (m.product_biased_exp[i], neg),
            ])],
            values: vec![Column::single(m.lane_shift_power[i])],
            filter: Filter::default(),
        });
    }

    // ---- FP16POW2: the incoming-carry alignment, anchored at the producing row (reads the
    // consuming/next row), filtered off when that carry is zero. Out-of-domain negative keys
    // prove GROUP_MAX_BIASED_EXPONENT' >= the carry's exponent; the cyclic wrap is inert because
    // row 0's INCOMING_CARRY_IS_ZERO is anchored to 1 (MA6). ----
    lookups.push(LutLookup {
        table: LutTable::Fp16Pow2,
        keys: vec![Column::linear_combination_and_next_row_with_constant(
            vec![(m.out_biased_exp, neg)],
            vec![(m.group_max_biased_exponent, one)],
            F::ZERO,
        )],
        values: vec![Column::single_next_row(m.carry_shift_power)],
        filter: Filter::from_column(Column::linear_combination_and_next_row_with_constant(
            vec![],
            vec![(m.incoming_carry_is_zero, neg)],
            one,
        )),
    });

    // ---- WIDTH32: the RZ truncate/lift powers, keyed on the claimed width (1..=32). ----
    let group_sum_is_nonzero = Filter::from_column(Column::linear_combination_with_constant([(m.group_sum_is_zero, neg)], one));
    lookups.push(LutLookup {
        table: LutTable::Width32,
        keys: vec![Column::single(m.group_sum_width)],
        values: Column::singles([m.truncation_power, m.lifting_power]).collect(),
        filter: group_sum_is_nonzero.clone(),
    });

    // ---- FP16POW2: the subnormal RZ divisor SUB_SHIFT_POWER = 2^k (MA13), keyed on the
    // subnormal shift exponent k = 1 - raw, filtered to subnormal-output rows. The key domain
    // (k in [1, 26]) bounds the shift; the lookup binds the committed power. ----
    lookups.push(LutLookup {
        table: LutTable::Fp16Pow2,
        keys: vec![Column::single(m.sub_shift_exp)],
        values: vec![Column::single(m.sub_shift_power)],
        filter: Filter::from_column(Column::single(m.out_is_subnormal)),
    });

    // ---- RANGE16: the FP32 cell-result limbs (MA9). ----
    lookups.push(LutLookup::rc16(Column::single(m.cell_result_f32_lo)));
    lookups.push(LutLookup::rc16(Column::single(m.cell_result_f32_hi)));

    // ---- RANGE16: the MA13 subnormal flag slack (`exp_slack < 2^16`; pins the flag to
    // `[raw <= 0]` by having no valid nonnegative witness for the wrong claim). ----
    lookups.push(LutLookup::rc16(Column::single(m.exp_slack)));

    // ---- RANGE16: MA12 limb-range splits of the Euclidean-floor witnesses. For each tracked
    // magnitude, `lo` is checked `< 2^16` and `2^6 * hi` is checked `< 2^16` (so `hi < 2^10`,
    // bounding the magnitude by `~2^26`). With MA12's reconstruction constraint
    // (`value = lo + 2^16*hi`) these make the per-lane and carry `floor` identities integer-exact
    // against Goldilocks field-fraction aliasing in a real FRI proof. ----
    let hi6 = F::from_canonical_u64(1 << 6);
    let mut split = |lo: usize, hi: usize| {
        lookups.push(LutLookup::rc16(Column::single(lo)));
        lookups.push(LutLookup::rc16(Column::linear_combination([(hi, hi6)])));
    };
    for i in 0..GROUP {
        split(m.aligned_mag_lo[i], m.aligned_mag_hi[i]);
        split(m.lane_rem_lo[i], m.lane_rem_hi[i]);
        split(m.lane_rem_bound_lo[i], m.lane_rem_bound_hi[i]);
    }
    split(m.aligned_carry_lo, m.aligned_carry_hi);
    split(m.carry_rem_lo, m.carry_rem_hi);
    split(m.carry_rem_bound_lo, m.carry_rem_bound_hi);
    split(m.norm_sig_lo, m.norm_sig_hi);
    // MA13: the subnormal mantissa OUT_SIG's limbs (0 off the subnormal path, so the check is
    // vacuous there; on subnormal rows it bounds OUT_SIG < 2^26). Combined with the NORM_SIG
    // normalized-range pin below (NORM_SIG < 2^24 on nonzero-sum rows) and the exact division
    // NORM_SIG = OUT_SIG * 2^k (k >= 1), this forces OUT_SIG < 2^23 — a valid subnormal mantissa.
    split(m.out_sig_lo, m.out_sig_hi);
    drop(split);

    // ---- RANGE16: pin NORM_SIG into the normalized range [2^23, 2^24) on nonzero-sum rows. ----
    // The MA12 split above only bounds NORM_SIG < 2^26. That is NOT enough: the MA7 width identity
    // `GROUP_SUM_ABS * LIFTING_POWER = NORM_SIG * TRUNCATION_POWER + TRUNC_REM` (with 0 <= TRUNC_REM
    // < TRUNCATION_POWER) then admits *any* GROUP_SUM_WIDTH in [1,32] for a given sum magnitude —
    // the prover can pick a false width, choose TRUNC_REM != 0, and so forge RZ_DROPPED (MA11),
    // hence GROUP_BREAKPOINT, hence the jackpot census (f_bp / rho). Forcing
    // `(NORM_SIG_HI - 128) * 2^9 < 2^16` pins NORM_SIG_HI in [128, 256), i.e. NORM_SIG in
    // [2^23, 2^24), which pins GROUP_SUM_WIDTH to the true bit-width and makes the RZ breakpoint
    // exact. This mirrors the FP8 B200 MB7 normalized-significand check (whose omission here was a
    // soundness gap); filtered off on zero-sum rows, where NORM_SIG is unconstrained (and 0).
    lookups.push(LutLookup::rc16_filtered(
        Column::linear_combination_with_constant(
            [(m.norm_sig_hi, F::from_canonical_u64(1 << 9))],
            -F::from_canonical_u64(128 << 9),
        ),
        group_sum_is_nonzero.clone(),
    ));

    // ---- RANGE16: the RZ Euclidean remainder and its two-sided bound (both < TRUNCATION_POWER
    // <= 2^6 < 2^16, so no limb split is needed). Together they pin `0 <= TRUNC_REM <
    // TRUNCATION_POWER` as genuine small integers, making the MA7 normalization floor exact. ----
    lookups.push(LutLookup::rc16(Column::single(m.trunc_rem)));
    lookups.push(LutLookup::rc16(Column::single(m.trunc_rem_bound)));

    lookups
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;

    use super::*;

    type F = GoldilocksField;

    #[test]
    fn cell_results_looked_half_is_well_formed() {
        // Any placeholder table index is fine; the real index is supplied at batch integration.
        let _ = ctl_cell_results_looked_matmul::<F>(0);
        assert_eq!(ctl_operand_codes_looking_matmul::<F>(0).len(), 2 * GROUP, "8 lanes x 2 operands");
    }

    #[test]
    fn inventory_matches_documented_counts() {
        let lookups = matmul_a100_lut_lookups::<F>();
        let count = |t: LutTable| lookups.iter().filter(|l| l.table == t).count();
        assert_eq!(count(LutTable::Fp16Decode), 2 * GROUP);
        // 8 lanes + 1 carry + 1 subnormal RZ divisor (MA13).
        assert_eq!(count(LutTable::Fp16Pow2), GROUP + 2);
        assert_eq!(count(LutTable::Width32), 1);
        // RANGE16: 2 cell-result limbs + 2 per MA12-split witness (3 per lane + 3 carry + norm_sig
        // + the MA13 out_sig mantissa = 29 witnesses -> 58 limb checks) + the RZ remainder and its
        // bound + the MA13 exp_slack + the NORM_SIG [2^23,2^24) normalized-range pin = 64.
        let range16 = 2 + 2 * (3 * GROUP + 3 + 1 + 1) + 2 + 1 + 1;
        assert_eq!(count(LutTable::Range16), range16);
        assert_eq!(count(LutTable::Range16), 64);
        assert_eq!(lookups.len(), 2 * GROUP + (GROUP + 2) + 1 + range16);
    }
}
