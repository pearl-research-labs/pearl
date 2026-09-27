//! Trace columns for one 32-lane H100 WGMMA accumulation step.
//!
//! A row aligns 32 fp8 products and the incoming carry to one FP22 exponent, sums the signed
//! terms, and truncates the result to H100's 14-bit carry precision. Here “FP22 exponent” means
//! the simulator's internal binary exponent shifted by +139; nonzero products occupy
//! `[127, 155]` and zero uses sentinel 114. Window-final rows promote into an FP32 total;
//! cell-final rows encode that total.
//!
//! [`MatmulColumnsView`] fixes committed column order; [`MATMUL_COL_MAP`] exposes the same
//! layout as flat indices for lookups and cross-table channels.

use crate::circuit::fp8::columns_view::columns_view;

/// Number of fp8 products per WGMMA row; including the incoming carry, each sum has 33 terms.
pub const GROUP_WIDTH: usize = 32;

/// Length of the attainment chain (M3): the running product of the 32 lane factors
/// `GROUP_MAX_PRODUCT_EXPONENT - PROD_FP22_EXP_i`, accumulated at most two new factors per
/// link so every link constraint stays degree <= 3 (`ATT_1` takes three factors, `ATT_16`
/// takes the last one).
pub const NUM_ATT_LINKS: usize = 16;

/// Exact two-input finite-FP32 promotion witness. Values use a shared
/// `floor(log2)+139` exponent frame and explicit normalized significands.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default)]
pub struct PromotionView<T: Copy> {
    /// 1 iff the global FP32 accumulator entering this row is zero.
    pub global_in_is_zero: T,
    /// Sign bit of the incoming global accumulator: 1 for negative, 0 for positive or zero.
    pub global_in_sign: T,
    /// `floor(log2(abs(global_in))) + 139` for a nonzero input; 0 for zero.
    pub global_in_exponent: T,
    /// Normalized 24-bit integer significand of the incoming global accumulator; 0 for zero.
    pub global_in_significand: T,
    /// For two nonzero inputs, selects the larger exponent: 1 for global, 0 for local (either on a tie).
    pub exponent_order: T,
    /// Absolute exponent difference between the two nonzero inputs.
    pub exponent_gap: T,
    /// `2^exponent_gap` on the near path; 1 on the far path.
    pub gap_power: T,
    /// 1 iff both inputs are nonzero and their exponent gap is at most 25.
    pub near_active: T,
    /// 1 iff both inputs are nonzero and their exponent gap exceeds 25.
    pub far_active: T,
    /// 1 iff the near sum is nonzero and its exact magnitude fits in at most 24 bits.
    pub near_low: T,
    /// 1 iff the near sum is nonzero and its exact magnitude needs more than 24 bits.
    pub near_high: T,
    /// Near-path multiplier for the global significand: `gap_power` if `exponent_order = 1`, otherwise 1.
    pub global_scale: T,
    /// Near-path global significand shifted into the smaller exponent's frame.
    pub scaled_global_significand: T,
    /// Near-path local significand, widened to 24 bits and shifted into the smaller exponent's frame.
    pub scaled_local_significand: T,
    /// Absolute value of the exact signed sum of the two aligned significands on the near path.
    pub exact_abs: T,
    /// Bit length of `exact_abs` on near nonzero rows, in `1..=49`.
    pub exact_width: T,
    /// `2^abs(exact_width - 24)`, used to normalize left or divide right on near nonzero rows.
    pub shift_power: T,
    /// Little-endian 16-bit limbs of the pre-rounding significand Q, in `[2^23, 2^24)` on near nonzero rows.
    pub quotient_limbs: [T; 2],
    /// Little-endian 16-bit limbs of `exact_abs` on near-low rows, or division remainder R on near-high rows.
    pub magnitude_or_remainder_limbs: [T; 2],
    /// Little-endian 16-bit limbs of `shift_power - 1 - R`, proving the near-high remainder bound.
    pub remainder_bound_limbs: [T; 2],
    /// On near-high rows, 1 iff `R >= shift_power / 2`.
    pub remainder_ge_half: T,
    /// On near-high rows, 1 iff `R = shift_power / 2`, marking an exact rounding tie.
    pub remainder_eq_half: T,
    /// Near-high comparison slack in little-endian 16-bit limbs: `R - half` if at/above half, else `half - 1 - R`.
    pub remainder_cmp_limbs: [T; 2],
    /// Field inverse of `2*R - shift_power` on near-high non-ties, proving the remainder is not halfway.
    pub remainder_eq_inverse: T,
    /// Low bit of Q on near-high rows, used to round exact ties to even.
    pub quotient_lsb: T,
    /// 1 iff ties-to-even rounding increments Q; 0 outside the near-high path.
    pub round_up: T,
    /// Near rounding carry or certified far-binade drop; 0 on other paths.
    pub binade_correction: T,
    /// Field inverse of `2^24 - Q - round_up` on near nonzero rows without a rounding carry.
    pub carry_slack_inverse: T,
    /// 1 iff this row's projected FP32 sum `global_in + local_out` is zero.
    pub global_out_is_zero: T,
    /// Sign bit of this row's projected FP32 sum; 0 for zero.
    pub global_out_sign: T,
    /// `floor(log2(abs(projected_sum))) + 139` for a nonzero FP32 projection; 0 for zero.
    pub global_out_exponent: T,
    /// Normalized 24-bit integer significand of this row's FP32 projection; 0 for zero.
    pub global_out_significand: T,
}

/// View of one MatmulStark trace row. The constraint labels (M1..M16) refer to
/// `super::stark`'s constraint groups.
///
/// Exponent conventions: `PROD_FP22_EXP`, `GROUP_MAX_PRODUCT_EXPONENT`, and
/// `GROUP_OUTPUT_EXPONENT` use the simulator's FP22 shift +139. Product sentinel 114 is the
/// minimum reachable carry exponent and therefore cannot exceed an honest nonzero maximum
/// (nonzero products span [127, 155]); zero outputs use sentinel 0.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct MatmulColumnsView<T: Copy> {
    // ------------------------------------------------------------------------------------------
    // Shared structural columns, class (a): verifier-recomputable from the program (geometry).
    // Committed *with the trace* (per-job data cannot live in a setup-time preprocessed
    // oracle), but the verifier recomputes them (`MatmulStarkH100::known_values`) and checks the
    // trace openings against its own values (`starky`'s batch "known columns") — that binding
    // carries every schedule fact (booleanness, the cell-final last row), so they appear in no
    // pinning constraint of their own.
    // ------------------------------------------------------------------------------------------
    /// Output cell index, constant across the cell's `k/32` rows; the XorFold channel key.
    pub cell_id: T,
    /// 1 on each cell's last row. Gates the f32 encode, the XorFold channel and the carry reset.
    pub is_cell_final: T,
    /// 1 on the last live atom of each window, including partial final windows.
    pub is_window_final: T,
    /// Operand flat-index base for this row's 32 A lanes (lane i reads
    /// `operand_index_base_a + i`).
    pub operand_index_base_a: T,
    /// Operand flat-index base for the B lanes; carries the constant `h*k` key offset matching
    /// InputQuantStark's B-channel keys.
    pub operand_index_base_b: T,
    /// 1 on the rows that read no operands: the trailing power-of-two padding rows (each its
    /// own single-row phantom cell, `IS_CELL_FINAL = 1`, `CELL_ID >= h*w`). Padding rows are
    /// excluded from the operand-code and cell-result channels, and their lanes are pinned to
    /// zero products (`is_padding * (PROD_FP22_EXP_i - 114) = 0` + the alignment-table
    /// binding).
    pub is_padding: T,

    // ------------------------------------------------------------------------------------------
    // Group-sum emulation (main).
    // ------------------------------------------------------------------------------------------
    /// The lane's fp8 A-operand code, received from InputQuant via the pair-packed CTL and
    /// individually pinned by the PRODALIGN15 tuple's `OPERAND_CODES_A` binding (M1).
    pub operand_codes_a: [T; GROUP_WIDTH],
    /// The lane's fp8 B-operand code (see `operand_codes_a`).
    pub operand_codes_b: [T; GROUP_WIDTH],
    /// The lane's aligned 15-bit-frame contribution as a *signed field element* (sign folded:
    /// negative = `p - v`; magnitude `floor(sig_a*sig_b*2^7 / 2^SHIFT)` with `sig_*` the fp8
    /// significands and `SHIFT = GROUP_MAX_PRODUCT_EXPONENT - PROD_FP22_EXP` the lane's gap to
    /// the frame anchor), bound by PRODALIGN15.
    pub aligned_lane_terms: [T; GROUP_WIDTH],
    /// The product's FP22 exponent, biased +139 (sentinel 114 for zero products), bound by
    /// PRODALIGN15; the lookup key's shift component and the attainment target. There is no
    /// per-lane zero flag: 114 is the floor of every reachable exponent, so the sentinel
    /// doubles as the zero flag and zero-lane attainment is self-defeating (M3).
    pub prod_fp22_exp: [T; GROUP_WIDTH],
    /// "The carry entering this row is zero" flag, pinned by propagation from the previous row's
    /// zero state (1 at cell starts; M4). The carry's sign/exponent/significand are *not*
    /// columns: they are the previous row's group-output fields, read by next-row references.
    pub incoming_carry_is_zero: T,
    /// `2^min(GROUP_MAX_PRODUCT_EXPONENT' - GROUP_OUTPUT_EXPONENT, 14)` from the POW2G LUT, on
    /// the consuming row (M5). The generator uses 1 on inactive zero-carry rows;
    /// the AIR only requires the aligned contribution to be zero there.
    pub incoming_carry_shift_power: T,
    /// The aligned carry
    /// `floor(GROUP_OUTPUT_SIGNIFICAND_prev / 2^min(shift, 14))`, on the consuming row; 0 on
    /// zero-carry rows.
    pub aligned_incoming_carry: T,
    /// Floor remainder witness:
    /// `GROUP_OUTPUT_SIGNIFICAND_prev - ALIGNED_INCOMING_CARRY * INCOMING_CARRY_SHIFT_POWER`
    /// (RC16'd).
    pub incoming_carry_remainder: T,
    /// The group's maximum FP22 exponent over 32 products + carry; pinned two-sidedly: LUT key
    /// domains give >= (negative shifts have no slot), mandatory attainment (M3) gives <=.
    pub group_max_product_exponent: T,
    /// Attainment chain (M3): the running product of the 33 attainment factors — 32 affine lane
    /// factors `GROUP_MAX_PRODUCT_EXPONENT - PROD_FP22_EXP_i` plus the zero-gated carry factor
    /// — accumulated at most two factors per link and constrained to vanish at the end. In a
    /// field a zero product means some factor is zero, i.e. some term attains
    /// `GROUP_MAX_PRODUCT_EXPONENT` exactly. This form needs no per-lane max flags.
    pub max_exponent_attainment: [T; NUM_ATT_LINKS],
    /// Sign bit of the signed frame sum.
    pub group_sum_sign: T,
    /// Magnitude of the signed frame sum (single column, < 2^20 — range-proven by the WIDTHNORM
    /// key domain).
    pub group_sum_abs: T,
    /// Zero-sum flag (boolean; `GROUP_SUM_IS_ZERO * GROUP_SUM_ABS = 0` proves one direction and
    /// M7's floor check proves the other). There is no separate all-zero column.
    pub group_sum_is_zero: T,
    /// Bit-width of the sum (in [1, 20]), bound by the WIDTHNORM tuple keyed `GROUP_SUM_ABS`.
    pub group_sum_width: T,
    /// Group-sum output FP22 exponent
    /// (`GROUP_MAX_PRODUCT_EXPONENT + GROUP_SUM_WIDTH - 14`; 0 on the zero path).
    pub group_output_exponent: T,
    /// Group-sum output 14-bit significand, directly bound by WIDTHNORM; 0 on the zero path.
    /// Its sign is `group_sum_sign`, including canonical +0.
    pub group_output_significand: T,
    /// Low 16-bit limb of the cell result's f32 word, encoded on `IS_CELL_FINAL` rows (M12);
    /// consumed by XorFold.
    pub cell_result_f32_lo: T,
    /// High 16-bit limb of the cell result's f32 word.
    pub cell_result_f32_hi: T,
    /// Window-local result promoted into the separate global FP32 accumulator.
    pub promotion: PromotionView<T>,

    // ------------------------------------------------------------------------------------------
    // Consolidated jackpot policy (M13): E_CELL = e(M)+139 and E_GRID =
    // max(e(Z)-13,e(M)-18)+157. For nonzero x, required range checks enforce:
    // E_CELL >= e(x)+139 for every product and exact unrounded C+c in the cell;
    // E_GRID >= E_CELL and E_GRID >= e(x)+144 for every product/local c in the window.
    // Understatement is unsatisfiable; overstatement only adds skips.
    // ------------------------------------------------------------------------------------------
    /// M13: the lane product's `floor(log2 |product|) + 139` (0 for a zero product), served by
    /// the same PRODALIGN15 lookup as the term. `RC16(E_CELL - LANE_BINADES_i)` proves the
    /// bound; nonzero binades are >= 121, so E_CELL = 0 implies an all-zero cell.
    pub lane_binades: [T; GROUP_WIDTH],
    /// M13: `e(M)+139`, constant across the cell. M covers every product and every exact,
    /// unrounded `C+c` after a live atom.
    pub e_cell: T,
    /// M13: `E_GRID`, constant across one at-most-four-atom window. It is at least E_CELL and
    /// at least each product/local-c binade plus five.
    pub e_grid: T,

    // ------------------------------------------------------------------------------------------
    // Consolidated jackpot census (M14-M16). Lane `u` of a nonzero grid is skippable iff
    // `LAMBDA_A + LAMBDA_B < 128 * E_GRID + 45952`;
    // claiming non-skip costs a range-check certificate, so the census can only be overstated,
    // and XorFold's budget gate rejects totals above the allowance.
    // ------------------------------------------------------------------------------------------
    /// M14: boolean; 0 forces `E_GRID = 0`, while 1 enables the non-skip certificates in M15.
    pub cell_nonzero: T,
    /// M15: the lane's A-operand summand score `lambda`, received from InputQuant on the
    /// widened operand-code channel (0 encodes "no finite summand") — the channel binds it
    /// per element, so the lane inherits InputQuant's exact value and range.
    pub lambda_a: [T; GROUP_WIDTH],
    /// The lane's B-operand summand score (see `lambda_a`).
    pub lambda_b: [T; GROUP_WIDTH],
    /// M15: the lane's skip verdict (boolean; 0 on padding rows and zero cells). One-sided:
    /// claiming *non-skip* costs the filtered RC16 certificate
    /// `LAMBDA_A + LAMBDA_B - 128*E_GRID - 45952 in [0, 2^16)`
    /// (filter `CELL_NONZERO * (1 - SKIP_FLAG)`), while claiming *skip* is free — the census
    /// can only be overstated. A true skip's key wraps far outside the range check.
    pub skip_flag: [T; GROUP_WIDTH],
    /// M16: in-cell running count of `SKIP_FLAG` (anchored to the row sum at each cell
    /// start); the cell-final value rides the result-and-census channel to XorFold's budget gate.
    pub cell_skips: T,
}

/// Total number of committed MatmulStark columns.
pub const NUM_MATMUL_COLUMNS: usize = size_of::<MatmulColumnsView<u8>>();

// 256 per-lane + 6 known schedule + 65 arithmetic (including 36 promotion)
// + 4 jackpot (E_CELL, E_GRID, CELL_NONZERO, CELL_SKIPS).
const _: () = assert!(size_of::<PromotionView<u8>>() == 36);
const _: () = assert!(NUM_MATMUL_COLUMNS == 331);

/// Number of MatmulStark public inputs — none: the AIR is program-independent and the former
/// `D_MEDIAN` policy threshold left with the jackpot policy.
pub const NUM_MATMUL_PUBLIC_INPUTS: usize = 0;

columns_view!(MatmulColumnsView, NUM_MATMUL_COLUMNS, MATMUL_COL_MAP);

/// Number of leading class (a) ("known") columns: `cell_id`, the cell/window boundary flags,
/// `operand_index_base_a`, `operand_index_base_b`, and `is_padding`. They are pure functions
/// of the public geometry and are re-checked against the trace openings.
pub const NUM_MATMUL_H100_KNOWN_COLUMNS: usize = MATMUL_COL_MAP.is_padding + 1;

#[cfg(test)]
mod tests {
    use core::borrow::Borrow;

    use super::*;

    #[test]
    fn col_map_is_the_identity_layout() {
        let as_array: [usize; NUM_MATMUL_COLUMNS] = MATMUL_COL_MAP.into();
        for (i, &c) in as_array.iter().enumerate() {
            assert_eq!(c, i);
        }
        // Class (a) columns come first (their indices feed `preprocessed_indices`).
        assert_eq!(MATMUL_COL_MAP.cell_id, 0);
        assert_eq!(MATMUL_COL_MAP.is_cell_final, 1);
        assert_eq!(MATMUL_COL_MAP.is_window_final, 2);
        assert_eq!(MATMUL_COL_MAP.operand_index_base_a, 3);
        assert_eq!(MATMUL_COL_MAP.operand_index_base_b, 4);
        assert_eq!(MATMUL_COL_MAP.is_padding, 5);
        assert_eq!(NUM_MATMUL_H100_KNOWN_COLUMNS, 6);
        assert_eq!(MATMUL_COL_MAP.cell_skips, NUM_MATMUL_COLUMNS - 1);
    }

    #[test]
    fn view_array_roundtrip() {
        let mut arr = [0u64; NUM_MATMUL_COLUMNS];
        for (i, v) in arr.iter_mut().enumerate() {
            *v = i as u64 * 3 + 1;
        }
        let view: MatmulColumnsView<u64> = arr.into();
        assert_eq!(view.cell_id, 1);
        assert_eq!(view.operand_codes_a[0], MATMUL_COL_MAP.operand_codes_a[0] as u64 * 3 + 1);
        assert_eq!(view.cell_skips, (NUM_MATMUL_COLUMNS as u64 - 1) * 3 + 1);
        let back: [u64; NUM_MATMUL_COLUMNS] = view.into();
        assert_eq!(back, arr);

        let borrowed: &MatmulColumnsView<u64> = arr.borrow();
        assert_eq!(
            borrowed.group_max_product_exponent,
            arr[MATMUL_COL_MAP.group_max_product_exponent]
        );
    }
}
