//! One trace row per output cell, followed by padding to a power of two.
//!
//! Each live row imports its magnitude bound and skip count from Matmul, and its
//! two noise scales from Scale. A nonzero cell claimed tamed supplies an integer
//! certificate `2^D <= Y`, with `Y = k*(sig_A*sig_B)^2 < 2^80`.
//!
//! [`super::stark`] derives `D`, including the fixed `tau_tame^2` factor moved
//! into its exponent. Untamed-cell and skip counts accumulate across the tile;
//! the final row checks both public budgets.

use crate::circuit::fp8::columns_view::columns_view;
use crate::circuit::fp8::luts::XFPOW2_LIMBS;

/// Limbs of one partial product `K * PP_LIMB` (`K = k <= 2^16`, limb < 2^16: the product is
/// < 2^32 — two 16-bit limbs).
pub const K_PP_PARTIAL_LIMBS: usize = 2;

/// Digits of the compared sides `2^A` and `Y` (both < 2^96: six base-2^16 digits —
/// `A <= 80` and `Y < 2^80`).
pub const COMPARISON_DIGITS: usize = 6;

/// View of one TamedStark trace row. Labels J1–J7 refer to [`super::stark`].
///
/// Trace generation fills the certificate columns with zero on untamed, zero-cell
/// and padding rows. Untamed and zero-cell flags disable certificate arithmetic
/// and the power lookup. Unfiltered limb and subtraction checks accept this zero fill.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct TamedColumnsView<T: Copy> {
    // Verifier-known schedule columns; see `known_values`.
    /// Cell index `i*w + j` (row-major, Matmul's convention): the E-cell channel key. Pad rows
    /// carry `h*w + t`.
    pub cell_id: T,
    /// The cell's tile-row group key `(i + 1)*k - 1` — the A-slot sigma channel key
    /// (ScaleStark's A-row `GROUP_KEY`). 0 on pads.
    pub a_group_key: T,
    /// The cell's tile-column group key `h*k + (j + 1)*k - 1` — the B-slot sigma channel key.
    /// 0 on pads.
    pub b_group_key: T,
    /// 1 on the last row only: the filter of the J6 TAME_LIMIT gate lookup.
    pub is_last_row: T,
    /// 1 on the trailing power-of-two padding rows: excluded from both import channels,
    /// `UNTAMED` pinned 0 (J1), certificate all-zero.
    pub is_pad: T,

    // CTL imports (each bound by its channel on live rows; free but harmless on pads).
    /// Upper bound on the cell's magnitude exponent, biased by +139, imported from Matmul.
    /// Zero implies an all-zero cell; overstating the bound can only make acceptance harder.
    pub cell_magnitude_exponent: T,
    /// The tile row's noise-std significand (< 2^16: a product of two bf16 significands,
    /// field-bound at the Scale side).
    pub sigma_a_significand: T,
    /// Its exponent, biased by 2048: `sigma = SIG * 2^(EXP - 2048)`.
    pub sigma_a_exp: T,
    /// See [`Self::sigma_a_significand`], for the tile column.
    pub sigma_b_significand: T,
    /// See [`Self::sigma_a_exp`].
    pub sigma_b_exp: T,
    /// Number of this cell's summands marked as skipped, imported from Matmul.
    /// MB15 allows overcounting but prevents undercounting. J7 requires zero on
    /// padding and rejects a tile-wide total above `SKIP_LIMIT`.
    pub cell_skips: T,

    // Verdicts and running counts (J1, J7).
    /// Adds one to the untamed count and disables the tamed certificate when set.
    /// A nonzero cell with this bit cleared must supply a certificate. The prover
    /// may overcount untamed cells; J6 still requires the total to fit `TAME_LIMIT`.
    pub untamed: T,
    /// Running untamed count (prefix sum including this row).
    pub running_untamed: T,
    /// Sum of `CELL_SKIPS` through this row, unchanged on padding.
    /// J7 checks the final total against `SKIP_LIMIT`.
    pub running_skips: T,
    /// Claims an all-zero cell, requiring `cell_magnitude_exponent = 0` (J1).
    /// Zero cells are tamed by definition and bypass the integer certificate.
    /// This flag is mutually exclusive with `untamed`.
    pub cell_is_zero: T,

    // J2: the sigma significand product, split into 16-bit limbs.
    /// `PP = SIGMA_A_SIGNIFICAND * SIGMA_B_SIGNIFICAND` as two range-checked limbs.
    pub sigma_product_limbs: [T; 2],

    // J3: `Y = K * PP^2 < 2^80`, built as `W = K*PP` (3 digits) then `Y = W*PP` (5 digits)
    // by exact base-2^16 schoolbook arithmetic.
    /// Limbs of `K * PP_LO` (< 2^32: two range-checked limbs).
    pub k_sigma_product_lo: [T; K_PP_PARTIAL_LIMBS],
    /// Limbs of `K * PP_HI` (< 2^32).
    pub k_sigma_product_hi: [T; K_PP_PARTIAL_LIMBS],
    /// Middle digit of `W`. Its low digit reuses `K_SIGMA_PRODUCT_LO[0]`; its high
    /// digit is reconstructed as `K_SIGMA_PRODUCT_HI[1] + K_SIGMA_PRODUCT_CARRIES[0] <= 2^16`.
    pub k_sigma_product_middle_limbs: [T; 1],
    /// The carry bit of the digit-aligned add `W = K*PP_LO + 2^16 * K*PP_HI`.
    pub k_sigma_product_carries: [T; 1],
    /// `Y`'s digits 0..=3 (range-checked; digit 4 is [`Self::bound_top`], digit 5 is structurally
    /// zero — `Y < 2^80`).
    pub bound_limbs: [T; 4],
    /// The `W*PP` position-0 carry (< 2^16).
    pub bound_carry_1: T,
    /// Low 16 bits of the position-1..=2 carries (each < 2^17: two 2^32 terms per position).
    pub bound_carries_lo: [T; 2],
    /// High bits of the position-1..=2 carries.
    pub bound_carries_bit: [T; 2],
    /// `Y`'s digit 4 — the position-3 carry (range-checked).
    pub bound_top: T,

    // J4: XFPOW2 supplies the left side of the comparison on tamed nonzero rows.
    /// `2^min(max(D, 0), 80)` as base-2^16 limbs. The doubled frame gap D includes
    /// `TAU_TAME_SQ_LOG2` through `FRAME_GAP_OFFSET`.
    pub comparison_power_limbs: [T; XFPOW2_LIMBS],

    // J5: the digit-wise borrow chain proving `2^A <= Y` (the range-checked per-digit keys live in
    // `super::ctl`; only the borrow bits are committed).
    /// Borrow bits of `Y - 2^A` at digit boundaries 1..=5; no borrow leaves digit 5.
    pub comparison_borrows: [T; COMPARISON_DIGITS - 1],

    // J7: the skip budget gate (`SKIP_LIMIT` can exceed one limb, so the gate difference is
    // committed as a two-limb split; both limbs range-checked, the recomposition constrained on the
    // last row only).
    /// Low 16 bits of `SKIP_LIMIT - RUNNING_SKIPS` on the last row; 0 elsewhere.
    pub skip_gate_slack_lo: T,
    /// High 16 bits of the same difference (< 2^32 for every sanctioned geometry).
    pub skip_gate_slack_hi: T,
}

pub const NUM_TAMED_COLUMNS: usize = size_of::<TamedColumnsView<u8>>();

const _: () = assert!(NUM_TAMED_COLUMNS == 46);

/// Inner dimension k. The fixed tau factor is absorbed into `FRAME_GAP_OFFSET`.
pub const TAME_K_PUBLIC_INPUT: usize = 0;
/// TamedStark public-input index: the untamed-cell allowance `floor(eps_tame * h * w)`.
pub const TAME_LIMIT_PUBLIC_INPUT: usize = 1;
/// TamedStark public-input index: the skippable-summand allowance
/// `floor(eps_pred * k * h * w)`.
pub const SKIP_LIMIT_PUBLIC_INPUT: usize = 2;
/// Number of TamedStark public inputs.
pub const NUM_TAMED_PUBLIC_INPUTS: usize = 3;

columns_view!(TamedColumnsView, NUM_TAMED_COLUMNS, TAMED_COL_MAP);

/// Number of leading verifier-known schedule columns.
pub const NUM_TAMED_KNOWN_COLUMNS: usize = TAMED_COL_MAP.is_pad + 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn col_map_is_the_identity_layout() {
        let as_array: [usize; NUM_TAMED_COLUMNS] = TAMED_COL_MAP.into();
        for (i, &c) in as_array.iter().enumerate() {
            assert_eq!(c, i);
        }
        assert_eq!(TAMED_COL_MAP.cell_id, 0);
        assert_eq!(TAMED_COL_MAP.a_group_key, 1);
        assert_eq!(TAMED_COL_MAP.b_group_key, 2);
        assert_eq!(TAMED_COL_MAP.is_last_row, 3);
        assert_eq!(TAMED_COL_MAP.is_pad, 4);
        assert_eq!(NUM_TAMED_KNOWN_COLUMNS, 5);
    }

    #[test]
    fn view_roundtrips_the_flat_array() {
        let mut arr = [0u64; NUM_TAMED_COLUMNS];
        for (i, v) in arr.iter_mut().enumerate() {
            *v = i as u64 * 3 + 1;
        }
        let view: TamedColumnsView<u64> = arr.into();
        assert_eq!(view.cell_id, 1);
        assert_eq!(
            view.sigma_product_limbs[1],
            TAMED_COL_MAP.sigma_product_limbs[1] as u64 * 3 + 1
        );
        assert_eq!(view.skip_gate_slack_hi, (NUM_TAMED_COLUMNS as u64 - 1) * 3 + 1);
    }
}
