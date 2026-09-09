//! Trace columns for TamedStark — jackpot check 3 (tamed products), one row per tile cell.
//!
//! A row imports its cell's replay-magnitude binade `E_CELL = floor(log2 M) + 139` (Matmul's
//! E-cell channel; sentinel 0 for the all-zero cell) and its tile row's/column's exact noise
//! stds `sigma = SIG * 2^(EXP - 2048)` (Scale's sigma channel), commits the cell's `UNTAMED`
//! verdict bit, and — on tamed nonzero rows — proves the certificate
//!
//! ```text
//! 2^(2*e_M) <= tau_tame^2 * K * (sigma_A * sigma_B)^2
//! <=>  2^A <= Y,   Y = K * PP^2 < 2^80,
//! ```
//!
//! where `PP = SIGMA_A_SIGNIFICAND * SIGMA_B_SIGNIFICAND < 2^32`, `K = k <= 2^16` is the
//! public input, and `2^A` is the XFPOW2 saturating power of the folded frame gap with
//! `tau_tame^2 = 2^14` absorbed into the shift (`super::stark` J4). The untamed count runs
//! monotonically and is gated against the `TAME_LIMIT` public input on the last row (J6);
//! the skip census (jackpot check 4) accumulates the same way and is gated against the
//! `SKIP_LIMIT` public input (J7).
//!
//! [`TamedColumnsView`] fixes committed column order; [`TAMED_COL_MAP`] exposes the same
//! layout as flat indices for lookups and cross-table channels.

use crate::circuit::fp8::columns_view::columns_view;
use crate::circuit::fp8::luts::XFPOW2_LIMBS;

/// Limbs of one partial product `K * PP_LIMB` (`K = k <= 2^16`, limb < 2^16: the product is
/// < 2^32 — two 16-bit limbs).
pub const K_PP_PARTIAL_LIMBS: usize = 2;

/// Digits of the compared sides `2^A` and `Y` (both < 2^96: six base-2^16 digits —
/// `A <= 80` and `Y < 2^80`).
pub const COMPARISON_DIGITS: usize = 6;

/// View of one TamedStark trace row. The constraint labels (J1..J7) refer to
/// `super::stark`'s constraint groups; the certificate columns (J2-J5) are all-zero on
/// untamed and zero-cell rows (their constraints are gated by `1 - UNTAMED - CELL_IS_ZERO`)
/// and on pad rows (whose all-zero imports satisfy the active gates vacuously).
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct TamedColumnsView<T: Copy> {
    // ------------------------------------------------------------------------------------------
    // Shared structural columns, class (a): verifier-recomputable from the program (geometry).
    // Committed *with the trace* but re-derived and checked by the batch verifier
    // (`TamedProgram::known_values`, `starky`'s batch "known columns") — that binding carries
    // every schedule fact, so they appear in no pinning constraint of their own.
    // ------------------------------------------------------------------------------------------
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

    // ------------------------------------------------------------------------------------------
    // CTL imports (each bound by its channel on live rows; free but harmless on pads).
    // ------------------------------------------------------------------------------------------
    /// The cell's replay-magnitude binade `E_CELL = floor(log2 M) + 139` (0 only for an
    /// all-zero cell, enforced by Matmul's M13/MB13).
    pub e_cell: T,
    /// The tile row's noise-std significand (< 2^16: a product of two bf16 significands,
    /// field-bound at the Scale side).
    pub sigma_a_significand: T,
    /// Its exponent, biased by 2048: `sigma = SIG * 2^(EXP - 2048)`.
    pub sigma_a_exp: T,
    /// See [`Self::sigma_a_significand`], for the tile column.
    pub sigma_b_significand: T,
    /// See [`Self::sigma_a_exp`].
    pub sigma_b_exp: T,
    /// The cell's skip census — how many of its `k` summands Matmul's per-lane certificates
    /// (M15/MB15) flagged as skippable; rides the E-cell channel. Pinned 0 on pads (J7);
    /// Matmul's census can only be overstated, and the J7 gate rejects totals above
    /// `SKIP_LIMIT`.
    pub cell_skips: T,

    // ------------------------------------------------------------------------------------------
    // Verdicts and censuses (J1, J7).
    // ------------------------------------------------------------------------------------------
    /// The cell's untamed bit. 0 activates the tamed certificate below, so a prover can only
    /// *overstate* the untamed count — and the J6 gate rejects counts above `TAME_LIMIT`.
    pub untamed: T,
    /// Running untamed count (prefix sum including this row).
    pub running_untamed: T,
    /// Running skip census (prefix sum of `CELL_SKIPS` including this row), frozen through
    /// pads; the last row's value faces the J7 budget gate.
    pub running_skips: T,
    /// 1 claims the all-zero cell (J1 pins `CELL_IS_ZERO * E_CELL = 0`): M = 0 is tamed by
    /// definition, so the certificate is skipped — it would demand `Y >= 2^A > 0`, which a
    /// zero cell need not prove. Mutually exclusive with `UNTAMED`.
    pub cell_is_zero: T,

    // ------------------------------------------------------------------------------------------
    // J2: the sigma significand product, split into 16-bit limbs.
    // ------------------------------------------------------------------------------------------
    /// `PP = SIGMA_A_SIGNIFICAND * SIGMA_B_SIGNIFICAND` as two RC16'd limbs.
    pub sigma_product_limbs: [T; 2],

    // ------------------------------------------------------------------------------------------
    // J3: `Y = K * PP^2 < 2^80`, built as `W = K*PP` (3 digits) then `Y = W*PP` (5 digits)
    // by exact base-2^16 schoolbook arithmetic.
    // ------------------------------------------------------------------------------------------
    /// Limbs of `K * PP_LO` (< 2^32: two RC16'd limbs).
    pub k_pp_lo_partial: [T; K_PP_PARTIAL_LIMBS],
    /// Limbs of `K * PP_HI` (< 2^32).
    pub k_pp_hi_partial: [T; K_PP_PARTIAL_LIMBS],
    /// `W`'s digit 1 (digit 0 is `K_PP_LO_PARTIAL[0]` verbatim; digit 2 is the
    /// constraint-side expression `K_PP_HI_PARTIAL[1] + K_PP_CARRIES[0] <= 2^16`).
    pub k_pp_mid_limbs: [T; 1],
    /// The carry bit of the digit-aligned add `W = K*PP_LO + 2^16 * K*PP_HI`.
    pub k_pp_carries: [T; 1],
    /// `Y`'s digits 0..=3 (RC16'd; digit 4 is [`Self::bound_top`], digit 5 is structurally
    /// zero — `Y < 2^80`).
    pub bound_limbs: [T; 4],
    /// The `W*PP` position-0 carry (< 2^16).
    pub bound_carry_1: T,
    /// Low 16 bits of the position-1..=2 carries (each < 2^17: two 2^32 terms per position).
    pub bound_carries_lo: [T; 2],
    /// High bits of the position-1..=2 carries.
    pub bound_carries_bit: [T; 2],
    /// `Y`'s digit 4 — the position-3 carry (RC16'd).
    pub bound_top: T,

    // ------------------------------------------------------------------------------------------
    // J4: the left side 2^A, straight from the XFPOW2 table. On live tamed nonzero rows the
    // lookup binds the limbs at key E_CELL - SIGMA_A_EXP - SIGMA_B_EXP + 4974; elsewhere it
    // is off and the limbs stay zero. The limbs are already 2^A's base-2^16 digits.
    // ------------------------------------------------------------------------------------------
    /// `2^A` as base-2^16 limbs (`A = min(max(D, 0), 80)`, `D` the doubled frame gap with
    /// `tau_tame^2 = 2^14` folded in).
    pub shift_a_limbs: [T; XFPOW2_LIMBS],

    // ------------------------------------------------------------------------------------------
    // J5: the digit-wise borrow chain proving `2^A <= Y` (the RC16'd per-digit keys live in
    // `super::ctl`; only the borrow bits are committed).
    // ------------------------------------------------------------------------------------------
    /// Borrow bits of `Y - 2^A` at digit boundaries 1..=5; no borrow leaves digit 5.
    pub comparison_borrows: [T; COMPARISON_DIGITS - 1],

    // ------------------------------------------------------------------------------------------
    // J7: the skip budget gate (`SKIP_LIMIT` can exceed one limb, so the gate difference is
    // committed as a two-limb split; both limbs RC16'd, the recomposition constrained on the
    // last row only).
    // ------------------------------------------------------------------------------------------
    /// Low 16 bits of `SKIP_LIMIT - RUNNING_SKIPS` on the last row; 0 elsewhere.
    pub skip_gate_slack_lo: T,
    /// High 16 bits of the same difference (< 2^32 for every sanctioned geometry).
    pub skip_gate_slack_hi: T,
}

/// Total number of committed TamedStark columns.
pub const NUM_TAMED_COLUMNS: usize = size_of::<TamedColumnsView<u8>>();

// Committed-column count: 5 class (a) + 6 imports + 4 verdict/census + 2 sigma product +
// 16 bound build + 6 shift limbs + 5 borrows + 2 gate slack limbs.
const _: () = assert!(NUM_TAMED_COLUMNS == 46);

/// TamedStark public-input index: the inner dimension `K = k`. The untamed threshold is
/// `tau_tame^2 * k * (sigma_A*sigma_B)^2` with `tau_tame^2 = 2^14` folded into the XFPOW2
/// shift (`super::stark::FRAME_GAP_OFFSET`), so only `k` rides the proof.
pub const TAME_K_PUBLIC_INPUT: usize = 0;
/// TamedStark public-input index: the untamed-cell allowance `floor(eps_tame * h * w)`.
pub const TAME_LIMIT_PUBLIC_INPUT: usize = 1;
/// TamedStark public-input index: the skippable-summand allowance
/// `floor(eps_pred * k * h * w)`.
pub const SKIP_LIMIT_PUBLIC_INPUT: usize = 2;
/// Number of TamedStark public inputs.
pub const NUM_TAMED_PUBLIC_INPUTS: usize = 3;

columns_view!(TamedColumnsView, NUM_TAMED_COLUMNS, TAMED_COL_MAP);

/// Number of leading class (a) ("known") columns: `cell_id`, `a_group_key`, `b_group_key`,
/// `is_last_row`, and `is_pad`. Pure functions of the public geometry, re-checked against the
/// trace openings.
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
        // Class (a) columns come first (their indices feed the known-column binding).
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
