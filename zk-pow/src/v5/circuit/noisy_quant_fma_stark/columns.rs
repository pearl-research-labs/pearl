//! Fixed trace layout for the FP16 single-rounding FMA AIR (group G2 of the fused noisy
//! quantization).
//!
//! One row per operand element `(i, j)`. The AIR proves `noised = fma(af, X, t)` bit-exact vs the
//! Rust `af.mul_add(fp16_to_f32(raw), t)`, where `af = bf16_to_f32(alpha)`, `X = fp16_to_f32(raw)`,
//! and `t` is group G1's output (`t = RNE_f32(bf*N)`, carried in as `(T_SIGN, T_MANT, T_EXP)`). It
//! is a signed, arbitrarily-aligned add of the exact product `af*X` and `t` with a SINGLE f32
//! round-to-nearest-ties-to-even, including cancellation. The result `noised` is produced as an
//! f32 and exposed (two limbs + decoded fields) to bind to the cast stage (group G3) and the FMA
//! hook of [`crate::v5::circuit::noisy_quant_stark`].
//!
//! The windowing mirrors the A100 accumulation AIR
//! ([`crate::v5::circuit::matmul_a100_stark`]): both addends are normalized to 24-bit
//! significands, aligned onto a common `2^(eta-26)` grid (`eta` = the max value-MSB, FP16POW2
//! shifts, Euclidean floors with a far-gap sticky), summed with sign, then renormalized into
//! `[2^23, 2^24)` (WIDTH32) and rounded once (RNE) with the exact quarter-ulp bracket / IS_BOTTOM
//! / is-zero technique. See [`super::stark`] for the constraint groups and the soundness envelope.
//!
//! [`FmaColumnsView`] is `#[repr(C)]`; declaration order is committed column order.

use crate::v4::circuit::columns_view::columns_view;

/// View of one `NoisyQuantFmaStark` trace row.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct FmaColumnsView<T: Copy> {
    // ------------------------------------------------------------------------------------------
    // Class (a) ("known").
    // ------------------------------------------------------------------------------------------
    /// 1 on trailing padding rows.
    pub is_pad: T,

    // ------------------------------------------------------------------------------------------
    // Inputs (CTL-bound to upstream stages later; decoded here).
    // ------------------------------------------------------------------------------------------
    /// The clean-operand scale `alpha` (BF16 code).
    pub alpha: T,
    /// BF16 exponent field of `alpha`.
    pub alpha_exp: T,
    /// BF16 mantissa field of `alpha` (`M_a = 128 + ALPHA_MANT`).
    pub alpha_mant: T,
    /// The clean FP16 operand code `X`.
    pub raw: T,
    /// FP16DECODE value 0: integer significand `M_x in [0, 2047]`.
    pub x_sig: T,
    /// FP16DECODE value 1: raw sign bit of `X`.
    pub x_sign: T,
    /// FP16DECODE value 2: biased stored exponent `eps + 15 in [1, 30]`.
    pub x_eps_biased: T,
    /// FP16DECODE value 3: `[M_x == 0]`.
    pub x_is_zero: T,
    /// `t` sign bit (from G1: `= NOISE_SIGN`).
    pub t_sign: T,
    /// `t` significand `M_t in [2^23, 2^24)` (from G1), or 0 when `t = 0`.
    pub t_mant: T,
    /// `t` f32 exponent field (from G1).
    pub t_exp: T,
    /// `[M_t == 0]` (t is zero).
    pub t_is_zero: T,
    /// Inverse witness for the `[M_t == 0]` is-zero gadget.
    pub t_mant_inv: T,

    // ------------------------------------------------------------------------------------------
    // Exact product P = af*X significand + 24-bit normalization.
    // ------------------------------------------------------------------------------------------
    /// `P_IS_ZERO = X_IS_ZERO` (the product is zero iff `X` is zero; `af` is always nonzero).
    pub p_is_zero: T,
    /// Exact product significand `Mp = M_a * M_x < 2^19`.
    pub mp: T,
    /// Bit-width `b_p = bitlen(Mp) in [1, 19]` (WIDTH32 key; filtered to nonzero-P rows).
    pub bp: T,
    /// WIDTH32 value `2^max(b_p-24, 0) = 1`.
    pub trunc_p: T,
    /// WIDTH32 value `2^max(24-b_p, 0) = 2^(24-b_p)`.
    pub lift_p: T,
    /// Normalized product significand `Mp_norm = Mp * 2^(24-b_p) in [2^23, 2^24)` (0 when P zero).
    pub mp_norm: T,
    /// Low 16-bit limb of `Mp_norm`.
    pub mp_norm_lo: T,
    /// High limb of `Mp_norm` in `[128, 256)`.
    pub mp_norm_hi: T,

    // ------------------------------------------------------------------------------------------
    // Window anchor eta (biased value-MSB max) and per-term relative shifts.
    // ------------------------------------------------------------------------------------------
    /// The window anchor `eta` = max biased value-MSB over nonzero terms (0 when both zero).
    pub eta: T,
    /// `rel_p = eta - EFF_P` (`EFF_P = (1 - P_IS_ZERO)*P_MSB_BIASED`), `>= 0` (RANGE16).
    pub rel_p: T,
    /// `rel_t = eta - EFF_T`, `>= 0` (RANGE16).
    pub rel_t: T,
    /// Far flag `[rel_p >= 27]` (the P term lies entirely below the window -> aligned 0, sticky).
    pub far_p: T,
    /// Two-sided slack pinning `far_p`: `far_p*(rel_p-27) + (1-far_p)*(26-rel_p) >= 0`.
    pub far_p_slack: T,
    /// Far flag `[rel_t >= 27]`.
    pub far_t: T,
    /// Two-sided slack pinning `far_t`.
    pub far_t_slack: T,
    /// `ACTIVE_P = (1 - far_p)*(1 - P_IS_ZERO)` — the P term aligns with a real FP16POW2 shift.
    pub active_p: T,
    /// `ACTIVE_T = (1 - far_t)*(1 - T_IS_ZERO)`.
    pub active_t: T,

    // ------------------------------------------------------------------------------------------
    // Alignment onto the 2^(eta-26) grid (GUARD = 3 extra low bits).
    // ------------------------------------------------------------------------------------------
    /// `2^rel_p` (FP16POW2, filtered to ACTIVE_P; 1 otherwise).
    pub pow_p: T,
    /// Aligned P magnitude `floor(Mp_norm * 8 / 2^rel_p) < 2^27` (0 when not ACTIVE_P).
    pub aligned_p: T,
    /// Low 16-bit limb of `aligned_p`.
    pub aligned_p_lo: T,
    /// High limb of `aligned_p` (`< 2^11`).
    pub aligned_p_hi: T,
    /// Alignment remainder `Mp_norm*8 - aligned_p*2^rel_p` (`< 2^rel_p <= 2^26`; 2-limb RANGE16).
    pub rem_p: T,
    pub rem_p_lo: T,
    pub rem_p_hi: T,
    /// Two-sided bound `2^rel_p - 1 - rem_p >= 0` (`< 2^26`; 2-limb RANGE16).
    pub rem_p_bound: T,
    pub rem_p_bound_lo: T,
    pub rem_p_bound_hi: T,
    /// `2^rel_t` (FP16POW2, filtered to ACTIVE_T; 1 otherwise).
    pub pow_t: T,
    /// Aligned t magnitude `floor(M_t * 8 / 2^rel_t) < 2^27`.
    pub aligned_t: T,
    pub aligned_t_lo: T,
    pub aligned_t_hi: T,
    pub rem_t: T,
    pub rem_t_lo: T,
    pub rem_t_hi: T,
    pub rem_t_bound: T,
    pub rem_t_bound_lo: T,
    pub rem_t_bound_hi: T,

    // ------------------------------------------------------------------------------------------
    // Sticky bits (dropped below the window).
    // ------------------------------------------------------------------------------------------
    /// `[rem_p != 0]`.
    pub rem_p_nz: T,
    pub rem_p_nz_inv: T,
    /// `[rem_t != 0]`.
    pub rem_t_nz: T,
    pub rem_t_nz_inv: T,
    /// `far_p OR [rem_p != 0]`.
    pub or_p: T,
    /// `far_t OR [rem_t != 0]`.
    pub or_t: T,
    /// P sticky `(1 - P_IS_ZERO)*OR_P`.
    pub sticky_p: T,
    /// t sticky `(1 - T_IS_ZERO)*OR_T`.
    pub sticky_t: T,
    /// `STICKY_P OR STICKY_T` — any bit dropped below the window.
    pub far_sticky: T,
    /// Sign-aware sticky: 1 iff the (single) sub-window sticky term has the SAME sign as the window
    /// result `W`, i.e. the dropped bits ADD to `|W|` (so a round-to-nearest tie rounds up). When the
    /// sticky term has the OPPOSITE sign (a subtraction/cancellation), the dropped bits make `|W|`
    /// slightly SMALLER, so the tie must round DOWN — the bug the plain `far_sticky` tie-break had.
    /// Only the non-dominant term (`rel > 0`) can carry a remainder, so at most one term is sticky.
    pub sticky_up: T,

    // ------------------------------------------------------------------------------------------
    // Signed window sum.
    // ------------------------------------------------------------------------------------------
    /// Sign of the signed window sum (boolean).
    pub w_sign: T,
    /// Magnitude `|W| < 2^28` of the signed window sum.
    pub w_abs: T,
    /// Low 16-bit limb of `|W|`.
    pub w_abs_lo: T,
    /// High limb of `|W|` (`< 2^12`).
    pub w_abs_hi: T,
    /// `[W == 0]` (zero result: both terms zero, or exact cancellation).
    pub w_is_zero: T,
    /// Inverse witness for the `[W == 0]` is-zero gadget.
    pub w_abs_inv: T,

    // ------------------------------------------------------------------------------------------
    // Renormalize |W| to a 24-bit significand (WIDTH32) + single RNE round.
    // ------------------------------------------------------------------------------------------
    /// Bit-width `w = bitlen(|W|) in [1, 28]` (WIDTH32 key; filtered to nonzero-W rows).
    pub ww: T,
    /// WIDTH32 value `2^max(w-24, 0)`.
    pub trunc_w: T,
    /// WIDTH32 value `2^max(24-w, 0)`.
    pub lift_w: T,
    /// Round-toward-zero significand `M_rz = floor(|W|*lift_w / trunc_w) in [2^23, 2^24)`.
    pub m_rz: T,
    /// Low 16-bit limb of `M_rz`.
    pub m_rz_lo: T,
    /// High limb of `M_rz` in `[128, 256)`.
    pub m_rz_hi: T,
    /// RZ remainder `|W|*lift_w - M_rz*trunc_w` (`< trunc_w`).
    pub rz_rem: T,
    /// Two-sided bound `trunc_w - 1 - rz_rem >= 0`.
    pub rz_rem_bound: T,
    /// `M_rz & 1` parity (rides the low limb, RANGE16).
    pub rz_parity: T,
    /// `(M_rz_lo - RZ_PARITY)/2` (RANGE16).
    pub rz_half: T,
    /// `[2*rz_rem > trunc_w]` (strict round-up).
    pub gt: T,
    /// Two-sided slack pinning `gt`: `gt*(2*rz_rem - trunc_w - 1) + (1-gt)*(trunc_w - 2*rz_rem) >= 0`.
    pub gt_slack: T,
    /// `[2*rz_rem == trunc_w]` (exact half -> ties-to-even).
    pub eq: T,
    /// Inverse witness for the `[trunc_w - 2*rz_rem == 0]` is-zero gadget.
    pub eq_inv: T,
    /// The tie-break predicate `STICKY_UP + (1 - FAR_STICKY)*RZ_PARITY` (boolean): round up at a tie
    /// iff a same-sign sub-window remainder pushes `|W|` above the tie, or (no sub-window remainder)
    /// the kept significand is odd (ties-to-even).
    pub or_rs: T,
    /// RNE round-up bit `gt + eq*OR_RS` (boolean).
    pub round_up: T,
    /// Mantissa-overflow carry `[M_rz + round_up == 2^24]`.
    pub carry: T,
    /// Inverse witness for the `[2^24 - M_rz - round_up == 0]` is-zero gadget.
    pub carry_inv: T,

    // ------------------------------------------------------------------------------------------
    // Result f32 `noised`.
    // ------------------------------------------------------------------------------------------
    /// f32 exponent field of `noised` (`= eta + w + carry - 412` on a nonzero result, else 0).
    pub noised_exp: T,
    /// Sign bit of `noised`.
    pub noised_sign: T,
    /// f32 23-bit trailing significand of `noised`.
    pub noised_mant: T,
    /// `noised` low 16-bit limb (the exposed f32 word).
    pub noised_lo: T,
    /// `noised` high 16-bit limb.
    pub noised_hi: T,
}

/// Total committed columns.
pub const NUM_FMA_COLUMNS: usize = size_of::<FmaColumnsView<u8>>();

const _: () = assert!(NUM_FMA_COLUMNS == 90);

/// No public inputs.
pub const NUM_FMA_PUBLIC_INPUTS: usize = 0;

columns_view!(FmaColumnsView, NUM_FMA_COLUMNS, FMA_COL_MAP);

/// Number of leading class (a) ("known") columns: just `IS_PAD`.
pub const NUM_FMA_KNOWN_COLUMNS: usize = FMA_COL_MAP.is_pad + 1;

#[cfg(test)]
mod tests {
    use core::borrow::Borrow;

    use super::*;

    #[test]
    fn col_map_is_the_identity_layout() {
        let as_array: [usize; NUM_FMA_COLUMNS] = FMA_COL_MAP.into();
        for (i, &c) in as_array.iter().enumerate() {
            assert_eq!(c, i);
        }
        assert_eq!(FMA_COL_MAP.is_pad, 0);
        assert_eq!(NUM_FMA_KNOWN_COLUMNS, 1);
        assert_eq!(FMA_COL_MAP.noised_hi, NUM_FMA_COLUMNS - 1);
    }

    #[test]
    fn view_array_roundtrip() {
        let mut arr = [0u64; NUM_FMA_COLUMNS];
        for (i, v) in arr.iter_mut().enumerate() {
            *v = i as u64 * 3 + 1;
        }
        let view: FmaColumnsView<u64> = arr.into();
        assert_eq!(view.is_pad, 1);
        assert_eq!(view.w_abs, FMA_COL_MAP.w_abs as u64 * 3 + 1);
        let back: [u64; NUM_FMA_COLUMNS] = view.into();
        assert_eq!(back, arr);
        let borrowed: &FmaColumnsView<u64> = arr.borrow();
        assert_eq!(borrowed.noised_hi, arr[FMA_COL_MAP.noised_hi]);
    }
}
