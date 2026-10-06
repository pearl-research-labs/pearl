//! Fixed trace layout for the FP16 fused noisy-quantization AIR.
//!
//! One row per operand element `(i, j)`. The AIR proves two of the three f32 roundings of the
//! plaintext ground truth [`crate::v5::api::quantization::noisy_quantize`]'s per-element fused
//! kernel
//!
//! ```text
//! t      = RNE_f32(bf * N_ij)                       // group G1 (pre-FMA f32 multiply)
//! noised = fma(af, X_ij, t)   // SINGLE-rounding f32 FMA — GROUP G2, DEFERRED (see mod.rs)
//! out    = f32_to_fp16(clamp(noised, -MAX, MAX))    // group G3 (clamp + FP16 cast)
//! ```
//!
//! with `af = bf16_to_f32(alpha)`, `bf = bf16_to_f32(beta)`, `X_ij = fp16_to_f32(raw_ij)`.
//!
//! `alpha`, `beta`, `raw`, the noise word `N_ij` and the fused f32 `noised` all enter as witness
//! **input** columns; binding `raw`/`N`/`noised`/`out` to their producing/consuming stages (the
//! operand commitment, the `N = E@F^T` matmul, the single-rounding FMA, and the output tile) is a
//! set of separate later CTL stages — this AIR proves the arithmetic of G1 and G3 only, and
//! [`super::ctl`] exposes the parameterized channel hooks. See [`super::stark`] for the
//! constraint groups and the documented soundness envelope.
//!
//! [`NoisyQuantColumnsView`] is `#[repr(C)]`; declaration order is committed column order.

use crate::v4::circuit::columns_view::columns_view;

/// View of one `NoisyQuantStark` trace row.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct NoisyQuantColumnsView<T: Copy> {
    // ------------------------------------------------------------------------------------------
    // Class (a) ("known") column (verifier-recomputed, checked against the trace openings). Must
    // come first.
    // ------------------------------------------------------------------------------------------
    /// 1 on the trailing padding rows (live elements padded to a power of two).
    pub is_pad: T,
    /// The operand-row index this element belongs to (`floor(element / k)`): A rows `0..h`, B rows
    /// `h..h+w`, in one disjoint space. The scales-import CTL's key, matching
    /// [`crate::v5::circuit::row_scale_stark`]'s `OPERAND_ROW_INDEX` so each element consumes its
    /// own operand row's `(alpha, beta)`. A known column (recomputed by the verifier). Held at the
    /// final live index on padding rows (filtered out of the CTL).
    pub operand_row_index: T,
    /// The global operand element index this row proves (`e` in `0..(h+w)*k`, the trace row index on
    /// live rows; held at the final live index on padding). Known column (recomputed by the
    /// verifier). The two operand-provenance channels key on it:
    /// * **operand-bytes** (6c): key `2*ELEMENT_INDEX` = the element's low-byte offset in the Blake3
    ///   operand stream, so the committed byte pair (Blake3) equals this element's `RAW`.
    /// * **operand-codes** (6d): key `ELEMENT_INDEX` = the matmul's `operand_index_base_{a,b} + lane`,
    ///   so the matmul's operand code equals this element's noised `OUT`.
    pub element_index: T,
    /// The matmul reuse multiplicity of this element (`w` for an A-side element — reused in `w`
    /// output cells — and `h` for a B-side element). Known column (recomputed by the verifier). It is
    /// the per-row multiplicity the operand-codes (6d) looked side carries, so the single
    /// `ELEMENT_INDEX` key balances the matmul's repeated lookups of each operand element. Held at the
    /// final live element's multiplicity on padding rows (filtered out of the CTL by `1 - IS_PAD`).
    pub operand_mult: T,

    // ------------------------------------------------------------------------------------------
    // Pure CTL-hook inputs (NOT constrained by this AIR; bound to their producing stages later).
    // ------------------------------------------------------------------------------------------
    /// The per-row clean-operand scale `alpha` (BF16 code). Hook for the deferred G2 FMA.
    pub alpha: T,
    /// The clean FP16 operand code `X_ij`. Hook for the operand-commitment binding.
    pub raw: T,
    /// The deferred-G2 FMA output `noised` as an f32, low 16-bit limb. Hook + G3 decodes it.
    pub noised_lo: T,
    /// `noised` high 16-bit limb.
    pub noised_hi: T,

    // ==========================================================================================
    // GROUP G1 — t = RNE_f32(bf * N_ij).
    // ==========================================================================================
    /// The per-row noise scale `beta` (BF16 code); decoded below.
    pub beta: T,
    /// BF16 exponent field of `beta` (structurally normal positive).
    pub beta_exp: T,
    /// BF16 mantissa field of `beta` in `[0, 127]` (`M_beta = 128 + BETA_MANT`).
    pub beta_mant: T,
    /// The noise word `N_ij` as an f32, low 16-bit limb. Hook for the `N = E@F^T` binding.
    pub noise_lo: T,
    /// `N_ij` high 16-bit limb.
    pub noise_hi: T,
    /// Sign bit of `N_ij`.
    pub noise_sign: T,
    /// f32 exponent field of `N_ij` (`[0, 254]`; envelope: normal, or 0 meaning exact zero).
    pub noise_exp: T,
    /// f32 23-bit trailing significand of `N_ij`.
    pub noise_mant: T,
    /// Low 16-bit limb of `NOISE_MANT`.
    pub noise_mant_lo: T,
    /// High 7-bit limb of `NOISE_MANT` (`NOISE_MANT = lo + 2^16*hi`).
    pub noise_mant_hi: T,
    /// `[NOISE_EXP == 0]` flag — a zero (or, out of envelope, subnormal) noise word gives `t = 0`.
    pub noise_is_zero: T,
    /// Inverse witness for the `[NOISE_EXP == 0]` is-zero gadget.
    pub noise_exp_inv: T,
    /// Noise significand `M_N = (1 - NOISE_IS_ZERO)*2^23 + NOISE_MANT` (24-bit, 0 when zero).
    pub mn: T,
    /// Exact product significand `P = M_beta * M_N < 2^32` (0 when the noise is zero).
    pub pm: T,
    /// Low 16-bit limb of `P`.
    pub pm_lo: T,
    /// High 16-bit limb of `P` (`P = lo + 2^16*hi`).
    pub pm_hi: T,
    /// RNE rounding shift `d1 = bitlen(P) - 24 in {7, 8}` (FP16POW2 key; 0 when the noise is zero).
    pub t_shift: T,
    /// `2^d1` (FP16POW2 value; 1 when the noise is zero).
    pub t_pow: T,
    /// 24-bit significand `M_t = RNE(P / 2^d1) in [2^23, 2^24)` of `t` (0 when the noise is zero).
    pub t_mant: T,
    /// Low 16-bit limb of `M_t`.
    pub t_mant_lo: T,
    /// High limb of `M_t` in `[128, 256)` (`M_t = lo + 2^16*hi`).
    pub t_mant_hi: T,
    /// Binade-bottom flag `[M_t == 2^23]` (significand is a power of two): lower boundary is
    /// `-ulp/4`, not `-ulp/2`. Boolean, pinned by the is-zero gadget on `M_t - 2^23`.
    pub t_bottom: T,
    /// Inverse witness for the `[M_t == 2^23]` is-zero gadget.
    pub t_bottom_inv: T,
    /// `M_t & 1` (ties-to-even parity; boolean).
    pub t_parity: T,
    /// `(T_MANT_LO - T_PARITY) / 2` (RANGE16; two-sided parity split on M_t's low limb, which
    /// shares M_t's parity since the high limb's `2^16` weight is even — keeps T_HALF a single limb).
    pub t_half: T,
    /// Lower quarter-ulp boundary product `(4*M_t - 2 + T_BOTTOM) * 2^d1`.
    pub t_blo: T,
    /// Upper quarter-ulp boundary product `(4*M_t + 2) * 2^d1`.
    pub t_bhi: T,
    /// Middle limb of the lower bracket slack `4*P - T_BLO - T_PARITY` (`< 2^34`).
    pub t_sl_mid: T,
    /// Top limb (bits 32..) of the lower bracket slack.
    pub t_sl_hi: T,
    /// Middle limb of the upper bracket slack `T_BHI - 4*P - T_PARITY`.
    pub t_su_mid: T,
    /// Top limb of the upper bracket slack.
    pub t_su_hi: T,
    /// f32 exponent field of `t` (`BETA_EXP + NOISE_EXP + T_SHIFT - 134` when nonzero, else 0).
    pub t_exp: T,

    // ==========================================================================================
    // GROUP G3 — out = f32_to_fp16(noised). (The clamp is subsumed: f32_to_fp16 already
    // saturates, so f32_to_fp16(clamp(x, +/-MAX)) == f32_to_fp16(x) for every finite x.)
    // ==========================================================================================
    /// Sign bit of `noised`.
    pub noised_sign: T,
    /// f32 exponent field of `noised` in `[1, 254]` (any normal f32 — the FP16 cast then lands in
    /// the normal/subnormal/zero/saturating branch per its exponent).
    pub noised_exp: T,
    /// f32 23-bit trailing significand of `noised`.
    pub noised_mant: T,
    /// Low 16-bit limb of `NOISED_MANT`.
    pub noised_mant_lo: T,
    /// High 7-bit limb of `NOISED_MANT`.
    pub noised_mant_hi: T,
    // --- Branch classification by the f32 exponent E: saturate (E>=143), normal (113<=E<=142),
    //     subnormal (102<=E<=112), zero (E<=101). Three monotone flags + two-sided slacks pin them.
    /// `F_SAT = [E >= 143]`.
    pub f_sat: T,
    /// Two-sided slack: `F_SAT*(E-143) + (1-F_SAT)*(142-E) >= 0`.
    pub sat_slack: T,
    /// `GE113 = [E >= 113]`.
    pub ge113: T,
    /// Two-sided slack: `GE113*(E-113) + (1-GE113)*(112-E) >= 0`.
    pub ge113_slack: T,
    /// Normal-output flag `F_NORM = GE113*(1 - F_SAT)`.
    pub f_norm: T,
    /// `GE102 = [E >= 102]`.
    pub ge102: T,
    /// Two-sided slack: `GE102*(E-102) + (1-GE102)*(101-E) >= 0`.
    pub ge102_slack: T,
    /// Subnormal-output flag `F_SUB = GE102*(1 - GE113)`.
    pub f_sub: T,
    /// Zero-output flag `F_ZERO = 1 - GE102`.
    pub f_zero: T,
    /// Cast rounding shift: `13` on the normal branch, `126 - E` on the subnormal branch, `0`
    /// otherwise (FP16POW2 key; `<= 24`).
    pub cast_shift: T,
    /// `2^CAST_SHIFT` (FP16POW2 value; `1` off the rounding branches).
    pub cast_pow: T,
    /// Rounded significand `q = RNE((2^23 + NOISED_MANT) / 2^CAST_SHIFT)` on the rounding branches:
    /// `[2^10, 2^11]` (normal) or `[0, 2^10]` (subnormal).
    pub q: T,
    /// RANGE16 slack `q - 2^10 >= 0` (normal branch only).
    pub q_lo_slack: T,
    /// Binade-bottom flag `[q == 2^10]`.
    pub q_bottom: T,
    /// Inverse witness for the `[q == 2^10]` is-zero gadget (on `q - 2^10`).
    pub q_bottom_inv: T,
    /// `q & 1` (ties-to-even parity; boolean).
    pub q_parity: T,
    /// `(q - Q_PARITY) / 2` (RANGE16).
    pub q_half: T,
    /// Lower quarter-ulp boundary product `(4*q - 2 + Q_BOTTOM) * CAST_POW`.
    pub cast_blo: T,
    /// Upper quarter-ulp boundary product `(4*q + 2) * CAST_POW`.
    pub cast_bhi: T,
    /// High 16-bit limb of the lower bracket slack `4*(2^23 + NOISED_MANT) - CAST_BLO - Q_PARITY`.
    pub cast_sl_hi: T,
    /// High 16-bit limb of the upper bracket slack `CAST_BHI - 4*(2^23 + NOISED_MANT) - Q_PARITY`.
    pub cast_su_hi: T,
    /// Rounding-carry flag `[q == 2^11]` (normal round overflowed the significand; exponent bumps).
    pub carry: T,
    /// Inverse witness for the `[q == 2^11]` is-zero gadget (on `2^11 - q`).
    pub carry_inv: T,
    /// `[NOISED_EXP == 142]` flag (a normal carry here means `e_adj = 16`, i.e. saturation).
    pub e142: T,
    /// Inverse witness for the `[NOISED_EXP == 142]` is-zero gadget (on `142 - NOISED_EXP`).
    pub e142_inv: T,
    /// `CCE = CARRY * E142` (committed to keep the saturation-fold degree <= 3).
    pub cce: T,
    /// Effective saturation flag `IS_SAT_EFF = F_SAT + F_NORM*CCE` (also the FP16-inf carry case).
    pub is_sat_eff: T,
    /// Non-saturating normal flag `FNNS = F_NORM*(1 - CCE)`.
    pub fnns: T,
    /// Normal FP16 output mantissa field `(1 - CARRY)*(q - 2^10)` (degree-budget helper).
    pub mf: T,
    /// `(1 - IS_PAD) * (1 - NOISE_IS_ZERO)` — the live-and-nonzero-noise gate for G1's bracket
    /// (committed so that gating degree-2 boundary constraints stays degree <= 3; boolean).
    pub nz_live: T,
    /// The output FP16 code.
    pub out: T,
}

/// Total number of committed columns.
pub const NUM_NOISY_QUANT_COLUMNS: usize = size_of::<NoisyQuantColumnsView<u8>>();

const _: () = assert!(NUM_NOISY_QUANT_COLUMNS == 76);

/// No public inputs: the identity is program-independent (geometry enters via `IS_PAD` + height).
pub const NUM_NOISY_QUANT_PUBLIC_INPUTS: usize = 0;

columns_view!(NoisyQuantColumnsView, NUM_NOISY_QUANT_COLUMNS, NOISY_QUANT_COL_MAP);

/// Number of leading class (a) ("known") columns: `IS_PAD`, `OPERAND_ROW_INDEX`, `ELEMENT_INDEX`
/// and `OPERAND_MULT`.
pub const NUM_NOISY_QUANT_KNOWN_COLUMNS: usize = NOISY_QUANT_COL_MAP.operand_mult + 1;

#[cfg(test)]
mod tests {
    use core::borrow::Borrow;

    use super::*;

    #[test]
    fn col_map_is_the_identity_layout() {
        let as_array: [usize; NUM_NOISY_QUANT_COLUMNS] = NOISY_QUANT_COL_MAP.into();
        for (i, &c) in as_array.iter().enumerate() {
            assert_eq!(c, i);
        }
        assert_eq!(NOISY_QUANT_COL_MAP.is_pad, 0);
        assert_eq!(NOISY_QUANT_COL_MAP.operand_row_index, 1);
        assert_eq!(NOISY_QUANT_COL_MAP.element_index, 2);
        assert_eq!(NOISY_QUANT_COL_MAP.operand_mult, 3);
        assert_eq!(NUM_NOISY_QUANT_KNOWN_COLUMNS, 4);
        assert_eq!(NOISY_QUANT_COL_MAP.out, NUM_NOISY_QUANT_COLUMNS - 1);
    }

    #[test]
    fn view_array_roundtrip() {
        let mut arr = [0u64; NUM_NOISY_QUANT_COLUMNS];
        for (i, v) in arr.iter_mut().enumerate() {
            *v = i as u64 * 3 + 1;
        }
        let view: NoisyQuantColumnsView<u64> = arr.into();
        assert_eq!(view.is_pad, 1);
        assert_eq!(view.pm, NOISY_QUANT_COL_MAP.pm as u64 * 3 + 1);
        let back: [u64; NUM_NOISY_QUANT_COLUMNS] = view.into();
        assert_eq!(back, arr);

        let borrowed: &NoisyQuantColumnsView<u64> = arr.borrow();
        assert_eq!(borrowed.out, arr[NOISY_QUANT_COL_MAP.out]);
    }
}
