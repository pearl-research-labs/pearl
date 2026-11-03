//! Fixed trace layout for the FP16 noise-line normalization AIR.
//!
//! One row per noise entry (one raw XOF byte), plus trailing all-zero `IS_PAD = 1` rows that
//! pad the live `rank` rows to a power of two. The **line-level** scalars (the sum of squares,
//! the integer square root, and the BF16 scale derivation) are a pure function of the whole
//! line, so they are replicated identically on every row (pinned to the final prefix sum by the
//! last-row equality and the transition-equality of `TOTAL_SUMSQ`); each row then also proves
//! its **own** entry (`entry_i = fp16(bf16(x_i) * scale)`) from that shared scale.
//!
//! All fixed-point BF16 roundings are proved with round-to-nearest / ties-to-even *bracket*
//! gadgets (quarter-ulp integer comparisons, mirroring `circuit::fp8::scale_stark`'s sqrt
//! bracket), reusing only the shared `RANGE16` and `POW2D` committed LUTs — no new table is
//! added. See `stark.rs` for the per-group constraint derivations and the documented
//! soundness envelope.
//!
//! [`NoiseColumnsView`] is `#[repr(C)]`; declaration order is committed column order.

use crate::v4::circuit::columns_view::columns_view;

/// View of one NoiseStark trace row.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct NoiseColumnsView<T: Copy> {
    // ------------------------------------------------------------------------------------------
    // Class (a) ("known") columns: pure functions of the geometry `(h, w, k, rank)`, verifier-
    // recomputed and checked against the trace openings (`starky`'s batch known columns). Must come
    // first. The extended AIR lays `num_lines = h + w + 2k` rank-entry blocks end to end — in the
    // order `E_A` (`h` lines), `F_A` (`k` lines), `E_B` (`w` lines), `F_B` (`k` lines), mirroring the
    // noise-matmul operand index layout — each block a self-contained `normalize_line` derivation,
    // followed by trailing padding. Exactly the multi-block structure of
    // [`crate::v5::circuit::row_scale_stark`].
    // ------------------------------------------------------------------------------------------
    /// 1 on the trailing padding rows (`num_lines * rank` live rows padded to the trace height).
    pub is_pad: T,
    /// 1 on the first row of each rank-entry block — the live blocks (`row % rank == 0`) and the
    /// first padding row. Gates the per-block prefix-sum reset and the scalar jumps.
    pub is_block_start: T,
    /// 1 on the last row of each LIVE block (`row % rank == rank-1`, live). Pins `TOTAL_SUMSQ` to
    /// that block's finished prefix sum. Padding carries 0.
    pub is_block_final: T,
    /// The global entry index `line * rank + entry` on live rows (= the live trace-row index, since
    /// blocks are contiguous). The E/F-binding CTL key; held at the final live index on padding.
    pub global_index: T,
    /// The noise-matmul reuse multiplicity of this line's entries (`E_A`/`E_B` lines: `k`; `F_A`
    /// lines: `h`; `F_B` lines: `w`). The E/F-binding looked side carries it so each proven entry
    /// balances the matmul operand's per-cell reuse. Held at the final live value on padding (the CTL
    /// filters padding out via `1 - IS_PAD`).
    pub operand_mult: T,
    /// **ZK noise binding (6e-3c).** 1 on the even live rows that open a two-byte XOF limb (`entry`
    /// even and `entry + 1 < rank`): the filter of the egress-pair looked channel
    /// ([`super::ctl::ctl_noise_egress_pair_looked`]) binding this line's XOF bytes to the
    /// seed-derived keyed-BLAKE3 egress ([`crate::v5::circuit::noise_blake3`]). Class (a).
    pub is_egress_pair: T,
    /// **ZK noise binding (6e-3c).** On an `IS_EGRESS_PAIR` row, the egress channel key
    /// `line*16 + entry/2` (= `GLOBAL_INDEX / 2`), matching the noise-BLAKE3 line compression's egress
    /// base `line*16` plus the limb index. 0 elsewhere. Class (a).
    pub egress_key: T,

    // ------------------------------------------------------------------------------------------
    // Per-entry raw byte decode: `x_i = (1 - 2*SIGN_BIT) * MAGNITUDE`,
    // `MAGNITUDE = MAG_MINUS_1 + 1 in [1, 128]`. `BYTE` is a witness input (CTL-bound to the
    // keyed-BLAKE3 XOF in a later stage; decode alone pins `BYTE in [0, 255]`).
    // ------------------------------------------------------------------------------------------
    /// The raw XOF byte.
    pub byte: T,
    /// Bit 7 of the byte (the sign).
    pub sign_bit: T,
    /// `BYTE & 0x7F` (the magnitude minus one), range `[0, 127]` (PAIR128).
    pub mag_minus_1: T,
    /// `(1 - IS_PAD) * (MAG_MINUS_1 + 1)` — the signed magnitude `|x_i|` (0 on pad rows).
    pub magnitude: T,

    // ------------------------------------------------------------------------------------------
    // Line sum of squares and its integer square root.
    // ------------------------------------------------------------------------------------------
    /// Prefix sum of `MAGNITUDE^2` through this row.
    pub running_sumsq: T,
    /// The full line sum of squares `S = sum_i x_i^2` (constant across all rows).
    pub total_sumsq: T,
    /// High 16-bit limb of `TOTAL_SUMSQ` (`S < 2^32`; low limb is the RC16'd remainder).
    pub total_hi: T,

    /// `norm_scaled = isqrt(S * 1024)` — the fixed-point line norm (`INT_SQRT_PREC = 32`).
    pub norm_scaled: T,
    /// High 16-bit limb of `NORM_SCALED` (`q < 2^21`).
    pub norm_scaled_hi: T,
    /// Euclidean remainder `X - q^2` with `X = S*1024`, `q = NORM_SCALED` (`in [0, 2q]`).
    pub isqrt_rem: T,
    /// High 16-bit limb of `ISQRT_REM` (`< 2^22`).
    pub isqrt_rem_hi: T,
    /// Euclidean upper slack `2q - ISQRT_REM >= 0`.
    pub isqrt_s2: T,
    /// High 16-bit limb of `ISQRT_S2`.
    pub isqrt_s2_hi: T,

    // ------------------------------------------------------------------------------------------
    // `denom = f32_to_bf16(norm_scaled)` — RNE of the integer `q` to a normal BF16 (group D).
    // Value `= M_DEN * 2^(DENOM_EXP - 134)`, `M_DEN = 128 + DENOM_MANT`.
    // ------------------------------------------------------------------------------------------
    /// BF16 exponent field of `denom` (`= E*(denom)`, structurally normal).
    pub denom_exp: T,
    /// BF16 mantissa field of `denom`, `in [0, 127]` (PAIR128).
    pub denom_mant: T,
    /// `2^(DENOM_EXP - 134)` (POW2D value; the key range proves `DENOM_EXP in [134, 153]`).
    pub denom_pow: T,
    /// `DENOM_MANT & 1` (ties-to-even parity; boolean).
    pub denom_parity: T,
    /// `(DENOM_MANT - DENOM_PARITY) / 2` (RC16; makes the parity split two-sided).
    pub denom_half: T,
    /// Binade-bottom flag `IS_BOTTOM = [DENOM_MANT == 0]` (value is a power of two): its lower
    /// rounding boundary is `-ulp/4`, not `-ulp/2`. Boolean, pinned exactly by the is-zero
    /// gadget (`denom_mant_inv`). `DENOM_EXP >= 2` is structural here (POW2D domain), so no
    /// separate exponent witness is needed.
    pub denom_bottom: T,
    /// Inverse witness for the `[DENOM_MANT == 0]` is-zero gadget.
    pub denom_mant_inv: T,
    /// Lower quarter-ulp boundary product `(4*M_DEN - 2 + DENOM_BOTTOM) * DENOM_POW`.
    pub denom_blo: T,
    /// Upper quarter-ulp boundary product `(4*M_DEN + 2) * DENOM_POW`.
    pub denom_bhi: T,
    /// High 16-bit limb of the lower bracket slack.
    pub denom_sl_hi: T,
    /// High 16-bit limb of the upper bracket slack.
    pub denom_su_hi: T,

    // ------------------------------------------------------------------------------------------
    // `scale = bf16_div(numer, denom)` — RNE of `NUMER_VAL / denom` to a normal BF16 (group V).
    // `numer = f32_to_bf16(NOISE_TARGET_NORM * 32) = 8192 = 2^13` is a compile-time constant
    // power of two, so the division bracket reduces to a single power-of-two `2^A` comparison
    // against `(4*M_S +/- 2) * M_DEN`, with `A = 283 - DENOM_EXP - SCALE_EXP`.
    // ------------------------------------------------------------------------------------------
    /// BF16 exponent field of `scale` (normal).
    pub scale_exp: T,
    /// BF16 mantissa field of `scale`, `in [0, 127]` (PAIR128); `M_S = 128 + SCALE_MANT`.
    pub scale_mant: T,
    /// `2^A`, `A = 283 - DENOM_EXP - SCALE_EXP` (POW2D value; key range proves `A in [0, 19]`).
    pub div_pow: T,
    /// `SCALE_MANT & 1` (parity; boolean).
    pub scale_parity: T,
    /// `(SCALE_MANT - SCALE_PARITY) / 2` (RC16).
    pub scale_half: T,
    /// Binade-bottom flag `IS_BOTTOM = [SCALE_MANT == 0]` (see `denom_bottom`).
    pub scale_bottom: T,
    /// Inverse witness for the `[SCALE_MANT == 0]` is-zero gadget.
    pub scale_mant_inv: T,
    /// Lower boundary product `(4*M_S - 2 + SCALE_BOTTOM) * M_DEN`.
    pub div_blo: T,
    /// Upper boundary product `(4*M_S + 2) * M_DEN`.
    pub div_bhi: T,
    /// High 16-bit limb of the lower division-bracket slack.
    pub div_sl_hi: T,
    /// High 16-bit limb of the upper division-bracket slack.
    pub div_su_hi: T,

    // ------------------------------------------------------------------------------------------
    // Per-entry `entry_bf16 = bf16_mul(bf16(x_i), scale)` then `entry = f32_to_fp16(entry_bf16)`
    // (group E). The exact pre-rounding product significand is the integer
    // `P = MAGNITUDE * M_S < 2^16`; RNE to 8 bits gives `M_E = 128 + ENTRY_MANT` at shift `gm`.
    // The BF16 -> FP16 cast is an exact widening (entries are always FP16-normal in the scheme;
    // see the envelope note in `stark.rs`). All group-E columns are 0 on pad rows.
    // ------------------------------------------------------------------------------------------
    /// BF16 mantissa field of `entry_bf16` (`M_E = 128 + ENTRY_MANT`); `in [0, 127]` (PAIR128).
    pub entry_mant: T,
    /// Rounding shift `gm = bitlen(P) - 8` (POW2D key; `in [0, 19]`).
    pub entry_gm: T,
    /// `2^gm` (POW2D value).
    pub entry_pow: T,
    /// `ENTRY_MANT & 1` (parity; boolean).
    pub entry_parity: T,
    /// `(ENTRY_MANT - ENTRY_PARITY) / 2` (RC16).
    pub entry_half: T,
    /// Binade-bottom flag `IS_BOTTOM = [ENTRY_MANT == 0]` (see `denom_bottom`); gated off on pad
    /// rows (the is-zero gadget would otherwise force it to 1 on the zero-mantissa pads).
    pub entry_bottom: T,
    /// Inverse witness for the `[ENTRY_MANT == 0]` is-zero gadget (live rows only).
    pub entry_mant_inv: T,
    /// Exact product significand `P = MAGNITUDE * M_S`.
    pub entry_prod: T,
    /// Lower boundary product `(4*M_E - 2 + ENTRY_BOTTOM) * ENTRY_POW`.
    pub entry_blo: T,
    /// Upper boundary product `(4*M_E + 2) * ENTRY_POW`.
    pub entry_bhi: T,
    /// High 16-bit limb of the lower multiply-bracket slack.
    pub entry_sl_hi: T,
    /// High 16-bit limb of the upper multiply-bracket slack.
    pub entry_su_hi: T,
    /// FP16 biased exponent field of the entry (`= ENTRY_GM + SCALE_EXP - 112`, `in [1, 30]`).
    pub entry_fp16_exp: T,
    /// The output FP16 code: `SIGN_BIT*2^15 + ENTRY_FP16_EXP*2^10 + 8*ENTRY_MANT`.
    pub entry_fp16: T,
    /// **ZK noise binding (6e-3c).** On an `IS_EGRESS_PAIR` row, the little-endian 16-bit XOF limb
    /// `BYTE + 2^8 * BYTE'` (this row's byte and the next row's byte), bound by the transition
    /// constraint `IS_EGRESS_PAIR * (BYTE_PAIR - BYTE - 256*BYTE') = 0`. The egress-pair channel
    /// exports it against the noise-BLAKE3 line compression's `cv_egress_limbs[entry/2]`, forcing the
    /// normalized line's raw XOF bytes to equal the seed-derived keyed-XOF — so the noise is no longer
    /// grindable. 0 (free, unconstrained) off egress-pair rows. Witness.
    pub byte_pair: T,
}

/// Total number of committed NoiseStark columns.
pub const NUM_NOISE_COLUMNS: usize = size_of::<NoiseColumnsView<u8>>();

// 7 class (a) columns (incl. the two egress-pair known columns) + 49 original main columns + 1
// BYTE_PAIR witness (the egress-pair binding, 6e-3c).
const _: () = assert!(NUM_NOISE_COLUMNS == 57);

/// The NoiseStark AIR has no public inputs: `rank` enters only through the known `IS_PAD`
/// column and the trace height.
pub const NUM_NOISE_PUBLIC_INPUTS: usize = 0;

columns_view!(NoiseColumnsView, NUM_NOISE_COLUMNS, NOISE_COL_MAP);

/// Number of leading class (a) ("known") columns: `IS_PAD`, `IS_BLOCK_START`, `IS_BLOCK_FINAL`,
/// `GLOBAL_INDEX`, `OPERAND_MULT`, and the egress-pair `IS_EGRESS_PAIR` / `EGRESS_KEY` (6e-3c) — pure
/// functions of the geometry re-checked by the batch verifier against the trace openings.
pub const NUM_NOISE_KNOWN_COLUMNS: usize = NOISE_COL_MAP.egress_key + 1;

#[cfg(test)]
mod tests {
    use core::borrow::Borrow;

    use super::*;

    #[test]
    fn col_map_is_the_identity_layout() {
        let as_array: [usize; NUM_NOISE_COLUMNS] = NOISE_COL_MAP.into();
        for (i, &c) in as_array.iter().enumerate() {
            assert_eq!(c, i);
        }
        // The class (a) columns come first (their indices feed `preprocessed_indices`).
        assert_eq!(NOISE_COL_MAP.is_pad, 0);
        assert_eq!(NOISE_COL_MAP.is_block_start, 1);
        assert_eq!(NOISE_COL_MAP.is_block_final, 2);
        assert_eq!(NOISE_COL_MAP.global_index, 3);
        assert_eq!(NOISE_COL_MAP.operand_mult, 4);
        assert_eq!(NOISE_COL_MAP.is_egress_pair, 5);
        assert_eq!(NOISE_COL_MAP.egress_key, 6);
        assert_eq!(NUM_NOISE_KNOWN_COLUMNS, 7);
        // BYTE_PAIR (the egress-pair limb witness) is the trailing column; ENTRY_FP16 just precedes it.
        assert_eq!(NOISE_COL_MAP.byte_pair, NUM_NOISE_COLUMNS - 1);
        assert_eq!(NOISE_COL_MAP.entry_fp16, NUM_NOISE_COLUMNS - 2);
    }

    #[test]
    fn view_array_roundtrip() {
        let mut arr = [0u64; NUM_NOISE_COLUMNS];
        for (i, v) in arr.iter_mut().enumerate() {
            *v = i as u64 * 3 + 1;
        }
        let view: NoiseColumnsView<u64> = arr.into();
        assert_eq!(view.is_pad, 1);
        assert_eq!(view.norm_scaled, NOISE_COL_MAP.norm_scaled as u64 * 3 + 1);
        let back: [u64; NUM_NOISE_COLUMNS] = view.into();
        assert_eq!(back, arr);

        let borrowed: &NoiseColumnsView<u64> = arr.borrow();
        assert_eq!(borrowed.entry_fp16, arr[NOISE_COL_MAP.entry_fp16]);
    }
}
