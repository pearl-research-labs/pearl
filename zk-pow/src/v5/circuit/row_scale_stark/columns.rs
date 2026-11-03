//! Fixed trace layout for the FP16 per-row scale-derivation AIR.
//!
//! One row per FP16 operand element (one `code`), plus trailing all-zero `IS_PAD = 1` rows that
//! pad the live `k` elements to a power of two. Two kinds of column:
//!
//! * **Per-element** columns prove the running reduction over the row — the sequential f32 sum of
//!   squares `sumsq = Σ (fp16_to_f32(x))^2` (one RNE f32 add per element, mirroring
//!   [`crate::v5::circuit::noisy_quant_fma_stark`]'s windowed fold) and the running maximum
//!   `|x|` (used for `linf`).
//! * **Scalar** columns are a pure function of the whole row (the two norms and the two bf16
//!   scales). They are replicated identically on every row (held constant by transition
//!   constraints and pinned to the final reduction on the last row), so every row re-proves the
//!   shared derivation — exactly the structure of [`crate::v5::circuit::noise_stark`].
//!
//! All ties-to-even roundings are proved with quarter-ulp *bracket* gadgets (the parity split plus
//! the binade-bottom `IS_BOTTOM` correction where the binade is asymmetric), reusing only the
//! FP16-batch committed LUTs (`FP16DECODE`, `RANGE16`, `FP16POW2`, `WIDTH32`) — no new table and no
//! grinding slack. See `stark.rs` for the constraint groups and the documented soundness envelope.
//!
//! [`RowScaleColumnsView`] is `#[repr(C)]`; declaration order is committed column order.

use crate::v4::circuit::columns_view::columns_view;

/// View of one RowScaleStark trace row.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct RowScaleColumnsView<T: Copy> {
    // ==============================================================================================
    // Class (a) ("known") columns: pure functions of the geometry `(num_operand_rows, k)`. Must come
    // first. The extended AIR lays `num_operand_rows` k-element blocks end to end (A rows then B
    // rows), each block a self-contained row-norm + scale derivation, followed by trailing padding.
    // ==============================================================================================
    /// 1 on the trailing padding rows (the `num_operand_rows * k` live rows padded up to the trace
    /// height).
    pub is_pad: T,
    /// The operand-row index this trace row belongs to (`floor(row / k)` on live rows; the scales
    /// channel's key, keeping A rows `0..h` and B rows `h..h+w` in one disjoint space). Held at the
    /// final live block index on padding rows (filtered out of every CTL anyway).
    pub operand_row_index: T,
    /// 1 on the first row of each k-element block — the live blocks (`row % k == 0`) and the first
    /// padding row. Gates the per-block sum-of-squares / running-max reset and the scalar jumps.
    pub is_block_start: T,
    /// 1 on the last row of each LIVE k-element block (`row % k == k-1`, live). Pins `sumsq` /
    /// `max_abs_code` to that block's finished reduction. Padding carries 0 (no scales emitted).
    pub is_block_final: T,

    // ==============================================================================================
    // Per-element decode (FP16DECODE) + running max |x| (group D / M).
    // ==============================================================================================
    /// The raw FP16 operand code `x_i` (witness input; CTL-bound to the operand commitment later).
    pub code: T,
    /// Integer significand of `code` (`0`, `1..=1023` subnormal, `1024..=2047` normal) — FP16DECODE.
    pub x_sig: T,
    /// Sign bit of `code` (bit 15) — FP16DECODE.
    pub x_sign: T,
    /// `stored_exponent + 15 >= 0` — FP16DECODE.
    pub x_eps_biased: T,
    /// `[x_sig == 0]` — FP16DECODE.
    pub x_is_zero: T,
    /// `code & 0x7FFF` (the magnitude code; value-ordered on nonnegative FP16).
    pub abs_code: T,
    /// Running max of `ABS_CODE` through this row (= max magnitude code so far).
    pub run_max: T,
    /// `[ABS_CODE >= prev RUN_MAX]` (boolean; the max compare-select).
    pub max_ge: T,
    /// Two-sided slack of the max compare (`MAX_GE ? ABS_CODE-prev : prev-ABS_CODE-1`).
    pub max_slack: T,

    // ==============================================================================================
    // Per-element square `sq = x_i^2` normalized to a 24-bit significand (group Q).
    // ==============================================================================================
    /// Exact product significand `SQ_SIG = X_SIG^2` (`< 2^22`).
    pub sq_sig: T,
    /// Bit-width of `SQ_SIG` (`0` when zero; WIDTH32 key on nonzero-square rows).
    pub sq_w: T,
    /// `2^(24 - SQ_W)` (WIDTH32 value; lifts `SQ_SIG` to a 24-bit significand).
    pub sq_lift: T,
    /// `SQ_NORM = SQ_SIG * SQ_LIFT in [2^23, 2^24)` (0 on zero-square rows).
    pub sq_norm: T,
    pub sq_norm_lo: T,
    pub sq_norm_hi: T,

    // ==============================================================================================
    // Accumulation fold `acc_out = RNE_f32(acc_in + sq)` (both nonnegative) (group A).
    // `acc_in`/`acc_out` carry an f32 as `(SIG in [2^23,2^24) or 0, EXP biased, ZERO)`.
    // ==============================================================================================
    /// Running sum significand BEFORE adding this element (24-bit or 0).
    pub acc_in_sig: T,
    pub acc_in_sig_lo: T,
    pub acc_in_sig_hi: T,
    /// Biased f32 exponent of `acc_in` (0 when zero).
    pub acc_in_exp: T,
    /// `[acc_in == 0]`.
    pub acc_in_zero: T,

    /// Biased value-MSB of the `sq` term (0 when `sq` is zero).
    pub sq_msb: T,
    /// Biased value-MSB of the `acc_in` term (0 when zero).
    pub acc_msb: T,
    /// Window anchor `eta = max(sq_msb, acc_msb)` over nonzero terms.
    pub eta: T,
    /// `eta - sq_msb` (0 if `sq` zero).
    pub rel_sq: T,
    /// `eta - acc_msb` (0 if `acc_in` zero).
    pub rel_acc: T,
    pub far_sq: T,
    pub far_sq_slack: T,
    pub far_acc: T,
    pub far_acc_slack: T,
    pub active_sq: T,
    pub active_acc: T,
    /// `2^rel_sq` (FP16POW2) on active rows.
    pub pow_sq: T,
    pub aligned_sq: T,
    pub aligned_sq_lo: T,
    pub aligned_sq_hi: T,
    pub rem_sq: T,
    pub rem_sq_lo: T,
    pub rem_sq_hi: T,
    pub rem_sq_bound: T,
    pub rem_sq_bound_lo: T,
    pub rem_sq_bound_hi: T,
    /// `2^rel_acc` (FP16POW2) on active rows.
    pub pow_acc: T,
    pub aligned_acc: T,
    pub aligned_acc_lo: T,
    pub aligned_acc_hi: T,
    pub rem_acc: T,
    pub rem_acc_lo: T,
    pub rem_acc_hi: T,
    pub rem_acc_bound: T,
    pub rem_acc_bound_lo: T,
    pub rem_acc_bound_hi: T,
    pub rem_sq_nz: T,
    pub rem_sq_nz_inv: T,
    pub rem_acc_nz: T,
    pub rem_acc_nz_inv: T,
    pub or_sq: T,
    pub or_acc: T,
    pub sticky_sq: T,
    pub sticky_acc: T,
    pub far_sticky: T,
    /// Nonnegative window sum `|W| = aligned_sq + aligned_acc` (`< 2^29`).
    pub w_abs: T,
    pub w_abs_lo: T,
    pub w_abs_hi: T,
    pub w_is_zero: T,
    pub w_abs_inv: T,
    /// Bit-width of `|W|` (WIDTH32 key).
    pub ww: T,
    pub trunc_w: T,
    pub lift_w: T,
    pub m_rz: T,
    pub m_rz_lo: T,
    pub m_rz_hi: T,
    pub rz_rem: T,
    pub rz_rem_bound: T,
    pub rz_parity: T,
    pub rz_half: T,
    pub gt: T,
    pub gt_slack: T,
    pub eq: T,
    pub eq_inv: T,
    pub or_rs: T,
    pub round_up: T,
    pub carry: T,
    pub carry_inv: T,
    /// `acc_out` significand (24-bit or 0) — next row's `acc_in`.
    pub acc_out_sig: T,
    pub acc_out_sig_lo: T,
    pub acc_out_sig_hi: T,
    pub acc_out_exp: T,
    pub acc_out_zero: T,

    // ==============================================================================================
    // ENTRY-LIVENESS gate (whitepaper "Shared checks"). Per element, `dead = [|x_iu| >= 4*l2_i]`;
    // the per-side dead counts are then gated `64 * dead_side <= rows_side * k` (eps_idle = 1/64).
    // These are PER-ELEMENT / per-row (they vary within a block), so they live BEFORE the scalar
    // region (the scalar-constancy transition pins everything from `sumsq_sig` on). All class (b)
    // (witness) -- NO new known column, so the wrapper PI layout and the native header-bound
    // verifier are untouched. See `stark.rs` group L for the constraints.
    // ==============================================================================================
    /// 1 iff this element is dead: `|x| >= 4*l2` (the block's floored `l2`). `0` on padding/zero `x`.
    pub dead: T,
    /// 1 iff `s = (x_eps_biased - 25) - (l2_e - 132) >= 0` (the aligned-compare direction). Pins
    /// `dead_key = |s|` via `dead_key = s * (2*dead_sign - 1)` (FP16POW2 bounds `dead_key` to its
    /// nonneg key domain, which pins the sign).
    pub dead_sign: T,
    /// `|s|` -- the FP16POW2 alignment key (shift between `|x|`'s and `4*l2`'s integer significands).
    pub dead_key: T,
    /// `2^min(dead_key, 26)` (FP16POW2 value). Saturation at 2^26 is sound: beyond the active
    /// window the larger side already dominates, so the comparison outcome never flips.
    pub dead_pow: T,
    /// Alignment power applied to the `|x|` side: `dead_sign ? dead_pow : 1` (the `4*l2` side gets
    /// `dead_pow + 1 - dead_pow_a`).
    pub dead_pow_a: T,
    /// `x_sig * dead_pow_a` (the `|x|` side of the aligned integer compare).
    pub dead_lhs: T,
    /// `(128 + l2_m) * dead_pow_b` (the `4*l2` side).
    pub dead_rhs: T,
    /// Nonnegative two-sided slack `dead ? (lhs - rhs) : (rhs - lhs - 1)`, `< 2^37`, as 16/16/5-bit
    /// limbs (RANGE16): `dead_slack = lo + 2^16*mid + 2^32*hi`.
    pub dead_slack_lo: T,
    pub dead_slack_mid: T,
    pub dead_slack_hi: T,
    /// 1 iff this trace row belongs to the B operand (`operand_row_index >= num_a_rows`). Masks the
    /// per-side dead accumulators. Pinned two-sided by `b_side_slack` (RANGE16).
    pub is_b_side: T,
    pub b_side_slack: T,
    /// Inclusive running count of dead A-side elements (`sum over live A rows of dead`); frozen on
    /// B/padding rows, so it holds the A-side total on the last row.
    pub dead_run_a: T,
    /// Inclusive running count of dead B-side elements; holds the B-side total on the last row.
    pub dead_run_b: T,
    /// Low/high limbs of the A gate slack `a_gate = num_a_rows*k - 64*dead_run_a >= 0` (last row).
    pub a_gate_slack_lo: T,
    pub a_gate_slack_hi: T,
    /// Low/high limbs of the B gate slack `b_gate = num_b_rows*k - 64*dead_run_b >= 0` (last row).
    pub b_gate_slack_lo: T,
    pub b_gate_slack_hi: T,

    // ==============================================================================================
    // SCALAR columns (replicated on every row; held constant; pinned to the last row).
    // ==============================================================================================
    /// `sumsq` significand (= `acc_out` of the last live row).
    pub sumsq_sig: T,
    pub sumsq_sig_lo: T,
    pub sumsq_sig_hi: T,
    pub sumsq_exp: T,
    pub sumsq_zero: T,
    /// Max magnitude code (= `run_max` of the last live row).
    pub max_abs_code: T,

    // ---- q = RNE_f32(sumsq / k) (group V) ----
    pub q_sig: T,
    pub q_sig_lo: T,
    pub q_sig_hi: T,
    pub q_exp: T,
    pub q_zero: T,
    pub q_bottom: T,
    pub q_bottom_inv: T,
    pub q_parity: T,
    pub q_half: T,
    /// `2^Dq` alignment power for the division bracket (FP16POW2).
    pub q_pow: T,
    pub q_dexp: T,
    pub q_blo: T,
    pub q_bhi: T,
    pub q_sl_lo: T,
    pub q_sl_mid: T,
    pub q_sl_hi: T,
    pub q_su_lo: T,
    pub q_su_mid: T,
    pub q_su_hi: T,

    // ---- s = RNE_f32(sqrt(q)) (group S) ----
    pub s_sig: T,
    pub s_sig_lo: T,
    pub s_sig_hi: T,
    pub s_exp: T,
    pub s_bottom: T,
    pub s_bottom_inv: T,
    pub s_parity: T,
    pub s_half: T,
    /// `2^Ds` alignment power for the sqrt bracket (FP16POW2).
    pub s_pow: T,
    pub s_dexp: T,
    /// Squared lower/upper quarter-ulp boundaries (`~ 2^52`).
    pub s_blo: T,
    pub s_bhi: T,
    pub s_sl_lo: T,
    pub s_sl_mid: T,
    pub s_sl_hi: T,
    pub s_su_lo: T,
    pub s_su_mid: T,
    pub s_su_hi: T,

    // ---- l2raw = f32_to_bf16(s) (round 24-bit -> 8-bit, shift 16) (group C) ----
    pub l2raw_mant: T,
    pub l2raw_exp: T,
    pub l2raw_carry: T,
    pub l2raw_bottom: T,
    pub l2raw_bottom_inv: T,
    pub l2raw_parity: T,
    pub l2raw_half: T,
    pub l2raw_blo: T,
    pub l2raw_bhi: T,
    pub l2raw_sl_lo: T,
    pub l2raw_sl_hi: T,
    pub l2raw_su_lo: T,
    pub l2raw_su_hi: T,
    /// `l2raw` bf16 code (`l2raw_exp<<7 | l2raw_mant`); `0` iff `sumsq == 0`.
    pub l2raw_code: T,

    // ---- l2grid = round_l2_to_grid(l2raw) (clear low 2 mantissa bits, ties up) (group G) ----
    /// `(l2raw_code + 2) >> 2` (the kept high bits).
    pub grid_q: T,
    /// `l2raw_code + 2 - 4*grid_q in {0,1,2,3}` (the cleared low bits + half).
    pub grid_r: T,
    pub grid_r_b0: T,
    pub grid_r_b1: T,
    /// `l2grid` bf16 code (`4 * grid_q`).
    pub l2grid_code: T,

    // ---- l2 = bf16_max(l2grid, floor) (group F) ----
    pub l2_ge: T,
    pub l2_floor_slack: T,
    pub l2_code: T,
    /// Exponent/mantissa fields of the floored `l2_code` (`l2_code = 128*l2_e + l2_m`).
    pub l2_e: T,
    pub l2_m: T,

    // ==============================================================================================
    // linf = bf16_max(f32_to_bf16(max|x|), floor).
    // ==============================================================================================
    /// FP16DECODE of `MAX_ABS_CODE` (`max_sign` is always 0 — the magnitude code).
    pub max_sig: T,
    pub max_sign: T,
    pub max_eps_biased: T,
    pub max_is_zero: T,
    /// Bit-width of `MAX_SIG` (WIDTH32 key; nonzero-max rows).
    pub max_w: T,
    pub max_lift: T,
    /// `MAX_SIG * MAX_LIFT in [2^23, 2^24)` (24-bit significand of `max|x|`).
    pub max_norm: T,
    pub max_norm_lo: T,
    pub max_norm_hi: T,
    /// Round of `max_norm` (24-bit) to the 8-bit bf16 significand (shift 16).
    pub linf_raw_mant: T,
    pub linf_raw_exp: T,
    pub linf_raw_carry: T,
    pub linf_raw_bottom: T,
    pub linf_raw_bottom_inv: T,
    pub linf_raw_parity: T,
    pub linf_raw_half: T,
    pub linf_raw_blo: T,
    pub linf_raw_bhi: T,
    pub linf_raw_sl_lo: T,
    pub linf_raw_sl_hi: T,
    pub linf_raw_su_lo: T,
    pub linf_raw_su_hi: T,
    /// `linf_raw` bf16 code (normal when `max|x| != 0`).
    pub linf_raw_code: T,
    pub linf_ge: T,
    pub linf_floor_slack: T,
    pub linf_code: T,
    /// Exponent/mantissa fields of the floored `linf_code` (`linf_code = 128*linf_e + linf_m`).
    pub linf_e: T,
    pub linf_m: T,

    // ==============================================================================================
    // noised_bound = bf16_fma(dr, l2, linf) (dr a compile-time constant; all operands nonneg) (H1).
    // Because `linf >= l2 ~ dr*l2` (max >= rms), the product `dr*l2` is the finer term: its LSB sits
    // `nb_shift` bits below `linf`'s, so `W = dr.M*l2.M + linf.M * 2^nb_shift` is the EXACT sum of
    // the two operands (no lost bits, no sticky). `W` is then rounded once to the 8-bit significand.
    // ==============================================================================================
    /// Exact product significand `dr.M * l2.M` (`< 2^16`).
    pub nb_prod: T,
    /// `nb_shift = E(linf) - E(l2) + 6 >= 0` (FP16POW2 key); aligns `linf` onto the product grid.
    pub nb_shift: T,
    /// `2^nb_shift` (FP16POW2 value).
    pub nb_shiftpow: T,
    /// Exact window sum `W = nb_prod + linf.M * nb_shiftpow`.
    pub nb_w: T,
    pub nb_w_lo: T,
    pub nb_w_hi: T,
    /// Rounding shift `nb_gm = bitlen(W) - 8` (FP16POW2 key; pinned by `nb_mant in [0,127]`).
    pub nb_gm: T,
    pub nb_pow: T,
    pub nb_mant: T,
    pub nb_exp: T,
    pub nb_bottom: T,
    pub nb_bottom_inv: T,
    pub nb_parity: T,
    pub nb_half: T,
    pub nb_blo: T,
    pub nb_bhi: T,
    pub nb_sl_lo: T,
    pub nb_sl_hi: T,
    pub nb_su_lo: T,
    pub nb_su_hi: T,
    /// `noised_bound` bf16 code.
    pub nb_code: T,

    // ==============================================================================================
    // alpha = bf16_div(MAX_FP16_bf16, noised_bound); MAX_FP16_bf16 = 2^16 (power of two) (H3).
    // Single power-of-two division bracket (mirrors noise_stark's `scale`).
    // ==============================================================================================
    pub alpha_exp: T,
    pub alpha_mant: T,
    pub alpha_bottom: T,
    pub alpha_bottom_inv: T,
    pub alpha_parity: T,
    pub alpha_half: T,
    /// `2^A` division power (FP16POW2); `A = DIV_KEY_BASE - nb_exp - alpha_exp`.
    pub alpha_pow: T,
    pub alpha_blo: T,
    pub alpha_bhi: T,
    pub alpha_sl: T,
    pub alpha_su: T,
    pub alpha_code: T,

    // ==============================================================================================
    // m1 = bf16_mul(alpha, l2); beta = bf16_mul(m1, dos) (H4/H5). Two bf16 RNE multiplies.
    // ==============================================================================================
    pub m1_prod: T,
    pub m1_mant: T,
    pub m1_exp: T,
    pub m1_bottom: T,
    pub m1_bottom_inv: T,
    pub m1_parity: T,
    pub m1_half: T,
    pub m1_pow: T,
    pub m1_gm: T,
    pub m1_blo: T,
    pub m1_bhi: T,
    pub m1_sl: T,
    pub m1_su: T,
    pub m1_code: T,

    pub beta_prod: T,
    pub beta_mant: T,
    pub beta_exp: T,
    pub beta_bottom: T,
    pub beta_bottom_inv: T,
    pub beta_parity: T,
    pub beta_half: T,
    pub beta_pow: T,
    pub beta_gm: T,
    pub beta_blo: T,
    pub beta_bhi: T,
    pub beta_sl: T,
    pub beta_su: T,
    pub beta_code: T,
}

/// Total number of committed RowScaleStark columns.
pub const NUM_ROW_SCALE_COLUMNS: usize = size_of::<RowScaleColumnsView<u8>>();

/// RowScaleStark has no public inputs: the geometry enters through the known columns (`IS_PAD`,
/// `OPERAND_ROW_INDEX`, `IS_BLOCK_START`, `IS_BLOCK_FINAL`), the trace height, and the compile-time
/// constants baked into the AIR.
pub const NUM_ROW_SCALE_PUBLIC_INPUTS: usize = 0;

columns_view!(RowScaleColumnsView, NUM_ROW_SCALE_COLUMNS, ROW_SCALE_COL_MAP);

/// Number of leading class (a) ("known") columns: `IS_PAD`, `OPERAND_ROW_INDEX`, `IS_BLOCK_START`,
/// `IS_BLOCK_FINAL`.
pub const NUM_ROW_SCALE_KNOWN_COLUMNS: usize = ROW_SCALE_COL_MAP.is_block_final + 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn col_map_is_the_identity_layout() {
        let as_array: [usize; NUM_ROW_SCALE_COLUMNS] = ROW_SCALE_COL_MAP.into();
        for (i, &c) in as_array.iter().enumerate() {
            assert_eq!(c, i);
        }
        assert_eq!(ROW_SCALE_COL_MAP.is_pad, 0);
        assert_eq!(ROW_SCALE_COL_MAP.operand_row_index, 1);
        assert_eq!(ROW_SCALE_COL_MAP.is_block_start, 2);
        assert_eq!(ROW_SCALE_COL_MAP.is_block_final, 3);
        assert_eq!(NUM_ROW_SCALE_KNOWN_COLUMNS, 4);
        assert_eq!(ROW_SCALE_COL_MAP.beta_code, NUM_ROW_SCALE_COLUMNS - 1);
        // The entry-liveness columns are per-element: they must precede the scalar region
        // (`sumsq_sig`..), which the scalar-constancy transition pins constant within a block.
        assert!(ROW_SCALE_COL_MAP.dead < ROW_SCALE_COL_MAP.sumsq_sig);
        assert!(ROW_SCALE_COL_MAP.b_gate_slack_hi < ROW_SCALE_COL_MAP.sumsq_sig);
    }
}
