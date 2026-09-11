//! Column layout for Scale's norms, scale chain and policy gates; see [`super::stark`].
//!
//! The square-root certificate aligns squared rounding boundaries with InputQuant's
//! framed sum using eight base-`2^16` limb positions. Range checks on these limbs
//! make the comparisons hold over the integers.
//!
//! A zero claim needs only the upper boundary. At a power of two above the minimum
//! normal value, the lower boundary is one quarter of the current binade's spacing
//! away. [`super::stark`] derives these boundaries and defines the constraint labels.

use crate::circuit::fp8::columns_view::columns_view;

/// 16-bit limbs of the integer L2 frame sum `S < 2^62` (Q1).
pub const L2_SUM_LIMBS: usize = 4;
/// 16-bit limbs of the shifted claim-side midpoint products `B^2 * k * 2^15 < 2^53` (their limb
/// 0 is provably zero and not committed — `k` is a multiple of 32, so `2^20 | B^2*k*2^15`).
pub const CLAIM_LIMBS: usize = 3;
/// Claim-side limb positions `1..=7`; position 0 is always zero.
/// Borrow chains compare all eight positions. Q5 multiplies `SS` by `2^32`,
/// placing its five limbs at positions `2..=6`.
pub const ALIGNED_LIMBS: usize = 7;
/// Borrow bits per chain (positions `0..=6`; position 7 admits no borrow-out).
pub const BORROW_BITS: usize = 7;

/// One shared bf16 multiply gadget instance (the M1-M6 constraint pattern, columns only):
/// `out = RNE_bf16(a * b)` with the exact significand product, the CLAMP22 subnormal cut and the
/// shared RNERND rounding back-end. Instantiated for `beta_1 = alpha * l2f` (H4) and
/// `beta = beta_1 * dos` (H5).
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct MulBlock<T: Copy> {
    /// M1: the exact significand product `M(a) * M(b) < 2^16`.
    pub sig_product: T,
    /// M2: RNERND slot `clamp(135 - E*(a) - E*(b), -7, 18) + 7` — the signed subnormal fade
    /// cut biased by +7 — pinned by CLAMP22 (key `535 - E*(a) - E*(b)`, i.e. the signed fade
    /// argument + 400).
    pub cut_depth: T,
    /// M4: output exponent field (0 on subnormal/zero outputs).
    pub out_exp: T,
    /// M3: output mantissa field, RNERND-bound (implicit 128 stripped on normal outputs).
    pub out_mantissa: T,
    /// M3/M4: RNERND's `final pos + 134` (`= width(V) + 126 + round_carry` on width-dominated
    /// rounding); `OUT_EXP = E*(a) + E*(b) - 268 + WIDTH_ADJUST` on normal outputs.
    pub width_adjust: T,
    /// M3/M5: RNERND's "output is +-0" flag.
    pub out_is_zero: T,
    /// M3/M5: RNERND's "output exponent field is 0" (subnormal-or-zero) flag.
    pub out_exp_is_zero: T,
}

/// Nonnegative FMA `RNE_bf16(dr*l2f + linf_f)` (H1, W1–W12).
/// The product's LSB exponent is EP = E*(dr) + E*(l2f) - 268; the addend's is
/// EC = E*(linf_f) - 134. Positive operands need neither a sign nor a zero-sign check.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct FmaBlock<T: Copy> {
    /// W1: the exact product significand `M(dr) * M(l2f) < 2^16`.
    pub sig_product: T,
    /// W2: `GE = 1 <=> EP >= EC` — which operand's *unit* is coarser (not which is bigger).
    pub product_scale_ge_addend: T,
    /// W2: two-sided order witness `GE*(EP - EC) + (1 - GE)*(EC - EP - 1)` (range-checked to 16 bits).
    pub scale_gap_slack: T,
    /// W3: far-gap flag (`d >= THR`, `THR = 18 - 8*GE`): beyond the threshold the finer
    /// operand cannot affect more than the sticky bit, so the alignment gap is capped. (The
    /// pure-addition fold has no borrow case, hence a lower threshold than InputQuant's
    /// subtracting FMA.)
    pub is_far_gap: T,
    /// W3: far-gap witness `d - THR` on far rows (range-checked to 16 bits; 0 on near rows).
    pub far_gap_slack: T,
    /// W4: the gap actually used by the fold: `d` on near rows, `THR - 1` on far rows; POW2D key
    /// (domain [0, 17] — its range proof).
    pub exp_gap_capped: T,
    /// W4: `2^EXP_GAP_CAPPED` (POW2D value).
    pub exp_gap_pow2: T,
    /// W5: the product magnitude in the common finer unit (committed for the degree budget).
    pub aligned_product: T,
    /// W5: the addend magnitude in the common finer unit.
    pub aligned_addend: T,
    /// W6: the exact fold `V = AL_P + AL_C < 2^34` (nonnegative variant: no sign, no zero pair).
    pub folded_magnitude: T,
    /// W8: wide flag — `V >= 2^17` overflows RNERND's key domain and is first compressed
    /// round-to-odd to 17 bits.
    pub is_wide: T,
    /// W8: round-to-odd shift `width(V) - 17` on wide rows (0 on narrow rows); POW2D key.
    pub compression_shift: T,
    /// W8: `2^SHIFT` (POW2D value; 1 on narrow rows).
    pub shift_pow2: T,
    /// W8: the compressed quotient `K = floor(V / 2^SHIFT) in [2^16, 2^17)` on wide rows; the
    /// sandwich (RC16 floor + `ROUNDING_SIGNIFICAND_KEY < 2^17`) forces `SHIFT = width(V) - 17` exactly.
    pub compression_quotient: T,
    /// W9: parity bit of `K` (bound two-sidedly by the `(K - K0)/2` RC16).
    pub compression_quotient_lsb: T,
    /// W8: round-to-odd remainder `R = V - K*2^SHIFT` (range-checked to 16 bits, and `< 2^SHIFT` by the
    /// `SHIFT_POW2 - 1 - R` RC16).
    pub compression_remainder: T,
    /// W9: inverse witness making `STICKY` two-sided:
    /// `COMPRESSION_REMAINDER * COMPRESSION_REMAINDER_INVERSE = STICKY`.
    pub compression_remainder_inverse: T,
    /// W9: `R != 0` — the discarded low bits' nonzeroness.
    pub sticky: T,
    /// W10: the RNERND significand key: `V` on narrow rows, `K + STICKY*(1 - K0)` (round-to-odd)
    /// on wide rows.
    pub rounding_significand_key: T,
    /// High bit of `rounding_significand_key`; together with its RC16 split, proves the key is
    /// below `2^17` and cannot alias into another RNERND cut slot.
    pub rounding_significand_key_high_bit: T,
    /// W10: the unit of `ROUNDING_SIGNIFICAND_KEY`'s LSB: `min(EP, EC) + far-gap restoration + SHIFT` (committed —
    /// CLAMP22's key must stay affine).
    pub key_scale: T,
    /// W11: RNERND slot `clamp(-133 - KEY_SCALE, -7, 18) + 7` — the signed subnormal fade cut
    /// biased by +7 — pinned by CLAMP22 (key `267 - KEY_SCALE`).
    pub cut_used: T,
    /// W12: output exponent field (`KEY_SCALE + WIDTH_ADJUST` on normal outputs, else 0).
    pub out_exp: T,
    /// W11: output mantissa field (RNERND-bound).
    pub out_mantissa: T,
    /// W11: RNERND's `final pos + 134` (see [`MulBlock::width_adjust`]).
    pub width_adjust: T,
    /// W11: RNERND zero flag.
    pub out_is_zero: T,
    /// W11: RNERND subnormal-or-zero flag.
    pub out_exp_is_zero: T,
}

/// View of one ScaleStark trace row (deviations are flagged here and derived in `stark.rs`).
///
/// Field conventions: a bf16 magnitude with fields `(EXP, MANTISSA)` and subnormal flag
/// `EXP_IS_ZERO = [EXP = 0]` has significand `M = 128*(1 - EXP_IS_ZERO) + MANTISSA`, effective
/// exponent `E* = EXP + EXP_IS_ZERO in [1, 254]`, and value `M * 2^(E* - 134)`.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct ScaleColumnsView<T: Copy> {
    // Verifier-known schedule columns; see `known_values`.
    /// The CTL key of this row's group tuple: `(g + 1)*k - 1` for A row `g`, `h*k + (c + 1)*k - 1`
    /// for B row `c` — InputQuantStark's `IS_GROUP_FINAL` element indices.
    pub group_key: T,
    /// 1 on the last row only: the filter of the T3 DEAD_LIMIT gate lookups.
    pub is_last_row: T,
    /// 1 on the `h` A-side rows, 0 on the `w` B-side rows (the A/B interleave of the group
    /// tuples).
    pub is_a_row: T,
    /// 1 on the trailing padding rows (`h + w` live rows padded to a power of two). Pad rows
    /// carry the all-zero tuple (a valid zero-sum row fill) and are excluded from the
    /// group-tuple CTL.
    pub is_pad: T,

    // The received group tuple (main; every field CTL-bound to InputQuantStark).
    /// The exact block-integer L2 frame sum
    /// `S = sum_b floor(p_b * 2^Wl2 / 2^(FRAME_DOUBLED_SCALE_EXPONENT - 2*E*(scale_b))) <
    /// 2^62`, `p_b = M(scale_b)^2 * n_b`, `n_b` the block's int8 square sum
    /// (InputQuantStark's group B; `Wl2 = 27 - ceil(log2 k)` is a per-job headroom constant).
    pub l2_frame_sum: T,
    /// The row frame exponent `E_MAX = max over live blocks of 2*E*(scale_b)` (0 if none) —
    /// already doubled/even; the L2 alignment frame (a *scale-plane* frame, not the decoded
    /// max element's exponent).
    pub frame_doubled_scale_exponent: T,
    /// bf16 code of the group's max `|X|` (sign cleared) — the linf code (group N).
    pub max_abs: T,
    /// Alpha claim, exponent field (alpha is structurally normal; group H3 rederives it).
    pub alpha_exp: T,
    /// Alpha claim, mantissa field.
    pub alpha_mantissa: T,
    /// Beta claim, exponent field (beta may be subnormal or zero).
    pub beta_exp: T,
    /// Beta claim, mantissa field.
    pub beta_mantissa: T,
    /// Beta claim, subnormal flag.
    pub beta_exp_is_zero: T,
    /// Dead-entry count imported from InputQuant. The tuple also binds the threshold
    /// to `code(l2f) + 256`, the BF16 encoding of `4*l2f`.
    pub dead_count: T,

    // Liveness totals (T): per-side dead counts, checked against public budgets on the last row.
    /// T2: prefix sum of `IS_A_ROW * DEAD_COUNT`; the last row holds the A side's dead total
    /// (frozen through pad rows), gated by the T3 RC16
    /// `DEAD_LIMIT_A - RUNNING_DEAD_A - 2^16·DEAD_SLACK_HI_A`.
    pub running_dead_a: T,
    /// T2: prefix sum of `(1 - IS_A_ROW)*(1 - IS_PAD) * DEAD_COUNT` (the B side's dead total).
    pub running_dead_b: T,
    /// T3 witness: the high bit of the A side's last-row slack
    /// `DEAD_LIMIT_A - RUNNING_DEAD_A` (boolean, T3a). An over-limit total wraps the slack
    /// far above `2^17`, where no boolean high bit can bring it back into RC16's window.
    pub dead_slack_hi_a: T,
    /// T3 witness for the B side (see `dead_slack_hi_a`).
    pub dead_slack_hi_b: T,

    // Noise floor (F): alpha*l2f >= 2. With exact significand product P and
    // exponent sum E, this is E >= 255, or E = 254 and P >= 2^15.
    // See `stark.rs` for the derivation and `ctl.rs` for the branch range checks.
    /// Selects the E >= 255 branch; a cleared bit requires E = 254 and P >= 2^15.
    pub sigma_exp_clears_floor: T,

    // Sqrt claim and squared rounding interval (Q); see `stark.rs` for the hat frame.
    // The certificate compares squared quarter-ulp boundaries against the framed mean
    // square. Odd significands require strict inequalities to implement ties-to-even.
    /// Claim exponent field (EXPINFO-bound).
    pub sqrt_exp: T,
    /// Claim mantissa field (7-bit, PAIR128-bound).
    pub sqrt_mantissa: T,
    /// Claim subnormal flag (EXPINFO value).
    pub sqrt_exp_is_zero: T,
    /// Two-sided `S = 0` flag (Q2): zero sums are forced to the +0 claim and gate the bracket
    /// off; booleanness is derived from the inverse pair.
    pub frame_sum_is_zero: T,
    /// Inverse witness of `S` on nonzero rows (`S * FRAME_SUM_INVERSE = 1 - FRAME_SUM_IS_ZERO`).
    pub frame_sum_inverse: T,
    /// Q2b: claim that the square root rounds to zero. Requires `t = 0` and disables
    /// the lower boundary check. On a nonzero sum, this permits `sqrt(v_hat) <= 2^-7`.
    /// Leaving the flag off for a zero claim is safe: it retains the stricter lower
    /// boundary `B_LO^2 = 4` and cannot admit an invalid square root.
    pub sqrt_claim_is_zero: T,
    /// Q2b: the committed lower-arm gate
    /// `(1 - FRAME_SUM_IS_ZERO) * (1 - SQRT_CLAIM_IS_ZERO)` (committed to keep the Q8
    /// aligned-claim mux at degree <= 3).
    pub lower_bracket_is_active: T,
    /// Claim mantissa parity (`t` odd <=> both bracket inequalities strict — ties-to-even).
    pub sqrt_mantissa_parity: T,
    /// `(SQRT_MANTISSA - SQRT_MANTISSA_PARITY) / 2`, range-checked by PAIR128.
    /// The bound makes the parity split exact over the integers; without it, either
    /// parity bit could satisfy the equation by field division.
    pub sqrt_mantissa_half: T,

    // Binade-bottom lower-midpoint correction (quarter-ulp soundness fix; see module docs).
    /// 1 iff the claim opens a binade above the subnormal boundary (`t = 128`, `EXP >= 2`):
    /// its lower rounding boundary is `y - ulp/4`, not `y - ulp/2`.
    pub sqrt_is_binade_bottom: T,
    /// `SQRT_MANTISSA = 0` flag (with inverse witness).
    pub sqrt_mantissa_is_zero: T,
    /// Inverse witness of `SQRT_MANTISSA`.
    pub sqrt_mantissa_inverse: T,
    /// `SQRT_EXP >= 2` flag (one-sided: `EXP_GE2 = 0` forces `EXP in {0, 1}`).
    pub sqrt_exponent_at_least_two: T,

    /// Q1: 16-bit limbs of `S` (range-checked; top limb capped `< 2^14` by `RC16(4 * limb_3)`).
    pub frame_sum_limbs: [T; L2_SUM_LIMBS],
    /// Q7: 16-bit limbs of the squared lower rounding boundary `B_LO^2 < 2^20`,
    /// `B_LO = 4t - 2 + SQRT_IS_BINADE_BOTTOM` (quarter-ulp scale). The upper boundary's
    /// square is the affine `B_LO^2 + 32t - 1021*SQRT_IS_BINADE_BOTTOM`.
    pub lower_boundary_squared_limbs: [T; 2],

    // Q5: let f = E*(sqrt claim) and F = frame_doubled_scale_exponent.
    //
    //   G      = 2*f + Wl2 - 4 - F
    //   G + 32 = 16*q + 15 - r2.
    //
    // One-hot q in [0, 4] and r2 in [0, 15] cover G in [-32, 47].
    // The +32 bias is balanced by placing the shifted sum two limbs higher in Q8.
    /// One-hot: `q = 1`.
    pub alignment_quotient_is_1: T,
    /// One-hot: `q = 2`.
    pub alignment_quotient_is_2: T,
    /// One-hot: `q = 3`.
    pub alignment_quotient_is_3: T,
    /// One-hot: `q = 4`.
    pub alignment_quotient_is_4: T,
    /// `r2 = 16q + 15 - (G + 32)`, the up-shift applied to `S` (POW2D key; `<= 15` by
    /// `RC16(15 - r2)`).
    pub sum_shift_remainder: T,
    /// `2^r2` (POW2D value).
    pub sum_shift_power: T,

    /// Q6: low four 16-bit limbs of `S * 2^r2 < 2^77` (range-checked); the top limb is the final carry.
    pub shifted_sum_limbs: [T; L2_SUM_LIMBS],
    /// Q6: per-limb carries of the `S * 2^r2` schoolbook product (range-checked).
    pub shifted_sum_carries: [T; L2_SUM_LIMBS],

    /// Q7: 16-bit limbs 1..=3 of `B_LO^2 * k * 2^15 < 2^53` (limb 0 provably zero; range-checked, top
    /// limb capped `< 2^8`).
    pub lower_boundary_product_limbs: [T; CLAIM_LIMBS],
    /// Q7: 16-bit limbs 1..=3 of `B_HI^2 * k * 2^15`, `B_HI^2 = B_LO^2 + 32t - 1021*b`.
    pub upper_boundary_product_limbs: [T; CLAIM_LIMBS],
    /// Q8: the lower boundary's limbs aligned at limb offset `q` (positions 1..=7), gated to 0
    /// by `LOWER_BRACKET_IS_ACTIVE` (zero-sum rows *and* live zero claims); bound by the
    /// one-hot mux, so each is a range-checked limb or 0.
    pub aligned_lower_boundary_limbs: [T; ALIGNED_LIMBS],
    /// Q8: the upper boundary's aligned limbs (positions 1..=7), gated by
    /// `1 - FRAME_SUM_IS_ZERO`.
    pub aligned_upper_boundary_limbs: [T; ALIGNED_LIMBS],
    /// Q8: borrow bits of the multi-limb comparison `ACL + ODD <= SS * 2^32`, where
    /// `ODD = SQRT_MANTISSA_PARITY` turns the bound strict exactly when `t` is odd (ties-to-even).
    pub lower_comparison_borrows: [T; BORROW_BITS],
    /// Q8: borrow bits of the mirror comparison `SS * 2^32 + ODD <= ACU`.
    pub upper_comparison_borrows: [T; BORROW_BITS],

    // Grid snap `l2 = grid4(y)` (group G): round the sqrt code to a multiple of four, ties up.
    // SQRT_CODE + 2 = 4*GRID_SNAP_QUOTIENT + GRID_SNAP_REMAINDER,
    // GRID_SNAP_REMAINDER in [0, 4), and L2_CODE = 4*GRID_SNAP_QUOTIENT.
    /// G1: the snap quotient (bounded via the G3 decode: `4*GRID_SNAP_QUOTIENT < 2^15 + 2^7`).
    pub grid_snap_quotient: T,
    /// G1/G2: the snap remainder, in [0, 4) by two RC16s.
    pub grid_snap_remainder: T,
    /// G3: l2 exponent field (EXPINFO-bound; an exponent-255 snap has no row and is rejected).
    pub l2_exp: T,
    /// G3: l2 mantissa field (PAIR128-bound, shared slot with LINF_MANTISSA).
    pub l2_mantissa: T,
    /// G3: l2 subnormal flag (EXPINFO value).
    pub l2_exp_is_zero: T,

    // Linf decode (group N): MAX_ABS *is* the linf code (no epsilon bump).
    /// N1: linf exponent field.
    pub linf_exp: T,
    /// N1: linf mantissa field.
    pub linf_mantissa: T,
    /// N1: linf subnormal flag.
    pub linf_exp_is_zero: T,

    // H0: floor both norms at 2^-32. Code order matches value order for nonnegative
    // BF16. The selected significand/exponent pairs are committed to keep downstream
    // FMA and lookup expressions within the degree bound.
    /// H0: `l2_code >= 0x2F80` (the bf16 code of `2^-32`) order bit.
    pub l2_ge_floor: T,
    /// H0: two-sided order slack for the l2 floor (range-checked).
    pub l2_floor_order_slack: T,
    /// H0: the floored l2 significand `M(l2f)` (128 — the floor's own `M` — on floored rows).
    pub l2_floored_significand: T,
    /// H0: the floored l2 effective exponent `E*(l2f)` (95 on floored rows).
    pub l2_floored_exponent: T,
    /// H0: `linf_code >= 0x2F80` order bit.
    pub linf_ge_floor: T,
    /// H0: two-sided order slack for the linf floor (range-checked).
    pub linf_floor_order_slack: T,
    /// H0: the floored linf significand `M(linf_f)`.
    pub linf_floored_significand: T,
    /// H0: the floored linf effective exponent `E*(linf_f)`.
    pub linf_floored_exponent: T,

    /// S1: the row's exact noise-scale significand `M(alpha) * M(l2f)`, in `[2^14, 2^16)`.
    /// Its exponent is `ALPHA_EXP + L2_FLOORED_EXPONENT - 269`, as derived in [`super::stark`].
    /// Committing the product keeps the exported sigma tuple linear. PAIR128/EXPINFO
    /// bounds on its factors already imply the significand range.
    pub sigma_significand: T,
    /// S2: 1 iff `SIGMA_SIGNIFICAND >= 2^15`, selecting one of its two possible binades.
    /// Each branch has an RC16 check, so the prover cannot choose the wrong width.
    /// InputQuant receives the resulting exponent for summand scoring:
    /// `e(sigma) + 268 = ALPHA_EXP + L2_FLOORED_EXPONENT + SIGMA_SIG_IS_WIDE + 13`.
    pub sigma_sig_is_wide: T,
    /// Sigma significand normalized to [2^15, 2^16) (S3).
    /// With exponent e = floor(log2(sigma)), `sigma = normalized_significand * 2^(e - 15)`.
    /// Exported to InputQuant for summand scoring.
    pub normalized_sigma_significand: T,

    // The scale chain (group H). No separate denominator floor exists (the reference floors
    // the norms, not the noised bound): `noised_bound >= linf_f >= 2^-32` by RNE monotonicity,
    // so DIV448 divides by the FMA output directly.
    /// H1: `noised_bound = RNE_bf16(dr * l2f + linf_f)` — the fused FMA gadget.
    pub noised_bound_fma: FmaBlock<T>,

    /// H4: `beta_1 = RNE_bf16(alpha * l2f)`.
    pub alpha_l2_multiply: MulBlock<T>,
    /// H5: `beta = RNE_bf16(beta_1 * dos)`; its outputs are bound to the tuple's beta claim.
    pub beta_scale_multiply: MulBlock<T>,
}

pub const NUM_SCALE_COLUMNS: usize = size_of::<ScaleColumnsView<u8>>();

const _: () = assert!(NUM_SCALE_COLUMNS == 145);

// The verifier derives dr/dos, geometry and policy budgets from the statement.
// The sqrt bounds require k % 32 = 0 and 2048 <= k <= 2^16.
/// `dr` exponent field.
pub const DR_EXP_PUBLIC_INPUT: usize = 0;
/// `dr` mantissa field.
pub const DR_MANTISSA_PUBLIC_INPUT: usize = 1;
/// `dos` exponent field.
pub const DOS_EXP_PUBLIC_INPUT: usize = 2;
/// `dos` mantissa field.
pub const DOS_MANTISSA_PUBLIC_INPUT: usize = 3;
/// The row length `k` (Q7's claim-side products are `B^2 * k * 2^15`). Sanctioned envelope:
/// `k % 32 == 0` (limb 0 of the claim-side products provably vanishes: `2^20 | B^2*k*2^15`)
/// and `2048 <= k <= 2^16` (the `< 2^53` claim-product cap in `ctl.rs` assumes `k <= 2^16`).
pub const K_PUBLIC_INPUT: usize = 4;
/// `Wl2 = 27 - ceil(log2 k)` — the block-L2 frame width (Q5's alignment window). In
/// `[11, 16]` over the sanctioned `k` range. The native verifier pins this slot and
/// InputQuantStark's `2^Wl2` from the same `k` (plaintext power); the wrapper just
/// exposes both wires.
pub const WL2_PUBLIC_INPUT: usize = 5;
/// The A side's dead-entry allowance `floor(eps_idle * h*k)` (jackpot check 1): the T3 gate
/// accepts iff `RUNNING_DEAD_A <= DEAD_LIMIT_A` on the last row, the integer form of the
/// plaintext `dead as f64 <= eps_idle * (h*k) as f64`. Below `2^16` for every sanctioned
/// geometry (`h*k < 2^22`, `eps_idle = 1/64`), so the gate is one RC16 on
/// `DEAD_LIMIT_A - RUNNING_DEAD_A - 2^16·DEAD_SLACK_HI_A` with a boolean high bit.
pub const DEAD_LIMIT_A_PUBLIC_INPUT: usize = 6;
/// The B side's dead-entry allowance `floor(eps_idle * w*k)` (see `DEAD_LIMIT_A_PUBLIC_INPUT`).
pub const DEAD_LIMIT_B_PUBLIC_INPUT: usize = 7;
/// The tile width `w` — the sigma channel's A-row multiplicity (each A row's sigma serves the
/// `w` cells of its tile row; jackpot check 3). A looked-side CTL filter term, not read by
/// any AIR constraint.
pub const W_MULT_PUBLIC_INPUT: usize = 8;
/// The tile height `h` — the sigma channel's B-row multiplicity.
pub const H_MULT_PUBLIC_INPUT: usize = 9;
/// Number of ScaleStark public inputs.
pub const NUM_SCALE_PUBLIC_INPUTS: usize = 10;

columns_view!(ScaleColumnsView, NUM_SCALE_COLUMNS, SCALE_COL_MAP);

/// Number of leading verifier-known schedule columns.
pub const NUM_SCALE_KNOWN_COLUMNS: usize = SCALE_COL_MAP.is_pad + 1;

#[cfg(test)]
mod tests {
    use core::borrow::Borrow;

    use super::*;

    #[test]
    fn col_map_is_the_identity_layout() {
        let as_array: [usize; NUM_SCALE_COLUMNS] = SCALE_COL_MAP.into();
        for (i, &c) in as_array.iter().enumerate() {
            assert_eq!(c, i);
        }
        assert_eq!(SCALE_COL_MAP.group_key, 0);
        assert_eq!(SCALE_COL_MAP.is_last_row, 1);
        assert_eq!(SCALE_COL_MAP.is_a_row, 2);
        assert_eq!(SCALE_COL_MAP.is_pad, 3);
        assert_eq!(SCALE_COL_MAP.beta_scale_multiply.out_exp_is_zero, NUM_SCALE_COLUMNS - 1);
    }

    #[test]
    fn view_array_roundtrip() {
        let mut arr = [0u64; NUM_SCALE_COLUMNS];
        for (i, v) in arr.iter_mut().enumerate() {
            *v = i as u64 * 3 + 1;
        }
        let view: ScaleColumnsView<u64> = arr.into();
        assert_eq!(view.group_key, 1);
        assert_eq!(view.frame_sum_limbs[2], SCALE_COL_MAP.frame_sum_limbs[2] as u64 * 3 + 1);
        assert_eq!(
            view.noised_bound_fma.rounding_significand_key,
            SCALE_COL_MAP.noised_bound_fma.rounding_significand_key as u64 * 3 + 1
        );
        assert_eq!(
            view.beta_scale_multiply.out_exp_is_zero,
            (NUM_SCALE_COLUMNS as u64 - 1) * 3 + 1
        );
        let back: [u64; NUM_SCALE_COLUMNS] = view.into();
        assert_eq!(back, arr);

        let borrowed: &ScaleColumnsView<u64> = arr.borrow();
        assert_eq!(borrowed.l2_floored_significand, arr[SCALE_COL_MAP.l2_floored_significand]);
    }
}
