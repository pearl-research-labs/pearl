//! Trace columns for one A100 `HMMA.16816.F32` accumulation step (one `G = 8`
//! group per row).
//!
//! A live row aligns 8 FP16 products and the incoming FP32 carry to the greatest
//! stored exponent `eta`, sums the signed terms as integers on the `2^(eta-24)`
//! grid, and truncates the result toward zero to a 24-bit FP32 significand. The
//! exponent axis is biased by +127 (the FP32 bias): a nonzero product's
//! `e_u = eps(a)+eps(b)` maps to `e_u + 127 in [99, 157]`, the carry's clamped
//! exponent `el` maps to `el + 127 in [1, 254]` (i.e. the FP32 exponent field),
//! and sentinel `0` marks a zero product / zero carry.
//!
//! See `docs/fp16_scheme/stark_feasibility.md` for the derivation and
//! `crate::v5::api::accumulate` for the bit-exact plaintext oracle this trace
//! must match. Constraint labels `MA1..MA10` refer to `super::stark`.

use crate::v4::circuit::columns_view::columns_view;

/// FP16 products per hardware accumulation group (A100 `HMMA` group size).
pub const GROUP: usize = 8;

/// Length of the eta-attainment chain (MA2): 8 affine lane factors accumulated at
/// most two new factors per link after a first link of three, so every link
/// constraint stays degree <= 3. `3 + 2 + 2 + 1 = 8`.
pub const NUM_ATT_LINKS: usize = 4;

/// View of one `MatmulStarkA100` trace row.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct MatmulA100ColumnsView<T: Copy> {
    // ---- Structural, class (a): verifier-recomputable from the AIR geometry. ----
    /// Output cell index, constant across the cell's `k/8` rows.
    pub cell_id: T,
    /// 1 on each cell's last row. Gates the f32 encode and the carry reset.
    pub is_cell_final: T,
    /// Operand flat-index base for this row's 8 A lanes.
    pub operand_index_base_a: T,
    /// Operand flat-index base for the 8 B lanes (carries the `h*k` B-plane offset).
    pub operand_index_base_b: T,
    /// 1 on trailing power-of-two padding rows (single-row phantom cells).
    pub is_padding: T,

    // ---- Per-lane operand decode (FP16DECODE LUT-served) + product (in-AIR). ----
    /// The lane's 16-bit FP16 A-operand code.
    pub operand_codes_a: [T; GROUP],
    /// The lane's 16-bit FP16 B-operand code.
    pub operand_codes_b: [T; GROUP],
    /// Decoded 11-bit A significand `m_a in [0, 2047]` (0 for a zero operand).
    /// FP16DECODE value 0, keyed on [`Self::operand_codes_a`].
    pub sig_a: [T; GROUP],
    /// Decoded 11-bit B significand `m_b in [0, 2047]`.
    pub sig_b: [T; GROUP],
    /// Raw sign bit of operand A (FP16DECODE value 1). Load-bearing only on nonzero lanes.
    pub sign_a: [T; GROUP],
    /// Raw sign bit of operand B (FP16DECODE value 1).
    pub sign_b: [T; GROUP],
    /// Biased stored exponent of operand A, `eps_a + 15 in [1, 30]` (FP16DECODE value 2).
    pub eps_a: [T; GROUP],
    /// Biased stored exponent of operand B, `eps_b + 15 in [1, 30]` (FP16DECODE value 2).
    pub eps_b: [T; GROUP],
    /// `[sig_a == 0]` (FP16DECODE value 3): operand A is a zero code.
    pub is_zero_a: [T; GROUP],
    /// `[sig_b == 0]` (FP16DECODE value 3): operand B is a zero code.
    pub is_zero_b: [T; GROUP],
    /// Product sign `sign_a xor sign_b` (boolean; MA1 derives it from the decoded sign bits).
    pub lane_sign: [T; GROUP],
    /// The product's biased stored exponent `ea + eb + 127 in [99, 157]`, or the
    /// sentinel 0 for a zero product. MA1 derives it from the decoded per-operand
    /// `eps_a/eps_b/is_zero_a/is_zero_b`:
    /// `(1-is_zero_a)*(1-is_zero_b) * (eps_a + eps_b + 97)`.
    pub product_biased_exp: [T; GROUP],
    /// Exact product significand `P = m_a * m_b in [0, 2^22)` (MA1: `sig_a*sig_b`).
    pub product_sig: [T; GROUP],

    // ---- Per-lane alignment (Euclidean truncation toward zero; MA3). ----
    /// `2^min(rel, 26)` for the lane's gap `rel = GROUP_MAX_BIASED_EXPONENT -
    /// PRODUCT_BIASED_EXPONENT` (POW2 LUT). 1 on padding / zero lanes.
    pub lane_shift_power: [T; GROUP],
    /// Unsigned aligned lane magnitude `floor(P*16 / 2^rel) < 2^26`.
    pub aligned_mag: [T; GROUP],
    /// Euclidean remainder `P*16 - ALIGNED_MAG * LANE_SHIFT_POWER` (< LANE_SHIFT_POWER).
    pub lane_rem: [T; GROUP],
    /// Two-sidedness witness `LANE_SHIFT_POWER - 1 - LANE_REM`.
    pub lane_rem_bound: [T; GROUP],
    /// Census: 1 iff this product's truncation discarded a nonzero bit
    /// (`LANE_REM != 0`). Boolean, pinned to `[LANE_REM != 0]` *exactly* by MA3's
    /// two-sided census (clean `(1-flag)*rem = 0`, tight `flag*(1 - rem*inv) = 0`).
    pub products_truncated_flag: [T; GROUP],
    /// Tight-census inverse witness: `LANE_REM^{-1}` when the lane truncated, else 0.
    /// MA3 pins `products_truncated_flag = 1 => LANE_REM != 0` via `flag*(rem*inv - 1) = 0`.
    pub lane_rem_inv: [T; GROUP],

    // ---- Incoming-carry alignment (reads previous row's output as the carry). ----
    /// "The carry entering this row is zero" flag (propagated; MA7).
    pub incoming_carry_is_zero: T,
    /// `2^min(rel_c, 26)` for the carry gap (POW2 LUT), on the consuming row.
    pub carry_shift_power: T,
    /// Unsigned aligned carry `floor(2 * prev.OUT_SIG / 2^rel_c) < 2^25`.
    pub aligned_carry: T,
    /// Euclidean remainder of the carry alignment and its two-sided bound.
    pub carry_rem: T,
    pub carry_rem_bound: T,
    /// Census: 1 iff the carry alignment discarded a nonzero bit (`CARRY_REM != 0`).
    /// Boolean, pinned to `[CARRY_REM != 0]` exactly by MA11 (one half of a breakpoint).
    pub carry_dropped: T,
    /// Tight-census inverse witness: `CARRY_REM^{-1}` when the carry truncated, else 0.
    pub carry_rem_inv: T,

    // ---- Window max + exact sum + round-toward-zero. ----
    /// The window anchor `eta + 127`: max biased stored exponent over nonzero
    /// products and the carry; sentinel 0 on an all-zero + zero-carry row.
    pub group_max_biased_exponent: T,
    /// Inverse witness pinning [`group_nonempty`] (`group_max * inv = nonempty`).
    pub group_max_inv: T,
    /// 1 iff the group did real work (`GROUP_MAX_BIASED_EXPONENT != 0`).
    pub group_nonempty: T,
    /// Attainment chain (MA2): running product of the 8 affine lane factors
    /// `GROUP_MAX_BIASED_EXPONENT - PRODUCT_BIASED_EXPONENT_i`; vanishes at the end.
    pub max_exponent_attainment: [T; NUM_ATT_LINKS],
    /// Sign of the signed window sum (boolean).
    pub group_sum_sign: T,
    /// Magnitude of the signed window sum (< 2^30).
    pub group_sum_abs: T,
    /// Zero-sum flag (boolean).
    pub group_sum_is_zero: T,
    /// Claimed bit width `W` of `GROUP_SUM_ABS` (in [1, 30]); WIDTH LUT key.
    pub group_sum_width: T,
    /// WIDTH-LUT pair for `W`: `2^max(W-24,0)` and `2^max(24-W,0)`.
    pub truncation_power: T,
    pub lifting_power: T,
    /// `floor(GROUP_SUM_ABS * LIFTING_POWER / TRUNCATION_POWER) in [2^23, 2^24)`.
    pub norm_sig: T,
    /// Euclidean remainder of the normalization and its two-sided bound.
    pub trunc_rem: T,
    pub trunc_rem_bound: T,
    /// Census: 1 iff the RZ normalization discarded a nonzero bit (`TRUNC_REM != 0`).
    /// Boolean, pinned to `[TRUNC_REM != 0]` exactly by MA11 (the other half of a breakpoint).
    pub rz_dropped: T,
    /// Tight-census inverse witness: `TRUNC_REM^{-1}` when the RZ dropped a bit, else 0.
    pub trunc_rem_inv: T,
    /// Census: 1 iff the carry alignment or the RZ normalization dropped a bit
    /// (a breakpoint). Boolean, pinned *exactly* to
    /// `GROUP_NONEMPTY * (CARRY_DROPPED OR RZ_DROPPED)` by MA11 — equals the
    /// `a100_dot` breakpoint census bit-for-bit, so it can neither be over- nor
    /// under-stated (the policy AIR's `rho`/`f_bp` gate depends on this tightness).
    pub group_breakpoint: T,

    // ---- Group output (= the carry entering the next row). ----
    /// Output sign (mux of the normal and +0 paths; MA8).
    pub out_sign: T,
    /// Output biased exponent: `GROUP_MAX_BIASED_EXPONENT + W - 25` (the FP32
    /// exponent field) on the normal path, sentinel 0 on both the zero path and
    /// the subnormal path (an FP32 subnormal has exponent field 0).
    pub out_biased_exp: T,
    /// Output significand: the normalized 24-bit `NORM_SIG` on the normal path,
    /// the (`< 2^23`) subnormal mantissa on the subnormal path, 0 on the zero path.
    pub out_sig: T,

    // ---- Subnormal FP32-output RZ branch (MA13). An output whose normal biased
    // exponent `GROUP_MAX_BIASED_EXPONENT + W - 25` would be `<= 0` is an FP32
    // subnormal (`|x| < 2^-126`): it is encoded on the `2^-149` grid with exponent
    // field 0 and a `< 2^23` mantissa holding fewer significand bits. The honest
    // from-zero matmul datapath never reaches this branch (FP16 products align at
    // `eta >= 99`, so a group output floors at `~2^-52`); the constraints below pin
    // the flag so the branch is a sound guard, and give the bit-exact `2^-149`
    // encode for the accumulation datapath (subnormal carry-in) the oracle
    // `a100_dot` already models. ----
    /// 1 iff this group's output is an FP32 subnormal (boolean; MA13). Forced 0 on
    /// the zero path and, in the from-zero AIR, on every live row (unreachable).
    pub out_is_subnormal: T,
    /// Nonnegative RANGE16 slack pinning the subnormal flag against the raw exponent
    /// `raw = GROUP_MAX_BIASED_EXPONENT + W - 25`: `raw - 1` on the normal path (so
    /// `raw >= 1`), `-raw` on the subnormal path (so `raw <= 0`). 0 on the zero path.
    pub exp_slack: T,
    /// Subnormal right-shift exponent `k = 1 - raw = 26 - GROUP_MAX_BIASED_EXPONENT - W`
    /// (`>= 1` on the subnormal path): the number of low `NORM_SIG` bits dropped to
    /// land the mantissa on the `2^-149` grid. FP16POW2 key (subnormal-filtered).
    pub sub_shift_exp: T,
    /// `2^sub_shift_exp` (FP16POW2 value, subnormal-filtered): the subnormal RZ divisor
    /// `NORM_SIG = OUT_SIG * SUB_SHIFT_POWER` (exact — the reachable subnormal outputs
    /// keep whole `NORM_SIG` high bits). 1 (inert) off the subnormal path.
    pub sub_shift_power: T,
    /// `OUT_SIG` low/high limbs on the subnormal path (`OUT_SIG < 2^23`): RANGE16-checked
    /// and reconstructed (`OUT_SIG = lo + 2^16*hi`) so the subnormal mantissa is a genuine
    /// small integer, not a field-wrapped quotient. 0 off the subnormal path.
    pub out_sig_lo: T,
    pub out_sig_hi: T,

    // ---- Limb splits of the 26-bit Euclidean-floor witnesses (MA12). Each tracked magnitude is
    // reconstructed as `lo + 2^16 * hi` with `lo` RANGE16-checked (`< 2^16`) and `hi` scaled by
    // `2^6` and RANGE16-checked (`< 2^10`, so the magnitude is `< 2^26`). This makes the per-lane
    // and carry `floor` identities integer-exact against Goldilocks field-fraction aliasing in a
    // real FRI proof: a wrapped quotient/remainder has no valid 16/10-bit limb witness. ----
    /// `aligned_mag[i]` low/high limbs.
    pub aligned_mag_lo: [T; GROUP],
    pub aligned_mag_hi: [T; GROUP],
    /// `lane_rem[i]` low/high limbs.
    pub lane_rem_lo: [T; GROUP],
    pub lane_rem_hi: [T; GROUP],
    /// `lane_rem_bound[i]` low/high limbs.
    pub lane_rem_bound_lo: [T; GROUP],
    pub lane_rem_bound_hi: [T; GROUP],
    /// `aligned_carry` low/high limbs.
    pub aligned_carry_lo: T,
    pub aligned_carry_hi: T,
    /// `carry_rem` low/high limbs.
    pub carry_rem_lo: T,
    pub carry_rem_hi: T,
    /// `carry_rem_bound` low/high limbs.
    pub carry_rem_bound_lo: T,
    pub carry_rem_bound_hi: T,
    /// `norm_sig` low/high limbs (`norm_sig < 2^24`, so the `2^6`-scaled hi check bounds it by
    /// `2^26` — ample for the RZ floor's no-wrap guarantee).
    pub norm_sig_lo: T,
    pub norm_sig_hi: T,

    // ---- Cell result + policy census totals. ----
    /// Low / high 16-bit limbs of the cell result's FP32 word (MA9, cell-final rows).
    pub cell_result_f32_lo: T,
    pub cell_result_f32_hi: T,
    /// In-cell running count of `products_truncated_flag` (rides the result channel).
    pub cell_products_truncated: T,
    /// In-cell running count of `group_breakpoint`.
    pub cell_breakpoints: T,
}

/// Total number of committed `MatmulStarkA100` columns.
pub const NUM_MATMUL_A100_COLUMNS: usize = size_of::<MatmulA100ColumnsView<u8>>();

// 19 per-lane arrays of 8 (= 152) + 5 structural + 7 carry + 19 window/RZ
// + 3 output + 4 result/census = 190, plus the MA12 limb splits: 6 per-lane arrays of 8 (= 48)
// + 8 carry/RZ limb singles = 56 -> 246, plus the MA13 subnormal-branch block
// (out_is_subnormal, exp_slack, sub_shift_exp, sub_shift_power, out_sig_lo, out_sig_hi) = 6 -> 252.
const _: () = assert!(NUM_MATMUL_A100_COLUMNS == 252);

columns_view!(MatmulA100ColumnsView, NUM_MATMUL_A100_COLUMNS, MATMUL_A100_COL_MAP);

/// Number of leading class (a) ("known") columns (pure functions of AIR geometry).
pub const NUM_MATMUL_A100_KNOWN_COLUMNS: usize = MATMUL_A100_COL_MAP.is_padding + 1;

/// Number of public inputs — none: the AIR is program-independent.
pub const NUM_MATMUL_A100_PUBLIC_INPUTS: usize = 0;

#[cfg(test)]
mod tests {
    use core::borrow::Borrow;

    use super::*;

    #[test]
    fn col_map_is_the_identity_layout() {
        let as_array: [usize; NUM_MATMUL_A100_COLUMNS] = MATMUL_A100_COL_MAP.into();
        for (i, &c) in as_array.iter().enumerate() {
            assert_eq!(c, i);
        }
        assert_eq!(MATMUL_A100_COL_MAP.cell_id, 0);
        assert_eq!(MATMUL_A100_COL_MAP.is_cell_final, 1);
        assert_eq!(MATMUL_A100_COL_MAP.operand_index_base_a, 2);
        assert_eq!(MATMUL_A100_COL_MAP.operand_index_base_b, 3);
        assert_eq!(MATMUL_A100_COL_MAP.is_padding, 4);
        assert_eq!(NUM_MATMUL_A100_KNOWN_COLUMNS, 5);
    }

    #[test]
    fn view_array_roundtrip() {
        let mut arr = [0u64; NUM_MATMUL_A100_COLUMNS];
        for (i, v) in arr.iter_mut().enumerate() {
            *v = i as u64 * 3 + 1;
        }
        let view: MatmulA100ColumnsView<u64> = arr.into();
        assert_eq!(view.cell_id, 1);
        assert_eq!(view.product_sig[0], MATMUL_A100_COL_MAP.product_sig[0] as u64 * 3 + 1);
        let back: [u64; NUM_MATMUL_A100_COLUMNS] = view.into();
        assert_eq!(back, arr);
        let borrowed: &MatmulA100ColumnsView<u64> = arr.borrow();
        assert_eq!(borrowed.group_max_biased_exponent, arr[MATMUL_A100_COL_MAP.group_max_biased_exponent]);
    }
}
