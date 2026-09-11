//! Trace columns for one B200 32-lane accumulation step.
//!
//! A row aligns 32 fp8 products and the incoming f32 carry to the greatest stored exponent,
//! sums the signed terms as integers in B200's 25-fractional-bit window, and truncates the
//! result toward zero to a 24-bit f32 significand. Product and carry stored exponents are
//! shifted by +38 so every nonzero value is positive and zero can use sentinel 0.
//!
//! The `#[repr(C)]` field order is committed column order; [`MATMUL_B200_COL_MAP`] exposes it
//! to lookups and cross-table channels.

use crate::circuit::fp8::columns_view::columns_view;

/// Number of fp8 products per row = per tcgen05 atom (the accumulation group is
/// carry + `GROUP_WIDTH` products, i.e. 33 summands).
pub const GROUP_WIDTH: usize = 32;

/// Length of the attainment chain (MB3): the running product of the 32 lane factors
/// `GROUP_MAX_BIASED_EXPONENT - PRODUCT_BIASED_EXPONENT_i`, accumulated at most two new factors per link so
/// every link constraint stays degree <= 3 (the first takes three factors and the last one).
pub const NUM_ATT_LINKS: usize = 16;

/// View of one MatmulB200Stark trace row. The constraint labels (MB1..MB16) refer to
/// `super::stark`'s constraint groups.
///
/// Exponent conventions: `PRODUCT_BIASED_EXPONENT`, `GROUP_MAX_BIASED_EXPONENT`, and
/// `GROUP_OUTPUT_BIASED_EXPONENT` use bias +38 and sentinel 0 (nonzero product exponents span
/// [26, 54]).
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct MatmulB200ColumnsView<T: Copy> {
    // Verifier-known schedule columns; see `known_values`.
    /// Output cell index, constant across the cell's `k/32` rows; the XorFold channel key.
    pub cell_id: T,
    /// 1 on each cell's last row. Gates the f32 encode, the XorFold channel and the carry reset.
    pub is_cell_final: T,
    /// Operand flat-index base for this row's 32 A lanes.
    pub operand_index_base_a: T,
    /// Operand flat-index base for the B lanes; carries the constant `h*k` key offset matching
    /// InputQuantStark's B-channel keys.
    pub operand_index_base_b: T,
    /// 1 on the rows that read no operands: the trailing power-of-two padding rows (single-row
    /// phantom cells, `IS_CELL_FINAL = 1`, `CELL_ID >= h*w`). Padding rows are excluded from
    /// the operand-code and cell-result channels, and their lanes are pinned to zero products.
    pub is_padding: T,

    // Window-sum emulation (main).
    /// The lane's fp8 A-operand code, received from InputQuant via the pair-packed CTL and
    /// individually pinned by the B200ALIGN tuple's `OPERAND_CODES_A` binding (MB1).
    pub operand_codes_a: [T; GROUP_WIDTH],
    /// The lane's fp8 B-operand code (see `operand_codes_a`).
    pub operand_codes_b: [T; GROUP_WIDTH],
    /// The lane's window contribution as a *signed field element* (a negative integer `-v` is
    /// represented as `p - v`, where `p` is the Goldilocks modulus). Its magnitude is
    /// `floor(m*2^19 / 2^REL) < 2^27`, with product significand
    /// `m = sig_a*sig_b in [0, 225]` and lane-to-anchor gap
    /// `REL = GROUP_MAX_BIASED_EXPONENT - PRODUCT_BIASED_EXPONENT`; this is hardware
    /// truncation toward zero, bound by B200ALIGN.
    pub aligned_lane_terms: [T; GROUP_WIDTH],
    /// The product's biased stored exponent `ea + eb + 24 in [26, 54]`, where `ea`/`eb` are
    /// the operands' subnormal-clamped fp8 exponent fields in [1, 15] (sentinel 0 for zero
    /// products); bound by B200ALIGN; the lookup key's slot component and the attainment
    /// target. There is no per-lane zero flag: the sentinel doubles as it (MB3/MB10).
    pub product_biased_exponents: [T; GROUP_WIDTH],
    /// "The carry entering this row is zero" flag, pinned by propagation from the previous row's
    /// zero state (1 at cell starts; MB4). The carry's sign/exponent/significand are *not*
    /// columns: they are the previous row's `group_output_sign`,
    /// `group_output_biased_exponent`, and `group_output_significand`, read by next-row
    /// references.
    pub incoming_carry_is_zero: T,
    /// `2^min(d, 26)` for the carry's gap to the anchor
    /// `d = GROUP_MAX_BIASED_EXPONENT' - GROUP_OUTPUT_BIASED_EXPONENT`, from the
    /// POW2GB LUT, on the consuming row (MB5). Equals 1 on zero-carry rows only by trace
    /// convention (nothing else reads it there).
    pub incoming_carry_shift_power: T,
    /// Low 16-bit limb of the aligned carry
    /// `floor(4*GROUP_OUTPUT_SIGNIFICAND_prev / 2^min(d, 26))`. The x4 lifts the 24-bit
    /// significand onto the window's 26-bit scale. Zero on zero-carry rows.
    pub aligned_incoming_carry_lo: T,
    /// High 10-bit limb of the aligned carry (range-checked as `HI * 2^6`).
    pub aligned_incoming_carry_hi: T,
    /// 16 + 10-bit limbs of the floor remainder
    /// `4*GROUP_OUTPUT_SIGNIFICAND_prev - ALIGNED_INCOMING_CARRY *
    /// INCOMING_CARRY_SHIFT_POWER` (< 2^26).
    pub incoming_carry_remainder_lo: T,
    pub incoming_carry_remainder_hi: T,
    /// 16 + 10-bit limbs of the two-sidedness witness
    /// `INCOMING_CARRY_SHIFT_POWER - 1 - INCOMING_CARRY_REMAINDER`.
    pub incoming_carry_remainder_bound_lo: T,
    pub incoming_carry_remainder_bound_hi: T,
    /// The window anchor, biased +38: the group's maximum biased stored exponent over 32
    /// products + carry; pinned two-sidedly: LUT key domains give >= (negative rel-shifts have
    /// no slot), mandatory attainment (MB3) gives <=.
    pub group_max_biased_exponent: T,
    /// Attainment chain (MB3): the running product of the 33 attainment factors — 32 affine
    /// lane factors `GROUP_MAX_BIASED_EXPONENT - PRODUCT_BIASED_EXPONENT_i` plus the zero-gated carry factor —
    /// accumulated at most two factors per link and constrained to vanish at the end. In a
    /// field, a zero product means some summand attains the group maximum exactly.
    pub max_exponent_attainment: [T; NUM_ATT_LINKS],
    /// Sign bit of the signed window sum.
    pub group_sum_sign: T,
    /// Magnitude of the signed window sum (single column, < 2^32 — range-pinned through MB7's
    /// Euclidean sandwich, there being no 2^32-key table).
    pub group_sum_abs: T,
    /// Zero-sum flag (boolean; the zero ban and MB7 normalization pin both directions).
    pub group_sum_is_zero: T,
    /// Claimed bit width of `GROUP_SUM_ABS` (in [1, 32]); proven exact by the MB7 sandwich.
    pub group_sum_width: T,
    /// The WIDTH32 pair for `W = GROUP_SUM_WIDTH`: the truncate-to-24-bits divisor and exact
    /// lift multiplier.
    pub truncation_power: T,
    pub lifting_power: T,
    /// 16 + 8-bit limbs of `floor(GROUP_SUM_ABS * 2^(24-W))`. The high limb is pinned into
    /// [128, 255] on live rows, proving the `[2^23, 2^24)` normalization range.
    pub normalized_group_sum_significand_lo: T,
    pub normalized_group_sum_significand_hi: T,
    /// Euclidean remainder of the truncation and its two-sided bound witness
    /// (< `TRUNCATION_POWER` <= 2^8; single RC16s).
    pub truncation_remainder: T,
    pub truncation_remainder_bound: T,
    /// Group-sum output sign (mux of the normal path and the +0 path; MB10). The next row reads
    /// `group_output_*` directly as its carry — no copy columns.
    pub group_output_sign: T,
    /// Group-sum output biased-38 exponent
    /// (`GROUP_MAX_BIASED_EXPONENT + GROUP_SUM_WIDTH - 26`; sentinel 0 on the zero path).
    pub group_output_biased_exponent: T,
    /// Group-sum output 24-bit significand (`NORMALIZED_GROUP_SUM_SIGNIFICAND`; 0 on zero).
    pub group_output_significand: T,
    /// Low 16-bit limb of the cell result's f32 word, encoded on `IS_CELL_FINAL` rows (MB12);
    /// consumed by XorFold.
    pub cell_result_f32_lo: T,
    /// High 16-bit limb of the cell result's f32 word.
    pub cell_result_f32_hi: T,

    // MB13: bound every product and partial-sum magnitude by one cell exponent.
    // The witness generator chooses the exact maximum; constraints permit larger bounds.
    /// MB13: the lane product's `floor(log2 |product|) + 139` (0 for a zero product), served
    /// by the same B200ALIGN lookup as the term. `RC16(CELL_MAGNITUDE_EXPONENT - LANE_BINADES_i)` proves the
    /// bound; nonzero binades are >= 121, so CELL_MAGNITUDE_EXPONENT = 0 implies an all-zero cell.
    pub lane_binades: [T; GROUP_WIDTH],
    /// MB13: CELL_MAGNITUDE_EXPONENT, constant across the cell's rows, exported to TamedStark on the cell-final
    /// row. The partial sums' bound is `RC16(CELL_MAGNITUDE_EXPONENT - GROUP_OUTPUT_BIASED_EXPONENT - 101)`
    /// (see `PARTIAL_BINADE_OFFSET`), filtered off on zero partials.
    pub cell_magnitude_exponent: T,

    // Jackpot check 4 (MB14-MB16): per-lane skip decisions and their running count.
    /// MB14: 0 claims a zero cell via `(1 - cell_nonzero) * cell_magnitude_exponent = 0`.
    /// Nonzero cells must set this flag: their lane binades force `cell_magnitude_exponent >= 121`.
    pub cell_nonzero: T,
    /// MB15: the lane's A-operand summand score `lambda`, received from InputQuant on the
    /// widened operand-code channel (0 encodes "no finite summand").
    pub summand_score_a: [T; GROUP_WIDTH],
    /// The lane's B-operand summand score (see `summand_score_a`).
    pub summand_score_b: [T; GROUP_WIDTH],
    /// MB15: mark this lane as skipped. Boolean; zero on padding rows and zero cells.
    /// A nonzero cell's unskipped lane must prove:
    /// `SUMMAND_SCORE_A + SUMMAND_SCORE_B - 128*CELL_MAGNITUDE_EXPONENT - skip_threshold_offset in [0, 2^16)`
    /// The filter is `CELL_NONZERO * (1 - SKIP_FLAG)`. Marking extra lanes as skipped
    /// is allowed and only makes the tile's skip budget harder to satisfy.
    pub skip_flag: [T; GROUP_WIDTH],
    /// MB16: sum of skip flags through this row within the current cell.
    /// The cell's final count is sent to Tamed for the tile-wide budget check.
    pub cell_skips: T,
}

pub const NUM_MATMUL_B200_COLUMNS: usize = size_of::<MatmulB200ColumnsView<u8>>();

const _: () = assert!(NUM_MATMUL_B200_COLUMNS == 304);

columns_view!(MatmulB200ColumnsView, NUM_MATMUL_B200_COLUMNS, MATMUL_B200_COL_MAP);

/// Number of leading verifier-known schedule columns.
pub const NUM_MATMUL_B200_KNOWN_COLUMNS: usize = MATMUL_B200_COL_MAP.is_padding + 1;

/// Number of MatmulB200Stark public inputs — none: the AIR is program-independent.
pub const NUM_MATMUL_PUBLIC_INPUTS: usize = 0;

#[cfg(test)]
mod tests {
    use core::borrow::Borrow;

    use super::*;

    #[test]
    fn col_map_is_the_identity_layout() {
        let as_array: [usize; NUM_MATMUL_B200_COLUMNS] = MATMUL_B200_COL_MAP.into();
        for (i, &c) in as_array.iter().enumerate() {
            assert_eq!(c, i);
        }
        assert_eq!(MATMUL_B200_COL_MAP.cell_id, 0);
        assert_eq!(MATMUL_B200_COL_MAP.is_cell_final, 1);
        assert_eq!(MATMUL_B200_COL_MAP.operand_index_base_a, 2);
        assert_eq!(MATMUL_B200_COL_MAP.operand_index_base_b, 3);
        assert_eq!(MATMUL_B200_COL_MAP.is_padding, 4);
        assert_eq!(NUM_MATMUL_B200_KNOWN_COLUMNS, 5);
        assert_eq!(MATMUL_B200_COL_MAP.cell_skips, NUM_MATMUL_B200_COLUMNS - 1);
    }

    #[test]
    fn view_array_roundtrip() {
        let mut arr = [0u64; NUM_MATMUL_B200_COLUMNS];
        for (i, v) in arr.iter_mut().enumerate() {
            *v = i as u64 * 3 + 1;
        }
        let view: MatmulB200ColumnsView<u64> = arr.into();
        assert_eq!(view.cell_id, 1);
        assert_eq!(view.operand_codes_a[0], MATMUL_B200_COL_MAP.operand_codes_a[0] as u64 * 3 + 1);
        assert_eq!(view.cell_skips, (NUM_MATMUL_B200_COLUMNS as u64 - 1) * 3 + 1);
        let back: [u64; NUM_MATMUL_B200_COLUMNS] = view.into();
        assert_eq!(back, arr);

        let borrowed: &MatmulB200ColumnsView<u64> = arr.borrow();
        assert_eq!(
            borrowed.group_max_biased_exponent,
            arr[MATMUL_B200_COL_MAP.group_max_biased_exponent]
        );
    }
}
