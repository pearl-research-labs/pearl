//! Device-specific trace layouts for InputQuant's per-element AIR.
//!
//! One trace row pairs the same flat element index from the A and B strips. A group has
//! `k` elements and each committed bf16 scale covers eight. The height rounds `max(h,w)*k`
//! upward to the next supported height (`trace_padding`); per-side liveness excludes
//! phantom elements from CTLs.
//! The two leading columns are verifier-known: both sides' packed binary32 noise words,
//! recomputed from the public seeds. The schedule block that follows (group G) is pinned by
//! the AIR instead of being recomputed: the noise decode fields with their recomposition and
//! range witnesses, the group/block/parity flags with their counters, the element and block
//! indices, and per-side liveness with the inverse witnesses that force each flip. B200 adds
//! the four-element coin-start flag, tied to the block counter by one more inverse witness.
//!
//! Each common side carries bf16 decode `X=RN_bf16(int8*scale)`, framed norm aggregates,
//! clean-data policy scores, the CTL-certified binary32 block coefficient, and binary32
//! fold/rounding/proxy witnesses. B200 extends each side with the conditional byte rewrite
//! and Gaussian row-sum witnesses. The two device layouts have different physical offsets;
//! `input_quant_column_map` selects the appropriate common views and optional extensions.
//!
//! The `#[repr(C)]` views define committed column order. Their flat index maps are used by
//! lookups and CTLs; changing field order changes the AIR and proof layout. Bf16 tuples use
//! `M=128*(1-exp_is_zero)+mantissa`, `E*=exp+exp_is_zero`, and value
//! `(-1)^sign*M*2^(E*-134)`. The binary32 representations and limb certificates live in
//! `crate::v4::circuit::fp32`; [`super::stark`] gives the full mathematical statement.

use crate::v4::api::public_params::Device;
use crate::v4::circuit::columns_view::columns_view;
use crate::v4::circuit::fp32::{AlignmentColumns, DownscaleColumns, Fp32Fields, Fp32Magnitude, OddProxyColumns, Round32Columns};

#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default)]
pub struct Bf16FieldsView<T: Copy> {
    /// IEEE sign bit as a field element (0 = positive, 1 = negative).
    pub sign: T,
    /// Raw biased 8-bit exponent field, in `[0, 254]` (255 = inf/NaN is banned everywhere).
    pub exp: T,
    /// 7-bit fraction field without the implicit leading 1, in `[0, 127]`.
    pub mantissa: T,
    /// 1 iff `exp = 0` (value is subnormal or ±0).
    pub exp_is_zero: T,
}

#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default)]
pub struct MulDecodeBlockView<T: Copy> {
    /// Exact significand product `M(INT) * M(SCALE)` (<= 255^2 < 2^16); RNERND key component.
    pub sig_product: T,
    /// RNERND slot `clamp(-133 - LSB_SCALE, -7, 18) + 7`: `-133 = 1 - 127 - 7` is bf16's
    /// smallest-subnormal exponent and the clamped signed fade cut is biased by +7 into the
    /// slot index. CLAMP22 embeds its signed fade argument by adding 400, giving key
    /// `535 - E*(INT) - E*(SCALE)`.
    pub cut_depth: T,
    /// RNERND-supplied exponent adjustment (see the module docs); enters the U4 exponent
    /// binding `OUT_EXP = LSB_SCALE + WIDTH_ADJUST`.
    pub width_adjust: T,
}

#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default)]
pub struct BlockL2Columns<T: Copy> {
    /// Running in-block sum of integer squares; define `n_b = block_int_squared_sum` on the
    /// block-final row (B1/B2).
    pub block_int_squared_sum: T,
    /// `block_l2_product = M(scale)² * block_int_squared_sum`; define `p_b` as its block-final
    /// value, which is below `2^33` (B3).
    pub block_l2_product: T,
    /// Inverse witness for `block_l2_product`; paired with `block_l2_product_nonzero` (B4).
    pub block_l2_product_inverse: T,
    /// One iff `block_l2_product` is nonzero (B4).
    pub block_l2_product_nonzero: T,
    /// Candidate doubled scale exponent `2*E*(scale)` on live block-final rows, else zero (B5).
    pub block_doubled_scale_exponent: T,
    /// Running maximum of `block_doubled_scale_exponent` in the group (B6).
    pub running_max_doubled_scale_exponent: T,
    /// Group-constant common exponent frame; its group-final value is
    /// `F = max_b 2*E*(scale_b)` over nonzero blocks (S4/B7).
    pub frame_doubled_scale_exponent: T,
    /// Four 16-bit limbs of `block_l2_product * 2^Wl2`, below `2^54` (B10).
    pub scaled_block_product_limbs: [T; 4],
    /// `2^r`, where `r = sigma_b mod 16`; used only on nonzero, block-final, non-far rows
    /// (B11).
    pub shift_remainder_power: T,
    /// Complement power `2^(16-r)` used to recompose the shifted high limbs (B11/B16).
    pub shift_complement_power: T,
    /// One-hot selector for the limb containing the shift boundary (B13).
    pub shift_limb_selector: [T; 4],
    /// Quotient of the selected limb by `2^r̂` (B14/B15).
    pub selected_limb_quotient: T,
    /// Remainder of the selected-limb split, below `2^r` (B14/B15).
    pub selected_limb_remainder: T,
    /// One on live block-final rows whose frame shift is at least 54 (B8).
    pub is_far_shift: T,
    /// Running framed block-term sum; define `S = running_l2_frame_sum` at group-final (B16).
    pub running_l2_frame_sum: T,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct InputQuantSideColumns<T: Copy> {
    pub int8_byte: T,
    pub int_bf16: Bf16FieldsView<T>,
    pub scale: Bf16FieldsView<T>,
    pub decode_multiply: MulDecodeBlockView<T>,
    pub x_sign: T,
    pub x_exp: T,
    pub x_mantissa: T,
    pub x_exp_is_zero: T,
    pub x_is_zero: T,
    pub block_l2: BlockL2Columns<T>,
    pub max_abs: T,
    pub alpha_exp: T,
    pub alpha_mantissa: T,
    pub beta_exp: T,
    pub beta_mantissa: T,
    pub beta_exp_is_zero: T,
    pub scaled_decoded_significand: T,
    pub code_noised: T,
    pub dead_bound: T,
    pub is_dead: T,
    pub dead_count: T,
    pub x_width: T,
    pub x_enc: T,
    pub sigma_enc: T,
    pub sigma_norm: T,
    pub x_pow: T,
    pub x_norm: T,
    pub x_dominates: T,
    pub exponent_gap: T,
    pub gap_is_far: T,
    pub near_gap: T,
    pub near_gap_pow: T,
    pub norm_big: T,
    pub norm_small: T,
    pub half_quotient: T,
    pub half_remainder: T,
    pub quotient: T,
    pub remainder: T,
    pub sum_of_squares_top: T,
    pub sum_of_squares_rest_lo: T,
    pub sum_of_squares_rest_hi: T,
    pub log_fraction: T,
    pub lambda: T,
    pub coefficient: Fp32Magnitude<T>,
    pub alignment: AlignmentColumns<T>,
    pub rounding: Round32Columns<T>,
    pub proxy: OddProxyColumns<T>,
}

/// The verifier-known prefix: each side's binary32 noise word
/// `SIGN*2^31 + EXPONENT*2^23 + FRACTION` (0 on dead rows), recomputed from the public seeds
/// and checked against the trace openings. Group G binds every other schedule column to
/// these two and to the row counter.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct InputQuantKnownColumns<T: Copy> {
    pub noise_word_a: T,
    pub noise_word_b: T,
}

/// Group G: the AIR-pinned noise decodes and schedule ([`super::stark`] documents G1-G13).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct InputQuantScheduleColumns<T: Copy> {
    /// A-side binary32 noise decode. G1 recomposes it to `noise_word_a` and G2 makes the
    /// sign boolean; the unfiltered EXPINFO lookup pins the exponent to `[0, 254]` with its
    /// zero flag, and the fraction split below bounds the fraction to `[0, 2^23)`.
    pub noise_a: Fp32Fields<T>,
    /// B-side noise decode (see `noise_a`).
    pub noise_b: Fp32Fields<T>,
    /// `floor(FRACTION / 2^16)` of the A-side noise, below 128 (PAIR128 with the B side);
    /// RC16 bounds the remainder `FRACTION - 2^16 * HIGH`.
    pub noise_fraction_high_a: T,
    /// B-side fraction high part (see `noise_fraction_high_a`).
    pub noise_fraction_high_b: T,
    /// 1 on the last row of each k-element group and on the trace's last row (G3/G4); CTL
    /// filter of the Scale channel; gates the accumulator resets.
    pub is_group_final: T,
    /// The row counter (G6); the A element CTL key. The B key is `ELEMENT_INDEX + h*k`.
    pub element_index: T,
    /// Period-2 flag (G7); filter of the pair-packed Blake3/Matmul channels.
    pub is_even_row: T,
    /// `[ELEMENT_INDEX ≡ 0 mod 8]` (G8-G10); filter of the scales channel and anchor of the
    /// scale-field copy constraints.
    pub is_block_start: T,
    /// `floor(ELEMENT_INDEX / 8)` (G11); the A scales CTL key. The B key is offset by `h*k/8`.
    pub block_index: T,
    /// `[ELEMENT_INDEX < h*k]` (G12); filters A-side CTLs, dead rows carry the phantom fill.
    pub a_live: T,
    /// `[ELEMENT_INDEX < w*k]` (G12).
    pub b_live: T,
    /// In-group position `ELEMENT_INDEX mod k` (G3/G4). A truncated tail closes on the last
    /// row without requiring this counter to equal `k-1`.
    pub group_counter: T,
    /// Inverse of `GROUP_COUNTER - (k-1)` off group-final rows (G4); forces IS_GROUP_FINAL
    /// exactly at the group boundary. Unconstrained on group-final rows.
    pub group_final_inv: T,
    /// `ELEMENT_INDEX mod 8` (G8-G10): the IS_BLOCK_START driver.
    pub block8_counter: T,
    /// Inverse of `BLOCK8_COUNTER - 7` off block-final rows (G9); forces IS_BLOCK_START
    /// exactly at block boundaries.
    pub block8_counter_inv: T,
    /// Inverse of `ELEMENT_INDEX - h*k` on A-live rows (G12); forces the A-liveness drop
    /// exactly at row `h*k`. Unconstrained on dead rows.
    pub a_flip_inv: T,
    /// Inverse of `ELEMENT_INDEX - w*k` on B-live rows (see `a_flip_inv`).
    pub b_flip_inv: T,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct InputQuantColumnsView<T: Copy> {
    pub known: InputQuantKnownColumns<T>,
    pub schedule: InputQuantScheduleColumns<T>,
    pub a: InputQuantSideColumns<T>,
    pub b: InputQuantSideColumns<T>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct B200InputQuantSideColumns<T: Copy> {
    pub common: InputQuantSideColumns<T>,
    pub downscale: DownscaleColumns<T>,
    /// Group-constant LOG16 region `1 + class` of the row's Gaussian lookups (0 on phantom rows).
    pub noise_coverage_region: T,
    /// Group-constant offset; the Gaussian lookup index is `lambda - noise_coverage_key_base`.
    pub noise_coverage_key_base: T,
    /// Q16 Gaussian kernel, including the small-noise cap.
    pub gaussian_value: T,
    /// Inclusive kernel sum, reset at each operand row.
    pub noise_coverage_sum: T,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct B200InputQuantColumnsView<T: Copy> {
    pub known: InputQuantKnownColumns<T>,
    pub schedule: InputQuantScheduleColumns<T>,
    /// `[BLOCK8_COUNTER ∈ {0, 4}]` (G13): the first element of each four-element coin block.
    pub is_coin_start: T,
    /// Inverse of `BLOCK8_COUNTER - 4` off half-block rows (G13); certifies the half-block
    /// test that, with IS_BLOCK_START, determines IS_COIN_START.
    pub coin_inv: T,
    pub a: B200InputQuantSideColumns<T>,
    pub b: B200InputQuantSideColumns<T>,
}

pub const NUM_INPUT_QUANT_COLUMNS: usize = size_of::<InputQuantColumnsView<u8>>();
pub const NUM_B200_INPUT_QUANT_COLUMNS: usize = size_of::<B200InputQuantColumnsView<u8>>();
/// Number of leading verifier-known columns (both devices): the two packed noise words.
pub const NUM_INPUT_QUANT_KNOWN_COLUMNS: usize = size_of::<InputQuantKnownColumns<u8>>();
pub const NUM_INPUT_QUANT_SCHEDULE_COLUMNS: usize = size_of::<InputQuantScheduleColumns<u8>>();
const _: () = assert!(NUM_INPUT_QUANT_KNOWN_COLUMNS == 2);
const _: () = assert!(NUM_INPUT_QUANT_SCHEDULE_COLUMNS == 23);
const _: () = assert!(size_of::<InputQuantSideColumns<u8>>() == 149);
const _: () = assert!(NUM_INPUT_QUANT_COLUMNS == 323);
const _: () = assert!(NUM_B200_INPUT_QUANT_COLUMNS == 343);
columns_view!(InputQuantColumnsView, NUM_INPUT_QUANT_COLUMNS, INPUT_QUANT_COL_MAP);
columns_view!(
    B200InputQuantColumnsView,
    NUM_B200_INPUT_QUANT_COLUMNS,
    B200_INPUT_QUANT_COL_MAP
);

#[derive(Clone, Copy)]
pub(crate) struct InputQuantColumnMap {
    pub schedule: InputQuantScheduleColumns<usize>,
    pub sides: [InputQuantSideColumns<usize>; 2],
    /// B200 only: the coin-start flag and the columns used for downscaling.
    pub downscale: Option<(usize, [DownscaleColumns<usize>; 2])>,
}

pub(crate) const fn input_quant_column_map(device: Device) -> InputQuantColumnMap {
    match device {
        Device::H100 => InputQuantColumnMap {
            schedule: INPUT_QUANT_COL_MAP.schedule,
            sides: [INPUT_QUANT_COL_MAP.a, INPUT_QUANT_COL_MAP.b],
            downscale: None,
        },
        Device::B200 => InputQuantColumnMap {
            schedule: B200_INPUT_QUANT_COL_MAP.schedule,
            sides: [B200_INPUT_QUANT_COL_MAP.a.common, B200_INPUT_QUANT_COL_MAP.b.common],
            downscale: Some((
                B200_INPUT_QUANT_COL_MAP.is_coin_start,
                [B200_INPUT_QUANT_COL_MAP.a.downscale, B200_INPUT_QUANT_COL_MAP.b.downscale],
            )),
        },
    }
}

/// `2^Wl2`, the job's block-L2 precision scale.
pub const WL2_POW_PUBLIC_INPUT: usize = 0;
/// `h*k`: the B slots' element-key offset and the A-liveness flip bound (G12).
pub const B_KEY_OFFSET_PUBLIC_INPUT: usize = 1;
/// `h*k/8`: the B slots' block-key offset.
pub const B_BLOCK_KEY_OFFSET_PUBLIC_INPUT: usize = 2;
/// `w`: the A operand-code multiplicity.
pub const OPERAND_MULT_A_PUBLIC_INPUT: usize = 3;
/// `h`: the B operand-code multiplicity.
pub const OPERAND_MULT_B_PUBLIC_INPUT: usize = 4;
/// `k`: the group length — the GROUP_COUNTER/IS_GROUP_FINAL tie (G3/G4).
pub const K_PUBLIC_INPUT: usize = 5;
/// `w*k`: the B-liveness flip bound (G12's B instance).
pub const B_LIVE_BOUND_PUBLIC_INPUT: usize = 6;
/// Number of public inputs. Geometry enters the CTL expressions and group G through these
/// slots, keeping the verifier circuit independent of job geometry; the native gateway pins
/// every slot to its statement-derived value.
pub const NUM_INPUT_QUANT_PUBLIC_INPUTS: usize = 7;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_are_identity_and_known_prefixes_are_contiguous() {
        let h100: [usize; NUM_INPUT_QUANT_COLUMNS] = INPUT_QUANT_COL_MAP.into();
        let b200: [usize; NUM_B200_INPUT_QUANT_COLUMNS] = B200_INPUT_QUANT_COL_MAP.into();
        assert_eq!(h100, core::array::from_fn(|i| i));
        assert_eq!(b200, core::array::from_fn(|i| i));
        // The verifier-known noise words lead; the schedule block follows at the same offsets
        // on both devices (the CTL helpers read it through the H100 map).
        for (known, schedule) in [
            (INPUT_QUANT_COL_MAP.known, INPUT_QUANT_COL_MAP.schedule),
            (B200_INPUT_QUANT_COL_MAP.known, B200_INPUT_QUANT_COL_MAP.schedule),
        ] {
            assert_eq!(known.noise_word_a, 0);
            assert_eq!(known.noise_word_b + 1, NUM_INPUT_QUANT_KNOWN_COLUMNS);
            assert_eq!(schedule.noise_a.sign, NUM_INPUT_QUANT_KNOWN_COLUMNS);
            assert_eq!(
                schedule.b_flip_inv + 1,
                NUM_INPUT_QUANT_KNOWN_COLUMNS + NUM_INPUT_QUANT_SCHEDULE_COLUMNS
            );
        }
        assert_eq!(INPUT_QUANT_COL_MAP.a.int8_byte, 25);
        assert_eq!(INPUT_QUANT_COL_MAP.b.int8_byte, 174);
        assert_eq!(B200_INPUT_QUANT_COL_MAP.is_coin_start, 25);
        assert_eq!(B200_INPUT_QUANT_COL_MAP.coin_inv, 26);
        assert_eq!(B200_INPUT_QUANT_COL_MAP.a.common.int8_byte, 27);
        assert_eq!(B200_INPUT_QUANT_COL_MAP.b.common.int8_byte, 185);
    }
}
