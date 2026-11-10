//! Cross-table and committed lookup relations used by InputQuantStark.
//!
//! A cross-table lookup (CTL) proves multiset equality between this trace and another STARK.
//! This file defines InputQuant's half of five channels: committed int8 bytes and block scales
//! from Blake3Stark, fp8 operand codes to the device-specific Matmul table, and completed row-group tuples to
//! ScaleStark, plus binary32 block coefficients from CoefficientStark. Their counterpart halves are assembled by
//! `super::super::ctl::all_cross_table_lookups`.
//!
//! The LUT descriptors range-check values and bind decode, rounding, and quantization table
//! outputs against the batch's committed lookup-table oracle. `super::stark` fills the witness
//! columns; the batch assembler includes these descriptors in the complete proof. Constraint
//! labels such as P1 and U2 refer to `super::stark`'s documented constraint groups; binary32
//! alignment/rounding certificates are documented in `super::super::fp32`.

use plonky2::field::types::Field;
use starky::cross_table_lookup::TableWithColumns;
use starky::lookup::{Column, Filter};

use super::super::ctl::Table;
use super::super::luts::LutTable;
use super::super::luts::ctl::LutLookup;
use super::columns::{
    B_BLOCK_KEY_OFFSET_PUBLIC_INPUT, B_KEY_OFFSET_PUBLIC_INPUT, BlockL2Columns, INPUT_QUANT_COL_MAP, InputQuantScheduleColumns,
    InputQuantSideColumns, OPERAND_MULT_A_PUBLIC_INPUT, OPERAND_MULT_B_PUBLIC_INPUT, input_quant_column_map,
};
use crate::v4::api::public_params::Device;

/// The bf16 code of a committed field triple, as the affine expression
/// `SIGN*2^15 + EXP*2^7 + MANTISSA`.
fn bf16_code_expr<F: Field>(sign: usize, exp: usize, mantissa: usize) -> Column<F> {
    Column::linear_combination([
        (sign, F::from_canonical_u64(1 << 15)),
        (exp, F::from_canonical_u64(1 << 7)),
        (mantissa, F::ONE),
    ])
}

/// `(this row's column) + 2^8*(next row's column)`: the pair-packing of two consecutive rows'
/// bytes, anchored at even rows.
fn byte_pair_expr<F: Field>(col: usize) -> Column<F> {
    Column::linear_combination_and_next_row_with_constant(
        vec![(col, F::ONE)],
        vec![(col, F::from_canonical_u64(1 << 8))],
        F::ZERO,
    )
}

/// The next row's column as an expression (the odd element of an even-anchored pair).
fn next_row_expr<F: Field>(col: usize) -> Column<F> {
    Column::linear_combination_and_next_row_with_constant(vec![], vec![(col, F::ONE)], F::ZERO)
}

/// `IS_EVEN_ROW * LIVE`: the pair-packing anchor restricted to the side's live rows (deg-2
/// filter; liveness is group-aligned and `h*k`/`w*k` are even, so a live pair never straddles
/// the boundary).
fn is_even_live_filter<F: Field>(live: usize) -> Filter<F> {
    Filter::new(
        vec![(Column::single(INPUT_QUANT_COL_MAP.schedule.is_even_row), Column::single(live))],
        vec![],
    )
}

/// The B slots' element key `ELEM_IDX + h*k`, with `h*k` read from the public input.
fn b_element_key<F: Field>() -> Column<F> {
    Column::single_plus_public_input(INPUT_QUANT_COL_MAP.schedule.element_index, B_KEY_OFFSET_PUBLIC_INPUT)
}

/// InputQuant's looked side of the strip-int8 channel, as the two slots `[A, B]` of one
/// [`starky::cross_table_lookup::CrossTableLookup`]:
/// `(key, INT8_BYTE + 2^8*INT8_BYTE')`, filtered by `IS_EVEN_ROW * LIVE`; the A key is
/// `ELEM_IDX`, the B key `ELEM_IDX + h*k` (public-input offset), keeping the two key
/// spaces disjoint. One tuple per byte pair = two consecutive elements, matching Blake3's
/// byte-pair tuples (its looking side is `blake3_stark::ctl::ctl_int8_bytes_looking_blake3`,
/// same low-element-in-low-byte packing; its `CTL_KEY_BASE` schedule carries the same `h*k` B
/// key offset). Sound as a pair-packing because each byte is individually pinned (BYTES2
/// there, the INT8DEC key domain here). Dead (phantom-fill) rows are excluded by the liveness
/// factor; the wrap instance of the next-row packing is filtered out: the last live row of a
/// side is odd (`h*k`/`w*k` are even).
pub fn ctl_int8_bytes_looked_input_quant<F: Field>(device: Device) -> Vec<TableWithColumns<F>> {
    let m = &input_quant_column_map(device);
    [
        (
            Column::single(m.schedule.element_index),
            m.sides[0].int8_byte,
            m.schedule.a_live,
        ),
        (b_element_key(), m.sides[1].int8_byte, m.schedule.b_live),
    ]
    .map(|(key, col, live)| {
        TableWithColumns::new(
            Table::InputQuant.into(),
            vec![key, byte_pair_expr(col)],
            is_even_live_filter(live),
        )
    })
    .to_vec()
}

/// InputQuant's looked side of the block-scales channel, slots `[A, B]`:
/// `(key, code(SCALE))`, filtered by `IS_BLOCK_START * LIVE`; the A key is `BLK_IDX`, the B
/// key `BLK_IDX + h*k/8` (public-input offset).
/// One bf16 block scale per tuple; the code expression is affine in the committed scale
/// fields, which P2 copies across the block (Blake3's looking side is
/// `blake3_stark::ctl::ctl_block_scales_looking_blake3`; its LE byte-pair packing equals this
/// affine code expression numerically). Dead rows' phantom scales are excluded by the
/// liveness factor.
pub fn ctl_block_scales_looked_input_quant<F: Field>(device: Device) -> Vec<TableWithColumns<F>> {
    let m = &input_quant_column_map(device);
    [
        (Column::single(m.schedule.block_index), &m.sides[0].scale, m.schedule.a_live),
        (
            Column::single_plus_public_input(m.schedule.block_index, B_BLOCK_KEY_OFFSET_PUBLIC_INPUT),
            &m.sides[1].scale,
            m.schedule.b_live,
        ),
    ]
    .map(|(key, scale, live)| {
        TableWithColumns::new(
            Table::InputQuant.into(),
            vec![key, bf16_code_expr(scale.sign, scale.exp, scale.mantissa)],
            Filter::new(
                vec![(Column::single(m.schedule.is_block_start), Column::single(live))],
                vec![],
            ),
        )
    })
    .to_vec()
}

/// InputQuant's side of the fp8 operand-code channel to Matmul. Each tuple contains an
/// element key (A: `ELEM_IDX`, B: `ELEM_IDX + h*k` — disjoint key spaces), two consecutive
/// noised fp8 codes packed into one field element, and the same two elements' summand
/// scores `LAMBDA` (jackpot check 3) as separate slots.
///
/// H100 exports QCAST's `code_noised`; B200 exports DOWNSCALEREWRITE's `output_code`, after
/// the conditional byte rewrite. Those lookups prove each packed component is a byte, and
/// Matmul's alignment lookup binds the same bytes there. The lambda scores still describe
/// the clean-data `alpha*X` and sigma, independently of that rewrite. Matmul reuses each A
/// element in `w` output cells and each B element in `h`
/// output cells, so the CTL filter carries the multiplicity — the public-input term
/// `w * IS_EVEN_ROW` (resp. `h * IS_EVEN_ROW`) times the liveness column — instead of
/// physically repeating each tuple.
pub fn ctl_operand_codes_looked_input_quant<F: Field>(device: Device) -> Vec<TableWithColumns<F>> {
    let m = &input_quant_column_map(device);
    [
        (
            Column::single(m.schedule.element_index),
            m.downscale.map_or(m.sides[0].code_noised, |(_, sides)| sides[0].output_code),
            m.sides[0].lambda,
            m.schedule.a_live,
            OPERAND_MULT_A_PUBLIC_INPUT,
        ),
        (
            b_element_key(),
            m.downscale.map_or(m.sides[1].code_noised, |(_, sides)| sides[1].output_code),
            m.sides[1].lambda,
            m.schedule.b_live,
            OPERAND_MULT_B_PUBLIC_INPUT,
        ),
    ]
    .map(|(key, col, lambda, live, mult)| {
        TableWithColumns::new(
            Table::InputQuant.into(),
            vec![key, byte_pair_expr(col), Column::single(lambda), next_row_expr(lambda)],
            Filter::new(
                vec![(
                    Column::single_times_public_input(m.schedule.is_even_row, mult),
                    Column::single(live),
                )],
                vec![],
            ),
        )
    })
    .to_vec()
}

/// InputQuant's looking side of the group-tuples channel into ScaleStark: the common 13-component
/// aggregate per matrix row, sent from *live* group-final rows (filter `IS_GROUP_FINAL *
/// LIVE`; dead groups have no ScaleStark counterpart). The group-final key `(g+1)*k - 1` is
/// ScaleStark's group key — `ELEM_IDX` for A, `ELEM_IDX + h*k` (public-input offset) for B.
/// The last two slots bind the group's committed `SIGMA_ENC` and `SIGMA_NORM` to ScaleStark's
/// exact sigma encoding and normalized significand. B200 appends the strip threshold,
/// Gaussian LOG16 region, key base, and kernel sum.
pub fn ctl_group_tuples_looking_input_quant<F: Field>(device: Device) -> Vec<TableWithColumns<F>> {
    let m = &input_quant_column_map(device);
    m.sides
        .iter()
        .enumerate()
        .map(|(index, side)| {
            let mut columns = vec![if index == 0 {
                Column::single(m.schedule.element_index)
            } else {
                b_element_key()
            }];
            columns.extend(Column::singles([
                side.block_l2.running_l2_frame_sum,
                side.block_l2.frame_doubled_scale_exponent,
                side.max_abs,
                side.alpha_exp,
                side.alpha_mantissa,
                side.beta_exp,
                side.beta_mantissa,
                side.beta_exp_is_zero,
                side.dead_bound,
                side.dead_count,
                side.sigma_enc,
                side.sigma_norm,
            ]));
            if device == Device::B200 {
                let map = &super::columns::B200_INPUT_QUANT_COL_MAP;
                let security = [&map.a, &map.b][index];
                columns.extend(Column::singles([
                    security.downscale.strip_threshold,
                    security.security_region,
                    security.security_key_base,
                    security.security_sum,
                ]));
            }
            let live = [m.schedule.a_live, m.schedule.b_live][index];
            TableWithColumns::new(
                Table::InputQuant.into(),
                columns,
                Filter::new(
                    vec![(Column::single(m.schedule.is_group_final), Column::single(live))],
                    vec![],
                ),
            )
        })
        .collect()
}

// ==================================================================================================
// Committed LUT instance inventory — the contract for the oracle wiring
// ==================================================================================================

/// The per-side column indices a symmetric lookup instance needs.
struct SideCols {
    int8_byte: usize,
    int_sign: usize,
    int_exp: usize,
    int_mantissa: usize,
    int_exp_is_zero: usize,
    scale_exp: usize,
    scale_exp_is_zero: usize,
    dec_sig: usize,
    dec_cut: usize,
    dec_wa: usize,
    x_exp: usize,
    x_mantissa: usize,
    x_exp_is_zero: usize,
    x_is_zero: usize,
    blk: BlockL2Columns<usize>,
    max_abs: usize,
    alpha_exp: usize,
    beta_exp: usize,
    beta_exp_is_zero: usize,
    scaled_decoded_significand: usize,
    is_group_final: usize,
    live: usize,
    dead_bound: usize,
    is_dead: usize,
    x_width: usize,
    x_pow: usize,
    exponent_gap: usize,
    gap_is_far: usize,
    near_gap: usize,
    near_gap_pow: usize,
    half_remainder: usize,
    remainder: usize,
    sum_of_squares_top: usize,
    sum_of_squares_rest_lo: usize,
    log_fraction: usize,
}

fn side_cols(side: &InputQuantSideColumns<usize>, schedule: &InputQuantScheduleColumns<usize>, live: usize) -> SideCols {
    SideCols {
        int8_byte: side.int8_byte,
        int_sign: side.int_bf16.sign,
        int_exp: side.int_bf16.exp,
        int_mantissa: side.int_bf16.mantissa,
        int_exp_is_zero: side.int_bf16.exp_is_zero,
        scale_exp: side.scale.exp,
        scale_exp_is_zero: side.scale.exp_is_zero,
        dec_sig: side.decode_multiply.sig_product,
        dec_cut: side.decode_multiply.cut_depth,
        dec_wa: side.decode_multiply.width_adjust,
        x_exp: side.x_exp,
        x_mantissa: side.x_mantissa,
        x_exp_is_zero: side.x_exp_is_zero,
        x_is_zero: side.x_is_zero,
        blk: side.block_l2,
        max_abs: side.max_abs,
        alpha_exp: side.alpha_exp,
        beta_exp: side.beta_exp,
        beta_exp_is_zero: side.beta_exp_is_zero,
        scaled_decoded_significand: side.scaled_decoded_significand,
        is_group_final: schedule.is_group_final,
        live,
        dead_bound: side.dead_bound,
        is_dead: side.is_dead,
        x_width: side.x_width,
        x_pow: side.x_pow,
        exponent_gap: side.exponent_gap,
        gap_is_far: side.gap_is_far,
        near_gap: side.near_gap,
        near_gap_pow: side.near_gap_pow,
        half_remainder: side.half_remainder,
        remainder: side.remainder,
        sum_of_squares_top: side.sum_of_squares_top,
        sum_of_squares_rest_lo: side.sum_of_squares_rest_lo,
        log_fraction: side.log_fraction,
    }
}

/// The RNERND key `SIG + 2^17*CUT` with the standard value tuple.
fn rnernd_lookup<F: Field>(
    sig: usize,
    cut: usize,
    mantissa: usize,
    wa: usize,
    is_zero: usize,
    exp_is_zero: usize,
) -> LutLookup<F> {
    LutLookup {
        table: LutTable::RneRnd,
        keys: vec![Column::linear_combination([
            (sig, F::ONE),
            (cut, F::from_canonical_u64(1 << 17)),
        ])],
        values: Column::singles([mantissa, wa, is_zero, exp_is_zero]).collect(),
        filter: Filter::default(),
    }
}

/// One side's LUT instances (everything except the cross-side PAIR128s and the QCASTs,
/// appended by [`input_quant_lut_lookups`]).
fn side_lut_lookups<F: Field>(s: &SideCols) -> Vec<LutLookup<F>> {
    let one = F::ONE;
    let neg = -F::ONE;
    let c254 = F::from_canonical_u64(254);
    let not_group_final = Filter::from_column(Column::linear_combination_with_constant([(s.is_group_final, neg)], one));
    // Exactly one shift-limb selector is active for a nonzero, block-final, non-far block.
    // Their sum is therefore the filter for every lookup used by that block's limb split.
    let nl_filter = || {
        Filter::new(
            vec![],
            vec![Column::linear_combination(s.blk.shift_limb_selector.map(|q| (q, one)))],
        )
    };
    // B11's shift remainder is
    // r = frame_doubled_scale_exponent - block_doubled_scale_exponent
    //     - 16*(selector_1 + 2*selector_2 + 3*selector_3).
    // It is affine, and the paired POW2D domains force r into [0, 16] on near-live rows.
    let r_hat = [
        (s.blk.frame_doubled_scale_exponent, one),
        (s.blk.block_doubled_scale_exponent, neg),
        (s.blk.shift_limb_selector[1], -F::from_canonical_u64(16)),
        (s.blk.shift_limb_selector[2], -F::from_canonical_u64(32)),
        (s.blk.shift_limb_selector[3], -F::from_canonical_u64(48)),
    ];
    let r_hat_complement = [
        (s.blk.frame_doubled_scale_exponent, neg),
        (s.blk.block_doubled_scale_exponent, one),
        (s.blk.shift_limb_selector[1], F::from_canonical_u64(16)),
        (s.blk.shift_limb_selector[2], F::from_canonical_u64(32)),
        (s.blk.shift_limb_selector[3], F::from_canonical_u64(48)),
    ];

    vec![
        // ---- S1: alpha positive-normal (RC16 x2): ALPHA_EXP - 1 and 254 - ALPHA_EXP. ----
        LutLookup::rc16(Column::linear_combination_with_constant([(s.alpha_exp, one)], neg)),
        LutLookup::rc16(Column::linear_combination_with_constant([(s.alpha_exp, neg)], c254)),
        LutLookup::rc16(Column::linear_combination_with_constant([(s.x_exp, neg)], c254)),
        // ---- F3: two range checks turn F2's either/or equation into a true running maximum.
        // `MAX_ABS' - ABS(X') >= 0` proves the updated value covers the new element.
        // `MAX_ABS' - MAX_ABS >= 0`, filtered to in-group transitions, proves it never drops.
        // At a group boundary F1 intentionally resets the maximum. ----
        LutLookup::rc16(Column::linear_combination_and_next_row_with_constant(
            vec![],
            vec![(s.max_abs, one), (s.x_exp, -F::from_canonical_u64(128)), (s.x_mantissa, neg)],
            F::ZERO,
        )),
        LutLookup::rc16_filtered(
            Column::linear_combination_and_next_row_with_constant(vec![(s.max_abs, neg)], vec![(s.max_abs, one)], F::ZERO),
            not_group_final.clone(),
        ),
        // ---- C1: the liveness certificates. IS_DEAD is boolean and zero on phantom rows
        // (stark.rs C1), so on live rows exactly one of the pair fires and pins
        // IS_DEAD = [ABS(X) >= DEAD_BOUND], with ABS(X) = 128*X_EXP + X_MANTISSA.
        // Dead: ABS(X) - DEAD_BOUND >= 0 (filter IS_DEAD).
        // Alive: DEAD_BOUND - 1 - ABS(X) >= 0 (filter (1 - IS_DEAD)*LIVE).
        // Both keys stay below 2^16: ABS <= 0x7F7F and DEAD_BOUND <= 0x7F7F + 256. ----
        LutLookup::rc16_filtered(
            Column::linear_combination([
                (s.x_exp, F::from_canonical_u64(128)),
                (s.x_mantissa, one),
                (s.dead_bound, neg),
            ]),
            Filter::from_column(Column::single(s.is_dead)),
        ),
        LutLookup::rc16_filtered(
            Column::linear_combination_with_constant(
                [
                    (s.dead_bound, one),
                    (s.x_exp, -F::from_canonical_u64(128)),
                    (s.x_mantissa, neg),
                ],
                neg,
            ),
            Filter::new(
                vec![(
                    Column::linear_combination_with_constant([(s.is_dead, neg)], one),
                    Column::single(s.live),
                )],
                vec![],
            ),
        ),
        // ---- P3/S2: EXPINFO on the scale and beta exponents (finiteness + the flag). ----
        LutLookup {
            table: LutTable::ExpInfo,
            keys: vec![Column::single(s.scale_exp)],
            values: vec![Column::single(s.scale_exp_is_zero)],
            filter: Filter::default(),
        },
        LutLookup {
            table: LutTable::ExpInfo,
            keys: vec![Column::single(s.beta_exp)],
            values: vec![Column::single(s.beta_exp_is_zero)],
            filter: Filter::default(),
        },
        // ---- P1: INT8DEC (key domain = the byte range proof). ----
        LutLookup {
            table: LutTable::Int8Dec,
            keys: vec![Column::single(s.int8_byte)],
            values: Column::singles([s.int_sign, s.int_exp, s.int_mantissa, s.int_exp_is_zero]).collect(),
            filter: Filter::default(),
        },
        // ---- U2: decode cut depth — CLAMP22(535 - E*(INT) - E*(SCALE)). ----
        LutLookup {
            table: LutTable::Clamp22,
            keys: vec![Column::linear_combination_with_constant(
                [
                    (s.int_exp, neg),
                    (s.int_exp_is_zero, neg),
                    (s.scale_exp, neg),
                    (s.scale_exp_is_zero, neg),
                ],
                F::from_canonical_u64(535),
            )],
            values: vec![Column::single(s.dec_cut)],
            filter: Filter::default(),
        },
        // ---- U3: decode RNERND — the outputs are the X fields. ----
        rnernd_lookup(s.dec_sig, s.dec_cut, s.x_mantissa, s.dec_wa, s.x_is_zero, s.x_exp_is_zero),
        // ---- B6: dominance and in-group monotonicity pin
        // `running_max_doubled_scale_exponent` to the running maximum of
        // `block_doubled_scale_exponent`. B7 copies its group-final value into
        // `frame_doubled_scale_exponent`; B8/B11 prove the resulting frame shift is nonnegative
        // (or at least 54 on `is_far_shift` rows). ----
        LutLookup::rc16(Column::linear_combination_and_next_row_with_constant(
            vec![],
            vec![
                (s.blk.running_max_doubled_scale_exponent, one),
                (s.blk.block_doubled_scale_exponent, neg),
            ],
            F::ZERO,
        )),
        LutLookup::rc16_filtered(
            Column::linear_combination_and_next_row_with_constant(
                vec![(s.blk.running_max_doubled_scale_exponent, neg)],
                vec![(s.blk.running_max_doubled_scale_exponent, one)],
                F::ZERO,
            ),
            not_group_final.clone(),
        ),
        // ---- B8: on `is_far_shift`, RC16 checks
        // frame_doubled_scale_exponent - block_doubled_scale_exponent - 54.
        // B10 caps the scaled block product below 2^54, so this branch's floor term is exactly
        // zero rather than a dropped contribution. ----
        LutLookup::rc16_filtered(
            Column::linear_combination_with_constant(
                [
                    (s.blk.frame_doubled_scale_exponent, one),
                    (s.blk.block_doubled_scale_exponent, neg),
                ],
                -F::from_canonical_u64(54),
            ),
            Filter::from_column(Column::single(s.blk.is_far_shift)),
        ),
        // ---- B10: range-check the four `scaled_block_product_limbs` and cap the top limb below
        // 2^6. This proves the dividend is below 2^54, giving a unique integer decomposition,
        // the far threshold's premise, and B16's no-alias bound. ----
        LutLookup::rc16(Column::single(s.blk.scaled_block_product_limbs[0])),
        LutLookup::rc16(Column::single(s.blk.scaled_block_product_limbs[1])),
        LutLookup::rc16(Column::single(s.blk.scaled_block_product_limbs[2])),
        LutLookup::rc16(Column::single(s.blk.scaled_block_product_limbs[3])),
        LutLookup::rc16(Column::linear_combination([(
            s.blk.scaled_block_product_limbs[3],
            F::from_canonical_u64(1 << 10),
        )])),
        // ---- B11: on an active block split, bind `shift_remainder_power = 2^r` and
        // `shift_complement_power = 2^(16-r)`. The two POW2D domains force `0 <= r <= 16`.
        // Other rows do not consume these power columns. ----
        LutLookup {
            table: LutTable::Pow2D,
            keys: vec![Column::linear_combination(r_hat)],
            values: vec![Column::single(s.blk.shift_remainder_power)],
            filter: nl_filter(),
        },
        LutLookup {
            table: LutTable::Pow2D,
            keys: vec![Column::linear_combination_with_constant(
                r_hat_complement,
                F::from_canonical_u64(16),
            )],
            values: vec![Column::single(s.blk.shift_complement_power)],
            filter: nl_filter(),
        },
        // ---- B15: range-check the selected-limb quotient and remainder, then prove
        // `selected_limb_remainder < shift_remainder_power`. Together with B14, this makes the
        // division by 2^r unique. ----
        LutLookup::rc16(Column::single(s.blk.selected_limb_quotient)),
        LutLookup::rc16(Column::single(s.blk.selected_limb_remainder)),
        LutLookup::rc16_filtered(
            Column::linear_combination_with_constant(
                [(s.blk.shift_remainder_power, one), (s.blk.selected_limb_remainder, neg)],
                neg,
            ),
            nl_filter(),
        ),
        // ---- V1: WIDTH16 binds X_WIDTH = bit_length(M(alpha)*M(X)) and pins the
        // product-nonzero flag to 1 - X_IS_ZERO (alpha's significand is at least 128, so the
        // product is zero exactly when X is; W1 pins the key to the committed factors). ----
        LutLookup {
            table: LutTable::Width16,
            keys: vec![Column::single(s.scaled_decoded_significand)],
            values: vec![
                Column::single(s.x_width),
                Column::linear_combination_with_constant([(s.x_is_zero, neg)], one),
            ],
            filter: Filter::default(),
        },
        // ---- V4: X_POW = 2^(16 - X_WIDTH), the significand normalizer (key domain [0, 19]
        // covers every width in [0, 16]). ----
        LutLookup {
            table: LutTable::Pow2D,
            keys: vec![Column::linear_combination_with_constant(
                [(s.x_width, neg)],
                F::from_canonical_u64(16),
            )],
            values: vec![Column::single(s.x_pow)],
            filter: Filter::default(),
        },
        // ---- V5: RC16(EXPONENT_GAP) forces the dominance bit: the wrongly signed gap wraps
        // far past 2^16. ----
        LutLookup::rc16(Column::single(s.exponent_gap)),
        // ---- V6: the far-cut certificates — EXPONENT_GAP - 16 >= 0 under GAP_IS_FAR, and
        // 15 - EXPONENT_GAP >= 0 under its complement. ----
        LutLookup::rc16_filtered(
            Column::linear_combination_with_constant([(s.exponent_gap, one)], -F::from_canonical_u64(16)),
            Filter::from_column(Column::single(s.gap_is_far)),
        ),
        LutLookup::rc16_filtered(
            Column::linear_combination_with_constant([(s.exponent_gap, neg)], F::from_canonical_u64(15)),
            Filter::from_column(Column::linear_combination_with_constant([(s.gap_is_far, neg)], one)),
        ),
        // ---- V7: NEAR_GAP_POW = 2^NEAR_GAP, the shared divisor of both floor stages. ----
        LutLookup {
            table: LutTable::Pow2D,
            keys: vec![Column::single(s.near_gap)],
            values: vec![Column::single(s.near_gap_pow)],
            filter: Filter::default(),
        },
        // ---- V9: both stage remainders lie in [0, NEAR_GAP_POW): each pair RC16(R) and
        // RC16(NEAR_GAP_POW - 1 - R) makes its floor division unique. ----
        LutLookup::rc16(Column::single(s.half_remainder)),
        LutLookup::rc16(Column::linear_combination_with_constant(
            [(s.near_gap_pow, one), (s.half_remainder, neg)],
            neg,
        )),
        LutLookup::rc16(Column::single(s.remainder)),
        LutLookup::rc16(Column::linear_combination_with_constant(
            [(s.near_gap_pow, one), (s.remainder, neg)],
            neg,
        )),
        // ---- V10: the sum-of-squares rest's low limb, and the fixed-point log of the top
        // slice — LOG16's logarithm region 0 bounds SUM_OF_SQUARES_TOP below 2^16 and serves
        // LOG_FRACTION = floor(64 * log2 SUM_OF_SQUARES_TOP) (0 at the phantom key 0). ----
        LutLookup::rc16(Column::single(s.sum_of_squares_rest_lo)),
        LutLookup {
            table: LutTable::Log16,
            keys: vec![Column::constant(F::ZERO), Column::single(s.sum_of_squares_top)],
            values: vec![Column::single(s.log_fraction)],
            filter: Filter::default(),
        },
    ]
}

/// Complete per-row inventory: 182 H100 instances (including group G's five noise-decode
/// bindings); B200 adds eight downscale instances and two Gaussian-kernel instances.
pub fn input_quant_lut_lookups<F: Field>(device: Device) -> Vec<LutLookup<F>> {
    let map = input_quant_column_map(device);
    let mut lookups = Vec::new();
    for (side, live) in map.sides.iter().zip([map.schedule.a_live, map.schedule.b_live]) {
        lookups.extend(side_lut_lookups(&side_cols(side, &map.schedule, live)));
        lookups.extend(crate::v4::circuit::fp32::alignment_lookups(&side.alignment));
        lookups.extend(crate::v4::circuit::fp32::rounding_lookups(
            &side.rounding,
            side.alignment.binary_scale,
        ));
        let half = F::TWO.inverse();
        lookups.push(LutLookup::rc16(Column::linear_combination([
            (side.rounding.output.fraction_high, half),
            (side.proxy.high_fraction_parity, -half),
        ])));
        lookups.push(LutLookup {
            table: LutTable::Qcast,
            keys: vec![Column::single(side.proxy.code)],
            values: vec![Column::single(side.code_noised)],
            filter: Filter::default(),
        });
    }
    // ---- G1: the noise decode's exponent (EXPINFO: `[0, 254]` and the zero flag) and the
    // fraction split `FRACTION = 2^16*HIGH + LOW` (RC16 on LOW here, PAIR128 on both HIGHs
    // below), unfiltered so the decode of the known word is unique on phantom rows too. ----
    for (noise, high) in [
        (&map.schedule.noise_a, map.schedule.noise_fraction_high_a),
        (&map.schedule.noise_b, map.schedule.noise_fraction_high_b),
    ] {
        lookups.push(LutLookup {
            table: LutTable::ExpInfo,
            keys: vec![Column::single(noise.exponent)],
            values: vec![Column::single(noise.is_exponent_zero)],
            filter: Filter::default(),
        });
        lookups.push(LutLookup::rc16(Column::linear_combination([
            (noise.fraction, F::ONE),
            (high, -F::from_canonical_u64(1 << 16)),
        ])));
    }
    for (first, second) in [
        (map.sides[0].scale.mantissa, map.sides[1].scale.mantissa),
        (map.sides[0].alpha_mantissa, map.sides[1].alpha_mantissa),
        (map.sides[0].beta_mantissa, map.sides[1].beta_mantissa),
        (map.schedule.noise_fraction_high_a, map.schedule.noise_fraction_high_b),
    ] {
        lookups.push(LutLookup {
            table: LutTable::Pair128,
            keys: Column::singles([first, second]).collect(),
            values: vec![],
            filter: Filter::default(),
        });
    }
    if let Some((start, downscale)) = map.downscale {
        let start_filter = Filter::from_column(Column::single(start));
        lookups.push(LutLookup {
            table: LutTable::Pair128,
            keys: downscale.iter().map(|side| Column::single(side.coin_field)).collect(),
            values: vec![],
            filter: start_filter.clone(),
        });
        lookups.push(LutLookup {
            table: LutTable::Bytes2,
            keys: map
                .sides
                .iter()
                .zip(downscale)
                .map(|(side, down)| {
                    Column::linear_combination([
                        (side.rounding.output.fraction_low, F::ONE),
                        (down.fraction_bit_15, -F::from_canonical_u64(32768)),
                        (down.coin_field, -F::from_canonical_u64(256)),
                    ])
                })
                .collect(),
            values: vec![],
            filter: start_filter,
        });
        for ((side, down), live) in map
            .sides
            .iter()
            .zip(downscale)
            .zip([map.schedule.a_live, map.schedule.b_live])
        {
            lookups.push(LutLookup::rc16_filtered(
                Column::linear_combination_with_constant([(down.coin_field, -F::ONE)], F::from_canonical_u64(111)),
                Filter::new(vec![(Column::single(start), Column::single(down.coin))], vec![]),
            ));
            lookups.push(LutLookup::rc16_filtered(
                Column::linear_combination_with_constant([(down.coin_field, F::ONE)], -F::from_canonical_u64(112)),
                Filter::new(
                    vec![(
                        Column::single(start),
                        Column::linear_combination_with_constant([(down.coin, -F::ONE)], F::ONE),
                    )],
                    vec![],
                ),
            ));
            // Quantize-first rewrite: pack (c,t,b,e) into one mixed-radix key
            //
            //   key = c + 2^8*t + 2^15*b + 2^16*e,
            //   c = code_noised in [0,256), t = strip_threshold in [0,128),
            //   b = coin in {0,1}, e = sigma_enc-269 in 0..=5.
            //
            // QCAST bounds c, Scale's group tuple/threshold LUT binds t and e, and the
            // coin extraction above plus evaluate_downscale bind b. The fields occupy
            // disjoint bit ranges: code 0..7, threshold 8..14, coin 15. The exponent
            // selects one of six folded slots, each offset by 65536, rather than a
            // separate online table. Since sigma_enc=floor(log2(s/2))+268 on B200,
            // e=floor(log2(s))-2 is precisely the replacement E4M3 exponent field.
            //
            // The table returns ((c&128)|(8*e)|(c&7)|1) when b && (c&127)<t,
            // and c otherwise. Binding output_code here authenticates the final byte
            // exported to Matmul. Only live elements participate in this relation.
            lookups.push(LutLookup {
                table: LutTable::DownscaleRewrite,
                keys: vec![Column::linear_combination_with_constant(
                    [
                        (side.code_noised, F::ONE),
                        (down.strip_threshold, F::from_canonical_u64(256)),
                        (down.coin, F::from_canonical_u64(32768)),
                        (side.sigma_enc, F::from_canonical_u64(65536)),
                    ],
                    -F::from_canonical_u64(65536 * 269),
                )],
                values: vec![Column::single(down.output_code)],
                filter: Filter::from_column(Column::single(live)),
            });
        }
    }
    // ---- Check 4: one Q16 Gaussian contribution per live B200 element. ----
    // LOG16 is shared: region 0 serves the score logarithm; region=1+class serves
    // min(T(d),cap_class), where
    //
    //   base = lambda(alpha*X=0, sigma),   d = lambda(alpha*X,sigma)-base,
    //   T(d) = floor(65536*exp(-(2^((50*d+51)/3200)-1)/2)).
    //
    // d measures the clean signal relative to that row's noise. The 51/50 score slack
    // bounds the score approximation conservatively; the kernel is zero for d>=290.
    // Scale's DOWNSCALETHRESHOLD lookup fixes the region and base from the exact row RMS,
    // and the group CTL binds both here. Keying (region,d) as two components is essential:
    // a negative d or d>=8192 has no row in this region and cannot borrow a neighboring
    // region's entry. InputQuant sums gaussian_value over the group and exports the total;
    // Scale multiplies it by the row coefficient and proves the mean proxy >=1/25.
    if device == Device::B200 {
        let m = &super::columns::B200_INPUT_QUANT_COL_MAP;
        for (side, live) in [(&m.a, m.schedule.a_live), (&m.b, m.schedule.b_live)] {
            lookups.push(LutLookup {
                table: LutTable::Log16,
                keys: vec![
                    Column::single(side.security_region),
                    Column::linear_combination([(side.common.lambda, F::ONE), (side.security_key_base, -F::ONE)]),
                ],
                values: vec![Column::single(side.gaussian_value)],
                filter: Filter::from_column(Column::single(live)),
            });
        }
    }
    lookups
}

pub(crate) fn coefficient_looking<F: Field>(device: Device) -> Vec<TableWithColumns<F>> {
    let map = input_quant_column_map(device);
    map.sides
        .iter()
        .zip([map.schedule.a_live, map.schedule.b_live])
        .enumerate()
        .map(|(index, (side, live))| {
            let mut tuple = vec![if index == 0 {
                Column::single(map.schedule.block_index)
            } else {
                Column::single_plus_public_input(map.schedule.block_index, B_BLOCK_KEY_OFFSET_PUBLIC_INPUT)
            }];
            tuple.extend(Column::singles([
                side.alpha_exp,
                side.alpha_mantissa,
                side.scale.exp,
                side.scale.mantissa,
                side.scale.exp_is_zero,
                side.coefficient.exponent,
                side.coefficient.fraction_low,
                side.coefficient.fraction_high,
                side.coefficient.is_exponent_zero,
            ]));
            TableWithColumns::new(
                Table::InputQuant.into(),
                tuple,
                Filter::new(
                    vec![(Column::single(map.schedule.is_block_start), Column::single(live))],
                    vec![],
                ),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use plonky2::field::goldilocks_field::GoldilocksField as F;

    #[test]
    fn lookup_shapes_and_counts_match_each_device() {
        for device in [Device::H100, Device::B200] {
            let lookups = input_quant_lut_lookups::<F>(device);
            let count = |table| lookups.iter().filter(|lookup| lookup.table == table).count();
            let downscales = device == Device::B200;
            assert_eq!(lookups.len(), if downscales { 192 } else { 182 });
            assert_eq!(count(LutTable::Range16), if downscales { 142 } else { 138 });
            assert_eq!(count(LutTable::Pair128), if downscales { 7 } else { 6 });
            for (table, total) in [
                (LutTable::ExpInfo, 10),
                (LutTable::Clamp22, 2),
                (LutTable::Int8Dec, 2),
                (LutTable::Pow2D, 8),
                (LutTable::RneRnd, 2),
                (LutTable::Width16, 2),
                (LutTable::Log16, if downscales { 4 } else { 2 }),
                (LutTable::Qcast, 2),
                (LutTable::Fp32Tools, 8),
            ] {
                assert_eq!(count(table), total, "{device:?} {table:?}");
            }
            assert_eq!(ctl_int8_bytes_looked_input_quant::<F>(device).len(), 2);
            assert_eq!(ctl_block_scales_looked_input_quant::<F>(device).len(), 2);
            assert_eq!(ctl_operand_codes_looked_input_quant::<F>(device).len(), 2);
            assert_eq!(ctl_group_tuples_looking_input_quant::<F>(device).len(), 2);
            assert_eq!(coefficient_looking::<F>(device).len(), 2);
        }
    }
}
