//! Cross-table and committed lookup relations used by InputQuantStark.
//!
//! A cross-table lookup (CTL) proves multiset equality between this trace and another STARK.
//! This file defines InputQuant's half of four channels: committed int8 bytes and block scales
//! from Blake3Stark, fp8 operand codes to MatmulB200Stark, and completed row-group tuples to
//! ScaleStark. Their counterpart halves are assembled by
//! `super::super::ctl::all_cross_table_lookups`.
//!
//! The LUT descriptors range-check values and bind decode, rounding, and quantization table
//! outputs against the batch's committed lookup-table oracle. `super::stark` fills the witness
//! columns; the batch assembler includes these descriptors in the complete proof. Constraint
//! labels such as P1, U2, and W11 refer to `super::stark`'s documented constraint groups.

use plonky2::field::types::Field;
use starky::cross_table_lookup::TableWithColumns;
use starky::lookup::{Column, Filter};

use super::super::ctl::{LutLookup, LutTable, Table};
use super::columns::{
    B_BLOCK_KEY_OFFSET_PUBLIC_INPUT, B_KEY_OFFSET_PUBLIC_INPUT, BlockL2Columns, INPUT_QUANT_COL_MAP, OPERAND_MULT_A_PUBLIC_INPUT,
    OPERAND_MULT_B_PUBLIC_INPUT,
};

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
        vec![(Column::single(INPUT_QUANT_COL_MAP.is_even_row), Column::single(live))],
        vec![],
    )
}

/// The B slots' element key `ELEM_IDX + h*k`, with `h*k` read from the public input.
fn b_element_key<F: Field>() -> Column<F> {
    Column::single_plus_public_input(INPUT_QUANT_COL_MAP.element_index, B_KEY_OFFSET_PUBLIC_INPUT)
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
pub fn ctl_int8_bytes_looked_input_quant<F: Field>() -> Vec<TableWithColumns<F>> {
    let m = &INPUT_QUANT_COL_MAP;
    [
        (Column::single(m.element_index), m.int8_byte_a, m.a_live),
        (b_element_key(), m.int8_byte_b, m.b_live),
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
pub fn ctl_block_scales_looked_input_quant<F: Field>() -> Vec<TableWithColumns<F>> {
    let m = &INPUT_QUANT_COL_MAP;
    [
        (Column::single(m.block_index), &m.scale_a, m.a_live),
        (
            Column::single_plus_public_input(m.block_index, B_BLOCK_KEY_OFFSET_PUBLIC_INPUT),
            &m.scale_b,
            m.b_live,
        ),
    ]
    .map(|(key, scale, live)| {
        TableWithColumns::new(
            Table::InputQuant.into(),
            vec![key, bf16_code_expr(scale.sign, scale.exp, scale.mantissa)],
            Filter::new(vec![(Column::single(m.is_block_start), Column::single(live))], vec![]),
        )
    })
    .to_vec()
}

/// InputQuant's side of the fp8 operand-code channel to MatmulB200Stark. Each tuple contains an
/// element key (A: `ELEM_IDX`, B: `ELEM_IDX + h*k` — disjoint key spaces), two consecutive
/// noised fp8 codes packed into one field element, and the same two elements' summand
/// scores `LAMBDA` (jackpot check 4) as separate slots.
///
/// QCAST proves each packed component is a byte here, and Matmul's alignment lookup binds the
/// same bytes there. Matmul reuses each A element in `w` output cells and each B element in `h`
/// output cells, so the CTL filter carries the multiplicity — the public-input term
/// `w * IS_EVEN_ROW` (resp. `h * IS_EVEN_ROW`) times the liveness column — instead of
/// physically repeating each tuple.
pub fn ctl_operand_codes_looked_input_quant<F: Field>() -> Vec<TableWithColumns<F>> {
    let m = &INPUT_QUANT_COL_MAP;
    [
        (
            Column::single(m.element_index),
            m.code_noised_a,
            m.lambda_a,
            m.a_live,
            OPERAND_MULT_A_PUBLIC_INPUT,
        ),
        (
            b_element_key(),
            m.code_noised_b,
            m.lambda_b,
            m.b_live,
            OPERAND_MULT_B_PUBLIC_INPUT,
        ),
    ]
    .map(|(key, col, lambda, live, mult)| {
        TableWithColumns::new(
            Table::InputQuant.into(),
            vec![key, byte_pair_expr(col), Column::single(lambda), next_row_expr(lambda)],
            Filter::new(
                vec![(Column::single_times_public_input(m.is_even_row, mult), Column::single(live))],
                vec![],
            ),
        )
    })
    .to_vec()
}

/// InputQuant's looking side of the group-tuples channel into ScaleStark: the 13-component
/// aggregate per matrix row, sent from *live* group-final rows (filter `IS_GROUP_FINAL *
/// LIVE`; dead groups have no ScaleStark counterpart). The group-final key `(g+1)*k - 1` is
/// ScaleStark's group key — `ELEM_IDX` for A, `ELEM_IDX + h*k` (public-input offset) for B.
/// The last two slots bind the group's committed `SIGMA_ENC` and `SIGMA_NORM` to ScaleStark's
/// exact sigma encoding and normalized significand.
pub fn ctl_group_tuples_looking_input_quant<F: Field>() -> Vec<TableWithColumns<F>> {
    let m = &INPUT_QUANT_COL_MAP;
    let gf_live = |live: usize| Filter::new(vec![(Column::single(m.is_group_final), Column::single(live))], vec![]);
    let a = TableWithColumns::new(
        Table::InputQuant.into(),
        vec![
            Column::single(m.element_index),
            Column::single(m.block_l2_a.running_l2_frame_sum),
            Column::single(m.block_l2_a.frame_doubled_scale_exponent),
            Column::single(m.max_abs_a),
            Column::single(m.alpha_a_exp),
            Column::single(m.alpha_a_mantissa),
            Column::single(m.beta_a_exp),
            Column::single(m.beta_a_mantissa),
            Column::single(m.beta_a_exp_is_zero),
            Column::single(m.dead_bound_a),
            Column::single(m.dead_count_a),
            Column::single(m.sigma_enc_a),
            Column::single(m.sigma_norm_a),
        ],
        gf_live(m.a_live),
    );
    let b = TableWithColumns::new(
        Table::InputQuant.into(),
        vec![
            b_element_key(),
            Column::single(m.block_l2_b.running_l2_frame_sum),
            Column::single(m.block_l2_b.frame_doubled_scale_exponent),
            Column::single(m.max_abs_b),
            Column::single(m.alpha_b_exp),
            Column::single(m.alpha_b_mantissa),
            Column::single(m.beta_b_exp),
            Column::single(m.beta_b_mantissa),
            Column::single(m.beta_b_exp_is_zero),
            Column::single(m.dead_bound_b),
            Column::single(m.dead_count_b),
            Column::single(m.sigma_enc_b),
            Column::single(m.sigma_norm_b),
        ],
        gf_live(m.b_live),
    );
    vec![a, b]
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
    beta_mul: MulCols,
    noise_exp: usize,
    noise_exp_is_zero: usize,
    fma: FmaCols,
    fma_sig_product: usize,
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

struct MulCols {
    sig: usize,
    cut: usize,
    out_exp: usize,
    out_mantissa: usize,
    wa: usize,
    out_is_zero: usize,
    out_exp_is_zero: usize,
}

struct FmaCols {
    scale_gap_slack: usize,
    far_gap_slack: usize,
    exp_gap_capped: usize,
    exp_gap_pow2: usize,
    is_wide: usize,
    compression_shift: usize,
    shift_pow2: usize,
    compression_quotient: usize,
    compression_quotient_lsb: usize,
    compression_remainder: usize,
    rounding_significand_key: usize,
    rounding_significand_key_high_bit: usize,
    key_scale: usize,
    cut_used: usize,
    out_exp: usize,
    out_mantissa: usize,
    wa: usize,
    out_is_zero: usize,
    out_exp_is_zero: usize,
}

fn mul_cols(m: &crate::circuit::fp8::input_quant_stark::columns::MulBlockView<usize>) -> MulCols {
    MulCols {
        sig: m.sig_product,
        cut: m.cut_depth,
        out_exp: m.out_exp,
        out_mantissa: m.out_mantissa,
        wa: m.width_adjust,
        out_is_zero: m.out_is_zero,
        out_exp_is_zero: m.out_exp_is_zero,
    }
}

fn fma_cols(f: &crate::circuit::fp8::input_quant_stark::columns::FmaBlockView<usize>) -> FmaCols {
    FmaCols {
        scale_gap_slack: f.scale_gap_slack,
        far_gap_slack: f.far_gap_slack,
        exp_gap_capped: f.exp_gap_capped,
        exp_gap_pow2: f.exp_gap_pow2,
        is_wide: f.is_wide,
        compression_shift: f.compression_shift,
        shift_pow2: f.shift_pow2,
        compression_quotient: f.compression_quotient,
        compression_quotient_lsb: f.compression_quotient_lsb,
        compression_remainder: f.compression_remainder,
        rounding_significand_key: f.rounding_significand_key,
        rounding_significand_key_high_bit: f.rounding_significand_key_high_bit,
        key_scale: f.key_scale,
        cut_used: f.cut_used,
        out_exp: f.out_exp,
        out_mantissa: f.out_mantissa,
        wa: f.width_adjust,
        out_is_zero: f.out_is_zero,
        out_exp_is_zero: f.out_exp_is_zero,
    }
}

fn side_cols_a() -> SideCols {
    let m = &INPUT_QUANT_COL_MAP;
    SideCols {
        int8_byte: m.int8_byte_a,
        int_sign: m.int_bf16_a.sign,
        int_exp: m.int_bf16_a.exp,
        int_mantissa: m.int_bf16_a.mantissa,
        int_exp_is_zero: m.int_bf16_a.exp_is_zero,
        scale_exp: m.scale_a.exp,
        scale_exp_is_zero: m.scale_a.exp_is_zero,
        dec_sig: m.decode_multiply_a.sig_product,
        dec_cut: m.decode_multiply_a.cut_depth,
        dec_wa: m.decode_multiply_a.width_adjust,
        x_exp: m.x_a_exp,
        x_mantissa: m.x_a_mantissa,
        x_exp_is_zero: m.x_a_exp_is_zero,
        x_is_zero: m.x_a_is_zero,
        blk: m.block_l2_a,
        max_abs: m.max_abs_a,
        alpha_exp: m.alpha_a_exp,
        beta_exp: m.beta_a_exp,
        beta_exp_is_zero: m.beta_a_exp_is_zero,
        beta_mul: mul_cols(&m.noise_scale_multiply_a),
        noise_exp: m.noise_a.exp,
        noise_exp_is_zero: m.noise_a.exp_is_zero,
        fma: fma_cols(&m.noised_value_fma_a),
        fma_sig_product: m.noised_fma_product_a,
        is_group_final: m.is_group_final,
        live: m.a_live,
        dead_bound: m.dead_bound_a,
        is_dead: m.is_dead_a,
        x_width: m.x_width_a,
        x_pow: m.x_pow_a,
        exponent_gap: m.exponent_gap_a,
        gap_is_far: m.gap_is_far_a,
        near_gap: m.near_gap_a,
        near_gap_pow: m.near_gap_pow_a,
        half_remainder: m.half_remainder_a,
        remainder: m.remainder_a,
        sum_of_squares_top: m.sum_of_squares_top_a,
        sum_of_squares_rest_lo: m.sum_of_squares_rest_lo_a,
        log_fraction: m.log_fraction_a,
    }
}

fn side_cols_b() -> SideCols {
    let m = &INPUT_QUANT_COL_MAP;
    SideCols {
        int8_byte: m.int8_byte_b,
        int_sign: m.int_bf16_b.sign,
        int_exp: m.int_bf16_b.exp,
        int_mantissa: m.int_bf16_b.mantissa,
        int_exp_is_zero: m.int_bf16_b.exp_is_zero,
        scale_exp: m.scale_b.exp,
        scale_exp_is_zero: m.scale_b.exp_is_zero,
        dec_sig: m.decode_multiply_b.sig_product,
        dec_cut: m.decode_multiply_b.cut_depth,
        dec_wa: m.decode_multiply_b.width_adjust,
        x_exp: m.x_b_exp,
        x_mantissa: m.x_b_mantissa,
        x_exp_is_zero: m.x_b_exp_is_zero,
        x_is_zero: m.x_b_is_zero,
        blk: m.block_l2_b,
        max_abs: m.max_abs_b,
        alpha_exp: m.alpha_b_exp,
        beta_exp: m.beta_b_exp,
        beta_exp_is_zero: m.beta_b_exp_is_zero,
        beta_mul: mul_cols(&m.noise_scale_multiply_b),
        noise_exp: m.noise_b.exp,
        noise_exp_is_zero: m.noise_b.exp_is_zero,
        fma: fma_cols(&m.noised_value_fma_b),
        fma_sig_product: m.noised_fma_product_b,
        is_group_final: m.is_group_final,
        live: m.b_live,
        dead_bound: m.dead_bound_b,
        is_dead: m.is_dead_b,
        x_width: m.x_width_b,
        x_pow: m.x_pow_b,
        exponent_gap: m.exponent_gap_b,
        gap_is_far: m.gap_is_far_b,
        near_gap: m.near_gap_b,
        near_gap_pow: m.near_gap_pow_b,
        half_remainder: m.half_remainder_b,
        remainder: m.remainder_b,
        sum_of_squares_top: m.sum_of_squares_top_b,
        sum_of_squares_rest_lo: m.sum_of_squares_rest_lo_b,
        log_fraction: m.log_fraction_b,
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
    let is_wide = Filter::from_column(Column::single(s.fma.is_wide));
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
        // ---- W2/W3: range-check the exact exponent-order slack and, on far rows, d - threshold. ----
        LutLookup::rc16(Column::single(s.fma.scale_gap_slack)),
        LutLookup::rc16(Column::single(s.fma.far_gap_slack)),
        // ---- W8/W9: prove the wide split has a 17-bit head, bounded remainder, and valid
        // quotient parity.
        LutLookup::rc16_filtered(
            Column::linear_combination_with_constant([(s.fma.compression_quotient, one)], -F::from_canonical_u64(1 << 16)),
            is_wide,
        ),
        LutLookup::rc16(Column::single(s.fma.compression_remainder)),
        LutLookup::rc16(Column::linear_combination_with_constant(
            [(s.fma.shift_pow2, one), (s.fma.compression_remainder, neg)],
            neg,
        )),
        // `(compression_quotient - compression_quotient_lsb)/2` is an integer below 2^16,
        // proving that the committed bit is the quotient's actual parity.
        LutLookup::rc16(Column::linear_combination([
            (s.fma.compression_quotient, F::TWO.inverse()),
            (s.fma.compression_quotient_lsb, -F::TWO.inverse()),
        ])),
        // ---- W10: prove ROUNDING_SIGNIFICAND_KEY < 2^17 from its high bit and low 16 bits.
        // This check is separate because the packed RNERND key could otherwise alias an
        // oversized significand into the next cut slot. ----
        LutLookup::rc16(Column::linear_combination([
            (s.fma.rounding_significand_key, one),
            (s.fma.rounding_significand_key_high_bit, -F::from_canonical_u64(1 << 16)),
        ])),
        // ---- U6/M6/W12: 254 - OUT_EXP overflow bans (RC16 x3 per side). ----
        LutLookup::rc16(Column::linear_combination_with_constant([(s.x_exp, neg)], c254)),
        LutLookup::rc16(Column::linear_combination_with_constant([(s.beta_mul.out_exp, neg)], c254)),
        LutLookup::rc16(Column::linear_combination_with_constant([(s.fma.out_exp, neg)], c254)),
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
        // ---- M2: beta-MUL cut depth — CLAMP22(535 - E*(BETA) - E*(NOISE)). ----
        LutLookup {
            table: LutTable::Clamp22,
            keys: vec![Column::linear_combination_with_constant(
                [
                    (s.beta_exp, neg),
                    (s.beta_exp_is_zero, neg),
                    (s.noise_exp, neg),
                    (s.noise_exp_is_zero, neg),
                ],
                F::from_canonical_u64(535),
            )],
            values: vec![Column::single(s.beta_mul.cut)],
            filter: Filter::default(),
        },
        // ---- M3: beta-MUL RNERND. ----
        rnernd_lookup(
            s.beta_mul.sig,
            s.beta_mul.cut,
            s.beta_mul.out_mantissa,
            s.beta_mul.wa,
            s.beta_mul.out_is_zero,
            s.beta_mul.out_exp_is_zero,
        ),
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
        // ---- W4/W8: bind 2^EXP_GAP_CAPPED and 2^COMPRESSION_SHIFT. POW2D's key domain
        // `[0, 19]` also range-checks both shifts; a narrow row uses shift zero and power one. ----
        LutLookup {
            table: LutTable::Pow2D,
            keys: vec![Column::single(s.fma.exp_gap_capped)],
            values: vec![Column::single(s.fma.exp_gap_pow2)],
            filter: Filter::default(),
        },
        LutLookup {
            table: LutTable::Pow2D,
            keys: vec![Column::single(s.fma.compression_shift)],
            values: vec![Column::single(s.fma.shift_pow2)],
            filter: Filter::default(),
        },
        // ---- W11: derive the RNERND slot `clamp(-133 - KEY_SCALE, -7, 18) + 7`. CLAMP22
        // embeds its signed fade argument by adding 400, hence lookup key `267 - KEY_SCALE`.
        // The signed cut makes `(ROUNDING_SIGNIFICAND_KEY, CUT_USED)` determine the
        // normal/subnormal classification for every reachable scale, including cancellation
        // results below 128. ----
        LutLookup {
            table: LutTable::Clamp22,
            keys: vec![Column::linear_combination_with_constant(
                [(s.fma.key_scale, neg)],
                F::from_canonical_u64(267),
            )],
            values: vec![Column::single(s.fma.cut_used)],
            filter: Filter::default(),
        },
        // ---- W11: round the bounded significand. The earlier high-bit/RC16 split, rather than
        // this packed lookup key, proves ROUNDING_SIGNIFICAND_KEY < 2^17. ----
        rnernd_lookup(
            s.fma.rounding_significand_key,
            s.fma.cut_used,
            s.fma.out_mantissa,
            s.fma.wa,
            s.fma.out_is_zero,
            s.fma.out_exp_is_zero,
        ),
        // ---- V1: WIDTH16 binds X_WIDTH = bit_length(M(alpha)*M(X)) and pins the
        // product-nonzero flag to 1 - X_IS_ZERO (alpha's significand is at least 128, so the
        // product is zero exactly when X is; W1 pins the key to the committed factors). ----
        LutLookup {
            table: LutTable::Width16,
            keys: vec![Column::single(s.fma_sig_product)],
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
        // slice — LOG16 bounds SUM_OF_SQUARES_TOP below 2^16 and serves
        // LOG_FRACTION = floor(64 * log2 SUM_OF_SQUARES_TOP) (0 at the phantom key 0). ----
        LutLookup::rc16(Column::single(s.sum_of_squares_rest_lo)),
        LutLookup {
            table: LutTable::Log16,
            keys: vec![Column::single(s.sum_of_squares_top)],
            values: vec![Column::single(s.log_fraction)],
            filter: Filter::default(),
        },
    ]
}

/// InputQuantStark's complete per-row lookup inventory: RC16 x70 (including two FMA
/// rounding-key range proofs, the four liveness certificates, and the sixteen summand-score
/// certificates of jackpot check 4), PAIR128 x3, EXPINFO x4, CLAMP22 x6, INT8DEC x2,
/// POW2D x12, RNERND x6, WIDTH16 x2, LOG16 x2, and QCAST x2. These 109 LUT instances plus
/// the eight CTL fractions above give 117 logup fractions in total.
pub fn input_quant_lut_lookups<F: Field>() -> Vec<LutLookup<F>> {
    let m = &INPUT_QUANT_COL_MAP;
    let a = side_cols_a();
    let b = side_cols_b();

    let mut lookups = side_lut_lookups::<F>(&a);
    lookups.extend(side_lut_lookups::<F>(&b));

    // ---- P3/S3: the cross-side mantissa pairs (PAIR128 x3). ----
    for (ka, kb) in [
        (m.scale_a.mantissa, m.scale_b.mantissa),
        (m.alpha_a_mantissa, m.alpha_b_mantissa),
        (m.beta_a_mantissa, m.beta_b_mantissa),
    ] {
        lookups.push(LutLookup {
            table: LutTable::Pair128,
            keys: vec![Column::single(ka), Column::single(kb)],
            values: vec![],
            filter: Filter::default(),
        });
    }
    // (No PAIR128 lookup on MAX_ABS: the L2 frame is `frame_doubled_scale_exponent`, pinned by
    // B6/B7/B8/B11; MAX_ABS feeds linf only.)

    // ---- Q1/Q2: QCAST on the noised output codes (the tuple binding pins the committed fp8
    // codes, byte range included). ----
    let qcast = |sign_expr: Column<F>, out: Column<F>| LutLookup {
        table: LutTable::Qcast,
        keys: vec![sign_expr],
        values: vec![out],
        filter: Filter::default(),
    };
    lookups.push(qcast(
        bf16_code_expr(
            m.noised_value_fma_a.out_sign,
            m.noised_value_fma_a.out_exp,
            m.noised_value_fma_a.out_mantissa,
        ),
        Column::single(m.code_noised_a),
    ));
    lookups.push(qcast(
        bf16_code_expr(
            m.noised_value_fma_b.out_sign,
            m.noised_value_fma_b.out_exp,
            m.noised_value_fma_b.out_mantissa,
        ),
        Column::single(m.code_noised_b),
    ));

    lookups
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;

    use super::*;

    type F = GoldilocksField;

    #[test]
    fn lut_inventory_matches_documented_counts() {
        // Pinned inventory: RC16 x70 (including two FMA rounding-key checks, four liveness
        // certificates, and sixteen summand-score certificates), PAIR128 x3, EXPINFO x4, CLAMP22 x6,
        // INT8DEC x2, POW2D x12, RNERND x6, WIDTH16 x2, LOG16 x2, and QCAST x2. Adding the
        // eight cross-table fractions gives 117.
        let lookups = input_quant_lut_lookups::<F>();
        let count = |t: LutTable| lookups.iter().filter(|l| l.table == t).count();
        assert_eq!(count(LutTable::Range16), 70);
        assert_eq!(count(LutTable::Pair128), 3);
        assert_eq!(count(LutTable::ExpInfo), 4);
        assert_eq!(count(LutTable::Clamp22), 6);
        assert_eq!(count(LutTable::Int8Dec), 2);
        assert_eq!(count(LutTable::Pow2D), 12);
        assert_eq!(count(LutTable::RneRnd), 6);
        assert_eq!(count(LutTable::Width16), 2);
        assert_eq!(count(LutTable::Log16), 2);
        assert_eq!(count(LutTable::Qcast), 2);
        assert_eq!(lookups.len(), 109);
        // Value-tuple arities per table.
        for l in &lookups {
            let expected_values = match l.table {
                LutTable::Range16 | LutTable::Pair128 => 0,
                LutTable::ExpInfo | LutTable::Clamp22 | LutTable::Pow2D | LutTable::Log16 | LutTable::Qcast => 1,
                LutTable::Width16 => 2,
                LutTable::Int8Dec | LutTable::RneRnd => 4,
                _ => unreachable!(),
            };
            assert_eq!(l.values.len(), expected_values, "wrong value arity for {:?}", l.table);
            let expected_keys = if l.table == LutTable::Pair128 { 2 } else { 1 };
            assert_eq!(l.keys.len(), expected_keys, "wrong key arity for {:?}", l.table);
        }
    }

    #[test]
    fn ctl_halves_are_well_formed() {
        assert_eq!(ctl_int8_bytes_looked_input_quant::<F>().len(), 2, "slots A, B");
        assert_eq!(ctl_block_scales_looked_input_quant::<F>().len(), 2);
        assert_eq!(ctl_operand_codes_looked_input_quant::<F>().len(), 2);
        let groups = ctl_group_tuples_looking_input_quant::<F>();
        assert_eq!(groups.len(), 2, "A and B group tuples");
    }
}
