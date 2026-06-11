use starky::evaluation_frame::{StarkEvaluationFrame, StarkFrame};

use crate::v1::circuit::{
    chip::jackpot::JackpotControlFields,
    pearl_layout::{pearl_columns, pearl_public},
    pearl_program::{JACKPOT_SIZE, LROT_PER_TILE},
    utils::{
        air_utils::{RowView, degree_2_indicators},
        evaluator::Evaluator,
    },
};

/// (bits >>> right_shift) as u32
fn bits_to_u32<V, S, E>(bits: &[V], right_shift: u32, eval: &mut E) -> V
where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    debug_assert_eq!(bits.len(), 32);
    let two = eval.i32(2);
    let rhs = right_shift as usize % 32;
    eval.polyval(&[&bits[rhs..], &bits[..rhs]].concat(), two)
}

fn bits_to_i32<V, S, E>(bits: &[V], eval: &mut E) -> V
where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    debug_assert_eq!(bits.len(), 32);
    let two = eval.i32(2);
    let low_31 = eval.polyval(&bits[..31], two);
    let c2_31 = eval.u64(1u64 << 31);
    let high_bit = eval.mul(bits[31], c2_31);
    eval.sub(low_31, high_bit)
}

pub fn eval_constraints<V, S, E>(
    vars: &StarkFrame<V, S, { pearl_columns::TOTAL }, { pearl_public::TOTAL }>,
    eval: &mut E,
    row_view: &mut RowView<V>,
    cf: &JackpotControlFields<V>,
    cumsum_tile: &[V],
) where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let next_trace = vars.get_next_values();
    let consts: [V; 17] = std::array::from_fn(|i| eval.u64(i as u64));
    let one = consts[1];

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // CUMSUM_BUFFER: 4 i32 elements cycling backwards
    // - If IS_DUMP_CUMSUM_BUFFER: load CUMSUM_TILE into buffer
    // - Otherwise: shift buffer left by 1 (cycling)
    debug_assert_eq!(row_view.offset, pearl_columns::CUMSUM_BUFFER);
    let cumsum_buffer = row_view.consume_few(pearl_columns::CUMSUM_BUFFER_LEN);
    let next_cumsum_buffer = &next_trace[pearl_columns::CUMSUM_BUFFER_RANGE];

    for i in 0..pearl_columns::CUMSUM_BUFFER_LEN {
        let expected = eval.mux(
            cf.is_dump_cumsum_buffer,
            next_cumsum_buffer[(i + 1) % pearl_columns::CUMSUM_BUFFER_LEN],
            cumsum_tile[i],
        );
        eval.constraint_eq(cumsum_buffer[i], expected);
    }
    debug_assert_eq!(row_view.offset, pearl_columns::CUMSUM_BUFFER_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // JACKPOT_MSG: 16-element u32 RAM with load/store operations
    // JACKPOT_IDX encoding (degree-2 indicator):
    //   - Indices 0..15:  Load jackpot[idx] into BIT_REG (rotated by LROT_PER_TILE)
    //   - Indices 16..31: Store rotated BIT_REG into jackpot[idx-16]
    debug_assert_eq!(row_view.offset, pearl_columns::JACKPOT_MSG);
    let jackpot_msg = row_view.consume_few(pearl_columns::JACKPOT_MSG_LEN);
    let next_jackpot = &next_trace[pearl_columns::JACKPOT_MSG_RANGE];
    let next_jackpot_idx = &next_trace[pearl_columns::JACKPOT_IDX_RANGE];
    let next_bit_reg = &next_trace[pearl_columns::BIT_REG_RANGE];

    // jackpot initialized with 0's
    for i in 0..pearl_columns::JACKPOT_MSG_LEN {
        eval.constraint_first_row(jackpot_msg[i]);
    }

    // Precompute BIT_REG rotations (LROT_PER_TILE bits per rotation)
    let next_bitreg_rot0 = bits_to_u32(next_bit_reg, 0, eval);
    let next_bitreg_rot1 = bits_to_u32(next_bit_reg, LROT_PER_TILE, eval);
    let next_bitreg_rot2 = bits_to_u32(next_bit_reg, 2 * LROT_PER_TILE, eval);

    // Load constraint: when IS_LOAD, bitreg_rot1 must equal selected jackpot value
    // Constraint: bitreg_rot1 * is_load == sum(load_ind[i] * jackpot[i])
    let next_load_idx = degree_2_indicators(eval, next_jackpot_idx, 0..JACKPOT_SIZE);
    let next_jackpot_load = eval.inner_product(&next_load_idx, next_jackpot);
    let next_is_load = next_trace[pearl_columns::IS_LOAD];
    let load_constraint = eval.msub(next_bitreg_rot1, next_is_load, next_jackpot_load);
    eval.constraint(load_constraint);

    // Store indicators and value being stored to jackpot
    let next_store_idx = degree_2_indicators(eval, next_jackpot_idx, JACKPOT_SIZE..2 * JACKPOT_SIZE);
    let next_jackpot_store = eval.inner_product(&next_store_idx, next_jackpot);

    // Persistence: unchanged jackpot elements must stay the same (transition constraint)
    for i in 0..pearl_columns::JACKPOT_MSG_LEN {
        let not_modified = eval.sub(one, next_store_idx[i]);
        let diff = eval.sub(jackpot_msg[i], next_jackpot[i]);
        let not_modified_constraint = eval.mul(not_modified, diff);
        eval.constraint_transition(not_modified_constraint);
    }

    // Store constraint: selected jackpot value must equal rotated bitreg
    // IS_STORE0/1/2 select which rotation to store (0, LROT_PER_TILE, 2*LROT_PER_TILE bits right)
    let next_is_store_rot0 = next_trace[pearl_columns::IS_STORE0];
    let next_is_store_rot1 = next_trace[pearl_columns::IS_STORE1];
    let next_is_store_rot2 = next_trace[pearl_columns::IS_STORE2];
    let next_bitreg_to_store = eval.inner_product(
        &[next_is_store_rot0, next_is_store_rot1, next_is_store_rot2],
        &[next_bitreg_rot0, next_bitreg_rot1, next_bitreg_rot2],
    );
    // Combined constraint: stored value must equal RAM access
    let next_jackpot_access = eval.add(next_jackpot_load, next_jackpot_store);
    eval.constraint_eq(next_bitreg_to_store, next_jackpot_access);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // BIT_REG: 32-bit register for XOR accumulation
    // Operations:
    //   - IS_XOR: XOR cumsum_buffer[0] into bit_reg
    //   - IS_SHIFT3: Rotate bit_reg right by 3*LROT_PER_TILE (39 bits)
    debug_assert_eq!(row_view.offset, pearl_columns::BIT_REG);
    let bit_reg = row_view.consume_few(pearl_columns::BIT_REG_LEN);
    for &bit in bit_reg {
        eval.constraint_bool(bit);
    }

    // IS_XOR constraint: bit_reg XOR next_bit_reg must equal cumsum_buffer[0] (as i32)
    let next_is_xor = next_trace[pearl_columns::IS_XOR];
    let xor_delta_bits: [V; 32] = std::array::from_fn(|i| eval.xor_bit(bit_reg[i], next_bit_reg[i]));
    let xor_delta_i32 = bits_to_i32(&xor_delta_bits, eval);
    eval.constraint_eq_if(next_is_xor, xor_delta_i32, cumsum_buffer[0]);

    // IS_SHIFT3 constraint: bit_reg rotated by 3*LROT must equal next_bitreg
    let next_is_shift3 = next_trace[pearl_columns::IS_SHIFT3];
    let bit_reg_rot3 = bits_to_u32(bit_reg, 3 * LROT_PER_TILE, eval);
    eval.constraint_eq_if(next_is_shift3, bit_reg_rot3, next_bitreg_rot0);

    debug_assert_eq!(row_view.offset, pearl_columns::BIT_REG_END);
}
