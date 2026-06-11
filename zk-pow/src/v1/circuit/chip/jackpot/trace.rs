use plonky2::hash::hash_types::RichField;

use crate::v1::circuit::{
    chip::{BitRegDst, BitRegSrc, JackpotLogic},
    pearl_layout::pearl_columns,
    pearl_program::LROT_PER_TILE,
    utils::trace_utils::{read_from_trace, write_to_trace, write_u32_as_bits},
};

/// Fill CUMSUM_BUFFER, BIT_REG, and JACKPOT_MSG columns.
///
/// CUMSUM_BUFFER: Filled backwards from is_dump_cumsum_buffer=true rows (which load from CUMSUM_TILE).
/// Other rows rotate cyclically from next row: cumsum_buffer[i] = next_cumsum_buffer[(i+1) % 4].
///
/// BIT_REG and JACKPOT_MSG: Single forward pass starting from zeros.
/// - src=Jackpot: bit_reg = jackpot_msg[idx].rotate_left(LROT)
/// - src=Xor: bit_reg ^= prev_cumsum_buffer[0]
/// - src=Shift3: bit_reg = bit_reg.rotate_right(3*LROT)
/// - Store (when src != Jackpot): jackpot_msg[idx] = bit_reg rotated by 0/LROT/2*LROT
pub fn fill_cumsum_buffer_xor_jackpot<F: RichField>(trace: &mut [[F; pearl_columns::TOTAL]], circuit: &[JackpotLogic]) {
    let num_rows = trace.len();

    // ========== CUMSUM_BUFFER ==========
    // Find all is_dump_cumsum_buffer row indices and initialize them from CUMSUM_TILE
    let dump_rows: Vec<usize> = (0..num_rows).filter(|&i| circuit[i].is_dump_cumsum_buffer).collect();

    for &row_idx in &dump_rows {
        let cumsum: [i32; 4] = read_from_trace(&trace[row_idx], pearl_columns::CUMSUM_TILE);
        write_to_trace(&mut trace[row_idx], pearl_columns::CUMSUM_BUFFER, &cumsum);
    }

    // Fill backwards from each is_dump_cumsum_buffer row to the previous one
    for i in 0..dump_rows.len() {
        let end_row = dump_rows[i];
        let start_row = dump_rows[if i == 0 { dump_rows.len() - 1 } else { i - 1 }];

        // Read initial value from is_dump_cumsum_buffer row, then propagate backwards with rotation
        let mut current: [i32; 4] = read_from_trace(&trace[end_row], pearl_columns::CUMSUM_BUFFER);
        let mut row_idx = (end_row + num_rows - 1) % num_rows;
        while row_idx != start_row {
            current = [current[1], current[2], current[3], current[0]];
            write_to_trace(&mut trace[row_idx], pearl_columns::CUMSUM_BUFFER, &current);
            row_idx = if row_idx == 0 { num_rows - 1 } else { row_idx - 1 };
        }
    }

    // ========== BIT_REG and JACKPOT_MSG ==========
    let mut bit_reg: u32 = 0;
    let mut jackpot_msg: [u32; 16] = [0; 16];

    for row_idx in 0..num_rows {
        let logic = &circuit[row_idx];
        let idx = logic.jackpot_idx;

        // Get prev_cumsum_buffer (row 0 uses zeros, which is correct for initial state)
        let prev_cumsum_buffer: [i32; 4] = if row_idx > 0 {
            read_from_trace(&trace[row_idx - 1], pearl_columns::CUMSUM_BUFFER)
        } else {
            [0i32; 4]
        };

        // Compute bit_reg based on src
        bit_reg = match logic.src {
            BitRegSrc::Jackpot => jackpot_msg[idx].rotate_left(LROT_PER_TILE),
            BitRegSrc::Xor => bit_reg ^ (prev_cumsum_buffer[0] as u32),
            BitRegSrc::Shift3 => bit_reg.rotate_right(3 * LROT_PER_TILE),
        };

        // Update jackpot_msg on store (when src != Jackpot)
        if !matches!(logic.src, BitRegSrc::Jackpot) {
            jackpot_msg[idx] = match logic.dst {
                BitRegDst::Store0 => bit_reg,
                BitRegDst::Store1 => bit_reg.rotate_right(LROT_PER_TILE),
                BitRegDst::Store2 => bit_reg.rotate_right(2 * LROT_PER_TILE),
            };
        }

        // Write to trace
        write_to_trace(&mut trace[row_idx], pearl_columns::JACKPOT_MSG, &jackpot_msg);
        write_u32_as_bits(&mut trace[row_idx], pearl_columns::BIT_REG, bit_reg);
    }
}
