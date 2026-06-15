use crate::{
    api::proof_utils::CompiledPublicParams,
    circuit::{
        chip::blake3::{
            blake3_compress::{BLAKE3_IV, BLAKE3_MSG_PERMUTATION, Blake3Tweak, blake3_compress},
            logic::{BlakeRoundLogic, MessageDataType},
        },
        pearl_layout::{BYTES_PER_GOLDILOCKS, pearl_columns},
        utils::trace_utils::bytes_to_words,
    },
};
use plonky2::hash::hash_types::RichField;
use plonky2_field::extension::Extendable;
use plonky2_maybe_rayon::*;

use crate::circuit::utils::trace_utils::{read_bits_as_u32s, read_from_trace, u64_pack_le, write_to_trace, write_u32s_as_bits};

/// Fill all BLAKE3-related columns using two-phase parallel algorithm.
pub fn fill_blake_data<F: RichField + Extendable<D>, const D: usize>(
    trace: &mut Vec<[F; pearl_columns::TOTAL]>,
    circuit: &[BlakeRoundLogic],
    compiled_params: &CompiledPublicParams,
) {
    let num_rows = trace.len();
    let job_key: [u32; 8] = bytes_to_words(&compiled_params.job_key);
    let (_, commitment_hash_bytes) = compiled_params.commitment_hash;
    let commitment_hash: [u32; 8] = bytes_to_words(&commitment_hash_bytes);

    // Phase 1: Sequential - buffer, CV_IN, BLAKE3_CV, MSG columns
    let mut buffer = [0u32; 16];
    let mut blocks: Vec<(usize, usize, Blake3Tweak)> = Vec::new();

    for row_idx in 0..num_rows {
        let blake = &circuit[row_idx];
        buffer = shift_buffer(&buffer);
        load_buffer(&mut buffer, row_idx, &blake.data_source, trace);

        // Write CV_IN and BLAKE3_CV for every row
        let cv_in: [u32; 8] = blake
            .idx_of_row_whence_to_read_cv
            .map(|i| read_from_trace(&trace[i], pearl_columns::CV_OUT))
            .unwrap_or([0; 8]);
        write_to_trace(&mut trace[row_idx], pearl_columns::CV_IN, &cv_in);
        let blake3_cv = if blake.is_use_commitment_hash() {
            commitment_hash
        } else if blake.is_use_job_key() {
            job_key
        } else {
            cv_in
        };
        write_to_trace(&mut trace[row_idx], pearl_columns::BLAKE3_CV, &blake3_cv);

        // Process block only at its end
        if circuit[(row_idx + 1) % num_rows].round_idx != 1 {
            continue;
        }

        let first = row_idx + 1 - blake.round_idx;

        // Write MSG columns and CV_OUT for complete 8-round blocks
        if blake.round_idx == 8 {
            debug_assert!(circuit[first].blake3_tweak.is_some());
            // Compute CV_OUT early so subsequent blocks can read it for their CV_IN
            let cv: [u32; 8] = read_from_trace(&trace[first], pearl_columns::BLAKE3_CV);
            // Note: can compute cv_out without blake3_compress
            let cv_out: [u32; 8] = bytes_to_words(&blake3_compress(
                &std::array::from_fn(|i| (buffer[i / 4] >> (8 * (i % 4))) as u8),
                std::array::from_fn(|i| (cv[i / 4] >> (8 * (i % 4))) as u8),
                circuit[first].blake3_tweak.unwrap(),
            ));
            write_to_trace(&mut trace[row_idx], pearl_columns::CV_OUT, &cv_out);

            write_to_trace(&mut trace[row_idx], pearl_columns::BLAKE3_MSG_BUFFER, &buffer);
            let mut msg = blake3_inverse_permute_msg(&buffer);
            write_to_trace(&mut trace[row_idx], pearl_columns::BLAKE3_MSG, &msg);
            for r in (row_idx.saturating_sub(7)..row_idx).rev() {
                msg = blake3_inverse_permute_msg(&msg);
                write_to_trace(&mut trace[r], pearl_columns::BLAKE3_MSG, &msg);
            }
            for r in row_idx.saturating_sub(7)..row_idx {
                let len = 2 * (r + 8 - row_idx);
                write_to_trace(&mut trace[r], pearl_columns::BLAKE3_MSG_BUFFER + 16 - len, &buffer[..len]);
            }
            let mut fwd = buffer;
            for off in 1..=7 {
                fwd = shift_buffer(&fwd);
                write_to_trace(
                    &mut trace[(row_idx + off) % num_rows],
                    pearl_columns::BLAKE3_MSG_BUFFER,
                    &fwd[..16 - 2 * off],
                );
            }
        }
        let blake3_tweak = circuit[first].blake3_tweak.unwrap_or_default();
        blocks.push((first, row_idx, blake3_tweak));
    }

    // Phase 2: Parallel - BLAKE3_ROUND and CV_OUT
    blocks.par_iter().for_each(|&(first, pivot, params)| {
        let cv: [u32; 8] = read_from_trace(&trace[first], pearl_columns::BLAKE3_CV);
        let mut state: [u32; 16] = std::array::from_fn(|i| match i {
            0..=7 => cv[i],
            8..=11 => BLAKE3_IV[i - 8],
            12 => params.counter_low,
            13 => params.counter_high as u32,
            14 => params.block_len,
            _ => params.flags,
        });

        for row_idx in first..=pivot {
            let row = unsafe { &mut *(trace.as_ptr() as *mut [F; pearl_columns::TOTAL]).add(row_idx) };
            write_blake3_state(row, pearl_columns::BLAKE3_ROUND, &state);

            let cv_out: [u32; 8] = if row_idx == pivot {
                std::array::from_fn(|i| state[i] ^ state[8 + i])
            } else {
                let init = state;
                let msg: [u32; 16] = read_from_trace(&trace[row_idx], pearl_columns::BLAKE3_MSG);
                let states = compute_blake3_round(&mut state, &msg);
                for (i, s) in states[..3].iter().enumerate() {
                    write_blake3_state(row, pearl_columns::BLAKE3_ROUND + (i + 1) * BLAKE3_STATE_SIZE, s);
                }
                std::array::from_fn(|i| {
                    if i < 4 {
                        states[0][4 + i] ^ states[0][12 + i]
                    } else {
                        init[i] ^ init[8 + i]
                    }
                })
            };
            write_to_trace(row, pearl_columns::CV_OUT, &cv_out);
        }
    });
    postprocess_round8_rows(trace, circuit);
}

/// Apply the inverse of BLAKE3_MSG_PERMUTATION
fn blake3_inverse_permute_msg(msg: &[u32; 16]) -> [u32; 16] {
    let mut result = [0u32; 16];
    for i in 0..16 {
        result[BLAKE3_MSG_PERMUTATION[i]] = msg[i];
    }
    result
}

/// Shift buffer left by 2: result[0..14] = buffer[2..16], result[14..16] = 0. Matches the BLAKE3_MSG_BUFFER advancement.
fn shift_buffer(buffer: &[u32; 16]) -> [u32; 16] {
    std::array::from_fn(|i| if i < 14 { buffer[i + 2] } else { 0 })
}

/// Load data into buffer[14..16] or buffer[8..16] or entire buffer, depending on data_source.
fn load_buffer<F: RichField>(
    buffer: &mut [u32; 16],
    row_idx: usize,
    data_source: &MessageDataType,
    trace: &[[F; pearl_columns::TOTAL]],
) {
    match *data_source {
        MessageDataType::Matrix { .. } | MessageDataType::AuxiliaryData { .. } | MessageDataType::RoutingData { .. } => {
            // Read from UINT8_DATA which was already filled by the main loop
            let bytes: [u8; 8] = read_from_trace(&trace[row_idx], pearl_columns::UINT8_DATA);
            buffer[14] = u64_pack_le(&bytes[..BYTES_PER_GOLDILOCKS], 8) as u32;
            buffer[15] = u64_pack_le(&bytes[BYTES_PER_GOLDILOCKS..], 8) as u32;
        }
        MessageDataType::PreviousCv { source_row_idx } => {
            buffer[8..16].copy_from_slice(&read_from_trace::<F, u32, 8>(&trace[source_row_idx], pearl_columns::CV_OUT));
        }
        MessageDataType::Jackpot => {
            // Read JACKPOT_MSG from trace (filled by fill_cumsum_buffer_xor_jackpot)
            let jackpot_msg: [u32; 16] = read_from_trace(&trace[row_idx], pearl_columns::JACKPOT_MSG);
            buffer.copy_from_slice(&jackpot_msg);
        }
        MessageDataType::None => {}
    }
}

/// Perform half of a quarter round on the given elements.
fn half_quarter_round(mut a: u32, mut b: u32, mut c: u32, mut d: u32, m: u32, is_second_half: bool) -> (u32, u32, u32, u32) {
    let (rot_1, rot_2) = if is_second_half { (8, 7) } else { (16, 12) };
    a = a.wrapping_add(b).wrapping_add(m);
    d = (d ^ a).rotate_right(rot_1);
    c = c.wrapping_add(d);
    b = (b ^ c).rotate_right(rot_2);
    (a, b, c, d)
}

/// Write a BLAKE3 state to trace at the given base column.
/// Layout: row1 (4 packed), row2 (128 bits), row3 (4 packed), row4 (128 bits)
fn write_blake3_state<F: RichField>(trace: &mut [F], base_col: usize, state: &[u32; 16]) {
    let mut col = base_col;
    // row1: state[0..4] packed
    for i in 0..4 {
        trace[col] = F::from_canonical_u64(state[i] as u64);
        col += 1;
    }
    // row2: state[4..8] as bits (32 bits each)
    for i in 4..8 {
        let mut val = state[i];
        for _ in 0..32 {
            trace[col] = F::from_canonical_u64((val & 1) as u64);
            col += 1;
            val >>= 1;
        }
    }
    // row3: state[8..12] packed
    for i in 8..12 {
        trace[col] = F::from_canonical_u64(state[i] as u64);
        col += 1;
    }
    // row4: state[12..16] as bits (32 bits each)
    for i in 12..16 {
        let mut val = state[i];
        for _ in 0..32 {
            trace[col] = F::from_canonical_u64((val & 1) as u64);
            col += 1;
            val >>= 1;
        }
    }
}

/// Compute one full BLAKE3 round (4 half quarter rounds) and return the 4 intermediate states.
/// Returns (state1, state2, state3, final_state) where final_state goes to next row's input.
fn compute_blake3_round(state: &mut [u32; 16], msg: &[u32; 16]) -> [[u32; 16]; 4] {
    let mut states = [[0u32; 16]; 4];

    // Column half round 1
    for i in 0..4 {
        let (a, b, c, d) = half_quarter_round(state[i], state[4 + i], state[8 + i], state[12 + i], msg[2 * i], false);
        state[i] = a;
        state[4 + i] = b;
        state[8 + i] = c;
        state[12 + i] = d;
    }
    states[0] = *state;

    // Column half round 2
    for i in 0..4 {
        let (a, b, c, d) = half_quarter_round(state[i], state[4 + i], state[8 + i], state[12 + i], msg[2 * i + 1], true);
        state[i] = a;
        state[4 + i] = b;
        state[8 + i] = c;
        state[12 + i] = d;
    }
    states[1] = *state;

    // Diagonal half round 1
    for i in 0..4 {
        let (a, b, c, d) = half_quarter_round(
            state[i],
            state[4 + (i + 1) % 4],
            state[8 + (i + 2) % 4],
            state[12 + (i + 3) % 4],
            msg[8 + 2 * i],
            false,
        );
        state[i] = a;
        state[4 + (i + 1) % 4] = b;
        state[8 + (i + 2) % 4] = c;
        state[12 + (i + 3) % 4] = d;
    }
    states[2] = *state;

    // Diagonal half round 2
    for i in 0..4 {
        let (a, b, c, d) = half_quarter_round(
            state[i],
            state[4 + (i + 1) % 4],
            state[8 + (i + 2) % 4],
            state[12 + (i + 3) % 4],
            msg[8 + 2 * i + 1],
            true,
        );
        state[i] = a;
        state[4 + (i + 1) % 4] = b;
        state[8 + (i + 2) % 4] = c;
        state[12 + (i + 3) % 4] = d;
    }
    states[3] = *state;

    states
}

/// Size of one BLAKE3 state in trace columns: 4 packed + 128 bits + 4 packed + 128 bits = 264
const BLAKE3_STATE_SIZE: usize = 4 + 128 + 4 + 128;

/// Post-process rows preceding round_idx == 1 rows (cyclically).
/// These are finalization rows that need STATE1-3 filled to satisfy add_unchecked constraints.
fn postprocess_round8_rows<F: RichField>(trace: &mut [[F; pearl_columns::TOTAL]], circuit: &[BlakeRoundLogic]) {
    let n = trace.len();
    for row_idx in 0..n {
        let next = (row_idx + 1) % n;
        if circuit[next].round_idx != 1 {
            continue;
        }

        let base = pearl_columns::BLAKE3_ROUND;
        let msg: [u32; 16] = read_from_trace(&trace[row_idx], pearl_columns::BLAKE3_MSG);
        let (r1, r2, r3): ([u32; 4], [u32; 4], [u32; 4]) = (
            read_from_trace(&trace[row_idx], base),
            read_bits_as_u32s(&trace[row_idx], base + 4),
            read_from_trace(&trace[row_idx], base + 4 + 128),
        );

        // STATE1: row2/row4 = input for finalization XOR; row1/row3 from half-round
        let (s1r1, s1r3): ([u32; 4], [u32; 4]) = (
            std::array::from_fn(|i| r1[i].wrapping_add(r2[i]).wrapping_add(msg[2 * i])),
            std::array::from_fn(|i| r3[i].wrapping_add(r3[i])),
        );
        let b1 = base + BLAKE3_STATE_SIZE;
        write_to_trace(&mut trace[row_idx], b1, &s1r1);
        write_u32s_as_bits(&mut trace[row_idx], b1 + 4, &r1);
        write_to_trace(&mut trace[row_idx], b1 + 4 + 128, &s1r3);
        write_u32s_as_bits(&mut trace[row_idx], b1 + 4 + 128 + 4, &r3);

        // STATE2: row2/row4 = 0; row1/row3 from constraints
        let s2r1: [u32; 4] = std::array::from_fn(|i| s1r1[i].wrapping_add(r1[i]).wrapping_add(msg[2 * i + 1]));
        let b2 = base + 2 * BLAKE3_STATE_SIZE;
        write_to_trace(&mut trace[row_idx], b2, &s2r1);
        write_u32s_as_bits(&mut trace[row_idx], b2 + 4, &[0u32; 4]);
        write_to_trace(&mut trace[row_idx], b2 + 4 + 128, &s1r3);
        write_u32s_as_bits(&mut trace[row_idx], b2 + 4 + 128 + 4, &[0u32; 4]);

        // STATE3: from diagonal constraints and next row's INPUT_STATE
        let (nr1, nr3, nr4): ([u32; 4], [u32; 4], [u32; 4]) = (
            read_from_trace(&trace[next], base),
            read_from_trace(&trace[next], base + 4 + 128),
            read_bits_as_u32s(&trace[next], base + 4 + 128 + 4),
        );
        let s3r1: [u32; 4] = std::array::from_fn(|i| s2r1[i].wrapping_add(msg[8 + 2 * i]));
        let (mut s3r2, mut s3r3, mut s3r4) = ([0u32; 4], [0u32; 4], [0u32; 4]);
        for i in 0..4 {
            let (b, c, d) = ((i + 1) % 4, (i + 2) % 4, (i + 3) % 4);
            s3r3[c] = nr3[c].wrapping_sub(nr4[d]);
            s3r4[d] = s3r3[c].wrapping_sub(s1r3[c]);
            s3r2[b] = nr1[i].wrapping_sub(s3r1[i]).wrapping_sub(msg[8 + 2 * i + 1]);
        }
        let b3 = base + 3 * BLAKE3_STATE_SIZE;
        write_to_trace(&mut trace[row_idx], b3, &s3r1);
        write_u32s_as_bits(&mut trace[row_idx], b3 + 4, &s3r2);
        write_to_trace(&mut trace[row_idx], b3 + 4 + 128, &s3r3);
        write_u32s_as_bits(&mut trace[row_idx], b3 + 4 + 128 + 4, &s3r4);
    }
}
