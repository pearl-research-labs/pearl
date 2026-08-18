use starky::evaluation_frame::{StarkEvaluationFrame, StarkFrame};

use crate::circuit::{
    chip::blake3::{Blake3ControlFields, blake3_air, blake3_compress::blake3_permute_msg, logic::decode_is_msg_bits},
    pearl_layout::{BYTES_PER_GOLDILOCKS, pearl_columns, pearl_public},
    utils::{air_utils::RowView, evaluator::Evaluator},
};

pub(crate) fn eval_constraints<V, S, E>(
    vars: &StarkFrame<V, S, { pearl_columns::TOTAL }, { pearl_public::TOTAL }>,
    eval: &mut E,
    row_view: &mut RowView<V>,
    cf: &Blake3ControlFields<V>,
    uint8_data: &[V],
) -> [V; 8]
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let local_trace = vars.get_local_values();
    let next_trace = vars.get_next_values();
    let public_inputs = vars.get_public_inputs();
    let consts: [V; 17] = std::array::from_fn(|i| eval.u64(i as u64));
    let one = consts[1];
    let c256 = eval.i32(256);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read BLAKE3_MSG_BUFFER
    debug_assert_eq!(row_view.offset, pearl_columns::BLAKE3_MSG_BUFFER);
    let blake3_msg_buffer = row_view.consume_few(pearl_columns::BLAKE3_MSG_BUFFER_LEN);
    debug_assert_eq!(row_view.offset, pearl_columns::BLAKE3_MSG_BUFFER_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read CV_OR_TWEAK_PREP (CV_IDX at rounds 1/4/8, Blake3Tweak at round 2)
    // Used directly as cv_idx since it's a preprocessed value (no range check needed)
    debug_assert_eq!(row_view.offset, pearl_columns::CV_OR_TWEAK_PREP);
    let _cv_idx = row_view.consume_single(); // CV_OR_TWEAK_PREP serves as cv_idx through lookups
    debug_assert_eq!(row_view.offset, pearl_columns::CV_OR_TWEAK_PREP_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read CV_IN
    debug_assert_eq!(row_view.offset, pearl_columns::CV_IN);
    let cv_in = row_view.consume_few(pearl_columns::CV_IN_LEN);
    debug_assert_eq!(row_view.offset, pearl_columns::CV_IN_END);

    // Check BLAKE3_MSG_BUFFER correctness (shift + data loading)
    let jackpot: [V; pearl_columns::JACKPOT_MSG_LEN] = local_trace[pearl_columns::JACKPOT_MSG_RANGE].try_into().unwrap();

    let (is_msg_jackpot, is_msg_uint8_data, is_msg_cv) = decode_is_msg_bits(eval, one, cf.is_msg_bits);
    verify_buffer_advancement(
        blake3_msg_buffer,
        &next_trace[pearl_columns::BLAKE3_MSG_BUFFER_RANGE],
        uint8_data,
        cv_in,
        &jackpot,
        is_msg_uint8_data,
        is_msg_cv,
        is_msg_jackpot,
        c256,
        eval,
    );

    // blake3_tweak is read from NEXT row's CV_OR_TWEAK_PREP (stored at round 2 and consumed in round 1)
    let blake3_tweak = next_trace[pearl_columns::CV_OR_TWEAK_PREP];

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read BLAKE3_MSG
    debug_assert_eq!(row_view.offset, pearl_columns::BLAKE3_MSG);
    let blake3_msg = row_view.consume_few(pearl_columns::BLAKE3_MSG_LEN);
    verify_msg_constraints(blake3_msg, blake3_msg_buffer, next_trace, cf.is_last_round, eval);
    debug_assert_eq!(row_view.offset, pearl_columns::BLAKE3_MSG_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read BLAKE3_CV
    debug_assert_eq!(row_view.offset, pearl_columns::BLAKE3_CV);
    let blake3_cv = row_view.consume_few(pearl_columns::BLAKE3_CV_LEN);
    // Check BLAKE3_CV equals CV_IN, JOB_KEY, or COMMITMENT_HASH based on flags.
    // is_use_job_key and is_use_commitment_hash are mutually exclusive.
    let job_key: [V; pearl_public::JOB_KEY_LEN] = std::array::from_fn(|i| eval.scalar(public_inputs[pearl_public::JOB_KEY + i]));
    let commitment_hash: [V; pearl_public::COMMITMENT_HASH_LEN] =
        std::array::from_fn(|i| eval.scalar(public_inputs[pearl_public::COMMITMENT_HASH + i]));
    let use_cv_in = eval.sub(one, cf.is_use_job_key);
    let use_cv_in = eval.sub(use_cv_in, cf.is_use_commitment_hash);
    let cv_selectors = [use_cv_in, cf.is_use_job_key, cf.is_use_commitment_hash];
    for i in 0..pearl_columns::BLAKE3_CV_LEN {
        let cv_sources = [cv_in[i], job_key[i], commitment_hash[i]];
        let expected_cv = eval.inner_product(&cv_selectors, &cv_sources);
        eval.constraint_eq(blake3_cv[i], expected_cv);
    }
    debug_assert_eq!(row_view.offset, pearl_columns::BLAKE3_CV_END);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read BLAKE3_ROUND and check constraints
    debug_assert_eq!(row_view.offset, pearl_columns::BLAKE3_ROUND);
    let next_is_new_blake = next_trace[pearl_columns::IS_NEW_BLAKE];
    let (blake3_output, init_state) = blake3_air::blake3_eval_transition_constraints(
        eval,
        row_view,
        blake3_msg,
        &next_trace[pearl_columns::BLAKE3_ROUND_RANGE],
        next_is_new_blake,
    );
    // Verify blake3 init state when this row starts a new blake3 (is_new_blake).
    blake3_air::verify_init_state(eval, &init_state, cf.is_new_blake, blake3_cv, blake3_tweak);
    debug_assert_eq!(row_view.offset, pearl_columns::BLAKE3_ROUND_END);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read CV_OUT
    debug_assert_eq!(row_view.offset, pearl_columns::CV_OUT);
    let cv_out = row_view.consume_few(pearl_columns::CV_OUT_LEN);
    for i in 0..pearl_columns::CV_OUT_LEN {
        eval.constraint_eq(cv_out[i], blake3_output[i]);
    }
    debug_assert_eq!(row_view.offset, pearl_columns::CV_OUT_END);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read CV_OUT_FREQ
    debug_assert_eq!(row_view.offset, pearl_columns::CV_OUT_FREQ);
    let _cv_out_freq = row_view.consume_single();
    debug_assert_eq!(row_view.offset, pearl_columns::CV_OUT_FREQ_END);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Check data when is_first_outer or is_second_outer: the first two limbs in UINT8_DATA should be the outer indices, and OUTER_INDEX_FIRST and OUTER_INDEX_SECOND should match.
    let first_limb = eval.polyval(&uint8_data[0..4], c256);
    let second_limb = eval.polyval(&uint8_data[4..8], c256);
    eval.constraint_eq_if(cf.is_first_outer, first_limb, cf.outer_index_first);
    eval.constraint_eq_if(cf.is_second_outer, second_limb, cf.outer_index_second);

    blake3_output
}

/// Check the message transition constraints.
/// - In rounds 2-8 (not IS_NEW_BLAKE): BLAKE3_MSG is a permuted version of prev msg.
/// - In round 8 (IS_LAST_ROUND): BLAKE3_MSG_BUFFER should equal permuted BLAKE3_MSG.
fn verify_msg_constraints<V, S, E>(blake3_msg: &[V], blake3_msg_buffer: &[V], next_trace: &[V], is_last_round: V, eval: &mut E)
where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    let next_blake3_msg = &next_trace[pearl_columns::BLAKE3_MSG_RANGE];
    let next_is_new_blake = next_trace[pearl_columns::IS_NEW_BLAKE];

    // If next is a middle round of blake3 (NOT a new blake), check msg is permuted version of current msg.
    let one = eval.u64(1);
    let next_is_same_blake = eval.sub(one, next_is_new_blake);
    let mut permuted_msg: [V; 16] = blake3_msg.try_into().unwrap();
    blake3_permute_msg(&mut permuted_msg);
    for i in 0..16 {
        eval.constraint_eq_if(next_is_same_blake, permuted_msg[i], next_blake3_msg[i]);
    }

    // In round 8 (IS_LAST_ROUND): check BLAKE3_MSG_BUFFER equals permuted BLAKE3_MSG.
    // This is because BLAKE3_MSG_BUFFER = original message and blake message permutation has order 8.
    for i in 0..16 {
        eval.constraint_eq_if(is_last_round, permuted_msg[i], blake3_msg_buffer[i]);
    }
}

/// Check BLAKE3_MSG_BUFFER advancement constraints.
/// The buffer shifts by 2 elements (1 dword) per round, and new data is loaded into the tail.
/// - If is_msg_uint8_data (Matrix or Aux/Routing): load uint8_data_packed into msg_buffer[14..16]
/// - If IS_MSG_CV: load CV_IN (8 elements) into msg_buffer[8..16]
/// - If IS_MSG_JACKPOT: verify entire buffer equals the selected jackpot slice
#[allow(clippy::too_many_arguments)]
fn verify_buffer_advancement<V, S, E>(
    msg_buffer: &[V],
    next_msg_buffer: &[V],
    uint8_data: &[V],
    cv_in: &[V],
    jackpot: &[V],
    is_msg_uint8_data: V,
    is_msg_cv: V,
    is_msg_jackpot: V,
    c256: V,
    eval: &mut E,
) where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    let buffer_len = pearl_columns::BLAKE3_MSG_BUFFER_LEN; // 16
    let shift_len = 2; // 1 dword = 2 packed elements

    // Buffer transition constraint: next_buffer[0..14] = buffer[2..16]
    for i in 0..(buffer_len - shift_len) {
        eval.constraint_eq(next_msg_buffer[i], msg_buffer[i + shift_len]);
    }

    let uint8_data_packed: [V; 2] =
        std::array::from_fn(|i| eval.polyval(&uint8_data[i * BYTES_PER_GOLDILOCKS..(i + 1) * BYTES_PER_GOLDILOCKS], c256));

    for i in 0..shift_len {
        eval.constraint_eq_if(is_msg_uint8_data, msg_buffer[14 + i], uint8_data_packed[i]);
    }

    for i in 0..pearl_columns::CV_IN_LEN {
        eval.constraint_eq_if(is_msg_cv, msg_buffer[8 + i], cv_in[i]);
    }

    debug_assert_eq!(jackpot.len(), pearl_columns::BLAKE3_MSG_LEN);
    for i in 0..pearl_columns::BLAKE3_MSG_LEN {
        eval.constraint_eq_if(is_msg_jackpot, msg_buffer[i], jackpot[i]);
    }
}
