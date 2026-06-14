use itertools::Itertools;
use starky::evaluation_frame::{StarkEvaluationFrame, StarkFrame};

use crate::v1::circuit::{
    pearl_layout::{BITS_PER_LIMB, BYTES_PER_GOLDILOCKS, pearl_columns, pearl_public},
    pearl_program::{TILE_D, TILE_H},
    utils::{air_utils::RowView, evaluator::Evaluator},
};

// pack 4 elements at a time
fn verify_packing_le<V, S, E>(unpacked: &[V], packed: &[V], c256: V, eval: &mut E)
where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    debug_assert_eq!(unpacked.len(), packed.len() * BYTES_PER_GOLDILOCKS);
    for (unpacked_chunk, &packed_elem) in unpacked.chunks_exact(BYTES_PER_GOLDILOCKS).zip_eq(packed) {
        let repacked = eval.polyval(unpacked_chunk, c256);
        eval.constraint_eq(packed_elem, repacked);
    }
}

pub(crate) fn eval_constraints<V, S, E>(
    vars: &StarkFrame<V, S, { pearl_columns::TOTAL }, { pearl_public::TOTAL }>,
    eval: &mut E,
    row_view: &mut RowView<V>,
) -> [V; pearl_columns::CUMSUM_TILE_LEN]
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let next_trace = vars.get_next_values();
    let c256 = eval.i32(256);
    let consts: [V; 17] = std::array::from_fn(|i| eval.u64(i as u64));
    let zero = consts[0];
    let climb = eval.i32(1 << BITS_PER_LIMB); // 2^13
    let climb_2 = eval.i32(1 << (2 * BITS_PER_LIMB)); // 2^26

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read A_ID_PREP
    debug_assert_eq!(row_view.offset, pearl_columns::AB_ID_PREP);
    let ab_id_prep = row_view.consume_single();
    let ab_id_limbs = row_view.consume_few(pearl_columns::AB_ID_LIMBS_LEN);
    debug_assert_eq!(pearl_columns::AB_ID_LIMBS_LEN, 4);
    // Each ID uses 2 limbs of BITS_PER_LIMB bits each
    let expected_a_id = eval.polyval(&ab_id_limbs[..2], climb);
    let expected_b_id = eval.polyval(&ab_id_limbs[2..], climb);
    let expected_ab_id = eval.polyval(&[expected_a_id, expected_b_id], climb_2);
    eval.constraint_eq(ab_id_prep, expected_ab_id);
    let a_id_prep = row_view.consume_single(); // Note: used through lookups
    let b_id_prep = row_view.consume_single(); // Note: used through lookups
    eval.constraint_eq(a_id_prep, expected_a_id);
    eval.constraint_eq(b_id_prep, expected_b_id);
    debug_assert_eq!(row_view.offset, pearl_columns::B_ID_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read A_NOISED
    debug_assert_eq!(row_view.offset, pearl_columns::A_NOISED);
    let a_noised = row_view.consume_few(pearl_columns::A_NOISED_LEN);
    debug_assert_eq!(row_view.offset, pearl_columns::A_NOISED_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read A_NOISED_UNPACK
    debug_assert_eq!(row_view.offset, pearl_columns::A_NOISED_UNPACK);
    let a_noised_unpack = row_view.consume_few(pearl_columns::A_NOISED_UNPACK_LEN);
    verify_packing_le(a_noised_unpack, a_noised, c256, eval);
    debug_assert_eq!(row_view.offset, pearl_columns::A_NOISED_UNPACK_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read B_NOISED
    debug_assert_eq!(row_view.offset, pearl_columns::B_NOISED);
    let b_noised = row_view.consume_few(pearl_columns::B_NOISED_LEN);
    debug_assert_eq!(row_view.offset, pearl_columns::B_NOISED_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read B_NOISED_UNPACK
    debug_assert_eq!(row_view.offset, pearl_columns::B_NOISED_UNPACK);
    let b_noised_unpack = row_view.consume_few(pearl_columns::B_NOISED_UNPACK_LEN);
    verify_packing_le(b_noised_unpack, b_noised, c256, eval);
    debug_assert_eq!(row_view.offset, pearl_columns::B_NOISED_UNPACK_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read CUMSUM_TILE
    debug_assert_eq!(row_view.offset, pearl_columns::CUMSUM_TILE);
    let cumsum_tile = row_view.consume_few(pearl_columns::CUMSUM_TILE_LEN);
    let next_cumsum_tile = &next_trace[pearl_columns::CUMSUM_TILE_RANGE];
    let next_a_noised_unpack = &next_trace[pearl_columns::A_NOISED_UNPACK_RANGE];
    let next_b_noised_unpack = &next_trace[pearl_columns::B_NOISED_UNPACK_RANGE];
    let next_is_reset_cumsum = next_trace[pearl_columns::IS_RESET_CUMSUM];
    let next_is_update_cumsum = next_trace[pearl_columns::IS_UPDATE_CUMSUM];
    let noised_a: [[V; TILE_D]; TILE_H] = std::array::from_fn(|i| std::array::from_fn(|k| next_a_noised_unpack[i * TILE_D + k]));
    let noised_b: [[V; TILE_D]; TILE_H] = std::array::from_fn(|j| std::array::from_fn(|k| next_b_noised_unpack[j * TILE_D + k]));
    for i in 0..TILE_H {
        for j in 0..TILE_H {
            let tile_idx = i * TILE_H + j;
            let prev_value = eval.mux(next_is_reset_cumsum, cumsum_tile[tile_idx], zero);
            let a_values = noised_a[i];
            let b_values = noised_b[j];
            let ip = eval.inner_product(&a_values, &b_values);
            let updated_jackpot = eval.add(prev_value, ip);
            let expected_jackpot = eval.mux(next_is_update_cumsum, prev_value, updated_jackpot);
            eval.constraint_eq(next_cumsum_tile[tile_idx], expected_jackpot);
        }
    }
    debug_assert_eq!(row_view.offset, pearl_columns::CUMSUM_TILE_END);

    cumsum_tile.try_into().unwrap()
}
