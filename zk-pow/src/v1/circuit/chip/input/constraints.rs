use crate::v1::circuit::{
    pearl_layout::{BYTES_PER_GOLDILOCKS, NOISE_PACKING_BASE, pearl_columns},
    utils::{air_utils::RowView, evaluator::Evaluator},
};
pub fn eval_constraints<V, S, E>(eval: &mut E, row_view: &mut RowView<V>) -> [V; pearl_columns::UINT8_DATA_LEN]
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let c_noise_base = eval.i32(NOISE_PACKING_BASE as i32);
    let c256 = eval.i32(256);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read MAT_UNPACK
    debug_assert_eq!(row_view.offset, pearl_columns::MAT_UNPACK);
    let mat_unpack = row_view.consume_few(pearl_columns::MAT_UNPACK_LEN);
    debug_assert_eq!(row_view.offset, pearl_columns::MAT_UNPACK_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read UINT8_DATA (MAT_UNPACK as u8 if IS_MSG_MAT, otherwise auxiliary data, checked through lookups)
    debug_assert_eq!(row_view.offset, pearl_columns::UINT8_DATA);
    let uint8_data = row_view.consume_few(pearl_columns::UINT8_DATA_LEN);
    debug_assert_eq!(row_view.offset, pearl_columns::UINT8_DATA_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read NOISE_PACKED_PREP
    debug_assert_eq!(row_view.offset, pearl_columns::NOISE_PACKED_PREP);
    let noise_packed = row_view.consume_single();
    let noise_unpack = row_view.consume_few(pearl_columns::NOISE_UNPACK_LEN);
    // pack 8 -64..=64 elements into goldilocks. Correct noise will turn out to be -63..=63.
    let noise_repacked = eval.polyval(noise_unpack, c_noise_base);
    eval.constraint_eq(noise_packed, noise_repacked);
    debug_assert_eq!(row_view.offset, pearl_columns::NOISE_UNPACK_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read NOISED_PACKED
    // Check: NOISED_PACKED = pack(MAT_UNPACK) + NOISE_PACKED_PREP
    debug_assert_eq!(row_view.offset, pearl_columns::NOISED_PACKED);
    let noised_packed = row_view.consume_few(pearl_columns::NOISED_PACKED_LEN);
    for (i, &noised) in noised_packed.iter().enumerate() {
        let mat_chunk = &mat_unpack[i * BYTES_PER_GOLDILOCKS..(i + 1) * BYTES_PER_GOLDILOCKS];
        let mat_packed = eval.polyval(mat_chunk, c256);
        let noise_chunk = &noise_unpack[i * BYTES_PER_GOLDILOCKS..(i + 1) * BYTES_PER_GOLDILOCKS];
        let noise_packed = eval.polyval(noise_chunk, c256);
        let expected_noised = eval.add(mat_packed, noise_packed);
        eval.constraint_eq(noised, expected_noised);
    }
    debug_assert_eq!(row_view.offset, pearl_columns::NOISED_PACKED_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read MAT_FREQ
    debug_assert_eq!(row_view.offset, pearl_columns::MAT_FREQ);
    let _mat_freq = row_view.consume_single();
    debug_assert_eq!(row_view.offset, pearl_columns::MAT_FREQ_END);

    uint8_data.try_into().unwrap()
}
