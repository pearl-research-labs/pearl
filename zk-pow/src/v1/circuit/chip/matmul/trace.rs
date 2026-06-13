use itertools::Itertools;
use plonky2::hash::hash_types::RichField;
use plonky2_field::extension::Extendable;

use crate::v1::circuit::{
    chip::MatmulChipConfig,
    pearl_layout::{BITS_PER_LIMB, BYTES_PER_GOLDILOCKS, pearl_columns},
    pearl_noise::MMSlice,
    pearl_program::{TILE_D, TILE_H},
    pearl_trace::read_tile,
    utils::trace_utils::{RowBuilder, i64_pack_base, u64_pack_le, u64_unpack_le},
};

pub fn fill_row_trace<'a, F, const D: usize>(
    secret_strips: &MMSlice,
    noise: &MMSlice,
    local_cumsum: &mut [[i32; TILE_H]; TILE_H],
    config: &MatmulChipConfig,
    row_idx: usize,
    row_builder: &mut RowBuilder<'a, F>,
) where
    F: RichField + Extendable<D>,
{
    let row_logic = config.logic[row_idx];

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Fill AB_ID_LIMBS + A_ID + B_ID from AB_ID_PREP
    debug_assert_eq!(row_builder.offset, pearl_columns::AB_ID_PREP);
    let ab_id_prep = row_builder.dump_noop().to_canonical_u64();

    let ab_id_limbs = u64_unpack_le(ab_id_prep, BITS_PER_LIMB, 4);
    for &limb in &ab_id_limbs {
        row_builder.dump_u64(limb); // AB_ID_LIMBS
    }

    row_builder.dump_u64(u64_pack_le(&ab_id_limbs[..2], BITS_PER_LIMB)); // a_id
    row_builder.dump_u64(u64_pack_le(&ab_id_limbs[2..], BITS_PER_LIMB)); // b_id
    debug_assert_eq!(row_builder.offset, pearl_columns::B_ID_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Fill A_NOISED and A_NOISED_UNPACK
    debug_assert_eq!(row_builder.offset, pearl_columns::A_NOISED);
    let (a_tile, a_noise) = if let Some(a_dword) = row_logic.a_dword {
        (read_tile(secret_strips, a_dword), read_tile(noise, a_dword))
    } else {
        ([0i8; TILE_H * TILE_D], [0i8; TILE_H * TILE_D])
    };
    let a_noised = a_tile.iter().zip(a_noise.iter()).map(|(a, n)| a + n).collect_vec();
    for a_n_chunk in a_noised.chunks_exact(BYTES_PER_GOLDILOCKS) {
        row_builder.dump_i64(i64_pack_base(a_n_chunk, 256));
    }
    for &a_n_unpack in &a_noised {
        row_builder.dump_i64(a_n_unpack as i64);
    }
    debug_assert_eq!(row_builder.offset, pearl_columns::A_NOISED_UNPACK_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Fill B_NOISED and B_NOISED_UNPACK
    debug_assert_eq!(row_builder.offset, pearl_columns::B_NOISED);
    let (b_tile, b_noise) = if let Some(b_dword) = row_logic.b_dword {
        (read_tile(secret_strips, b_dword), read_tile(noise, b_dword))
    } else {
        ([0i8; TILE_H * TILE_D], [0i8; TILE_H * TILE_D])
    };
    let b_noised = b_tile.iter().zip(b_noise.iter()).map(|(b, n)| b + n).collect_vec();
    for b_n_chunk in b_noised.chunks_exact(BYTES_PER_GOLDILOCKS) {
        row_builder.dump_i64(i64_pack_base(b_n_chunk, 256));
    }
    for &b_n_unpack in &b_noised {
        row_builder.dump_i64(b_n_unpack as i64);
    }
    debug_assert_eq!(row_builder.offset, pearl_columns::B_NOISED_UNPACK_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Fill CUMSUM_TILE
    debug_assert_eq!(row_builder.offset, pearl_columns::CUMSUM_TILE);
    if row_logic.is_reset_cumsum {
        *local_cumsum = [[0i32; TILE_H]; TILE_H];
    }
    if row_logic.is_update_cumsum() {
        // matmul
        for i in 0..TILE_H {
            for j in 0..TILE_H {
                for k in 0..TILE_D {
                    let a_elem = a_noised[i * TILE_D + k] as i32;
                    let b_elem = b_noised[j * TILE_D + k] as i32;
                    local_cumsum[i][j] += a_elem * b_elem;
                }
            }
        }
    }
    // TILE_H × TILE_H
    for row in local_cumsum {
        for elem in row {
            row_builder.dump_i64(*elem as i64);
        }
    }
    debug_assert_eq!(row_builder.offset, pearl_columns::CUMSUM_TILE_END);
}
