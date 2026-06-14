use plonky2::hash::hash_types::RichField;
use plonky2_field::extension::Extendable;

use crate::{
    api::proof::Hash256,
    circuit::{
        chip::blake3::{
            Blake3ChipConfig,
            logic::{AuxDataType, MessageDataType},
            program::DWORD_SIZE,
        },
        pearl_layout::{BYTES_PER_GOLDILOCKS, NOISE_PACKING_BASE, pearl_columns},
        pearl_noise::MMSlice,
        pearl_preprocess::{read_dword_from_matrix, read_routing_dword},
        utils::trace_utils::{RowBuilder, field_to_i64, i64_pack_base, i64_unpack_base},
    },
};

pub struct AuxData {
    // aux data from PrivateProofParams
    pub external_msgs: Vec<[u8; 64]>,
    pub external_cvs: Vec<Hash256>,
}

impl AuxData {
    /// Read a single dword (8 bytes) from auxiliary data (message or CV).
    pub fn read_dword(&self, aux_type: AuxDataType, dword_idx: usize) -> [u8; DWORD_SIZE] {
        match aux_type {
            AuxDataType::Msg { aux_msg_idx } => {
                debug_assert!(dword_idx < 8, "dword_idx must be < 8 for auxiliary message");
                let msg = &self.external_msgs[aux_msg_idx];
                msg[dword_idx * DWORD_SIZE..(dword_idx + 1) * DWORD_SIZE].try_into().unwrap()
            }
            AuxDataType::Cv { aux_cv_idx } => {
                debug_assert!(dword_idx < 4, "dword_idx must be < 4 for auxiliary CV");
                let cv = &self.external_cvs[aux_cv_idx];
                cv[dword_idx * DWORD_SIZE..(dword_idx + 1) * DWORD_SIZE].try_into().unwrap()
            }
        }
    }
}

pub fn fill_row_trace<'a, F, const D: usize>(
    row_idx: usize,
    row_builder: &mut RowBuilder<'a, F>,
    secret_strips: &MMSlice,
    aux_data: &AuxData,
    blake_config: &Blake3ChipConfig,
) where
    F: RichField + Extendable<D>,
{
    let row_logic = blake_config.logic[row_idx];
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Fill MAT_UNPACK
    debug_assert_eq!(row_builder.offset, pearl_columns::MAT_UNPACK);
    let mat_unpack: [i8; DWORD_SIZE] = if let MessageDataType::Matrix { dword_id } = row_logic.data_source {
        read_dword_from_matrix(secret_strips, dword_id)
    } else {
        [0i8; DWORD_SIZE]
    };
    for i8_byte in mat_unpack {
        row_builder.dump_i64(i8_byte as i64);
    }
    debug_assert_eq!(row_builder.offset, pearl_columns::MAT_UNPACK_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Fill UINT8_DATA
    debug_assert_eq!(row_builder.offset, pearl_columns::UINT8_DATA);
    let uint8_data = if let MessageDataType::Matrix { .. } = row_logic.data_source {
        mat_unpack.map(|i8_byte| i8_byte as u8)
    } else if let MessageDataType::RoutingData {
        hotspot_idx,
        idx_in_block,
    } = row_logic.data_source
    {
        read_routing_dword(secret_strips, hotspot_idx, idx_in_block)
    } else if let MessageDataType::AuxiliaryData { aux_type, dword_idx } = row_logic.data_source {
        aux_data.read_dword(aux_type, dword_idx)
    } else {
        [0u8; DWORD_SIZE]
    };
    for u8_byte in uint8_data {
        row_builder.dump_u64(u8_byte as u64);
    }
    debug_assert_eq!(row_builder.offset, pearl_columns::UINT8_DATA_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Fill NOISE_UNPACK + NOISED_PACKED
    debug_assert_eq!(row_builder.offset, pearl_columns::NOISE_PACKED_PREP);
    let noise_packed_prep = field_to_i64(row_builder.dump_noop()); // NOISE_PACKED_PREP is preprocessed

    let noise_unpack = i64_unpack_base(noise_packed_prep, NOISE_PACKING_BASE, DWORD_SIZE);
    for &n in &noise_unpack {
        row_builder.dump_i64(n); // NOISE_UNPACK
    }
    debug_assert_eq!(row_builder.offset, pearl_columns::NOISE_UNPACK_END);

    // Fill NOISED_PACKED (2 columns, mat + noise packed with base 256)
    for i in 0..pearl_columns::NOISED_PACKED_LEN {
        let mat_chunk = &mat_unpack[i * BYTES_PER_GOLDILOCKS..(i + 1) * BYTES_PER_GOLDILOCKS];
        let noise_chunk = &noise_unpack[i * BYTES_PER_GOLDILOCKS..(i + 1) * BYTES_PER_GOLDILOCKS];
        let noised: Vec<i64> = mat_chunk.iter().zip(noise_chunk).map(|(&m, &n)| m as i64 + n).collect();
        row_builder.dump_i64(i64_pack_base(&noised, 256)); // NOISED_PACKED
    }
    debug_assert_eq!(row_builder.offset, pearl_columns::NOISED_PACKED_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MAT_FREQ
    debug_assert_eq!(row_builder.offset, pearl_columns::MAT_FREQ);
    row_builder.dump_noop(); // To be filled later by frequency computation
    debug_assert_eq!(row_builder.offset, pearl_columns::MAT_FREQ_END);
}
