//! Input Chip
//! ----------
//! Each matrix-processing row loads a single dword (8 bytes) from a matrix strip, identified
//! by MAT_ID. The int7 values appear in MAT_UNPACK and serve two downstream consumers:
//!   - Blake3: a lookup converts int7 → uint8 in UINT8_DATA, which feeds the message buffer.
//!   - Matmul: noise from NOISE_PACKED_PREP is added to produce NOISED_PACKED, which feeds
//!     the matmul tile accumulation.
//!
//! This dual use ensures the data being hashed (for Merkle tree verification) and the data
//! being multiplied (for the inner product) are the same matrix elements.

mod constraints;
pub mod trace;

use plonky2::hash::hash_types::RichField;
use plonky2_field::extension::Extendable;

use crate::v1::circuit::{
    chip::{blake3::Blake3ChipConfig, input::trace::AuxData},
    pearl_layout::pearl_columns,
    pearl_noise::MMSlice,
    utils::{air_utils::RowView, evaluator::Evaluator, trace_utils::RowBuilder},
};

#[derive(Debug, Clone, Default)]
pub struct InputChip {}

impl InputChip {
    pub(crate) fn fill_row_trace<'a, F, const D: usize>(
        &self,
        row_idx: usize,
        row_builder: &mut RowBuilder<'a, F>,
        secret_strips: &MMSlice,
        aux_data: &AuxData,
        blake3_config: &Blake3ChipConfig,
    ) where
        F: RichField + Extendable<D>,
    {
        trace::fill_row_trace(row_idx, row_builder, secret_strips, aux_data, blake3_config);
    }

    pub(crate) fn eval_constraints<V, S, E>(&self, eval: &mut E, row_view: &mut RowView<V>) -> [V; pearl_columns::UINT8_DATA_LEN]
    where
        V: Copy + Default,
        S: Copy + Default,
        E: Evaluator<V, S>,
    {
        constraints::eval_constraints(eval, row_view)
    }
}
