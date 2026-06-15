//! Matmul Chip
//! -----------
//! Rows with IS_UPDATE_CUMSUM load TILE_H × TILE_D tiles from A and B via RAM lookups
//! indexed by A_ID/B_ID. The lookup ensures consistency with the NOISED_PACKED values at
//! the corresponding MAT_ID rows.
//!
//! CUMSUM_TILE (TILE_H × TILE_H) accumulates inner products across multiple rows, one
//! TILE_D chunk at a time. IS_RESET_CUMSUM zeros the accumulator at the start of a new
//! dot product. IS_UPDATE_CUMSUM controls whether a row contributes to the accumulation,
//! allowing idle matmul rows while other chips are active.

use plonky2::hash::hash_types::RichField;
use plonky2_field::extension::Extendable;
use starky::evaluation_frame::StarkFrame;

mod constraints;
mod logic;
mod trace;

pub use logic::MatmulLogic;

use crate::v1::circuit::{
    pearl_layout::{pearl_columns, pearl_public},
    pearl_noise::MMSlice,
    pearl_program::TILE_H,
    utils::{air_utils::RowView, evaluator::Evaluator, trace_utils::RowBuilder},
};

#[derive(Debug, Clone, Default)]
pub struct MatmulChipConfig {
    logic: Vec<MatmulLogic>,
}
impl MatmulChipConfig {
    pub fn new(logic: Vec<MatmulLogic>) -> Self {
        Self { logic }
    }
}
pub struct MatmulControlFields<V: Copy> {
    pub is_reset_cumsum: V,
    pub is_update_cumsum: V,
}

impl<V: Copy> MatmulControlFields<V> {
    pub(crate) fn consume_fields(row_view: &mut RowView<V>) -> Self {
        let is_reset_cumsum = row_view.consume_single();
        let is_update_cumsum = row_view.consume_single();

        Self {
            is_reset_cumsum,
            is_update_cumsum,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MatmulChip {
    local_cumsum: [[i32; TILE_H]; TILE_H],
}

impl MatmulChip {
    pub(crate) fn fill_row_trace<'a, F, const D: usize>(
        &mut self,
        row_idx: usize,
        row_builder: &mut RowBuilder<'a, F>,
        secret_strips: &MMSlice,
        noise: &MMSlice,
        config: &MatmulChipConfig,
    ) where
        F: RichField + Extendable<D>,
    {
        trace::fill_row_trace(secret_strips, noise, &mut self.local_cumsum, config, row_idx, row_builder);
    }

    pub(crate) fn eval_constraints<V, S, E>(
        &self,
        vars: &StarkFrame<V, S, { pearl_columns::TOTAL }, { pearl_public::TOTAL }>,
        eval: &mut E,
        row_view: &mut RowView<V>,
    ) -> [V; pearl_columns::CUMSUM_TILE_LEN]
    where
        V: Copy + Default,
        S: Copy + Default,
        E: Evaluator<V, S>,
    {
        constraints::eval_constraints(vars, eval, row_view)
    }
}
