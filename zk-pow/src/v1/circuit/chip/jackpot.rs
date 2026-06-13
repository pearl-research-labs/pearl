//! Jackpot Chip
//! ------------
//! Reduces the h × w matrix of inner products (from matmul) into a 16-word JACKPOT_MSG
//! by XOR-folding tile results with bit rotations.
//!
//! When a matmul tile completes, its CUMSUM_TILE is dumped into CUMSUM_BUFFER
//! (IS_DUMP_CUMSUM_BUFFER). The buffer then cyclically rotates to previous rows
//! feeding one word at a time to the XOR step (allowing concurrent computation with CUMSUM_TILE).
//!
//! BIT_REG is a 32-bit register with three operations:
//!   - LOAD: bit_reg = jackpot_msg[idx].rotate_left(LROT_PER_TILE). The implicit left
//!     rotation is the natural per-rank accumulation that occurs every `rank` elements.
//!   - XOR: bit_reg ^= cumsum_buffer[0]. Folds in one tile result element, verified via
//!     bitwise XOR constraints on individual bits.
//!   - SHIFT3: bit_reg = bit_reg.rotate_right(3 * LROT_PER_TILE). Used for back-shift
//!     compensation.
//!
//! After XOR folding, BIT_REG is stored back to jackpot_msg[idx] with a configurable
//! rotation (Store0/Store1/Store2). Non-modified entries are constrained unchanged.
//!
//! Back-shift compensation: since each LOAD applies rotate_left(LROT_PER_TILE), earlier
//! subtiles' contributions accumulate extra rotations from subsequent loads. The last write
//! from non-final subtiles uses extra SHIFT3+store operations to pre-compensate, so that
//! the final jackpot value has the correct net rotation matching compute_jackpot.

use plonky2::hash::hash_types::RichField;
use plonky2_field::extension::Extendable;
use starky::evaluation_frame::StarkFrame;

mod constraints;
pub mod helper;
pub mod logic;
mod trace;

use crate::v1::circuit::{
    chip::JackpotLogic,
    pearl_layout::{pearl_columns, pearl_public},
    pearl_stark::PearlStarkConfig,
    utils::{air_utils::RowView, evaluator::Evaluator, trace_utils::RowBuilder},
};

#[derive(Debug, Clone, Default)]
pub struct JackpotChipConfig {
    logic: Vec<JackpotLogic>,
}
impl JackpotChipConfig {
    pub fn new(logic: Vec<JackpotLogic>) -> Self {
        Self { logic }
    }
}
pub struct JackpotControlFields<V: Copy> {
    pub is_load: V,
    pub is_xor: V,
    pub is_shift3: V,
    pub is_store0: V,
    pub is_store1: V,
    pub is_store2: V,
    pub is_dump_cumsum_buffer: V,
    pub jackpot_idx: [V; pearl_columns::JACKPOT_IDX_LEN],
}

impl<V: Copy> JackpotControlFields<V> {
    pub(crate) fn consume_fields(row_view: &mut RowView<V>) -> Self {
        let is_load = row_view.consume_single();
        let is_xor = row_view.consume_single();
        let is_shift3 = row_view.consume_single();
        let is_store0 = row_view.consume_single();
        let is_store1 = row_view.consume_single();
        let is_store2 = row_view.consume_single();
        let is_dump_cumsum_buffer = row_view.consume_single();
        let jackpot_idx = row_view.consume_few(pearl_columns::JACKPOT_IDX_LEN).try_into().unwrap();

        Self {
            is_load,
            is_xor,
            is_shift3,
            is_store0,
            is_store1,
            is_store2,
            is_dump_cumsum_buffer,
            jackpot_idx,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct JackpotChip {}

impl JackpotChip {
    pub fn fill_full_trace<F: RichField + Extendable<D>, const D: usize>(
        &self,
        chip_config: &JackpotChipConfig,
        _config: &PearlStarkConfig,
        trace: &mut Vec<[F; pearl_columns::TOTAL]>,
    ) {
        trace::fill_cumsum_buffer_xor_jackpot(trace, &chip_config.logic);
    }

    pub(crate) fn fill_row_trace<'a, F, const D: usize>(&self, _row_idx: usize, row_builder: &mut RowBuilder<'a, F>)
    where
        F: RichField + Extendable<D>,
    {
        // fill_full_trace responsible for filling
        // CUMSUM_BUFFER
        // JACKPOT_MSG
        // BIT_REG
        debug_assert_eq!(row_builder.offset, pearl_columns::CUMSUM_BUFFER);
        row_builder.offset = pearl_columns::BIT_REG_END;
        debug_assert_eq!(row_builder.offset, pearl_columns::BIT_REG_END);
    }

    pub(crate) fn eval_constraints<V, S, E>(
        &self,
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
        constraints::eval_constraints(vars, eval, row_view, cf, cumsum_tile);
    }
}
