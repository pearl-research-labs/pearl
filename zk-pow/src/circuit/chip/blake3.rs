//! Blake3 Chip
//! -----------
//! Each Blake3 compression takes 8 consecutive rows: 7 round rows plus a finalization row.
//! The 7 rounds perform standard Blake3 quarter-round operations. The 8th row is a pure
//! readout: it XORs the two state halves to produce CV_OUT, with no blake3-round computation.
//!
//! BLAKE3_MSG_BUFFER assembles the 64-byte message over the 8 rows. It shifts left by 2
//! elements each row, with new data loaded into the tail. Data sources vary per row: matrix
//! dwords, auxiliary data (Merkle siblings), previous CV outputs, or the jackpot message.
//! At the finalization row, the fully assembled buffer is checked for consistency against
//! the permuted BLAKE3_MSG.
//!
//! BLAKE3_MSG is permuted between rounds following the standard Blake3 message schedule.
//! At IS_NEW_BLAKE boundaries (start of a new compression), the permutation chain resets.
//!
//! The input CV (BLAKE3_CV) is selected from JOB_KEY (for keyed leaf compressions),
//! COMMITMENT_HASH (for jackpot hashing), or a previous row's CV_OUT (for chained/parent
//! compressions). CV_IN is loaded from a previous CV_OUT via logup lookup indexed by
//! CV_OR_TWEAK_PREP.
//!
//! When IS_HASH_A, IS_HASH_B, or IS_HASH_JACKPOT is set (always at the finalization row),
//! CV_OUT is constrained to match the corresponding public input.

use plonky2::hash::hash_types::RichField;
use plonky2_field::extension::Extendable;
use starky::evaluation_frame::StarkFrame;

use crate::circuit::{
    chip::blake3::logic::BlakeRoundLogic,
    pearl_layout::{pearl_columns, pearl_public},
    pearl_stark::PearlStarkConfig,
    utils::{air_utils::RowView, evaluator::Evaluator, trace_utils::RowBuilder},
};

pub(crate) mod blake3_air;
pub mod blake3_compress;
pub mod constraints;
pub mod logic;
pub mod program;
pub mod trace;

pub struct Blake3ControlFields<V: Copy> {
    pub is_use_job_key: V,
    pub is_use_commitment_hash: V,
    pub is_hash_a: V,
    pub is_hash_b: V,
    pub is_hash_routing: V,
    pub is_hash_jackpot: V,
    pub is_cv_in: V,
    pub is_new_blake: V,
    pub is_last_round: V,
    pub is_msg_bits: [V; 3],
    pub is_first_outer: V,
    pub is_second_outer: V,
    /// Reconstructed outer index values from OUTER_INDICES_PACKED_PREP; set by the control packer.
    pub outer_index_first: V,
    pub outer_index_second: V,
}

impl<V: Copy> Blake3ControlFields<V> {
    pub(crate) fn consume_fields(row_view: &mut RowView<V>) -> Self {
        let is_use_job_key = row_view.consume_single();
        let is_use_commitment_hash = row_view.consume_single();
        let is_hash_a = row_view.consume_single();
        let is_hash_b = row_view.consume_single();
        let is_hash_routing = row_view.consume_single();
        let is_hash_jackpot = row_view.consume_single();
        let is_cv_in = row_view.consume_single();
        let is_new_blake = row_view.consume_single();
        let is_last_round = row_view.consume_single();
        let is_msg_bits = row_view.consume_few(3).try_into().unwrap();
        let is_first_outer = row_view.consume_single();
        let is_second_outer = row_view.consume_single();

        Self {
            is_use_job_key,
            is_use_commitment_hash,
            is_hash_a,
            is_hash_b,
            is_hash_routing,
            is_hash_jackpot,
            is_cv_in,
            is_new_blake,
            is_last_round,
            is_msg_bits,
            is_first_outer,
            is_second_outer,
            // Placeholder values; must be completed via with_outer_indices before use.
            outer_index_first: is_first_outer,
            outer_index_second: is_second_outer,
        }
    }

    /// Returns a copy of self with the outer index values replaced by the given computed values.
    pub(crate) fn with_outer_indices(mut self, first: V, second: V) -> Self {
        self.outer_index_first = first;
        self.outer_index_second = second;
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct Blake3ChipConfig {
    pub logic: Vec<BlakeRoundLogic>,
}

impl Blake3ChipConfig {
    pub fn new(logic: Vec<BlakeRoundLogic>) -> Self {
        Self { logic }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Blake3Chip {}

impl Blake3Chip {
    pub(crate) fn eval_constraints<V, S, E>(
        &self,
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
        constraints::eval_constraints(vars, eval, row_view, cf, uint8_data)
    }

    pub(crate) fn fill_row_trace<'a, F, const D: usize>(&self, _row_idx: usize, row_builder: &mut RowBuilder<'a, F>)
    where
        F: RichField + Extendable<D>,
    {
        // fill_full_trace responsible for filling
        // BLAKE3_MSG_BUFFER
        // CV_OR_TWEAK_PREP
        // CV_IN
        // BLAKE3_MSG
        // BLAKE3_CV
        // BLAKE3_ROUND
        // CV_OUT
        debug_assert_eq!(row_builder.offset, pearl_columns::BLAKE3_MSG_BUFFER);
        row_builder.offset = pearl_columns::CV_OUT_FREQ_END;
        debug_assert_eq!(row_builder.offset, pearl_columns::CV_OUT_FREQ_END);
    }

    pub fn fill_full_trace<F: RichField + Extendable<D>, const D: usize>(
        &self,
        chip_config: &Blake3ChipConfig,
        config: &PearlStarkConfig,
        trace: &mut Vec<[F; pearl_columns::TOTAL]>,
    ) {
        trace::fill_blake_data(trace, &chip_config.logic, &config.compiled_public_params);
    }
}
