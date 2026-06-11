//! Chip for unpacking and constraining the CONTROL_PREP column.
//!
//! CONTROL_PREP is a single field element that bit-packs all per-row control
//! flags (matmul, blake3, jackpot) together with the matrix ID (MAT_ID).
//! The trace side unpacks CONTROL_PREP into individual boolean flags and
//! MAT_ID limbs; the constraint side re-packs them via `polyval` and checks
//! equality with the original, ensuring every control bit is boolean and
//! MAT_ID is consistently derived.

use plonky2::hash::hash_types::RichField;
use plonky2_field::extension::Extendable;

use crate::v1::circuit::{
    chip::{JackpotControlFields, MatmulControlFields, blake3::Blake3ControlFields},
    pearl_layout::{BITS_PER_LIMB, pearl_columns},
    utils::{
        air_utils::RowView,
        evaluator::Evaluator,
        trace_utils::{RowBuilder, u64_unpack_le},
    },
};

#[derive(Debug, Default, Clone)]
pub struct ControlAndMatIDPackedChip {}
impl ControlAndMatIDPackedChip {
    pub fn new() -> Self {
        Self {}
    }
}

impl ControlAndMatIDPackedChip {
    pub(crate) fn fill_row_trace<'a, F, const D: usize>(&self, _row_idx: usize, row_builder: &mut RowBuilder<'a, F>)
    where
        F: RichField + Extendable<D>,
    {
        let limb_mask = (1usize << BITS_PER_LIMB) - 1;

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Fill control flags + MAT_ID_LIMBS
        // CONTROL_PREP packing: control bits + MAT_ID
        row_builder.offset = pearl_columns::CONTROL_PREP;
        let control = row_builder.dump_noop().to_canonical_u64();
        // Unpack control bits (first 29 bits)
        let bits = u64_unpack_le(control, 1, pearl_columns::JACKPOT_IDX_END - pearl_columns::IS_RESET_CUMSUM);
        for &bit in bits.iter() {
            row_builder.dump_u64(bit);
        }
        debug_assert_eq!(row_builder.offset, pearl_columns::JACKPOT_IDX_END);

        // Fill MAT_ID_LIMBS
        let mat_id_from_control = control >> bits.len();
        let mat_id_limb0 = mat_id_from_control & limb_mask as u64;
        let mat_id_limb1 = (mat_id_from_control >> BITS_PER_LIMB) & limb_mask as u64;
        row_builder.dump_u64(mat_id_limb0);
        row_builder.dump_u64(mat_id_limb1);
        debug_assert_eq!(row_builder.offset, pearl_columns::MAT_ID_LIMBS_END);

        // Fill MAT_ID
        debug_assert_eq!(row_builder.offset, pearl_columns::MAT_ID);
        let mat_id = mat_id_limb0 + (mat_id_limb1 << BITS_PER_LIMB);
        row_builder.dump_u64(mat_id);
        debug_assert_eq!(row_builder.offset, pearl_columns::MAT_ID_END);
    }
    pub(crate) fn eval_constraints<V, S, E>(
        &self,
        row_view: &mut RowView<V>,
        eval: &mut E,
    ) -> (MatmulControlFields<V>, Blake3ControlFields<V>, JackpotControlFields<V>)
    where
        V: Copy,
        S: Copy,
        E: Evaluator<V, S>,
    {
        let consts: [V; 17] = std::array::from_fn(|i| eval.u64(i as u64));
        let climb = eval.i32(1 << BITS_PER_LIMB); // 2^13

        debug_assert_eq!(row_view.offset, pearl_columns::CONTROL_PREP);
        let control_prep = row_view.consume_single();

        let matmul_cf = MatmulControlFields::consume_fields(row_view);
        let blake3_cf = Blake3ControlFields::consume_fields(row_view);
        let jackpot_cf = JackpotControlFields::consume_fields(row_view);

        let mat_id_limbs = row_view.consume_few(pearl_columns::MAT_ID_LIMBS_LEN);

        // Check CONTROL_PREP packing
        let control_bits = [
            &[
                matmul_cf.is_reset_cumsum,
                matmul_cf.is_update_cumsum,
                blake3_cf.is_use_job_key,
                blake3_cf.is_use_commitment_hash,
                blake3_cf.is_hash_a,
                blake3_cf.is_hash_b,
                blake3_cf.is_hash_jackpot,
                blake3_cf.is_cv_in,
                blake3_cf.is_new_blake,
                blake3_cf.is_last_round,
                blake3_cf.is_msg_mat,
                blake3_cf.is_msg_jackpot,
                blake3_cf.is_msg_aux_data,
                blake3_cf.is_msg_cv,
                jackpot_cf.is_load,
                jackpot_cf.is_xor,
                jackpot_cf.is_shift3,
                jackpot_cf.is_store0,
                jackpot_cf.is_store1,
                jackpot_cf.is_store2,
                jackpot_cf.is_dump_cumsum_buffer,
            ][..],
            &jackpot_cf.jackpot_idx,
        ]
        .concat();
        for &bit in control_bits.iter() {
            eval.constraint_bool(bit);
        }

        let expected_mat_id = eval.polyval(mat_id_limbs, climb);
        let flags_pack = eval.polyval(&[&control_bits[..], &[expected_mat_id]].concat(), consts[2]);
        eval.constraint_eq(control_prep, flags_pack);
        debug_assert_eq!(row_view.offset, pearl_columns::MAT_ID_LIMBS_END);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Read MAT_ID (derived from CONTROL_PREP, used for tile lookups)
        debug_assert_eq!(row_view.offset, pearl_columns::MAT_ID);
        let mat_id = row_view.consume_single(); // Could have used MAT_ID_LIMBS directly.
        eval.constraint_eq(mat_id, expected_mat_id);
        debug_assert_eq!(row_view.offset, pearl_columns::MAT_ID_END);

        (matmul_cf, blake3_cf, jackpot_cf)
    }
}
