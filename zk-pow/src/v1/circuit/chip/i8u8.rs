//! Lookup table chip for signed-to-unsigned 8-bit integer conversion.
//!
//! Enumerates all 256 (i8, u8) pairs by packing them as `signed * 256 + unsigned`
//! into a single field element (I8U8_TABLE). An auxiliary boolean column (I8U8_AUX)
//! transitions from 0 to 1 at the sign boundary (signed == -1 -> 0), splitting the
//! table into negative and non-negative halves. Constraints enforce monotonic
//! traversal from (-128, 128) to (127, 127) with a step of 257, except at the
//! sign-boundary wrap where the step is 1.

use plonky2::hash::hash_types::RichField;
use plonky2_field::extension::Extendable;

use crate::v1::circuit::{
    pearl_layout::pearl_columns,
    utils::{air_utils::RowView, evaluator::Evaluator, trace_utils::RowBuilder},
};

#[derive(Debug, Clone, Default)]
pub struct I8U8Chip {}

impl I8U8Chip {
    pub(crate) fn fill_row_trace<'a, F, const D: usize>(&self, row_idx: usize, row_builder: &mut RowBuilder<'a, F>)
    where
        F: RichField + Extendable<D>,
    {
        let idx8 = row_idx.min(255);
        let signed = idx8 as i64 - 128;
        let unsigned = signed.rem_euclid(256);
        row_builder.dump_i64(unsigned + signed * 256); // I8U8_TABLE
        row_builder.dump_i64(signed >= 0); // I8U8_AUX
        row_builder.dump_noop(); // I8U8_FREQ
        debug_assert_eq!(row_builder.offset, pearl_columns::I8U8_FREQ_END);
    }
    pub(crate) fn eval_constraints<V, S, E>(&self, row_view: &mut RowView<V>, next_trace: &[V], eval: &mut E)
    where
        V: Copy,
        S: Copy,
        E: Evaluator<V, S>,
    {
        let consts: [V; 17] = std::array::from_fn(|i| eval.u64(i as u64));

        let c256 = eval.i32(256);
        let c257 = eval.i32(257);
        let one = consts[1];

        debug_assert_eq!(row_view.offset, pearl_columns::I8U8_TABLE);
        let i8u8 = row_view.consume_single();
        let i8u8_aux = row_view.consume_single();
        let _freq = row_view.consume_single(); // skip FREQ column
        let next_i8u8 = next_trace[pearl_columns::I8U8_TABLE];
        let next_i8u8_aux = next_trace[pearl_columns::I8U8_AUX];
        let delta_i8u8 = eval.sub(next_i8u8, i8u8);
        let delta_aux = eval.sub(next_i8u8_aux, i8u8_aux);

        eval.constraint_bool(i8u8_aux);
        eval.constraint_first_row(i8u8_aux); // = 0 at first row
        eval.constraint_last_row_eq(i8u8_aux, one); // = 1 at last row
        let is_delta_i8u8_aux_bool = eval.msub(delta_aux, delta_aux, delta_aux);
        eval.constraint_transition(is_delta_i8u8_aux_bool); // i8u8_aux has shape 0000...1111

        let either_delta_aux_0_or_i8u8_is_m1 = eval.mad(i8u8, delta_aux, delta_aux);
        // i8u8_aux allowed & must transition 0 -> 1 when i8u8 == -1
        eval.constraint_transition(either_delta_aux_0_or_i8u8_is_m1);

        let i8u8_start = eval.i32(-128 * 256 + 128);
        let i8u8_end = eval.i32(127 * 256 + 127);
        eval.constraint_first_row_eq(i8u8, i8u8_start);
        eval.constraint_last_row_eq(i8u8, i8u8_end);
        // check delta_i8u8 is 257 except when i8u8 = -1, in which delta_i8u8 should be 1.
        let tot_delta = eval.mad(delta_aux, c256, delta_i8u8);
        let tot_delta_minus_257 = eval.sub(tot_delta, c257);
        let delta_delta = eval.sub(delta_i8u8, delta_aux);
        let deltas_equal_or_tot_delta_257 = eval.mul(delta_delta, tot_delta_minus_257);
        eval.constraint_transition(deltas_equal_or_tot_delta_257);

        debug_assert_eq!(row_view.offset, pearl_columns::I8U8_FREQ_END);
    }
}
