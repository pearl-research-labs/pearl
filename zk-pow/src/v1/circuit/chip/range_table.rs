//! Generic lookup table chip for bounded integer ranges.
//!
//! Parameterized by column index, MIN, and MAX. The table column walks from
//! MIN to MAX with steps of 0 or 1 (enforced as a boolean delta), paired with
//! a frequency column for log-derivative lookup accounting.
//! Concrete instantiations: `URange8Chip` (0..255), `URange13Chip` (0..8191),
//! `IRange7P1Chip` (-64..64), `IRange8Chip` (-128..127).

use plonky2::hash::hash_types::RichField;
use plonky2_field::extension::Extendable;

use crate::v1::circuit::{
    pearl_layout::{BITS_PER_LIMB, pearl_columns},
    utils::{air_utils::RowView, evaluator::Evaluator, trace_utils::RowBuilder},
};

pub type URange8Chip = RangeTableChip<{ pearl_columns::URANGE8_TABLE }, 0, 255>;
pub type URange13Chip = RangeTableChip<{ pearl_columns::URANGE13_TABLE }, 0, { (1 << BITS_PER_LIMB) - 1 }>;
pub type IRange7P1Chip = RangeTableChip<{ pearl_columns::IRANGE7P1_TABLE }, -64, 64>;
pub type IRange8Chip = RangeTableChip<{ pearl_columns::IRANGE8_TABLE }, -128, 127>;

#[derive(Debug, Default, Clone)]
pub struct RangeTableChip<const COLUMN: usize, const MIN: i32, const MAX: i32> {}
impl<const COLUMN: usize, const MIN: i32, const MAX: i32> RangeTableChip<COLUMN, MIN, MAX> {
    pub fn new() -> Self {
        assert!(MIN < MAX);
        Self {}
    }
}

impl<const COLUMN: usize, const MIN: i32, const MAX: i32> RangeTableChip<COLUMN, MIN, MAX> {
    pub(crate) fn fill_row_trace<'a, F, const D: usize>(&self, row_idx: usize, row_builder: &mut RowBuilder<'a, F>)
    where
        F: RichField + Extendable<D>,
    {
        debug_assert_eq!(row_builder.offset, COLUMN);
        let idx8 = row_idx.min((MAX - MIN) as usize);
        row_builder.dump_i64(idx8 as i32 + MIN);
        row_builder.dump_noop();
    }
    pub(crate) fn eval_constraints<V, S, E>(&self, row_view: &mut RowView<V>, next_trace: &[V], eval: &mut E)
    where
        V: Copy,
        S: Copy,
        E: Evaluator<V, S>,
    {
        debug_assert_eq!(row_view.offset, COLUMN);
        let next_value = next_trace[row_view.offset];
        let value = row_view.consume_single();
        let _freq = row_view.consume_single(); // Skip FREQ column

        let min_val = eval.i32(MIN);
        let max_val = eval.i32(MAX);

        eval.constraint_first_row_eq(value, min_val);
        eval.constraint_last_row_eq(value, max_val);

        let diff = eval.sub(next_value, value);
        let is_diff_bool = eval.msub(diff, diff, diff);
        eval.constraint_transition(is_diff_bool);
    }
}
