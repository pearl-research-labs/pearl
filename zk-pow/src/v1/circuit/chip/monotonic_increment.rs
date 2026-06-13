//! Generic chip that constrains a column to be a zero-based monotonically
//! incrementing counter (0, 1, 2, …). Parameterized by the target column
//! index so a single implementation can serve any counter column.
//! `StarkRowChip` is the concrete instantiation used as the global row index.

use plonky2::hash::hash_types::RichField;
use plonky2_field::extension::Extendable;

use crate::v1::circuit::{
    pearl_layout::pearl_columns,
    utils::{air_utils::RowView, evaluator::Evaluator, trace_utils::RowBuilder},
};

pub type StarkRowChip = MonotonicIncrement<{ pearl_columns::STARK_ROW_IDX }>;

#[derive(Debug, Default, Clone)]
pub struct MonotonicIncrement<const COLUMN: usize> {}
impl<const COLUMN: usize> MonotonicIncrement<COLUMN> {
    pub fn new() -> Self {
        Self {}
    }
}

impl<const COLUMN: usize> MonotonicIncrement<COLUMN> {
    pub(crate) fn fill_row_trace<'a, F, const D: usize>(&self, row_idx: usize, row_builder: &mut RowBuilder<'a, F>)
    where
        F: RichField + Extendable<D>,
    {
        debug_assert_eq!(row_builder.offset, COLUMN);
        row_builder.dump_u64(row_idx as u64);
    }
    pub(crate) fn eval_constraints<V, S, E>(&self, row_view: &mut RowView<V>, next_trace: &[V], eval: &mut E)
    where
        V: Copy,
        S: Copy,
        E: Evaluator<V, S>,
    {
        let consts: [V; 17] = std::array::from_fn(|i| eval.u64(i as u64));
        let zero = consts[0];
        let one = consts[1];

        debug_assert_eq!(row_view.offset, COLUMN);
        let stark_row_idx = row_view.consume_single();
        let next_stark_row_idx = next_trace[COLUMN];
        eval.constraint_first_row_eq(stark_row_idx, zero);
        let expected_next_row_idx = eval.add(stark_row_idx, one);
        eval.constraint_transition_eq(expected_next_row_idx, next_stark_row_idx);
    }
}
