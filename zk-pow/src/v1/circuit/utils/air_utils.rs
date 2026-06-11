use crate::v1::circuit::utils::{evaluator::Evaluator, trace_utils::index_to_pair};

#[derive(Debug, Copy, Clone)]
pub(crate) struct RowView<'a, P> {
    pub(crate) row: &'a [P],
    pub(crate) offset: usize,
}

impl<'a, P: Copy> RowView<'a, P> {
    pub(crate) fn new(row: &'a [P]) -> Self {
        Self { row, offset: 0 }
    }

    pub(crate) fn consume_single(&mut self) -> P {
        self.offset += 1;
        self.row[self.offset - 1]
    }

    pub(crate) fn consume_few<'s>(&mut self, num_values: usize) -> &'s [P]
    where
        'a: 's,
    {
        let start_idx = self.offset;
        self.offset += num_values;
        &self.row[start_idx..self.offset]
    }

    pub(crate) fn assert_end(&self) {
        debug_assert_eq!(
            self.offset,
            self.row.len(),
            "RowView not fully checked; remains [offset={}, row_len={})",
            self.offset,
            self.row.len()
        );
    }
}

/// Returns degree-2 indicator values for indices in range.
/// Each indicator[i] is 1 iff muxer_bits encodes index (range.start + i), else 0.
pub(crate) fn degree_2_indicators<V: Copy, S: Copy, E: Evaluator<V, S>>(
    eval: &mut E,
    muxer_bits: &[V],
    range: std::ops::Range<usize>,
) -> Vec<V> {
    let mut res = Vec::with_capacity(range.len());
    if !range.is_empty() {
        let sum_muxer_bits = eval.sum(muxer_bits);
        let c2 = eval.i32(2);
        let (mut i, mut j) = index_to_pair(range.start);
        for _ in range {
            if i == j {
                let is_active = muxer_bits[i];
                let is_participant = eval.msub(is_active, c2, sum_muxer_bits);
                res.push(eval.mul(is_participant, is_active));
                (i, j) = (0, j + 1);
            } else {
                res.push(eval.mul(muxer_bits[i], muxer_bits[j]));
                i += 1;
            }
        }
    }
    res
}
