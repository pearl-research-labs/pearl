//! Contiguous prover rows and their bit-reversed digest leaves.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
use crate::plonk::config::{GenericConfig, GenericHashOut, Hasher};
use crate::util::{log2_strict, reverse_bits};
use plonky2_maybe_rayon::*;

/// Hash natural-order rows into bit-reversed Merkle leaves.
///
/// Each row is hashed independently on the prover worker pool.
pub(super) fn hash_leaf_rows<F, C, const D: usize>(rows: &RowMatrix<F>) -> Vec<Vec<F>>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    let height = rows.values.len() / rows.width;
    let bits = log2_strict(height);
    (0..height)
        .into_par_iter()
        .map(|index| C::Hasher::hash_no_pad(rows.row(reverse_bits(index, bits))).to_vec())
        .collect()
}

/// Prover-only raw evaluations in natural point order; Merkle leaves are bit-reversed.
#[derive(Eq, PartialEq, Debug)]
pub(super) struct RowMatrix<F> {
    width: usize,
    values: Vec<F>,
}

impl<F: RichField> RowMatrix<F> {
    /// Transpose nonempty, equal-length columns into contiguous rows.
    pub(super) fn from_columns(columns: &[Vec<F>]) -> Self {
        let width = columns.len();
        let mut values = vec![F::ZERO; width * columns[0].len()];
        // Read short contiguous runs from each column while filling adjacent rows.
        values
            .par_chunks_mut(64 * width)
            .enumerate()
            .for_each(|(block, chunk)| {
                for (r, row) in chunk.chunks_exact_mut(width).enumerate() {
                    for (c, value) in row.iter_mut().enumerate() {
                        *value = columns[c][64 * block + r];
                    }
                }
            });
        Self { width, values }
    }

    pub(super) fn row(&self, index: usize) -> &[F] {
        &self.values[index * self.width..(index + 1) * self.width]
    }
}
