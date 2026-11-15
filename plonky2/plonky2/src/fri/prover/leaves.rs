//! Shared leaf gathering for single- and multi-height FRI proofs.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::field::extension::{Extendable, FieldExtension};
use crate::hash::hash_types::RichField;
use crate::util::{log2_strict, reverse_bits};
use plonky2_maybe_rayon::*;

/// Flattens each arity-sized chunk of the bit-reversed codeword into a Merkle leaf.
/// Gather from natural-order values directly, avoiding a full-buffer permutation.
pub(crate) fn fri_leaves<F: RichField + Extendable<D>, const D: usize>(
    values: &[F::Extension],
    arity_bits: usize,
) -> Vec<Vec<F>> {
    let arity = 1 << arity_bits;
    let chunk_bits = log2_strict(values.len()) - arity_bits;
    (0..values.len() / arity)
        .into_par_iter()
        .map(|row| {
            let low = reverse_bits(row, chunk_bits);
            let mut leaf = Vec::with_capacity(arity * D);
            for k in 0..arity {
                let index = (reverse_bits(k, arity_bits) << chunk_bits) | low;
                leaf.extend_from_slice(&values[index].to_basefield_array());
            }
            leaf
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::extension::flatten;
    use crate::field::goldilocks_field::GoldilocksField as F;
    use crate::field::types::Field;
    use crate::util::reverse_index_bits_in_place;

    #[test]
    fn gathered_leaves_match_bit_reversed_chunks() {
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    for bits in 0..=10 {
                        let values: Vec<_> = (0..1 << bits)
                            .map(|i| {
                                <F as Extendable<2>>::Extension::from_basefield_array([
                                    F::from_canonical_usize(i),
                                    F::from_canonical_usize(i ^ 123),
                                ])
                            })
                            .collect();
                        let mut reversed = values.clone();
                        reverse_index_bits_in_place(&mut reversed);
                        for arity_bits in 0..=bits {
                            let expected: Vec<_> = reversed
                                .chunks(1 << arity_bits)
                                .map(flatten::<F, 2>)
                                .collect();
                            assert_eq!(fri_leaves::<F, 2>(&values, arity_bits), expected);
                        }
                    }
                });
        }
    }
}
