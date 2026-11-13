//! Prover-only batch-tree row injection. The default constructor keeps the original path.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::BatchMerkleTree;
use crate::hash::hash_types::{RichField, NUM_HASH_OUT_ELTS};
use crate::plonk::config::{GenericHashOut, Hasher};
use plonky2_maybe_rayon::{
    IndexedParallelIterator, MaybeIntoParIter, MaybeParIter, ParallelIterator,
};

impl<F: RichField, H: Hasher<F>> BatchMerkleTree<F, H> {
    /// Builds the same tree with parallel prover row injection and disposal.
    pub(crate) fn new_prover(leaves: Vec<Vec<Vec<F>>>, cap_height: usize) -> Self {
        Self::new_with_options::<true>(leaves, cap_height)
    }
}

pub(super) fn inject_at_equal_height<F: RichField, H: Hasher<F>>(
    cap: &[H::Hash],
    rows: &[Vec<F>],
) -> Vec<H::Hash> {
    cap.par_iter()
        .zip(rows)
        .map(|(hash, row)| {
            let fields = hash.to_vec();
            if fields.len() == NUM_HASH_OUT_ELTS && row.len() == NUM_HASH_OUT_ELTS {
                let mut input = [F::ZERO; 2 * NUM_HASH_OUT_ELTS];
                input[..NUM_HASH_OUT_ELTS].copy_from_slice(&fields);
                input[NUM_HASH_OUT_ELTS..].copy_from_slice(row);
                H::hash_or_noop(&input)
            } else {
                let mut input = fields;
                input.extend(row);
                H::hash_or_noop(&input)
            }
        })
        .collect()
}

pub(super) fn inject_rows<F: RichField, H: Hasher<F>>(
    cap: &[H::Hash],
    rows: &[Vec<F>],
) -> Vec<Vec<F>> {
    cap.par_iter()
        .with_min_len(1 << 13)
        .zip(rows)
        .map(|(hash, row)| {
            let mut fields = Vec::with_capacity(NUM_HASH_OUT_ELTS + row.len());
            fields.extend(hash.to_vec());
            fields.extend(row);
            fields
        })
        .collect()
}

pub(super) fn drop_rows<F: RichField>(rows: Vec<Vec<F>>) {
    rows.into_par_iter().with_min_len(1 << 13).for_each(drop);
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField as F;
    use crate::field::types::Field;

    #[test]
    fn prover_tree_matches_original_constructor() {
        fn check<H: Hasher<F>>() {
            for heights in [vec![5], vec![5, 3, 2], vec![5, 5, 3, 3, 2]] {
                let leaves: Vec<Vec<Vec<F>>> = heights
                    .iter()
                    .enumerate()
                    .map(|(group, &height)| {
                        (0..1 << height)
                            .map(|row| {
                                (0..[0, 3, 4, 7, 1][group])
                                    .map(|column| {
                                        F::from_canonical_usize(group * 1000 + row * 10 + column)
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect();
                for cap in 0..=*heights.last().unwrap() {
                    assert_eq!(
                        BatchMerkleTree::<F, H>::new_prover(leaves.clone(), cap),
                        BatchMerkleTree::<F, H>::new(leaves.clone(), cap)
                    );
                }
            }
        }
        check::<crate::hash::poseidon::PoseidonHash>();
        check::<crate::hash::keccak::KeccakHash<32>>();
    }
}
