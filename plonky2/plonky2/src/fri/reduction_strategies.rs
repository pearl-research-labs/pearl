#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use log::debug;
use serde::Serialize;
#[cfg(feature = "timing")]
use web_time::Instant;

use crate::hash::hash_types::RichField;

/// A method for deciding what arity to use at each reduction layer.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub enum FriReductionStrategy {
    /// Specifies the exact sequence of arities (expressed in bits) to use.
    Fixed(Vec<usize>),

    /// `ConstantArityBits(arity_bits, final_poly_bits)` applies reductions of arity `2^arity_bits`
    /// until the polynomial degree is less than or equal to `2^final_poly_bits` or until any further
    /// `arity_bits`-reduction makes the last FRI tree have height less than `cap_height`.
    /// This tends to work well in the recursive setting, as it avoids needing multiple configurations
    /// of gates used in FRI verification, such as `InterpolationGate`.
    ConstantArityBits(usize, usize),

    /// `MinSize(opt_max_arity_bits)` searches for an optimal sequence of reduction arities, with an
    /// optional max `arity_bits`. If this proof will have recursive proofs on top of it, a max
    /// `arity_bits` of 3 is recommended.
    MinSize(Option<usize>),

    /// `Ladder(boundaries)`: a strictly descending ladder of fold boundaries (degree bits).
    /// The schedule for `degree_bits` (which must be a boundary) is the ladder's *suffix*
    /// from `degree_bits` down to the last boundary: one fold per gap between consecutive
    /// boundaries, gaps split into steps of at most 3 bits (arity <= 8, the recursion-friendly
    /// cap). Unlike [`Self::Fixed`], [`Self::serialize`] emits the boundary list itself —
    /// independent of `degree_bits` — so transcripts absorbing the strategy agree across
    /// proofs of different degrees whose ladders coincide (the universal-verifier
    /// prerequisite: a shallower proof's schedule is exactly a deeper one's suffix).
    Ladder(Vec<usize>),
}

impl FriReductionStrategy {
    /// The arity of each FRI reduction step, expressed as the log2 of the actual arity.
    pub fn reduction_arity_bits(
        &self,
        mut degree_bits: usize,
        rate_bits: usize,
        cap_height: usize,
        num_queries: usize,
    ) -> Vec<usize> {
        match self {
            FriReductionStrategy::Fixed(reduction_arity_bits) => reduction_arity_bits.to_vec(),
            &FriReductionStrategy::ConstantArityBits(arity_bits, final_poly_bits) => {
                let mut result = Vec::new();
                while degree_bits > final_poly_bits
                    && degree_bits + rate_bits - arity_bits >= cap_height
                {
                    result.push(arity_bits);
                    assert!(degree_bits >= arity_bits);
                    degree_bits -= arity_bits;
                }
                result.shrink_to_fit();
                result
            }
            FriReductionStrategy::MinSize(opt_max_arity_bits) => {
                min_size_arity_bits(degree_bits, rate_bits, num_queries, *opt_max_arity_bits)
            }
            FriReductionStrategy::Ladder(boundaries) => {
                let start = boundaries
                    .iter()
                    .position(|&b| b == degree_bits)
                    .unwrap_or_else(|| {
                        panic!("degree {degree_bits} is not a ladder boundary ({boundaries:?})")
                    });
                let mut arities = Vec::new();
                for w in boundaries[start..].windows(2) {
                    assert!(w[0] > w[1], "ladder boundaries must be strictly descending");
                    let mut gap = w[0] - w[1];
                    while gap > 0 {
                        let step = gap.min(3);
                        arities.push(step);
                        gap -= step;
                    }
                }
                arities
            }
        }
    }

    pub fn serialize<F: RichField>(&self) -> Vec<F> {
        match self {
            FriReductionStrategy::Fixed(reduction_arity_bits) => core::iter::once(F::ZERO)
                .chain(
                    reduction_arity_bits
                        .iter()
                        .map(|&x| F::from_canonical_usize(x)),
                )
                .collect(),
            FriReductionStrategy::ConstantArityBits(arity_bits, final_poly_bits) => {
                vec![
                    F::ONE,
                    F::from_canonical_usize(*arity_bits),
                    F::from_canonical_usize(*final_poly_bits),
                ]
            }
            FriReductionStrategy::MinSize(opt_max_arity_bits) => {
                let max_arity = opt_max_arity_bits.unwrap_or(0);
                vec![F::TWO, F::from_canonical_usize(max_arity)]
            }
            FriReductionStrategy::Ladder(boundaries) => {
                core::iter::once(F::from_canonical_usize(3))
                    .chain(boundaries.iter().map(|&b| F::from_canonical_usize(b)))
                    .collect()
            }
        }
    }
}

fn min_size_arity_bits(
    degree_bits: usize,
    rate_bits: usize,
    num_queries: usize,
    opt_max_arity_bits: Option<usize>,
) -> Vec<usize> {
    // 2^4 is the largest arity we see in optimal reduction sequences in practice. For 2^5 to occur
    // in an optimal sequence, we would need a really massive polynomial.
    let max_arity_bits = opt_max_arity_bits.unwrap_or(4);

    #[cfg(feature = "timing")]
    let start = Instant::now();
    let (mut arity_bits, fri_proof_size) =
        min_size_arity_bits_helper(degree_bits, rate_bits, num_queries, max_arity_bits, vec![]);
    arity_bits.shrink_to_fit();

    #[cfg(feature = "timing")]
    debug!(
        "min_size_arity_bits took {:.3}s",
        start.elapsed().as_secs_f32()
    );
    debug!(
        "Smallest arity_bits {arity_bits:?} results in estimated FRI proof size of {fri_proof_size} elements",
    );

    arity_bits
}

/// Return `(arity_bits, fri_proof_size)`.
fn min_size_arity_bits_helper(
    degree_bits: usize,
    rate_bits: usize,
    num_queries: usize,
    global_max_arity_bits: usize,
    prefix: Vec<usize>,
) -> (Vec<usize>, usize) {
    let sum_of_arities: usize = prefix.iter().sum();
    let current_layer_bits = degree_bits + rate_bits - sum_of_arities;
    assert!(current_layer_bits >= rate_bits);

    let mut best_arity_bits = prefix.clone();
    let mut best_size = relative_proof_size(degree_bits, rate_bits, num_queries, &prefix);

    // The largest next_arity_bits to search. Note that any optimal arity sequence will be
    // monotonically non-increasing, as a larger arity will shrink more Merkle proofs if it occurs
    // earlier in the sequence.
    let max_arity_bits = prefix
        .last()
        .copied()
        .unwrap_or(global_max_arity_bits)
        .min(current_layer_bits - rate_bits);

    for next_arity_bits in 1..=max_arity_bits {
        let mut extended_prefix = prefix.clone();
        extended_prefix.push(next_arity_bits);

        let (arity_bits, size) = min_size_arity_bits_helper(
            degree_bits,
            rate_bits,
            num_queries,
            max_arity_bits,
            extended_prefix,
        );
        if size < best_size {
            best_arity_bits = arity_bits;
            best_size = size;
        }
    }

    (best_arity_bits, best_size)
}

/// Compute the approximate size of a FRI proof with the given reduction arities. Note that this
/// ignores initial evaluations, which aren't affected by arities, and some other minor
/// contributions. The result is measured in field elements.
fn relative_proof_size(
    degree_bits: usize,
    rate_bits: usize,
    num_queries: usize,
    arity_bits: &[usize],
) -> usize {
    const D: usize = 4;

    let mut current_layer_bits = degree_bits + rate_bits;

    let mut total_elems = 0;
    for arity_bits in arity_bits {
        let arity = 1 << arity_bits;

        // Add neighboring evaluations, which are extension field elements.
        total_elems += (arity - 1) * D * num_queries;
        // Add siblings in the Merkle path.
        total_elems += current_layer_bits * 4 * num_queries;

        current_layer_bits -= arity_bits;
    }

    // Add the final polynomial's coefficients.
    assert!(current_layer_bits >= rate_bits);
    let final_poly_len = 1 << (current_layer_bits - rate_bits);
    total_elems += D * final_poly_len;

    total_elems
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::Field;

    #[test]
    fn ladder_schedules_are_suffixes_with_constant_serialization() {
        type F = GoldilocksField;
        let ladder = FriReductionStrategy::Ladder(vec![21, 20, 19, 17, 13, 11, 10, 5]);

        // The schedule from a boundary is the ladder's suffix: gaps split into <= 3-bit steps.
        assert_eq!(
            ladder.reduction_arity_bits(21, 1, 4, 10),
            vec![1, 1, 2, 3, 1, 2, 1, 3, 2]
        );
        assert_eq!(
            ladder.reduction_arity_bits(17, 1, 4, 10),
            vec![3, 1, 2, 1, 3, 2]
        );
        assert_eq!(
            ladder.reduction_arity_bits(5, 1, 4, 10),
            Vec::<usize>::new()
        );
        let deep = ladder.reduction_arity_bits(21, 1, 4, 10);
        let shallow = ladder.reduction_arity_bits(13, 1, 4, 10);
        assert_eq!(&deep[deep.len() - shallow.len()..], shallow.as_slice());

        // Serialization is the boundary list (tag 3), independent of any degree.
        let expected: Vec<F> = [3, 21, 20, 19, 17, 13, 11, 10, 5]
            .into_iter()
            .map(F::from_canonical_usize)
            .collect();
        assert_eq!(ladder.serialize::<F>(), expected);
    }

    #[test]
    #[should_panic(expected = "not a ladder boundary")]
    fn ladder_rejects_off_ladder_degrees() {
        FriReductionStrategy::Ladder(vec![10, 8, 5]).reduction_arity_bits(9, 1, 4, 10);
    }
}
