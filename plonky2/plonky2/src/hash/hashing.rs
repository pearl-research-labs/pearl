//! Concrete instantiation of a hash function.
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::fmt::Debug;

use crate::field::extension::Extendable;
use crate::field::types::Field;
use crate::hash::hash_types::{HashOut, HashOutTarget, RichField, NUM_HASH_OUT_ELTS};
use crate::hash::poseidon::{SPONGE_RATE, SPONGE_WIDTH};
use crate::iop::target::Target;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::AlgebraicHasher;
use plonky2_field::packed::PackedField;

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilder<F, D> {
    pub fn hash_or_noop<H: AlgebraicHasher<F>>(&mut self, inputs: Vec<Target>) -> HashOutTarget {
        let zero = self.zero();
        if inputs.len() <= NUM_HASH_OUT_ELTS {
            HashOutTarget::from_partial(&inputs, zero)
        } else {
            self.hash_n_to_hash_no_pad::<H>(inputs)
        }
    }

    pub fn hash_n_to_hash_no_pad<H: AlgebraicHasher<F>>(
        &mut self,
        inputs: Vec<Target>,
    ) -> HashOutTarget {
        HashOutTarget::from_vec(self.hash_n_to_m_no_pad::<H>(inputs, NUM_HASH_OUT_ELTS))
    }

    pub fn hash_n_to_m_no_pad<H: AlgebraicHasher<F>>(
        &mut self,
        inputs: Vec<Target>,
        num_outputs: usize,
    ) -> Vec<Target> {
        let zero = self.zero();
        let mut state = H::AlgebraicPermutation::new(core::iter::repeat(zero));

        // Absorb all input chunks.
        for input_chunk in inputs.chunks(H::AlgebraicPermutation::RATE) {
            // Overwrite the first r elements with the inputs. This differs from a standard sponge,
            // where we would xor or add in the inputs. This is a well-known variant, though,
            // sometimes called "overwrite mode".
            state.set_from_slice(input_chunk, 0);
            state = self.permute::<H>(state);
        }

        // Squeeze until we have the desired number of outputs.
        let mut outputs = Vec::with_capacity(num_outputs);
        loop {
            for &s in state.squeeze() {
                outputs.push(s);
                if outputs.len() == num_outputs {
                    return outputs;
                }
            }
            state = self.permute::<H>(state);
        }
    }
}

/// Permutation that can be used in the sponge construction for an algebraic hash.
pub trait PlonkyPermutation<T: Copy + Default>:
    AsRef<[T]> + Copy + Debug + Default + Eq + Sync + Send
{
    const RATE: usize;
    const WIDTH: usize;

    /// Initialises internal state with values from `iter` until
    /// `iter` is exhausted or `Self::WIDTH` values have been
    /// received; remaining state (if any) initialised with
    /// `T::default()`. To initialise remaining elements with a
    /// different value, instead of your original `iter` pass
    /// `iter.chain(core::iter::repeat(F::from_canonical_u64(12345)))`
    /// or similar.
    fn new<I: IntoIterator<Item = T>>(iter: I) -> Self;

    /// Set idx-th state element to be `elt`. Panics if `idx >= WIDTH`.
    fn set_elt(&mut self, elt: T, idx: usize);

    /// Set state element `i` to be `elts[i] for i =
    /// start_idx..start_idx + n` where `n = min(elts.len(),
    /// WIDTH-start_idx)`. Panics if `start_idx > WIDTH`.
    fn set_from_iter<I: IntoIterator<Item = T>>(&mut self, elts: I, start_idx: usize);

    /// Same semantics as for `set_from_iter` but probably faster than
    /// just calling `set_from_iter(elts.iter())`.
    fn set_from_slice(&mut self, elts: &[T], start_idx: usize);

    /// Apply permutation to internal state
    fn permute(&mut self);

    /// Apply permutation, but fills only the first N elements of state.
    /// After that, the object should not be used except for reading the first N elements.
    fn permute_n<const N: usize>(&mut self) {
        self.permute();
    }

    /// Return a slice of `RATE` elements
    fn squeeze(&self) -> &[T];
}

/// A one-way compression function which takes two ~256 bit inputs and returns a ~256 bit output.
pub fn compress<F: Field, P: PlonkyPermutation<F>>(x: HashOut<F>, y: HashOut<F>) -> HashOut<F> {
    // TODO: With some refactoring, this function could be implemented as
    // hash_n_to_m_no_pad(chain(x.elements, y.elements), NUM_HASH_OUT_ELTS).

    debug_assert_eq!(x.elements.len(), NUM_HASH_OUT_ELTS);
    debug_assert_eq!(y.elements.len(), NUM_HASH_OUT_ELTS);
    debug_assert!(P::RATE >= NUM_HASH_OUT_ELTS);

    let mut perm = P::new(core::iter::repeat(F::ZERO));
    perm.set_from_slice(&x.elements, 0);
    perm.set_from_slice(&y.elements, NUM_HASH_OUT_ELTS);

    perm.permute();

    HashOut {
        elements: perm.squeeze()[..NUM_HASH_OUT_ELTS].try_into().unwrap(),
    }
}

/// Hash a message without any padding step. Note that this can enable length-extension attacks.
/// However, it is still collision-resistant in cases where the input has a fixed length.
pub fn hash_n_to_m_no_pad<F: RichField, P: PlonkyPermutation<F>>(
    inputs: &[F],
    num_outputs: usize,
) -> Vec<F> {
    let mut perm = P::new(core::iter::repeat(F::ZERO));

    // Absorb all input chunks.
    for input_chunk in inputs.chunks(P::RATE) {
        perm.set_from_slice(input_chunk, 0);
        perm.permute();
    }

    // Squeeze until we have the desired number of outputs.
    let mut outputs = Vec::new();
    loop {
        for &item in perm.squeeze() {
            outputs.push(item);
            if outputs.len() == num_outputs {
                return outputs;
            }
        }
        perm.permute();
    }
}

pub fn hash_n_to_hash_no_pad<F: RichField, P: PlonkyPermutation<F>>(inputs: &[F]) -> HashOut<F> {
    HashOut::from_vec(hash_n_to_m_no_pad::<F, P>(inputs, NUM_HASH_OUT_ELTS))
}

/// Batched, packed-field version of [`hash_n_to_hash_no_pad`] for Poseidon
/// over `F::Packable::Packing`.
///
/// Hashes `P::WIDTH` independent messages of the same length in lockstep,
/// one message per SIMD lane, using the packed Poseidon permutation. For
/// wide messages (many RATE-sized chunks) this amortizes the permutation
/// cost across lanes; for the Merkle-leaf use case (one RATE chunk) it
/// reduces the hashing work per leaf to roughly one packed permutation.
///
/// Requires every message to have the same length (fixed-width leaves);
/// lengths need not be a multiple of the sponge rate — the final partial
/// chunk overwrites only the first `chunk_len` state slots, exactly like
/// the scalar sponge.
pub fn poseidon_hash_batch_no_pad<F, P>(messages: &[&[F]]) -> Vec<HashOut<F>>
where
    F: RichField,
    P: PackedField<Scalar = F>,
{
    assert!(!messages.is_empty());
    assert!(messages.len() <= P::WIDTH);
    let len = messages[0].len();
    assert!(messages.iter().all(|m| m.len() == len));

    let width = messages.len();
    debug_assert!(width <= P::WIDTH);
    let mut state = [P::ZEROS; SPONGE_WIDTH];

    // Absorb all input chunks, overwriting the sponge rate portion like
    // `hash_n_to_m_no_pad`. A partial final chunk overwrites only its own
    // slots; the rest keep their pre-permutation values, matching the
    // scalar sponge. Lane k absorbs messages[k]; unused lanes stay zero,
    // which keeps them independent of the hashed lanes.
    let mut lane = vec![F::ZERO; P::WIDTH];
    for chunk_start in (0..len).step_by(SPONGE_RATE) {
        let chunk_len = SPONGE_RATE.min(len - chunk_start);
        for i in 0..chunk_len {
            for (k, msg) in messages.iter().enumerate() {
                lane[k] = msg[chunk_start + i];
            }
            state[i] = *P::from_slice(&lane);
        }
        state = F::poseidon_packed_base(state);
    }

    // Squeeze NUM_HASH_OUT_ELTS base-field elements per lane.
    let mut outs = Vec::with_capacity(width);
    for k in 0..width {
        let mut elements = [F::ZERO; NUM_HASH_OUT_ELTS];
        for (i, elem) in elements.iter_mut().enumerate() {
            *elem = state[i].as_slice()[k];
        }
        outs.push(HashOut { elements });
    }
    outs
}

#[cfg(test)]
mod packed_tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::hash::poseidon::{Poseidon, PoseidonHash};
    use crate::plonk::config::Hasher;
    use plonky2_field::packable::Packable;

    type F = GoldilocksField;
    type P = <F as Packable>::Packing;

    #[test]
    fn packed_batch_hash_matches_scalar() {
        // RATE-multiple and partial-chunk lengths (v4 digest leaf widths
        // like 237 are not RATE multiples; the final partial chunk must
        // overwrite only its own sponge slots).
        for len in [
            SPONGE_RATE,
            2 * SPONGE_RATE,
            3 * SPONGE_RATE,
            237,
            222,
            339,
            5,
        ] {
            let msgs: Vec<Vec<F>> = (0..P::WIDTH)
                .map(|k| {
                    (0..len)
                        .map(|i| F::from_canonical_u64((k as u64) * 1009 + i as u64))
                        .collect()
                })
                .collect();
            let refs: Vec<&[F]> = msgs.iter().map(|m| &m[..]).collect();
            let packed = poseidon_hash_batch_no_pad::<F, P>(&refs);
            for (k, msg) in msgs.iter().enumerate() {
                let scalar = PoseidonHash::hash_no_pad(msg);
                assert_eq!(packed[k], scalar, "len={} lane={}", len, k);
            }
        }
    }

    #[test]
    fn packed_permute_matches_scalar() {
        // The packed permutation with a single active lane must equal the
        // scalar permutation (unused lanes are zero, and Poseidon's zero
        // state is a fixed point, so lane 0 is unaffected by idle lanes).
        let input: [F; SPONGE_WIDTH] =
            core::array::from_fn(|i| F::from_canonical_u64(17 * i as u64 + 1));
        let scalar = F::poseidon(input);

        let mut state = [P::ZEROS; SPONGE_WIDTH];
        for (i, x) in input.iter().enumerate() {
            state[i] = P::from(*x);
        }
        let packed = F::poseidon_packed_base(state);
        for i in 0..SPONGE_WIDTH {
            assert_eq!(packed[i].as_slice()[0], scalar[i]);
        }
    }
}
