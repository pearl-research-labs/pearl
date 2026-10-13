#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::hash::hash_types::{BytesHash, RichField};
use crate::hash::hashing::PlonkyPermutation;
use crate::plonk::config::Hasher;
use crate::util::serialization::Write;

pub const SPONGE_RATE: usize = 8;
pub const SPONGE_CAPACITY: usize = 4;
pub const SPONGE_WIDTH: usize = SPONGE_RATE + SPONGE_CAPACITY;

/// Blake3 pseudo-permutation used in the challenger.
/// Hashes the state using Blake3 XOF mode and fills output via rejection sampling.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct Blake3Permutation<F: RichField> {
    state: [F; SPONGE_WIDTH],
}

impl<F: RichField> AsRef<[F]> for Blake3Permutation<F> {
    fn as_ref(&self) -> &[F] {
        &self.state
    }
}

/// Similar to KeccakPermutation, but without the property that squeeze() determines the entire state.
impl<F: RichField> PlonkyPermutation<F> for Blake3Permutation<F> {
    const RATE: usize = SPONGE_RATE;
    const WIDTH: usize = SPONGE_WIDTH;

    fn new<I: IntoIterator<Item = F>>(elts: I) -> Self {
        let mut perm = Self {
            state: [F::default(); SPONGE_WIDTH],
        };
        perm.set_from_iter(elts, 0);
        perm
    }

    fn set_elt(&mut self, elt: F, idx: usize) {
        self.state[idx] = elt;
    }

    fn set_from_slice(&mut self, elts: &[F], start_idx: usize) {
        self.state[start_idx..start_idx + elts.len()].copy_from_slice(elts);
    }

    fn set_from_iter<I: IntoIterator<Item = F>>(&mut self, elts: I, start_idx: usize) {
        for (s, e) in self.state[start_idx..].iter_mut().zip(elts) {
            *s = e;
        }
    }

    fn permute(&mut self) {
        self.permute_n::<{ SPONGE_WIDTH }>();
    }

    fn permute_n<const N: usize>(&mut self) {
        debug_assert_eq!(F::BITS, 64);
        // Serialize state to bytes
        let mut state_bytes = [0u8; SPONGE_WIDTH * 8];
        for (chunk, field) in state_bytes
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(&self.state)
        {
            *chunk = field.to_canonical_u64().to_le_bytes();
        }

        let mut reader = blake3::Hasher::new().update(&state_bytes).finalize_xof();
        let mut idx = 0;
        let mut buf = [0u8; 64];
        while idx < N {
            reader.fill(&mut buf);
            for chunk in buf.as_chunks::<8>().0 {
                let word = u64::from_le_bytes(*chunk);
                if word < F::ORDER {
                    self.state[idx] = F::from_canonical_u64(word);
                    idx += 1;
                    if idx == N {
                        return;
                    }
                }
            }
        }
    }

    fn squeeze(&self) -> &[F] {
        &self.state[..Self::RATE]
    }
}

/// Blake3 hash function.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Blake3Hash<const N: usize>;

impl<F: RichField, const N: usize> Hasher<F> for Blake3Hash<N> {
    const HASH_SIZE: usize = N;
    type Hash = BytesHash<N>;
    type Permutation = Blake3Permutation<F>;

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        let mut buffer = Vec::with_capacity(input.len() * F::BITS.div_ceil(8));
        buffer.write_field_vec(input).unwrap();
        BytesHash(blake3::hash(&buffer).as_bytes()[..N].try_into().unwrap())
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&left.0);
        hasher.update(&right.0);
        BytesHash(hasher.finalize().as_bytes()[..N].try_into().unwrap())
    }
}
