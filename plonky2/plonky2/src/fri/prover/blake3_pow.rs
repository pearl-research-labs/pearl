//! Batched native Blake3 grinding. Verifiers use the existing permutation.
//!
//! Each lane hashes the same 96-byte challenger state with a different nonce.
//! The nonce is in the first 64-byte block. The second block contains 32 bytes;
//! its root compression produces the first 64 XOF bytes. Noncanonical output
//! words fall back to the existing permutation, including further XOF blocks.

#[cfg(feature = "parallel")]
use plonky2_maybe_rayon::*;

use crate::hash::hash_types::RichField;
use crate::hash::hashing::PlonkyPermutation;

const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];
const SCHEDULE: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

// The four disjoint G operations in each half of a BLAKE3 round.
const COLUMN_G: [(usize, usize, usize, usize); 4] =
    [(0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15)];
const DIAGONAL_G: [(usize, usize, usize, usize); 4] =
    [(0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14)];

// Transposed u32 lanes. The same compression code serves portable and AVX2 lanes.
trait Words: Copy {
    const LANES: usize;
    fn splat(x: u32) -> Self;
    fn from_fn(f: impl FnMut(usize) -> u32) -> Self;
    fn add(self, rhs: Self) -> Self;
    fn xor(self, rhs: Self) -> Self;
    fn and(self, rhs: Self) -> Self;
    fn or(self, rhs: Self) -> Self;
    fn rotate<const N: u32>(self) -> Self;
    fn eq(self, rhs: Self) -> Self;
    fn le(self, rhs: Self) -> Self;
    fn mask(self) -> u64;
    #[inline(always)]
    fn mix(s: &mut [Self; 16], a: usize, b: usize, c: usize, d: usize, x: Self, y: Self) {
        s[a] = s[a].add(s[b]).add(x);
        s[d] = s[d].xor(s[a]).rotate::<16>();
        s[c] = s[c].add(s[d]);
        s[b] = s[b].xor(s[c]).rotate::<12>();
        s[a] = s[a].add(s[b]).add(y);
        s[d] = s[d].xor(s[a]).rotate::<8>();
        s[c] = s[c].add(s[d]);
        s[b] = s[b].xor(s[c]).rotate::<7>();
    }
}

#[derive(Clone, Copy)]
struct Portable<const L: usize>([u32; L]);
impl<const L: usize> Words for Portable<L> {
    const LANES: usize = L;
    #[inline(always)]
    fn splat(x: u32) -> Self {
        Self([x; L])
    }
    #[inline(always)]
    fn from_fn(f: impl FnMut(usize) -> u32) -> Self {
        Self(core::array::from_fn(f))
    }
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_fn(|i| self.0[i].wrapping_add(rhs.0[i]))
    }
    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        Self::from_fn(|i| self.0[i] ^ rhs.0[i])
    }
    #[inline(always)]
    fn and(self, rhs: Self) -> Self {
        Self::from_fn(|i| self.0[i] & rhs.0[i])
    }
    #[inline(always)]
    fn or(self, rhs: Self) -> Self {
        Self::from_fn(|i| self.0[i] | rhs.0[i])
    }
    #[inline(always)]
    fn rotate<const N: u32>(self) -> Self {
        Self::from_fn(|i| self.0[i].rotate_right(N))
    }
    #[inline(always)]
    fn eq(self, rhs: Self) -> Self {
        Self::from_fn(|i| 0u32.wrapping_sub((self.0[i] == rhs.0[i]) as u32))
    }
    #[inline(always)]
    fn le(self, rhs: Self) -> Self {
        Self::from_fn(|i| 0u32.wrapping_sub((self.0[i] <= rhs.0[i]) as u32))
    }
    #[inline(always)]
    fn mask(self) -> u64 {
        self.0
            .iter()
            .enumerate()
            .fold(0, |bits, (i, x)| bits | (((x >> 31) as u64) << i))
    }
    // Keeping each lane's operations together lets LLVM vectorize the portable loop.
    #[inline(always)]
    fn mix(s: &mut [Self; 16], a: usize, b: usize, c: usize, d: usize, x: Self, y: Self) {
        for i in 0..L {
            s[a].0[i] = s[a].0[i].wrapping_add(s[b].0[i]).wrapping_add(x.0[i]);
            s[d].0[i] = (s[d].0[i] ^ s[a].0[i]).rotate_right(16);
            s[c].0[i] = s[c].0[i].wrapping_add(s[d].0[i]);
            s[b].0[i] = (s[b].0[i] ^ s[c].0[i]).rotate_right(12);
            s[a].0[i] = s[a].0[i].wrapping_add(s[b].0[i]).wrapping_add(y.0[i]);
            s[d].0[i] = (s[d].0[i] ^ s[a].0[i]).rotate_right(8);
            s[c].0[i] = s[c].0[i].wrapping_add(s[d].0[i]);
            s[b].0[i] = (s[b].0[i] ^ s[c].0[i]).rotate_right(7);
        }
    }
}

#[inline(always)]
fn g<W: Words>(s: &mut [W; 16], a: usize, b: usize, c: usize, d: usize, x: W, y: W) {
    W::mix(s, a, b, c, d, x, y);
}

#[inline(always)]
fn round<W: Words>(s: &mut [W; 16], m: &[W; 16], r: usize) {
    let p = SCHEDULE[r];
    g(s, 0, 4, 8, 12, m[p[0]], m[p[1]]);
    g(s, 1, 5, 9, 13, m[p[2]], m[p[3]]);
    g(s, 2, 6, 10, 14, m[p[4]], m[p[5]]);
    g(s, 3, 7, 11, 15, m[p[6]], m[p[7]]);
    g(s, 0, 5, 10, 15, m[p[8]], m[p[9]]);
    g(s, 1, 6, 11, 12, m[p[10]], m[p[11]]);
    g(s, 2, 7, 8, 13, m[p[12]], m[p[13]]);
    g(s, 3, 4, 9, 14, m[p[14]], m[p[15]]);
}

/// Compress the fixed 32-byte tail as the root block, producing 64 XOF bytes.
#[inline(always)]
fn compress_root<W: Words>(cv: &[W; 8], block: &[W; 16]) -> [W; 16] {
    let mut s = [W::splat(0); 16];
    s[..8].copy_from_slice(cv);
    for i in 0..4 {
        s[i + 8] = W::splat(IV[i]);
    }
    s[14] = W::splat(32);
    s[15] = W::splat(2 | 8); // CHUNK_END | ROOT
    round(&mut s, block, 0);
    round(&mut s, block, 1);
    round(&mut s, block, 2);
    round(&mut s, block, 3);
    round(&mut s, block, 4);
    round(&mut s, block, 5);
    round(&mut s, block, 6);
    for i in 0..8 {
        s[i] = s[i].xor(s[i + 8]);
        s[i + 8] = s[i + 8].xor(cv[i]);
    }
    s
}

/// Precompute the first-round G operations that do not read the nonce.
/// `pos < 8` identifies its field slot; `hash_batch` completes the remaining Gs.
fn prepare(input: &[u32; 24], pos: usize) -> [u32; 16] {
    let mut s = [Portable([0u32]); 16];
    for i in 0..8 {
        s[i] = Portable([IV[i]]);
    }
    for i in 0..4 {
        s[8 + i] = Portable([IV[i]]);
    }
    s[14] = Portable([64]);
    s[15] = Portable([1]);
    let words = input.map(|x| Portable([x]));
    for (i, (a, b, c, d)) in COLUMN_G.into_iter().enumerate() {
        if pos != i {
            g(&mut s, a, b, c, d, words[2 * i], words[2 * i + 1]);
        }
    }
    if pos >= 4 {
        for (i, (a, b, c, d)) in DIAGONAL_G.into_iter().enumerate() {
            if pos != i + 4 {
                g(&mut s, a, b, c, d, words[8 + 2 * i], words[9 + 2 * i]);
            }
        }
    }
    s.map(|x| x.0[0])
}

/// Hash consecutive nonces using a prepared first-round state for slot `POS`.
/// Returns all 64 XOF bytes as transposed u32s, before field rejection sampling.
#[inline(always)]
fn hash_batch<W: Words, const POS: usize>(
    input: &[u32; 24],
    prepared: &[u32; 16],
    base: u64,
) -> [W; 16] {
    let mut block = core::array::from_fn(|i| W::splat(input[i]));
    let lo = W::from_fn(|i| (base + i as u64) as u32);
    let hi = if base & (W::LANES as u64 - 1) == 0 {
        W::splat((base >> 32) as u32)
    } else {
        W::from_fn(|i| ((base + i as u64) >> 32) as u32)
    };
    block[2 * POS] = lo;
    block[2 * POS + 1] = hi;
    let mut s = prepared.map(W::splat);
    if POS < 4 {
        let (a, b, c, d) = COLUMN_G[POS];
        g(&mut s, a, b, c, d, lo, hi);
        g(&mut s, 0, 5, 10, 15, block[8], block[9]);
        g(&mut s, 1, 6, 11, 12, block[10], block[11]);
        g(&mut s, 2, 7, 8, 13, block[12], block[13]);
        g(&mut s, 3, 4, 9, 14, block[14], block[15]);
    } else {
        let (a, b, c, d) = DIAGONAL_G[POS - 4];
        g(&mut s, a, b, c, d, lo, hi);
    }
    round(&mut s, &block, 1);
    round(&mut s, &block, 2);
    round(&mut s, &block, 3);
    round(&mut s, &block, 4);
    round(&mut s, &block, 5);
    round(&mut s, &block, 6);
    let cv = core::array::from_fn(|i| s[i].xor(s[i + 8]));
    let tail = core::array::from_fn(|i| W::splat(if i < 8 { input[16 + i] } else { 0 }));
    compress_root(&cv, &tail)
}

/// Return (qualifying lanes, lanes requiring rejection-sampling fallback).
/// The field is Goldilocks: a u64 word is noncanonical iff high == MAX and low != 0.
/// A qualifying lane has eight canonical words and response <= `limit`.
#[inline(always)]
fn masks<W: Words>(output: &[W; 16], limit: u64) -> (u64, u64) {
    let zero = W::splat(0);
    let ones = W::splat(u32::MAX);
    let high = W::splat((limit >> 32) as u32);
    let accepted = if limit as u32 == u32::MAX {
        output[15].le(high)
    } else {
        output[15].le(high).and(
            output[15]
                .eq(high)
                .xor(ones)
                .or(output[14].le(W::splat(limit as u32))),
        )
    };
    let mut possible_rejection = zero;
    for i in 0..8 {
        possible_rejection = possible_rejection.or(output[2 * i + 1].eq(ones));
    }
    if possible_rejection.mask() == 0 {
        return (accepted.mask(), 0);
    }
    let mut rejected = zero;
    for i in 0..8 {
        rejected = rejected.or(output[2 * i + 1]
            .eq(ones)
            .and(output[2 * i].eq(zero).xor(ones)));
    }
    (accepted.and(rejected.xor(ones)).mask(), rejected.mask())
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use core::arch::x86_64::*;

    // Every construction/use is inside a runtime-checked AVX2 entry point.
    #[derive(Clone, Copy)]
    struct Avx2(__m256i);
    impl Words for Avx2 {
        const LANES: usize = 8;
        #[inline(always)]
        fn splat(x: u32) -> Self {
            Self(unsafe { _mm256_set1_epi32(x as i32) })
        }
        #[inline(always)]
        fn from_fn(f: impl FnMut(usize) -> u32) -> Self {
            let x: [u32; 8] = core::array::from_fn(f);
            Self(unsafe { _mm256_loadu_si256(x.as_ptr().cast()) })
        }
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self(unsafe { _mm256_add_epi32(self.0, rhs.0) })
        }
        #[inline(always)]
        fn xor(self, rhs: Self) -> Self {
            Self(unsafe { _mm256_xor_si256(self.0, rhs.0) })
        }
        #[inline(always)]
        fn and(self, rhs: Self) -> Self {
            Self(unsafe { _mm256_and_si256(self.0, rhs.0) })
        }
        #[inline(always)]
        fn or(self, rhs: Self) -> Self {
            Self(unsafe { _mm256_or_si256(self.0, rhs.0) })
        }
        #[inline(always)]
        fn rotate<const N: u32>(self) -> Self {
            Self(unsafe {
                if N == 16 {
                    _mm256_shuffle_epi8(
                        self.0,
                        _mm256_setr_epi8(
                            2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7,
                            4, 5, 10, 11, 8, 9, 14, 15, 12, 13,
                        ),
                    )
                } else if N == 8 {
                    _mm256_shuffle_epi8(
                        self.0,
                        _mm256_setr_epi8(
                            1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0, 5, 6,
                            7, 4, 9, 10, 11, 8, 13, 14, 15, 12,
                        ),
                    )
                } else if N == 12 {
                    _mm256_or_si256(
                        _mm256_srli_epi32::<12>(self.0),
                        _mm256_slli_epi32::<20>(self.0),
                    )
                } else {
                    debug_assert_eq!(N, 7);
                    _mm256_or_si256(
                        _mm256_srli_epi32::<7>(self.0),
                        _mm256_slli_epi32::<25>(self.0),
                    )
                }
            })
        }
        #[inline(always)]
        fn eq(self, rhs: Self) -> Self {
            Self(unsafe { _mm256_cmpeq_epi32(self.0, rhs.0) })
        }
        #[inline(always)]
        fn le(self, rhs: Self) -> Self {
            Self(unsafe { _mm256_cmpeq_epi32(self.0, _mm256_min_epu32(self.0, rhs.0)) })
        }
        #[inline(always)]
        fn mask(self) -> u64 {
            unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(self.0)) as u64 }
        }
    }

    #[cfg(test)]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn test_hash(input: &[u32; 24], pos: usize, base: u64) -> [[u32; 8]; 16] {
        hash_at_position::<Avx2>(input, pos, base).map(|w| {
            let mut out = [0u32; 8];
            _mm256_storeu_si256(out.as_mut_ptr().cast(), w.0);
            out
        })
    }
    #[cfg(test)]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn test_masks(input: &[[u32; 8]; 16], limit: u64) -> (u64, u64) {
        masks(&input.map(|w| Avx2::from_fn(|i| w[i])), limit)
    }

    pub(super) fn available() -> bool {
        #[cfg(feature = "std")]
        {
            std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(feature = "std"))]
        {
            cfg!(target_feature = "avx2")
        }
    }

    /// Caller must check AVX2 support. All vector loads read fixed eight-u32 arrays.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn batch<const POS: usize>(
        input: &[u32; 24],
        prepared: &[u32; 16],
        base: u64,
        limit: u64,
    ) -> (u64, u64) {
        masks(&hash_batch::<Avx2, POS>(input, prepared, base), limit)
    }
}

/// Search a Blake3 transcript, returning None if the nonce search is exhausted.
/// `state` has absorbed buffered inputs but not the nonce; `pos` is its input slot.
/// The caller must select `Blake3GoldilocksConfig`; it also absorbs and checks
/// the returned nonce through its existing Challenger path.
pub(super) fn find_witness<F: RichField, P: PlonkyPermutation<F>>(
    state: P,
    pos: usize,
    bits: u32,
) -> Option<u64> {
    let mut input = [0u32; 24];
    for (words, field) in input.as_chunks_mut::<2>().0.iter_mut().zip(state.as_ref()) {
        let x = field.to_canonical_u64();
        words[0] = x as u32;
        words[1] = (x >> 32) as u32;
    }
    match pos {
        0 => find_witness_at::<F, _, 0>(state, &input, bits),
        1 => find_witness_at::<F, _, 1>(state, &input, bits),
        2 => find_witness_at::<F, _, 2>(state, &input, bits),
        3 => find_witness_at::<F, _, 3>(state, &input, bits),
        4 => find_witness_at::<F, _, 4>(state, &input, bits),
        5 => find_witness_at::<F, _, 5>(state, &input, bits),
        6 => find_witness_at::<F, _, 6>(state, &input, bits),
        7 => find_witness_at::<F, _, 7>(state, &input, bits),
        _ => unreachable!("nonce slot is outside the challenger rate"),
    }
}

/// Grind with a fixed nonce slot, reusing the nonce-independent first-round work.
fn find_witness_at<F: RichField, P: PlonkyPermutation<F>, const POS: usize>(
    state: P,
    input: &[u32; 24],
    bits: u32,
) -> Option<u64> {
    let prepared = prepare(input, POS);
    let limit = u64::MAX >> bits;
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = avx2::available();
    #[cfg(target_arch = "x86_64")]
    let lanes = if use_avx2 { 8 } else { Portable::<16>::LANES };
    #[cfg(not(target_arch = "x86_64"))]
    let lanes = Portable::<16>::LANES;
    // Complete SIMD batches are needed even at low difficulty.
    let chunk_size = (1u64 << bits.saturating_sub(16)).max(lanes as u64);
    let chunks = F::NEG_ONE.to_canonical_u64() / chunk_size;
    let search_chunk = |j| {
        let base = j * chunk_size;
        for offset in (0..chunk_size).step_by(lanes) {
            let first = base + offset;
            #[cfg(target_arch = "x86_64")]
            let (mut accepted, rejected) = if use_avx2 {
                // SAFETY: availability was checked once before starting parallel work.
                unsafe { avx2::batch::<POS>(input, &prepared, first, limit) }
            } else {
                masks(
                    &hash_batch::<Portable<16>, POS>(input, &prepared, first),
                    limit,
                )
            };
            #[cfg(not(target_arch = "x86_64"))]
            let (mut accepted, rejected) = masks(
                &hash_batch::<Portable<16>, POS>(input, &prepared, first),
                limit,
            );
            accepted = resolve_rejections::<F, _>(state, POS, first, limit, (accepted, rejected));
            if accepted != 0 {
                return Some(first + accepted.trailing_zeros() as u64);
            }
        }
        None
    };
    #[cfg(feature = "parallel")]
    let witness = (0..chunks).into_par_iter().find_map_any(search_chunk);
    #[cfg(not(feature = "parallel"))]
    let witness = (0..chunks).find_map(search_chunk);
    witness
}

/// Replay only lanes whose XOF words need field rejection sampling.
/// `masks` contains disjoint accepted/rejected bitsets for consecutive nonces.
fn resolve_rejections<F: RichField, P: PlonkyPermutation<F>>(
    state: P,
    pos: usize,
    base: u64,
    limit: u64,
    masks: (u64, u64),
) -> u64 {
    let (mut accepted, mut rejected) = masks;
    while rejected != 0 {
        let i = rejected.trailing_zeros();
        rejected &= rejected - 1;
        let mut fallback = state;
        fallback.set_elt(F::from_canonical_u64(base + i as u64), pos);
        fallback.permute_n::<8>();
        if fallback.squeeze()[7].to_canonical_u64() <= limit {
            accepted |= 1 << i;
        }
    }
    accepted
}

#[cfg(test)]
#[inline(always)]
fn hash_at_position<W: Words>(input: &[u32; 24], pos: usize, base: u64) -> [W; 16] {
    let prepared = prepare(input, pos);
    match pos {
        0 => hash_batch::<W, 0>(input, &prepared, base),
        1 => hash_batch::<W, 1>(input, &prepared, base),
        2 => hash_batch::<W, 2>(input, &prepared, base),
        3 => hash_batch::<W, 3>(input, &prepared, base),
        4 => hash_batch::<W, 4>(input, &prepared, base),
        5 => hash_batch::<W, 5>(input, &prepared, base),
        6 => hash_batch::<W, 6>(input, &prepared, base),
        7 => hash_batch::<W, 7>(input, &prepared, base),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField as F;
    use crate::field::types::{Field, Field64, PrimeField64};
    use crate::hash::blake3_perm::Blake3Permutation;
    use crate::plonk::config::{Blake3GoldilocksConfig, GenericConfig, PoseidonGoldilocksConfig};

    fn fields(pattern: usize) -> [F; 12] {
        let mut x = 829175889023u64 ^ (pattern as u64).wrapping_mul(0x9e3779b97f4a7c15);
        core::array::from_fn(|i| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            match pattern {
                0 => F::ZERO,
                1 => F(if i % 2 == 0 { F::ORDER } else { u64::MAX }),
                _ => F::from_noncanonical_u64(x),
            }
        })
    }

    #[test]
    fn full_xof_matches_existing_blake3() {
        for pattern in 0..18 {
            let fields = fields(pattern);
            let mut bytes = [0u8; 96];
            for (part, f) in bytes.as_chunks_mut::<8>().0.iter_mut().zip(fields) {
                part.copy_from_slice(&f.to_canonical_u64().to_le_bytes());
            }
            let input = core::array::from_fn(|i| {
                u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
            });
            for pos in 0..8 {
                for base in [0, 1, u32::MAX as u64 - 3, 1 << 32, F::ORDER - 4096] {
                    let output = hash_at_position::<Portable<16>>(&input, pos, base);
                    for i in 0..16 {
                        let mut message = bytes;
                        message[pos * 8..pos * 8 + 8]
                            .copy_from_slice(&(base + i as u64).to_le_bytes());
                        let mut expected = [0u8; 64];
                        blake3::Hasher::new()
                            .update(&message)
                            .finalize_xof()
                            .fill(&mut expected);
                        let got: [u8; 64] =
                            core::array::from_fn(|j| output[j / 4].0[i].to_le_bytes()[j % 4]);
                        assert_eq!(got, expected);
                    }
                    #[cfg(target_arch = "x86_64")]
                    if avx2::available() {
                        // SAFETY: runtime AVX2 support was checked.
                        let raw = unsafe { avx2::test_hash(&input, pos, base) };
                        for j in 0..16 {
                            assert_eq!(raw[j], output[j].0[..8]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn threshold_and_field_boundaries() {
        for bits in [0, 1, 17, 20, 23, 25, 31, 32, 33, 40, 63] {
            let limit = u64::MAX >> bits;
            let mut raw = [[0u64; 8]; 16];
            for (i, lane) in raw.iter_mut().enumerate() {
                lane[7] = [0, limit, limit.saturating_add(1), u64::MAX][i % 4];
            }
            raw[4][0] = F::ORDER;
            raw[5][3] = F::ORDER - 1;
            raw[6][6] = F::ORDER + 1;
            raw[7][1] = u64::MAX;
            let output: [Portable<16>; 16] = core::array::from_fn(|j| {
                Portable::from_fn(|i| (raw[i][j / 2] >> (32 * (j % 2))) as u32)
            });
            let (mut accepted, mut rejected) = (0, 0);
            for (i, lane) in raw.iter().enumerate() {
                if lane.iter().any(|&x| x >= F::ORDER) {
                    rejected |= 1 << i;
                } else if lane[7] <= limit {
                    accepted |= 1 << i;
                }
            }
            assert_eq!(masks(&output, limit), (accepted, rejected));
            #[cfg(target_arch = "x86_64")]
            if avx2::available() {
                let input = core::array::from_fn(|j| output[j].0[..8].try_into().unwrap());
                // SAFETY: runtime AVX2 support was checked.
                assert_eq!(
                    unsafe { avx2::test_masks(&input, limit) },
                    (accepted & 255, rejected & 255)
                );
            }
        }
    }

    #[test]
    fn forced_rejections_replay_original_permutation() {
        for pos in 0..8 {
            let state = Blake3Permutation::new(fields(1));
            for bits in [0, 1, 23, 63] {
                let limit = u64::MAX >> bits;
                let mut expected = 0;
                for i in 0..16 {
                    let mut s = state;
                    s.set_elt(F::from_canonical_u64(i), pos);
                    s.permute_n::<8>();
                    if s.squeeze()[7].to_canonical_u64() <= limit {
                        expected |= 1 << i;
                    }
                }
                for rejected in [1, 0x5555, 0xaaaa, 0xffff] {
                    assert_eq!(
                        resolve_rejections::<F, _>(
                            state,
                            pos,
                            0,
                            limit,
                            (expected & !rejected, rejected)
                        ),
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn witnesses_verify_at_all_batch_sizes() {
        let state = Blake3Permutation::new(fields(2));
        fn check_caller<C: GenericConfig<2, F = F>>(bits: u32) {
            let mut challenger = crate::iop::challenger::Challenger::new();
            let mut replay = challenger.clone();
            let witness = crate::fri::prover::proof_of_work::<F, C, 2>(&mut challenger, bits);
            replay.observe_element(witness);
            assert!(replay.get_challenge().to_canonical_u64().leading_zeros() >= bits);
        }
        for bits in [0, 1, 4, 19] {
            check_caller::<Blake3GoldilocksConfig>(bits);
            check_caller::<PoseidonGoldilocksConfig>(bits);
        }
        for threads in [1, 4] {
            let pool = plonky2_maybe_rayon::rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for bits in [0, 1, 16, 17, 18, 19, 20] {
                    for pos in 0..8 {
                        let witness = find_witness::<F, _>(state, pos, bits).unwrap();
                        let mut replay = state;
                        replay.set_elt(F::from_canonical_u64(witness), pos);
                        replay.permute_n::<8>();
                        assert!(replay.squeeze()[7].to_canonical_u64().leading_zeros() >= bits);
                    }
                }
            });
        }
    }
}
