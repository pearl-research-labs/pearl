//! Batched native Blake3 grinding. Verifiers use the existing permutation.
//!
//! Each lane hashes the same 96-byte challenger state with a different nonce.
//! The nonce is in the first 64-byte block. The second block contains 32 bytes;
//! its root compression produces the first 64 XOF bytes. Noncanonical output
//! words fall back to the existing permutation, including further XOF blocks.

use core::any::TypeId;
#[cfg(feature = "parallel")]
use plonky2_maybe_rayon::*;

use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
use crate::hash::hashing::PlonkyPermutation;
use crate::plonk::config::{Blake3GoldilocksConfig, GenericConfig, Hasher};

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

#[inline(always)]
fn compress<W: Words>(cv: &[W; 8], block: &[W; 16], len: u32, flags: u32) -> [W; 16] {
    let mut s = [W::splat(0); 16];
    s[..8].copy_from_slice(cv);
    for i in 0..4 {
        s[i + 8] = W::splat(IV[i]);
    }
    s[14] = W::splat(len);
    s[15] = W::splat(flags);
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

/// Return the full first XOF block for consecutive nonces, in transposed u32s.
/// `input` is the canonical little-endian 96-byte state split into words;
/// `pos < 8` selects the overwritten field element. Each lane is independent.
#[inline(always)]
fn hash_batch<W: Words>(input: &[u32; 24], pos: usize, base: u64) -> [W; 16] {
    let mut block = core::array::from_fn(|i| W::splat(input[i]));
    block[2 * pos] = W::from_fn(|i| (base + i as u64) as u32);
    block[2 * pos + 1] = if base & (W::LANES as u64 - 1) == 0 {
        W::splat((base >> 32) as u32)
    } else {
        W::from_fn(|i| ((base + i as u64) >> 32) as u32)
    };
    let first = compress(&IV.map(W::splat), &block, 64, 1); // CHUNK_START
    let cv = core::array::from_fn(|i| first[i]);
    let tail = core::array::from_fn(|i| W::splat(if i < 8 { input[16 + i] } else { 0 }));
    compress(&cv, &tail, 32, 2 | 8) // CHUNK_END | ROOT, XOF counter zero
}

/// Return (qualifying lanes, lanes requiring rejection-sampling fallback).
/// The field is Goldilocks: a u64 word is noncanonical iff high == MAX and low != 0.
/// A qualifying lane has eight canonical words and response <= `limit`.
#[inline(always)]
fn masks<W: Words>(output: &[W; 16], limit: u64) -> (u64, u64) {
    let zero = W::splat(0);
    let ones = W::splat(u32::MAX);
    let mut rejected = zero;
    for i in 0..8 {
        rejected = rejected.or(output[2 * i + 1]
            .eq(ones)
            .and(output[2 * i].eq(zero).xor(ones)));
    }
    let high = W::splat((limit >> 32) as u32);
    let accepted = output[15].le(high).and(
        output[15]
            .eq(high)
            .xor(ones)
            .or(output[14].le(W::splat(limit as u32))),
    );
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
        hash_batch::<Avx2>(input, pos, base).map(|w| {
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
    pub(super) unsafe fn batch(input: &[u32; 24], pos: usize, base: u64, limit: u64) -> (u64, u64) {
        masks(&hash_batch::<Avx2>(input, pos, base), limit)
    }
}

/// Search a Blake3 transcript, or return None for the unchanged generic grinder.
/// `state` has absorbed buffered inputs but not the nonce; `pos` is its input slot.
/// Only the concrete Blake3 config at 20 or more bits is specialized; cheaper
/// grinds keep the generic path. The final nonce absorption and
/// response check remain in the caller's existing Challenger path.
pub(super) fn find_witness<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    state: <C::Hasher as Hasher<F>>::Permutation,
    pos: usize,
    bits: u32,
) -> Option<u64> {
    if TypeId::of::<C>() != TypeId::of::<Blake3GoldilocksConfig>() || bits < 20 {
        return None;
    }
    let mut input = [0u32; 24];
    for (words, field) in input.as_chunks_mut::<2>().0.iter_mut().zip(state.as_ref()) {
        let x = field.to_canonical_u64();
        words[0] = x as u32;
        words[1] = (x >> 32) as u32;
    }
    let limit = u64::MAX >> bits;
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = avx2::available();
    #[cfg(target_arch = "x86_64")]
    let lanes = if use_avx2 { 8 } else { Portable::<16>::LANES };
    #[cfg(not(target_arch = "x86_64"))]
    let lanes = Portable::<16>::LANES;
    // At bits >= 20 every chunk is a multiple of both supported batch widths.
    let chunk_size = 1u64 << bits.saturating_sub(16);
    let chunks = F::NEG_ONE.to_canonical_u64() / chunk_size;
    let search_chunk = |j| {
        let base = j * chunk_size;
        for offset in (0..chunk_size).step_by(lanes) {
            let first = base + offset;
            #[cfg(target_arch = "x86_64")]
            let (mut accepted, rejected) = if use_avx2 {
                // SAFETY: availability was checked once before starting parallel work.
                unsafe { avx2::batch(&input, pos, first, limit) }
            } else {
                masks(&hash_batch::<Portable<16>>(&input, pos, first), limit)
            };
            #[cfg(not(target_arch = "x86_64"))]
            let (mut accepted, rejected) =
                masks(&hash_batch::<Portable<16>>(&input, pos, first), limit);
            accepted = resolve_rejections::<F, _>(state, pos, first, limit, (accepted, rejected));
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
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField as F;
    use crate::field::types::{Field, Field64, PrimeField64};
    use crate::hash::blake3_perm::Blake3Permutation;
    use crate::plonk::config::PoseidonGoldilocksConfig;

    fn fields(pattern: usize) -> [F; 12] {
        let mut x = 829175889023u64;
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
        for pattern in 0..3 {
            let fields = fields(pattern);
            let mut bytes = [0u8; 96];
            for (part, f) in bytes.as_chunks_mut::<8>().0.iter_mut().zip(fields) {
                part.copy_from_slice(&f.to_canonical_u64().to_le_bytes());
            }
            let input = core::array::from_fn(|i| {
                u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
            });
            for pos in 0..8 {
                for base in [0, u32::MAX as u64 - 3, F::ORDER - 4096] {
                    let output = hash_batch::<Portable<16>>(&input, pos, base);
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
    fn witnesses_verify_and_other_configs_keep_generic_path() {
        let poseidon = <<PoseidonGoldilocksConfig as GenericConfig<2>>::Hasher as Hasher<F>>::Permutation::default();
        assert!(find_witness::<F, PoseidonGoldilocksConfig, 2>(poseidon, 0, 23).is_none());
        let state = Blake3Permutation::new(fields(2));
        for bits in [0, 19] {
            assert!(find_witness::<F, Blake3GoldilocksConfig, 2>(state, 7, bits).is_none());
        }
        for threads in [1, 4] {
            let pool = plonky2_maybe_rayon::rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for pos in [0, 7] {
                    let witness =
                        find_witness::<F, Blake3GoldilocksConfig, 2>(state, pos, 20).unwrap();
                    let mut replay = state;
                    replay.set_elt(F::from_canonical_u64(witness), pos);
                    replay.permute_n::<8>();
                    assert!(replay.squeeze()[7].to_canonical_u64().leading_zeros() >= 20);
                }
            });
        }
    }
}
