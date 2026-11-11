//! Prover-only BLAKE3 nonce batching. Ordinary hashing and verification are unchanged.

#[cfg(feature = "parallel")]
use plonky2_maybe_rayon::*;

use crate::hash::hash_types::RichField;
use crate::hash::hashing::PlonkyPermutation;

// Independent candidates in each u32 lane let LLVM vectorize the compression.
const LANES: usize = 16;
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
type Word = [u32; LANES];

macro_rules! mix {
    ($a:ident, $b:ident, $c:ident, $d:ident, $x:expr, $y:expr) => {
        for lane in 0..LANES {
            $a[lane] = $a[lane].wrapping_add($b[lane]).wrapping_add($x[lane]);
            $d[lane] = ($d[lane] ^ $a[lane]).rotate_right(16);
            $c[lane] = $c[lane].wrapping_add($d[lane]);
            $b[lane] = ($b[lane] ^ $c[lane]).rotate_right(12);
            $a[lane] = $a[lane].wrapping_add($b[lane]).wrapping_add($y[lane]);
            $d[lane] = ($d[lane] ^ $a[lane]).rotate_right(8);
            $c[lane] = $c[lane].wrapping_add($d[lane]);
            $b[lane] = ($b[lane] ^ $c[lane]).rotate_right(7);
        }
    };
}

/// Compress one block in every lane, with zero block counters.
#[inline(always)]
#[unroll::unroll_for_loops]
fn compress(cv: &[Word; 8], block: &[Word; 16], len: u32, flags: u32) -> [Word; 16] {
    let [mut v0, mut v1, mut v2, mut v3, mut v4, mut v5, mut v6, mut v7] = *cv;
    let [mut v8, mut v9, mut v10, mut v11] = core::array::from_fn(|i| [IV[i]; LANES]);
    let (mut v12, mut v13) = ([0; LANES], [0; LANES]);
    let (mut v14, mut v15) = ([len; LANES], [flags; LANES]);
    for round in 0..7 {
        let p = SCHEDULE[round];
        mix!(v0, v4, v8, v12, block[p[0]], block[p[1]]);
        mix!(v1, v5, v9, v13, block[p[2]], block[p[3]]);
        mix!(v2, v6, v10, v14, block[p[4]], block[p[5]]);
        mix!(v3, v7, v11, v15, block[p[6]], block[p[7]]);
        mix!(v0, v5, v10, v15, block[p[8]], block[p[9]]);
        mix!(v1, v6, v11, v12, block[p[10]], block[p[11]]);
        mix!(v2, v7, v8, v13, block[p[12]], block[p[13]]);
        mix!(v3, v4, v9, v14, block[p[14]], block[p[15]]);
    }
    let mut s = [
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
    ];
    for i in 0..8 {
        for lane in 0..LANES {
            s[i][lane] ^= s[i + 8][lane];
            s[i + 8][lane] ^= cv[i][lane];
        }
    }
    s
}

/// First 64 XOF bytes for consecutive nonces in a 96-byte challenger state.
/// `pos < 8` is the nonce's field slot; output words are transposed by lane.
#[inline]
fn hash_batch(input: &[u32; 24], pos: usize, base: u64) -> [Word; 16] {
    let mut block = core::array::from_fn(|i| [input[i]; LANES]);
    block[2 * pos] = core::array::from_fn(|lane| (base + lane as u64) as u32);
    block[2 * pos + 1] = core::array::from_fn(|lane| ((base + lane as u64) >> 32) as u32);
    let first = compress(&IV.map(|x| [x; LANES]), &block, 64, 1); // CHUNK_START
    let cv = core::array::from_fn(|i| first[i]);
    let tail = core::array::from_fn(|i| [if i < 8 { input[16 + i] } else { 0 }; LANES]);
    compress(&cv, &tail, 32, 2 | 8) // CHUNK_END | ROOT
}

/// PoW decisions for the consecutive nonces `first..first + LANES`.
/// Require `bits` leading zeroes; replay original XOF sampling for exceptional lanes.
/// Goldilocks words with high halves below u32::MAX are always canonical.
#[inline]
fn qualifies<F: RichField, P: PlonkyPermutation<F>>(
    output: &[Word; 16],
    state: P,
    pos: usize,
    first: u64,
    bits: u32,
) -> [bool; LANES] {
    let mut fallback = [false; LANES];
    for i in 0..8 {
        for lane in 0..LANES {
            fallback[lane] |= output[2 * i + 1][lane] == u32::MAX;
        }
    }
    core::array::from_fn(|lane| {
        if fallback[lane] {
            let mut replay = state;
            replay.set_elt(F::from_canonical_u64(first + lane as u64), pos);
            replay.permute_n::<8>();
            replay.squeeze()[7].to_canonical_u64().leading_zeros() >= bits
        } else if bits <= 32 {
            output[15][lane].leading_zeros() >= bits
        } else {
            output[15][lane] == 0 && output[14][lane].leading_zeros() >= bits - 32
        }
    })
}

/// Search a BLAKE3/Goldilocks transcript, returning a canonical nonce or None.
/// `state` has absorbed buffered inputs, but not the nonce at `pos < 8`.
/// The caller selects Blake3GoldilocksConfig and rechecks the result normally.
pub(super) fn find_witness<F: RichField, P: PlonkyPermutation<F>>(
    state: P,
    pos: usize,
    bits: u32,
) -> Option<u64> {
    let input = core::array::from_fn(|i| {
        (state.as_ref()[i / 2].to_canonical_u64() >> (32 * (i % 2))) as u32
    });
    // Whole batches are required even when the difficulty is small.
    let chunk_size = (1u64 << bits.saturating_sub(16)).max(LANES as u64);
    let chunks = F::NEG_ONE.to_canonical_u64() / chunk_size;
    let search = |j| {
        let base = j * chunk_size;
        for offset in (0..chunk_size).step_by(LANES) {
            let first = base + offset;
            let output = hash_batch(&input, pos, first);
            for (lane, valid) in qualifies::<F, _>(&output, state, pos, first, bits)
                .into_iter()
                .enumerate()
            {
                if valid {
                    return Some(first + lane as u64);
                }
            }
        }
        None
    };
    #[cfg(feature = "parallel")]
    return (0..chunks).into_par_iter().find_map_any(search);
    #[cfg(not(feature = "parallel"))]
    (0..chunks).find_map(search)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField as F;
    use crate::field::types::{Field, Field64, PrimeField64};
    use crate::hash::blake3_perm::Blake3Permutation;

    fn fields() -> [F; 12] {
        let values = [0, 1, F::ORDER, u64::MAX, 0x123456789abcdef0];
        core::array::from_fn(|i| F(values[i % values.len()]))
    }

    #[test]
    fn batched_xof_matches_blake3() {
        let state = Blake3Permutation::new(fields());
        let bytes: [u8; 96] =
            core::array::from_fn(|i| state.as_ref()[i / 8].to_canonical_u64().to_le_bytes()[i % 8]);
        let input = core::array::from_fn(|i| {
            u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
        });
        for pos in 0..8 {
            for base in [0, 1, (1 << 32) - 3, 1 << 32, F::ORDER - 32] {
                let output = hash_batch(&input, pos, base);
                for lane in 0..LANES {
                    let mut message = bytes;
                    message[8 * pos..8 * pos + 8]
                        .copy_from_slice(&(base + lane as u64).to_le_bytes());
                    let mut expected = [0; 64];
                    blake3::Hasher::new()
                        .update(&message)
                        .finalize_xof()
                        .fill(&mut expected);
                    let actual: [u8; 64] =
                        core::array::from_fn(|i| output[i / 4][lane].to_le_bytes()[i % 4]);
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    #[test]
    fn qualification_handles_canonical_and_exceptional_words() {
        let state = Blake3Permutation::new(fields());
        let p = F::ORDER;
        let values = [
            0,
            1,
            (1 << 32) - 1,
            1 << 32,
            1 << 63,
            p - 1,
            p,
            p + 1,
            u64::MAX,
        ];
        let raw: [u64; LANES] = core::array::from_fn(|lane| values[lane % values.len()]);
        for pos in 0..8 {
            let first = [0, (1 << 32) - 3, p - 32][pos % 3];
            for word in 0..8 {
                let mut output = [[0; LANES]; 16];
                output[2 * word] = raw.map(|v| v as u32);
                output[2 * word + 1] = raw.map(|v| (v >> 32) as u32);
                let responses: [u64; LANES] = core::array::from_fn(|lane| {
                    if raw[lane] >= p - 1 {
                        // p-1 conservatively replays too.
                        let mut replay = state;
                        replay.set_elt(F::from_canonical_u64(first + lane as u64), pos);
                        replay.permute_n::<8>();
                        replay.squeeze()[7].to_canonical_u64()
                    } else if word == 7 {
                        raw[lane]
                    } else {
                        0
                    }
                });
                for bits in [0, 1, 23, 31, 32, 33, 48, 63] {
                    assert_eq!(
                        qualifies::<F, _>(&output, state, pos, first, bits),
                        responses.map(|r| r.leading_zeros() >= bits)
                    );
                }
            }
        }
    }

    #[test]
    fn witnesses_replay_at_low_and_high_difficulties() {
        use crate::iop::challenger::Challenger;
        use crate::plonk::config::{
            Blake3GoldilocksConfig, GenericConfig, PoseidonGoldilocksConfig,
        };
        fn check<C: GenericConfig<2, F = F>>(pos: usize, bits: u32) {
            let mut challenger = Challenger::new();
            challenger.observe_elements(&fields()[..pos]);
            let mut replay = challenger.clone();
            let nonce = crate::fri::prover::proof_of_work::<F, C, 2>(&mut challenger, bits);
            replay.observe_element(nonce);
            assert!(replay.get_challenge().to_canonical_u64().leading_zeros() >= bits);
        }
        for pos in 0..8 {
            check::<PoseidonGoldilocksConfig>(pos, 4);
            for bits in [0, 1, 4, 16, 17, 18, 19, 20] {
                check::<Blake3GoldilocksConfig>(pos, bits);
            }
        }
    }
}
