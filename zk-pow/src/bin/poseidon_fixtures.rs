//! C1 bit-exactness oracle: emit (input[12] -> poseidon(input)[12]) fixtures for
//! the GPU Goldilocks-Poseidon kernel to validate against. Canonical u64, hex.
//!
//! The GPU kernel implements the *naive* permutation (`poseidon_naive`), which
//! plonky2 ships as the verified bit-identical oracle of the production fast
//! permutation. We emit the production `poseidon()` output here; if the kernel
//! matches it, the naive==fast equivalence is exercised end-to-end too.
//!
//! Run:  cargo run --release --bin poseidon_fixtures -- 512 > /tmp/poseidon_fixtures.txt
//! Format (one fixture per line): 12 input limbs, 12 output limbs, space-separated hex.

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, Field64, PrimeField64};
use plonky2::hash::poseidon::{Poseidon, SPONGE_WIDTH};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

fn perm(input: [GoldilocksField; SPONGE_WIDTH]) -> [u64; SPONGE_WIDTH] {
    let out = <GoldilocksField as Poseidon>::poseidon(input);
    std::array::from_fn(|i| out[i].to_canonical_u64())
}

fn emit(input: [u64; SPONGE_WIDTH]) {
    let f: [GoldilocksField; SPONGE_WIDTH] =
        std::array::from_fn(|i| GoldilocksField::from_canonical_u64(input[i] % GoldilocksField::ORDER));
    let canon_in: [u64; SPONGE_WIDTH] =
        std::array::from_fn(|i| input[i] % GoldilocksField::ORDER);
    let out = perm(f);
    let mut parts: Vec<String> = Vec::with_capacity(24);
    for v in canon_in.iter().chain(out.iter()) {
        parts.push(format!("{:016x}", v));
    }
    println!("{}", parts.join(" "));
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    // Edge cases first.
    emit([0u64; SPONGE_WIDTH]);
    emit([GoldilocksField::ORDER - 1; SPONGE_WIDTH]);
    emit(std::array::from_fn(|i| i as u64));
    emit(std::array::from_fn(|i| GoldilocksField::ORDER - 1 - i as u64));
    // EPSILON / order-boundary stressors (noncanonical-looking inputs get reduced).
    emit([0xFFFF_FFFFu64; SPONGE_WIDTH]); // EPSILON
    emit([0xFFFF_FFFF_0000_0000u64; SPONGE_WIDTH]);

    // Random fixtures (deterministic seed for reproducibility).
    let mut rng = StdRng::seed_from_u64(0xC1_0DE_5EED);
    for _ in 0..n {
        let input: [u64; SPONGE_WIDTH] = std::array::from_fn(|_| rng.next_u64());
        emit(input);
    }
}
