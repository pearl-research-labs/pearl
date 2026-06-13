//! C1 Merkle oracle: build a plonky2 `MerkleTree<GoldilocksField, PoseidonHash>`
//! over random leaves and emit leaves + cap for the GPU kernel to match
//! bit-for-bit. Also prints the CPU build time (rayon, all cores) to stderr as a
//! throughput baseline.
//!
//! Run:  cargo run --release --bin merkle_fixtures -- <log2_n> <leaf_width> <cap_height> > /tmp/merkle_fixtures.txt
//! stdout format:
//!   n k cap_height
//!   <leaf 0: k hex>            (n lines)
//!   ...
//!   CAP
//!   <cap entry: 4 hex>         (2^cap_height lines, in subtree order)

use std::time::Instant;

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, Field64, PrimeField64};
use plonky2::hash::merkle_tree::MerkleTree;
use plonky2::hash::poseidon::PoseidonHash;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

type F = GoldilocksField;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let log2_n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(16);
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(135);
    let cap_height: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let n = 1usize << log2_n;

    let mut rng = StdRng::seed_from_u64(0xC1_4E_47_1E);
    let leaves: Vec<Vec<F>> = (0..n)
        .map(|_| (0..k).map(|_| F::from_canonical_u64(rng.next_u64() % F::ORDER)).collect())
        .collect();

    let t0 = Instant::now();
    let tree = MerkleTree::<F, PoseidonHash>::new(leaves.clone(), cap_height);
    let build = t0.elapsed();
    eprintln!(
        "CPU MerkleTree::new: n=2^{}={} leaf_width={} cap_height={} -> {:?} (rayon all-cores)",
        log2_n, n, k, cap_height, build
    );

    println!("{} {} {}", n, k, cap_height);
    for leaf in &leaves {
        let s: Vec<String> = leaf.iter().map(|x| format!("{:016x}", x.to_canonical_u64())).collect();
        println!("{}", s.join(" "));
    }
    println!("CAP");
    for h in &tree.cap.0 {
        let s: Vec<String> = h.elements.iter().map(|x| format!("{:016x}", x.to_canonical_u64())).collect();
        println!("{}", s.join(" "));
    }
}
