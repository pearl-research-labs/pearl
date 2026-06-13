//! C-step e2e: the STARK#0 trace commitment = `PolynomialBatch::from_values`
//! (ifft -> coset-LDE all columns -> transpose -> bit-reverse leaves -> Merkle).
//! This is the dominant pool-side cost (~45 s of the 61.5 s STARK#0). Times it
//! (the Threadripper/CPU baseline) and, with `emit`, dumps a byte-exact oracle
//! (trace + both root tables + cap) for the GPU `from_values_test`.
//!
//! Run (bench):  cargo run --release --bin commit_bench -- <lg_n> <num_cols> <rate_bits> <cap_height>
//! Run (oracle): cargo run --release --bin commit_bench -- <lg_n> <num_cols> <rate_bits> <cap_height> emit > /tmp/commit_fixture.txt

use std::time::Instant;

use plonky2::field::fft::fft_root_table;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::{Field, Field64, PrimeField64};
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::plonk::config::PoseidonGoldilocksConfig;
use plonky2::util::timing::TimingTree;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

type F = GoldilocksField;
type C = PoseidonGoldilocksConfig;
const D: usize = 2;

fn line(xs: &[F]) -> String {
    xs.iter()
        .map(|x| format!("{:016x}", x.to_canonical_u64()))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let lg_n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(17);
    let num_cols: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(140);
    let rate_bits: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);
    let cap_height: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(5);
    let emit = a.get(5).map(|s| s == "emit").unwrap_or(false);
    let n = 1usize << lg_n;

    let mut rng = StdRng::seed_from_u64(0xC0DE_0017u64);
    let trace: Vec<PolynomialValues<F>> = (0..num_cols)
        .map(|_| PolynomialValues::new((0..n).map(|_| F::from_canonical_u64(rng.next_u64() % F::ORDER)).collect()))
        .collect();

    let rt_lde = fft_root_table::<F>(n << rate_bits);

    // Warm + timed runs (blinding=false for a byte-exact, deterministic commitment).
    let mut cap_hex = String::new();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut best = std::time::Duration::from_secs(9999);
    let reps = if lg_n >= 17 { 3 } else { 5 };
    for _ in 0..reps {
        let t0 = Instant::now();
        let pb = PolynomialBatch::<F, C, D>::from_values(
            trace.clone(),
            rate_bits,
            false,
            cap_height,
            &mut TimingTree::default(),
            Some(&rt_lde),
        );
        let dt = t0.elapsed();
        if dt < best {
            best = dt;
        }
        if emit && cap_hex.is_empty() {
            cap_hex = pb
                .merkle_tree
                .cap
                .0
                .iter()
                .map(|h| line(&h.elements))
                .collect::<Vec<_>>()
                .join("\n");
        }
    }
    eprintln!(
        "CPU from_values: lg_n={} n={} num_cols={} rate_bits={} cap_height={} threads={} -> best {:?}",
        lg_n, n, num_cols, rate_bits, cap_height, threads, best
    );

    if emit {
        let rt_n = fft_root_table::<F>(n);
        let n_inv = F::inverse_2exp(lg_n);
        // All hex (the GPU side parses every header token as base-16).
        println!(
            "{:x} {:x} {:x} {:x} {:016x} {:016x}",
            lg_n,
            num_cols,
            rate_bits,
            cap_height,
            F::coset_shift().to_canonical_u64(),
            n_inv.to_canonical_u64()
        );
        println!("TRACE");
        for col in &trace {
            println!("{}", line(&col.values));
        }
        println!("RT_N"); // ifft root table (size n)
        for row in &rt_n {
            println!("{}", line(row));
        }
        println!("RT_LDE"); // coset-fft root table (size n<<r)
        for row in &rt_lde {
            println!("{}", line(row));
        }
        println!("CAP");
        println!("{}", cap_hex);
        // Full digests vec (plonky2 layout) for FFI byte-exact validation.
        let pb = PolynomialBatch::<F, C, D>::from_values(
            trace.clone(),
            rate_bits,
            false,
            cap_height,
            &mut TimingTree::default(),
            Some(&rt_lde),
        );
        println!("DIGESTS {}", pb.merkle_tree.digests.len());
        for h in &pb.merkle_tree.digests {
            println!("{}", line(&h.elements));
        }
    }
}
