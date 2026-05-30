//! C1 NTT oracle: emit (coeffs -> fft(coeffs)) plus the plonky2 root table, so
//! the GPU NTT can use plonky2's exact twiddles and match bit-for-bit. Also
//! prints CPU fft time to stderr as a throughput baseline.
//!
//! Run: cargo run --release --bin fft_fixtures -- <lg_n> > /tmp/fft_fixtures.txt
//! stdout:
//!   n lg_n
//!   <n coeff hex>            (input, 1 line)
//!   <n value hex>            (fft output, 1 line)
//!   <root_table[i] hex...>   (lg_n lines; line i = stage-i twiddles)

use std::time::Instant;

use plonky2::field::fft::{fft, fft_root_table};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::polynomial::PolynomialCoeffs;
use plonky2::field::types::{Field, Field64, PrimeField64};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

type F = GoldilocksField;

fn line(xs: &[F]) -> String {
    xs.iter()
        .map(|x| format!("{:016x}", x.to_canonical_u64()))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let lg_n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(16);
    let n = 1usize << lg_n;

    let mut rng = StdRng::seed_from_u64(0xC1_FF_7000);
    let coeffs: Vec<F> = (0..n).map(|_| F::from_canonical_u64(rng.next_u64() % F::ORDER)).collect();
    let poly = PolynomialCoeffs { coeffs: coeffs.clone() };

    let t0 = Instant::now();
    let values = fft(poly.clone());
    let dt = t0.elapsed();
    eprintln!("CPU fft: n=2^{}={} -> {:?}", lg_n, n, dt);

    let rt = fft_root_table::<F>(n);

    println!("{} {}", n, lg_n);
    println!("{}", line(&coeffs));
    println!("{}", line(&values.values));
    for row in &rt {
        println!("{}", line(row));
    }
}
