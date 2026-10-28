//! Parallel FFT for large transforms, via the classic four-step
//! (Cooley-Tukey) decomposition.
//!
//! A serial radix-2 FFT over $N = 2^{23}$ extension-field elements takes
//! hundreds of milliseconds and no outer loop can subdivide it, so the
//! remaining serial FFTs inside otherwise-parallel phases dominate their
//! wall time. The four-step split factors the transform into
//! $N = N_1 \cdot N_2$ independent sub-FFTs of lengths $N_1$ and $N_2$
//! which run through the shared rayon pool.
//!
//! With input indexed $n = N_2 n_1 + n_2$ and output $k = k_1 + N_1 k_2$:
//!
//! $$X[k_1 + N_1 k_2] = \sum_{n_2} \omega_N^{n_2 k_1}\, \omega_{N_2}^{n_2 k_2}
//!   \left[\sum_{n_1} x[N_2 n_1 + n_2]\, \omega_{N_1}^{n_1 k_1}\right]$$
//!
//! i.e. (1) one length-$N_1$ FFT per stride-$N_2$ residue class $n_2$,
//! (2) a twiddle pass, (3) one length-$N_2$ FFT per output column $k_1$,
//! (4) a final transpose into natural output order.

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use plonky2_field::fft::{fft_root_table, fft_with_options};
use plonky2_field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;
use plonky2_util::log2_strict;

/// Transforms below this size run faster serially; the four-step's extra
/// passes and task overheads only pay off on the big buffers.
const MIN_LG: usize = 18;

/// Parallel FFT of `coeffs` (natural-order in and out), returned as values.
///
/// Below the size threshold this is exactly `fft`; above it the transform
/// decomposes into $\sqrt{N}$-sized sub-FFTs that share the rayon pool.
pub fn fft_par<F: Field + Send + Sync>(coeffs: PolynomialCoeffs<F>) -> PolynomialValues<F> {
    let PolynomialCoeffs { coeffs } = coeffs;
    let n = coeffs.len();
    if n < (1 << MIN_LG) {
        return fft_with_options(PolynomialCoeffs::new(coeffs), None, None);
    }

    let lg_n = log2_strict(n);
    // Balanced factorization: both sub-FFT lengths sit near sqrt(N).
    let lg_n1 = lg_n.div_ceil(2);
    let lg_n2 = lg_n - lg_n1;
    let n1 = 1usize << lg_n1;
    let n2 = 1usize << lg_n2;

    let omega = F::primitive_root_of_unity(lg_n);

    // Sub-FFT root tables, built once and shared across all row tasks.
    let table1 = fft_root_table::<F>(n1);
    let table2 = fft_root_table::<F>(n2);

    // Step 1: A[n2][k1] = DFT_{N1} over n1 of the stride-N2 residue class n2.
    // Gather each class first so the sub-FFTs run on contiguous rows.
    let a: Vec<Vec<F>> = (0..n2)
        .into_par_iter()
        .map(|n2_idx| {
            let class: Vec<F> = (0..n1).map(|n1_idx| coeffs[n2_idx + n1_idx * n2]).collect();
            fft_with_options(PolynomialCoeffs::new(class), None, Some(&table1)).values
        })
        .collect();

    // Steps 2+3: B[k1][k2] = DFT_{N2} over n2 of
    //   A[n2][k1] * omega_N^{n2 k1},
    // the twiddle folded into the column gather.
    let b: Vec<Vec<F>> = (0..n1)
        .into_par_iter()
        .map(|k1| {
            let step = omega.exp_u64(k1 as u64); // omega_N^{k1}
            let mut w = F::ONE;
            let column: Vec<F> = (0..n2)
                .map(|n2_idx| {
                    let v = a[n2_idx][k1] * w;
                    w *= step;
                    v
                })
                .collect();
            fft_with_options(PolynomialCoeffs::new(column), None, Some(&table2)).values
        })
        .collect();

    // Step 4: transpose B into natural order, X[k1 + N1*k2] = B[k1][k2].
    // Consecutive outputs run over k1 first, so each chunk of N1 holds one k2.
    let mut out = vec![F::ZERO; n];
    out.par_chunks_mut(n1)
        .enumerate()
        .for_each(|(k2, chunk)| {
            for (k1, dst) in chunk.iter_mut().enumerate() {
                *dst = b[k1][k2];
            }
        });
    PolynomialValues::new(out)
}

/// `fft_par` followed by the coset pointwise scaling: the evaluation of
/// `coeffs` on the coset `shift * H`, in canonical coset order.
pub fn coset_fft_par<F: Field + Send + Sync>(
    coeffs: &PolynomialCoeffs<F>,
    shift: F,
) -> PolynomialValues<F> {
    // Scale coeffs[i] *= shift^i in parallel, then transform.
    let mut scaled = vec![F::ZERO; coeffs.len()];
    scaled
        .par_chunks_mut(1 << 12)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let mut w = shift.exp_u64((ci << 12) as u64);
            for (dst, &src) in chunk.iter_mut().zip(&coeffs.coeffs[ci << 12..]) {
                *dst = src * w;
                w *= shift;
            }
        });
    fft_par(PolynomialCoeffs::new(scaled))
}

#[cfg(test)]
mod tests {
    use plonky2_field::fft::fft;
    use plonky2_field::goldilocks_field::GoldilocksField;
    use plonky2_field::polynomial::PolynomialCoeffs;
    use plonky2_field::types::Field;

    use super::*;

    type F = GoldilocksField;

    #[test]
    fn matches_serial_below_threshold() {
        let n = 1 << 10;
        let coeffs: Vec<F> = (0..n).map(|i| F::from_canonical_usize(i * 977 % 101)).collect();
        let poly = PolynomialCoeffs::new(coeffs.clone());
        assert_eq!(fft_par(poly.clone()).values, fft(poly).values);
    }

    #[test]
    fn matches_serial_above_threshold() {
        // Cross the parallel threshold and exercise the unbalanced split
        // (lg 19 = 10 + 9).
        let n = 1 << 19;
        let coeffs: Vec<F> = (0..n).map(|i| F::from_canonical_u64((i as u64) * 0x9E37_79B9_7F4A_7C15 % (1 << 32))).collect();
        let poly = PolynomialCoeffs::new(coeffs);
        let par = fft_par(poly.clone()).values;
        let serial = fft(poly).values;
        assert_eq!(par, serial);
    }

    #[test]
    fn coset_matches_serial() {
        let n = 1 << 19;
        let coeffs: Vec<F> = (0..n).map(|i| F::from_canonical_u64((i as u64) * 0xBF58_476D_1CE4_E5B9 % (1 << 32))).collect();
        let poly = PolynomialCoeffs::new(coeffs);
        let shift = F::coset_shift();
        let par = coset_fft_par(&poly, shift).values;
        let serial = poly.coset_fft(shift).values;
        assert_eq!(par, serial);
    }
}
