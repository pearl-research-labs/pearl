//! Explicit prover commitment constructors; setup keeps the original constructors.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::BatchFriOracle;
use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::hash::hash_types::RichField;
use crate::plonk::config::GenericConfig;
use crate::util::timing::TimingTree;
use plonky2_maybe_rayon::*;

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    BatchFriOracle<F, C, D>
{
    /// Commits prover columns with optimized LDE and tree injection.
    pub fn from_values_prover(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: &[Option<&FftRootTable<F>>],
    ) -> Self {
        let coeffs = crate::timed!(
            timing,
            "IFFT",
            values.into_par_iter().map(|v| v.ifft()).collect()
        );
        Self::from_coeffs_prover(
            coeffs,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }

    /// Prover-only counterpart of `from_coeffs`, with identical commitments.
    pub fn from_coeffs_prover(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: &[Option<&FftRootTable<F>>],
    ) -> Self {
        Self::from_coeffs_with_options::<true>(
            polynomials,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }
}
