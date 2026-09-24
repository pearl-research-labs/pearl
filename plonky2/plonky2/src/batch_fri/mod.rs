//! Batched FRI protocol. It allows proving multiple polynomials
//! of different degrees with a single FRI argument: the FRI folding starts
//! from the largest LDE and, whenever the folded codeword reaches the size
//! of the next (smaller) instance, that instance's reduced openings are
//! absorbed into the running codeword.
//!
//! Compared to upstream plonky2, this port additionally supports initial
//! oracles whose tallest polynomial group is *smaller* than the tallest
//! instance (e.g. a preprocessed-data oracle containing only small lookup
//! tables). Query indices are shifted per-oracle accordingly.
//!
//! The batch FRI implementation does not support zero-knowledge (hiding)
//! mode; `FriParams::hiding` must be `false`.

pub mod oracle;
pub mod prover;
pub mod recursive_verifier;
pub mod verifier;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::field::extension::Extendable;
use crate::fri::structure::{FriInstanceInfo, FriInstanceInfoTarget, FriPolynomialInfo};
use crate::hash::hash_types::RichField;

/// One contiguous run of same-oracle polynomials inside one opening batch of
/// one batched-FRI instance, tagged with the global `alpha` exponent of its
/// first claim.
///
/// Batched FRI folds every opened claim `(polynomial, point)` into a single
/// codeword using powers of one challenge `alpha`:
///
///   `E_i(X) = Σ_runs alpha^offset(run) · Σ_j alpha^j · (f_run,j(X) − f_run,j(z_run)) / (X − z_run)`
///
/// Exponents are assigned *canonically*: runs are enumerated
/// `(oracle_index, poly_start, batch)`-ascending and offsets accumulate run
/// lengths in that order, so every claim gets a distinct power and a claim's
/// power depends only on which claims are opened — not on how the instances
/// group by degree. Prover and verifiers (native and recursive) must use the
/// same assignment; a universal recursive verifier additionally relies on it
/// being independent of the tables' degree profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlphaRun {
    /// Index of the run's instance (degree group).
    pub instance: usize,
    /// Index of the run's batch (opening point) within the instance.
    pub batch: usize,
    /// Position of the run's first polynomial in the batch's flat polynomial list.
    pub flat_start: usize,
    /// The oracle holding the run's polynomials.
    pub oracle_index: usize,
    /// First polynomial of the run within the oracle.
    pub poly_start: usize,
    /// Number of polynomials in the run.
    pub len: usize,
    /// Global `alpha` exponent of the run's first claim.
    pub alpha_offset: usize,
}

/// The canonical [`AlphaRun`]s of a batch, `(instance, batch, flat_start)`-ascending.
pub fn batch_alpha_runs<F: RichField + Extendable<D>, const D: usize>(
    instances: &[FriInstanceInfo<F, D>],
) -> Vec<AlphaRun> {
    alpha_runs(
        instances
            .iter()
            .map(|inst| {
                inst.batches
                    .iter()
                    .map(|b| b.polynomials.as_slice())
                    .collect()
            })
            .collect(),
    )
}

/// Circuit-shape twin of [`batch_alpha_runs`]: the assignment depends only on
/// the polynomial structure, which `FriInstanceInfoTarget` shares.
pub fn batch_alpha_runs_target<const D: usize>(
    instances: &[FriInstanceInfoTarget<D>],
) -> Vec<AlphaRun> {
    alpha_runs(
        instances
            .iter()
            .map(|inst| {
                inst.batches
                    .iter()
                    .map(|b| b.polynomials.as_slice())
                    .collect()
            })
            .collect(),
    )
}

fn alpha_runs(instances: Vec<Vec<&[FriPolynomialInfo]>>) -> Vec<AlphaRun> {
    let mut runs = Vec::new();
    for (i, batches) in instances.iter().enumerate() {
        for (k, polys) in batches.iter().enumerate() {
            let mut start = 0;
            while start < polys.len() {
                let oracle_index = polys[start].oracle_index;
                let poly_start = polys[start].polynomial_index;
                let mut len = 1;
                while start + len < polys.len()
                    && polys[start + len].oracle_index == oracle_index
                    && polys[start + len].polynomial_index == poly_start + len
                {
                    len += 1;
                }
                runs.push(AlphaRun {
                    instance: i,
                    batch: k,
                    flat_start: start,
                    oracle_index,
                    poly_start,
                    len,
                    alpha_offset: 0,
                });
                start += len;
            }
        }
    }
    // Canonical exponent assignment: enumerate runs by claim identity,
    // accumulating lengths; runs stay (instance, batch, flat_start)-ascending.
    let mut order: Vec<usize> = (0..runs.len()).collect();
    order.sort_by_key(|&r| (runs[r].oracle_index, runs[r].poly_start, runs[r].batch));
    let mut acc = 0;
    for &r in &order {
        runs[r].alpha_offset = acc;
        acc += runs[r].len;
    }
    runs
}
