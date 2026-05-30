//! GPU offload of `PolynomialBatch::from_values` for GoldilocksField /
//! PoseidonGoldilocksConfig / D=2 / blinding=false (the STARK#0 trace + aux +
//! quotient commitments). Byte-exact-by-value to the CPU path (validated by
//! `pearl_gpu_commit`); GoldilocksField compares by canonical value, so the real
//! prover + verifier accept the GPU-built commitment unchanged.

use core::any::TypeId;
use core::mem::{forget, transmute_copy};

use plonky2_maybe_rayon::*;

use plonky2_field::fft::fft_root_table;
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2_field::types::{Field, Field64, PrimeField64};

use crate::fri::oracle::PolynomialBatch;
use crate::hash::hash_types::HashOut;
use crate::hash::merkle_tree::{MerkleCap, MerkleTree};
use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use crate::util::log2_strict;

type GF = GoldilocksField;
type PC = PoseidonGoldilocksConfig;

unsafe extern "C" {
    fn pearl_gpu_from_values_f64(
        values: *const u64, num_polys: i32, degree_log: i32, rate_bits: i32, cap_height: i32,
        rtn: *const u64, offn: *const i32, rtl: *const u64, offl: *const i32,
        coset_shift: u64, n_inv: u64,
        out_coeffs: *mut u64, out_leaves: *mut u64, out_digests: *mut u64, out_cap: *mut u64,
    ) -> i32;
}

fn flat_root_table(m: usize) -> (Vec<u64>, Vec<i32>) {
    let rt = fft_root_table::<GF>(m);
    let mut flat = Vec::new();
    let mut off = Vec::with_capacity(rt.len());
    for row in &rt {
        off.push(flat.len() as i32);
        for x in row {
            flat.push(x.to_canonical_u64());
        }
    }
    (flat, off)
}

pub(crate) fn try_gpu_from_values<F, C, const D: usize>(
    values: &[PolynomialValues<F>],
    rate_bits: usize,
    cap_height: usize,
) -> Option<PolynomialBatch<F, C, D>>
where
    F: crate::hash::hash_types::RichField + plonky2_field::extension::Extendable<D> + 'static,
    C: GenericConfig<D, F = F> + 'static,
{
    if TypeId::of::<F>() != TypeId::of::<GF>() || TypeId::of::<C>() != TypeId::of::<PC>() {
        return None;
    }
    if values.is_empty() {
        return None;
    }
    // F == GoldilocksField here, so this reinterpretation is sound.
    let vals: &[PolynomialValues<GF>] =
        unsafe { &*(values as *const [PolynomialValues<F>] as *const [PolynomialValues<GF>]) };
    let num_polys = vals.len();
    let n = vals[0].values.len();
    let degree_log = log2_strict(n);
    let total = n << rate_bits;
    let len_cap = 1usize << cap_height;
    let dig_len = 2 * (total - len_cap);

    let input: Vec<u64> = vals
        .par_iter()
        .flat_map_iter(|p| p.values.iter().map(|v| v.to_canonical_u64()))
        .collect();
    let (rtn, offn) = flat_root_table(n);
    let (rtl, offl) = flat_root_table(total);
    let n_inv = GF::inverse_2exp(degree_log).to_canonical_u64();
    let shift = GF::coset_shift().to_canonical_u64();

    let mut out_coeffs = vec![0u64; num_polys * n];
    let mut out_leaves = vec![0u64; total * num_polys];
    let mut out_digests = vec![0u64; dig_len * 4];
    let mut out_cap = vec![0u64; len_cap * 4];
    let rc = unsafe {
        pearl_gpu_from_values_f64(
            input.as_ptr(), num_polys as i32, degree_log as i32, rate_bits as i32, cap_height as i32,
            rtn.as_ptr(), offn.as_ptr(), rtl.as_ptr(), offl.as_ptr(), shift, n_inv,
            out_coeffs.as_mut_ptr(), out_leaves.as_mut_ptr(), out_digests.as_mut_ptr(), out_cap.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return None;
    }

    // Reconstruct in parallel (rayon) — this dominates the host cost. The GPU
    // already canonicalized, so from_canonical_u64 is a no-op wrap in release.
    let polynomials: Vec<PolynomialCoeffs<GF>> = out_coeffs
        .par_chunks(n)
        .map(|ch| PolynomialCoeffs { coeffs: ch.iter().map(|&u| GF::from_noncanonical_u64(u)).collect() })
        .collect();
    let leaves: Vec<Vec<GF>> = out_leaves
        .par_chunks(num_polys)
        .map(|ch| ch.iter().map(|&u| GF::from_noncanonical_u64(u)).collect())
        .collect();
    let digests: Vec<HashOut<GF>> = out_digests
        .par_chunks(4)
        .map(|c| HashOut { elements: [c[0], c[1], c[2], c[3]].map(GF::from_noncanonical_u64) })
        .collect();
    let cap: Vec<HashOut<GF>> = out_cap
        .par_chunks(4)
        .map(|c| HashOut { elements: [c[0], c[1], c[2], c[3]].map(GF::from_noncanonical_u64) })
        .collect();

    let merkle_tree = MerkleTree::<GF, <PC as GenericConfig<2>>::Hasher> {
        leaves,
        digests,
        cap: MerkleCap(cap),
    };
    let concrete: PolynomialBatch<GF, PC, 2> = PolynomialBatch {
        polynomials,
        merkle_tree,
        degree_log,
        rate_bits,
        blinding: false,
    };
    // Same concrete types (verified via TypeId) => identical layout.
    let out: PolynomialBatch<F, C, D> = unsafe { transmute_copy(&concrete) };
    forget(concrete);
    Some(out)
}
