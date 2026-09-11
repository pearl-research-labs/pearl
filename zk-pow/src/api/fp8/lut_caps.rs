//! LUT Merkle cap loaded from committed `lut_caps.bin`.
//!
//! The LUT tables (`crate::circuit::fp8::luts`) share a job-independent
//! commitment layout ([`Fp8System::preprocessed_data`](crate::circuit::fp8::driver::Fp8System::preprocessed_data)).
//! [`committed_lut_cap`] reads the file without rebuilding the oracle.
//!
//! After changes to the tables, commitment layout or STARK configuration,
//! regenerate the cap and matching verifier cache together:
//! `cargo run --release --no-default-features --bin build_cache src/api/fp8/fp8_cache.bin`.
//! The tool uses [`derive_lut_caps_bytes`]; `lut_caps_file_matches_the_lut_tables`
//! checks the committed cap against a fresh derivation.

use anyhow::{Result, ensure};
use plonky2::field::types::{Field, Field64, PrimeField64};
use plonky2::hash::hash_types::HashOut;
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::plonk::config::GenericConfig;
use plonky2::util::timing::TimingTree;

use super::zk::{C, D, F, Fp8Job, sample_dense_statement};
use crate::api::primitives::IncompleteBlockHeader;
use crate::circuit::fp8::driver::STARK_CAP_HEIGHT;

/// The Merkle cap type of the committed LUT oracle.
pub type LutCap = MerkleCap<F, <C as GenericConfig<D>>::Hasher>;

/// The committed caps file: one cap of [`LUT_CAP_LEN`] hashes, each four
/// little-endian `u64` field elements.
const LUT_CAPS_DATA: &[u8] = include_bytes!("lut_caps.bin");

pub(crate) const LUT_CAP_LEN: usize = 1 << STARK_CAP_HEIGHT;
const CAP_BYTES: usize = LUT_CAP_LEN * 4 * 8;

// Check the embedded file length at compile time.
const _: () = assert!(LUT_CAPS_DATA.len() == CAP_BYTES);

/// The LUT setup cap, parsed from the committed file.
pub fn committed_lut_cap() -> LutCap {
    cap_from_file_bytes(LUT_CAPS_DATA)
}

/// Like [`committed_lut_cap`], reading explicit bytes for build tooling.
pub fn cap_from_file_bytes(bytes: &[u8]) -> LutCap {
    assert_eq!(bytes.len(), CAP_BYTES, "malformed LUT caps file ({} bytes)", bytes.len());
    MerkleCap(
        bytes
            .as_chunks::<32>()
            .0
            .iter()
            .map(|hash| HashOut {
                elements: std::array::from_fn(|i| {
                    let element = u64::from_le_bytes(hash[i * 8..][..8].try_into().unwrap());
                    assert!(element < F::ORDER, "non-canonical field element in the LUT caps file");
                    F::from_canonical_u64(element)
                }),
            })
            .collect(),
    )
}

/// Derive fresh `lut_caps.bin` bytes. `build_cache` also compiles
/// `fp8_cache.bin` against this cap.
pub fn derive_lut_caps_bytes() -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(CAP_BYTES);
    // The fixture supplies the batch layout; preprocessing is job-independent.
    let params = sample_dense_statement()?;
    let job = Fp8Job::derive(&params, &IncompleteBlockHeader::zero())?;
    let cap = job.system.preprocessed_data::<C>(&mut TimingTree::default()).cap();
    ensure!(
        cap.0.len() == LUT_CAP_LEN,
        "LUT cap height differs from the consensus StarkConfig"
    );
    for hash in &cap.0 {
        for element in hash.elements {
            bytes.extend_from_slice(&element.to_canonical_u64().to_le_bytes());
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Check the committed cap against the LUT tables. Regenerate both files
    /// with `build_cache src/api/fp8/fp8_cache.bin` after a setup change.
    #[test]
    fn lut_caps_file_matches_the_lut_tables() {
        let fresh = derive_lut_caps_bytes().expect("deriving the LUT caps from the tables");
        assert!(
            fresh == LUT_CAPS_DATA,
            "the committed lut_caps.bin drifted from the LUT tables; regenerate it (and fp8_cache.bin) with build_cache"
        );
    }
}
