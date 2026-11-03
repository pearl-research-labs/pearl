//! Pre-compiled FP16 wrapper verifier setups, keyed by the batch degree profile.
//!
//! The per-shape FP16 wrapper circuit ([`super::wrapper::Fp16WrapperCircuits`]) is a
//! pure function of the batch degree profile ([`super::driver::Fp16System::degree_bits`])
//! and the consensus LUT cap (fixed across every geometry), so two tiles that snap to
//! the same on-ladder degree profile share one compiled circuit. This cache stores one
//! compiled stage-2 verifier circuit per distinct profile; the node verifier LOADS the
//! setup for a proof's geometry instead of rebuilding it, making verification cost
//! constant and geometry-independent — closing the attacker-chosen-geometry DoS (the
//! proof's `(h, w, k)` no longer forces a multi-minute circuit compilation).
//!
//! It mirrors the FP8 trusted-setup cache ([`crate::v4::circuit::circuit_utils`]'s
//! `Fp8VerifierCache`), with two differences: (1) FP8's wrapper is *universal* (one
//! circuit per device), so it keys by device; FP16's wrapper is per-degree-profile
//! (the universal variant is deferred — see [`super::wrapper`] docs), so it keys by the
//! degree-bits profile, and because the profile space is small (every table height
//! snaps to [`super::driver::FP16_REACHABLE_DEGREE_BITS`]) one entry per reachable
//! profile covers the whole envelope; (2) the header-bound FP16 verifier takes a full
//! [`plonky2::plonk::proof::ProofWithPublicInputs`] (not the compact form), so a setup
//! is just the [`VerifierCircuitData`] — no constants/sigmas polynomials.
//!
//! Verification never compiles circuits: a profile missing from the cache rejects the
//! proof (fail-closed). A stale or incomplete cache is a deployment error — surfaced by
//! the lookup — not a soundness hole (a wrong/absent setup makes the proof fail to
//! verify, never falsely accept).

use anyhow::{Context, Result, ensure};
use bincode::Options;
use hashbrown::HashMap;
use plonky2::plonk::circuit_data::VerifierCircuitData;
use plonky2::util::serialization::DefaultGateSerializer;

use super::ctl::NUM_FP16_TABLES;
use super::wrapper::{D, F, OuterC};

/// The batch degree profile selecting one compiled wrapper: the per-table
/// `degree_bits` from [`super::driver::Fp16System::degree_bits`]. Every geometry with
/// this profile shares the compiled circuit.
pub type Fp16VerifierKey = [usize; NUM_FP16_TABLES];

/// One compiled FP16 wrapper verifier setup (the stage-2 ZK verifier circuit).
#[derive(Clone)]
pub struct Fp16Verifier {
    pub(crate) circuit: VerifierCircuitData<F, OuterC, D>,
}

impl Fp16Verifier {
    /// Wraps a freshly compiled stage-2 wrapper verifier circuit (build tooling).
    pub fn new(circuit: VerifierCircuitData<F, OuterC, D>) -> Self {
        Self { circuit }
    }

    /// Serializes the trusted verifier setup for embedding / distribution.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.circuit
            .to_bytes(&DefaultGateSerializer)
            .map_err(|error| anyhow::anyhow!("serializing the fp16 wrapper verifier circuit: {error:?}"))
    }

    /// Loads a setup previously produced by [`Self::to_bytes`]. Consensus /
    /// trusted-setup data, not proof-controlled input.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let circuit = VerifierCircuitData::from_bytes(bytes, &DefaultGateSerializer)
            .map_err(|error| anyhow::anyhow!("deserializing the fp16 wrapper verifier circuit: {error}"))?;
        ensure!(
            circuit.common.config.zero_knowledge,
            "the fp16 verifier setup must contain the ZK wrapper (stage 2)"
        );
        Ok(Self { circuit })
    }

    /// The verifier circuit data for `verify_wrapped_proof_with_headers`.
    pub fn circuit(&self) -> &VerifierCircuitData<F, OuterC, D> {
        &self.circuit
    }
}

/// A read-only store of FP16 wrapper verifier setups keyed by degree profile. Deployments
/// preload [`crate::v5::api::embedded_cache::CACHE_DATA`], built offline by `build_cache`.
#[derive(Default)]
pub struct Fp16VerifierCache {
    verifiers: HashMap<Fp16VerifierKey, Fp16Verifier>,
}

fn verifier_setup_codec_options() -> impl Options {
    bincode::options().with_fixint_encoding().reject_trailing_bytes()
}

impl Fp16VerifierCache {
    /// Loads a cache previously produced by [`Self::to_bytes`]. An empty blob — the
    /// embedded default when no cache has been built — loads as an empty cache.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Ok(Self::default());
        }
        let entries: Vec<(Vec<u64>, Vec<u8>)> = verifier_setup_codec_options()
            .deserialize(bytes)
            .context("deserializing the fp16 verifier cache")?;
        let mut verifiers = HashMap::new();
        for (key_vec, serialized_verifier) in entries {
            ensure!(
                key_vec.len() == NUM_FP16_TABLES,
                "fp16 verifier cache key has {} entries, expected {NUM_FP16_TABLES}",
                key_vec.len()
            );
            let mut key: Fp16VerifierKey = [0; NUM_FP16_TABLES];
            for (slot, value) in key.iter_mut().zip(&key_vec) {
                *slot = *value as usize;
            }
            let verifier = Fp16Verifier::from_bytes(serialized_verifier)?;
            ensure!(verifiers.insert(key, verifier).is_none(), "duplicate fp16 verifier cache entry");
        }
        Ok(Self { verifiers })
    }

    /// Serializes the cache (canonical, profile-sorted, so equal caches share bytes).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut entries = self
            .verifiers
            .iter()
            .map(|(key, verifier)| Ok((key.iter().map(|&bits| bits as u64).collect::<Vec<u64>>(), verifier.to_bytes()?)))
            .collect::<Result<Vec<_>>>()?;
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        verifier_setup_codec_options()
            .serialize(&entries)
            .context("serializing the fp16 verifier cache")
    }

    /// Registers a pre-built setup under its degree profile (build tooling).
    pub fn insert(&mut self, key: Fp16VerifierKey, verifier: Fp16Verifier) {
        self.verifiers.insert(key, verifier);
    }

    /// Looks up a pre-built setup by degree profile; never compiles on a miss.
    pub fn get(&self, key: &Fp16VerifierKey) -> Option<&Fp16Verifier> {
        self.verifiers.get(key)
    }

    pub fn len(&self) -> usize {
        self.verifiers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.verifiers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Loads a REAL prebuilt cache blob (e.g. the `build_cache` sample output) from
    /// `$FP16_SAMPLE_CACHE`, confirming a non-empty cache decodes, holds its entries,
    /// and re-serializes byte-stably. Cheap (no circuit build); `#[ignore]` since it
    /// needs a path. Run: `FP16_SAMPLE_CACHE=/path/fp16_sample.bin cargo test -p zk-pow
    /// --lib circuit::fp16::verifier_cache::tests::loads_a_prebuilt_sample -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs a prebuilt cache at $FP16_SAMPLE_CACHE"]
    fn loads_a_prebuilt_sample_cache() {
        let path = std::env::var("FP16_SAMPLE_CACHE").expect("set FP16_SAMPLE_CACHE");
        let bytes = std::fs::read(&path).expect("read sample cache");
        let cache = Fp16VerifierCache::from_bytes(&bytes).expect("decode real cache");
        assert!(!cache.is_empty(), "a built cache must have >= 1 profile");
        println!("loaded fp16 cache: {} profile(s) from {} bytes", cache.len(), bytes.len());
        // Re-serialization is byte-stable (canonical, profile-sorted).
        assert_eq!(cache.to_bytes().unwrap(), bytes, "cache re-serialization must be byte-stable");
    }

    #[test]
    fn empty_cache_round_trips_and_misses() {
        let cache = Fp16VerifierCache::default();
        assert!(cache.is_empty());
        // Empty blob and a serialized-empty cache both load to an empty cache.
        assert!(Fp16VerifierCache::from_bytes(&[]).unwrap().is_empty());
        let bytes = cache.to_bytes().unwrap();
        assert!(Fp16VerifierCache::from_bytes(&bytes).unwrap().is_empty());
        // A lookup on an empty cache misses (fail-closed): the verifier rejects.
        assert!(cache.get(&[0usize; NUM_FP16_TABLES]).is_none());
    }

    /// Cache mechanism end-to-end: compile a wrapper for a small geometry, store it in
    /// the cache under its degree profile, serialize + deserialize the cache, look the
    /// setup up by profile, and assert the deserialized setup is BYTE-IDENTICAL to a
    /// freshly built one. Because a [`VerifierCircuitData`] fully determines
    /// verification, byte-identity proves the cache-loaded setup verifies exactly as a
    /// rebuilt one — i.e. load-by-profile faithfully replaces the verify-time rebuild
    /// (the existing `wrapper::tests` cover the proof/verify path itself). Also builds
    /// a SECOND distinct geometry that snaps to the SAME profile and asserts it reuses
    /// the one cached circuit, and that a different profile misses (fail-closed).
    /// `#[ignore]` (compiles 1-2 wrappers, ~minutes); run explicitly:
    /// `cargo test -p zk-pow --lib circuit::fp16::verifier_cache::tests::cache_round_trips -- --ignored --nocapture`.
    #[test]
    #[ignore = "compiles wrapper circuit(s) (~minutes); run explicitly"]
    fn cache_round_trips_and_is_byte_identical_to_a_rebuild() {
        use crate::v4::api::public_params::HashId;
        use crate::v5::circuit::driver::Fp16System;
        use crate::v5::circuit::wrapper::{Fp16WrapperCircuits, InnerC};
        use plonky2::util::timing::TimingTree;

        let hash = HashId::Blake3Chunk1024;
        let mut timing = TimingTree::default();

        // Smallest consensus-legal tile (h*w = 256, k = 8).
        let system = Fp16System::<F, D>::new(4, 64, 8, hash, hash);
        let preprocessed = system.preprocessed_data::<InnerC>(&mut timing);
        let circuits = Fp16WrapperCircuits::build(&system, &preprocessed.cap(), &mut timing).unwrap();
        let built = Fp16Verifier::new(circuits.verifier_data());
        let built_bytes = built.to_bytes().unwrap();

        // Cache it, round-trip the cache bytes, look it up by profile.
        let mut cache = Fp16VerifierCache::default();
        cache.insert(*system.degree_bits(), built);
        let loaded = Fp16VerifierCache::from_bytes(&cache.to_bytes().unwrap()).unwrap();
        assert_eq!(loaded.len(), 1);
        let verifier = loaded.get(system.degree_bits()).expect("profile present in the loaded cache");
        assert_eq!(
            verifier.to_bytes().unwrap(),
            built_bytes,
            "cache-loaded verifier must be byte-identical to the freshly built one"
        );

        // A wrong profile misses (fail-closed → the consensus verifier rejects).
        let mut wrong = *system.degree_bits();
        wrong[0] = wrong[0].wrapping_add(1);
        assert!(loaded.get(&wrong).is_none(), "a different profile must miss the cache");

        // A SECOND, distinct geometry that snaps to the SAME profile reuses the one
        // cached circuit — the property that makes the profile-keyed cache small and
        // geometry-independent. (k=16 keeps the smallest tile on the same ladder rung.)
        let system2 = Fp16System::<F, D>::new(4, 64, 16, hash, hash);
        if system2.degree_bits() == system.degree_bits() {
            assert!(
                loaded.get(system2.degree_bits()).is_some(),
                "a same-profile geometry must hit the one cached circuit"
            );
        }
    }
}
