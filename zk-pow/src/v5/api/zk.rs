//! FP16 (A100) ZK prover: turns a winning tile into the header-bound recursive
//! certificate the consensus path now carries.
//!
//! This is the prover-side analogue of [`crate::v4::api::zk::Fp8Prover`], and the
//! exact inverse of the consensus verifier
//! ([`crate::v5::circuit::wrapper::verify_wrapped_proof_with_headers`], wired at
//! the FFI by `verify_fp16_zk_cert_ffi`). Where the retired plaintext path opened
//! the operand strips and let the verifier replay the tile, the prover now:
//!
//! 1. derives the per-side opening keys from the headers exactly as the verifier
//!    does — `keyA = key_a(proposed)`, `keyB = key_b(ancestor)` — and the noise
//!    seeds from the committed operand roots + keys + the job's `p` encoding
//!    ([`Fp16System::root_derived_seeds`]), so the lottery-compression jackpot key
//!    is the header/root-derived `jackpot_key(seed_a)` the verifier pins;
//! 2. proves the FP16 batch ([`Fp16System::prove`]) under those header-derived
//!    keys, deriving the Fiat-Shamir statement digest from the proven
//!    `HASH_JACKPOT`;
//! 3. wraps it to the constant-size stage-2 recursive proof
//!    ([`Fp16WrapperCircuits::prove`]) and packages it with the public job
//!    statement into an [`Fp16ZkCertificate`].
//!
//! An honestly-produced certificate therefore passes
//! `verify_wrapped_proof_with_headers` under the same headers, because every key /
//! seed / digest it commits is the header-derived value the verifier recomputes.
//!
//! # Setup cost / caching
//!
//! The FP16 wrapper is compiled **per tile shape** (the universal wrapper is the
//! documented residual, `docs/fp16_scheme/stark_feasibility.md §8`): compiling the
//! two wrapper stages for one `(h, w, k, hash_id)` geometry costs minutes.
//! [`set_noise_seed_params`] only affects the witness (the seed/public-input
//! values), never the circuit shape, so [`Fp16Prover`] caches one setup per
//! geometry and reuses it across blocks — each block then pays only the (still
//! minutes-long) proving, not recompilation. This mirrors [`Fp8Prover`] caching
//! per device; FP16 caches per geometry because its wrapper is not yet universal.
//!
//! [`Fp8Prover`]: crate::v4::api::zk::Fp8Prover
//! [`set_noise_seed_params`]: crate::v5::circuit::driver::Fp16System::set_noise_seed_params

use std::collections::HashMap;

use anyhow::{ensure, Result};
use plonky2::util::timing::TimingTree;
use starky::batch_prover::BatchStarkPreprocessedData;

use super::noise::commitment_keys;
use super::plain_proof::{Fp16JobParams, Fp16PlainProof};
use super::zk_cert::Fp16ZkCertificate;
use crate::v4::api::public_params::HashId;
use crate::v4::api::transcript::jackpot_key;
use crate::v4::api::primitives::{IncompleteBlockHeader, Sides};
use crate::v5::circuit::ctl::BLAKE3_A100_TABLE;
use crate::v5::circuit::driver::Fp16System;
use crate::v5::circuit::wrapper::{Fp16WrapperCircuits, D, F, InnerC};

/// One tile shape's compiled prover setup: the LUT precommitment and the
/// two-stage wrapper circuits. Shape-only — reused across blocks/jobs. The
/// `Fp16System` itself is NOT retained (it holds non-`Send` STARK trait objects,
/// and reconstructing it from the geometry is cheap — construction only, no
/// circuit work), so this setup stays `Send` and the prover is a plain pyclass.
struct Fp16ProverSetup {
    preprocessed: BatchStarkPreprocessedData<F, InnerC, D>,
    circuits: Fp16WrapperCircuits,
}

/// A reusable FP16 prover. Retains one compiled setup per tile geometry
/// `(h, w, k, hash_id_A, hash_id_B)` and reuses it across proofs.
#[derive(Default)]
pub struct Fp16Prover {
    setups: HashMap<(usize, usize, usize, u8, u8), Fp16ProverSetup>,
}

impl Fp16Prover {
    /// An empty prover; geometries are compiled lazily on first [`Self::prove`]
    /// (or eagerly via [`Self::setup_geometry`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// Compiles and retains the wrapper circuits for one tile geometry now, so a
    /// later [`Self::prove`] of that shape does not pay the compilation cost.
    /// A no-op if the geometry is already cached.
    pub fn setup_geometry(&mut self, h: usize, w: usize, k: usize, a_hash: HashId, b_hash: HashId) -> Result<()> {
        let mut timing = TimingTree::default();
        self.ensure_setup(h, w, k, a_hash, b_hash, &mut timing)
    }

    fn ensure_setup(
        &mut self,
        h: usize,
        w: usize,
        k: usize,
        a_hash: HashId,
        b_hash: HashId,
        timing: &mut TimingTree,
    ) -> Result<()> {
        let key = (h, w, k, a_hash as u8, b_hash as u8);
        if !self.setups.contains_key(&key) {
            let system = Fp16System::<F, D>::new(h, w, k, a_hash, b_hash);
            let preprocessed = system.preprocessed_data::<InnerC>(timing);
            let circuits = Fp16WrapperCircuits::build(&system, &preprocessed.cap(), timing)?;
            self.setups.insert(key, Fp16ProverSetup { preprocessed, circuits });
        }
        Ok(())
    }

    /// Proves one winning tile and returns the header-bound [`Fp16ZkCertificate`]
    /// (the bytes the node's `CertificateV5.ProofData` carries).
    ///
    /// `a_codes` is the opened `h x k` A tile and `b_codes` the `w x k` B tile
    /// (row-major FP16 bit patterns, the same codes the miner's search ran over);
    /// `job` is the public statement (ancestor header, `k`/`r`, per-side
    /// `num_rows`/`hash_id`/pattern) whose patterns fix `(h, w)`. `proposed_header`
    /// keys the A side and the A branch of the seed chain.
    pub fn prove(
        &mut self,
        proposed_header: &IncompleteBlockHeader,
        job: &Fp16JobParams,
        a_codes: &[u16],
        b_codes: &[u16],
    ) -> Result<Fp16ZkCertificate> {
        let mut timing = TimingTree::default();
        self.prove_with_timing(proposed_header, job, a_codes, b_codes, &mut timing)
    }

    /// Proves directly from an [`Fp16PlainProof`] — the opener bundle the miner
    /// already assembles for a winning tile. This authenticates and re-opens the
    /// tile codes from the committed Merkle openings under `proposed_header`
    /// (so a malformed opening is caught before the minutes-long prove), then
    /// proves them under the bundle's job. The retired plaintext certificate is
    /// thus repurposed as the prover's input, not a submitted artifact.
    pub fn prove_from_plain_proof(
        &mut self,
        proposed_header: &IncompleteBlockHeader,
        plain_proof: &Fp16PlainProof,
    ) -> Result<Fp16ZkCertificate> {
        let opened = plain_proof.parse_proof(proposed_header)?;
        self.prove(proposed_header, &plain_proof.job, &opened.a_rows, &opened.b_rows)
    }

    fn prove_with_timing(
        &mut self,
        proposed_header: &IncompleteBlockHeader,
        job: &Fp16JobParams,
        a_codes: &[u16],
        b_codes: &[u16],
        timing: &mut TimingTree,
    ) -> Result<Fp16ZkCertificate> {
        let h = job.operands.a.pattern.tile_size() as usize;
        let w = job.operands.b.pattern.tile_size() as usize;
        let k = job.k as usize;
        let (a_hash, b_hash) = (job.operands.a.hash_id, job.operands.b.hash_id);
        ensure!(a_codes.len() == h * k, "A operand must be h*k = {h}*{k} FP16 codes, got {}", a_codes.len());
        ensure!(b_codes.len() == w * k, "B operand must be w*k = {w}*{k} FP16 codes, got {}", b_codes.len());

        // Header-derived per-side opening keys (bit-exact with the verifier).
        let keys = commitment_keys(proposed_header, &job.ancestor_header);

        // Compile (or reuse) the geometry's wrapper circuits + LUT precommitment.
        self.ensure_setup(h, w, k, a_hash, b_hash, timing)?;
        let setup = self.setups.get(&(h, w, k, a_hash as u8, b_hash as u8)).expect("setup ensured above");

        // Reconstruct the (cheap, non-`Send`) system for this geometry and fold
        // THIS job's `p` into its witness seed chain. `p` affects only the
        // seed/public-input witness, never the circuit shape, so the cached
        // `preprocessed`/`circuits` (built from an identical system) stay valid.
        let mut system = Fp16System::<F, D>::new(h, w, k, a_hash, b_hash);
        system.set_noise_seed_params(Sides { a: job.encode_p_a(), b: job.encode_p_b() });

        // The lottery-compression key the verifier pins: jackpot_key(seed_a) over
        // the header/root-derived seed_a. `root_derived_seeds` commits the operands
        // under the header keys and runs the B-then-A chain, bit-exact with
        // `api::fp16::noise::noise_seeds`.
        let seeds = system.root_derived_seeds(a_codes, b_codes, keys.a, keys.b)?;
        let jk = jackpot_key(&seeds.a);

        // Prove the batch under the header-derived keys, then wrap to stage 2. The
        // statement digest is DERIVED from the proven HASH_JACKPOT (not caller
        // salt), exactly what the verifier recomputes and pins.
        let batch = system.prove::<InnerC>(a_codes, b_codes, keys.a, keys.b, jk, &setup.preprocessed, timing)?;
        let digest = Fp16System::<F, D>::statement_digest(&batch.public_inputs[BLAKE3_A100_TABLE]);
        let wrapped = setup.circuits.prove(&system, &batch, digest, timing)?;

        Ok(Fp16ZkCertificate { job: job.clone(), proof_bytes: wrapped.to_bytes() })
    }
}

#[cfg(test)]
mod fixture {
    //! Generates + self-verifies the committed Go/Rust FP16 **ZK-cert** fixture
    //! (`node/zkpow/testdata/fp16_zk_cert_a100.bin`): `header(76) | u32le cert_len |
    //! Fp16ZkCertificate::to_bytes()`. The self-check is a full header-bound
    //! prove -> verify-through-the-production-cache cycle (mirroring the FFI), plus a
    //! tamper-rejects assertion. Heavy (recursive wrap ~min); run explicitly:
    //! ```text
    //! cargo test -p zk-pow --lib --no-default-features -- \
    //!   api::fp16::zk::fixture::regenerate_zk_cert_fixture --ignored --exact --nocapture
    //! ```
    use super::*;
    use crate::v5::api::commitment::{commit_operand, open_rows};
    use crate::v5::api::dtype::f32_to_fp16;
    use crate::v5::api::noise::{key_a, key_b};
    use crate::v5::api::params::Fp16Device;
    use crate::v5::api::plain_proof::{Fp16MatrixProof, Fp16OperandParams, Fp16PlainProof};
    use crate::v4::api::layout::AxisPattern;
    use crate::v4::api::layout::DimType::{Blake, Fold};
    use crate::v4::api::primitives::Sides;
    use crate::v5::circuit::verifier_cache::Fp16VerifierCache;
    use crate::v5::circuit::wrapper::verify_wrapped_proof_with_headers;
    use plonky2::plonk::proof::ProofWithPublicInputs;

    const M: usize = 8;
    const N: usize = 128;
    const K: usize = 256;
    const R: u32 = 32;
    const EASY: u32 = 0x207f_ffff;
    const HASH: HashId = HashId::Blake3Chunk1024;

    /// The asymmetric `honest_fixture` header (matches `api::fp16::verify` tests):
    /// version 0, prev_block [0..31], merkle_root [0x40..0x5f], ts 0x66666666.
    fn asymmetric_test_header(nbits: u32) -> IncompleteBlockHeader {
        IncompleteBlockHeader {
            version: 0,
            prev_block: std::array::from_fn(|i| i as u8),
            merkle_root: std::array::from_fn(|i| 0x40u8 + i as u8),
            timestamp: 0x6666_6666,
            nbits,
        }
    }

    /// Spread-magnitude FP16 operands (the honest/admissible regime), byte-identical
    /// to `api::fp16::verify`'s `Gen`.
    fn spread_operands(na: usize, nb: usize, seed: u64) -> (Vec<u16>, Vec<u16>) {
        let mut s = seed;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut op = |n: usize| -> Vec<u16> {
            (0..n)
                .map(|_| {
                    let r = next();
                    let sign = if r & 1 == 0 { 1.0 } else { -1.0 };
                    let exp = ((r >> 1) % 9) as i32 - 3;
                    let mant = 1.0 + ((r >> 8) % 1024) as f32 / 1024.0;
                    f32_to_fp16(sign * mant * 2f32.powi(exp)).unwrap()
                })
                .collect()
        };
        (op(na), op(nb))
    }

    fn build_plain_proof(header: &IncompleteBlockHeader, a_full: &[u16], b_full: &[u16]) -> Fp16PlainProof {
        let rp = AxisPattern::new(&[(4, Blake)]).unwrap(); // h = 4
        let cp = AxisPattern::new(&[(4, Blake), (16, Fold)]).unwrap(); // w = 64
        let tree_a = commit_operand(a_full, M, K, HASH, key_a(header)).unwrap();
        let tree_b = commit_operand(b_full, N, K, HASH, key_b(header)).unwrap();
        let a_idx: Vec<usize> = rp.tile_offsets().iter().map(|&o| o as usize).collect();
        let b_idx: Vec<usize> = cp.tile_offsets().iter().map(|&o| o as usize).collect();
        let pa = open_rows(&tree_a, &a_idx, M, K, HASH).unwrap();
        let pb = open_rows(&tree_b, &b_idx, N, K, HASH).unwrap();
        Fp16PlainProof {
            job: Fp16JobParams {
                ancestor_header: *header,
                device: Fp16Device::A100,
                k: K as u32,
                r: R,
                operands: Sides {
                    a: Fp16OperandParams { num_rows: M as u32, hash_id: HASH, pattern: rp },
                    b: Fp16OperandParams { num_rows: N as u32, hash_id: HASH, pattern: cp },
                },
            },
            values: Sides {
                a: Fp16MatrixProof { proof: pa, row_indices: a_idx },
                b: Fp16MatrixProof { proof: pb, row_indices: b_idx },
            },
        }
    }

    #[test]
    #[ignore = "recursive FP16 wrap (~min) + writes the committed ZK-cert fixture; run explicitly"]
    fn regenerate_zk_cert_fixture() {
        let header = asymmetric_test_header(EASY);
        let (a_full, b_full) = spread_operands(M * K, N * K, 0xdead_beef_0bad_f00d);
        let plain = build_plain_proof(&header, &a_full, &b_full);

        // PROVE (header-bound ZK cert from the opener bundle).
        let mut prover = Fp16Prover::new();
        let cert = prover.prove_from_plain_proof(&header, &plain).expect("prove the winning tile");

        // VERIFY through the PRODUCTION cache, exactly as the FFI does.
        let (h, w, k) = cert.tile_geometry();
        let (a_hash, b_hash) = (cert.job.operands.a.hash_id, cert.job.operands.b.hash_id);
        let cache_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/api/fp16/fp16_cache.bin");
        let cache = Fp16VerifierCache::from_bytes(&std::fs::read(&cache_path).expect("read fp16_cache.bin"))
            .expect("parse production cache");
        let mut system = Fp16System::<F, D>::new(h, w, k, a_hash, b_hash);
        let verifier = cache.get(system.degree_bits()).expect("production cache covers the fixture profile");
        let vd = verifier.circuit();
        let proof = ProofWithPublicInputs::from_bytes(cert.proof_bytes.clone(), &vd.common).expect("deserialize");
        // The verifier folds THIS job's `p` into the system internally.
        verify_wrapped_proof_with_headers(&mut system, vd, &proof, &header, &cert.job, EASY)
            .expect("header-bound prove -> verify-through-cache must accept the honest cert");

        // TAMPER: a flipped proof byte must be rejected.
        let mut bad = cert.proof_bytes.clone();
        *bad.last_mut().unwrap() ^= 1;
        assert!(
            ProofWithPublicInputs::from_bytes(bad, &vd.common)
                .map_err(|e| e.to_string())
                .and_then(|p| verify_wrapped_proof_with_headers(&mut system, vd, &p, &header, &cert.job, EASY).map_err(|e| e.to_string()))
                .is_err(),
            "a tampered wrapped proof must be rejected"
        );

        // WRITE the fixture: header(76) | u32le cert_len | cert bytes.
        let cert_bytes = cert.to_bytes().expect("serialize Fp16ZkCertificate");
        let mut out = Vec::with_capacity(76 + 4 + cert_bytes.len());
        out.extend_from_slice(&header.to_bytes());
        out.extend_from_slice(&(cert_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&cert_bytes);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../node/zkpow/testdata/fp16_zk_cert_a100.bin");
        std::fs::write(&path, &out).expect("write ZK-cert fixture");
        println!("wrote FP16 ZK-cert fixture ({} bytes) to {}", out.len(), path.display());
    }
}
