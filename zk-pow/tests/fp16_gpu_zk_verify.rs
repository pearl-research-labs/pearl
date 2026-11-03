//! On-hardware FP16/A100 end-to-end ZK consensus validation.
//!
//! Ingests the plain-proof bundle produced by the REAL sm_80 GPU lottery search
//! (`pearl_gemm.fp16_miner.search_block` on the local CMP 170HX, dumped by the
//! driver as `header(76) | u32le len | Fp16PlainProof::to_bytes()`), proves it
//! into the header-bound `Fp16ZkCertificate` with the production `Fp16Prover`,
//! and verifies that wrapped proof through the PRODUCTION verifier cache
//! (`src/api/fp16/fp16_cache.bin`) via `verify_wrapped_proof_with_headers` --
//! exactly the path the node FFI (`verify_fp16_zk_cert_ffi`) runs. A flipped
//! proof byte must be rejected.
//!
//! Heavy (recursive wrap ~12-15 min); ignored by default. Run:
//! ```text
//! GPU_FP16_BUNDLE=/mnt/raid/projects/pearl-scratch/gpu_fp16_plain_proof.bin \
//!   cargo test -p zk-pow --test fp16_gpu_zk_verify --release -- \
//!   gpu_tile_proves_and_verifies_through_cache --ignored --exact --nocapture
//! ```

use plonky2::plonk::proof::ProofWithPublicInputs;

use zk_pow::v5::api::zk::Fp16Prover;
use zk_pow::v5::api::plain_proof::Fp16PlainProof;
use zk_pow::v4::api::primitives::IncompleteBlockHeader;
use zk_pow::v5::circuit::driver::Fp16System;
use zk_pow::v5::circuit::verifier_cache::Fp16VerifierCache;
use zk_pow::v5::circuit::wrapper::{verify_wrapped_proof_with_headers, D, F};

const EASY: u32 = 0x207f_ffff;

#[test]
#[ignore = "GPU-sourced recursive FP16 wrap (~12-15 min); run explicitly with GPU_FP16_BUNDLE set"]
fn gpu_tile_proves_and_verifies_through_cache() {
    // 1. Load the GPU-produced bundle: header(76) | u32le len | plain-proof bytes.
    let path = std::env::var("GPU_FP16_BUNDLE")
        .expect("set GPU_FP16_BUNDLE to the GPU plain-proof bundle path");
    let raw = std::fs::read(&path).expect("read GPU plain-proof bundle");
    assert!(raw.len() > 80, "bundle too short");
    let header = IncompleteBlockHeader::from_bytes(&raw[..76]).expect("parse header");
    let plen = u32::from_le_bytes(raw[76..80].try_into().unwrap()) as usize;
    assert_eq!(80 + plen, raw.len(), "bundle length mismatch");
    let plain = Fp16PlainProof::from_bytes(&raw[80..80 + plen]).expect("parse Fp16PlainProof");
    println!(
        "GPU bundle: header nbits={:#010x}, plain-proof {} bytes, geometry h={} w={} k={}",
        header.nbits,
        plen,
        plain.job.operands.a.pattern.tile_size(),
        plain.job.operands.b.pattern.tile_size(),
        plain.job.k,
    );

    // 2. PROVE the GPU-found tile into the header-bound ZK certificate.
    let mut prover = Fp16Prover::new();
    let cert = prover
        .prove_from_plain_proof(&header, &plain)
        .expect("prove the GPU-found winning tile");
    println!("proved: Fp16ZkCertificate carries {} proof bytes", cert.proof_bytes.len());

    // 3. VERIFY through the PRODUCTION cache, exactly as the node FFI does.
    let (h, w, k) = cert.tile_geometry();
    let (a_hash, b_hash) = (cert.job.operands.a.hash_id, cert.job.operands.b.hash_id);
    let cache_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/api/fp16/fp16_cache.bin");
    let cache = Fp16VerifierCache::from_bytes(&std::fs::read(&cache_path).expect("read fp16_cache.bin"))
        .expect("parse production cache");
    let mut system = Fp16System::<F, D>::new(h, w, k, a_hash, b_hash);
    let verifier = cache
        .get(system.degree_bits())
        .expect("production cache covers the fixture profile");
    let vd = verifier.circuit();
    let proof = ProofWithPublicInputs::from_bytes(cert.proof_bytes.clone(), &vd.common)
        .expect("deserialize wrapped proof");
    verify_wrapped_proof_with_headers(&mut system, vd, &proof, &header, &cert.job, EASY)
        .expect("GPU tile -> ZK prove -> verify-through-production-cache must ACCEPT");
    println!("ACCEPT: GPU-found tile verified through the production verifier cache (consensus path).");

    // 4. TAMPER: a flipped proof byte must be rejected.
    let mut bad = cert.proof_bytes.clone();
    *bad.last_mut().unwrap() ^= 1;
    let rejected = ProofWithPublicInputs::from_bytes(bad, &vd.common)
        .map_err(|e| e.to_string())
        .and_then(|p| {
            verify_wrapped_proof_with_headers(&mut system, vd, &p, &header, &cert.job, EASY)
                .map_err(|e| e.to_string())
        })
        .is_err();
    assert!(rejected, "a tampered wrapped proof must be rejected");
    println!("REJECT: tampered proof correctly rejected. End-to-end GPU->ZK->consensus verify PASSED.");
}
