//! FP16 (A100) ZK consensus-certificate verification FFI.
//!
//! This is the FP16 analogue of the FP8 v4 wrapped-proof path
//! ([`crate::verify::verify_zk_proof_v4`]): it deserializes an
//! [`Fp16ZkCertificate`] (the public job statement + the constant-size stage-2
//! recursive proof), authenticates the proof-carried ancestor against the
//! supplied header window, header-binds the proof's public inputs, and verifies
//! the wrapped plonky2 proof.
//!
//! It REPLACES the former plaintext FP16 certificate FFI
//! (`verify_fp16_plain_proof_ffi` / `verify_fp16_cert_ffi`). The plaintext
//! certificate carried the opened operand strips (cert size scaled with the
//! tile, up to several MiB); the ZK certificate is a constant-size recursive
//! proof, so the wire-size pressure the plaintext path put on the V5 certificate
//! and block/headers message caps is gone.
//!
//! # PROVISIONING (read before activating V5 on a network)
//!
//! Unlike FP8 — whose verifier uses a preloaded *universal* trusted setup
//! (`fp8_cache.bin`, one circuit covering every envelope-legal geometry) — the
//! FP16 wrapper is compiled **per degree profile** (the universal FP16 wrapper is
//! the documented residual, `docs/fp16_scheme/stark_feasibility.md §8`). To avoid
//! an attacker-chosen geometry forcing a multi-minute circuit build at verify time,
//! this entry NEVER compiles: it loads the pre-compiled wrapper for the proof's
//! degree profile from the embedded [`FP16_VERIFIER_CACHE`] (the per-profile
//! analogue of FP8's universal per-device cache) and rejects any profile the cache
//! does not contain. Verification is therefore constant-cost and geometry-bounded;
//! the former verify-time-rebuild DoS is closed.
//!
//! The one remaining provisioning step before activation is the cache CONTENTS:
//! the embedded `fp16_cache.bin` must enumerate every consensus-legal degree
//! profile (built offline by `build_cache … src/api/fp16/fp16_cache.bin`; the
//! sample/bootstrap blob covers only the smallest profile and is for building and
//! exercising the load path, not for consensus). A legal geometry whose profile is
//! absent from the embedded cache is rejected — fail-closed, but it would reject
//! honest work, so the full cache must be embedded before V5 is allowed.

use std::os::raw::c_char;
use std::slice;

use anyhow::{ensure, Result};
use lazy_static::lazy_static;
use plonky2::plonk::proof::ProofWithPublicInputs;
use sha2::{Digest, Sha256};

use zk_pow::v5::api::embedded_cache;
use zk_pow::v5::api::zk_cert::Fp16ZkCertificate;
use zk_pow::v4::api::primitives::IncompleteBlockHeader as Fp16BlockHeader;
use zk_pow::v5::circuit::driver::Fp16System;
use zk_pow::v5::circuit::verifier_cache::Fp16VerifierCache;
use zk_pow::v5::circuit::wrapper::{verify_wrapped_proof_with_headers, D, F};

lazy_static! {
    /// The embedded FP16 wrapper verifier cache (one compiled stage-2 verifier per
    /// reachable degree profile). Loaded once; verification looks up the proof's
    /// geometry profile and NEVER compiles a circuit, so an attacker-chosen geometry
    /// cannot force a multi-minute setup build. An empty embedded blob (feature off /
    /// cache not yet built) loads as an empty cache and every verify then rejects.
    static ref FP16_VERIFIER_CACHE: Fp16VerifierCache =
        Fp16VerifierCache::from_bytes(embedded_cache::CACHE_DATA)
            .expect("the embedded fp16 verifier cache must decode");
}

use crate::common::{catch_panic, set_error_msg};

/// Canonical full block-header wire length (incomplete 76-byte projection plus
/// the 32-byte proof commitment), matching the FP8 v4 ancestry codec.
const FULL_BLOCK_HEADER_SIZE: usize = 108;

/// Maximum accepted FP16 ZK certificate byte length. The stage-2 wrapped proof is
/// constant-size (~74 KiB) and the job statement is small. The ceiling is bounded
/// by the p2p relay budget: a `headers` message carries up to `MaxBlockHeadersPerMsg`
/// (100) headers each budgeting a certificate, and must fit `MaxProtocolMessageLength`
/// (8 MB), so a V5 certificate cannot exceed ~79.6 KiB. This value MUST equal the
/// Go `MaxFp16ZkCertProofSize` (node/wire/certificate.go). `Fp16ZkCertificate::from_bytes`
/// has no internal limit, so an untrusted blob is capped here (denial-of-service guard).
const MAX_FP16_ZK_CERT_SIZE: usize = 79_000;

/// Authenticate the supplied full headers against the proof-carried ancestor,
/// returning the proposed header's incomplete projection.
///
/// `headers` is the concatenation of canonical 108-byte wire headers in
/// proposed, parent, grandparent order (length 108, 216, or 324). Each link is
/// authenticated by SHA256d of the full header (including its proof commitment)
/// equalling the next header's `prev_block`. The proof-carried ancestor
/// ([`Fp16ZkCertificate`]'s `job.ancestor_header`, which keys the B-side
/// commitment + noise seeds) must equal the incomplete projection of one of these
/// headers — i.e. it lies within the depth-`D` (`D <= 2`) state window. This
/// mirrors the FP8 v4 `check_certificate_ancestors` gate.
fn check_fp16_certificate_ancestors(headers: &[u8], ancestor: &Fp16BlockHeader) -> Result<Fp16BlockHeader> {
    ensure!(
        matches!(headers.len(), 108 | 216 | 324),
        "invalid fp16 headers length {}",
        headers.len()
    );
    let (headers, _) = headers.as_chunks::<FULL_BLOCK_HEADER_SIZE>();
    let (proposed_bytes, ancestors) = headers.split_first().unwrap();
    let proposed = Fp16BlockHeader::from_bytes(&proposed_bytes[..Fp16BlockHeader::SERIALIZED_SIZE])?;
    let mut matched = ancestor == &proposed;
    let mut prev_hash = &proposed_bytes[4..36];
    for (i, header) in ancestors.iter().enumerate() {
        let hash = Sha256::digest(Sha256::digest(header));
        ensure!(
            hash[..] == prev_hash[..],
            "fp16 ancestor header at depth {} does not connect",
            i + 1
        );
        let parsed = Fp16BlockHeader::from_bytes(&header[..Fp16BlockHeader::SERIALIZED_SIZE])?;
        matched |= ancestor == &parsed;
        prev_hash = &header[4..36];
    }
    ensure!(
        matched,
        "fp16 proof-carried ancestor is not the proposed header, its parent, or its grandparent"
    );
    Ok(proposed)
}

/// Verify an FP16 (A100) ZK consensus certificate: authenticate the proof-carried
/// ancestor against the supplied header window, then header-bind and verify the
/// wrapped recursive proof against the proposed header.
///
/// `headers` is proposed ‖ parent ‖ grandparent in canonical 108-byte wire form
/// (`headers_len` must be 108, 216, or 324). `cert_bytes` is a serialized
/// [`Fp16ZkCertificate`] (`{ job, proof_bytes }`). The job's ancestor header keys
/// the B-side commitment + noise seeds, which [`check_fp16_certificate_ancestors`]
/// confirms is a member of the state window before verification.
///
/// `nbits_override`: `0` uses the proposed header's own `nbits` (a full-block
/// check); a non-zero value is a pool-share target.
///
/// # Returns
/// - 0: certificate verified and accepted
/// - 1: certificate rejected (malformed, oversized, out-of-window ancestor, bad
///   proof, inadmissible tile, or unmet difficulty)
/// - 2: system error (null/empty input or internal panic)
///
/// # Safety
/// - `headers` must point to `headers_len` readable bytes
/// - `cert_bytes` must point to `cert_len` readable bytes
/// - `error_msg_out` must be null or a valid pointer to a caller-allocated buffer
///   of `ERROR_MSG_MAX_SIZE` bytes
#[no_mangle]
pub unsafe extern "C" fn verify_fp16_zk_cert_ffi(
    headers: *const u8,
    headers_len: usize,
    cert_bytes: *const u8,
    cert_len: usize,
    nbits_override: u32,
    error_msg_out: *mut c_char,
) -> i32 {
    if headers.is_null() || cert_bytes.is_null() || cert_len == 0 {
        set_error_msg(error_msg_out, "Null/empty input");
        return 2;
    }
    if !matches!(headers_len, 108 | 216 | 324) {
        set_error_msg(error_msg_out, &format!("invalid fp16 headers length {}", headers_len));
        return 1;
    }
    if cert_len > MAX_FP16_ZK_CERT_SIZE {
        set_error_msg(error_msg_out, "FP16 ZK certificate too large");
        return 1;
    }

    let headers = slice::from_raw_parts(headers, headers_len);
    let bytes = slice::from_raw_parts(cert_bytes, cert_len);

    let result = catch_panic(|| {
        let cert = match Fp16ZkCertificate::from_bytes(bytes) {
            Ok(c) => c,
            Err(e) => return (1, format!("deserialize: {e}")),
        };
        let proposed = match check_fp16_certificate_ancestors(headers, &cert.job.ancestor_header) {
            Ok(h) => h,
            Err(e) => return (1, format!("rejected: {e}")),
        };
        let nbits = if nbits_override == 0 { proposed.nbits } else { nbits_override };
        match verify_fp16_zk_cert(&cert, &proposed, nbits) {
            Ok(()) => (0, "accepted".to_string()),
            Err(e) => (1, format!("rejected: {e}")),
        }
    });

    match result {
        Ok((code, msg)) => {
            set_error_msg(error_msg_out, &msg);
            code
        }
        Err(panic_msg) => {
            set_error_msg(error_msg_out, &format!("panic: {panic_msg}"));
            2
        }
    }
}

/// Load the pre-compiled verifier setup for the proof's geometry and run the
/// header-bound wrapped-proof verifier. Verification cost is constant and
/// geometry-INDEPENDENT: `Fp16System::new` is cheap (closed-form snapped degree
/// bits), and the expensive wrapper circuit is LOADED from the embedded cache by
/// its degree profile — never compiled on the verify path. An attacker-chosen
/// geometry therefore cannot force a circuit build: a legal geometry hits a cached
/// profile; a geometry whose profile is absent (out of envelope, or a stale cache)
/// is rejected before any circuit work. This closes the former verify-time-rebuild
/// DoS. The cache is a consensus / trusted-setup artifact, keyed by degree profile
/// (the per-geometry analogue of FP8's universal per-device cache).
fn verify_fp16_zk_cert(cert: &Fp16ZkCertificate, proposed: &Fp16BlockHeader, nbits: u32) -> Result<()> {
    let (h, w, k) = cert.tile_geometry();
    let a_hash = cert.job.operands.a.hash_id;
    let b_hash = cert.job.operands.b.hash_id;

    // Reject any geometry outside the consensus / wrapper-legal envelope BEFORE
    // constructing the system (whose ladder height-snap would otherwise panic on an
    // oversize tile). This is a clean fail-closed rejection, not a system error, and
    // mirrors the api-level envelope the miner and plaintext oracle enforce.
    zk_pow::v5::api::params::Fp16Params {
        device: cert.job.device,
        h,
        w,
        k,
        r: cert.job.r as usize,
    }
    .validate()
    .map_err(|e| anyhow::anyhow!("fp16 geometry rejected: {e}"))?;

    // Cheap: AIR definitions + the closed-form, ladder-snapped degree profile that
    // keys the cache. No trace generation, no circuit compilation. (The verifier
    // folds the job's `p` into it; hence `mut`.)
    let mut system = Fp16System::<F, D>::new(h, w, k, a_hash, b_hash);
    let verifier = FP16_VERIFIER_CACHE.get(system.degree_bits()).ok_or_else(|| {
        anyhow::anyhow!(
            "no cached fp16 verifier setup for degree profile {:?} (geometry h={h} w={w} k={k}); \
             the geometry is out of the compiled envelope or fp16_cache.bin is stale \
             (regenerate with build_cache)",
            system.degree_bits()
        )
    })?;
    let verifier_data = verifier.circuit();

    // Deserialize the stage-2 proof against the cached wrapper common data.
    let proof = ProofWithPublicInputs::from_bytes(cert.proof_bytes.clone(), &verifier_data.common)
        .map_err(|e| anyhow::anyhow!("deserialize wrapped proof: {e}"))?;

    // Header-bound verification: pins KEY_A/KEY_B/jackpot-key + noise seeds to the
    // header, derives the statement digest from the proven HASH_JACKPOT, pins the
    // whole PI vector, verifies the ZK proof, and checks difficulty vs `nbits`.
    verify_wrapped_proof_with_headers(&mut system, verifier_data, &proof, proposed, &cert.job, nbits)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 108-byte canonical wire header: the fixed 76-byte incomplete projection
    /// plus a 32-byte proof commitment.
    fn full_header(proj76: &[u8; 76], commitment: [u8; 32]) -> Vec<u8> {
        let mut full = proj76.to_vec();
        full.extend_from_slice(&commitment);
        full
    }

    fn proj76() -> [u8; 76] {
        // Asymmetric, so any header-orientation regression on the FFI seam shows.
        let mut p = [0u8; 76];
        for (i, b) in p.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(7).wrapping_add(3);
        }
        p
    }

    /// Hash-walk unit check (does not need a wrapped proof): a connected depth-1
    /// chain authenticates a parent-as-ancestor, a broken link is rejected, and an
    /// ancestor outside the window is rejected.
    #[test]
    fn fp16_ancestor_hash_walk_connects() {
        let mut parent_proj = proj76();
        parent_proj[68] ^= 0x5A; // perturb the parent's timestamp vs proposed
        let parent_full = full_header(&parent_proj, [7u8; 32]);
        let parent_incomplete = Fp16BlockHeader::from_bytes(&parent_full[..76]).unwrap();

        // Link the proposed header to the parent: prev_block = SHA256d(parent).
        let link = Sha256::digest(Sha256::digest(&parent_full));
        let mut proposed_proj = proj76();
        proposed_proj[4..36].copy_from_slice(&link);
        let proposed_full = full_header(&proposed_proj, [0u8; 32]);
        let mut chain = proposed_full.clone();
        chain.extend_from_slice(&parent_full);

        assert!(check_fp16_certificate_ancestors(&chain, &parent_incomplete).is_ok());

        let mut broken = chain.clone();
        broken[4] ^= 0x01;
        assert!(check_fp16_certificate_ancestors(&broken, &parent_incomplete).is_err());

        let stranger = Fp16BlockHeader { version: 0xDEAD_BEEF, ..parent_incomplete };
        assert!(check_fp16_certificate_ancestors(&chain, &stranger).is_err());
    }

    /// An invalid header-chain length is rejected (must be 108/216/324).
    #[test]
    fn fp16_zk_cert_rejects_bad_headers_len() {
        let cert = [0xABu8; 64];
        let headers = vec![0u8; 100];
        let mut err = [0 as c_char; crate::common::ERROR_MSG_MAX_SIZE];
        let code = unsafe {
            verify_fp16_zk_cert_ffi(headers.as_ptr(), headers.len(), cert.as_ptr(), cert.len(), 0, err.as_mut_ptr())
        };
        assert_eq!(code, 1, "bad headers length must be rejected");
    }

    /// Garbage certificate bytes deserialize-fail and are rejected with code 1.
    #[test]
    fn fp16_zk_cert_rejects_garbage() {
        let garbage = [0xABu8; 64];
        let headers = full_header(&proj76(), [0u8; 32]);
        let mut err = [0 as c_char; crate::common::ERROR_MSG_MAX_SIZE];
        let code = unsafe {
            verify_fp16_zk_cert_ffi(headers.as_ptr(), headers.len(), garbage.as_ptr(), garbage.len(), 0, err.as_mut_ptr())
        };
        assert_eq!(code, 1, "garbage cert must be rejected");
    }

    /// Null / empty input is a bad-input system error (code 2).
    #[test]
    fn fp16_zk_cert_rejects_null_and_empty() {
        let cert = [0xABu8; 64];
        let headers = full_header(&proj76(), [0u8; 32]);
        let mut err = [0 as c_char; crate::common::ERROR_MSG_MAX_SIZE];
        let code = unsafe {
            verify_fp16_zk_cert_ffi(std::ptr::null(), 108, cert.as_ptr(), cert.len(), 0, err.as_mut_ptr())
        };
        assert_eq!(code, 2, "null headers must be bad input");
        let code = unsafe {
            verify_fp16_zk_cert_ffi(headers.as_ptr(), headers.len(), cert.as_ptr(), 0, 0, err.as_mut_ptr())
        };
        assert_eq!(code, 2, "empty cert must be bad input");
    }

    // NOTE: an end-to-end accept test needs a real wrapped stage-2 proof, which
    // costs a ~12-minute recursive wrap (and the per-geometry circuit build). It
    // is therefore not a committed unit test here; regenerate the Go fixture (a
    // serialized `Fp16ZkCertificate`) with a dedicated, explicitly-run tool once
    // the FP16 verifier cache / universal wrapper (see the module-level blocker)
    // lands, mirroring the FP8 `fp8_zk_proof_b200.bin` fixture flow.
}
