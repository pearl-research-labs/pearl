//! ZK Proof Verification FFI (Security Critical)
//!
//! This module contains ZK proof verification FFI functions.
//! Extra care must be taken when modifying this code as it's critical for security.

use std::os::raw::c_char;
use std::slice;

use anyhow::{ensure, Result};
use sha2::{Digest, Sha256};

use crate::common::{MAX_FP8_PROOF_SIZE, MAX_ZK_PROOF_SIZE};
use zk_pow::api::fp8::public_params::PublicParams;
use zk_pow::api::primitives::IncompleteBlockHeader as Fp8BlockHeader;
use zk_pow::api::seed::SeedDerivation;
use zk_pow::v2::api::proof::{IncompleteBlockHeader, PublicProofParams, ZKProof};
use zk_pow::v2::api::sanity_checks;
use zk_pow::v2::api::verify;

use crate::common::{acquire_cache, catch_panic, fp8_cache, set_error_msg, CZKProof};

// ============================================================================
// ZK Proof Verification FFI
// ============================================================================

/// Shared implementation for ZK proof verification.
///
/// # Safety
/// - All pointers must be valid
/// - `zk_proof.proof_blob` must be a valid pointer to `proof_blob_len` bytes
/// - `error_msg_out` must be null or a valid pointer to a caller-allocated buffer of `ERROR_MSG_MAX_SIZE` bytes
unsafe fn verify_zk_proof_inner(
    block_header: *const IncompleteBlockHeader,
    zk_proof: *const CZKProof,
    nbits_override: Option<u32>,
    seed_derivation: SeedDerivation,
    error_msg_out: *mut c_char,
) -> i32 {
    // Wrap in catch_unwind to prevent panics from crossing FFI boundary
    let result = catch_panic(|| {
        // Validate input pointers
        if block_header.is_null() || zk_proof.is_null() {
            set_error_msg(error_msg_out, "Null pointer");
            return 2;
        }

        let zk_proof_ref = &*zk_proof;

        if zk_proof_ref.proof_blob.is_null() || zk_proof_ref.proof_blob_len == 0 {
            set_error_msg(error_msg_out, "Null or empty proof blob");
            return 1;
        }
        if zk_proof_ref.proof_blob_len > MAX_ZK_PROOF_SIZE {
            set_error_msg(error_msg_out, "ZK Proof too large");
            return 1;
        }

        if !PublicProofParams::is_valid_wire_size(zk_proof_ref.public_data_len) {
            set_error_msg(
                error_msg_out,
                &format!("invalid public_data_len {}", zk_proof_ref.public_data_len),
            );
            return 1;
        }

        let plonky2_proof = slice::from_raw_parts(zk_proof_ref.proof_blob, zk_proof_ref.proof_blob_len);
        let public_data = &zk_proof_ref.public_data[..zk_proof_ref.public_data_len];
        let (params, zk_proof) = match ZKProof::deserialize(*block_header, seed_derivation, public_data, plonky2_proof) {
            Ok(r) => r,
            Err(e) => {
                set_error_msg(error_msg_out, &format!("{}", e));
                return 1;
            }
        };

        // Acquire circuit cache (immutable - verifier doesn't modify cache)
        let cache = acquire_cache();

        // Verify using cached circuits only (no compilation)
        match verify::verify_block_cached_circuits_only(&params, &zk_proof, &cache, nbits_override) {
            Ok(_) => {
                set_error_msg(error_msg_out, "Proof verified successfully");
                0
            }
            Err(e) => {
                set_error_msg(error_msg_out, &format!("{}", e));
                1
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(panic_msg) => {
            set_error_msg(error_msg_out, &format!("Internal panic: {}", panic_msg));
            2
        }
    }
}

/// Verify a ZK proof against public parameters.
///
/// # Security Considerations
/// - This is the primary entry point for verifying ZK proofs
/// - Validates all input parameters before verification
/// - Uses panic handling to prevent crashes from poisoning global state
/// - All validation errors return specific codes and messages
///
/// # Returns
/// - 0: Proof verified and accepted
/// - 1: Proof verified but rejected (proof is invalid)
/// - 2: System error (could not run verification)
///
/// # Safety
/// - All pointers must be valid
/// - `zk_proof.proof_blob` must be a valid pointer to `proof_blob_len` bytes
/// - `error_msg_out` must be null or a valid pointer to a caller-allocated buffer of `ERROR_MSG_MAX_SIZE` bytes
///
/// Verify a ZK proof. Panics are caught to prevent undefined behavior at FFI boundary.
/// Returns: 0 = success, 1 = proof rejected, 2 = system error.
#[no_mangle]
pub unsafe extern "C" fn verify_zk_proof_v2(
    block_header: *const IncompleteBlockHeader,
    zk_proof: *const CZKProof,
    error_msg_out: *mut c_char,
) -> i32 {
    verify_zk_proof_inner(block_header, zk_proof, None, SeedDerivation::Legacy, error_msg_out)
}

/// Verify a ZK proof against public parameters, overriding the difficulty with the given nbits.
///
/// Identical to `verify_zk_proof_v2` except the difficulty target is derived from `nbits_override`
/// instead of the block header's nbits field.
///
/// # Returns
/// - 0: Proof verified and accepted
/// - 1: Proof verified but rejected (proof is invalid)
/// - 2: System error (could not run verification)
///
/// # Safety
/// - All pointers must be valid
/// - `zk_proof.proof_blob` must be a valid pointer to `proof_blob_len` bytes
/// - `error_msg_out` must be null or a valid pointer to a caller-allocated buffer of `ERROR_MSG_MAX_SIZE` bytes
#[no_mangle]
pub unsafe extern "C" fn verify_zk_proof_v2_with_nbits(
    block_header: *const IncompleteBlockHeader,
    zk_proof: *const CZKProof,
    nbits_override: u32,
    error_msg_out: *mut c_char,
) -> i32 {
    verify_zk_proof_inner(
        block_header,
        zk_proof,
        Some(nbits_override),
        SeedDerivation::Legacy,
        error_msg_out,
    )
}

/// `verify_zk_proof_v2` with the salted (V3) noise-seed derivation.
#[no_mangle]
pub unsafe extern "C" fn verify_zk_proof_v3(
    block_header: *const IncompleteBlockHeader,
    zk_proof: *const CZKProof,
    error_msg_out: *mut c_char,
) -> i32 {
    verify_zk_proof_inner(block_header, zk_proof, None, SeedDerivation::Salted, error_msg_out)
}

/// `verify_zk_proof_v3` with the difficulty from `nbits_override`.
#[no_mangle]
pub unsafe extern "C" fn verify_zk_proof_v3_with_nbits(
    block_header: *const IncompleteBlockHeader,
    zk_proof: *const CZKProof,
    nbits_override: u32,
    error_msg_out: *mut c_char,
) -> i32 {
    verify_zk_proof_inner(
        block_header,
        zk_proof,
        Some(nbits_override),
        SeedDerivation::Salted,
        error_msg_out,
    )
}

/// Check wire `public_data` against the rank-penalty rule.
///
/// # Returns
/// - 0: rule satisfied
/// - 1: rule violated (the block must be rejected)
/// - 2: System error (could not run the check)
///
/// # Safety
/// - `public_data` must be a valid pointer to `public_data_len` bytes
/// - `error_msg_out` must be null or a valid pointer to a caller-allocated buffer of `ERROR_MSG_MAX_SIZE` bytes
#[no_mangle]
pub unsafe extern "C" fn check_rank_penalty(
    nbits: u32,
    public_data: *const u8,
    public_data_len: usize,
    error_msg_out: *mut c_char,
) -> i32 {
    let result = catch_panic(|| {
        if public_data.is_null() {
            set_error_msg(error_msg_out, "Null pointer");
            return 2;
        }
        if !PublicProofParams::is_valid_wire_size(public_data_len) {
            set_error_msg(error_msg_out, &format!("invalid public_data_len {}", public_data_len));
            return 1;
        }

        let public_data = slice::from_raw_parts(public_data, public_data_len);
        let (mining_config, hash_jackpot) = match PublicProofParams::mining_config_and_jackpot_from_wire_bytes(public_data) {
            Ok(fields) => fields,
            Err(e) => {
                set_error_msg(error_msg_out, &format!("{}", e));
                return 1;
            }
        };

        match sanity_checks::check_rank_penalty(&mining_config, &hash_jackpot, nbits) {
            Ok(()) => {
                set_error_msg(error_msg_out, "Rank penalty rule satisfied");
                0
            }
            Err(e) => {
                set_error_msg(error_msg_out, &format!("{}", e));
                1
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(panic_msg) => {
            set_error_msg(error_msg_out, &format!("Internal panic: {}", panic_msg));
            2
        }
    }
}

// ============================================================================
// FP8 ZK Proof Verification FFI
// ============================================================================

const FULL_BLOCK_HEADER_SIZE: usize = 108;

/// Authenticate all supplied full headers, then locate the proof-carried ancestor.
/// Hash raw wire bytes, including ProofCommitment; parsed hash fields use display order.
fn check_certificate_ancestors(headers: &[u8], public_data: &[u8]) -> Result<Fp8BlockHeader> {
    ensure!(
        matches!(headers.len(), 108 | 216 | 324),
        "invalid v4 headers length {}",
        headers.len()
    );
    let statement = PublicParams::from_bytes(public_data)?;
    let mut headers = headers.chunks_exact(FULL_BLOCK_HEADER_SIZE);
    let proposed_bytes = headers.next().unwrap();
    let proposed = Fp8BlockHeader::from_bytes(&proposed_bytes[..Fp8BlockHeader::SERIALIZED_SIZE])?;
    let mut matched = statement.ancestor_header() == &proposed;
    let mut prev_hash = &proposed_bytes[4..36];
    for (i, header) in headers.enumerate() {
        let hash = Sha256::digest(Sha256::digest(header));
        ensure!(
            hash[..] == prev_hash[..],
            "v4 ancestor header at depth {} does not connect",
            i + 1
        );
        let ancestor = Fp8BlockHeader::from_bytes(&header[..Fp8BlockHeader::SERIALIZED_SIZE])?;
        matched |= statement.ancestor_header() == &ancestor;
        prev_hash = &header[4..36];
    }
    ensure!(
        matched,
        "v4 ancestor header is not the proposed header, its parent, or its grandparent"
    );
    Ok(proposed)
}

/// Shared implementation for fp8 proof verification: validate pointers and sizes,
/// authenticate the certificate's ancestor, then run the cached proof verifier.
///
/// # Safety
/// Same contract as [`verify_zk_proof_v4`].
unsafe fn verify_zk_proof_v4_inner(
    headers: *const u8,
    headers_len: usize,
    zk_proof: *const CZKProof,
    nbits_override: Option<u32>,
    error_msg_out: *mut c_char,
) -> i32 {
    let result = catch_panic(|| {
        if headers.is_null() || zk_proof.is_null() {
            set_error_msg(error_msg_out, "Null pointer");
            return 2;
        }
        if !matches!(headers_len, 108 | 216 | 324) {
            set_error_msg(error_msg_out, &format!("invalid v4 headers length {}", headers_len));
            return 1;
        }

        let zk_proof_ref = &*zk_proof;

        if zk_proof_ref.proof_blob.is_null() || zk_proof_ref.proof_blob_len == 0 {
            set_error_msg(error_msg_out, "Null or empty proof blob");
            return 1;
        }
        if zk_proof_ref.proof_blob_len > MAX_FP8_PROOF_SIZE {
            set_error_msg(error_msg_out, "FP8 proof too large");
            return 1;
        }
        if !PublicParams::is_valid_wire_size(zk_proof_ref.public_data_len) {
            set_error_msg(
                error_msg_out,
                &format!("invalid public_data_len {}", zk_proof_ref.public_data_len),
            );
            return 1;
        }

        let proof_data = slice::from_raw_parts(zk_proof_ref.proof_blob, zk_proof_ref.proof_blob_len);
        let public_data = &zk_proof_ref.public_data[..zk_proof_ref.public_data_len];
        let headers = slice::from_raw_parts(headers, headers_len);
        let header = match check_certificate_ancestors(headers, public_data) {
            Ok(header) => header,
            Err(error) => {
                set_error_msg(error_msg_out, &error.to_string());
                return 1;
            }
        };

        // The statement's trusted setup comes from the global read-only cache: the
        // embedded `fp8_cache.bin` preloads the universal (D1) wrapper circuits,
        // which cover every envelope-legal geometry and degree profile. A device
        // missing from a stale cache rejects the proof — setups are never compiled
        // on demand, so no proof can force that cost.
        let cache = fp8_cache();
        let verdict = match nbits_override {
            None => cache.verify_block(&header, public_data, proof_data),
            Some(nbits) => cache.verify_share(&header, public_data, proof_data, nbits),
        };
        match verdict {
            Ok(()) => {
                set_error_msg(error_msg_out, "Proof verified successfully");
                0
            }
            Err(e) => {
                set_error_msg(error_msg_out, &format!("{}", e));
                1
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(panic_msg) => {
            set_error_msg(error_msg_out, &format!("Internal panic: {}", panic_msg));
            2
        }
    }
}

/// Verify an FP8 ZK block proof: the published `public_data` / `proof_data` pair carried
/// by `zk_proof` against the caller's expected block header.
///
/// `headers` contains canonical 108-byte wire headers in proposed, parent, grandparent
/// order. The proposed header is required; zero to two ancestors may follow, so
/// `headers_len` must be 108, 216, or 324. Every supplied link is authenticated by
/// SHA256d of the full header, including its proof commitment. The statement's
/// ancestor must match the incomplete projection of one of these headers.
///
/// The trusted verifier setup is not an argument: setups live in a global read-only
/// cache preloaded from the embedded `fp8_cache.bin`, resolved by the statement's
/// device byte. A device outside the cache rejects the proof (code 1) — setups are
/// never compiled on demand, so no proof can force an expensive circuit build. The
/// verdict is binary: the jackpot policy accepts or rejects, nothing else is reported.
///
/// # Returns
/// - 0: Proof verified and accepted
/// - 1: Proof rejected (malformed, oversized, wrong header, or verification failure)
/// - 2: System error (null pointers or internal panic)
///
/// # Safety
/// - `headers` must point to `headers_len` readable bytes
/// - `zk_proof` must be a valid pointer; `zk_proof.proof_blob` must be a valid pointer to
///   `proof_blob_len` bytes
/// - `error_msg_out` must be null or a valid pointer to a caller-allocated buffer of `ERROR_MSG_MAX_SIZE` bytes
#[no_mangle]
pub unsafe extern "C" fn verify_zk_proof_v4(
    headers: *const u8,
    headers_len: usize,
    zk_proof: *const CZKProof,
    error_msg_out: *mut c_char,
) -> i32 {
    verify_zk_proof_v4_inner(headers, headers_len, zk_proof, None, error_msg_out)
}

/// Verify an FP8 ZK share proof: identical to [`verify_zk_proof_v4`] except the
/// difficulty target is derived from `nbits_override` (e.g. a pool share target) instead of
/// the block header's own nbits field.
///
/// # Returns
/// Same as [`verify_zk_proof_v4`].
///
/// # Safety
/// Same as [`verify_zk_proof_v4`].
#[no_mangle]
pub unsafe extern "C" fn verify_zk_proof_v4_with_nbits(
    headers: *const u8,
    headers_len: usize,
    zk_proof: *const CZKProof,
    nbits_override: u32,
    error_msg_out: *mut c_char,
) -> i32 {
    verify_zk_proof_v4_inner(headers, headers_len, zk_proof, Some(nbits_override), error_msg_out)
}

/// Verify a V1 (version 1, master-format) ZK proof.
/// Uses the V1 circuit cache which contains master-compatible verifier circuits.
///
/// # Returns
/// - 0: Proof verified and accepted
/// - 1: Proof verified but rejected
/// - 2: System error
///
/// # Safety
/// Same requirements as `verify_zk_proof_v2`.
#[no_mangle]
pub unsafe extern "C" fn verify_zk_proof_v1(
    block_header: *const IncompleteBlockHeader,
    zk_proof: *const CZKProof,
    error_msg_out: *mut c_char,
) -> i32 {
    let result = catch_panic(|| {
        if block_header.is_null() || zk_proof.is_null() {
            set_error_msg(error_msg_out, "Null pointer");
            return 2;
        }

        let zk_proof_ref = &*zk_proof;

        if zk_proof_ref.proof_blob.is_null() || zk_proof_ref.proof_blob_len == 0 {
            set_error_msg(error_msg_out, "Null or empty proof blob");
            return 1;
        }
        if zk_proof_ref.proof_blob_len > MAX_ZK_PROOF_SIZE {
            set_error_msg(error_msg_out, "ZK Proof too large");
            return 1;
        }

        let expected_len = zk_pow::v1::api::proof::PublicProofParams::PUBLICDATA_SIZE;
        if zk_proof_ref.public_data_len != expected_len {
            set_error_msg(
                error_msg_out,
                &format!(
                    "v1 proof requires {} byte public_data, got {}",
                    expected_len, zk_proof_ref.public_data_len
                ),
            );
            return 1;
        }

        let block_header_ref = &*block_header;
        let block_header_bytes = block_header_ref.to_bytes();
        let public_data = &zk_proof_ref.public_data[..zk_proof_ref.public_data_len];
        let proof_data = slice::from_raw_parts(zk_proof_ref.proof_blob, zk_proof_ref.proof_blob_len);

        let cache = crate::common::acquire_v1_cache();

        match zk_pow::v1::verify_v1(&block_header_bytes, public_data, proof_data, &cache, None) {
            Ok(_) => {
                set_error_msg(error_msg_out, "V1 proof verified successfully");
                0
            }
            Err(e) => {
                set_error_msg(error_msg_out, &format!("{}", e));
                1
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(panic_msg) => {
            set_error_msg(error_msg_out, &format!("Internal panic: {}", panic_msg));
            2
        }
    }
}

#[cfg(test)]
mod fp8_ancestor_tests {
    use super::*;

    fn linked_headers() -> [[u8; FULL_BLOCK_HEADER_SIZE]; 3] {
        let grandparent = std::array::from_fn(|i| i as u8);
        let mut parent = std::array::from_fn(|i| (i as u8).wrapping_add(113));
        parent[4..36].copy_from_slice(&Sha256::digest(Sha256::digest(grandparent)));
        let mut proposed = std::array::from_fn(|i| (i as u8).wrapping_add(211));
        proposed[4..36].copy_from_slice(&Sha256::digest(Sha256::digest(parent)));
        [proposed, parent, grandparent]
    }

    // Reuse only the unchanged fixture's canonical statement, replacing its ancestor.
    fn public_data(ancestor: &[u8; FULL_BLOCK_HEADER_SIZE]) -> Vec<u8> {
        let fixture = include_bytes!("../../../../node/zkpow/testdata/fp8_zk_proof_b200.bin");
        let length = u32::from_le_bytes(fixture[76..80].try_into().unwrap()) as usize;
        let mut data = fixture[80..80 + length].to_vec();
        data[..76].copy_from_slice(&ancestor[..76]);
        data
    }

    #[test]
    fn full_header_wire_encoding() {
        // Independent wire vector: every byte, including ProofCommitment, is distinct.
        let header: [u8; FULL_BLOCK_HEADER_SIZE] = std::array::from_fn(|i| i as u8);
        let expected_hash = [
            0x35, 0x54, 0xa2, 0x7c, 0xb6, 0x00, 0x80, 0x65, 0xba, 0x72, 0xf6, 0xa9, 0x1f, 0x25, 0xc4, 0xb2, 0x06, 0xa6, 0x82,
            0x95, 0xc0, 0x71, 0xfb, 0x50, 0x71, 0xed, 0xa6, 0x95, 0x68, 0x4b, 0xac, 0xf1,
        ];
        assert_eq!(Sha256::digest(Sha256::digest(header))[..], expected_hash);
        let parsed = Fp8BlockHeader::from_bytes(&header[..76]).unwrap();
        assert_eq!(parsed.version, 0x03020100);
        assert_eq!(parsed.prev_block, std::array::from_fn(|i| (35 - i) as u8));
        assert_eq!(parsed.merkle_root, std::array::from_fn(|i| (67 - i) as u8));
        assert_eq!(parsed.timestamp, 0x47464544);
        assert_eq!(parsed.nbits, 0x4b4a4948);
        assert_eq!(parsed.to_bytes(), header[..76]);
    }

    #[test]
    fn accepts_ancestors_at_each_depth() {
        let headers = linked_headers();
        let proposed = Fp8BlockHeader::from_bytes(&headers[0][..76]).unwrap();
        for depth in 0..=2 {
            let public = public_data(&headers[depth]);
            for supplied_depth in depth..=2 {
                assert_eq!(
                    check_certificate_ancestors(&headers[..=supplied_depth].concat(), &public).unwrap(),
                    proposed,
                    "ancestor depth {depth}, supplied depth {supplied_depth}"
                );
            }
        }

        // The proposed header's commitment is excluded from the proof statement.
        let mut changed_proposed = headers;
        changed_proposed[0][76] ^= 1;
        assert_eq!(
            check_certificate_ancestors(&changed_proposed.concat(), &public_data(&headers[0])).unwrap(),
            proposed
        );
    }

    #[test]
    fn rejects_invalid_ancestors() {
        let headers = linked_headers();
        let mut parent_commitment = headers;
        parent_commitment[1][76] ^= 1;
        let mut grandparent_commitment = headers;
        grandparent_commitment[2][76] ^= 1;
        let mut reversed_prev = public_data(&headers[0]);
        reversed_prev[4..36].reverse();
        let mut reversed_merkle = public_data(&headers[0]);
        reversed_merkle[36..68].reverse();
        let unrelated = [0x55; FULL_BLOCK_HEADER_SIZE];
        for (name, headers, public) in [
            ("outside window", headers.concat(), public_data(&unrelated)),
            ("missing ancestor", headers[..1].concat(), public_data(&headers[1])),
            ("reversed previous hash", headers[..1].concat(), reversed_prev),
            ("reversed merkle root", headers[..1].concat(), reversed_merkle),
        ] {
            let err = check_certificate_ancestors(&headers, &public).unwrap_err();
            assert!(
                err.to_string()
                    .contains("not the proposed header, its parent, or its grandparent"),
                "{name}: {err}"
            );
        }
        for (name, headers, public) in [
            (
                "missing intermediate",
                [headers[0], headers[2]].concat(),
                public_data(&headers[2]),
            ),
            ("parent commitment", parent_commitment.concat(), public_data(&headers[1])),
            (
                "grandparent commitment",
                grandparent_commitment.concat(),
                public_data(&headers[2]),
            ),
            (
                "invalid after depth 0 match",
                [headers[0], unrelated].concat(),
                public_data(&headers[0]),
            ),
            (
                "invalid after depth 1 match",
                [headers[0], headers[1], unrelated].concat(),
                public_data(&headers[1]),
            ),
        ] {
            let err = check_certificate_ancestors(&headers, &public).unwrap_err();
            assert!(err.to_string().contains("does not connect"), "{name}: {err}");
        }
    }

    #[test]
    fn rejects_invalid_framing_and_statements() {
        let headers = linked_headers();
        let public = public_data(&headers[0]);
        for length in [0, 76, 107, 109, 215, 217, 323, 325, 432] {
            let err = check_certificate_ancestors(&vec![0; length], &public).unwrap_err();
            assert!(
                err.to_string().contains("invalid v4 headers length"),
                "length {length}: {err}"
            );
        }
        let headers = headers.concat();
        for length in 0..public.len() {
            assert!(
                check_certificate_ancestors(&headers, &public[..length]).is_err(),
                "public data truncated at {length}"
            );
        }
        let mut trailing = public;
        trailing.push(0);
        assert!(check_certificate_ancestors(&headers, &trailing).is_err());
    }
}
