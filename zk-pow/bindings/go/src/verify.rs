//! ZK Proof Verification FFI (Security Critical)
//!
//! This module contains ZK proof verification FFI functions.
//! Extra care must be taken when modifying this code as it's critical for security.

use std::os::raw::c_char;
use std::slice;

use anyhow::{ensure, Result};

use crate::common::MAX_ZK_PROOF_SIZE;
use zk_pow::api::fp8::public_params::PublicParams;
use zk_pow::api::primitives::BlockHeader;
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

/// Split `σ̂ ‖ σ_1 ‖ … ‖ σ_{d-1}` (108-byte wire headers) into the proposed header's
/// incomplete projection and the ancestor chain, which the verifier authenticates
/// ([`zk_pow::api::fp8::public_params::JobParams::check_ancestry`]).
fn split_v4_headers(headers: &[u8]) -> Result<(Fp8BlockHeader, Vec<BlockHeader>)> {
    let mut headers = BlockHeader::chain_from_bytes(headers)?;
    ensure!(!headers.is_empty(), "missing the proposed v4 header");
    let proposed = headers.remove(0).incomplete;
    Ok((proposed, headers))
}

/// Shared implementation for fp8 proof verification: validate pointers and sizes,
/// then run the cached proof verifier (which authenticates the certificate's
/// ancestor chain) against the single explicit `verification_nbits` target.
///
/// # Safety
/// Same contract as [`verify_zk_proof_v4`].
unsafe fn verify_zk_proof_v4_inner(
    headers: *const u8,
    headers_len: usize,
    zk_proof: *const CZKProof,
    verification_nbits: u32,
    error_msg_out: *mut c_char,
) -> i32 {
    let result = catch_panic(|| {
        if headers.is_null() || zk_proof.is_null() {
            set_error_msg(error_msg_out, "Null pointer");
            return 2;
        }

        let zk_proof_ref = &*zk_proof;

        if zk_proof_ref.proof_blob.is_null() || zk_proof_ref.proof_blob_len == 0 {
            set_error_msg(error_msg_out, "Null or empty proof blob");
            return 1;
        }
        if zk_proof_ref.proof_blob_len > MAX_ZK_PROOF_SIZE {
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
        let (header, ancestor_chain) = match split_v4_headers(headers) {
            Ok(split) => split,
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
        let verdict = cache.verify_share(&header, &ancestor_chain, public_data, proof_data, verification_nbits);
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

/// Verify an FP8 ZK proof: the published `public_data` / `proof_data` pair carried
/// by `zk_proof` against the caller's expected block header and the single explicit
/// difficulty target `verification_nbits`.
///
/// Consensus passes the header's own `nbits`; pool shares pass the share target —
/// both through this one function, so header-bound and share-target verification
/// can never diverge into separate C paths.
///
/// `headers` contains canonical 108-byte wire headers: the proposed header `σ̂`, then
/// the ancestor chain `σ_1..σ_{d-1}` strictly between it and the statement's
/// complete ancestor header `σ_d` (parent first), so `headers_len` is 108·d for
/// `d = 1..=4`. Each link — ending at `σ_d` — must be the SHA256d of the full parent
/// header, including its proof commitment; extra headers are rejected. The proposed
/// header's own proof commitment is ignored.
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
    verification_nbits: u32,
    error_msg_out: *mut c_char,
) -> i32 {
    verify_zk_proof_v4_inner(headers, headers_len, zk_proof, verification_nbits, error_msg_out)
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
    use sha2::{Digest, Sha256};

    #[test]
    fn full_header_wire_encoding() {
        // Independent wire vector: every byte, including ProofCommitment, is distinct.
        let header: [u8; BlockHeader::SERIALIZED_SIZE] = std::array::from_fn(|i| i as u8);
        let expected_hash = [
            0x35, 0x54, 0xa2, 0x7c, 0xb6, 0x00, 0x80, 0x65, 0xba, 0x72, 0xf6, 0xa9, 0x1f, 0x25, 0xc4, 0xb2, 0x06, 0xa6, 0x82,
            0x95, 0xc0, 0x71, 0xfb, 0x50, 0x71, 0xed, 0xa6, 0x95, 0x68, 0x4b, 0xac, 0xf1,
        ];
        assert_eq!(Sha256::digest(Sha256::digest(header))[..], expected_hash);
        let parsed = BlockHeader::from_bytes(&header).unwrap();
        assert_eq!(parsed.incomplete.version, 0x03020100);
        assert_eq!(parsed.incomplete.prev_block, std::array::from_fn(|i| (35 - i) as u8));
        assert_eq!(parsed.incomplete.merkle_root, std::array::from_fn(|i| (67 - i) as u8));
        assert_eq!(parsed.incomplete.timestamp, 0x47464544);
        assert_eq!(parsed.incomplete.nbits, 0x4b4a4948);
        assert_eq!(parsed.proof_commitment[..], header[76..]);
        assert_eq!(parsed.to_bytes(), header);
        // The block hash is reported in `prev_block` (display) order.
        let mut display_hash = expected_hash;
        display_hash.reverse();
        assert_eq!(parsed.block_hash(), display_hash);
    }

    #[test]
    fn splits_the_proposed_header_from_its_chain() {
        let headers: Vec<u8> = (0..3 * BlockHeader::SERIALIZED_SIZE).map(|i| i as u8).collect();
        let (proposed, chain) = split_v4_headers(&headers).unwrap();
        // The proposed header's own proof commitment is not part of the statement.
        assert_eq!(proposed.to_bytes(), headers[..Fp8BlockHeader::SERIALIZED_SIZE]);
        assert_eq!(
            chain.iter().flat_map(BlockHeader::to_bytes).collect::<Vec<_>>(),
            headers[BlockHeader::SERIALIZED_SIZE..]
        );
        for length in [0, 76, 107, 109, 215, 217] {
            assert!(split_v4_headers(&headers[..length]).is_err(), "length {length}");
        }
    }
}
