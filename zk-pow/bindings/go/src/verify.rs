//! ZK Proof Verification FFI (Security Critical)
//!
//! This module contains ZK proof verification FFI functions.
//! Extra care must be taken when modifying this code as it's critical for security.

use std::os::raw::c_char;
use std::slice;

use crate::common::{MAX_FP8_PROOF_SIZE, MAX_ZK_PROOF_SIZE};
use zk_pow::api::fp8::public_params::PublicParams;
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

/// The FFI block-header type is the frozen v2 clone; the fp8 (v4) API binds
/// the mainline type. The two structs are field-identical.
fn header_to_v4(header: &IncompleteBlockHeader) -> zk_pow::api::primitives::IncompleteBlockHeader {
    zk_pow::api::primitives::IncompleteBlockHeader {
        version: header.version,
        prev_block: header.prev_block,
        merkle_root: header.merkle_root,
        timestamp: header.timestamp,
        nbits: header.nbits,
    }
}

/// Shared implementation for fp8 proof verification — the same shape as
/// [`verify_zk_proof_inner`]: pointer and size validation, then a single call into the
/// fp8 API, which deserializes both blobs (header binding included) and runs the full
/// verification.
///
/// # Safety
/// Same contract as [`verify_zk_proof_v4`].
unsafe fn verify_zk_proof_v4_inner(
    block_header: *const IncompleteBlockHeader,
    zk_proof: *const CZKProof,
    nbits_override: Option<u32>,
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

        // The statement's trusted setup comes from the global read-only cache: the
        // embedded `fp8_cache.bin` preloads the universal (D1) wrapper circuits,
        // which cover every envelope-legal geometry and degree profile. A device
        // missing from a stale cache rejects the proof — setups are never compiled
        // on demand, so no proof can force that cost.
        let cache = fp8_cache();
        let header = header_to_v4(&*block_header);
        // The statement carries its own `ancestor_header` (σ_Δ) inside
        // `public_data`'s 76-byte prefix, so the single `block_header`
        // argument (σ̂) is all the binding the FFI needs. Authenticating
        // σ_Δ against the caller's blockchain context (hash-walking the
        // `prev_block` chain) is the caller's responsibility — see
        // `verify_zk_proof_v4`'s docs.
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
/// by `zk_proof` against the caller's expected block header (`block_header` = σ̂).
///
/// **Caller responsibility — `ancestor_header` (σ_Δ) authentication.** The statement
/// carries its own 76-byte `ancestor_header` (σ_Δ) in the `public_data` prefix; the
/// proof's B side is keyed by it, but the proof does *not* establish that this header
/// is the caller's real ancestor. The caller must hash-walk `prev_block` links from
/// the proposed header's parent and confirm the proof-carried σ_Δ appears in that
/// window before accepting the block.
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
/// - `block_header` must be a valid pointer
/// - `zk_proof` must be a valid pointer; `zk_proof.proof_blob` must be a valid pointer to
///   `proof_blob_len` bytes
/// - `error_msg_out` must be null or a valid pointer to a caller-allocated buffer of `ERROR_MSG_MAX_SIZE` bytes
#[no_mangle]
pub unsafe extern "C" fn verify_zk_proof_v4(
    block_header: *const IncompleteBlockHeader,
    zk_proof: *const CZKProof,
    error_msg_out: *mut c_char,
) -> i32 {
    verify_zk_proof_v4_inner(block_header, zk_proof, None, error_msg_out)
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
    block_header: *const IncompleteBlockHeader,
    zk_proof: *const CZKProof,
    nbits_override: u32,
    error_msg_out: *mut c_char,
) -> i32 {
    verify_zk_proof_v4_inner(block_header, zk_proof, Some(nbits_override), error_msg_out)
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
