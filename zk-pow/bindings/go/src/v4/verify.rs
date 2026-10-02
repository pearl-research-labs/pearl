//! FP8 ZK Proof Verification FFI (Security Critical)
//!
//! This module contains the cert v4 (FP8) ZK proof verification FFI function.
//! Extra care must be taken when modifying this code as it's critical for security.

use std::os::raw::c_char;
use std::slice;

use anyhow::{ensure, Result};

use crate::common::MAX_ZK_PROOF_SIZE;
use zk_pow::v4::api::primitives::BlockHeader;
use zk_pow::v4::api::primitives::IncompleteBlockHeader as Fp8BlockHeader;
use zk_pow::v4::api::public_params::{PublicParams, STATE_WINDOW_DEPTH};

use crate::common::{catch_panic, set_error_msg, CZKProof};
use crate::v4::fp8_cache;

// ============================================================================
// FP8 ZK Proof Verification FFI
// ============================================================================

/// Split `σ̂ ‖ σ_1 ‖ … ‖ σ_{d-1}` (108-byte wire headers) into the proposed header's
/// incomplete projection and the ancestor chain, which the verifier authenticates
/// ([`zk_pow::v4::api::public_params::JobParams::check_ancestry`]).
fn split_v4_headers(headers: &[u8]) -> Result<(Fp8BlockHeader, Vec<BlockHeader>)> {
    ensure!(
        valid_v4_headers_len(headers.len()),
        "invalid v4 headers length {}",
        headers.len()
    );
    let (proposed, chain) = headers.split_at(BlockHeader::SERIALIZED_SIZE);
    Ok((
        Fp8BlockHeader::from_bytes(&proposed[..Fp8BlockHeader::SERIALIZED_SIZE])?,
        BlockHeader::chain_from_bytes(chain)?,
    ))
}

fn valid_v4_headers_len(len: usize) -> bool {
    (BlockHeader::SERIALIZED_SIZE..=STATE_WINDOW_DEPTH * BlockHeader::SERIALIZED_SIZE).contains(&len)
        && len.is_multiple_of(BlockHeader::SERIALIZED_SIZE)
}

/// Verify FP8 `public_data` / `proof_data` against the expected proposed header.
/// Pass its `nbits` for consensus or the explicit share target for pool shares.
///
/// `headers` is `σ̂ ‖ σ_1 ‖ … ‖ σ_{d-1}` in canonical 108-byte wire headers,
/// parent first; `headers_len` must be 108·d for d = 1..=4. The statement carries
/// the complete `σ_d`. Every link authenticates the full parent's SHA256d,
/// including its proof commitment; the proposed header's proof commitment is ignored.
/// Setups come from the device-indexed, read-only embedded cache; uncached devices
/// reject, and verification never compiles a setup on demand.
///
/// # Returns
/// 0 = accepted, 1 = rejected (malformed, oversized, wrong header or invalid proof),
/// 2 = system error (null pointers or internal panic).
///
/// # Safety
/// - `headers` must point to `headers_len` readable bytes
/// - `zk_proof` must be valid; its `proof_blob` must point to `proof_blob_len` readable bytes
/// - `error_msg_out` must be null or point to a caller-allocated buffer of `ERROR_MSG_MAX_SIZE` bytes
#[no_mangle]
pub unsafe extern "C" fn verify_zk_proof_v4(
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
        if !valid_v4_headers_len(headers_len) {
            set_error_msg(error_msg_out, &format!("invalid v4 headers length {}", headers_len));
            return 1;
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
        use crate::common::{ERROR_MSG_MAX_SIZE, PUBLICDATA_MAX_SIZE};
        use std::ffi::CStr;

        let headers: Vec<u8> = (0..5 * BlockHeader::SERIALIZED_SIZE).map(|i| i as u8).collect();
        for depth in 1..=STATE_WINDOW_DEPTH {
            let bytes = &headers[..depth * BlockHeader::SERIALIZED_SIZE];
            let (proposed, chain) = split_v4_headers(bytes).unwrap();
            // The proposed header's own proof commitment is not part of the statement.
            assert_eq!(proposed.to_bytes(), bytes[..Fp8BlockHeader::SERIALIZED_SIZE]);
            assert_eq!(
                chain.iter().flat_map(BlockHeader::to_bytes).collect::<Vec<_>>(),
                bytes[BlockHeader::SERIALIZED_SIZE..]
            );
        }
        let proof = CZKProof {
            public_data_len: 0,
            public_data: [0; PUBLICDATA_MAX_SIZE],
            proof_blob_len: 0,
            proof_blob: std::ptr::null_mut(),
        };
        for length in [0, 76, 107, 109, 215, 217, 433, 540] {
            assert!(split_v4_headers(&headers[..length]).is_err(), "length {length}");
            let mut err = [0 as c_char; ERROR_MSG_MAX_SIZE];
            let code = unsafe { verify_zk_proof_v4(headers.as_ptr(), length, &proof, 0x207fffff, err.as_mut_ptr()) };
            assert_eq!(code, 1);
            let message = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy();
            assert!(message.contains("invalid v4 headers length"), "{message}");
        }
    }
}
