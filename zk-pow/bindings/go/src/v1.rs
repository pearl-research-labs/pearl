//! V1 (master-format) ZK Proof Verification FFI (Security Critical)
//!
//! Extra care must be taken when modifying this code as it's critical for security.

use std::os::raw::c_char;
use std::slice;
use std::sync::Mutex;

use crate::common::{catch_panic, set_error_msg, CZKProof, IncompleteBlockHeader, MAX_ZK_PROOF_SIZE};

type V1CircuitCache = zk_pow::v1::circuit::circuit_utils::CircuitCache;

lazy_static::lazy_static! {
    /// V1 circuit cache for verifying version-1 (master-format) proofs.
    pub static ref V1_CIRCUIT_CACHE: Mutex<V1CircuitCache> = {
        use zk_pow::v1::embedded_cache;
        Mutex::new(V1CircuitCache::from_bytes(embedded_cache::CACHE_DATA)
            .expect("V1 circuit cache is missing or corrupt; cannot verify V1 proofs"))
    };
}

/// Acquires the V1 circuit cache for version-1 proof verification.
pub(crate) fn acquire_v1_cache() -> std::sync::MutexGuard<'static, V1CircuitCache> {
    V1_CIRCUIT_CACHE.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
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

        let cache = acquire_v1_cache();

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
