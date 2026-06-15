//! Mining FFI - performs mining and returns a ZK proof directly.

use std::os::raw::c_char;
use std::slice;

use zk_pow::api::proof::{IncompleteBlockHeader, MiningConfiguration, PublicProofParams};
use zk_pow::api::prove;
use zk_pow::ffi::mine::{mine as ffi_mine, mine_moe as ffi_mine_moe};
use zk_pow::ffi::plain_proof::PlainProof;

use crate::common::{acquire_cache, catch_panic, set_error_msg, CZKProof, MAX_ZK_PROOF_SIZE};

// ============================================================================
// Public FFI
// ============================================================================

/// Perform mining and generate a standard (non-MoE) ZK proof in one step.
///
/// # Returns
/// - 0: Mining and proof generation successful
/// - 1: Invalid input
/// - 2: System error
///
/// # Safety
/// - All pointers must be valid
/// - `zk_proof_out.proof_blob` must have capacity `MAX_ZK_PROOF_SIZE`
/// - `error_msg_out` must be null or a valid pointer to a caller-allocated buffer of `ERROR_MSG_MAX_SIZE` bytes
#[no_mangle]
pub unsafe extern "C" fn mine(
    m: u32,
    n: u32,
    block_header: *const IncompleteBlockHeader,
    mining_config: *const [u8; crate::common::MINING_CONFIG_SERIALIZED_SIZE],
    zk_proof_out: *mut CZKProof,
    error_msg_out: *mut c_char,
) -> i32 {
    mine_inner(block_header, mining_config, zk_proof_out, error_msg_out, |header, config| {
        ffi_mine(
            m as usize,
            n as usize,
            config.common_dim as usize,
            header,
            config,
            None,
            false,
        )
    })
}

/// Perform MoE mining and generate a ZK proof in one step.
///
/// # Returns
/// - 0: Mining and proof generation successful
/// - 1: Invalid input
/// - 2: System error
///
/// # Safety
/// - All pointers must be valid
/// - `zk_proof_out.proof_blob` must have capacity `MAX_ZK_PROOF_SIZE`
/// - `error_msg_out` must be null or a valid pointer to a caller-allocated buffer of `ERROR_MSG_MAX_SIZE` bytes
///
/// `e` and `top_k` are read from the serialized `mining_config` trailer (committed in the job_key).
#[no_mangle]
pub unsafe extern "C" fn mine_moe(
    m: u32,
    n: u32,
    block_header: *const IncompleteBlockHeader,
    mining_config: *const [u8; crate::common::MINING_CONFIG_SERIALIZED_SIZE],
    zk_proof_out: *mut CZKProof,
    error_msg_out: *mut c_char,
) -> i32 {
    mine_inner(block_header, mining_config, zk_proof_out, error_msg_out, |header, config| {
        ffi_mine_moe(
            m as usize,
            n as usize,
            config.common_dim as usize,
            header,
            config,
            None,
            false,
        )
    })
}

// ============================================================================
// Shared helpers
// ============================================================================

/// Shared boilerplate for both `mine` and `mine_moe`.
///
/// Parses the mining config, validates pointers, calls the provided mining
/// closure, then copies the result into `zk_proof_out`.
unsafe fn mine_inner(
    block_header: *const IncompleteBlockHeader,
    mining_config: *const [u8; crate::common::MINING_CONFIG_SERIALIZED_SIZE],
    zk_proof_out: *mut CZKProof,
    error_msg_out: *mut c_char,
    do_mine: impl FnOnce(IncompleteBlockHeader, MiningConfiguration) -> anyhow::Result<PlainProof>,
) -> i32 {
    if block_header.is_null() || mining_config.is_null() || zk_proof_out.is_null() {
        set_error_msg(error_msg_out, "Null pointer");
        return 2;
    }

    let config = match MiningConfiguration::from_bytes(&*mining_config) {
        Ok(c) => c,
        Err(e) => {
            set_error_msg(error_msg_out, &format!("Invalid mining config: {}", e));
            return 2;
        }
    };

    let header = *block_header;
    let out = &mut *zk_proof_out;
    if out.proof_blob.is_null() {
        set_error_msg(error_msg_out, "proof_blob buffer is null");
        return 2;
    }

    let result = match mine_and_prove(error_msg_out, header, || do_mine(header, config)) {
        Some(r) => r,
        None => return 2,
    };

    let pd_len = result.public_data.len();
    if !PublicProofParams::is_valid_wire_size(pd_len) {
        set_error_msg(error_msg_out, &format!("public_data length {} is out of valid range", pd_len));
        return 2;
    }
    out.public_data_len = pd_len;
    out.public_data[..pd_len].copy_from_slice(&result.public_data);

    let buffer = slice::from_raw_parts_mut(out.proof_blob, MAX_ZK_PROOF_SIZE);
    buffer[..result.proof_data.len()].copy_from_slice(&result.proof_data);
    out.proof_blob_len = result.proof_data.len();

    set_error_msg(error_msg_out, "Mining and proof generation successful");
    0
}

/// Mine with the given closure and ZK-prove the result. Returns `None` on
/// error (after writing the message to `error_msg_out`).
unsafe fn mine_and_prove(
    error_msg_out: *mut c_char,
    header: IncompleteBlockHeader,
    mine_fn: impl FnOnce() -> anyhow::Result<PlainProof>,
) -> Option<prove::ProveResult> {
    let proof = match catch_panic(mine_fn) {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            set_error_msg(error_msg_out, &format!("Mining failed: {}", e));
            return None;
        }
        Err(panic_msg) => {
            set_error_msg(error_msg_out, &format!("Mining panic: {}", panic_msg));
            return None;
        }
    };

    let mut cache = acquire_cache();
    match catch_panic(|| prove::zk_prove_plain_proof(header, &proof, &mut cache, false)) {
        Ok(Ok(r)) => Some(r),
        Ok(Err(e)) => {
            set_error_msg(error_msg_out, &format!("Prove failed: {}", e));
            None
        }
        Err(panic_msg) => {
            set_error_msg(error_msg_out, &format!("Prove panic: {}", panic_msg));
            None
        }
    }
}
