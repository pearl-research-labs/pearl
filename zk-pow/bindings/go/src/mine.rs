//! Mining FFI - performs mining and returns a ZK proof directly.

use std::os::raw::c_char;

use zk_pow::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, MoEConfig, PeriodicPattern};
use zk_pow::api::prove;
use zk_pow::ffi::mine::{mine as ffi_mine, mine_moe as ffi_mine_moe};
use zk_pow::ffi::plain_proof::PlainProof;

use crate::common::{catch_panic, copy_prove_result, set_error_msg, zk_prove, CZKProof, MIN_NOISE_RANK};

// ============================================================================
// Default mining configuration
// ============================================================================

/// Noise rank the default configuration mines at. Sitting on the rank-penalty
/// floor keeps mined blocks valid once that rule activates, without paying the
/// penalty a larger rank would.
const DEFAULT_RANK: u16 = MIN_NOISE_RANK;

/// Smallest common dimension the sanity checks allow for [`DEFAULT_RANK`] (`k >= 16 * r`).
const DEFAULT_COMMON_DIM: u32 = 16 * DEFAULT_RANK as u32;

/// Index patterns partitioning the jackpot tile: 4 rows of A by 8 columns of B.
const DEFAULT_ROWS_PATTERN: [u32; 4] = [0, 8, 64, 72];
const DEFAULT_COLS_PATTERN: [u32; 8] = [0, 1, 8, 9, 32, 33, 40, 41];

// ============================================================================
// Public FFI
// ============================================================================

/// Serializes the default mining configuration.
///
/// `e == 0` selects a standard job; otherwise it selects a GROUPED_GEMM job
/// routing each token to `top_k` experts.
///
/// # Returns
/// - 0: configuration written to `mining_config_out`
/// - 2: System error
///
/// # Safety
/// - `mining_config_out` must be a valid pointer to `MINING_CONFIG_SERIALIZED_SIZE` bytes
/// - `error_msg_out` must be null or a valid pointer to a caller-allocated buffer of `ERROR_MSG_MAX_SIZE` bytes
#[no_mangle]
pub unsafe extern "C" fn default_mining_config(
    e: u16,
    top_k: u16,
    mining_config_out: *mut [u8; crate::common::MINING_CONFIG_SERIALIZED_SIZE],
    error_msg_out: *mut c_char,
) -> i32 {
    let result = catch_panic(|| {
        if mining_config_out.is_null() {
            set_error_msg(error_msg_out, "Null pointer");
            return 2;
        }

        match build_default_config(e, top_k) {
            Ok(config) => {
                *mining_config_out = config.to_bytes();
                set_error_msg(error_msg_out, "Default mining config written");
                0
            }
            Err(err) => {
                set_error_msg(error_msg_out, &format!("Invalid default mining config: {err}"));
                2
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(panic_msg) => {
            set_error_msg(error_msg_out, &format!("Internal panic: {panic_msg}"));
            2
        }
    }
}

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

fn build_default_config(e: u16, top_k: u16) -> anyhow::Result<MiningConfiguration> {
    Ok(MiningConfiguration {
        common_dim: DEFAULT_COMMON_DIM,
        rank: DEFAULT_RANK,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&DEFAULT_ROWS_PATTERN)?,
        cols_pattern: PeriodicPattern::from_list(&DEFAULT_COLS_PATTERN)?,
        moe: (e != 0).then_some(MoEConfig { e, top_k }),
    })
}

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

    if !copy_prove_result(error_msg_out, out, &result) {
        return 2;
    }

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

    zk_prove(error_msg_out, header, &proof)
}
