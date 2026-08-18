//! C-compatible structs and utilities for Go FFI.

use std::os::raw::c_char;
use std::panic::AssertUnwindSafe;
use std::slice;
use std::sync::Mutex;

use anyhow::Result;
use zk_pow::api::proof::{IncompleteBlockHeader, MiningConfiguration, PublicProofParams, SeedDerivation};
use zk_pow::api::prove;
use zk_pow::circuit::pearl_circuit::{PearlRecursion, RecursionCircuit};
use zk_pow::ffi::plain_proof::PlainProof;

/// Size of reserved field in MiningConfiguration (exported to C header).
pub const MINING_CONFIG_RESERVED_SIZE: usize = 32;

/// Size of serialized MiningConfiguration in bytes (exported to C header).
/// Note: IncompleteBlockHeader (76) + MiningConfiguration (52) = 128 bytes = 2 blake3 blocks.
pub const MINING_CONFIG_SERIALIZED_SIZE: usize = 52;

/// Maximum size of the error message buffer passed from Go (exported to C header).
pub const ERROR_MSG_MAX_SIZE: usize = 128;

/// Maximum size of a serialized ZK proof blob (excluding IncompleteBlockHeader and MiningConfiguration, including everything else).
pub const MAX_ZK_PROOF_SIZE: usize = 60000;

/// Smallest noise rank the rank-penalty rule accepts (exported to C header).
pub const MIN_NOISE_RANK: u16 = 128;

// Compile-time assertions to ensure constants stay in sync
const _: () = assert!(MINING_CONFIG_RESERVED_SIZE == MiningConfiguration::RESERVED_SIZE);
const _: () = assert!(MINING_CONFIG_SERIALIZED_SIZE == MiningConfiguration::SERIALIZED_SIZE);
const _: () = assert!(MIN_NOISE_RANK as usize == zk_pow::api::sanity_checks::PENALTY_BASE_RANK);

type CircuitCache = <PearlRecursion as RecursionCircuit>::CircuitCache;
type V1CircuitCache = zk_pow::v1::circuit::circuit_utils::CircuitCache;

lazy_static::lazy_static! {
    /// Global circuit cache shared across Go FFI functions (verify and prove).
    /// Protected by a Mutex for thread-safe access from multiple Go goroutines.
    pub static ref CIRCUIT_CACHE: Mutex<CircuitCache> = {
        use zk_pow::circuit::embedded_cache;
        Mutex::new(CircuitCache::from_bytes(embedded_cache::CACHE_DATA)
            .expect("V2 circuit cache is missing or corrupt; cannot verify proofs"))
    };

    /// V1 circuit cache for verifying version-1 (master-format) proofs.
    pub static ref V1_CIRCUIT_CACHE: Mutex<V1CircuitCache> = {
        use zk_pow::v1::embedded_cache;
        Mutex::new(V1CircuitCache::from_bytes(embedded_cache::CACHE_DATA)
            .expect("V1 circuit cache is missing or corrupt; cannot verify V1 proofs"))
    };
}

/// Acquires the circuit cache. Recovers from poisoned mutex if a prior panic occurred.
/// The cache data is still valid for verifier after a panic, since the CircuitCache is read only.
pub(crate) fn acquire_cache() -> std::sync::MutexGuard<'static, CircuitCache> {
    CIRCUIT_CACHE.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Acquires the V1 circuit cache for version-1 proof verification.
pub(crate) fn acquire_v1_cache() -> std::sync::MutexGuard<'static, V1CircuitCache> {
    V1_CIRCUIT_CACHE.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Catches panics from a closure and returns Ok(result) or Err(panic_message).
/// The closure is wrapped in AssertUnwindSafe internally.
pub(crate) fn catch_panic<F, R>(f: F) -> Result<R>
where
    F: FnOnce() -> R,
{
    std::panic::catch_unwind(AssertUnwindSafe(f)).map_err(|e| {
        let msg = e
            .downcast::<String>()
            .map(|s| *s)
            .or_else(|e| e.downcast::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|_| "Unknown panic".to_string());
        let first_line = msg.lines().next().unwrap_or(&msg).to_string();
        anyhow::anyhow!(first_line)
    })
}

/// Size of the committed public data in bytes for a standard (non-MoE) ZK proof (exported to C header).
pub const PUBLICDATA_SIZE: usize = 164;
const _: () = assert!(PUBLICDATA_SIZE == PublicProofParams::WIRE_SIZE);

/// Maximum `public_data` buffer length, sized for largest MoE proofs (exported to C header).
pub const PUBLICDATA_MAX_SIZE: usize = 4807;
const _: () = assert!(PUBLICDATA_MAX_SIZE == PublicProofParams::MAX_WIRE_SIZE);

/// Go-owned ZK proof structure. Buffer is sized for the largest MoE proof;
/// `public_data_len` indicates how many bytes are actually used.
#[repr(C)]
pub struct CZKProof {
    pub public_data_len: usize,
    pub public_data: [u8; PUBLICDATA_MAX_SIZE],
    pub proof_blob_len: usize,
    pub proof_blob: *mut u8,
}

/// Writes an error message into a caller-allocated buffer of ERROR_MSG_MAX_SIZE bytes.
/// The message is always null-terminated. Truncation respects UTF-8 char boundaries.
/// # Safety
/// `out` must be null or a valid pointer to a buffer of at least `ERROR_MSG_MAX_SIZE` bytes.
pub(crate) unsafe fn set_error_msg(out: *mut c_char, msg: &str) {
    if out.is_null() {
        return;
    }
    let buf = slice::from_raw_parts_mut(out as *mut u8, ERROR_MSG_MAX_SIZE);
    // Truncate at a UTF-8 char boundary that fits in ERROR_MSG_MAX_SIZE-1 bytes (reserve 1 for null)
    let max_len = ERROR_MSG_MAX_SIZE - 1;
    let mut end = msg.len().min(max_len);
    while end > 0 && !msg.is_char_boundary(end) {
        end -= 1;
    }
    buf[..end].copy_from_slice(&msg.as_bytes()[..end]);
    buf[end] = 0;
}

/// ZK-proves a PlainProof, catching panics so they never cross the FFI boundary. On failure the
/// reason is written to `error_msg_out` and `None` is returned. Shared by the `mine`/`mine_moe`
/// (mine.rs) and `prove_plain_proof_ffi` (plain.rs) entry points.
pub(crate) unsafe fn zk_prove(
    error_msg_out: *mut c_char,
    header: IncompleteBlockHeader,
    proof: &PlainProof,
    seed_derivation: SeedDerivation,
) -> Option<prove::ProveResult> {
    let mut cache = acquire_cache();
    match catch_panic(|| prove::zk_prove_plain_proof(header, proof, &mut cache, false, seed_derivation)) {
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

/// Copies a prove result into a caller-allocated `CZKProof` (variable-length `public_data` plus the
/// proof blob), validating both sizes. On overflow / invalid wire size the reason is written to
/// `error_msg_out` and `false` is returned.
/// # Safety
/// `out.proof_blob` must be non-null and point to a buffer of at least `MAX_ZK_PROOF_SIZE` bytes.
pub(crate) unsafe fn copy_prove_result(error_msg_out: *mut c_char, out: &mut CZKProof, result: &prove::ProveResult) -> bool {
    if result.proof_data.len() > MAX_ZK_PROOF_SIZE {
        set_error_msg(error_msg_out, "proof exceeds MAX_ZK_PROOF_SIZE");
        return false;
    }
    let pd_len = result.public_data.len();
    if !PublicProofParams::is_valid_wire_size(pd_len) {
        set_error_msg(error_msg_out, &format!("public_data length {} is out of valid range", pd_len));
        return false;
    }
    out.public_data_len = pd_len;
    out.public_data[..pd_len].copy_from_slice(&result.public_data);

    let buffer = slice::from_raw_parts_mut(out.proof_blob, MAX_ZK_PROOF_SIZE);
    buffer[..result.proof_data.len()].copy_from_slice(&result.proof_data);
    out.proof_blob_len = result.proof_data.len();
    true
}
