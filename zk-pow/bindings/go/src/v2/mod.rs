//! V2 Go FFI, which also serves cert v3 (same circuits, salted noise seed): the circuit cache,
//! the wire constants Go mirrors, and the prove helpers shared by [`mine`] and [`plain`].

use std::os::raw::c_char;
use std::slice;
use std::sync::Mutex;

use zk_pow::v2::api::proof::{IncompleteBlockHeader, MiningConfiguration, PublicProofParams};
use zk_pow::v2::api::prove;
use zk_pow::v2::api::seed::SeedDerivation;
use zk_pow::v2::circuit::pearl_circuit::{PearlRecursion, RecursionCircuit};
use zk_pow::v2::ffi::plain_proof::PlainProof;

use crate::common::{catch_panic, set_error_msg, CZKProof, MAX_ZK_PROOF_SIZE};

pub mod mine;
pub mod plain;
pub mod verify;

/// Size of reserved field in MiningConfiguration (exported to C header).
pub const MINING_CONFIG_RESERVED_SIZE: usize = 32;

/// Size of serialized MiningConfiguration in bytes (exported to C header).
/// Note: IncompleteBlockHeader (76) + MiningConfiguration (52) = 128 bytes = 2 blake3 blocks.
pub const MINING_CONFIG_SERIALIZED_SIZE: usize = 52;

/// Smallest noise rank the rank-penalty rule accepts (exported to C header).
pub const MIN_NOISE_RANK: u16 = 128;

// Compile-time assertions to ensure constants stay in sync
const _: () = assert!(MINING_CONFIG_RESERVED_SIZE == MiningConfiguration::RESERVED_SIZE);
const _: () = assert!(MINING_CONFIG_SERIALIZED_SIZE == MiningConfiguration::SERIALIZED_SIZE);
const _: () = assert!(MIN_NOISE_RANK as usize == zk_pow::v2::api::sanity_checks::PENALTY_BASE_RANK);

type CircuitCache = <PearlRecursion as RecursionCircuit>::CircuitCache;

lazy_static::lazy_static! {
    /// Global circuit cache shared across Go FFI functions (verify and prove).
    /// Protected by a Mutex for thread-safe access from multiple Go goroutines.
    pub static ref CIRCUIT_CACHE: Mutex<CircuitCache> = {
        use zk_pow::v2::circuit::embedded_cache;
        Mutex::new(CircuitCache::from_bytes(embedded_cache::CACHE_DATA)
            .expect("V2 circuit cache is missing or corrupt; cannot verify proofs"))
    };
}

/// Acquires the circuit cache. Recovers from poisoned mutex if a prior panic occurred.
/// The cache data is still valid for verifier after a panic, since the CircuitCache is read only.
pub(crate) fn acquire_cache() -> std::sync::MutexGuard<'static, CircuitCache> {
    CIRCUIT_CACHE.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Size of the committed public data in bytes for a standard (non-MoE) ZK proof (exported to C header).
pub const PUBLICDATA_SIZE: usize = 164;
const _: () = assert!(PUBLICDATA_SIZE == PublicProofParams::WIRE_SIZE);

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
