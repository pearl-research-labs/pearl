//! C-compatible structs and utilities for Go FFI, shared by every version module.

use std::os::raw::c_char;
use std::panic::AssertUnwindSafe;
use std::slice;

use anyhow::Result;
use zk_pow::v2::api::proof::PublicProofParams;
use zk_pow::v4::api::public_params::PublicParams;

/// The C-facing block header (the v2 struct) that every version's header-taking entry point accepts.
pub use zk_pow::v2::api::proof::IncompleteBlockHeader;

/// Maximum size of the error message buffer passed from Go (exported to C header).
pub const ERROR_MSG_MAX_SIZE: usize = 128;

/// Maximum size of a serialized ZK proof blob (excluding IncompleteBlockHeader and MiningConfiguration, including everything else).
pub const MAX_ZK_PROOF_SIZE: usize = 60000;

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

/// Maximum `public_data` buffer length over all schemes. Equal to the v2 MoE
/// maximum (`PublicProofParams::MAX_WIRE_SIZE`); the v4/fp8 maximum
/// (`PublicParams::MAX_WIRE_SIZE`) must fit inside it. `CZKProof.public_data` is
/// shared by the v2 and fp8 paths, so it must fit both. Exported to C and
/// mirrored as the FFI statement buffer; the Go node's `wire.PublicDataMaxSizeV2`
/// caps v2 certificates separately.
///
/// Kept literal so cbindgen can emit a `#define`. The compile-time assertions
/// below guard the value against drift.
pub const PUBLICDATA_MAX_SIZE: usize = 4807;
const _: () = {
    assert!(PUBLICDATA_MAX_SIZE >= PublicParams::MAX_WIRE_SIZE);
    assert!(PUBLICDATA_MAX_SIZE == PublicProofParams::MAX_WIRE_SIZE);
};

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
