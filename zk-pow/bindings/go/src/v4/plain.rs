//! Cert v4 (FP8) PlainProofV4 FFI for the mining pool: `verify_plain_proof_v4_ffi`, cheap share
//! validation (no plonky2).

use std::os::raw::c_char;
use std::slice;

use zk_pow::v4::api::plain_proof::PlainProofV4;
use zk_pow::v4::api::primitives::IncompleteBlockHeader as Fp8BlockHeader;
use zk_pow::v4::api::verify as fp8_verify;

use crate::common::{catch_panic, set_error_msg};

/// Verify a bincode-serialized cert v4 `PlainProofV4` (FP8) against the proposed header, given as
/// its 76 canonical wire bytes (`IncompleteBlockHeader::SERIALIZED_SIZE`; the proof commitment is
/// not part of the statement). The proof's ancestor `σ_d` is authenticated by hash-walking the
/// proof's own `ancestor_chain` from the header's `prev_block`. The jackpot difficulty is checked
/// against `nbits_override` (0 = the header's own nbits). No plonky2.
/// Returns 0 = accepted, 1 = rejected, 2 = bad input / panic; the reason is written to `error_msg_out`.
///
/// # Warning
/// This function does not bound `pp_len`. `PlainProofV4::from_bytes` has no internal byte limit,
/// so callers must reject oversized proofs before invoking this function.
///
/// # Safety
/// - `block_header` must point to 76 readable bytes
/// - `pp_bytes` must point to `pp_len` readable bytes
/// - `error_msg_out` must be null or a valid pointer to a caller-allocated buffer of `ERROR_MSG_MAX_SIZE` bytes
#[no_mangle]
pub unsafe extern "C" fn verify_plain_proof_v4_ffi(
    block_header: *const u8,
    pp_bytes: *const u8,
    pp_len: usize,
    nbits_override: u32,
    error_msg_out: *mut c_char,
) -> i32 {
    if block_header.is_null() || pp_bytes.is_null() || pp_len == 0 {
        set_error_msg(error_msg_out, "Null/empty input");
        return 2;
    }
    let header = slice::from_raw_parts(block_header, Fp8BlockHeader::SERIALIZED_SIZE);
    let bytes = slice::from_raw_parts(pp_bytes, pp_len);

    let result = catch_panic(|| {
        let header = match Fp8BlockHeader::from_bytes(header) {
            Ok(h) => h,
            Err(e) => return (2, format!("header: {e}")),
        };
        let pp = match PlainProofV4::from_bytes(bytes) {
            Ok(p) => p,
            Err(e) => return (1, format!("deserialize: {e:#}")),
        };
        let nover = if nbits_override == 0 { None } else { Some(nbits_override) };
        match fp8_verify::verify_plain_proof(&header, &pp, nover) {
            Ok(()) => (0, "accepted".to_string()),
            Err(e) => (1, format!("rejected: {e:#}")),
        }
    });

    match result {
        Ok((code, msg)) => {
            set_error_msg(error_msg_out, &msg);
            code
        }
        Err(panic_msg) => {
            set_error_msg(error_msg_out, &format!("panic: {panic_msg}"));
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::ffi::CStr;

    use crate::common::ERROR_MSG_MAX_SIZE;

    fn err_str(buf: &[c_char]) -> String {
        unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
    }

    /// The v4 entry point: null input is bad input (2); undecodable proof bytes reject (1).
    /// Acceptance runs the same `zk_pow::v4::api::verify::verify_plain_proof` covered in zk-pow.
    #[test]
    fn verify_plain_proof_v4_ffi_rejects_bad_input() {
        let header = [0u8; Fp8BlockHeader::SERIALIZED_SIZE];
        let garbage = [0xFFu8; 64];
        let mut err = [0 as c_char; ERROR_MSG_MAX_SIZE];

        let code = unsafe { verify_plain_proof_v4_ffi(std::ptr::null(), garbage.as_ptr(), garbage.len(), 0, err.as_mut_ptr()) };
        assert_eq!(code, 2, "{}", err_str(&err));

        let code = unsafe { verify_plain_proof_v4_ffi(header.as_ptr(), garbage.as_ptr(), garbage.len(), 0, err.as_mut_ptr()) };
        assert_eq!(code, 1, "{}", err_str(&err));
        assert!(err_str(&err).contains("deserialize"), "{}", err_str(&err));
    }
}
