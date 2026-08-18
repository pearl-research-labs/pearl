//! PlainProof FFI for the mining pool. Two entry points the pool links:
//!   - `verify_plain_proof_ffi` — cheap blake3 share validation (no plonky2).
//!   - `prove_plain_proof_ffi`  — plonky2 ZK certificate for block submission.
//!
//! Public data is variable-length (V2/MoE): `CZKProof.public_data_len` holds the used size.

use std::os::raw::c_char;
use std::slice;

use zk_pow::api::proof::IncompleteBlockHeader;
use zk_pow::api::verify;
use zk_pow::ffi::plain_proof::{check_cert_version_eligible, PlainProof};

use crate::common::{catch_panic, copy_prove_result, set_error_msg, zk_prove, CZKProof};

/// Verify a bincode-serialized PlainProof against the block header. The jackpot difficulty is
/// checked against `nbits_override` (the pool share target; 0 = use the header's own nbits, i.e. a
/// full-block check). The header's nbits is NOT modified — the proof commitment is derived from the
/// header including its nbits, so it must stay what the miner mined. blake3-only (no plonky2).
/// `cert_version` selects the noise-seed derivation and gates eligibility (MoE requires >= 2).
/// Returns 0 = accepted, 1 = rejected, 2 = bad input / panic; the reason is written to `error_msg_out`.
///
/// # Warning
/// This function does not bound `pp_len`. `PlainProof::deserialize_compat` has no internal byte
/// limit, so arbitrarily large untrusted input causes unbounded work inside this call. Callers
/// (the pool) must reject oversized shares before invoking this function. Suggested cap: ~8 MiB.
#[no_mangle]
pub unsafe extern "C" fn verify_plain_proof_ffi(
    block_header: *const IncompleteBlockHeader,
    pp_bytes: *const u8,
    pp_len: usize,
    cert_version: u32,
    nbits_override: u32,
    error_msg_out: *mut c_char,
) -> i32 {
    if block_header.is_null() || pp_bytes.is_null() || pp_len == 0 {
        set_error_msg(error_msg_out, "Null/empty input");
        return 2; // bad input (consistent with verify.rs and the docstring)
    }
    let header = *block_header;
    let bytes = slice::from_raw_parts(pp_bytes, pp_len);

    let result = catch_panic(|| {
        let pp: PlainProof = match PlainProof::deserialize_compat(bytes) {
            Ok(p) => p,
            Err(e) => return (1, format!("deserialize: {e}")),
        };
        let version = match check_cert_version_eligible(cert_version, &pp) {
            Ok(v) => v,
            Err(e) => return (1, format!("rejected: {e}")),
        };
        let nover = if nbits_override == 0 { None } else { Some(nbits_override) };
        match verify::verify_plain_proof(&header, &pp, nover, version.seed_derivation()) {
            Ok(()) => (0, "accepted".to_string()),
            Err(e) => (1, format!("rejected: {e}")),
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

/// Generate a plonky2 ZK certificate from a (bincode) PlainProof — the master's block-finalization
/// step (EXPENSIVE; needs the circuit cache). `cert_version` is as in `verify_plain_proof_ffi`.
/// Fills `zk_proof_out`: `public_data_len` + `public_data`
/// (variable-length; the caller copies `public_data[..public_data_len]` into the block certificate),
/// and `proof_blob`/`proof_blob_len` (the caller-allocated blob must hold `MAX_ZK_PROOF_SIZE` bytes).
/// Returns 0 = success, 2 = bad input / prove failure / panic.
///
/// # Warning
/// This function does not bound `pp_len`. `PlainProof::deserialize_compat` has no internal byte
/// limit, so arbitrarily large untrusted input causes unbounded work inside this call. Callers
/// (the pool) must reject oversized shares before invoking this function. Suggested cap: ~8 MiB.
#[no_mangle]
pub unsafe extern "C" fn prove_plain_proof_ffi(
    block_header: *const IncompleteBlockHeader,
    pp_bytes: *const u8,
    pp_len: usize,
    cert_version: u32,
    zk_proof_out: *mut CZKProof,
    error_msg_out: *mut c_char,
) -> i32 {
    if block_header.is_null() || pp_bytes.is_null() || pp_len == 0 || zk_proof_out.is_null() {
        set_error_msg(error_msg_out, "Null/empty input");
        return 2;
    }
    let header = *block_header;
    let bytes = slice::from_raw_parts(pp_bytes, pp_len);
    let out = &mut *zk_proof_out;
    if out.proof_blob.is_null() {
        set_error_msg(error_msg_out, "proof_blob buffer is null");
        return 2;
    }

    // Deserialize inside catch_panic so malformed input can't unwind across the FFI boundary (UB).
    let pp = match catch_panic(|| PlainProof::deserialize_compat(bytes)) {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            set_error_msg(error_msg_out, &format!("deserialize: {e}"));
            return 2;
        }
        Err(panic_msg) => {
            set_error_msg(error_msg_out, &format!("deserialize panic: {panic_msg}"));
            return 2;
        }
    };

    let version = match check_cert_version_eligible(cert_version, &pp) {
        Ok(v) => v,
        Err(e) => {
            set_error_msg(error_msg_out, &format!("cert version: {e}"));
            return 2;
        }
    };

    let result = match zk_prove(error_msg_out, header, &pp, version.seed_derivation()) {
        Some(r) => r,
        None => return 2,
    };
    if !copy_prove_result(error_msg_out, out, &result) {
        return 2;
    }

    set_error_msg(error_msg_out, "proof generation successful");
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::ffi::CStr;

    use bincode::Options;
    use rand_chacha::rand_core::SeedableRng;
    use zk_pow::api::proof::{MMAType, MiningConfiguration, MoEConfig, PeriodicPattern, SeedDerivation};
    use zk_pow::ffi::mine::try_mine_one_moe;

    use crate::common::{ERROR_MSG_MAX_SIZE, MAX_ZK_PROOF_SIZE, PUBLICDATA_MAX_SIZE};
    use crate::verify::verify_zk_proof_v2;

    /// CI-fast MoE parameters with permissive difficulty, mirroring zk-pow's moe_test baseline.
    fn test_params() -> (IncompleteBlockHeader, MiningConfiguration, usize, usize, usize) {
        let k = 1024usize;
        let header = IncompleteBlockHeader {
            version: 0,
            prev_block: [0; 32],
            merkle_root: *b"0123456789abcdef0123456789abcdef",
            timestamp: 0x66666666,
            nbits: 0x207FFFFF,
        };
        let config = MiningConfiguration {
            common_dim: k as u32,
            rank: 32,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41]).unwrap(),
            moe: Some(MoEConfig { e: 4, top_k: 1 }),
        };
        (header, config, 1024, 128, k)
    }

    /// Mine one PlainProof and serialize it the way a miner submits it (the inverse of
    /// `PlainProof::deserialize_compat`).
    fn mine_plain_proof_bytes() -> (IncompleteBlockHeader, Vec<u8>) {
        let (header, config, m, n, k) = test_params();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0xC0FFEE);
        let mut pp = None;
        for _ in 0..1000 {
            if let Some(found) = try_mine_one_moe(&mut rng, m, n, k, header, config, None, false, SeedDerivation::Legacy).unwrap()
            {
                pp = Some(found);
                break;
            }
        }
        let pp = pp.unwrap_or_else(|| panic!("no MoE proof found in 1000 attempts — check test difficulty params"));
        let bytes = bincode::options().with_fixint_encoding().serialize(&pp).unwrap();
        (header, bytes)
    }

    fn err_str(buf: &[c_char]) -> String {
        unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
    }

    fn new_czkproof(blob: &mut [u8]) -> CZKProof {
        CZKProof {
            public_data_len: 0,
            public_data: [0u8; PUBLICDATA_MAX_SIZE],
            proof_blob_len: 0,
            proof_blob: blob.as_mut_ptr(),
        }
    }

    /// End-to-end FFI flow: miner -> plain_proof -> verify_plain_proof_ffi -> prove_plain_proof_ffi
    /// -> verify_zk_proof_v2. Guards the whole Go-facing boundary against regressions.
    #[test]
    fn plain_proof_ffi_flow() {
        let (header, pp_bytes) = mine_plain_proof_bytes();
        let mut err = [0 as c_char; ERROR_MSG_MAX_SIZE];

        // 1. cheap blake3 share verify (V2 = legacy derivation)
        let code = unsafe { verify_plain_proof_ffi(&header, pp_bytes.as_ptr(), pp_bytes.len(), 2, 0, err.as_mut_ptr()) };
        assert_eq!(code, 0, "verify_plain_proof_ffi rejected a valid proof: {}", err_str(&err));

        // 2. plonky2 prove into a caller-allocated CZKProof
        let mut blob = vec![0u8; MAX_ZK_PROOF_SIZE];
        let mut zk = new_czkproof(&mut blob);
        let code = unsafe { prove_plain_proof_ffi(&header, pp_bytes.as_ptr(), pp_bytes.len(), 2, &mut zk, err.as_mut_ptr()) };
        assert_eq!(code, 0, "prove_plain_proof_ffi failed: {}", err_str(&err));
        assert!(zk.proof_blob_len > 0, "empty proof blob");
        assert!(zk.public_data_len >= 24, "V2 public_data too short: {}", zk.public_data_len);

        // 3. verify the produced V2 certificate
        let code = unsafe { verify_zk_proof_v2(&header, &zk, err.as_mut_ptr()) };
        assert_eq!(code, 0, "verify_zk_proof_v2 rejected the produced cert: {}", err_str(&err));
    }

    /// Malformed input must be rejected with the bad-input code 2 and a deserialize error message.
    /// Note: this only exercises the `Err` path of `PlainProof::deserialize_compat` — bincode is
    /// panic-free on malformed input, so a genuine unwind cannot be triggered here to regression-test
    /// the `catch_panic` wrapping of deserialization (the actual UB guard) directly. That guard is
    /// verified by code review (deserialization happens inside `catch_panic` in `plain.rs`), not by
    /// this test.
    #[test]
    fn prove_plain_proof_ffi_rejects_malformed_input() {
        let header = test_params().0;
        let garbage = [0xFFu8; 64];
        let mut blob = vec![0u8; MAX_ZK_PROOF_SIZE];
        let mut zk = new_czkproof(&mut blob);
        let mut err = [0 as c_char; ERROR_MSG_MAX_SIZE];

        let code = unsafe { prove_plain_proof_ffi(&header, garbage.as_ptr(), garbage.len(), 2, &mut zk, err.as_mut_ptr()) };
        assert_eq!(code, 2, "expected bad-input code 2, got {}: {}", code, err_str(&err));
    }
}
