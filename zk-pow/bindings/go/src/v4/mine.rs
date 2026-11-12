//! CPU-only V4 mining and ZK proving for tests.

use std::os::raw::c_char;
use std::slice;

use anyhow::{ensure, Result};
use zk_pow::v4::api::mine::mine;
use zk_pow::v4::api::primitives::{BlockHeader, IncompleteBlockHeader};
use zk_pow::v4::api::zk::Fp8Prover;

use crate::common::{catch_panic, set_error_msg, CZKProof, MAX_ZK_PROOF_SIZE, PUBLICDATA_MAX_SIZE};

/// Mines a fixed dense B200-emulated V4 job on CPU, then generates its ZK proof.
/// Random candidates are checked against the proposed header's nbits until one
/// wins. Intended for easy regtest/integration targets; no GPU, attempt limit,
/// cancellation, or optimized search. Prover setup is rebuilt on every call.
/// The immediate parent is the statement's ancestor, so no extra chain is needed.
/// Returns 0 on success or 2 on invalid input, proving error, or panic.
/// Output lengths are zero on failure.
///
/// # Safety
/// - `block_header` points to 76 canonical readable wire bytes (no commitment).
/// - `parent_header` points to 108 canonical readable wire bytes (with commitment).
/// - `zk_proof_out` points to writable `CZKProof`; its `proof_blob` points to a
///   writable buffer of at least `MAX_ZK_PROOF_SIZE` bytes. Buffers must not alias.
/// - `error_msg_out` is null or points to `ERROR_MSG_MAX_SIZE` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn mine_v4(
    block_header: *const u8,
    parent_header: *const u8,
    zk_proof_out: *mut CZKProof,
    error_msg_out: *mut c_char,
) -> i32 {
    if zk_proof_out.is_null() {
        set_error_msg(error_msg_out, "Null output");
        return 2;
    }
    // SAFETY: the caller supplies a writable output and distinct input buffers.
    let out = &mut *zk_proof_out;
    out.public_data_len = 0;
    out.proof_blob_len = 0;
    if block_header.is_null() || parent_header.is_null() || out.proof_blob.is_null() {
        set_error_msg(error_msg_out, "Null input or proof buffer");
        return 2;
    }
    let result = catch_panic(|| -> Result<()> {
        // SAFETY: the caller supplies headers of the documented fixed wire sizes.
        let header =
            IncompleteBlockHeader::from_bytes(slice::from_raw_parts(block_header, IncompleteBlockHeader::SERIALIZED_SIZE))?;
        let parent = BlockHeader::from_bytes(slice::from_raw_parts(parent_header, BlockHeader::SERIALIZED_SIZE))?;
        let plain = mine(&header, &parent)?;
        let (public, proof) = Fp8Prover::setup(plain.job.common.device)?.prove(&header, &plain)?;
        ensure!(public.len() <= PUBLICDATA_MAX_SIZE, "V4 public data exceeds buffer");
        ensure!(proof.len() <= MAX_ZK_PROOF_SIZE, "V4 proof exceeds buffer");
        // SAFETY: the caller supplies MAX_ZK_PROOF_SIZE writable bytes.
        slice::from_raw_parts_mut(out.proof_blob, MAX_ZK_PROOF_SIZE)[..proof.len()].copy_from_slice(&proof);
        out.public_data[..public.len()].copy_from_slice(&public);
        out.public_data_len = public.len();
        out.proof_blob_len = proof.len();
        Ok(())
    });
    match result {
        Ok(Ok(())) => {
            set_error_msg(error_msg_out, "Mining successful");
            0
        }
        Ok(Err(e)) => {
            set_error_msg(error_msg_out, &format!("V4 mining failed: {e:#}"));
            2
        }
        Err(e) => {
            set_error_msg(error_msg_out, &format!("V4 mining panic: {e}"));
            2
        }
    }
}
