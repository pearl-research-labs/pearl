//! V4 (FP8) functions. The prover/verifier classes live in `zk_pow::ffi::py_v4`.

use pyo3::prelude::*;

use zk_pow::v4::api::plain_proof::PlainProofV4;
use zk_pow::v4::api::verify as fp8_verify;

use crate::common::IncompleteBlockHeader;

/// Cert v4 (PlainFP8) plain verification against the mainline FP8 verifier.
///
/// The proof carries its own `ancestor_header` (σ_d) in the job and the headers
/// linking it to `block_header` (σ̂) in `ancestor_chain`; the verifier
/// authenticates σ_d by hash-walking that chain from σ̂'s `prev_block`.
#[pyfunction]
#[pyo3(signature = (block_header, plain_proof, nbits_override=None))]
pub fn verify_plain_proof_v4(
    block_header: IncompleteBlockHeader,
    plain_proof: PlainProofV4,
    nbits_override: Option<u32>,
) -> PyResult<(bool, String)> {
    match fp8_verify::verify_plain_proof(&block_header, &plain_proof, nbits_override) {
        Ok(()) => Ok((true, "Mining solution verified successfully".into())),
        Err(e) => Ok((false, e.to_string())),
    }
}
