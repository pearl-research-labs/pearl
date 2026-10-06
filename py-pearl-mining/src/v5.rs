//! V5 (FP16 / A100) functions. The prover class lives in `zk_pow::ffi::py_v5`.

use pyo3::prelude::*;

use zk_pow::ffi::py_v5::PyFp16Prover;
use zk_pow::v5::api::params::Fp16Device;
use zk_pow::v5::api::plain_proof::{Fp16JobParams, Fp16MatrixProof, Fp16OperandParams, Fp16PlainProof};
use zk_pow::v5::api::verify as fp16_verify;

use crate::common::IncompleteBlockHeader;

/// Cert v5 (PlainFP16 / A100) plain verification against the FP16 verifier.
///
/// Like v4, the proof carries its own `ancestor_header` (σ_Δ) in the job;
/// `block_header` (σ̂) keys the A side and supplies the default difficulty.
/// Authenticating σ_Δ (a member of the state window) is the caller's
/// responsibility. NOTE: the plaintext path is retired as a consensus path (the
/// wired V5 consensus artifact is the header-bound ZK certificate); this entry
/// remains for prover-side checks.
#[pyfunction]
#[pyo3(signature = (block_header, plain_proof, nbits_override=None))]
pub fn verify_fp16_plain_proof(
    block_header: IncompleteBlockHeader,
    plain_proof: Fp16PlainProof,
    nbits_override: Option<u32>,
) -> PyResult<(bool, String)> {
    match fp16_verify::verify_fp16_plain_proof(&block_header, &plain_proof, nbits_override) {
        Ok(()) => Ok((true, "Mining solution verified successfully".into())),
        Err(e) => Ok((false, e.to_string())),
    }
}

pub fn register_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Fp16PlainProof>()?;
    m.add_class::<Fp16JobParams>()?;
    m.add_class::<Fp16OperandParams>()?;
    m.add_class::<Fp16MatrixProof>()?;
    m.add_class::<Fp16Device>()?;
    Ok(())
}

pub fn register_proofs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(verify_fp16_plain_proof, m)?)?;
    // FP16 (A100) ZK prover: produces the Fp16ZkCertificate bytes the node's
    // CertificateV5.ProofData carries.
    m.add_class::<PyFp16Prover>()?;
    Ok(())
}
