//! Certificate-version dispatchers.
//!
//! One entry point per operation, keyed on the certificate version the node
//! reports in getblocktemplate (`requiredcertversion`), so callers cannot pick
//! the wrong circuit around the MoE fork crossover. The only module that calls
//! into every version module.

use pyo3::prelude::*;

use zk_pow::ffi::CertificateVersion;
use zk_pow::v2::ffi::plain_proof::PlainProof;
use zk_pow::v4::api::plain_proof::PlainProofV4;

use crate::common::{value_err, IncompleteBlockHeader, PyProof};
use crate::v1::{generate_proof_v1, verify_plain_proof_v1, verify_proof_v1};
use crate::v2::{
    generate_proof_v2, generate_proof_v3, verify_plain_proof_v2, verify_plain_proof_v3,
    verify_proof_v2, verify_proof_v3,
};
use crate::v4::verify_plain_proof_v4;

/// Cert v4 (PlainFP8) blocks certify the plain proof itself; no ZK circuit
/// is wired for v4 proofs yet, so the ZK entry points fail closed.
fn not_implemented_fp8() -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err(
        "cert v4 (PlainFP8) has no ZK proof yet; use verify_plain_proof_for_cert_version",
    )
}

#[pyfunction]
#[pyo3(name = "check_cert_version_eligible")]
pub fn py_check_cert_version_eligible(
    cert_version: u32,
    plain_proof: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match CertificateVersion::try_from(cert_version).map_err(value_err)? {
        CertificateVersion::ZkDense | CertificateVersion::ZkMoe | CertificateVersion::ZkV3 => {
            extract_int7(plain_proof)?
                .check_cert_version_eligible(cert_version)
                .map_err(value_err)?;
            Ok(())
        }
        CertificateVersion::PlainFp8 => {
            extract_v4(plain_proof)?;
            Ok(())
        }
    }
}

#[pyfunction]
pub fn generate_proof_for_cert_version(
    cert_version: u32,
    block_header: IncompleteBlockHeader,
    plain_proof: PlainProof,
) -> PyResult<PyProof> {
    match plain_proof
        .check_cert_version_eligible(cert_version)
        .map_err(value_err)?
    {
        CertificateVersion::ZkDense => generate_proof_v1(block_header, plain_proof),
        CertificateVersion::ZkMoe => generate_proof_v2(block_header, plain_proof),
        CertificateVersion::ZkV3 => generate_proof_v3(block_header, plain_proof),
        CertificateVersion::PlainFp8 => Err(not_implemented_fp8()),
    }
}

#[pyfunction]
pub fn verify_proof_for_cert_version(
    cert_version: u32,
    block_header: IncompleteBlockHeader,
    proof: &PyProof,
) -> PyResult<(bool, String)> {
    // No PlainProof here, so only the version itself can be validated.
    match CertificateVersion::try_from(cert_version).map_err(value_err)? {
        CertificateVersion::ZkDense => verify_proof_v1(block_header, proof),
        CertificateVersion::ZkMoe => verify_proof_v2(block_header, proof),
        CertificateVersion::ZkV3 => verify_proof_v3(block_header, proof),
        CertificateVersion::PlainFp8 => Err(not_implemented_fp8()),
    }
}

#[pyfunction]
#[pyo3(signature = (cert_version, block_header, plain_proof, nbits_override=None))]
pub fn verify_plain_proof_for_cert_version(
    cert_version: u32,
    block_header: IncompleteBlockHeader,
    plain_proof: &Bound<'_, PyAny>,
    nbits_override: Option<u32>,
) -> PyResult<(bool, String)> {
    match CertificateVersion::try_from(cert_version).map_err(value_err)? {
        CertificateVersion::ZkDense => {
            let proof = extract_int7(plain_proof)?;
            proof
                .check_cert_version_eligible(cert_version)
                .map_err(value_err)?;
            verify_plain_proof_v1(block_header, proof, nbits_override)
        }
        CertificateVersion::ZkMoe => {
            let proof = extract_int7(plain_proof)?;
            proof
                .check_cert_version_eligible(cert_version)
                .map_err(value_err)?;
            verify_plain_proof_v2(block_header, proof, nbits_override)
        }
        CertificateVersion::ZkV3 => {
            let proof = extract_int7(plain_proof)?;
            proof
                .check_cert_version_eligible(cert_version)
                .map_err(value_err)?;
            verify_plain_proof_v3(block_header, proof, nbits_override)
        }
        CertificateVersion::PlainFp8 => {
            let proof = extract_v4(plain_proof)?;
            verify_plain_proof_v4(block_header, proof, nbits_override)
        }
    }
}

fn extract_int7(plain_proof: &Bound<'_, PyAny>) -> PyResult<PlainProof> {
    plain_proof.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("certificate versions 1/2/3 require PlainProof")
    })
}

fn extract_v4(plain_proof: &Bound<'_, PyAny>) -> PyResult<PlainProofV4> {
    plain_proof.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("certificate version 4 requires PlainProofV4")
    })
}
