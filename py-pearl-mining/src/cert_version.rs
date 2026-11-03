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
use zk_pow::v5::api::plain_proof::Fp16PlainProof;

use crate::common::{value_err, IncompleteBlockHeader, PyProof};
use crate::v1::{generate_proof_v1, verify_plain_proof_v1, verify_proof_v1};
use crate::v2::{
    generate_proof_v2, generate_proof_v3, verify_plain_proof_v2, verify_plain_proof_v3,
    verify_proof_v2, verify_proof_v3,
};
use crate::v4::verify_plain_proof_v4;
use crate::v5::verify_fp16_plain_proof;

/// Cert v4 (PlainFP8) blocks certify the plain proof itself; no ZK circuit
/// is wired for v4 proofs yet, so the ZK entry points fail closed.
fn not_implemented_fp8() -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err(
        "cert v4 (PlainFP8) has no ZK proof yet; use verify_plain_proof_for_cert_version",
    )
}

/// Cert v5 (FP16/A100) proving does not go through this int7-shaped dispatcher:
/// the FP16 ZK prover takes the opened tile operand codes + the public job (not a
/// `PlainProof`), so the miner drives it directly via the `Fp16Prover` class.
fn not_implemented_fp16() -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err(
        "cert v5 (FP16) proving uses the Fp16Prover class directly (prove(header, job, a_codes, b_codes)), not generate_proof_for_cert_version",
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
        CertificateVersion::PlainFp16 => {
            extract_fp16(plain_proof)?;
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
        CertificateVersion::PlainFp16 => Err(not_implemented_fp16()),
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
        CertificateVersion::PlainFp16 => Err(not_implemented_fp16()),
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
        CertificateVersion::PlainFp16 => {
            let proof = extract_fp16(plain_proof)?;
            verify_fp16_plain_proof(block_header, proof, nbits_override)
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

fn extract_fp16(plain_proof: &Bound<'_, PyAny>) -> PyResult<Fp16PlainProof> {
    plain_proof.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("certificate version 5 requires Fp16PlainProof")
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CERT_VERSION_ZK_DENSE", CertificateVersion::ZkDense as u32)?;
    m.add("CERT_VERSION_ZK_MOE", CertificateVersion::ZkMoe as u32)?;
    m.add("CERT_VERSION_ZK_V3", CertificateVersion::ZkV3 as u32)?;
    m.add(
        "CERT_VERSION_PLAIN_FP8",
        CertificateVersion::PlainFp8 as u32,
    )?;
    m.add(
        "CERT_VERSION_PLAIN_FP16",
        CertificateVersion::PlainFp16 as u32,
    )?;
    m.add_function(wrap_pyfunction!(py_check_cert_version_eligible, m)?)?;
    m.add_function(wrap_pyfunction!(generate_proof_for_cert_version, m)?)?;
    m.add_function(wrap_pyfunction!(verify_proof_for_cert_version, m)?)?;
    m.add_function(wrap_pyfunction!(verify_plain_proof_for_cert_version, m)?)?;
    Ok(())
}
