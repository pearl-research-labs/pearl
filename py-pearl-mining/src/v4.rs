//! V4 (FP8) functions. The prover/verifier classes live in `zk_pow::ffi::py_v4`.

use pyo3::prelude::*;

use zk_pow::ffi::py_v4::{PyFp8Prover, PyFp8Verifier};
use zk_pow::v4::api::layout::{AxisPattern, DimType};
use zk_pow::v4::api::plain_proof::{MoeWitness, PlainProofV4};
use zk_pow::v4::api::primitives::BlockHeader;
use zk_pow::v4::api::public_params::{CommonParams, Device, HashId, MoeParams, OperandParams, Quant};
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

pub fn register_header(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BlockHeader>()?;
    Ok(())
}

pub fn register_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PlainProofV4>()?;
    m.add_class::<MoeWitness>()?;
    m.add_class::<CommonParams>()?;
    m.add_class::<OperandParams>()?;
    m.add_class::<MoeParams>()?;
    m.add_class::<HashId>()?;
    m.add_class::<Quant>()?;
    m.add_class::<DimType>()?;
    m.add_class::<AxisPattern>()?;
    m.add_class::<Device>()?;
    Ok(())
}

pub fn register_proofs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(verify_plain_proof_v4, m)?)?;
    m.add_class::<PyFp8Prover>()?;
    m.add_class::<PyFp8Verifier>()?;
    Ok(())
}
