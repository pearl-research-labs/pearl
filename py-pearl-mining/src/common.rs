//! Python-facing pieces shared by every version module.

use blake3::CHUNK_LEN;
use pearl_blake3::{MerkleProof, MerkleTree, pad_to_chunk_boundary};
use pyo3::prelude::*;
use zk_pow::v2::api::proof::PublicProofParams;

/// The Python-facing block header every version's functions take; the v1/v2 modules
/// convert it into their own identical struct.
pub use zk_pow::v4::api::primitives::IncompleteBlockHeader;

pub fn py_err(msg: &str, e: impl std::fmt::Display) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}: {}", msg, e))
}

pub fn value_err(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

// ============================================================================
// ZK Proof (only type defined in the binding crate)
// ============================================================================

#[pyclass(name = "ZKProof", get_all)]
#[derive(Clone)]
pub struct PyProof {
    pub public_data: Vec<u8>,
    pub proof_data: Vec<u8>,
}

#[pymethods]
impl PyProof {
    #[new]
    fn new(public_data: Vec<u8>, proof_data: Vec<u8>) -> PyResult<Self> {
        if !PublicProofParams::is_valid_wire_size(public_data.len()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "public_data length must be {} bytes (non-MoE) or {}..={} (MoE)",
                PublicProofParams::WIRE_SIZE,
                PublicProofParams::MIN_MOE_WIRE_SIZE,
                PublicProofParams::MAX_WIRE_SIZE,
            )));
        }
        Ok(Self {
            public_data,
            proof_data,
        })
    }
}

#[pyfunction]
#[pyo3(name = "pad_to_chunk_boundary")]
pub fn py_pad_to_chunk_boundary(data: &[u8]) -> Vec<u8> {
    pad_to_chunk_boundary(data)
}

pub fn register_constants(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("MERKLE_LEAF_SIZE", CHUNK_LEN)?;
    Ok(())
}

pub fn register_merkle_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MerkleTree>()?;
    m.add_class::<MerkleProof>()?;
    Ok(())
}

pub fn register_header(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IncompleteBlockHeader>()?;
    Ok(())
}

pub fn register_proof_type(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProof>()?;
    Ok(())
}

pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_pad_to_chunk_boundary, m)?)?;
    Ok(())
}
