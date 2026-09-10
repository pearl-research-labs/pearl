//! Python bindings for [`MerkleTree`] and [`MerkleProof`].
//!
//! Provides `#[pymethods]` implementations that extend the pyo3 classes defined in [`merkle`].

use blake3::{CHUNK_LEN, OUT_LEN};
use pyo3::{exceptions::PyValueError, pymethods, types::PyBytes, Bound, PyResult, Python};

use crate::hasher::Digest;
use crate::merkle::{MerkleProof, MerkleTree};

#[pymethods]
impl MerkleTree {
    #[new]
    #[pyo3(signature = (data, key, chunk_len = CHUNK_LEN))]
    fn py_new(data: &[u8], key: &[u8], chunk_len: usize) -> PyResult<Self> {
        let key: [u8; OUT_LEN] = key.try_into().map_err(|_| {
            PyValueError::new_err(format!(
                "key must be exactly {} bytes, got {}",
                OUT_LEN,
                key.len()
            ))
        })?;
        Self::with_chunk_len(data, key, chunk_len).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter(root)]
    fn py_root<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.root())
    }

    #[getter(leaf_hashes)]
    fn py_leaf_hashes<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyBytes>> {
        self.leaf_hashes()
            .iter()
            .map(|h| PyBytes::new(py, h))
            .collect()
    }

    #[pyo3(name = "get_multileaf_proof")]
    fn py_get_multileaf_proof(&self, leaf_indices: Vec<usize>) -> MerkleProof {
        self.get_multileaf_proof(&leaf_indices)
    }

    #[staticmethod]
    #[pyo3(name = "compute_leaf_indices_from_rows")]
    #[pyo3(signature = (row_indices, shape, chunk_len = CHUNK_LEN))]
    fn py_compute_leaf_indices_from_rows(
        row_indices: Vec<usize>,
        shape: (usize, usize),
        chunk_len: usize,
    ) -> PyResult<Vec<usize>> {
        Self::compute_leaf_indices_from_rows(&row_indices, shape, chunk_len)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pymethods]
impl MerkleProof {
    #[new]
    fn py_new(
        leaf_data: Vec<Vec<u8>>,
        leaf_indices: Vec<usize>,
        root: &[u8],
        siblings: Vec<Vec<u8>>,
        total_leaves: usize,
    ) -> PyResult<Self> {
        Self::check_equal_allowed_leaves(&leaf_data).map_err(PyValueError::new_err)?;
        let root: Digest = root.try_into().map_err(|_| {
            PyValueError::new_err(format!("root must be exactly {} bytes", OUT_LEN))
        })?;
        let siblings: Vec<Digest> = siblings
            .into_iter()
            .map(|v| {
                v.try_into().map_err(|_| {
                    PyValueError::new_err(format!(
                        "siblings entries must be exactly {} bytes",
                        OUT_LEN
                    ))
                })
            })
            .collect::<PyResult<_>>()?;
        Ok(Self {
            leaf_data,
            leaf_indices,
            total_leaves,
            root,
            siblings,
        })
    }

    #[getter(leaf_data)]
    fn py_leaf_data(&self) -> Vec<Vec<u8>> {
        self.leaf_data.clone()
    }

    #[getter(leaf_indices)]
    fn py_leaf_indices(&self) -> Vec<usize> {
        self.leaf_indices.clone()
    }

    #[getter(total_leaves)]
    fn py_total_leaves(&self) -> usize {
        self.total_leaves
    }

    #[getter(root)]
    fn py_root<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.root)
    }

    #[getter(siblings)]
    fn py_siblings<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyBytes>> {
        self.siblings
            .iter()
            .map(|sibling| PyBytes::new(py, sibling))
            .collect()
    }

    #[getter(chunk_len)]
    fn py_chunk_len(&self) -> usize {
        self.chunk_len()
    }

    #[pyo3(name = "verify")]
    fn py_verify(&self, key: &[u8]) -> PyResult<bool> {
        let key: Digest = key
            .try_into()
            .map_err(|_| PyValueError::new_err(format!("key must be exactly {} bytes", OUT_LEN)))?;
        Ok(self.verify(key))
    }

    #[pyo3(name = "extract_bytes")]
    fn py_extract_bytes(&self, global_start: usize, length: usize) -> PyResult<Vec<u8>> {
        self.extract_bytes(global_start, length)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}
