//! Python bindings for [`MerkleTree`] and [`MerkleProof`].
//!
//! Provides `#[pymethods]` implementations that extend the pyo3 classes defined in [`merkle`].

use blake3::{CHUNK_LEN, OUT_LEN};
use pyo3::{
    buffer::PyBuffer, exceptions::PyValueError, pymethods, types::PyBytes, Bound, PyResult, Python,
};

use crate::hasher::Digest;
use crate::merkle::{padded_chunk_len, MerkleProof, MerkleTree};

#[pymethods]
impl MerkleTree {
    #[new]
    #[pyo3(signature = (data, key))]
    fn py_new(data: &[u8], key: &[u8]) -> PyResult<Self> {
        let key: [u8; OUT_LEN] = key.try_into().map_err(|_| {
            PyValueError::new_err(format!(
                "key must be exactly {} bytes, got {}",
                OUT_LEN,
                key.len()
            ))
        })?;
        Ok(Self::new(data, key))
    }

    #[staticmethod]
    #[pyo3(name = "from_buffer", signature = (data, key))]
    fn py_from_buffer(py: Python<'_>, data: PyBuffer<i8>, key: Digest) -> PyResult<Self> {
        if !data.is_c_contiguous() {
            return Err(PyValueError::new_err("data must be C-contiguous"));
        }
        let (addr, len) = (data.buf_ptr() as usize, data.len_bytes());
        Ok(py.allow_threads(|| {
            let mut padded = vec![0u8; padded_chunk_len(len)];
            unsafe { std::ptr::copy_nonoverlapping(addr as *const u8, padded.as_mut_ptr(), len) };
            Self::from_vec(padded, key)
        }))
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
    fn py_compute_leaf_indices_from_rows(
        row_indices: Vec<usize>,
        shape: (usize, usize),
    ) -> Vec<usize> {
        Self::compute_leaf_indices_from_rows(&row_indices, shape)
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
        let leaf_data: Vec<[u8; CHUNK_LEN]> = leaf_data
            .into_iter()
            .map(|v| {
                v.try_into().map_err(|_| {
                    PyValueError::new_err(format!("leaf data must be exactly {} bytes", CHUNK_LEN))
                })
            })
            .collect::<PyResult<_>>()?;
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
}
