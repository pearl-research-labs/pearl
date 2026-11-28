//! Python bindings for [`MerkleTree`] and [`MerkleProof`], and the tree's
//! keyed chunk and parent compressions.
//!
//! Provides `#[pymethods]` implementations that extend the pyo3 classes defined in [`merkle`],
//! and [`chunk_cvs`], [`parent_cvs`] and [`root_cv`]: a caller that holds only part of a
//! tree's leaves (an expert-parallel miner's slice of B) rebuilds and opens its nodes with
//! them, exactly as [`MerkleTree`] hashes them.

use blake3::{CHUNK_LEN, OUT_LEN};
use pyo3::{
    exceptions::PyValueError, pyfunction, pymethods, types::PyBytes, Bound, PyResult, Python,
};

use crate::hasher::{Blake3Hasher, Digest};
use crate::merkle::{subtree_cvs, MerkleProof, MerkleTree};

fn parse_key(key: &[u8]) -> PyResult<Digest> {
    key.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "key must be exactly {} bytes, got {}",
            OUT_LEN,
            key.len()
        ))
    })
}

fn parse_digests(bytes: &[u8], name: &str) -> PyResult<Vec<Digest>> {
    if !bytes.len().is_multiple_of(OUT_LEN) {
        return Err(PyValueError::new_err(format!(
            "{name} ({} bytes) must be whole {OUT_LEN}-byte digests",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(OUT_LEN)
        .map(|digest| digest.try_into().expect("chunks_exact yields digests"))
        .collect())
}

/// The keyed chaining values of the consecutive `chunk_len`-byte leaves `data`,
/// the first being leaf `first_leaf` of its tree (leaf `i` hashes at chunk
/// counter `i`), concatenated: the tree's leaf nodes, as [`MerkleTree`] hashes them.
#[pyfunction]
#[cfg_attr(not(feature = "extension-module"), allow(dead_code))]
#[pyo3(signature = (data, key, first_leaf, chunk_len = CHUNK_LEN))]
pub(crate) fn chunk_cvs<'py>(
    py: Python<'py>,
    data: &[u8],
    key: &[u8],
    first_leaf: u64,
    chunk_len: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let key = parse_key(key)?;
    let cvs = py
        .allow_threads(|| subtree_cvs(data, key, first_leaf, chunk_len, 1))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, cvs.as_flattened()))
}

/// The keyed non-root parents of the concatenated child chaining values
/// `left[i]` and `right[i]`, concatenated: one tree layer up.
#[pyfunction]
#[cfg_attr(not(feature = "extension-module"), allow(dead_code))]
pub(crate) fn parent_cvs<'py>(
    py: Python<'py>,
    left: &[u8],
    right: &[u8],
    key: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let hasher = Blake3Hasher::with_key(parse_key(key)?);
    let (left, right) = (parse_digests(left, "left")?, parse_digests(right, "right")?);
    if left.len() != right.len() {
        return Err(PyValueError::new_err(format!(
            "left holds {} digests, right {}",
            left.len(),
            right.len()
        )));
    }
    let parents: Vec<Digest> = left
        .iter()
        .zip(&right)
        .map(|(l, r)| hasher.parent_cv(l, r))
        .collect();
    Ok(PyBytes::new(py, parents.as_flattened()))
}

/// The keyed root of the tree whose top two nodes are `left` and `right`
/// (each one 32-byte chaining value): [`MerkleTree`]'s ROOT-flagged root.
#[pyfunction]
#[cfg_attr(not(feature = "extension-module"), allow(dead_code))]
pub(crate) fn root_cv<'py>(
    py: Python<'py>,
    left: &[u8],
    right: &[u8],
    key: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let hasher = Blake3Hasher::with_key(parse_key(key)?);
    let one = |bytes: &[u8], name: &str| -> PyResult<Digest> {
        bytes.try_into().map_err(|_| {
            PyValueError::new_err(format!(
                "{name} must be exactly {OUT_LEN} bytes, got {}",
                bytes.len()
            ))
        })
    };
    Ok(PyBytes::new(
        py,
        &hasher.root_cv(&one(left, "left")?, &one(right, "right")?),
    ))
}

#[pymethods]
impl MerkleTree {
    #[new]
    #[pyo3(signature = (data, key, chunk_len = CHUNK_LEN))]
    fn py_new(data: &[u8], key: &[u8], chunk_len: usize) -> PyResult<Self> {
        let key = parse_key(key)?;
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
