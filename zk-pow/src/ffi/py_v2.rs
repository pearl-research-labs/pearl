//! Python bindings for the v2 (Int7) proof types: `PeriodicPattern` / `MiningConfiguration` /
//! `MoEConfig` / `MMAType`, and the plain-proof witness `PlainProof` with its
//! `MatrixMerkleProof` / `MoEProofParams` parts.

#[cfg(feature = "pyo3")]
use pearl_blake3::MerkleProof;

#[cfg(feature = "pyo3")]
use crate::v2::api::proof::{MMAType, MiningConfiguration, MoEConfig, PeriodicPattern};
#[cfg(feature = "pyo3")]
use crate::v2::ffi::plain_proof::{MatrixMerkleProof, MoEProofParams, PlainProof};

// =============================================================================
// Python bindings (constructors for core types with #[pyclass] attribute)
// =============================================================================

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl PeriodicPattern {
    #[classattr]
    #[pyo3(name = "NUM_DIMS")]
    fn get_max_dims() -> usize {
        3
    }

    #[new]
    fn py_new(shape: Vec<(u32, u32)>) -> pyo3::PyResult<Self> {
        if shape.len() != Self::NUM_DIMS {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "shape must have exactly {} elements",
                Self::NUM_DIMS
            )));
        }
        let shape_arr: [(u32, u32); 3] = shape
            .try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("shape conversion failed"))?;
        Ok(Self { shape: shape_arr })
    }

    #[staticmethod]
    #[pyo3(name = "from_bytes")]
    fn py_from_bytes(data: Vec<u8>) -> pyo3::PyResult<Self> {
        PeriodicPattern::from_bytes(&data).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    #[pyo3(name = "from_list")]
    fn py_from_list(pattern: Vec<u32>) -> pyo3::PyResult<Self> {
        PeriodicPattern::from_list(&pattern).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[pyo3(name = "to_bytes")]
    fn py_to_bytes(&self) -> Vec<u8> {
        PeriodicPattern::to_bytes(self).to_vec()
    }

    #[pyo3(name = "to_list")]
    fn py_to_list(&self) -> Vec<u32> {
        PeriodicPattern::to_list(self)
    }

    #[pyo3(name = "offset_is_valid")]
    fn py_offset_is_valid(&self, offset: u32) -> bool {
        PeriodicPattern::offset_is_valid(self, offset)
    }

    #[pyo3(name = "is_valid")]
    fn py_is_valid(&self) -> bool {
        PeriodicPattern::is_valid(self)
    }

    #[getter]
    fn get_period(&self) -> u32 {
        PeriodicPattern::period(self)
    }

    #[getter]
    fn get_size(&self) -> u32 {
        PeriodicPattern::size(self)
    }

    #[getter]
    fn get_shape(&self) -> Vec<(u32, u32)> {
        self.shape.to_vec()
    }
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl MoEConfig {
    #[new]
    fn py_new(e: u16, top_k: u16) -> Self {
        Self { e, top_k }
    }
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl MiningConfiguration {
    /// Size of serialized MiningConfiguration in bytes.
    #[classattr]
    #[pyo3(name = "SERIALIZED_SIZE")]
    fn py_serialized_size() -> usize {
        Self::SERIALIZED_SIZE
    }

    /// Construct a mining configuration. Pass `moe=None` for a standard job, or a
    /// `MoEConfig` to select GROUPED_GEMM mode (committed in the job_key).
    #[new]
    #[pyo3(signature = (common_dim, rank, mma_type, rows_pattern, cols_pattern, moe=None))]
    fn py_new(
        common_dim: u32,
        rank: u16,
        mma_type: MMAType,
        rows_pattern: PeriodicPattern,
        cols_pattern: PeriodicPattern,
        moe: Option<MoEConfig>,
    ) -> pyo3::PyResult<Self> {
        Ok(Self {
            common_dim,
            rank,
            mma_type,
            rows_pattern,
            cols_pattern,
            moe,
        })
    }

    /// Serialize to bytes (52 bytes).
    #[pyo3(name = "to_bytes")]
    fn py_to_bytes(&self) -> Vec<u8> {
        MiningConfiguration::to_bytes(self).to_vec()
    }

    /// Deserialize from bytes (52 bytes).
    #[staticmethod]
    #[pyo3(name = "from_bytes")]
    fn py_from_bytes(data: Vec<u8>) -> pyo3::PyResult<Self> {
        MiningConfiguration::from_bytes(&data).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Height of the hash tile (number of rows in the pattern).
    #[getter]
    fn get_hash_tile_h(&self) -> u32 {
        self.rows_pattern.size()
    }

    /// Width of the hash tile (number of columns in the pattern).
    #[getter]
    fn get_hash_tile_w(&self) -> u32 {
        self.cols_pattern.size()
    }

    #[getter]
    fn get_rounded_common_dim(&self) -> u32 {
        self.dot_product_length() as u32
    }
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl MMAType {
    /// Returns the torch dtype name for this MMA type.
    #[getter]
    fn get_tensor_dtype(&self) -> &'static str {
        match self {
            MMAType::Int7xInt7ToInt32 => "int8",
        }
    }
}

// =============================================================================
// Plain-proof witness bindings
// =============================================================================

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl MoEProofParams {
    #[new]
    fn py_new(
        e: usize,
        top_k: usize,
        expert_idx: u16,
        routing_end_offsets: Vec<u32>,
        inner_a_rows: Vec<usize>,
        routing_proof: MerkleProof,
    ) -> Self {
        Self {
            e,
            top_k,
            expert_idx,
            routing_end_offsets,
            inner_a_rows,
            routing_proof,
        }
    }
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl MatrixMerkleProof {
    #[new]
    fn py_new(proof: &MerkleProof, row_indices: Vec<usize>) -> Self {
        Self {
            proof: proof.clone(),
            row_indices,
        }
    }

    #[getter]
    fn row_indices(&self) -> Vec<usize> {
        self.row_indices.clone()
    }

    #[getter]
    fn root<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, &self.proof.root)
    }
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl PlainProof {
    #[new]
    #[pyo3(signature = (m, n, k, noise_rank, a_merkle_proof, bt_merkle_proof, moe=None))]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        m: usize,
        n: usize,
        k: usize,
        noise_rank: usize,
        a_merkle_proof: MatrixMerkleProof,
        bt_merkle_proof: MatrixMerkleProof,
        moe: Option<MoEProofParams>,
    ) -> Self {
        Self {
            m,
            n,
            k,
            noise_rank,
            a: a_merkle_proof,
            bt: bt_merkle_proof,
            moe,
        }
    }

    /// The lowest block certificate version that can certify this proof.
    #[getter(min_cert_version)]
    fn py_min_cert_version(&self) -> u32 {
        self.min_cert_version() as u32
    }

    fn to_base64(&self) -> pyo3::PyResult<String> {
        use base64::{Engine as _, engine::general_purpose::STANDARD};
        let bytes = bincode::serialize(self)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Serialization failed: {}", e)))?;
        Ok(STANDARD.encode(bytes))
    }

    #[staticmethod]
    fn from_base64(data: &str) -> pyo3::PyResult<Self> {
        use base64::{Engine as _, engine::general_purpose::STANDARD};
        let bytes = STANDARD
            .decode(data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Base64 decode failed: {}", e)))?;
        Self::deserialize_compat(&bytes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Deserialization failed: {}", e)))
    }
}
