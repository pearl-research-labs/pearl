//! Python bindings for core proof types.

#[cfg(feature = "pyo3")]
use crate::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, MoEConfig, PeriodicPattern};

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
impl IncompleteBlockHeader {
    /// Size of serialized IncompleteBlockHeader in bytes.
    #[classattr]
    #[pyo3(name = "SERIALIZED_SIZE")]
    fn py_serialized_size() -> usize {
        Self::SERIALIZED_SIZE
    }

    #[new]
    fn py_new(version: u32, prev_block: Vec<u8>, merkle_root: Vec<u8>, timestamp: u32, nbits: u32) -> pyo3::PyResult<Self> {
        if prev_block.len() != 32 || merkle_root.len() != 32 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "prev_block and merkle_root must be 32 bytes",
            ));
        }
        Ok(Self {
            version,
            prev_block: prev_block.try_into().unwrap(),
            merkle_root: merkle_root.try_into().unwrap(),
            timestamp,
            nbits,
        })
    }

    /// Format: version(4) | prev_block(32, reversed) | merkle_root(32, reversed) | timestamp(4) | nbits(4)
    #[pyo3(name = "to_bytes")]
    fn py_to_bytes(&self) -> Vec<u8> {
        IncompleteBlockHeader::to_bytes(self).to_vec()
    }

    #[staticmethod]
    #[pyo3(name = "from_bytes")]
    fn py_from_bytes(data: Vec<u8>) -> pyo3::PyResult<Self> {
        IncompleteBlockHeader::from_bytes(&data).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
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

    /// Serialize to bytes (44 bytes).
    #[pyo3(name = "to_bytes")]
    fn py_to_bytes(&self) -> Vec<u8> {
        MiningConfiguration::to_bytes(self).to_vec()
    }

    /// Deserialize from bytes (44 bytes).
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
