//! Python bindings for core proof types.
//!
//! Int7 types: `PeriodicPattern` / `MiningConfiguration` / `MoEConfig` / `MMAType`.
//! FP8 witnesses cross as [`crate::api::fp8::plain_proof::PlainProofV4`].
//! `IncompleteBlockHeader` is shared. The node-side cache
//! ([`crate::api::fp8::zk::Fp8VerifierCache`]) is not bound: Python is the
//! prover/miner surface; the Go node uses its own embedded cache.

#[cfg(feature = "pyo3")]
use crate::api::primitives::IncompleteBlockHeader;
#[cfg(feature = "pyo3")]
use crate::v2::api::proof::{MMAType, MiningConfiguration, MoEConfig, PeriodicPattern};

#[cfg(feature = "pyo3")]
pub use fp8::{PyFp8Prover, PyFp8Verifier};

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
// FP8 ZK prove/verify bindings
// =============================================================================

/// Python `Fp8Prover` / `Fp8Verifier` over [`crate::api::fp8::zk`].
/// Witnesses are [`PlainProofV4`].
#[cfg(feature = "pyo3")]
mod fp8 {
    use plonky2::util::timing::TimingTree;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    use crate::api::fp8::plain_proof::PlainProofV4;
    use crate::api::fp8::zk::{Fp8Prover, Fp8Verifier, decode_statement};
    use crate::api::primitives::IncompleteBlockHeader;

    fn value_err(error: impl core::fmt::Display) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(error.to_string())
    }

    fn runtime_err(context: &str, error: impl core::fmt::Display) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(format!("{context}: {error}"))
    }

    /// Shape-specific fp8 prover setup (LUT precommitment + compiled
    /// recursive-wrapper circuits). Create via [`PyFp8Prover::setup`].
    #[pyclass(name = "Fp8Prover")]
    pub struct PyFp8Prover {
        inner: Fp8Prover,
    }

    #[pymethods]
    impl PyFp8Prover {
        /// Builds prover data for the job shape of `plain_proof` (an FP8
        /// witness). The `ancestor_header` carried inside the proof's job
        /// keys the B side; `block_header` (σ̂) keys the A side.
        #[staticmethod]
        fn setup(block_header: IncompleteBlockHeader, plain_proof: PlainProofV4) -> PyResult<PyFp8Prover> {
            let prover = Fp8Prover::setup(&block_header, &plain_proof).map_err(|e| runtime_err("fp8 setup failed", e))?;
            Ok(Self { inner: prover })
        }

        /// Proves one block and returns the published `(public_data,
        /// proof_data)` byte pair.
        fn prove<'py>(
            &mut self,
            py: Python<'py>,
            block_header: IncompleteBlockHeader,
            plain_proof: PlainProofV4,
        ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
            let (public_data, proof_data) = self
                .inner
                .prove(&block_header, &plain_proof)
                .map_err(|e| runtime_err("fp8 prove failed", e))?;
            Ok((PyBytes::new(py, &public_data), PyBytes::new(py, &proof_data)))
        }
    }

    /// Trusted fp8 verifier setup (the universal wrapper covers every
    /// envelope-legal geometry and degree profile).
    /// Obtain from [`PyFp8Verifier::generate`] or [`PyFp8Verifier::from_bytes`].
    #[pyclass(name = "Fp8Verifier")]
    pub struct PyFp8Verifier {
        inner: Fp8Verifier,
    }

    #[pymethods]
    impl PyFp8Verifier {
        /// Generates the trusted setup for a published statement
        /// (`public_data`): reads the committed LUT cap and compiles the
        /// universal wrapper circuits. Expensive — once per deployment.
        #[staticmethod]
        fn generate(public_data: &[u8]) -> PyResult<Self> {
            let statement = decode_statement(public_data).map_err(value_err)?;
            // The setup is job-independent, so the statement's own `ancestor_header`
            // anchors the job derivation (the committed cache is built from the
            // canonical `sample_dense_statement` dummy of the same shape).
            Fp8Verifier::generate(&statement, statement.ancestor_header(), &mut TimingTree::default())
                .map(|verifier| Self { inner: verifier })
                .map_err(|e| runtime_err("fp8 verifier setup generation failed", e))
        }

        /// Verifies a published `(public_data, proof_data)` pair against the
        /// expected block header. Raises on rejection.
        fn verify_block(&self, block_header: IncompleteBlockHeader, public_data: &[u8], proof_data: &[u8]) -> PyResult<()> {
            self.inner
                .verify_block(&block_header, public_data, proof_data)
                .map_err(|e| runtime_err("fp8 verification rejected the proof", e))
        }

        /// Verifies a pool share under an explicit share target (`share_nbits`).
        fn verify_share(
            &self,
            block_header: IncompleteBlockHeader,
            public_data: &[u8],
            proof_data: &[u8],
            share_nbits: u32,
        ) -> PyResult<()> {
            self.inner
                .verify_share(&block_header, public_data, proof_data, share_nbits)
                .map_err(|e| runtime_err("fp8 verification rejected the share", e))
        }

        /// Serializes the trusted verifier setup for distribution.
        fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            let bytes = self
                .inner
                .to_bytes()
                .map_err(|e| runtime_err("serializing fp8 verifier setup failed", e))?;
            Ok(PyBytes::new(py, &bytes))
        }

        /// Loads verifier setup previously produced by `to_bytes`. The bytes
        /// are consensus/trusted-setup data, not proof-controlled input.
        #[staticmethod]
        fn from_bytes(data: Vec<u8>) -> PyResult<Self> {
            Fp8Verifier::from_bytes(&data)
                .map(|verifier| Self { inner: verifier })
                .map_err(value_err)
        }
    }
}
