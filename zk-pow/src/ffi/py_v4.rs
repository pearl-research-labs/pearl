//! Python bindings for the v4 (FP8) proof types.
//!
//! FP8 witnesses cross as [`crate::v4::api::plain_proof::PlainProofV4`].
//! `IncompleteBlockHeader` and the complete `BlockHeader` are shared. The node-side cache
//! ([`crate::v4::api::zk::Fp8VerifierCache`]) is not bound: Python is the
//! prover/miner surface; the Go node uses its own embedded cache.

#[cfg(feature = "pyo3")]
use crate::v4::api::primitives::{BlockHeader, IncompleteBlockHeader};

#[cfg(feature = "pyo3")]
pub use fp8::{PyFp8Prover, PyFp8Verifier};

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
impl BlockHeader {
    /// Size of a serialized (wire) BlockHeader in bytes.
    #[classattr]
    #[pyo3(name = "SERIALIZED_SIZE")]
    fn py_serialized_size() -> usize {
        Self::SERIALIZED_SIZE
    }

    #[new]
    fn py_new(incomplete: IncompleteBlockHeader, proof_commitment: Vec<u8>) -> pyo3::PyResult<Self> {
        let proof_commitment = proof_commitment
            .try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("proof_commitment must be 32 bytes"))?;
        Ok(Self {
            incomplete,
            proof_commitment,
        })
    }

    /// Format: IncompleteBlockHeader wire bytes(76) | proof_commitment(32, as stored)
    #[pyo3(name = "to_bytes")]
    fn py_to_bytes(&self) -> Vec<u8> {
        BlockHeader::to_bytes(self).to_vec()
    }

    #[staticmethod]
    #[pyo3(name = "from_bytes")]
    fn py_from_bytes(data: Vec<u8>) -> pyo3::PyResult<Self> {
        BlockHeader::from_bytes(&data).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// SHA256d of the wire header, in `prev_block` (display) byte order.
    #[pyo3(name = "block_hash")]
    fn py_block_hash(&self) -> Vec<u8> {
        BlockHeader::block_hash(self).to_vec()
    }
}

// =============================================================================
// FP8 ZK prove/verify bindings
// =============================================================================

/// Python `Fp8Prover` / `Fp8Verifier` over [`crate::v4::api::zk`].
/// Witnesses are [`PlainProofV4`].
#[cfg(feature = "pyo3")]
mod fp8 {
    use plonky2::util::timing::TimingTree;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    use crate::v4::api::plain_proof::PlainProofV4;
    use crate::v4::api::primitives::{BlockHeader, IncompleteBlockHeader};
    use crate::v4::api::public_params::Device;
    use crate::v4::api::zk::{Fp8Prover, Fp8Verifier, decode_statement};

    fn value_err(error: impl core::fmt::Display) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(error.to_string())
    }

    fn runtime_err(context: &str, error: impl core::fmt::Display) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(format!("{context}: {error}"))
    }

    /// Reusable fp8 prover setup (LUT precommitments + compiled recursive
    /// wrapper circuits). Create via [`PyFp8Prover::setup`].
    #[pyclass(name = "Fp8Prover")]
    pub struct PyFp8Prover {
        inner: Fp8Prover,
    }

    #[pymethods]
    impl PyFp8Prover {
        /// Precompiles one device's prover setup. Another device is initialized
        /// lazily if a later proof requires it.
        #[staticmethod]
        fn setup(device: Device) -> PyResult<PyFp8Prover> {
            let prover = Fp8Prover::setup(device).map_err(|e| runtime_err("fp8 setup failed", e))?;
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
            // The setup is job-independent; a zero header suffices for derivation.
            Fp8Verifier::generate(&statement, &IncompleteBlockHeader::zero(), &mut TimingTree::default())
                .map(|verifier| Self { inner: verifier })
                .map_err(|e| runtime_err("fp8 verifier setup generation failed", e))
        }

        /// Verifies a published `(public_data, proof_data)` pair against the
        /// expected block header; `ancestor_chain` holds the headers strictly
        /// between it and the statement's `σ_d`, parent first. Raises on rejection.
        fn verify_block(
            &self,
            block_header: IncompleteBlockHeader,
            ancestor_chain: Vec<BlockHeader>,
            public_data: &[u8],
            proof_data: &[u8],
        ) -> PyResult<()> {
            self.inner
                .verify_block(&block_header, &ancestor_chain, public_data, proof_data)
                .map_err(|e| runtime_err("fp8 verification rejected the proof", e))
        }

        /// Verifies a pool share under an explicit share target (`share_nbits`).
        fn verify_share(
            &self,
            block_header: IncompleteBlockHeader,
            ancestor_chain: Vec<BlockHeader>,
            public_data: &[u8],
            proof_data: &[u8],
            share_nbits: u32,
        ) -> PyResult<()> {
            self.inner
                .verify_share(&block_header, &ancestor_chain, public_data, proof_data, share_nbits)
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
