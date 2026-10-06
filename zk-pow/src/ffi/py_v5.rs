//! Python bindings for the v5 (FP16 / A100) proof types.
//!
//! The FP16 ZK prover turns a winning A100 tile (opened operand codes + the
//! public `Fp16JobParams`) into the serialized `Fp16ZkCertificate` bytes the
//! node's `CertificateV5.ProofData` carries. There is no FP16 verifier binding:
//! FP16 consensus verification is the node's job via `verify_fp16_zk_cert_ffi`
//! (the header-bound gateway), not a Python entry point.

#[cfg(feature = "pyo3")]
pub use fp16::PyFp16Prover;

/// Python `Fp16Prover` over [`crate::v5::api::zk`]. Turns a winning A100 tile
/// (opened operand codes + the public [`Fp16JobParams`]) into the serialized
/// [`Fp16ZkCertificate`] bytes the node's `CertificateV5.ProofData` carries.
///
/// There is no FP16 verifier binding here: FP16 consensus verification is the
/// node's job via the `verify_fp16_zk_cert_ffi` FFI (the header-bound gateway),
/// not a Python entry point.
///
/// [`Fp16JobParams`]: crate::v5::api::plain_proof::Fp16JobParams
/// [`Fp16ZkCertificate`]: crate::v5::api::zk_cert::Fp16ZkCertificate
#[cfg(feature = "pyo3")]
mod fp16 {
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    use crate::v5::api::plain_proof::{Fp16JobParams, Fp16PlainProof};
    use crate::v5::api::zk::Fp16Prover;
    use crate::v4::api::primitives::IncompleteBlockHeader;

    fn runtime_err(context: &str, error: impl core::fmt::Display) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(format!("{context}: {error}"))
    }

    /// Reusable FP16 prover. Retains one compiled wrapper per tile geometry
    /// (the FP16 wrapper is per-shape; see [`crate::v5::api::zk`]).
    #[pyclass(name = "Fp16Prover")]
    pub struct PyFp16Prover {
        inner: Fp16Prover,
    }

    #[pymethods]
    impl PyFp16Prover {
        /// A fresh prover; geometries compile lazily on first `prove`.
        #[new]
        fn new() -> Self {
            Self { inner: Fp16Prover::new() }
        }

        /// Eagerly compiles the wrapper for one tile geometry so a later `prove`
        /// of that shape does not pay the (minutes-long) compilation cost. `(h, w)`
        /// are the tile sizes (`P_A.tile_size()` / `P_B.tile_size()`); `hash_id`
        /// is the operand Merkle chunk id (both sides use it here).
        fn setup_geometry(&mut self, h: usize, w: usize, k: usize, hash_id: crate::v4::api::public_params::HashId) -> PyResult<()> {
            self.inner
                .setup_geometry(h, w, k, hash_id, hash_id)
                .map_err(|e| runtime_err("fp16 wrapper compilation failed", e))
        }

        /// Proves one winning tile and returns the serialized `Fp16ZkCertificate`
        /// bytes (`CertificateV5.ProofData`). `a_codes`/`b_codes` are the opened
        /// `h x k` / `w x k` FP16 operand bit patterns; `job` fixes the geometry
        /// and the noise-seed `p` encodings; `proposed_header` keys the A side.
        fn prove<'py>(
            &mut self,
            py: Python<'py>,
            proposed_header: IncompleteBlockHeader,
            job: Fp16JobParams,
            a_codes: Vec<u16>,
            b_codes: Vec<u16>,
        ) -> PyResult<Bound<'py, PyBytes>> {
            let cert = self
                .inner
                .prove(&proposed_header, &job, &a_codes, &b_codes)
                .map_err(|e| runtime_err("fp16 prove failed", e))?;
            let bytes = cert.to_bytes().map_err(|e| runtime_err("serializing fp16 certificate failed", e))?;
            Ok(PyBytes::new(py, &bytes))
        }

        /// Proves directly from the `Fp16PlainProof` opener bundle the miner
        /// already assembles for a winning tile: it authenticates + re-opens the
        /// tile codes under `proposed_header` (catching a malformed opening before
        /// the minutes-long prove), then proves. Returns the serialized
        /// `Fp16ZkCertificate` bytes (`CertificateV5.ProofData`).
        fn prove_from_plain_proof<'py>(
            &mut self,
            py: Python<'py>,
            proposed_header: IncompleteBlockHeader,
            plain_proof: Fp16PlainProof,
        ) -> PyResult<Bound<'py, PyBytes>> {
            let cert = self
                .inner
                .prove_from_plain_proof(&proposed_header, &plain_proof)
                .map_err(|e| runtime_err("fp16 prove failed", e))?;
            let bytes = cert.to_bytes().map_err(|e| runtime_err("serializing fp16 certificate failed", e))?;
            Ok(PyBytes::new(py, &bytes))
        }
    }
}
