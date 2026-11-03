//! FP16 (A100) ZK consensus certificate: the wire container the node's
//! `CertificateV5` carries once the scheme's consensus path is the recursive ZK
//! proof rather than the plaintext replay.
//!
//! This is the ZK analogue of [`crate::v5::api::plain_proof::Fp16PlainProof`].
//! Where the plaintext certificate carried the opened operand strips (so cert
//! size scaled with the tile, up to several MiB), the ZK certificate carries a
//! **constant-size** recursive proof plus the public job statement:
//!
//! ```text
//! Fp16ZkCertificate = { job: Fp16JobParams, proof_bytes: Vec<u8> }
//! ```
//!
//! * `job` — the public statement ([`Fp16JobParams`]: the proof-carried ancestor
//!   header `σ_Δ`, device, `k`, `r`, and per-side `num_rows`/`hash_id`/pattern).
//!   It fixes the tile geometry `(h, w, k)` (`h = P_A.tile_size()`,
//!   `w = P_B.tile_size()`) and feeds the noise-seed public-parameter encodings
//!   `p_A`/`p_B` ([`Fp16JobParams::encode_p_a`]/`encode_p_b`), so the verifier
//!   can header-bind the proof exactly as the plaintext path did.
//! * `proof_bytes` — the serialized stage-2 wrapped plonky2 proof
//!   (`ProofWithPublicInputs<F, OuterC, D>::to_bytes()`). It is opaque here: the
//!   committed operand roots `HASH_A`/`HASH_B`, the jackpot hash `HASH_JACKPOT`,
//!   and the statement digest all live inside the proof's public inputs, and are
//!   recovered + header-bound by the verifier
//!   ([`crate::v5::circuit::wrapper::verify_wrapped_proof_with_headers`]).
//!   Deserialization needs the wrapper's `CommonCircuitData`, which the verifier
//!   reconstructs from `job`'s geometry, so this container keeps the proof as
//!   bytes (the FP8 `proof_blob` discipline).
//!
//! The codec is canonical fixint bincode and rejects trailing bytes, matching
//! [`Fp16PlainProof`].

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::plain_proof::Fp16JobParams;

/// The FP16 (A100) ZK consensus certificate (see module docs).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "Fp16ZkCertificate"))]
pub struct Fp16ZkCertificate {
    /// The public job statement (fixes geometry + the noise-seed `p` encodings).
    pub job: Fp16JobParams,
    /// The serialized stage-2 wrapped recursive proof
    /// (`ProofWithPublicInputs::to_bytes()`); opaque to this container.
    pub proof_bytes: Vec<u8>,
}

impl Fp16ZkCertificate {
    /// Strict fixint bincode (matches [`Fp16PlainProof::to_bytes`]).
    ///
    /// [`Fp16PlainProof::to_bytes`]: super::plain_proof::Fp16PlainProof::to_bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        use bincode::Options;
        bincode::options()
            .with_fixint_encoding()
            .serialize(self)
            .map_err(|e| anyhow::anyhow!("serialize Fp16ZkCertificate: {e}"))
    }

    /// Inverse of [`Self::to_bytes`]. No compat ladder; rejects trailing bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        use bincode::Options;
        bincode::options()
            .with_fixint_encoding()
            .reject_trailing_bytes()
            .deserialize(bytes)
            .map_err(|e| anyhow::anyhow!("deserialize Fp16ZkCertificate: {e}"))
    }

    /// The tile geometry `(h, w, k)` the statement fixes — what the consensus
    /// verifier rebuilds the `Fp16System`/wrapper circuit for.
    pub fn tile_geometry(&self) -> (usize, usize, usize) {
        (
            self.job.operands.a.pattern.tile_size() as usize,
            self.job.operands.b.pattern.tile_size() as usize,
            self.job.k as usize,
        )
    }
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl Fp16ZkCertificate {
    #[new]
    fn py_new(job: Fp16JobParams, proof_bytes: Vec<u8>) -> Self {
        Self { job, proof_bytes }
    }

    #[getter]
    fn job(&self) -> Fp16JobParams {
        self.job.clone()
    }

    #[getter]
    fn proof_bytes<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, &self.proof_bytes)
    }

    #[getter]
    fn min_cert_version(&self) -> u32 {
        crate::ffi::CertificateVersion::PlainFp16 as u32
    }

    #[pyo3(name = "to_bytes")]
    fn py_to_bytes(&self) -> pyo3::PyResult<Vec<u8>> {
        Self::to_bytes(self).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    #[pyo3(name = "from_bytes")]
    fn py_from_bytes(data: Vec<u8>) -> pyo3::PyResult<Self> {
        Self::from_bytes(&data).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v5::api::params::Fp16Device;
    use crate::v5::api::plain_proof::{Fp16JobParams, Fp16OperandParams};
    use crate::v4::api::public_params::HashId;
    use crate::v4::api::layout::AxisPattern;
    use crate::v4::api::layout::DimType::{Blake, Fold};
    use crate::v4::api::primitives::{IncompleteBlockHeader, Sides};

    fn sample() -> Fp16ZkCertificate {
        let rp = AxisPattern::new(&[(4, Blake)]).unwrap(); // h = 4
        let cp = AxisPattern::new(&[(4, Blake), (16, Fold)]).unwrap(); // w = 64
        Fp16ZkCertificate {
            job: Fp16JobParams {
                ancestor_header: IncompleteBlockHeader::new_for_test(0x207f_ffff),
                device: Fp16Device::A100,
                k: 256,
                r: 32,
                operands: Sides {
                    a: Fp16OperandParams { num_rows: 8, hash_id: HashId::Blake3Chunk1024, pattern: rp },
                    b: Fp16OperandParams { num_rows: 128, hash_id: HashId::Blake3Chunk1024, pattern: cp },
                },
            },
            proof_bytes: (0u16..5000).flat_map(|x| x.to_le_bytes()).collect(),
        }
    }

    #[test]
    fn roundtrips_and_rejects_trailing_bytes() {
        let cert = sample();
        let bytes = cert.to_bytes().expect("serialize");
        let back = Fp16ZkCertificate::from_bytes(&bytes).expect("deserialize");
        assert_eq!(back.job, cert.job);
        assert_eq!(back.proof_bytes, cert.proof_bytes);
        assert_eq!(back.tile_geometry(), (4, 64, 256));

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(Fp16ZkCertificate::from_bytes(&trailing).is_err(), "a trailing byte must be rejected");
    }

    #[test]
    fn tile_geometry_matches_patterns() {
        assert_eq!(sample().tile_geometry(), (4, 64, 256));
    }
}
