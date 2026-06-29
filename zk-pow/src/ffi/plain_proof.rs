//! Plain proof structures and parsing for FFI.
//!
//! This module contains the core data structures (`PlainProof`, `MatrixMerkleProof`)
//! and `PlainProof::parse_proof`, used to convert plain proofs to ZK proof parameters.

use anyhow::{Context, Result, bail, ensure};
use blake3::BLOCK_LEN;
use pearl_blake3::MerkleProof;
use serde::{Deserialize, Serialize};

use crate::api::proof::{
    IncompleteBlockHeader, MMAType, MiningConfiguration, MoEConfig, MoEParams, PeriodicPattern, PrivateProofParams,
    PublicProofParams,
};
use crate::circuit::chip::blake3::program::{AuxiliaryCvLocation, AuxiliaryMsgLocation, ProofSource, routing_blake_hotspot_rows};
use crate::circuit::utils::macros::ensure_eq;
use pearl_blake3::{BLAKE3_CHUNK_LEN, BLAKE3_DIGEST_SIZE};

/// Merkle proof data for a single matrix.
#[derive(Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "MatrixMerkleProof"))]
pub struct MatrixMerkleProof {
    pub proof: MerkleProof,
    pub row_indices: Vec<usize>,
}

impl std::fmt::Debug for MatrixMerkleProof {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatrixMerkleProof")
            .field("leaf_indices", &self.proof.leaf_indices)
            .field("row_indices", &self.row_indices)
            .field("root", &self.proof.root)
            .field("leaf_count", &self.proof.leaf_data.len())
            .field("sibling_count", &self.proof.siblings.len())
            .finish()
    }
}

impl MatrixMerkleProof {
    /// Construct from merkle proof components.
    pub fn new(
        leaf_data: Vec<[u8; BLAKE3_CHUNK_LEN]>,
        leaf_indices: Vec<usize>,
        row_indices: Vec<usize>,
        total_leaves: usize,
        root: [u8; BLAKE3_DIGEST_SIZE],
        siblings: Vec<[u8; BLAKE3_DIGEST_SIZE]>,
    ) -> Self {
        Self {
            proof: MerkleProof {
                leaf_data,
                leaf_indices,
                total_leaves,
                root,
                siblings,
            },
            row_indices,
        }
    }

    /// Returns the merkle indices and leaves for internal processing.
    pub fn data(&self) -> (&[usize], &[[u8; BLAKE3_CHUNK_LEN]]) {
        (&self.proof.leaf_indices, &self.proof.leaf_data)
    }
}

/// Plain proof structure for FFI.
///
/// This represents a proof before ZK transformation, containing the raw merkle
/// proof data for both matrices A and B^T.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "PlainProof", get_all))]
pub struct PlainProof {
    // Shared fields (dense + MoE)
    pub m: usize,
    pub n: usize, // For MoE: n_e (per-expert intermediate dim)
    pub k: usize,
    pub noise_rank: usize,
    pub a: MatrixMerkleProof,
    pub bt: MatrixMerkleProof,

    // Optional MoE fields (None for dense)
    pub moe: Option<MoEProofParams>,
}

/// MoE-specific proof parameters to be included in the `PlainProof` for MoE proofs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "MoEProofParams", get_all))]
pub struct MoEProofParams {
    /// Total number of experts.
    pub e: usize,
    /// Number of experts each token is routed to.
    pub top_k: usize,
    /// Index of the current expert.
    pub expert_idx: u16,
    /// Cumulative token counts per expert (exclusive ends into the flat routing array).
    /// Entry `i` equals the total number of tokens assigned to experts `0..=i`;
    /// the last entry equals `m * top_k`.
    pub routing_end_offsets: Vec<u32>,
    /// Inner row indices within the expert's token subset (before routing to global indices).
    /// Used to reconstruct the correct rows_pattern for job_key computation.
    pub inner_a_rows: Vec<usize>,
    /// Merkle proof for the flat routing data (all experts' token indices concatenated as little-endian u32s).
    /// The Merkle tree is built with `job_key` as the Blake3 key.
    pub routing_proof: MerkleProof,
}

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

/// In one proof, the tiles row indices are given relative to the actual matrix multiplication being carried out.
/// In the MoE setting, the rows need to be "decoded" into indices in the global token matrix that was committed.
/// [`OuterIndices`] represents the decoded global indices corresponding to the proof's local row indices.
pub type OuterIndices = Vec<u32>;

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

/// Block certificate version (the wire format a block's certificate uses).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CertificateVersion {
    /// V1: dense (non-MoE) proofs only.
    ZkDense = 1,
    /// V2: MoE and dense proofs.
    ZkMoe = 2,
}

impl TryFrom<u32> for CertificateVersion {
    type Error = anyhow::Error;

    fn try_from(version: u32) -> Result<Self> {
        match version {
            v if v == Self::ZkDense as u32 => Ok(Self::ZkDense),
            v if v == Self::ZkMoe as u32 => Ok(Self::ZkMoe),
            v => bail!("unknown certificate version: {v}"),
        }
    }
}

/// Checks that `proof` can be certified at `cert_version` and returns the
/// parsed version.
pub fn check_cert_version_eligible(cert_version: u32, proof: &PlainProof) -> Result<CertificateVersion> {
    let version = CertificateVersion::try_from(cert_version)?;
    let min_version = proof.min_cert_version() as u32;
    ensure!(
        min_version <= cert_version,
        "proof requires certificate version >= {min_version}, but the block requires version {cert_version} \
         (MoE proofs are only valid at or after the V2 crossover)"
    );
    Ok(version)
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl PlainProof {
    #[new]
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

fn extract_routing_strips(p: &PlainProof, params: &PublicProofParams) -> Result<Vec<Vec<u8>>> {
    let moe_proof = p
        .moe
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("PlainProof has no MoE data; cannot extract routing strips"))?;
    let moe_public = params
        .moe
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("PublicProofParams has no MoE data; cannot extract routing strips"))?;

    let inner = params.a_inner_indices();
    ensure_eq!(
        inner.len(),
        p.a.row_indices.len(),
        "inner A indices and outer row count must match"
    );

    routing_blake_hotspot_rows(moe_public.expert_start_offset(), &inner)
        .iter()
        .map(|&hotspot_idx| {
            let block_start = hotspot_idx as usize * BLOCK_LEN;
            moe_proof
                .routing_proof
                .extract_bytes(block_start, BLOCK_LEN)
                .with_context(|| format!("routing strip: extract 64 bytes at row_start={}", block_start))
        })
        .collect()
}

fn extract_strips(indices: &[usize], k: usize, strip_len: usize, proof: &MerkleProof) -> Result<Vec<Vec<i8>>> {
    indices
        .iter()
        .map(|&idx| {
            proof
                .extract_bytes(idx * k, strip_len)
                .map(|b| b.into_iter().map(|x| x as i8).collect())
                .context("Failed to extract strip")
        })
        .collect()
}

fn compute_external_cvs(
    locs: &[AuxiliaryCvLocation],
    p: &PlainProof,
    k: usize,
    key: [u8; BLAKE3_DIGEST_SIZE],
) -> Result<Vec<[u8; BLAKE3_DIGEST_SIZE]>> {
    let total_b_cols = p.total_b_cols();
    let a_ranges = p.a.proof.compute_sibling_ranges(pearl_blake3::padded_chunk_len(p.m * k));
    let b_ranges =
        p.bt.proof
            .compute_sibling_ranges(pearl_blake3::padded_chunk_len(total_b_cols * k));

    let routing = p.moe.as_ref().map(|moe| {
        let total_routing_entries = p.m * moe.top_k;
        let raw_len = total_routing_entries * std::mem::size_of::<u32>();
        let ranges = moe
            .routing_proof
            .compute_sibling_ranges(pearl_blake3::padded_chunk_len(raw_len));
        (&moe.routing_proof, ranges)
    });

    locs.iter()
        .map(|loc| {
            let (proof, ranges) = match loc.source {
                ProofSource::A => (&p.a.proof, &a_ranges),
                ProofSource::B => (&p.bt.proof, &b_ranges),
                ProofSource::Routing => {
                    let (proof, ranges) = routing
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("ProofSource::Routing requested on a non-MoE PlainProof"))?;
                    (*proof, ranges)
                }
            };
            if let Some(&(_, _, h)) = ranges.iter().find(|(s, e, _)| *s == loc.global_start && *e == loc.global_end) {
                return Ok(h);
            }
            log::warn!("CV not in proof, computing recursively");
            proof
                .compute_cv(loc.global_start, loc.global_end, ranges, key)
                .context("Failed to compute CV")
        })
        .collect()
}

/// Converts indices to a PeriodicPattern and base offset.
pub fn list_to_pattern(indices: &[u32]) -> Result<(PeriodicPattern, u32)> {
    if indices.is_empty() {
        bail!("pattern parsing error: Empty indices");
    }
    if !indices.windows(2).all(|w| w[0] < w[1]) {
        bail!("pattern parsing error: Indices not strictly increasing");
    }

    let offset = indices[0];
    let normalized: Vec<u32> = indices.iter().map(|&i| i - offset).collect();
    let pattern = PeriodicPattern::from_list(&normalized).context("pattern parsing error")?;

    if !pattern.offset_is_valid(offset) {
        bail!("pattern parsing error: offset {} is not valid for pattern", offset);
    }

    Ok((pattern, offset))
}

/// bincode tag byte for `Option::None`, appended to legacy V1 blobs to make
/// them parse as the current format (whose trailing field is `moe: Option<_>`).
const BINCODE_OPTION_NONE_TAG: u8 = 0x00;

impl PlainProof {
    /// The lowest block certificate version that can certify this proof.
    pub fn min_cert_version(&self) -> CertificateVersion {
        if self.moe.is_some() {
            CertificateVersion::ZkMoe
        } else {
            CertificateVersion::ZkDense
        }
    }

    /// Deserializes a `PlainProof`, accepting both the current format and the
    /// legacy V1 format (same layout, missing the trailing `moe` Option tag).
    pub fn deserialize_compat(bytes: &[u8]) -> Result<Self> {
        use bincode::Options;
        let strict = bincode::options().with_fixint_encoding();
        strict.deserialize(bytes).or_else(|_| {
            let mut padded = Vec::with_capacity(bytes.len() + 1);
            padded.extend_from_slice(bytes);
            padded.push(BINCODE_OPTION_NONE_TAG);
            strict
                .deserialize(&padded)
                .context("not a valid PlainProof (tried current and legacy V1 formats)")
        })
    }

    /// Returns the merkle proof for a given proof source. Errors if `Routing`
    /// is requested on a non-MoE proof.
    fn proof_for(&self, source: ProofSource) -> Result<&MerkleProof> {
        match source {
            ProofSource::A => Ok(&self.a.proof),
            ProofSource::B => Ok(&self.bt.proof),
            ProofSource::Routing => self
                .moe
                .as_ref()
                .map(|moe| &moe.routing_proof)
                .ok_or_else(|| anyhow::anyhow!("ProofSource::Routing requested on a non-MoE PlainProof")),
        }
    }

    /// Extracts external messages from the proof based on the provided locations.
    fn extract_external_messages(&self, locs: &[AuxiliaryMsgLocation]) -> Result<Vec<[u8; 64]>> {
        locs.iter()
            .map(|loc| {
                let proof = self.proof_for(loc.source)?;
                proof.extract_bytes(loc.global_start, 64).map(|b| b.try_into().unwrap())
            })
            .collect()
    }

    fn total_b_cols(&self) -> usize {
        if let Some(moe) = &self.moe { self.n * moe.e } else { self.n }
    }

    /// Derives the inner A/B index lists used to build the periodic patterns,
    /// plus the public `MoEParams` (when this is an MoE proof).
    fn moe_inner_indices(&self) -> Result<(Vec<u32>, Vec<u32>, Option<MoEParams>)> {
        let a_indices: Vec<u32> = self.a.row_indices.iter().map(|&x| x as u32).collect();
        let bt_indices: Vec<u32> = self.bt.row_indices.iter().map(|&x| x as u32).collect();

        let Some(moe) = &self.moe else {
            return Ok((a_indices, bt_indices, None));
        };

        ensure!((moe.expert_idx as usize) < moe.e);
        ensure!(moe.e == moe.routing_end_offsets.len());

        let weight_col_offset = (moe.expert_idx as usize) * self.n;
        // p.bt.row_indices are already global indices (offset by expert_idx * n_e in mining).
        for &idx in &self.bt.row_indices {
            ensure!(
                idx >= weight_col_offset && idx < weight_col_offset + self.n,
                "B column index {} out of range for expert {} (expected [{}, {}))",
                idx,
                moe.expert_idx,
                weight_col_offset,
                weight_col_offset + self.n
            );
        }
        // In the MoE case, n represents the per-expert intermediate dimension n_e.
        ensure!(
            self.bt.row_indices.len() < self.n,
            "B^T row indices length {} is not less than n_e {}",
            self.bt.row_indices.len(),
            self.n
        );

        ensure!(moe.routing_end_offsets.len() == moe.e);

        let inner_a: Vec<u32> = moe.inner_a_rows.iter().map(|&x| x as u32).collect();
        let inner_b: Vec<u32> = bt_indices.iter().map(|&idx| idx - weight_col_offset as u32).collect();

        ensure!(
            moe.e <= PublicProofParams::MAX_NUM_EXPERTS,
            "number of experts {} exceeds maximum {}",
            moe.e,
            PublicProofParams::MAX_NUM_EXPERTS
        );
        let moe_params = MoEParams {
            routing_offsets: moe.routing_end_offsets.clone(),
            expert_idx: moe.expert_idx,
            hash_routing: moe.routing_proof.root,
            outer_indices: a_indices,
        };
        Ok((inner_a, inner_b, Some(moe_params)))
    }

    /// Converts plain proof to Rust proof types, checks a,bt merkle roots match provided hashes.
    pub fn parse_proof(&self, header: IncompleteBlockHeader) -> Result<(PrivateProofParams, PublicProofParams)> {
        let (m, n, k) = (self.m, self.n, self.k);

        for &tok in &self.a.row_indices {
            ensure!(tok < m, "routing entry {} out of range for t={}", tok, m);
        }

        let (inner_a_indices, inner_b_indices, moe_params) = self.moe_inner_indices()?;
        let (rows_pattern, t_rows) = list_to_pattern(&inner_a_indices)?;
        let (cols_pattern, t_cols) = list_to_pattern(&inner_b_indices)?;

        let public = PublicProofParams {
            block_header: header,
            mining_config: MiningConfiguration {
                common_dim: k as u32,
                rank: self.noise_rank as u16,
                mma_type: MMAType::Int7xInt7ToInt32,
                rows_pattern,
                cols_pattern,
                moe: self.moe.as_ref().map(|m| MoEConfig {
                    e: m.e as u16,
                    top_k: m.top_k as u16,
                }),
            },
            hash_a: self.a.proof.root,
            hash_b: self.bt.proof.root,
            hash_jackpot: [0xFFu8; 32], // Consumed only by ZK verifier
            m: m as u32,
            n: n as u32,
            t_rows,
            t_cols,
            moe: moe_params,
        };

        let (compiled, msg_locs, cv_locs) = public.compile();
        let strip_len = public.dot_product_length();

        let s_routing = if self.moe.is_some() {
            let strips = extract_routing_strips(self, &public)?;
            ensure_eq!(
                strips.len(),
                compiled.blake_proof.num_routing_strips,
                "MoE s_routing strips must match num_routing_strips"
            );
            strips
        } else {
            vec![]
        };

        let private = PrivateProofParams {
            s_a: extract_strips(&self.a.row_indices, k, strip_len, &self.a.proof)?,
            s_b: extract_strips(&self.bt.row_indices, k, strip_len, &self.bt.proof)?,
            s_routing,
            external_msgs: self.extract_external_messages(&msg_locs)?,
            external_cvs: compute_external_cvs(&cv_locs, self, k, public.job_key())?,
        };

        let opt_hash_routing = self.moe.as_ref().map(|moe| moe.routing_proof.root);
        let (hash_a, hash_b) = compiled
            .blake_proof
            .evaluate_blake(compiled.job_key, &private, opt_hash_routing)?;
        ensure_eq!(hash_a, self.a.proof.root, "Hash A mismatch, job_key={:?}", compiled.job_key);
        ensure_eq!(hash_b, self.bt.proof.root, "Hash B mismatch, job_key={:?}", compiled.job_key);

        if let Some(moe) = &self.moe {
            verify_moe_routing(moe, &self.a.row_indices, compiled.job_key)?;
        }

        Ok((private, public))
    }
}

/// Verifies the routing Merkle membership proof (recomputed root matches the committed root)
/// and that, for each sampled outer index, `routing[expert_idx][inner_idx]` equals it.
fn verify_moe_routing(moe: &MoEProofParams, outer_indices: &[usize], key: [u8; BLAKE3_DIGEST_SIZE]) -> Result<()> {
    let computed_root = moe
        .routing_proof
        .compute_root(key)
        .ok_or_else(|| anyhow::anyhow!("routing Merkle proof has no leaves"))?;
    ensure_eq!(
        computed_root,
        moe.routing_proof.root,
        "routing Merkle membership proof failed: computed root does not match hash_routing"
    );

    // The routing is serialized as a flat array of little-endian u32 values.
    // `routing_end_offsets` stores exclusive ends (cumulative counts), so the
    // start of `expert_idx` is the previous expert's end (0 for expert 0).
    let routing_start_offset = match moe.expert_idx {
        0 => 0u32,
        idx => moe.routing_end_offsets[idx as usize - 1],
    };
    for (i, &inner_idx) in moe.inner_a_rows.iter().enumerate() {
        let byte_offset = (routing_start_offset as usize + inner_idx) * std::mem::size_of::<u32>();
        let bytes = moe
            .routing_proof
            .extract_bytes(byte_offset, std::mem::size_of::<u32>())
            .with_context(|| {
                format!(
                    "failed to extract routing entry at inner_idx={} (byte_offset={})",
                    inner_idx, byte_offset
                )
            })?;
        let routing_value = u32::from_le_bytes(bytes.try_into().unwrap());
        ensure_eq!(
            routing_value as usize,
            outer_indices[i],
            "routing mismatch: routing[{}][{}] = {} but outer_indices[{}] = {}",
            moe.expert_idx,
            inner_idx,
            routing_value,
            i,
            outer_indices[i]
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_merkle_proof() -> MerkleProof {
        MerkleProof {
            leaf_data: vec![],
            leaf_indices: vec![],
            total_leaves: 0,
            root: [0u8; BLAKE3_DIGEST_SIZE],
            siblings: vec![],
        }
    }

    fn dummy_matrix_proof() -> MatrixMerkleProof {
        MatrixMerkleProof {
            proof: dummy_merkle_proof(),
            row_indices: vec![1, 2, 3],
        }
    }

    fn dense_proof() -> PlainProof {
        PlainProof {
            m: 8,
            n: 4,
            k: 16,
            noise_rank: 2,
            a: dummy_matrix_proof(),
            bt: dummy_matrix_proof(),
            moe: None,
        }
    }

    fn moe_proof() -> PlainProof {
        PlainProof {
            moe: Some(MoEProofParams {
                e: 4,
                top_k: 2,
                expert_idx: 1,
                routing_end_offsets: vec![2, 4, 6, 8],
                inner_a_rows: vec![0, 1],
                routing_proof: dummy_merkle_proof(),
            }),
            ..dense_proof()
        }
    }

    #[test]
    fn min_cert_version_dense_is_v1_moe_is_v2() {
        assert_eq!(dense_proof().min_cert_version(), CertificateVersion::ZkDense);
        assert_eq!(moe_proof().min_cert_version(), CertificateVersion::ZkMoe);
    }

    #[test]
    fn dense_proof_eligible_under_both_versions() {
        assert_eq!(
            check_cert_version_eligible(CertificateVersion::ZkDense as u32, &dense_proof()).unwrap(),
            CertificateVersion::ZkDense
        );
        assert_eq!(
            check_cert_version_eligible(CertificateVersion::ZkMoe as u32, &dense_proof()).unwrap(),
            CertificateVersion::ZkMoe
        );
    }

    #[test]
    fn moe_proof_eligible_only_under_v2() {
        assert_eq!(
            check_cert_version_eligible(CertificateVersion::ZkMoe as u32, &moe_proof()).unwrap(),
            CertificateVersion::ZkMoe
        );
        let err = check_cert_version_eligible(CertificateVersion::ZkDense as u32, &moe_proof()).unwrap_err();
        assert!(err.to_string().contains("crossover"), "unexpected error: {err}");
    }

    #[test]
    fn unknown_cert_versions_rejected() {
        for version in [0u32, 3, u32::MAX] {
            assert!(check_cert_version_eligible(version, &dense_proof()).is_err());
        }
    }

    #[test]
    fn deserialize_compat_roundtrips_current_format() {
        for proof in [dense_proof(), moe_proof()] {
            let bytes = bincode::serialize(&proof).unwrap();
            let parsed = PlainProof::deserialize_compat(&bytes).unwrap();
            assert_eq!(parsed.m, proof.m);
            assert_eq!(parsed.moe.is_some(), proof.moe.is_some());
        }
    }

    #[test]
    fn deserialize_compat_accepts_legacy_v1_format() {
        // A legacy V1 blob is the current dense serialization minus the
        // trailing Option::None tag for `moe`.
        let bytes = bincode::serialize(&dense_proof()).unwrap();
        assert_eq!(*bytes.last().unwrap(), BINCODE_OPTION_NONE_TAG);
        let legacy = &bytes[..bytes.len() - 1];

        let parsed = PlainProof::deserialize_compat(legacy).unwrap();
        assert_eq!(parsed.m, dense_proof().m);
        assert!(parsed.moe.is_none());
    }

    #[test]
    fn deserialize_compat_rejects_garbage_and_trailing_bytes() {
        assert!(PlainProof::deserialize_compat(&[0xAB; 7]).is_err());

        let mut bytes = bincode::serialize(&dense_proof()).unwrap();
        bytes.extend_from_slice(&[0x01, 0x02]);
        assert!(PlainProof::deserialize_compat(&bytes).is_err());
    }
}
