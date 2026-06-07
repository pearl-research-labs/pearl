//! Plain proof structures and parsing for FFI.
//!
//! This module contains the core data structures (`PlainProof`, `MatrixMerkleProof`)
//! and the `parse_plain_proof` function used to convert plain proofs to ZK proof parameters.

use anyhow::{Context, Result, bail};
use pearl_blake3::MerkleProof;
use serde::{Deserialize, Serialize};

use crate::api::proof::{
    IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, PrivateProofParams, PublicProofParams,
};
use crate::circuit::chip::blake3::program::{AuxiliaryCvLocation, AuxiliaryMsgLocation};
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
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub noise_rank: usize,
    pub a: MatrixMerkleProof,
    pub bt: MatrixMerkleProof,
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
    fn py_new(
        m: usize,
        n: usize,
        k: usize,
        noise_rank: usize,
        a_merkle_proof: MatrixMerkleProof,
        bt_merkle_proof: MatrixMerkleProof,
    ) -> Self {
        Self {
            m,
            n,
            k,
            noise_rank,
            a: a_merkle_proof,
            bt: bt_merkle_proof,
        }
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
        bincode::deserialize(&bytes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Deserialization failed: {}", e)))
    }
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

fn extract_external_messages(locs: &[AuxiliaryMsgLocation], p: &PlainProof) -> Result<Vec<[u8; 64]>> {
    locs.iter()
        .map(|loc| {
            let proof = if loc.is_b { &p.bt.proof } else { &p.a.proof };
            proof.extract_bytes(loc.global_start, 64).map(|b| b.try_into().unwrap())
        })
        .collect()
}

fn compute_external_cvs(
    locs: &[AuxiliaryCvLocation],
    p: &PlainProof,
    m: usize,
    n: usize,
    k: usize,
    key: [u8; BLAKE3_DIGEST_SIZE],
) -> Result<Vec<[u8; BLAKE3_DIGEST_SIZE]>> {
    let a_ranges = p.a.proof.compute_sibling_ranges(pearl_blake3::padded_chunk_len(m * k));
    let b_ranges = p.bt.proof.compute_sibling_ranges(pearl_blake3::padded_chunk_len(n * k));

    locs.iter()
        .map(|loc| {
            let (ranges, proof) = if loc.is_b {
                (&b_ranges, &p.bt.proof)
            } else {
                (&a_ranges, &p.a.proof)
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

fn usize_to_u32(field: &str, value: usize) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{field}={value} does not fit in u32"))
}

fn usize_indices_to_u32(field: &str, indices: &[usize]) -> Result<Vec<u32>> {
    indices
        .iter()
        .enumerate()
        .map(|(i, &value)| u32::try_from(value).with_context(|| format!("{field}[{i}]={value} does not fit in u32")))
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

/// Converts plain proof to Rust proof types, checks a,bt merkle roots match provided hashes.
pub fn parse_plain_proof(header: IncompleteBlockHeader, p: &PlainProof) -> Result<(PrivateProofParams, PublicProofParams)> {
    let m = usize_to_u32("m", p.m)?;
    let n = usize_to_u32("n", p.n)?;
    let k = usize_to_u32("k", p.k)?;
    let noise_rank = usize_to_u32("noise_rank", p.noise_rank)?;

    let a_indices = usize_indices_to_u32("a.row_indices", &p.a.row_indices)?;
    let bt_indices = usize_indices_to_u32("bt.row_indices", &p.bt.row_indices)?;
    let (rows_pattern, t_rows) = list_to_pattern(&a_indices)?;
    let (cols_pattern, t_cols) = list_to_pattern(&bt_indices)?;

    let public = PublicProofParams {
        block_header: header,
        mining_config: MiningConfiguration {
            common_dim: k,
            rank: noise_rank as u16,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern,
            cols_pattern,
            reserved: MiningConfiguration::RESERVED_VALUE,
        },
        hash_a: p.a.proof.root,
        hash_b: p.bt.proof.root,
        hash_jackpot: [0xFFu8; 32], // Consumed only by ZK verifier
        m,
        n,
        t_rows,
        t_cols,
    };

    crate::api::sanity_checks::validate_tile_offsets(&public.mining_config, m, n, t_rows, t_cols)?;

    let (compiled, msg_locs, cv_locs) = public.compile();

    let strip_len = public.dot_product_length();

    let private = PrivateProofParams {
        s_a: extract_strips(&p.a.row_indices, p.k, strip_len, &p.a.proof)?,
        s_b: extract_strips(&p.bt.row_indices, p.k, strip_len, &p.bt.proof)?,
        external_msgs: extract_external_messages(&msg_locs, p)?,
        external_cvs: compute_external_cvs(&cv_locs, p, p.m, p.n, p.k, public.job_key())?,
    };

    let (hash_a, hash_b) = compiled.blake_proof.evaluate_blake(compiled.job_key, &private)?;
    ensure_eq!(hash_a, p.a.proof.root, "Hash A mismatch, job_key={:?}", compiled.job_key);
    ensure_eq!(hash_b, p.bt.proof.root, "Hash B mismatch, job_key={:?}", compiled.job_key);

    Ok((private, public))
}
