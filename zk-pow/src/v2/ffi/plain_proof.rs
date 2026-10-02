//! Int7 plain-proof wire types (`PlainProof`, `MoEProofParams`, `MatrixMerkleProof`) and
//! [`PlainProof::deserialize_compat`]. Their Python methods and the certificate-version
//! eligibility rules live in the shared binding layer, `crate::ffi`.

use anyhow::{Context, Result};
use pearl_blake3::MerkleProof;
use serde::{Deserialize, Serialize};

use pearl_blake3::BLAKE3_DIGEST_SIZE;

/// Merkle proof data for a single matrix.
///
/// Default serde is v1–v3 [`MerkleProof::serialize_chunk_1024`]. V4 openings
/// use [`MerkleProof::serialize_variable_chunk`].
#[derive(Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "MatrixMerkleProof"))]
pub struct MatrixMerkleProof {
    #[serde(
        serialize_with = "MerkleProof::serialize_chunk_1024",
        deserialize_with = "MerkleProof::deserialize_chunk_1024"
    )]
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
        leaf_data: Vec<Vec<u8>>,
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
    pub fn data(&self) -> (&[usize], &[Vec<u8>]) {
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
    /// Certificate v1–v3 `PlainProof` encoding ([`MerkleProof::serialize_chunk_1024`]).
    #[serde(
        serialize_with = "MerkleProof::serialize_chunk_1024",
        deserialize_with = "MerkleProof::deserialize_chunk_1024"
    )]
    pub routing_proof: MerkleProof,
}

/// In one proof, the tiles row indices are given relative to the actual matrix multiplication being carried out.
/// In the MoE setting, the rows need to be "decoded" into indices in the global token matrix that was committed.
/// [`OuterIndices`] represents the decoded global indices corresponding to the proof's local row indices.
pub type OuterIndices = Vec<u32>;

/// bincode tag byte for `Option::None`, appended to legacy V1 blobs to make
/// them parse as the current format (whose trailing field is `moe: Option<_>`).
const BINCODE_OPTION_NONE_TAG: u8 = 0x00;

impl PlainProof {
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
    fn deserialize_compat_roundtrips_master_format() {
        for proof in [dense_proof(), moe_proof()] {
            let bytes = bincode::serialize(&proof).unwrap();
            let parsed = PlainProof::deserialize_compat(&bytes).unwrap();
            assert_eq!(parsed.m, proof.m);
            assert_eq!(parsed.moe.is_some(), proof.moe.is_some());
        }
    }

    #[test]
    fn deserialize_compat_accepts_legacy_v1_missing_moe_tag() {
        let bytes = bincode::serialize(&dense_proof()).unwrap();
        assert_eq!(bytes[bytes.len() - 1], 0u8, "moe None is a trailing zero tag");
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

    fn chunk_merkle_proof() -> MerkleProof {
        MerkleProof {
            leaf_data: vec![vec![7u8; pearl_blake3::BLAKE3_CHUNK_LEN]],
            leaf_indices: vec![0],
            total_leaves: 1,
            root: [1u8; BLAKE3_DIGEST_SIZE],
            siblings: vec![],
        }
    }

    #[derive(Serialize, Deserialize)]
    struct AsChunk1024(
        #[serde(
            serialize_with = "MerkleProof::serialize_chunk_1024",
            deserialize_with = "MerkleProof::deserialize_chunk_1024"
        )]
        MerkleProof,
    );

    #[derive(Serialize, Deserialize)]
    struct AsVariableChunk(
        #[serde(
            serialize_with = "MerkleProof::serialize_variable_chunk",
            deserialize_with = "MerkleProof::deserialize_variable_chunk"
        )]
        MerkleProof,
    );

    /// Master's `leaf_data` helper: length-prefixed bytes that must be 1024.
    mod master_chunk_vec {
        use pearl_blake3::BLAKE3_CHUNK_LEN;
        use serde::{Deserialize, Deserializer, Serialize, Serializer};

        pub fn serialize<S: Serializer>(data: &[[u8; BLAKE3_CHUNK_LEN]], serializer: S) -> Result<S::Ok, S::Error> {
            let vecs: Vec<&[u8]> = data.iter().map(|a| a.as_slice()).collect();
            vecs.serialize(serializer)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<[u8; BLAKE3_CHUNK_LEN]>, D::Error> {
            let vecs: Vec<Vec<u8>> = Vec::deserialize(deserializer)?;
            vecs.into_iter()
                .map(|v| {
                    v.try_into()
                        .map_err(|_| serde::de::Error::custom("leaf data must be CHUNK_LEN bytes"))
                })
                .collect()
        }
    }

    #[test]
    fn chunk_1024_merkle_encoding_matches_master_chunk_vec() {
        use pearl_blake3::BLAKE3_CHUNK_LEN;

        #[derive(Serialize, Deserialize)]
        struct Master {
            #[serde(with = "master_chunk_vec")]
            leaf_data: Vec<[u8; BLAKE3_CHUNK_LEN]>,
            leaf_indices: Vec<usize>,
            total_leaves: usize,
            root: [u8; BLAKE3_DIGEST_SIZE],
            siblings: Vec<[u8; BLAKE3_DIGEST_SIZE]>,
        }

        let proof = chunk_merkle_proof();
        let master = Master {
            leaf_data: vec![[7u8; BLAKE3_CHUNK_LEN]],
            leaf_indices: vec![0],
            total_leaves: 1,
            root: [1u8; BLAKE3_DIGEST_SIZE],
            siblings: vec![],
        };
        let master_bytes = bincode::serialize(&master).unwrap();
        let chunk_1024_bytes = bincode::serialize(&AsChunk1024(proof.clone())).unwrap();
        assert_eq!(chunk_1024_bytes, master_bytes);

        let variable_bytes = bincode::serialize(&AsVariableChunk(proof.clone())).unwrap();
        assert_eq!(
            variable_bytes, master_bytes,
            "1024-byte leaves are length-prefixed in both chunk_1024 and variable_chunk"
        );

        let AsChunk1024(decoded) = bincode::deserialize(&master_bytes).unwrap();
        assert_eq!(decoded.leaf_data, proof.leaf_data);
        assert_eq!(decoded.leaf_indices, proof.leaf_indices);
    }

    #[test]
    fn chunk_1024_serde_rejects_non_1024_leaves() {
        let proof = MerkleProof {
            leaf_data: vec![vec![0u8; 128]],
            leaf_indices: vec![0],
            total_leaves: 1,
            root: [0u8; BLAKE3_DIGEST_SIZE],
            siblings: vec![],
        };
        assert!(bincode::serialize(&AsChunk1024(proof.clone())).is_err());
        assert!(bincode::serialize(&AsVariableChunk(proof.clone())).is_ok());
        let bytes = bincode::serialize(&AsVariableChunk(proof)).unwrap();
        assert!(bincode::deserialize::<AsChunk1024>(&bytes).is_err());
    }

    #[test]
    fn variable_chunk_serde_roundtrips_128_byte_leaves() {
        let proof = MerkleProof {
            leaf_data: vec![vec![3u8; 128]],
            leaf_indices: vec![0],
            total_leaves: 1,
            root: [2u8; BLAKE3_DIGEST_SIZE],
            siblings: vec![],
        };
        let bytes = bincode::serialize(&AsVariableChunk(proof.clone())).unwrap();
        let AsVariableChunk(decoded) = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.leaf_data, proof.leaf_data);
        assert_eq!(decoded.leaf_data[0].len(), 128);
    }

    #[test]
    fn variable_chunk_deserialize_rejects_unsorted_indices() {
        let proof = MerkleProof {
            leaf_data: vec![vec![0u8; 128], vec![1u8; 128]],
            leaf_indices: vec![2, 0],
            total_leaves: 4,
            root: [0u8; BLAKE3_DIGEST_SIZE],
            siblings: vec![],
        };
        let bytes = bincode::serialize(&AsVariableChunk(proof)).unwrap();
        assert!(bincode::deserialize::<AsVariableChunk>(&bytes).is_err());
    }

    #[test]
    fn plain_proof_with_leaves_roundtrips_int7() {
        let mut p = dense_proof();
        p.a.proof = chunk_merkle_proof();
        p.bt.proof = chunk_merkle_proof();
        let bytes = bincode::serialize(&p).unwrap();
        let parsed = PlainProof::deserialize_compat(&bytes).unwrap();
        assert_eq!(parsed.a.proof.leaf_data[0].len(), pearl_blake3::BLAKE3_CHUNK_LEN);
        assert_eq!(parsed.a.proof.leaf_data[0][0], 7);
    }
}
