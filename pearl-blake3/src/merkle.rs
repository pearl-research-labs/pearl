//! Merkle tree construction and proof generation/verification.
//!
//! `MerkleTree` builds a BLAKE3 Merkle tree from raw bytes and generates multi-leaf proofs.
//! `MerkleProof` verifies proofs and provides byte extraction utilities.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{ensure, Result};
use blake3::{CHUNK_LEN, OUT_LEN};
use rayon::prelude::*;

use crate::hasher::{Blake3Hasher, Digest};

/// Leaf sizes this crate will build or accept in a proof.
pub const ALLOWED_CHUNK_LENS: [usize; 4] = [128, 256, 512, 1024];

pub fn is_allowed_chunk_len(n: usize) -> bool {
    ALLOWED_CHUNK_LENS.contains(&n)
}

/// Round `raw_len` up to the next multiple of the native BLAKE3 chunk (1024).
pub fn padded_chunk_len(raw_len: usize) -> usize {
    raw_len.div_ceil(CHUNK_LEN) * CHUNK_LEN
}

/// Zero-pad `data` so its length is a multiple of the native BLAKE3 chunk (1024).
///
/// Matrix data must be padded to a BLAKE3 chunk boundary before building a
/// Merkle tree.
pub fn pad_to_chunk_boundary(data: &[u8]) -> Vec<u8> {
    let mut padded = data.to_vec();
    padded.resize(padded_chunk_len(data.len()), 0);
    padded
}

fn hash_leaves(hasher: &Blake3Hasher, data: &[u8], chunk_len: usize) -> Vec<Digest> {
    if chunk_len == CHUNK_LEN {
        return hasher.hash_chunks(data);
    }
    data.par_chunks(chunk_len)
        .enumerate()
        .map(|(i, chunk)| hasher.chunk_cv(chunk, i as u64))
        .collect()
}

// ============================================================================
// MerkleTree
// ============================================================================

/// BLAKE3 Merkle tree with multi-leaf proof generation.
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "MerkleTree"))]
pub struct MerkleTree {
    key: Digest,
    layers: Vec<Vec<Digest>>,
    data: Vec<u8>,
    chunk_len: usize,
}

impl MerkleTree {
    /// Build a Merkle tree from `data` using keyed BLAKE3 and 1024-byte leaves.
    pub fn new(data: &[u8], key: Digest) -> Self {
        Self::with_chunk_len(data, key, CHUNK_LEN).expect("1024 is in ALLOWED_CHUNK_LENS")
    }

    /// Build a Merkle tree whose leaf size is `chunk_len` (must be in [`ALLOWED_CHUNK_LENS`]).
    pub fn with_chunk_len(data: &[u8], key: Digest, chunk_len: usize) -> Result<Self> {
        ensure!(
            is_allowed_chunk_len(chunk_len),
            "chunk_len {chunk_len} is not in {ALLOWED_CHUNK_LENS:?}"
        );
        let hasher = Blake3Hasher::with_key(key);
        let layers = if data.is_empty() {
            vec![vec![]]
        } else if data.len() <= chunk_len {
            // One leaf: the root is the ROOT-finalized keyed hash, not a chunk CV.
            vec![vec![hasher.hash(data)]]
        } else {
            let mut layers = vec![hash_leaves(&hasher, data, chunk_len)];
            while layers.last().unwrap().len() > 2 {
                let prev = layers.last().unwrap();
                layers.push(hasher.combine_layer(prev));
            }
            let last = layers.last().unwrap();
            if last.len() == 2 {
                let root = hasher.root_cv(&last[0], &last[1]);
                layers.push(vec![root]);
            }
            layers
        };
        Ok(Self {
            key,
            layers,
            data: data.to_vec(),
            chunk_len,
        })
    }

    /// The BLAKE3 key this tree was built with.
    pub fn key(&self) -> Digest {
        self.key
    }

    pub fn root(&self) -> Digest {
        self.layers.last().map(|l| l[0]).unwrap_or([0u8; OUT_LEN])
    }

    pub fn leaf_hashes(&self) -> &[Digest] {
        &self.layers[0]
    }

    pub fn num_leaves(&self) -> usize {
        self.layers[0].len()
    }

    /// Generate a multi-leaf proof. Returns a complete `MerkleProof`.
    pub fn get_multileaf_proof(&self, leaf_indices: &[usize]) -> MerkleProof {
        assert!(!leaf_indices.is_empty(), "leaf_indices must be non-empty");

        let unique: BTreeSet<usize> = leaf_indices.iter().copied().collect();
        let total_leaves = self.num_leaves();

        assert!(
            *unique.last().unwrap() < total_leaves,
            "leaf index out of bounds"
        );

        // Collect leaf data
        let sorted_indices: Vec<usize> = unique.iter().copied().collect();
        let chunk_len = self.chunk_len;
        let leaf_data: Vec<Vec<u8>> = sorted_indices
            .iter()
            .map(|&i| {
                let start = i * chunk_len;
                let end = (start + chunk_len).min(self.data.len());
                let mut chunk = vec![0u8; chunk_len];
                chunk[..end - start].copy_from_slice(&self.data[start..end]);
                chunk
            })
            .collect();

        // Walk tree level-by-level to collect sibling hashes
        let mut siblings: Vec<Digest> = Vec::new();
        let mut current_set = unique;
        let mut level_len = total_leaves;

        let mut level = 0;
        while level_len > 1 && !current_set.is_empty() {
            let level_nodes = &self.layers[level];

            for &i in &current_set {
                if i % 2 == 1 {
                    if !current_set.contains(&(i - 1)) {
                        siblings.push(level_nodes[i - 1]);
                    }
                } else if !current_set.contains(&(i + 1)) && (i + 1) < level_len {
                    siblings.push(level_nodes[i + 1]);
                }
            }

            current_set = current_set.iter().map(|&i| i / 2).collect();
            level_len = level_len.div_ceil(2);
            level += 1;
        }

        MerkleProof {
            leaf_data,
            leaf_indices: sorted_indices,
            total_leaves,
            root: self.root(),
            siblings,
        }
    }

    /// Compute which leaf indices are needed to prove the given matrix rows.
    /// `chunk_len` must be in [`ALLOWED_CHUNK_LENS`].
    pub fn compute_leaf_indices_from_rows(
        row_indices: &[usize],
        shape: (usize, usize),
        chunk_len: usize,
    ) -> Result<Vec<usize>> {
        ensure!(
            is_allowed_chunk_len(chunk_len),
            "chunk_len {chunk_len} is not in {ALLOWED_CHUNK_LENS:?}"
        );
        let cols = shape.1;
        let mut indices = BTreeSet::new();
        for &row in row_indices {
            let first = (row * cols) / chunk_len;
            let last = ((row + 1) * cols - 1) / chunk_len;
            for i in first..=last {
                indices.insert(i);
            }
        }
        Ok(indices.into_iter().collect())
    }
}

// ============================================================================
// MerkleProof
// ============================================================================

/// Multi-leaf keyed-BLAKE3 Merkle proof.
///
/// Leaves are full chunks of one length in [`ALLOWED_CHUNK_LENS`]. A short last
/// chunk in the tree is zero-padded into `leaf_data`; proofs never store a
/// truncated leaf. Enforced by [`Self::sanity_check`], V4 deserialize, and the
/// Python constructor. A Rust struct literal does not check, but
/// [`Self::compute_root`] (and thus [`Self::verify`]) re-runs
/// [`Self::sanity_check`] and fails closed on malformed proofs.
///
/// No default serde: [`Self::serialize_chunk_1024`] is cert v1–v3;
/// [`Self::serialize_variable_chunk`] is cert v4.
#[derive(Clone)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "MerkleProof"))]
pub struct MerkleProof {
    /// Full chunks of one allowed length; never a truncated last leaf.
    pub leaf_data: Vec<Vec<u8>>,
    pub leaf_indices: Vec<usize>,
    pub total_leaves: usize,
    pub root: Digest,
    pub siblings: Vec<Digest>,
}

impl std::fmt::Debug for MerkleProof {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MerkleProof")
            .field("leaf_indices", &self.leaf_indices)
            .field("total_leaves", &self.total_leaves)
            .field("root", &self.root)
            .field("leaf_count", &self.leaf_data.len())
            .field("sibling_count", &self.siblings.len())
            .finish()
    }
}

impl MerkleProof {
    /// Validate proof structure: matching leaf/index counts, sorted unique
    /// indices, every index below `total_leaves`, and equal-length leaves of
    /// an allowed size.
    pub fn sanity_check(&self) -> Result<()> {
        ensure!(
            !self.leaf_indices.is_empty(),
            "leaf_indices must be non-empty"
        );
        ensure!(
            self.leaf_indices.len() == self.leaf_data.len(),
            "leaf_indices and leaf_data must have the same length"
        );
        ensure!(
            self.leaf_indices.windows(2).all(|w| w[0] < w[1]),
            "leaf_indices must be sorted and unique"
        );
        let max_index = *self.leaf_indices.last().unwrap();
        ensure!(
            max_index < self.total_leaves,
            "leaf index {max_index} out of range for a tree of {} leaves",
            self.total_leaves
        );
        Self::check_equal_allowed_leaves(&self.leaf_data).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    }

    /// Equal-length leaves in an [`ALLOWED_CHUNK_LENS`] size. Empty `leaf_data` is allowed.
    pub(crate) fn check_equal_allowed_leaves(leaf_data: &[Vec<u8>]) -> Result<(), String> {
        let Some(first) = leaf_data.first() else {
            return Ok(());
        };
        if !is_allowed_chunk_len(first.len()) {
            return Err(format!(
                "leaf data length {} is not in {ALLOWED_CHUNK_LENS:?}",
                first.len(),
            ));
        }
        if leaf_data.iter().any(|leaf| leaf.len() != first.len()) {
            return Err("all leaves in a proof must have the same length".into());
        }
        Ok(())
    }

    /// Leaf size in bytes, inferred from the opened leaves (1024 if none).
    pub fn chunk_len(&self) -> usize {
        self.leaf_data
            .first()
            .map(|leaf| leaf.len())
            .filter(|&n| is_allowed_chunk_len(n))
            .unwrap_or(CHUNK_LEN)
    }

    /// Compute leaf hashes (chunk CVs) from the raw leaf data.
    pub fn leaf_hashes(&self, key: Digest) -> Vec<Digest> {
        let hasher = Blake3Hasher::with_key(key);
        self.leaf_indices
            .par_iter()
            .zip(self.leaf_data.par_iter())
            .map(|(&idx, data)| hasher.chunk_cv(data, idx as u64))
            .collect()
    }

    /// Reconstruct the Merkle root from leaf hashes and proof siblings.
    ///
    /// Returns `None` if the proof is malformed: anything [`Self::sanity_check`]
    /// rejects (mismatched leaf/index counts, unsorted or duplicate indices, an
    /// index `>= total_leaves`, bad leaf lengths), or a sibling list that does
    /// not exactly cover the reconstruction. The shape checks are
    /// security-critical, not just hygiene: an out-of-range or duplicate index
    /// would let the reconstruction rebuild the root from the supplied siblings
    /// without ever consuming the corresponding leaf data, so [`Self::verify`]
    /// would accept leaf bytes the root does not commit to.
    pub fn compute_root(&self, key: Digest) -> Option<Digest> {
        if self.sanity_check().is_err() {
            return None;
        }

        let hasher = Blake3Hasher::with_key(key);

        if self.total_leaves == 1 {
            // A single-chunk tree's root is the ROOT-finalized keyed hash of
            // the chunk itself (see `MerkleTree::new`), not its non-root
            // chunk CV.
            return if self.leaf_indices == [0] && self.siblings.is_empty() {
                Some(hasher.hash(&self.leaf_data[0]))
            } else {
                None
            };
        }

        let leaf_hashes = self.leaf_hashes(key);

        let mut current: BTreeMap<usize, Digest> =
            self.leaf_indices.iter().copied().zip(leaf_hashes).collect();
        let mut level_len = self.total_leaves;

        let mut sib_iter = self.siblings.iter();

        while level_len > 2 {
            let mut next: BTreeMap<usize, Digest> = BTreeMap::new();

            for (&i, &cv) in &current {
                if i % 2 == 0 {
                    let left = cv;
                    let right = if let Some(&r) = current.get(&(i + 1)) {
                        Some(r)
                    } else if (i + 1) < level_len {
                        Some(*sib_iter.next()?)
                    } else {
                        None
                    };
                    next.insert(
                        i / 2,
                        match right {
                            Some(r) => hasher.parent_cv(&left, &r),
                            None => left,
                        },
                    );
                } else if current.contains_key(&(i - 1)) {
                    continue;
                } else {
                    let right = cv;
                    let left = *sib_iter.next()?;
                    next.insert(i / 2, hasher.parent_cv(&left, &right));
                }
            }

            current = next;
            level_len = level_len.div_ceil(2);
        }

        let left = match current.get(&0) {
            Some(&v) => v,
            None => *sib_iter.next()?,
        };
        let right = match current.get(&1) {
            Some(&v) => v,
            None => *sib_iter.next()?,
        };

        if sib_iter.next().is_some() {
            return None;
        }

        Some(hasher.root_cv(&left, &right))
    }

    /// Verify that the proof reconstructs the stored root.
    pub fn verify(&self, key: Digest) -> bool {
        self.compute_root(key) == Some(self.root)
    }

    /// Extract bytes from sparse merkle leaves.
    pub fn extract_bytes(&self, global_start: usize, length: usize) -> Result<Vec<u8>> {
        let mut result = vec![0u8; length];
        let global_end = global_start + length;
        let mut copied = 0;

        for (&leaf_idx, data) in self.leaf_indices.iter().zip(&self.leaf_data) {
            let chunk_len = data.len();
            let leaf_start = leaf_idx * chunk_len;
            let leaf_end = leaf_start + chunk_len;
            if leaf_start < global_end && global_start < leaf_end {
                let copy_start = global_start.max(leaf_start);
                let copy_end = global_end.min(leaf_end);
                let len = copy_end - copy_start;
                result[copy_start - global_start..][..len]
                    .copy_from_slice(&data[copy_start - leaf_start..][..len]);
                copied += len;
            }
        }
        ensure!(copied == length, "Not all required bytes covered by leaves");
        Ok(result)
    }

    /// Compute byte ranges covered by proof siblings.
    pub fn compute_sibling_ranges(&self, total_size: usize) -> Vec<(usize, usize, Digest)> {
        if self.leaf_indices.is_empty() || self.siblings.is_empty() {
            return Vec::new();
        }

        let mut current: BTreeSet<usize> = self.leaf_indices.iter().copied().collect();

        let chunk_len = self.chunk_len();
        let mut result = Vec::new();
        let mut proof_idx = 0;
        let mut level_size = total_size.div_ceil(chunk_len);
        let mut level = 0;

        while level_size > 1 && !current.is_empty() && proof_idx < self.siblings.len() {
            let mut next = BTreeSet::new();
            for &idx in &current {
                let sib = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
                let need_sib = if idx % 2 == 1 {
                    !current.contains(&sib)
                } else {
                    sib < level_size && !current.contains(&sib)
                };

                if need_sib && proof_idx < self.siblings.len() {
                    let chunk = chunk_len << level;
                    result.push((
                        sib * chunk,
                        ((sib + 1) * chunk).min(total_size),
                        self.siblings[proof_idx],
                    ));
                    proof_idx += 1;
                }
                next.insert(idx / 2);
            }
            current = next;
            level_size = level_size.div_ceil(2);
            level += 1;
        }
        result
    }

    /// Recursively compute a CV for a byte range from sibling ranges and leaf data.
    pub fn compute_cv(
        &self,
        start: usize,
        end: usize,
        ranges: &[(usize, usize, Digest)],
        key: Digest,
    ) -> Result<Digest> {
        if let Some(&(_, _, h)) = ranges.iter().find(|(s, e, _)| *s == start && *e == end) {
            return Ok(h);
        }
        let chunk_len = self.chunk_len();
        if end - start == chunk_len {
            let data = self.extract_bytes(start, end - start)?;
            return Ok(Blake3Hasher::with_key(key).chunk_cv(&data, (start / chunk_len) as u64));
        }
        let mid = start + (end - start).next_power_of_two() / 2;
        let hasher = Blake3Hasher::with_key(key);
        let left = self.compute_cv(start, mid, ranges, key)?;
        let right = self.compute_cv(mid, end, ranges, key)?;
        Ok(hasher.parent_cv(&left, &right))
    }
}

// ============================================================================
// Serde: chunk_1024 (cert v1–v3 PlainProof) vs variable_chunk (cert v4 PlainProofV4)
// ============================================================================

#[cfg(feature = "serde")]
impl MerkleProof {
    /// Certificate v1–v3 `PlainProof` encoding: length-prefixed leaves, each 1024 bytes.
    pub fn serialize_chunk_1024<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        Self::check_chunk_1024(&self.leaf_data).map_err(serde::ser::Error::custom)?;
        self.serialize_fields(serializer)
    }

    /// Certificate v1–v3 `PlainProof` encoding.
    pub fn deserialize_chunk_1024<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        Self::deserialize_fields(deserializer, Self::check_chunk_1024)
    }

    /// Certificate v4 `PlainProofV4` encoding: length-prefixed leaves, equal length in
    /// [`ALLOWED_CHUNK_LENS`].
    pub fn serialize_variable_chunk<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        Self::check_equal_allowed_leaves(&self.leaf_data).map_err(serde::ser::Error::custom)?;
        self.serialize_fields(serializer)
    }

    /// Certificate v4 `PlainProofV4` encoding. Rejects proofs that fail [`Self::sanity_check`].
    pub fn deserialize_variable_chunk<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        let proof = Self::deserialize_fields(deserializer, Self::check_equal_allowed_leaves)?;
        proof.sanity_check().map_err(serde::de::Error::custom)?;
        Ok(proof)
    }

    fn serialize_fields<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("MerkleProof", 5)?;
        s.serialize_field("leaf_data", &self.leaf_data)?;
        s.serialize_field("leaf_indices", &self.leaf_indices)?;
        s.serialize_field("total_leaves", &self.total_leaves)?;
        s.serialize_field("root", &self.root)?;
        s.serialize_field("siblings", &self.siblings)?;
        s.end()
    }

    fn deserialize_fields<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
        check: fn(&[Vec<u8>]) -> Result<(), String>,
    ) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Fields {
            leaf_data: Vec<Vec<u8>>,
            leaf_indices: Vec<usize>,
            total_leaves: usize,
            root: Digest,
            siblings: Vec<Digest>,
        }
        use serde::Deserialize;
        let fields = Fields::deserialize(deserializer)?;
        check(&fields.leaf_data).map_err(serde::de::Error::custom)?;
        Ok(Self {
            leaf_data: fields.leaf_data,
            leaf_indices: fields.leaf_indices,
            total_leaves: fields.total_leaves,
            root: fields.root,
            siblings: fields.siblings,
        })
    }

    fn check_chunk_1024(leaf_data: &[Vec<u8>]) -> Result<(), String> {
        if leaf_data.iter().any(|leaf| leaf.len() != CHUNK_LEN) {
            Err(format!(
                "chunk_1024 MerkleProof leaves must be {CHUNK_LEN} bytes"
            ))
        } else {
            Ok(())
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data(size: usize) -> Vec<u8> {
        (0..size).map(|i| (i % 256) as u8).collect()
    }

    fn test_key() -> Digest {
        *b"0123456789abcdef0123456789abcdef"
    }

    // ---- Tree construction ----

    #[test]
    fn test_root_matches_blake3_hash() {
        let key = test_key();
        for num_chunks in [1, 2, 3, 5, 7, 10] {
            let data = test_data(num_chunks * CHUNK_LEN);
            let tree = MerkleTree::new(&data, key);
            let expected = Blake3Hasher::with_key(key).hash(&data);
            assert_eq!(
                tree.root(),
                expected,
                "Root mismatch for {num_chunks} chunks"
            );
        }
    }

    #[test]
    fn test_root_matches_partial_last_chunk() {
        let key = test_key();
        for size in [1, 100, 1023, 1025, 2000, 3000, 7777] {
            let data = test_data(size);
            let tree = MerkleTree::new(&data, key);
            let expected = Blake3Hasher::with_key(key).hash(&data);
            assert_eq!(tree.root(), expected, "Root mismatch for size {size}");
        }
    }

    #[test]
    fn test_leaf_count() {
        let key = test_key();
        for (size, expected_leaves) in [(1024, 1), (2048, 2), (3072, 3), (1025, 2), (100, 1)] {
            let data = test_data(size);
            let tree = MerkleTree::new(&data, key);
            assert_eq!(
                tree.num_leaves(),
                expected_leaves,
                "Wrong leaf count for size {size}"
            );
        }
    }

    #[test]
    fn test_single_leaf_tree() {
        let key = test_key();
        let data = test_data(CHUNK_LEN);
        let tree = MerkleTree::new(&data, key);
        assert_eq!(tree.key(), key);
        assert_eq!(tree.num_leaves(), 1);
        assert_eq!(tree.root(), tree.leaf_hashes()[0]);
    }

    // ---- Proof generation + verification ----

    #[test]
    fn test_single_leaf_proof() {
        let data = test_data(8 * CHUNK_LEN);
        let tree = MerkleTree::new(&data, test_key());

        for idx in [0, 2, 4, 7] {
            let proof = tree.get_multileaf_proof(&[idx]);
            assert!(
                proof.verify(tree.key()),
                "Single leaf proof failed for index {idx}"
            );
        }
    }

    #[test]
    fn test_single_chunk_tree_proof_roundtrip() {
        // A tree with exactly one chunk: the root is the ROOT-finalized hash
        // of the chunk, and a proof opening it must verify (regression:
        // compute_root used to return the non-root chunk CV).
        let key = test_key();
        let data = test_data(CHUNK_LEN);
        let tree = MerkleTree::new(&data, key);
        let proof = tree.get_multileaf_proof(&[0]);
        assert_eq!(proof.total_leaves, 1);
        assert_eq!(proof.compute_root(key), Some(tree.root()));
        assert!(proof.verify(key));

        // Tampered chunk bytes fail closed.
        let mut tampered = proof;
        tampered.leaf_data[0][0] ^= 1;
        assert!(!tampered.verify(key));
    }

    #[test]
    fn test_multi_leaf_proof_consecutive() {
        let data = test_data(8 * CHUNK_LEN);
        let tree = MerkleTree::new(&data, test_key());

        let proof = tree.get_multileaf_proof(&[0, 1]);
        assert!(proof.verify(tree.key()));

        let proof = tree.get_multileaf_proof(&[2, 3, 4]);
        assert!(proof.verify(tree.key()));
    }

    #[test]
    fn test_multi_leaf_proof_non_consecutive() {
        let key = test_key();
        let data = test_data(8 * CHUNK_LEN);
        let tree = MerkleTree::new(&data, key);

        for indices in [
            vec![0, 2],
            vec![0, 3],
            vec![1, 3, 5],
            vec![0, 1, 4, 5],
            vec![0, 2, 3, 7],
        ] {
            let proof = tree.get_multileaf_proof(&indices);
            assert!(proof.verify(key), "Proof failed for indices {indices:?}");
        }
    }

    #[test]
    fn test_proof_dedup_and_ordering() {
        let key = test_key();
        let data = test_data(8 * CHUNK_LEN);
        let tree = MerkleTree::new(&data, key);

        let proof1 = tree.get_multileaf_proof(&[5, 3, 4, 3]);
        let proof2 = tree.get_multileaf_proof(&[3, 4, 5]);
        assert_eq!(proof1.siblings, proof2.siblings);
        assert!(proof1.verify(key));
    }

    #[test]
    fn test_full_tree_proof() {
        let key = test_key();
        let data = test_data(8 * CHUNK_LEN);
        let tree = MerkleTree::new(&data, key);

        let all: Vec<usize> = (0..tree.num_leaves()).collect();
        let proof = tree.get_multileaf_proof(&all);
        assert!(proof.verify(key));
        assert!(proof.siblings.is_empty());
    }

    #[test]
    fn test_small_tree_3_leaves() {
        let key = test_key();
        let data = test_data(3 * CHUNK_LEN);
        let tree = MerkleTree::new(&data, key);
        assert_eq!(tree.num_leaves(), 3);

        for i in 0..3 {
            let proof = tree.get_multileaf_proof(&[i]);
            assert!(proof.verify(key), "3-leaf tree: single leaf {i} failed");
        }

        let proof = tree.get_multileaf_proof(&[0, 2]);
        assert!(proof.verify(key));
    }

    #[test]
    fn test_4_leaf_proof_lengths() {
        let key = test_key();
        let data = test_data(4 * CHUNK_LEN);
        let tree = MerkleTree::new(&data, key);

        let adjacent = tree.get_multileaf_proof(&[0, 1]);
        let cross = tree.get_multileaf_proof(&[0, 2]);
        let all = tree.get_multileaf_proof(&[0, 1, 2, 3]);

        assert_eq!(adjacent.siblings.len(), 1);
        assert_eq!(cross.siblings.len(), 2);
        assert_eq!(all.siblings.len(), 0);
    }

    // ---- Verification rejection ----

    #[test]
    fn test_reject_wrong_leaf_data() {
        let key = test_key();
        let data = test_data(8 * CHUNK_LEN);
        let tree = MerkleTree::new(&data, key);

        let mut proof = tree.get_multileaf_proof(&[2, 3]);
        proof.leaf_data[0] = vec![0xFFu8; CHUNK_LEN];
        assert!(!proof.verify(key));
    }

    #[test]
    fn test_reject_wrong_root() {
        let key = test_key();
        let data = test_data(8 * CHUNK_LEN);
        let tree = MerkleTree::new(&data, key);

        let mut proof = tree.get_multileaf_proof(&[1, 2]);
        proof.root = [0xAA; OUT_LEN];
        assert!(!proof.verify(key));
    }

    #[test]
    fn test_reject_wrong_indices() {
        let key = test_key();
        let data = test_data(8 * CHUNK_LEN);
        let tree = MerkleTree::new(&data, key);

        let mut proof = tree.get_multileaf_proof(&[2, 3]);
        proof.leaf_indices = vec![3, 4];
        assert!(!proof.verify(key));
    }

    #[test]
    fn test_reject_extra_siblings() {
        let key = test_key();
        let data = test_data(8 * CHUNK_LEN);
        let tree = MerkleTree::new(&data, key);

        let mut proof = tree.get_multileaf_proof(&[0, 2, 5]);
        proof.siblings.push([0xBB; OUT_LEN]);
        assert!(!proof.verify(key));
    }

    #[test]
    fn test_reject_out_of_range_leaf_indices() {
        // Regression (security): with an index >= total_leaves, the final
        // combine used to take both children from the sibling list, so the
        // root was reconstructed without ever hashing the supplied leaf data
        // and verify() accepted leaf bytes the root does not commit to.
        let key = test_key();
        let data = test_data(2 * CHUNK_LEN);
        let tree = MerkleTree::new(&data, key);

        let forged = MerkleProof {
            leaf_data: vec![vec![0xEE; CHUNK_LEN]], // arbitrary, never hashed pre-fix
            leaf_indices: vec![2],                  // out of range: tree has 2 leaves
            total_leaves: 2,
            root: tree.root(),
            siblings: tree.leaf_hashes().to_vec(), // real CVs fill positions 0 and 1
        };
        assert!(forged.sanity_check().is_err());
        assert_eq!(forged.compute_root(key), None);
        assert!(!forged.verify(key));

        // Same forgery one level deeper: the out-of-range leaf survives one
        // merge round and is then silently dropped at the final combine.
        let data = test_data(4 * CHUNK_LEN);
        let tree = MerkleTree::new(&data, key);
        let forged = MerkleProof {
            leaf_data: vec![vec![0xEE; CHUNK_LEN]],
            leaf_indices: vec![4], // out of range: tree has 4 leaves
            total_leaves: 4,
            root: tree.root(),
            siblings: tree.layers[1].clone(), // both level-1 CVs
        };
        assert!(forged.sanity_check().is_err());
        assert_eq!(forged.compute_root(key), None);
        assert!(!forged.verify(key));
    }

    #[test]
    fn test_compute_root_rejects_malformed_shape() {
        let key = test_key();
        let data = test_data(4 * CHUNK_LEN);
        let tree = MerkleTree::new(&data, key);

        // More indices than leaf data: the index/hash zip would silently drop
        // the extra index instead of failing.
        let mut mismatched = tree.get_multileaf_proof(&[0]);
        mismatched.leaf_indices = vec![0, 1];
        assert_eq!(mismatched.compute_root(key), None);
        assert!(!mismatched.verify(key));

        // Duplicate index: the BTreeMap would silently keep only one of the
        // two leaves' hashes, leaving the other leaf unverified.
        let mut duplicated = tree.get_multileaf_proof(&[0, 1]);
        duplicated.leaf_indices = vec![1, 1];
        assert_eq!(duplicated.compute_root(key), None);
        assert!(!duplicated.verify(key));
    }

    // ---- Randomized round-trips ----

    #[test]
    fn test_random_round_trips() {
        use std::collections::HashSet;
        let key = test_key();

        for seed in 0u64..20 {
            let n = 3 + (seed % 20) as usize;
            let data = test_data(n * CHUNK_LEN);
            let tree = MerkleTree::new(&data, key);

            let num_selected = 1 + (seed as usize % tree.num_leaves());
            let mut indices: HashSet<usize> = HashSet::new();
            let mut val = seed;
            while indices.len() < num_selected {
                val = val.wrapping_mul(6364136223846793005).wrapping_add(1);
                indices.insert((val as usize) % tree.num_leaves());
            }
            let mut indices: Vec<usize> = indices.into_iter().collect();
            indices.sort();

            let proof = tree.get_multileaf_proof(&indices);
            assert!(
                proof.verify(key),
                "Random round-trip failed: n={n}, indices={indices:?}"
            );
        }
    }

    // ---- compute_leaf_indices_from_rows ----

    #[test]
    fn test_leaf_indices_from_rows() {
        // 4 rows of 1024 bytes each = 4 leaves, 1:1 mapping
        let indices =
            MerkleTree::compute_leaf_indices_from_rows(&[0, 2], (4, CHUNK_LEN), CHUNK_LEN).unwrap();
        assert_eq!(indices, vec![0, 2]);

        // 4 rows of 512 bytes each = 2 leaves (2 rows per leaf)
        let indices =
            MerkleTree::compute_leaf_indices_from_rows(&[0, 2], (4, 512), CHUNK_LEN).unwrap();
        assert_eq!(indices, vec![0, 1]);

        // Row spans two leaves
        let indices =
            MerkleTree::compute_leaf_indices_from_rows(&[0], (2, 1500), CHUNK_LEN).unwrap();
        assert_eq!(indices, vec![0, 1]);
    }

    // ---- extract_bytes ----

    #[test]
    fn test_extract_bytes() {
        let key = test_key();
        let data = test_data(4 * CHUNK_LEN);
        let tree = MerkleTree::new(&data, key);
        let proof = tree.get_multileaf_proof(&[1, 2]);

        let extracted = proof.extract_bytes(CHUNK_LEN, 64).unwrap();
        assert_eq!(extracted, &data[CHUNK_LEN..CHUNK_LEN + 64]);

        // Spanning two leaves
        let extracted = proof.extract_bytes(2 * CHUNK_LEN - 32, 64).unwrap();
        assert_eq!(extracted, &data[2 * CHUNK_LEN - 32..2 * CHUNK_LEN + 32]);
    }

    #[test]
    fn test_extract_bytes_fails_for_missing_leaves() {
        let key = test_key();
        let data = test_data(4 * CHUNK_LEN);
        let tree = MerkleTree::new(&data, key);
        let proof = tree.get_multileaf_proof(&[1]);

        // Leaf 0 not in proof
        assert!(proof.extract_bytes(0, 64).is_err());
    }

    // ---- compute_sibling_ranges + compute_cv ----

    #[test]
    fn test_compute_sibling_ranges() {
        let key = test_key();
        let data = test_data(4 * CHUNK_LEN);
        let tree = MerkleTree::new(&data, key);
        let proof = tree.get_multileaf_proof(&[0]);

        let total_size = data.len();
        let ranges = proof.compute_sibling_ranges(total_size);
        assert!(!ranges.is_empty());

        for (start, end, _hash) in &ranges {
            assert!(*start < *end);
            assert!(*end <= total_size);
        }
    }

    #[test]
    fn test_compute_cv_round_trip() {
        let key = test_key();
        let data = test_data(4 * CHUNK_LEN);
        let tree = MerkleTree::new(&data, key);

        let all_indices: Vec<usize> = (0..4).collect();
        let proof = tree.get_multileaf_proof(&all_indices);
        let ranges = proof.compute_sibling_ranges(data.len());

        let cv = proof.compute_cv(0, CHUNK_LEN, &ranges, key).unwrap();
        let expected = Blake3Hasher::with_key(key).chunk_cv(&data[..CHUNK_LEN], 0);
        assert_eq!(cv, expected);
    }

    // ---- Sanity check ----

    #[test]
    fn test_sanity_check() {
        let proof = MerkleProof {
            leaf_data: vec![vec![0u8; CHUNK_LEN], vec![1u8; CHUNK_LEN]],
            leaf_indices: vec![0, 2],
            total_leaves: 4,
            root: [0u8; OUT_LEN],
            siblings: vec![],
        };
        assert!(proof.sanity_check().is_ok());

        let bad = MerkleProof {
            leaf_data: vec![],
            leaf_indices: vec![],
            total_leaves: 0,
            root: [0u8; OUT_LEN],
            siblings: vec![],
        };
        assert!(bad.sanity_check().is_err());

        let unsorted = MerkleProof {
            leaf_data: vec![vec![0u8; CHUNK_LEN], vec![1u8; CHUNK_LEN]],
            leaf_indices: vec![2, 0],
            total_leaves: 4,
            root: [0u8; OUT_LEN],
            siblings: vec![],
        };
        assert!(unsorted.sanity_check().is_err());

        let out_of_range = MerkleProof {
            leaf_data: vec![vec![0u8; CHUNK_LEN]],
            leaf_indices: vec![4],
            total_leaves: 4,
            root: [0u8; OUT_LEN],
            siblings: vec![],
        };
        assert!(out_of_range.sanity_check().is_err());
    }

    #[test]
    fn test_padded_chunk_len() {
        assert_eq!(padded_chunk_len(0), 0);
        assert_eq!(padded_chunk_len(1), CHUNK_LEN);
        assert_eq!(padded_chunk_len(CHUNK_LEN - 1), CHUNK_LEN);
        assert_eq!(padded_chunk_len(CHUNK_LEN), CHUNK_LEN);
        assert_eq!(padded_chunk_len(CHUNK_LEN + 1), 2 * CHUNK_LEN);
        assert_eq!(padded_chunk_len(3 * CHUNK_LEN), 3 * CHUNK_LEN);
    }

    #[test]
    fn test_pad_to_chunk_boundary() {
        assert!(pad_to_chunk_boundary(&[]).is_empty());

        let single_byte = pad_to_chunk_boundary(&[1]);
        assert_eq!(single_byte.len(), CHUNK_LEN);
        assert_eq!(single_byte[0], 1);
        assert_eq!(single_byte[1], 0);

        let aligned = test_data(CHUNK_LEN);
        assert_eq!(pad_to_chunk_boundary(&aligned), aligned);

        let unaligned = test_data(CHUNK_LEN + 1);
        let padded = pad_to_chunk_boundary(&unaligned);
        assert_eq!(padded.len(), 2 * CHUNK_LEN);
        assert_eq!(&padded[..CHUNK_LEN + 1], &unaligned[..]);
        assert!(padded[CHUNK_LEN + 1..].iter().all(|&b| b == 0));
    }

    #[test]
    fn test_padded_tree_root_differs_from_unpadded() {
        let key = test_key();
        let data = test_data(CHUNK_LEN + 500);
        let padded = pad_to_chunk_boundary(&data);

        let tree_raw = MerkleTree::new(&data, key);
        let tree_padded = MerkleTree::new(&padded, key);

        assert_ne!(
            tree_raw.root(),
            tree_padded.root(),
            "Padded and unpadded trees must have different roots for non-aligned data"
        );
        assert_eq!(tree_padded.num_leaves(), 2);
        assert_eq!(tree_raw.num_leaves(), 2);
    }

    #[test]
    fn allowed_chunk_lens() {
        assert_eq!(ALLOWED_CHUNK_LENS, [128, 256, 512, CHUNK_LEN]);
        assert!(is_allowed_chunk_len(128));
        assert!(is_allowed_chunk_len(CHUNK_LEN));
        assert!(!is_allowed_chunk_len(64));
        assert!(MerkleTree::with_chunk_len(&[0u8; 64], test_key(), 64).is_err());
        assert!(MerkleTree::compute_leaf_indices_from_rows(&[0], (1, 64), 64).is_err());
    }

    #[test]
    fn test_variable_chunk_len_round_trip() {
        let key = test_key();
        for chunk_len in ALLOWED_CHUNK_LENS {
            let raw_len = chunk_len * 3 + 17;
            let mut data = test_data(raw_len);
            data.resize(raw_len.div_ceil(chunk_len) * chunk_len, 0);
            let tree = MerkleTree::with_chunk_len(&data, key, chunk_len).unwrap();
            assert_eq!(tree.num_leaves(), data.len() / chunk_len);

            let proof = tree.get_multileaf_proof(&[0, tree.num_leaves() - 1]);
            assert_eq!(proof.chunk_len(), chunk_len);
            assert!(proof.verify(key));
            assert_eq!(proof.leaf_data[0].len(), chunk_len);
        }
    }
}
