//! Merkle tree construction and proof generation/verification.
//!
//! `MerkleTree` builds a BLAKE3 Merkle tree from raw bytes and generates multi-leaf proofs.
//! `MerkleProof` verifies proofs and provides byte extraction utilities.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{ensure, Result};
use blake3::{CHUNK_LEN, OUT_LEN};
use rayon::prelude::*;

use crate::hasher::{Blake3Hasher, Digest};

/// Round `raw_len` up to the next multiple of `CHUNK_LEN` (1024).
pub fn padded_chunk_len(raw_len: usize) -> usize {
    raw_len.div_ceil(CHUNK_LEN) * CHUNK_LEN
}

/// Zero-pad `data` so its length is a multiple of `CHUNK_LEN` (1024).
///
/// Matrix data must be padded to a BLAKE3 chunk boundary before building a
/// Merkle tree.  Uses [`padded_chunk_len`] for the target size.
pub fn pad_to_chunk_boundary(data: &[u8]) -> Vec<u8> {
    let mut padded = data.to_vec();
    padded.resize(padded_chunk_len(data.len()), 0);
    padded
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
}

impl MerkleTree {
    /// Build a Merkle tree from `data` using keyed BLAKE3.
    pub fn new(data: &[u8], key: Digest) -> Self {
        let hasher = Blake3Hasher::with_key(key);
        if data.is_empty() {
            return Self {
                key,
                layers: vec![vec![]],
                data: vec![],
            };
        }

        // Single chunk or less: hash directly
        if data.len() <= CHUNK_LEN {
            let root = hasher.hash(data);
            return Self {
                key,
                layers: vec![vec![root]],
                data: data.to_vec(),
            };
        }

        let chunk_cvs = hasher.hash_chunks(data);
        let mut layers: Vec<Vec<Digest>> = vec![chunk_cvs];

        while layers.last().unwrap().len() > 2 {
            let prev = layers.last().unwrap();
            layers.push(hasher.combine_layer(prev));
        }

        let last = layers.last().unwrap();
        if last.len() == 2 {
            let root = hasher.root_cv(&last[0], &last[1]);
            layers.push(vec![root]);
        }

        Self {
            key,
            layers,
            data: data.to_vec(),
        }
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
        let leaf_data: Vec<[u8; CHUNK_LEN]> = sorted_indices
            .iter()
            .map(|&i| {
                let start = i * CHUNK_LEN;
                let end = (start + CHUNK_LEN).min(self.data.len());
                let mut chunk = [0u8; CHUNK_LEN];
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
    pub fn compute_leaf_indices_from_rows(
        row_indices: &[usize],
        shape: (usize, usize),
    ) -> Vec<usize> {
        let cols = shape.1;
        let mut indices = BTreeSet::new();
        for &row in row_indices {
            let first = (row * cols) / CHUNK_LEN;
            let last = ((row + 1) * cols - 1) / CHUNK_LEN;
            for i in first..=last {
                indices.insert(i);
            }
        }
        indices.into_iter().collect()
    }
}

// ============================================================================
// MerkleProof
// ============================================================================

/// Generic multi-leaf Merkle proof, domain-agnostic.
#[derive(Clone)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "MerkleProof"))]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(crate = "serde")
)]
pub struct MerkleProof {
    #[cfg_attr(feature = "serde", serde(with = "serde_chunk_vec"))]
    pub leaf_data: Vec<[u8; CHUNK_LEN]>,
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
    /// Validate proof structure.
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
        Ok(())
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
    /// Returns `None` if the proof has incorrect shape.
    pub fn compute_root(&self, key: Digest) -> Option<Digest> {
        if self.leaf_indices.is_empty() {
            return None;
        }

        let hasher = Blake3Hasher::with_key(key);
        let leaf_hashes = self.leaf_hashes(key);

        let mut current: BTreeMap<usize, Digest> =
            self.leaf_indices.iter().copied().zip(leaf_hashes).collect();
        let mut level_len = self.total_leaves;

        if level_len == 1 {
            return if self.siblings.is_empty() {
                Some(current[&0])
            } else {
                None
            };
        }

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
            let leaf_start = leaf_idx * CHUNK_LEN;
            let leaf_end = leaf_start + CHUNK_LEN;
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

        let mut result = Vec::new();
        let mut proof_idx = 0;
        let mut level_size = total_size.div_ceil(CHUNK_LEN);
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
                    let chunk = CHUNK_LEN << level;
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
        if end - start == CHUNK_LEN {
            let data = self.extract_bytes(start, end - start)?;
            return Ok(Blake3Hasher::with_key(key).chunk_cv(&data, (start / CHUNK_LEN) as u64));
        }
        let mid = start + (end - start).next_power_of_two() / 2;
        let hasher = Blake3Hasher::with_key(key);
        let left = self.compute_cv(start, mid, ranges, key)?;
        let right = self.compute_cv(mid, end, ranges, key)?;
        Ok(hasher.parent_cv(&left, &right))
    }
}

// ============================================================================
// Serde helpers for fixed-size arrays
// ============================================================================

#[cfg(feature = "serde")]
mod serde_chunk_vec {
    use blake3::CHUNK_LEN;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        data: &[[u8; CHUNK_LEN]],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let vecs: Vec<&[u8]> = data.iter().map(|a| a.as_slice()).collect();
        vecs.serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<[u8; CHUNK_LEN]>, D::Error> {
        let vecs: Vec<Vec<u8>> = Vec::deserialize(deserializer)?;
        vecs.into_iter()
            .map(|v| {
                v.try_into()
                    .map_err(|_| serde::de::Error::custom("leaf data must be CHUNK_LEN bytes"))
            })
            .collect()
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
        proof.leaf_data[0] = [0xFFu8; CHUNK_LEN];
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
        let indices = MerkleTree::compute_leaf_indices_from_rows(&[0, 2], (4, CHUNK_LEN));
        assert_eq!(indices, vec![0, 2]);

        // 4 rows of 512 bytes each = 2 leaves (2 rows per leaf)
        let indices = MerkleTree::compute_leaf_indices_from_rows(&[0, 2], (4, 512));
        assert_eq!(indices, vec![0, 1]);

        // Row spans two leaves
        let indices = MerkleTree::compute_leaf_indices_from_rows(&[0], (2, 1500));
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
            leaf_data: vec![[0u8; CHUNK_LEN], [1u8; CHUNK_LEN]],
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
            leaf_data: vec![[0u8; CHUNK_LEN], [1u8; CHUNK_LEN]],
            leaf_indices: vec![2, 0],
            total_leaves: 4,
            root: [0u8; OUT_LEN],
            siblings: vec![],
        };
        assert!(unsorted.sanity_check().is_err());
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
}
