//! Merkle commitment over raw FP16 operand rows, with minimal multi-row openings.
//!
//! The FP16 scheme commits each operand as a keyed-BLAKE3 Merkle tree directly
//! over its FP16 rows (`u16`, little-endian, row-major) — there is no int8 +
//! block-scale prequant layer, so the committed leaves ARE the FP16 values. This
//! is strictly simpler than the FP8 commitment ([`crate::v4::api::openings`]),
//! which commits a separate values tree and scales tree and runs the compiled
//! Blake program; here one tree per operand suffices.
//!
//! The tree/key/hash-id discipline matches FP8: leaves are fixed-size chunks
//! ([`HashId::chunk_len`]) of the zero-padded row bytes, built under a per-tree
//! key. An opening discloses exactly the selected tile rows as the unique-minimal
//! leaf/sibling set; [`verify_and_open_rows`] rebuilds the root under the key,
//! checks it against the claimed root, and extracts the opened FP16 rows.

use anyhow::{Context, Result, ensure};
use pearl_blake3::{MerkleProof, MerkleTree};

use super::dtype::fp16_to_f32;
use crate::v4::api::public_params::HashId;
use crate::v4::api::primitives::Hash256;
use crate::ensure_eq;

/// Bytes per committed FP16 row: `k` values, 2 bytes each (little-endian).
pub fn row_bytes(k: usize) -> usize {
    k * 2
}

/// Little-endian byte image of an FP16 row-major matrix (the committed leaf data,
/// before hash-id padding).
pub fn rows_to_bytes(rows: &[u16]) -> Vec<u8> {
    rows.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Builds the keyed Merkle tree committing `rows` (`num_rows x k` FP16 values,
/// row-major) under `key`, with leaves of `hash_id` chunk length.
pub fn commit_operand(rows: &[u16], num_rows: usize, k: usize, hash_id: HashId, key: Hash256) -> Result<MerkleTree> {
    ensure!(rows.len() == num_rows * k, "operand is not num_rows x k ({} != {num_rows}*{k})", rows.len());
    let bytes = rows_to_bytes(rows);
    MerkleTree::with_chunk_len(&hash_id.pad(&bytes), key, hash_id.chunk_len())
}

/// Opens exactly the `row_indices` rows of a committed tree: the unique-minimal
/// leaf/sibling set for those rows. `row_indices` must be valid rows of the
/// `num_rows x k` matrix.
pub fn open_rows(tree: &MerkleTree, row_indices: &[usize], num_rows: usize, k: usize, hash_id: HashId) -> Result<MerkleProof> {
    ensure!(!row_indices.is_empty(), "must open at least one row");
    ensure!(
        row_indices.iter().all(|&r| r < num_rows),
        "opened row index out of range for {num_rows} rows"
    );
    let rb = row_bytes(k);
    let leaves = MerkleTree::compute_leaf_indices_from_rows(row_indices, (num_rows, rb), hash_id.chunk_len())?;
    Ok(tree.get_multileaf_proof(&leaves))
}

/// Authenticates `proof` as the unique-minimal opening of `row_indices` from the
/// `num_rows x k` operand committed under `key` with root `claimed_root`, then
/// extracts and returns the opened FP16 rows (flattened, in `row_indices` order,
/// `row_indices.len() * k` values).
///
/// Rejects: a leaf set that is not the unique-minimal one for the opened rows, a
/// wrong total-leaf count, a root that does not reconstruct under `key` (wrong
/// key / wrong header / tampered leaf or sibling), a root that disagrees with the
/// claimed root, and any opened FP16 value that is NaN/±inf.
pub fn verify_and_open_rows(
    proof: &MerkleProof,
    row_indices: &[usize],
    num_rows: usize,
    k: usize,
    hash_id: HashId,
    key: Hash256,
    claimed_root: &Hash256,
) -> Result<Vec<u16>> {
    ensure!(!row_indices.is_empty(), "must open at least one row");
    ensure!(
        row_indices.iter().all(|&r| r < num_rows),
        "opened row index out of range for {num_rows} rows"
    );
    let chunk_len = hash_id.chunk_len();
    let rb = row_bytes(k);

    // Structural gates before any Merkle work (sanity_check bounds proof fields).
    proof.sanity_check().context("FP16 operand: merkle proof structure")?;
    if let Some(leaf) = proof.leaf_data.first() {
        ensure_eq!(leaf.len(), chunk_len, "FP16 operand: leaf length must match hash_id chunk_len");
    }
    let tree_bytes = num_rows
        .checked_mul(rb)
        .ok_or_else(|| anyhow::anyhow!("FP16 operand: matrix byte length overflow"))?;
    ensure_eq!(
        proof.total_leaves,
        hash_id.padded_len(tree_bytes) / chunk_len,
        "FP16 operand: total_leaves mismatch"
    );

    // The opened leaves must be exactly the unique-minimal set for these rows.
    let minimal = MerkleTree::compute_leaf_indices_from_rows(row_indices, (num_rows, rb), chunk_len)?;
    ensure_eq!(
        proof.leaf_indices,
        minimal,
        "FP16 operand: leaf set is not the unique minimal set for the opened rows"
    );
    ensure_eq!(
        proof.leaf_data.len(),
        minimal.len(),
        "FP16 operand: leaf_data count must match the unique leaf set"
    );

    // Reconstruct the root under the key and bind it to the claimed root.
    let root = proof
        .compute_root(key)
        .ok_or_else(|| anyhow::anyhow!("FP16 operand: Merkle reconstruction failed"))?;
    ensure_eq!(&root, claimed_root, "FP16 operand: reconstructed root != claimed root");
    ensure_eq!(&proof.root, claimed_root, "FP16 operand: proof root != claimed root");

    // Extract the opened rows and decode FP16 (rejecting NaN/inf).
    let mut out = Vec::with_capacity(row_indices.len() * k);
    for &idx in row_indices {
        let start = idx.checked_mul(rb).ok_or_else(|| anyhow::anyhow!("row offset overflow"))?;
        let bytes = proof.extract_bytes(start, rb).context("extract FP16 row")?;
        for pair in bytes.as_chunks::<2>().0 {
            let v = u16::from_le_bytes(*pair);
            ensure!((v >> 10) & 0x1F != 0x1F, "opened FP16 value {v:#06x} is NaN/inf");
            out.push(v);
        }
    }
    debug_assert_eq!(out.len(), row_indices.len() * k);
    // Touch the decoder so a future layout change that breaks decoding is caught.
    debug_assert!(out.iter().all(|&v| fp16_to_f32(v).is_finite()));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::dtype::f32_to_fp16;

    const KEY: Hash256 = [7u8; 32];

    /// Valid (finite) FP16 row-major matrix with spread magnitudes.
    fn matrix(num_rows: usize, k: usize) -> Vec<u16> {
        (0..num_rows * k)
            .map(|i| f32_to_fp16(((i * 37 % 193) as f32 - 96.0) * 0.5).unwrap())
            .collect()
    }

    #[test]
    fn round_trips_the_opened_rows() {
        let (num_rows, k) = (8usize, 256usize);
        let rows = matrix(num_rows, k);
        for hash_id in HashId::ALL {
            let tree = commit_operand(&rows, num_rows, k, hash_id, KEY).unwrap();
            let open_idx = vec![0usize, 3, 7];
            let proof = open_rows(&tree, &open_idx, num_rows, k, hash_id).unwrap();
            let opened = verify_and_open_rows(&proof, &open_idx, num_rows, k, hash_id, KEY, &tree.root()).unwrap();
            assert_eq!(opened.len(), open_idx.len() * k);
            for (slot, &row) in open_idx.iter().enumerate() {
                assert_eq!(&opened[slot * k..slot * k + k], &rows[row * k..row * k + k], "row {row}");
            }
        }
    }

    #[test]
    fn wrong_key_or_root_rejected() {
        let (num_rows, k) = (8usize, 128usize);
        let rows = matrix(num_rows, k);
        let hash_id = HashId::Blake3Chunk1024;
        let tree = commit_operand(&rows, num_rows, k, hash_id, KEY).unwrap();
        let open_idx = vec![1usize, 2];
        let proof = open_rows(&tree, &open_idx, num_rows, k, hash_id).unwrap();

        // Wrong key -> reconstruction differs from the claimed root.
        assert!(verify_and_open_rows(&proof, &open_idx, num_rows, k, hash_id, [9u8; 32], &tree.root()).is_err());
        // Wrong claimed root.
        assert!(verify_and_open_rows(&proof, &open_idx, num_rows, k, hash_id, KEY, &[0u8; 32]).is_err());
        // Honest opening still passes.
        assert!(verify_and_open_rows(&proof, &open_idx, num_rows, k, hash_id, KEY, &tree.root()).is_ok());
    }

    #[test]
    fn tampered_leaf_rejected() {
        let (num_rows, k) = (8usize, 128usize);
        let rows = matrix(num_rows, k);
        let hash_id = HashId::Blake3Chunk1024;
        let tree = commit_operand(&rows, num_rows, k, hash_id, KEY).unwrap();
        let open_idx = vec![0usize, 4];
        let mut proof = open_rows(&tree, &open_idx, num_rows, k, hash_id).unwrap();
        proof.leaf_data[0][0] ^= 0xFF;
        assert!(verify_and_open_rows(&proof, &open_idx, num_rows, k, hash_id, KEY, &tree.root()).is_err());
    }

    #[test]
    fn wrong_leaf_set_rejected() {
        let (num_rows, k) = (8usize, 128usize);
        let rows = matrix(num_rows, k);
        let hash_id = HashId::Blake3Chunk1024;
        let tree = commit_operand(&rows, num_rows, k, hash_id, KEY).unwrap();
        // Open rows [0,4] but claim we opened [1,2]: the minimal leaf set won't match.
        let proof = open_rows(&tree, &[0usize, 4], num_rows, k, hash_id).unwrap();
        assert!(verify_and_open_rows(&proof, &[1usize, 2], num_rows, k, hash_id, KEY, &tree.root()).is_err());
    }

    /// Additive oracle dump for the sm_80 miner commitment kernel. Prints, for
    /// several operand shapes / keys / hash-ids, the committed root (and a few
    /// leaf-hash bytes) so the GPU kernel can be asserted bit-exact against it.
    /// `cargo test -p zk-pow api::fp16::commitment::dump_commit_oracle -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn dump_commit_oracle() {
        // Deterministic raw u16 rows (arbitrary bit patterns; commit does not
        // validate finiteness), reproducible in the Python/GPU oracle.
        fn gen_rows(n: usize, seed: u32) -> Vec<u16> {
            (0..n)
                .map(|i| ((i as u32).wrapping_mul(40503).wrapping_add(seed) & 0xFFFF) as u16)
                .collect()
        }
        let cases: &[(usize, usize, u8, u8)] = &[
            // (num_rows, k, chunk_len_id(0..3), key_byte)
            (1, 1, 3, 0x11),   // tiny single-leaf
            (2, 64, 3, 0x22),  // 256 bytes, single 1024 leaf
            (8, 256, 3, 0x07), // 4096 bytes = 4 leaves of 1024
            (8, 256, 0, 0x5A), // same bytes, 128-byte leaves (32 leaves)
            (8, 256, 1, 0xA5), // 256-byte leaves (16 leaves)
            (8, 256, 2, 0x33), // 512-byte leaves (8 leaves)
            (5, 100, 2, 0x44), // 1000 bytes, 512-byte leaves -> padded 1024 -> 2 leaves
            (7, 333, 0, 0x99), // odd shapes, 128-byte leaves
            (13, 777, 3, 0xFE),// non-power-of-two leaf count, 1024-byte leaves
            (3, 1500, 1, 0x80),// 9000 bytes, 256-byte leaves
        ];
        for (idx, &(num_rows, k, cl_id, kb)) in cases.iter().enumerate() {
            let hash_id = HashId::try_from(cl_id).unwrap();
            let key = [kb; 32];
            let rows = gen_rows(num_rows * k, (idx as u32).wrapping_mul(2654435761));
            let tree = commit_operand(&rows, num_rows, k, hash_id, key).unwrap();
            let root = tree.root();
            let seed = (idx as u32).wrapping_mul(2654435761);
            let root_hex: String = root.iter().map(|b| format!("{b:02x}")).collect();
            println!("ORACLE {num_rows} {k} {} {kb} {seed} {root_hex}", hash_id.chunk_len());
        }
    }

    #[test]
    fn nan_value_rejected() {
        let (num_rows, k) = (2usize, 64usize);
        let mut rows = matrix(num_rows, k);
        rows[0] = 0x7C01; // FP16 NaN (exp=0x1F, man!=0)
        let hash_id = HashId::Blake3Chunk1024;
        let tree = commit_operand(&rows, num_rows, k, hash_id, KEY).unwrap();
        let proof = open_rows(&tree, &[0usize], num_rows, k, hash_id).unwrap();
        let err = verify_and_open_rows(&proof, &[0usize], num_rows, k, hash_id, KEY, &tree.root()).unwrap_err();
        assert!(err.to_string().contains("NaN/inf"), "got: {err}");
    }
}
