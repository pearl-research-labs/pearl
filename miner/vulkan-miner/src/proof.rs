use anyhow::Result;
use zk_pow::ffi::plain_proof::{MatrixMerkleProof, PlainProof};

// ---- Deterministic xorshift32 matching GLSL k1_random_fill.comp ----

fn xorshift_int8(state: &mut u32) -> i8 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    ((*state & 0x7F) as i8).wrapping_sub(64)
}

fn element_value(seed: u32, idx: u32) -> i8 {
    let mut state = seed ^ idx.wrapping_mul(0x9E3779B9);
    xorshift_int8(&mut state)
}

/// Re-derive the A and B matrix bytes from the iteration seed (same xorshift32
/// used by the K1 GLSL shader).  Returns `(a_bytes, bt_bytes)` where `bt_bytes`
/// is B^T (transpose of B, N×K row-major).
pub fn derive_matrix_bytes(
    seed: u32,
    m: u32,
    n: u32,
    k: u32,
) -> (Vec<u8>, Vec<u8>) {
    let a_len = (m * k) as usize;
    let b_len = (k * n) as usize;
    let mut a = vec![0u8; a_len];
    let mut b = vec![0i8; b_len];

    for idx in 0..a_len {
        a[idx] = element_value(seed, idx as u32) as u8;
    }
    for idx in 0..b_len {
        b[idx] = element_value(seed, (a_len + idx) as u32);
    }

    // Transpose B (K×N) → B^T (N×K)
    let mut bt = vec![0u8; (n * k) as usize];
    for i in 0..k as usize {
        for j in 0..n as usize {
            bt[j * k as usize + i] = b[i * n as usize + j] as u8;
        }
    }
    (a, bt)
}

/// Build a Merkle proof for a winning tile after a block is found.
///
/// `a_bytes` / `b_bytes` are the raw row-major matrix bytes (use
/// [`derive_matrix_bytes`] to get them from the seed).
pub fn build_merkle_proof(
    a_bytes: &[u8],
    b_bytes: &[u8],
    m: usize,
    n: usize,
    k: usize,
    tile_row: u32,
    tile_col: u32,
    tile_m: u32,
    tile_n: u32,
    rows_pattern: &[u32],
    cols_pattern: &[u32],
    job_key: &[u8; 32],
    noise_rank: usize,
) -> Result<PlainProof> {
    // Compute row indices within the winning tile
    let a_rows: Vec<usize> = rows_pattern
        .iter()
        .map(|&r| (tile_row * tile_m + r) as usize)
        .filter(|&r| r < m)
        .collect();

    let b_cols: Vec<usize> = cols_pattern
        .iter()
        .map(|&c| (tile_col * tile_n + c) as usize)
        .filter(|&c| c < n)
        .collect();

    let num_cols_a = k;
    let num_cols_b = k;

    let a_proof = build_matrix_proof(a_bytes, job_key, &a_rows, m, num_cols_a);
    let b_proof = build_matrix_proof(b_bytes, job_key, &b_cols, n, num_cols_b);

    Ok(PlainProof {
        m,
        n,
        k,
        noise_rank,
        a: a_proof,
        bt: b_proof,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Verify that a proof built from `derive_matrix_bytes` + `build_merkle_proof`
    /// is internally consistent: each component Merkle proof verifies against its
    /// tree root.
    #[test]
    fn test_proof_round_trip() {
        let job_key = *blake3::hash(b"test_key").as_bytes();
        let seed = 0x12345678;
        let m = 128u32;
        let n = 128u32;
        let k = 128u32;
        let rank = 16;

        let (a_bytes, bt_bytes) = derive_matrix_bytes(seed, m, n, k);

        let tile_m = 16u32;
        let tile_n = 16u32;
        let tile_row = 1u32;
        let tile_col = 2u32;

        let rows_pattern: Vec<u32> = (0..tile_m).collect();
        let cols_pattern: Vec<u32> = (0..tile_n).collect();

        let proof = build_merkle_proof(
            &a_bytes,
            &bt_bytes,
            m as usize,
            n as usize,
            k as usize,
            tile_row,
            tile_col,
            tile_m,
            tile_n,
            &rows_pattern,
            &cols_pattern,
            &job_key,
            rank as usize,
        )
        .unwrap();

        assert!(proof.a.proof.verify(job_key), "A Merkle proof should verify");
        assert!(!proof.a.row_indices.is_empty(), "A should have row indices");
        assert!(!proof.a.proof.leaf_data.is_empty(), "A should have leaf data");

        // Verify B^T Merkle proof
        assert!(proof.bt.proof.verify(job_key), "B^T Merkle proof should verify");
        assert!(!proof.bt.row_indices.is_empty(), "B^T should have row indices");
        assert!(!proof.bt.proof.leaf_data.is_empty(), "B^T should have leaf data");

        // Check row indices are in range
        for &r in &proof.a.row_indices {
            assert!(r < m as usize, "A row {} out of range", r);
        }
        for &r in &proof.bt.row_indices {
            assert!(r < n as usize, "B^T row {} out of range", r);
        }
    }

    /// Verify that different seeds produce different matrix data.
    #[test]
    fn test_different_seed_different_matrices() {

        let m = 16u32;
        let n = 16u32;
        let k = 16u32;

        let (a1, bt1) = derive_matrix_bytes(0xAAAAAAAA, m, n, k);
        let (a2, bt2) = derive_matrix_bytes(0xBBBBBBBB, m, n, k);

        assert_ne!(a1, a2, "Different seeds should produce different A matrices");
        assert_ne!(bt1, bt2, "Different seeds should produce different B^T matrices");
    }
}

fn build_matrix_proof(
    flat_data: &[u8],
    job_key: &[u8; 32],
    row_indices: &[usize],
    num_rows: usize,
    num_cols: usize,
) -> MatrixMerkleProof {
    let padded = pearl_blake3::pad_to_chunk_boundary(flat_data);
    let tree = pearl_blake3::MerkleTree::new(&padded, *job_key);
    let leaf_indices =
        pearl_blake3::MerkleTree::compute_leaf_indices_from_rows(row_indices, (num_rows, num_cols));
    let proof = tree.get_multileaf_proof(&leaf_indices);

    MatrixMerkleProof {
        proof,
        row_indices: row_indices.to_vec(),
    }
}
