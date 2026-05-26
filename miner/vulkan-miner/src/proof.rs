use anyhow::Result;
use zk_pow::ffi::plain_proof::{MatrixMerkleProof, PlainProof};

/// Build a Merkle proof for a winning tile after a block is found.
///
/// The A and B matrices are recomputed on CPU from the same seed to
/// avoid reading back from GPU (saving PCIe bandwidth).
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
