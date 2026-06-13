use plonky2_maybe_rayon::*;

use crate::v1::api::proof_utils::CompiledPublicParams;
use pearl_blake3::{BLAKE3_DIGEST_SIZE, blake3_digest};

#[derive(Debug, Clone)]
pub struct MMSlice {
    pub a: Vec<Vec<i8>>, // a
    pub b: Vec<Vec<i8>>, // b^t
}

const NOISE_RANGE: usize = 128;
const IDXS_PER_COL: usize = 2;
const UNIFORM_NOISE_RANGE: usize = NOISE_RANGE / IDXS_PER_COL;
const ZERO_POINT_TRANSLATION: i8 = (UNIFORM_NOISE_RANGE / 2) as i8;
const RANGE_MASK: u8 = (UNIFORM_NOISE_RANGE - 1) as u8;

const fn padded_seed_label(label: [u8; 8]) -> [u8; 32] {
    let mut result = [0u8; 32];
    let mut i = 0;
    while i < label.len() {
        result[i] = label[i];
        i += 1;
    }
    result
}

const SEED_LABEL_A: [u8; 32] = padded_seed_label(*b"A_tensor");
const SEED_LABEL_B: [u8; 32] = padded_seed_label(*b"B_tensor");

/// A*v with each row of A having exactly two non-zero entries:
/// +1 at A[i][0] and -1 at A[i][1].
/// Result[i] = v[A[i][0]] - v[A[i][1]]
fn matvec_sparse_perm(perm: &[[u32; 2]], vec: &[i8]) -> Vec<i8> {
    perm.iter()
        .map(|&[first_idx, second_idx]| {
            let pos_val = vec[first_idx as usize] as i32;
            let neg_val = vec[second_idx as usize] as i32;
            (pos_val - neg_val) as i8
        })
        .collect()
}

// Helper function to generate random hash for noise generation
pub fn get_random_hash(index: usize, seed: &[u8; 32], key: &[u8; 32], prepend_index: usize) -> [u8; 32] {
    let mut message = vec![0u8; 32 + 32]; // 8 i32 prepend slots + 32-byte seed

    // Write the prepend_index value at the correct position (as i32 little-endian)
    let prepend_value = (1 + index) as i32;
    message[prepend_index * 4..(prepend_index * 4 + 4)].copy_from_slice(&prepend_value.to_le_bytes());

    // Copy seed
    message[32..64].copy_from_slice(seed);

    blake3_digest(&message, Some(*key))
}

// Generate uniform random matrix (A_L or B_R_transposed)
// Only guaranteed to be correct on row_indices
pub fn generate_uniform_random_matrix(seed: &[u8; 32], key: &[u8; 32], row_indices: &[usize], num_cols: usize) -> Vec<Vec<i8>> {
    row_indices
        .iter()
        .map(|&row_idx| {
            let start_idx = row_idx * num_cols;
            (start_idx / BLAKE3_DIGEST_SIZE..(start_idx + num_cols).div_ceil(BLAKE3_DIGEST_SIZE))
                .flat_map(|block| {
                    get_random_hash(block, seed, key, 0)
                        .into_iter()
                        .enumerate()
                        .filter_map(move |(k, byte)| {
                            let idx = block * BLAKE3_DIGEST_SIZE + k;
                            (idx >= start_idx && idx < start_idx + num_cols)
                                .then(|| (byte & RANGE_MASK) as i8 - ZERO_POINT_TRANSLATION)
                        })
                })
                .collect()
        })
        .collect()
}

// Compute high 32 bits of 64-bit product of two unsigned 32-bit integers
pub fn mul_hi_u32(a: u32, b: u32) -> u32 {
    ((a as u64 * b as u64) >> 32) as u32
}

/// Generate sparse permutation matrix (A_R_transposed or B_L).
/// Returns k pairs of indices, where each pair [first_idx, second_idx] represents
/// a row with +1 at first_idx and -1 at second_idx.
pub fn generate_permutation_matrix(seed: &[u8; 32], key: &[u8; 32], k: usize, noise_rank: usize) -> Vec<[u32; 2]> {
    const BYTES_PER_LINE: usize = 4;
    const LINES_PER_HASH: usize = BLAKE3_DIGEST_SIZE / BYTES_PER_LINE;

    let rank_mask = (noise_rank - 1) as u32;
    let mut res = vec![[0u32; 2]; k];

    // Each hash covers one LINES_PER_HASH-sized chunk; par_chunks_mut truncates the last chunk naturally.
    res.par_chunks_mut(LINES_PER_HASH).enumerate().for_each(|(i, chunk)| {
        let random_hash = get_random_hash(i, seed, key, 1);
        for (j, slot) in chunk.iter_mut().enumerate() {
            let random_uint32 = u32::from_le_bytes([
                random_hash[j * 4],
                random_hash[j * 4 + 1],
                random_hash[j * 4 + 2],
                random_hash[j * 4 + 3],
            ]);

            let first_idx = random_uint32 & rank_mask;
            let second_idx = first_idx ^ (1 + mul_hi_u32((noise_rank - 1) as u32, random_uint32));

            *slot = [first_idx, second_idx];
        }
    });

    res
}

pub fn compute_noise_for_indices(
    k: usize,
    noise_rank: usize,
    commitment_hash: ([u8; 32], [u8; 32]),
    a_rows_indices: &[usize],
    b_cols_indices: &[usize],
) -> MMSlice {
    // Validate noise_rank constraints
    debug_assert!(
        noise_rank > 0 && (noise_rank & (noise_rank - 1)) == 0,
        "noise_rank must be a power of two"
    );
    debug_assert_eq!(
        noise_rank % BLAKE3_DIGEST_SIZE,
        0,
        "noise_rank must be divisible by BLAKE3_DIGEST_SIZE"
    );

    let (b_noise_seed, a_noise_seed) = commitment_hash;

    let seed_a = SEED_LABEL_A;
    let seed_b = SEED_LABEL_B;

    // Generate the 4 matrices
    let e_al = generate_uniform_random_matrix(&seed_a, &a_noise_seed, a_rows_indices, noise_rank);
    let e_ar_transposed = generate_permutation_matrix(&seed_a, &a_noise_seed, k, noise_rank);
    let e_bl = generate_permutation_matrix(&seed_b, &b_noise_seed, k, noise_rank);
    let e_br_transposed = generate_uniform_random_matrix(&seed_b, &b_noise_seed, b_cols_indices, noise_rank);

    // Compute only the needed rows of NOISE_A = E_AL * E_AR
    // e_ar_transposed is k × r, row is r × 1
    let noise_a: Vec<Vec<i8>> = e_al.par_iter().map(|row| matvec_sparse_perm(&e_ar_transposed, row)).collect();

    // Compute only the needed columns of NOISE_B = E_BL * E_BR
    // e_bl is k × r, col is r × 1
    let noise_b_t: Vec<Vec<i8>> = e_br_transposed.par_iter().map(|col| matvec_sparse_perm(&e_bl, col)).collect();

    MMSlice {
        a: noise_a,
        b: noise_b_t,
    }
}

// Compute noise rows for A and noise columns for B
// noise_A -- array of TILE_H noise rows (selected by a_rows_indices).
// noise_B -- array of TILE_W noise columns (selected by b_cols_indices).
pub fn compute_noise(params: &CompiledPublicParams) -> MMSlice {
    compute_noise_for_indices(
        params.k,
        params.r,
        params.commitment_hash,
        &params.a_rows_indices,
        &params.b_cols_indices,
    )
}

#[cfg(test)]
mod tests {
    use crate::v1::api::proof::PublicProofParams;
    #[cfg(debug_assertions)]
    use crate::v1::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern};

    use super::*;

    #[test]
    fn test_compute_noise_output_shape() {
        let mut params = PublicProofParams::new_for_tests(1024, 768, 512);

        // Test with r = 128
        let result = compute_noise(&CompiledPublicParams::from(&params));

        // Verify shape of noise_a: should have h rows, each of length k
        assert_eq!(result.a.len(), params.h(), "noise_a should have h rows");
        for row in &result.a {
            assert_eq!(row.len(), params.common_dim(), "Each noise_a row should have length k");
        }

        // Verify shape of noise_b: should have w columns, each of length k
        assert_eq!(result.b.len(), params.w(), "noise_b should have w columns");
        for col in &result.b {
            assert_eq!(col.len(), params.common_dim(), "Each noise_b column should have length k");
        }

        println!(
            "Test with r=128 passed: noise_a shape = {}x{}, noise_b shape = {}x{}",
            result.a.len(),
            result.a[0].len(),
            result.b.len(),
            result.b[0].len()
        );

        // Test with different noise_rank (r = 64)
        params.mining_config.rank = 64; // Must be power of 2 and divisible by 32
        let result2 = compute_noise(&CompiledPublicParams::from(&params));

        // Verify shapes remain the same regardless of r
        assert_eq!(result2.a.len(), params.h(), "noise_a should have h rows with r=64");
        for row in &result2.a {
            assert_eq!(
                row.len(),
                params.common_dim(),
                "Each noise_a row should have length k with r=64"
            );
        }

        assert_eq!(result2.b.len(), params.w(), "noise_b should have w columns with r=64");
        for col in &result2.b {
            assert_eq!(
                col.len(),
                params.common_dim(),
                "Each noise_b column should have length k with r=64"
            );
        }

        println!(
            "Test with r=64 passed: noise_a shape = {}x{}, noise_b shape = {}x{}",
            result2.a.len(),
            result2.a[0].len(),
            result2.b.len(),
            result2.b[0].len()
        );

        // Test with r = 32 (minimum valid value since must be divisible by BLAKE3_DIGEST_SIZE=32)
        params.mining_config.rank = 32;
        let result3 = compute_noise(&CompiledPublicParams::from(&params));

        assert_eq!(result3.a.len(), params.h(), "noise_a should have h rows with r=32");
        assert_eq!(result3.b.len(), params.w(), "noise_b should have w columns with r=32");

        println!(
            "Test with r=32 passed: noise_a shape = {}x{}, noise_b shape = {}x{}",
            result3.a.len(),
            result3.a[0].len(),
            result3.b.len(),
            result3.b[0].len()
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "noise_rank must be a power of two")]
    fn test_invalid_noise_rank_not_power_of_two() {
        let block_header = IncompleteBlockHeader::new_for_test(0x207FFFFF);
        let mining_configuration = MiningConfiguration {
            common_dim: 512,
            rank: 100, // Not a power of 2
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41, 64, 65, 72, 73, 96, 97, 104, 105]).unwrap(),
            reserved: MiningConfiguration::RESERVED_VALUE,
        };

        let params = PublicProofParams::new_dummy(block_header, mining_configuration, 1024, 768, 0, 0);

        compute_noise(&CompiledPublicParams::from(&params)); // Should panic in debug mode
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "noise_rank must be divisible by BLAKE3_DIGEST_SIZE")]
    fn test_invalid_noise_rank_not_divisible_by_32() {
        let block_header = IncompleteBlockHeader::new_for_test(0x207FFFFF);

        let mining_configuration = MiningConfiguration {
            common_dim: 512,
            rank: 16, // Power of 2 but not divisible by 32
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41, 64, 65, 72, 73, 96, 97, 104, 105]).unwrap(),
            reserved: MiningConfiguration::RESERVED_VALUE,
        };

        let params = PublicProofParams::new_dummy(block_header, mining_configuration, 1024, 768, 0, 0);

        compute_noise(&CompiledPublicParams::from(&params)); // Should panic in debug mode
    }
}
