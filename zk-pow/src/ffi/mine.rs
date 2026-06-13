//! Shared mining implementation for both Go FFI and Python bindings.

use anyhow::Result;
use primitive_types::U256;
use rand::Rng;

use crate::api::proof::{IncompleteBlockHeader, MiningConfiguration, PeriodicPattern};
use crate::api::proof_utils::{compute_hash_activations, compute_jackpot_hash};
use crate::api::sanity_checks::extract_difficulty_bound;
use crate::circuit::pearl_noise::compute_noise_for_indices;
use crate::circuit::pearl_program::{JACKPOT_SIZE, LROT_PER_TILE};
use crate::ffi::plain_proof::{MatrixMerkleProof, MoEProofParams, PlainProof};
use pearl_blake3::blake3_digest;

const SIGNAL_MIN: i8 = -64;
const SIGNAL_MAX: i8 = 64;

#[allow(clippy::too_many_arguments)]
pub fn try_mine_one<R: Rng>(
    rng: &mut R,
    m: usize,
    n: usize,
    k: usize,
    header: IncompleteBlockHeader,
    config: MiningConfiguration,
    signal_range: Option<(i8, i8)>,
    wrong_jackpot_hash: bool,
) -> Result<Option<PlainProof>> {
    let rank = config.rank as usize;
    let (signal_min, signal_max) = signal_range.unwrap_or((SIGNAL_MIN, SIGNAL_MAX));

    // Generate random matrices A (m×k) and B (k×n)
    let a_matrix: Vec<Vec<i8>> = (0..m)
        .map(|_| (0..k).map(|_| rng.random_range(signal_min..=signal_max)).collect())
        .collect();

    let b_matrix: Vec<Vec<i8>> = (0..k)
        .map(|_| (0..n).map(|_| rng.random_range(signal_min..=signal_max)).collect())
        .collect();

    // Transpose B for column-major format
    let b_transposed: Vec<Vec<i8>> = (0..n).map(|i| (0..k).map(|j| b_matrix[j][i]).collect()).collect();

    let job_key = compute_job_key(&header, &config);

    let a_row_major = pearl_blake3::pad_to_chunk_boundary(&flatten_matrix(&a_matrix));
    let b_col_major = pearl_blake3::pad_to_chunk_boundary(&flatten_matrix(&b_transposed));
    let (b_noise_seed, a_noise_seed) = compute_commitment_hash(&job_key, &a_row_major, &b_col_major);

    // Compute noise using shared implementation from pearl_noise
    let a_all_rows: Vec<usize> = (0..m).collect();
    let b_all_cols: Vec<usize> = (0..n).collect();
    let noise = compute_noise_for_indices(k, rank, (b_noise_seed, a_noise_seed), &a_all_rows, &b_all_cols);

    // Add noise to matrices (noise.a is m×k, noise.b is n×k as transposed columns)
    let a_noised: Vec<Vec<i32>> = a_matrix
        .iter()
        .zip(&noise.a)
        .map(|(a_row, n_row)| a_row.iter().zip(n_row).map(|(&a, &n)| a as i32 + n as i32).collect())
        .collect();

    // noise.b contains columns of B's noise as rows, need to transpose for b_matrix (k×n)
    let b_noised: Vec<Vec<i32>> = b_matrix
        .iter()
        .enumerate()
        .map(|(row_idx, b_row)| {
            b_row
                .iter()
                .enumerate()
                .map(|(col_idx, &b)| b as i32 + noise.b[col_idx][row_idx] as i32)
                .collect()
        })
        .collect();

    let b_noised_t: Vec<Vec<i32>> = (0..n).map(|i| (0..k).map(|j| b_noised[j][i]).collect()).collect();

    // Mine using pattern partitions
    for a_rows in threads_partition(&config.rows_pattern, m) {
        for b_cols in threads_partition(&config.cols_pattern, n) {
            // same as compute_jackpot but with a and b matrices pre-noised
            let tile_h = a_rows.len();
            let tile_w = b_cols.len();
            let mut jackpot_tile: Vec<Vec<i32>> = vec![vec![0; tile_w]; tile_h];
            let mut jackpot: [u32; 16] = [0; 16];

            for ll in (rank..=k).step_by(rank) {
                for (u, &a_idx) in a_rows.iter().enumerate() {
                    for (v, &b_idx) in b_cols.iter().enumerate() {
                        for l in ll - rank..ll {
                            jackpot_tile[u][v] += a_noised[a_idx][l] * b_noised_t[b_idx][l];
                        }
                    }
                }

                let xored_tile = jackpot_tile.iter().flatten().fold(0u32, |a, &x| a ^ x as u32);
                let tid = (ll / rank - 1) % JACKPOT_SIZE;
                jackpot[tid] = jackpot[tid].rotate_left(LROT_PER_TILE) ^ xored_tile;
            }
            let jackpot_hash = compute_jackpot_hash(&jackpot, a_noise_seed);
            let jackpot_bound = extract_difficulty_bound(header.nbits, &config);
            if (U256::from_little_endian(&jackpot_hash) <= jackpot_bound) != wrong_jackpot_hash {
                let a_proof = build_matrix_proof(&a_matrix, &job_key, &a_rows, k);
                let b_proof = build_matrix_proof(&b_transposed, &job_key, &b_cols, k);

                return Ok(Some(PlainProof {
                    m,
                    n,
                    k,
                    noise_rank: rank,
                    a: a_proof,
                    bt: b_proof,
                    moe: None,
                }));
            }
        }
    }

    Ok(None)
}

#[allow(clippy::too_many_arguments)]
pub fn try_mine_one_moe<R: Rng>(
    rng: &mut R,
    m: usize,
    n: usize,
    k: usize,
    header: IncompleteBlockHeader,
    config: MiningConfiguration,
    signal_range: Option<(i8, i8)>,
    wrong_jackpot_hash: bool,
) -> Result<Option<PlainProof>> {
    // If we are in the non-moe case, mine via the original function.
    if config.moe.is_none() {
        return try_mine_one(rng, m, n, k, header, config, signal_range, wrong_jackpot_hash);
    }

    let moe_cfg = config.moe.unwrap();
    let e = moe_cfg.e as usize;
    let top_k = moe_cfg.top_k as usize;
    assert!(top_k > 0 && top_k < e, "top_k must be in (0, e)");
    assert!(e > 0, "e must be greater than 0 in the MoE setting.");

    let rank = config.rank as usize;
    let (signal_min, signal_max) = signal_range.unwrap_or((SIGNAL_MIN, SIGNAL_MAX));

    // Generate random matrices A (m × k) and B (k x (n_e x n))
    let a_matrix: Vec<Vec<i8>> = (0..m)
        .map(|_| (0..k).map(|_| rng.random_range(signal_min..=signal_max)).collect())
        .collect();

    let b_matrix: Vec<Vec<i8>> = (0..k)
        .map(|_| (0..n * e).map(|_| rng.random_range(signal_min..=signal_max)).collect())
        .collect();

    // Transpose B for column-major format
    let b_transposed: Vec<Vec<i8>> = (0..n * e).map(|i| (0..k).map(|j| b_matrix[j][i]).collect()).collect();

    let job_key = compute_job_key(&header, &config);

    let a_row_major = pearl_blake3::pad_to_chunk_boundary(&flatten_matrix(&a_matrix));
    let b_col_major = pearl_blake3::pad_to_chunk_boundary(&flatten_matrix(&b_transposed));

    // Build a random routing where each token is assigned to `top_k` distinct experts.
    // Expert 1 is the "hot" expert: each token includes it with probability HOT_BIAS,
    // and the remaining slots are filled uniformly at random from the other experts.
    const HOT_EXPERT: usize = 1;
    const HOT_BIAS: f64 = 0.7;
    let mut routing: Vec<Vec<u32>> = vec![Vec::new(); e];
    for token_idx in 0..m {
        let mut selected: Vec<usize> = Vec::with_capacity(top_k);
        if rng.random::<f64>() < HOT_BIAS {
            selected.push(HOT_EXPERT);
        }
        while selected.len() < top_k {
            let cand = rng.random_range(0..e);
            if !selected.contains(&cand) {
                selected.push(cand);
            }
        }
        for &expert_idx in &selected {
            routing[expert_idx].push(token_idx as u32);
        }
    }

    let (b_noise_seed, a_noise_seed, routing_end_offsets) =
        compute_commitment_hash_with_offsets(&job_key, &a_row_major, &b_col_major, Some(&routing[..]));

    // Compute noise using shared implementation from pearl_noise
    let a_all_rows: Vec<usize> = (0..m).collect();
    let b_all_cols: Vec<usize> = (0..n * e).collect();
    let noise = compute_noise_for_indices(k, rank, (b_noise_seed, a_noise_seed), &a_all_rows, &b_all_cols);

    // Add noise to matrices (noise.a is m×k, noise.b is (n_e×n)×k as transposed columns)
    let a_noised: Vec<Vec<i32>> = a_matrix
        .iter()
        .zip(&noise.a)
        .map(|(a_row, n_row)| a_row.iter().zip(n_row).map(|(&a, &n)| a as i32 + n as i32).collect())
        .collect();

    // noise.b contains columns of B's noise as rows, need to transpose for b_matrix (k×(n_e×n))
    let b_noised: Vec<Vec<i32>> = b_matrix
        .iter()
        .enumerate()
        .map(|(row_idx, b_row)| {
            b_row
                .iter()
                .enumerate()
                .map(|(col_idx, &b)| b as i32 + noise.b[col_idx][row_idx] as i32)
                .collect()
        })
        .collect();

    let b_noised_t: Vec<Vec<i32>> = (0..n * e).map(|i| (0..k).map(|j| b_noised[j][i]).collect()).collect();

    // Mine using pattern partitions
    // Here, first, define the new a/b matrices as the subtoken matrices
    for expert_idx in 0..e {
        // Submatrices here:
        // - A_sub: rows corresponding to this expert (routing[expert_idx].len() rows)
        // - B_sub: shape = k x (n_e x n)
        // Floor the per-expert token count to a multiple of the row-pattern period
        // so `threads_partition` can evenly partition the rows. Tokens in the tail
        // are ignored for this expert's mining attempt but still live in the
        // committed routing (and contribute to the `m * top_k` routing entries).
        let row_period = config.rows_pattern.period() as usize;
        let expert_tokens = (routing[expert_idx].len() / row_period) * row_period;
        if expert_tokens == 0 {
            continue;
        }
        for a_rows in threads_partition(&config.rows_pattern, expert_tokens) {
            for b_cols in threads_partition(&config.cols_pattern, n) {
                // same as compute_jackpot but with a and b matrices pre-noised
                let tile_h = a_rows.len();
                let tile_w = b_cols.len();
                let mut jackpot_tile: Vec<Vec<i32>> = vec![vec![0; tile_w]; tile_h];
                let mut jackpot: [u32; 16] = [0; 16];

                for ll in (rank..=k).step_by(rank) {
                    for (u, &a_idx) in a_rows.iter().enumerate() {
                        for (v, &b_idx) in b_cols.iter().enumerate() {
                            for l in ll - rank..ll {
                                let global_a_idx = routing[expert_idx][a_idx] as usize;
                                let global_b_idx = expert_idx * n + b_idx;
                                jackpot_tile[u][v] += a_noised[global_a_idx][l] * b_noised_t[global_b_idx][l];
                            }
                        }
                    }

                    let xored_tile = jackpot_tile.iter().flatten().fold(0u32, |a, &x| a ^ x as u32);
                    let tid = (ll / rank - 1) % JACKPOT_SIZE;
                    jackpot[tid] = jackpot[tid].rotate_left(LROT_PER_TILE) ^ xored_tile;
                }
                let jackpot_hash = compute_jackpot_hash(&jackpot, a_noise_seed);
                let jackpot_bound = extract_difficulty_bound(header.nbits, &config);
                if (U256::from_little_endian(&jackpot_hash) <= jackpot_bound) != wrong_jackpot_hash {
                    let global_a_rows: Vec<usize> = a_rows.iter().map(|&idx| routing[expert_idx][idx] as usize).collect();
                    let global_b_cols: Vec<usize> = b_cols.iter().map(|&idx| expert_idx * n + idx).collect();
                    let a_proof = build_matrix_proof(&a_matrix, &job_key, &global_a_rows, k);
                    let b_proof = build_matrix_proof(&b_transposed, &job_key, &global_b_cols, k);

                    let routing_proof = build_routing_proof(&routing, &job_key, expert_idx, &a_rows);

                    let moe_params = MoEProofParams {
                        e,
                        expert_idx: expert_idx as u16,
                        inner_a_rows: a_rows.clone(),
                        routing_proof: routing_proof.proof,
                        routing_end_offsets,
                        top_k,
                    };
                    return Ok(Some(PlainProof {
                        m,
                        k,
                        n,
                        noise_rank: rank,
                        a: a_proof,
                        bt: b_proof,
                        moe: Some(moe_params),
                    }));
                }
            }
        }
    }

    Ok(None)
}

/// Mines a proof for the given block header and configuration.
///
/// * `signal_range` - Optional custom signal range [min, max] for testing. Default: (-64, 64)
/// * `wrong_jackpot_hash` - Accept wrong jackpot hash (for testing only)
#[allow(clippy::too_many_arguments)]
pub fn mine(
    m: usize,
    n: usize,
    k: usize,
    header: IncompleteBlockHeader,
    config: MiningConfiguration,
    signal_range: Option<(i8, i8)>,
    wrong_jackpot_hash: bool,
) -> Result<PlainProof> {
    let mut rng = rand::rng();

    loop {
        let proof = try_mine_one(&mut rng, m, n, k, header, config, signal_range, wrong_jackpot_hash)?;
        if let Some(proof) = proof {
            return Ok(proof);
        }
    }
}

pub fn mine_moe(
    m: usize,
    n: usize,
    k: usize,
    header: IncompleteBlockHeader,
    config: MiningConfiguration,
    signal_range: Option<(i8, i8)>,
    wrong_jackpot_hash: bool,
) -> Result<PlainProof> {
    let mut rng = rand::rng();

    loop {
        let proof = try_mine_one_moe(&mut rng, m, n, k, header, config, signal_range, wrong_jackpot_hash)?;
        if let Some(proof) = proof {
            return Ok(proof);
        }
    }
}

/// Build a MatrixMerkleProof for the given matrix and row indices using pearl_blake3.
fn build_matrix_proof(matrix: &[Vec<i8>], job_key: &[u8; 32], row_indices: &[usize], num_cols: usize) -> MatrixMerkleProof {
    let padded = pearl_blake3::pad_to_chunk_boundary(&flatten_matrix(matrix));
    let tree = pearl_blake3::MerkleTree::new(&padded, *job_key);
    let leaf_indices = pearl_blake3::MerkleTree::compute_leaf_indices_from_rows(row_indices, (matrix.len(), num_cols));
    let proof = tree.get_multileaf_proof(&leaf_indices);
    MatrixMerkleProof {
        proof,
        row_indices: row_indices.to_vec(),
    }
}

/// Flattens the jagged per-expert routing into the byte layout committed in the
/// routing Merkle tree: every token index as a little-endian u32, experts concatenated.
fn flatten_routing(routing: &[Vec<u32>]) -> Vec<u8> {
    routing
        .iter()
        .flat_map(|tokens| tokens.iter().flat_map(|&idx| idx.to_le_bytes()))
        .collect()
}

/// Root of the routing commitment, computed the same way as the routing proof's root:
/// `blake3(pad_to_chunk_boundary(flatten_routing), key=job_key)`, which equals
/// `MerkleTree::new(padded, job_key).root()` (see [`build_routing_proof`]). Used in the
/// commitment hash so the miner and verifier agree on `hash_routing` without needing a
/// specific expert's membership proof.
fn routing_hash(routing: &[Vec<u32>], job_key: &[u8; 32]) -> [u8; 32] {
    let padded = pearl_blake3::pad_to_chunk_boundary(&flatten_routing(routing));
    blake3_digest(&padded, Some(*job_key))
}

fn build_routing_proof(
    routing: &[Vec<u32>],
    job_key: &[u8; 32],
    expert_idx: usize,
    inner_indices: &[usize],
) -> MatrixMerkleProof {
    let padded = pearl_blake3::pad_to_chunk_boundary(&flatten_routing(routing));
    let tree = pearl_blake3::MerkleTree::new(&padded, *job_key);
    // Routing is a jagged 2D array; treat the flattened form as a 1D array of u32
    // entries and address inner indices by prefix-summing expert lengths.
    let expert_offset: usize = routing[..expert_idx].iter().map(|r| r.len()).sum();
    let total_entries: usize = routing.iter().map(|r| r.len()).sum();
    let row_indices: Vec<usize> = inner_indices.iter().map(|&i| expert_offset + i).collect();
    let leaf_indices =
        pearl_blake3::MerkleTree::compute_leaf_indices_from_rows(&row_indices, (total_entries, std::mem::size_of::<u32>()));
    let proof = tree.get_multileaf_proof(&leaf_indices);
    MatrixMerkleProof { proof, row_indices }
}

fn compute_job_key(header: &IncompleteBlockHeader, config: &MiningConfiguration) -> [u8; 32] {
    let mut data = Vec::with_capacity(128);
    data.extend_from_slice(&header.to_bytes());
    data.extend_from_slice(&config.to_bytes());
    blake3_digest(&data, None)
}

fn compute_commitment_hash(job_key: &[u8; 32], a_row_major: &[u8], b_col_major: &[u8]) -> ([u8; 32], [u8; 32]) {
    let (b_seed, a_seed, _) = compute_commitment_hash_with_offsets(job_key, a_row_major, b_col_major, None);

    (b_seed, a_seed)
}

type MoECaseOutputs = ([u8; 32], [u8; 32], Vec<u32>);
fn compute_commitment_hash_with_offsets(
    job_key: &[u8; 32],
    a_row_major: &[u8],
    b_col_major: &[u8],
    opt_routing: Option<&[Vec<u32>]>,
) -> MoECaseOutputs {
    let hash_a = blake3_digest(a_row_major, Some(*job_key));
    let hash_b = blake3_digest(b_col_major, Some(*job_key));

    let (hash_activations, routing_offsets) = match opt_routing {
        Some(r) => {
            let mut acc = 0u32;
            let routing_offsets: Vec<u32> = r
                .iter()
                .map(|expert_routing| {
                    acc = acc
                        .checked_add(u32::try_from(expert_routing.len()).expect("expert token count exceeds u32"))
                        .expect("cumulative routing offset exceeds u32; m * top_k too large");
                    acc
                })
                .collect();

            let hash_routing_data = routing_hash(r, job_key);
            let ha = compute_hash_activations(&hash_a, &hash_routing_data, &routing_offsets, job_key);
            (ha, routing_offsets)
        }
        None => (hash_a, vec![]),
    };

    let b_noise_seed = blake3_digest(&[&job_key[..], &hash_b[..]].concat(), None);
    let a_noise_seed = blake3_digest(&[&b_noise_seed[..], &hash_activations[..]].concat(), None);

    (b_noise_seed, a_noise_seed, routing_offsets)
}

fn flatten_matrix(matrix: &[Vec<i8>]) -> Vec<u8> {
    matrix.iter().flatten().map(|&x| x as u8).collect()
}

fn threads_partition(pattern: &PeriodicPattern, total_dimension: usize) -> Vec<Vec<usize>> {
    let period = pattern.period() as usize;
    if !total_dimension.is_multiple_of(period) {
        panic!("total_dimension must be divisible by pattern period");
    }

    let base_indices: Vec<usize> = pattern.to_list().iter().map(|&i| i as usize).collect();

    (0..total_dimension)
        .filter(|&i| pattern.offset_is_valid(i as u32))
        .map(|offset| base_indices.iter().map(|&d| offset + d).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::api::proof::MoEConfig;
    use crate::api::{proof::MMAType, verify::verify_plain_proof};

    use super::*;
    use blake3::CHUNK_LEN;

    const TEST_MATRIX_MOD: usize = 251;

    fn test_matrix(num_rows: usize, num_cols: usize) -> Vec<Vec<i8>> {
        (0..num_rows)
            .map(|r| (0..num_cols).map(|c| ((r * num_cols + c) % TEST_MATRIX_MOD) as i8).collect())
            .collect()
    }

    #[test]
    fn test_build_matrix_proof_pads_to_chunk_boundary() {
        // 3 rows x 500 cols = 1500 bytes, not a multiple of CHUNK_LEN (1024)
        let num_rows = 3;
        let num_cols = 500;
        assert_ne!((num_rows * num_cols) % CHUNK_LEN, 0);

        let matrix = test_matrix(num_rows, num_cols);
        let key = [42u8; 32];

        let proof = build_matrix_proof(&matrix, &key, &[0, 2], num_cols);

        let padded = pearl_blake3::pad_to_chunk_boundary(&flatten_matrix(&matrix));
        let expected_root = pearl_blake3::Blake3Hasher::with_key(key).hash(&padded);
        assert_eq!(
            proof.proof.root, expected_root,
            "Merkle root must equal blake3 of chunk-padded data"
        );
    }

    #[test]
    fn test_build_matrix_proof_aligned_unchanged() {
        // 4 rows x 256 cols = 1024 bytes, exactly one chunk
        let num_rows = 4;
        let num_cols = 256;
        assert_eq!((num_rows * num_cols) % CHUNK_LEN, 0);

        let matrix = test_matrix(num_rows, num_cols);
        let key = [7u8; 32];

        let proof = build_matrix_proof(&matrix, &key, &[1, 3], num_cols);

        let flat = flatten_matrix(&matrix);
        let expected_root = pearl_blake3::Blake3Hasher::with_key(key).hash(&flat);
        assert_eq!(
            proof.proof.root, expected_root,
            "Aligned data should produce identical root with or without padding"
        );
    }

    #[test]
    fn test_padded_blake3_hash_equals_merkle_root() {
        // The commitment hash derives noise seeds from blake3(padded_data, key).
        // This must equal MerkleTree::new(padded_data, key).root() so that
        // the verifier's Merkle proof check is consistent with the miner's
        // commitment.
        let num_rows = 3;
        let num_cols = 500;
        let matrix = test_matrix(num_rows, num_cols);
        let key = [99u8; 32];

        let padded = pearl_blake3::pad_to_chunk_boundary(&flatten_matrix(&matrix));
        let hash_via_digest = blake3_digest(&padded, Some(key));
        let hash_via_tree = pearl_blake3::MerkleTree::new(&padded, key).root();
        assert_eq!(hash_via_digest, hash_via_tree);
    }

    #[test]
    fn test_moe_commitment_matches_verifier_non_aligned_routing() {
        use crate::api::proof::{MoEParams, PublicProofParams};

        // The miner (`compute_commitment_hash_with_offsets`) and the verifier
        // (`PublicProofParams::commitment_hash`) must derive identical noise seeds, even
        // when the flattened routing length is NOT a multiple of the BLAKE3 chunk size.
        // This only holds because `routing_hash` pads exactly like the routing Merkle
        // tree — an unpadded routing hash matches for chunk-aligned dims but fails here.
        let job_key = [0x5au8; 32];
        let top_k = 2u16;

        // 5 experts, 50 routing entries total -> 200 bytes, not a multiple of CHUNK_LEN (1024).
        let routing: Vec<Vec<u32>> = vec![
            (0..10).collect(),
            (10..22).collect(),
            (22..30).collect(),
            (30..45).collect(),
            (45..50).collect(),
        ];
        let total_bytes = routing.iter().map(|r| r.len()).sum::<usize>() * std::mem::size_of::<u32>();
        assert_ne!(total_bytes % CHUNK_LEN, 0, "test must use non-chunk-aligned routing");

        // Arbitrary padded A/B blobs; only their keyed hashes feed the commitment.
        let a_row_major = pearl_blake3::pad_to_chunk_boundary(&[1u8; 100]);
        let b_col_major = pearl_blake3::pad_to_chunk_boundary(&[2u8; 100]);

        // Miner side.
        let (b_mine, a_mine, offsets) =
            compute_commitment_hash_with_offsets(&job_key, &a_row_major, &b_col_major, Some(&routing));

        // Verifier side: take `hash_routing` from the actual routing Merkle proof root
        // (what the wire carries), then recompute the seeds via the canonical path.
        let hash_routing = build_routing_proof(&routing, &job_key, 1, &[0]).proof.root;
        let params = PublicProofParams {
            block_header: IncompleteBlockHeader::new_for_test(0x207FFFFF),
            mining_config: MiningConfiguration {
                common_dim: 1024,
                rank: 32,
                mma_type: MMAType::Int7xInt7ToInt32,
                rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
                cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9]).unwrap(),
                moe: Some(MoEConfig {
                    e: routing.len() as u16,
                    top_k,
                }),
            },
            hash_a: blake3_digest(&a_row_major, Some(job_key)),
            hash_b: blake3_digest(&b_col_major, Some(job_key)),
            hash_jackpot: [0u8; 32],
            m: 0,
            n: 0,
            t_rows: 0,
            t_cols: 0,
            moe: Some(MoEParams {
                expert_idx: 0,
                routing_offsets: offsets.clone(),
                hash_routing,
                outer_indices: vec![],
            }),
        };
        let (b_verify, a_verify) = params.commitment_hash(job_key);

        assert_eq!(b_mine, b_verify, "b_noise_seed must match between miner and verifier");
        assert_eq!(a_mine, a_verify, "a_noise_seed must match between miner and verifier");
    }

    #[test]
    fn test_verify_moe() {
        let e = 4;
        let m = 1024;
        let n = 1024; // per-expert cols in B (total = n * e = 4096)
        let k = 1024;
        let rank = 32; // noise rank

        let header = IncompleteBlockHeader::new_for_test(0x207FFFFF);
        let config = MiningConfiguration {
            common_dim: k,
            rank,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41, 64, 65, 72, 73, 96, 97, 104, 105]).unwrap(),
            moe: Some(MoEConfig { e: e as u16, top_k: 1 }),
        };
        let proof = mine_moe(m, n, k as usize, header, config, None, false).unwrap();
        verify_plain_proof(&header, &proof, None).unwrap();
    }
}
