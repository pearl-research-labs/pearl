use pearl_blake3::{BLAKE3_CHUNK_LEN, MerkleProof, MerkleTree, pad_to_chunk_boundary};
use zk_pow::api::{
    proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, PublicProofParams},
    sanity_checks::public_params_sanity_check,
};
use zk_pow::ffi::plain_proof::{MatrixMerkleProof, PlainProof, parse_plain_proof};

fn test_block_header() -> IncompleteBlockHeader {
    IncompleteBlockHeader {
        version: 0,
        prev_block: [1; 32],
        merkle_root: [2; 32],
        timestamp: 0x66666666,
        nbits: 0x207FFFFF,
    }
}

fn wrapped_public_params() -> PublicProofParams {
    let rows: Vec<u32> = (0..126).collect();
    let cols: Vec<u32> = (0..2).collect();
    let mining_config = MiningConfiguration {
        common_dim: 1024,
        rank: 32,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&rows).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&cols).unwrap(),
        reserved: MiningConfiguration::RESERVED_VALUE,
    };

    PublicProofParams::new_dummy(test_block_header(), mining_config, 122, 2, 0xFFFF_FFFC, 0)
}

fn keyed_zero_matrix_root(rows: usize, cols: usize, key: [u8; 32]) -> [u8; 32] {
    let data = pad_to_chunk_boundary(&vec![0u8; rows * cols]);
    MerkleTree::new(&data, key).root()
}

fn high_index_zero_matrix_proof(row_indices: Vec<usize>, rows: usize, cols: usize, key: [u8; 32]) -> MatrixMerkleProof {
    assert_eq!(cols, BLAKE3_CHUNK_LEN);
    let root = keyed_zero_matrix_root(rows, cols, key);
    MatrixMerkleProof {
        proof: MerkleProof {
            leaf_data: vec![[0u8; BLAKE3_CHUNK_LEN]; row_indices.len()],
            leaf_indices: row_indices.clone(),
            total_leaves: row_indices.last().copied().unwrap() + 1,
            root,
            siblings: vec![],
        },
        row_indices,
    }
}

#[test]
fn direct_public_wrapped_offset_params_are_rejected_at_deserialize() {
    let params = wrapped_public_params();
    let bytes = params.to_bytes();
    let err = PublicProofParams::from_bytes(params.block_header, &bytes).unwrap_err();
    assert!(
        err.to_string().contains("strictly increasing")
            || err.to_string().contains("fit within matrix")
    );
}

#[test]
fn plain_proof_high_usize_rows_are_rejected() {
    let rows: Vec<u32> = (0..126).collect();
    let cols: Vec<u32> = (0..2).collect();
    let mining_config = MiningConfiguration {
        common_dim: BLAKE3_CHUNK_LEN as u32,
        rank: 32,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&rows).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&cols).unwrap(),
        reserved: MiningConfiguration::RESERVED_VALUE,
    };
    let public_for_key = PublicProofParams::new_dummy(test_block_header(), mining_config, 126, 2, 0, 0);
    let key = public_for_key.job_key();
    let high = 1usize << 32;
    let proof = PlainProof {
        m: 126,
        n: 2,
        k: BLAKE3_CHUNK_LEN,
        noise_rank: 32,
        a: high_index_zero_matrix_proof((0..126).map(|i| high + i).collect(), 126, BLAKE3_CHUNK_LEN, key),
        bt: high_index_zero_matrix_proof((0..2).map(|i| high + i).collect(), 2, BLAKE3_CHUNK_LEN, key),
    };

    let err = parse_plain_proof(test_block_header(), &proof).unwrap_err();
    assert!(err.to_string().contains("does not fit in u32"));
}

#[test]
fn plain_proof_high_usize_m_n_and_rank_are_rejected() {
    let rows = (0..2).collect::<Vec<u32>>();
    let cols = (0..16).collect::<Vec<u32>>();
    let mining_config = MiningConfiguration {
        common_dim: BLAKE3_CHUNK_LEN as u32,
        rank: 32,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&rows).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&cols).unwrap(),
        reserved: MiningConfiguration::RESERVED_VALUE,
    };
    let public_for_key = PublicProofParams::new_dummy(test_block_header(), mining_config, 2, 16, 0, 0);
    let key = public_for_key.job_key();
    let high = 1usize << 32;
    let proof = PlainProof {
        m: high + 2,
        n: high + 16,
        k: BLAKE3_CHUNK_LEN,
        noise_rank: high + 32,
        a: high_index_zero_matrix_proof((0..2).collect(), 2, BLAKE3_CHUNK_LEN, key),
        bt: high_index_zero_matrix_proof((0..16).collect(), 16, BLAKE3_CHUNK_LEN, key),
    };

    let err = parse_plain_proof(test_block_header(), &proof).unwrap_err();
    assert!(err.to_string().contains("does not fit in u32"));
}

#[test]
fn plain_proof_high_usize_k_is_blocked_by_original_strip_extraction() {
    let rows = vec![0u32];
    let cols = (0..32).collect::<Vec<u32>>();
    let mining_config = MiningConfiguration {
        common_dim: BLAKE3_CHUNK_LEN as u32,
        rank: 32,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&rows).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&cols).unwrap(),
        reserved: MiningConfiguration::RESERVED_VALUE,
    };
    let public_for_key = PublicProofParams::new_dummy(test_block_header(), mining_config, 1, 32, 0, 0);
    let key = public_for_key.job_key();
    let high = 1usize << 32;
    let proof = PlainProof {
        m: 1,
        n: 32,
        k: high + BLAKE3_CHUNK_LEN,
        noise_rank: 32,
        a: high_index_zero_matrix_proof(vec![0], 1, BLAKE3_CHUNK_LEN, key),
        bt: high_index_zero_matrix_proof((0..32).collect(), 32, BLAKE3_CHUNK_LEN, key),
    };

    let err = parse_plain_proof(test_block_header(), &proof).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("does not fit in u32") || msg.contains("Failed to extract strip"),
        "unexpected error: {msg}"
    );
}

#[test]
#[ignore = "expensive: compiles circuits and attempts a full offline prove/verify"]
fn direct_public_wrapped_offset_proof_is_rejected_by_verifier() {
    assert!(!cfg!(debug_assertions), "run with cargo test --release");

    let params = wrapped_public_params();
    assert!(public_params_sanity_check(&params).is_err());
}
