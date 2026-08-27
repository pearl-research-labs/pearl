use blake3::CHUNK_LEN;

use zk_pow::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, SeedDerivation};
use zk_pow::api::verify::verify_plain_proof;
use zk_pow::ffi::mine::mine;

#[test]
fn rejects_dimension_wraparound_after_leaf_count_check() {
    let (m, n, k) = (256usize, 128usize, 1024usize);
    let header = IncompleteBlockHeader {
        version: 0,
        prev_block: [1; 32],
        merkle_root: [2; 32],
        timestamp: 0x6666_6666,
        nbits: 0x207f_ffff,
    };
    let config = MiningConfiguration {
        common_dim: k as u32,
        rank: 32,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41]).unwrap(),
        moe: None,
    };

    let mut proof = mine(m, n, k, header, config, None, false, SeedDerivation::Salted).unwrap();
    verify_plain_proof(&header, &proof, None, SeedDerivation::Salted).unwrap();

    proof.m += 1usize << 40;
    proof.a.proof.total_leaves = pearl_blake3::padded_chunk_len(proof.m * proof.k) / CHUNK_LEN;

    assert!(
        verify_plain_proof(&header, &proof, None, SeedDerivation::Salted).is_err(),
        "wrapped m bypassed the declared Merkle leaf-count binding"
    );
}
