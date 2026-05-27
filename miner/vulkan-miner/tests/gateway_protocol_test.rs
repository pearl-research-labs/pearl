//! Integration test for the JSON-RPC 2.0 gateway protocol.
//!
//! Starts a mock gateway server in a background task, connects the real
//! GatewayClient, issues getMiningInfo and submitPlainProof calls, and
//! verifies correct request/response parsing.

mod common;

#[tokio::test]
async fn test_get_mining_info_roundtrip() {
    let addr = common::start_mock_gateway().await;

    let mut client =
        vulkan_miner::gateway::client::GatewayClient::connect(&addr).await.unwrap();

    let job = client.get_job().await.unwrap();

    // Verify the header fields match mock_header_b64()
    let version = u32::from_le_bytes(job.incomplete_header_bytes[0..4].try_into().unwrap());
    assert_eq!(version, 1);
    let prev_block_expected: [u8; 32] = std::array::from_fn(|i| (i + 1) as u8);
    assert_eq!(job.prev_block, prev_block_expected, "prev_block mismatch");
    let merkle_root_expected: [u8; 32] = std::array::from_fn(|i| (i + 0x11) as u8);
    assert_eq!(job.merkle_root, merkle_root_expected, "merkle_root mismatch");
    assert_eq!(job.target[24..32], 0x00FFFFFF_00000000u64.to_be_bytes(), "target mismatch");
    assert!(job.target[..24].iter().all(|&b| b == 0), "upper 24 target bytes should be zero");
}

#[tokio::test]
async fn test_submit_plain_proof_roundtrip() {
    let addr = common::start_mock_gateway().await;

    let mut client =
        vulkan_miner::gateway::client::GatewayClient::connect(&addr).await.unwrap();

    let job = client.get_job().await.unwrap();

    // Build a real proof using the same functions as the production miner
    let job_key = blake3::hash(&job.prev_block);
    let seed = vulkan_miner::rng::compute_seed(&job_key, 0);
    let (a_bytes, bt_bytes) = vulkan_miner::proof::derive_matrix_bytes(seed, 8, 8, 8);
    let proof = vulkan_miner::proof::build_merkle_proof(
        &a_bytes, &bt_bytes,
        8, 8, 8,
        0, 0,  // tile_row, tile_col
        4, 4,  // tile_m, tile_n
        &[0, 1, 2, 3], &[0, 1, 2, 3],
        job_key.as_bytes(),
        16,
    ).unwrap();

    let proof_bytes = bincode::serialize(&proof).unwrap();
    client.submit_plain_proof(&proof_bytes, &job).await.unwrap();
}
