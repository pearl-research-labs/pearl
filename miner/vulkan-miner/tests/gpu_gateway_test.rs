//! End-to-end integration test: mock gateway + real GPU pipeline.
//!
//! 1. Starts a mock JSON-RPC gateway on a random TCP port.
//! 2. Initialises Vulkan, allocates buffers, creates pipelines.
//! 3. Connects to the mock gateway and fetches a `MiningJob`.
//! 4. Runs one GPU mining iteration with an all-0xFF target.
//! 5. Builds a real `PlainProof` from the winning tile coordinates.
//! 6. Serialises and submits the proof via the gateway client.
//!
//! Run:  cargo +nightly test --test gpu_gateway_test -- --ignored --nocapture

mod common;

use anyhow::Result;
use vulkan_miner::gateway::client::GatewayClient;
use vulkan_miner::mining::MiningLoop;
use vulkan_miner::proof::{build_merkle_proof, derive_matrix_bytes};
use vulkan_miner::rng::compute_seed;
use vulkan_wrappers::{Buffers, Pipelines, VulkanContext};

const M: u32 = 8;
const N: u32 = 8;
const K: u32 = 8;
const R: u32 = 16;
const TILE_M: u32 = 4;
const TILE_N: u32 = 4;
const TILE_K: u32 = 4;

#[tokio::test]
#[ignore]
async fn gpu_pipeline_with_mock_gateway() -> Result<()> {
    let addr = common::start_mock_gateway().await;

    let ctx = VulkanContext::init()?;
    let buffers = Buffers::new(&ctx, M, N, K, R)?;
    let pipelines = Pipelines::new(&ctx, &buffers)?;

    let mut mining_loop = MiningLoop::new(ctx, pipelines, buffers, TILE_M, TILE_N, TILE_K);

    let mut client = GatewayClient::connect(&addr).await?;
    let job = client.get_job().await?;

    let job_key = blake3::hash(&job.prev_block);

    let target = [0xFFu8; 32];

    match mining_loop.run_iteration(
        &job_key,
        0,
        M, N, K, R,
        &[0u8; 32],
        &[0u8; 32],
        &target,
    )? {
        Some((tile_row, tile_col)) => {
            let seed = compute_seed(&job_key, 0);
            let (a_bytes, bt_bytes) = derive_matrix_bytes(seed, M, N, K);
            let rows_pattern: Vec<u32> = (0..TILE_M).collect();
            let cols_pattern: Vec<u32> = (0..TILE_N).collect();
            let proof = build_merkle_proof(
                &a_bytes,
                &bt_bytes,
                M as usize, N as usize, K as usize,
                tile_row, tile_col,
                TILE_M, TILE_N,
                &rows_pattern, &cols_pattern,
                job_key.as_bytes(),
                R as usize,
            )?;

            let proof_bytes = bincode::serialize(&proof)?;
            client.submit_plain_proof(&proof_bytes, &job).await?;

            Ok(())
        }
        None => {
            panic!("GPU did not find a block with all-0xFF target; this is unexpected for small matrices");
        }
    }
}
