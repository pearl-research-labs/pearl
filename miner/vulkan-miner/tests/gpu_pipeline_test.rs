//! GPU pipeline integration test.
//!
//! Verifies the full K1→K2→K3 compute pipeline via `MiningLoop::run_iteration`.
//! Uses an all-0xFF target to guarantee a block is found on a small matrix.
//!
//! Run:  cargo +nightly test --test gpu_pipeline_test -- --ignored

use anyhow::Result;
use vulkan_miner::mining::MiningLoop;
use vulkan_wrappers::{Buffers, Pipelines, VulkanContext};

const M: u32 = 8;
const N: u32 = 8;
const K: u32 = 8;
const R: u32 = 16;
const TILE_M: u32 = 4;
const TILE_N: u32 = 4;
const TILE_K: u32 = 4;

#[test]
#[ignore]
fn gpu_pipeline_full_k1_k2_k3() -> Result<()> {
    let ctx = VulkanContext::init()?;
    let buffers = Buffers::new(&ctx, M, N, K, R)?;
    let pipelines = Pipelines::new(&ctx, &buffers)?;

    let mut mining_loop = MiningLoop::new(ctx, pipelines, buffers, TILE_M, TILE_N, TILE_K);

    let job_key = blake3::hash(b"gpu_test_job");
    let target = [0xFFu8; 32];

    match mining_loop.run_iteration(
        &job_key,
        0,
        M, N, K, R,
        &[0xABu8; 32],
        &[0xCDu8; 32],
        &target,
    )? {
        Some((tile_row, tile_col)) => {
            assert!(tile_row < M / TILE_M, "tile_row {} out of range", tile_row);
            assert!(tile_col < N / TILE_N, "tile_col {} out of range", tile_col);
            Ok(())
        }
        None => {
            panic!("GPU did not find a block with all-0xFF target; this is unexpected for small matrices");
        }
    }
}
