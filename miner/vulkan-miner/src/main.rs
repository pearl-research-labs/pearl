use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use vulkan_wrappers::{Buffers, Pipelines, VulkanContext};
use vulkan_miner::mining::MiningLoop;

#[derive(Parser)]
#[command(name = "vulkan-miner", about = "Vulkan 1.3 standalone PoW miner for pearl-gemm")]
struct Cli {
    /// Gateway address (UDS path on Unix, "host:port" on Windows)
    #[cfg_attr(unix, arg(long, default_value = "/tmp/pearl-gateway.sock"))]
    #[cfg_attr(windows, arg(long, default_value = "127.0.0.1:8337"))]
    gateway: String,

    /// Matrix dimensions
    #[arg(long, default_value_t = 1024)]
    m: u32,
    #[arg(long, default_value_t = 1024)]
    n: u32,
    #[arg(long, default_value_t = 1024)]
    k: u32,

    /// Noise rank (must be power of 2, divisible by 32)
    #[arg(long, default_value_t = 128)]
    rank: u32,

    /// K3 tile dimensions
    #[arg(long, default_value_t = 16)]
    tile_m: u32,
    #[arg(long, default_value_t = 16)]
    tile_n: u32,
    #[arg(long, default_value_t = 128)]
    tile_k: u32,

    /// Only run one iteration (for testing)
    #[arg(long)]
    once: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    tracing::info!(
        "Initialising Vulkan 1.3 compute context ({}x{}x{}, r={})",
        cli.m,
        cli.n,
        cli.k,
        cli.rank
    );

    // 1. Vulkan context
    let ctx = VulkanContext::init()?;
    tracing::info!("Vulkan context created");

    // 2. Allocate buffers
    let buffers = Buffers::new(&ctx, cli.m, cli.n, cli.k, cli.rank)?;
    tracing::info!(
        "Buffers allocated ({:.2} MB)",
        buffers.allocation_size as f64 / 1_048_576.0
    );

    // 3. Create pipelines
    let pipelines = Pipelines::new(&ctx, &buffers)?;
    tracing::info!("Compute pipelines created");

    // 4. Mining loop state
    let mut mining_loop = MiningLoop::new(ctx, pipelines, buffers, cli.tile_m, cli.tile_n, cli.tile_k);

    // 5. Connect to gateway (retry until available)
    let mut gateway = loop {
        match vulkan_miner::gateway::client::GatewayClient::connect(&cli.gateway).await {
            Ok(g) => break g,
            Err(e) => {
                tracing::warn!("Failed to connect to gateway: {}; retrying in 1s", e);
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    };
    tracing::info!("Connected to gateway at {}", cli.gateway);

    // Main mining loop
    let mut iteration = 0u64;
    loop {
        let job = gateway.get_job().await?;
        let job_key = blake3::hash(&job.prev_block);
        let hash_a = blake3::keyed_hash(job_key.as_bytes(), b"hash_a");
        let hash_b = blake3::keyed_hash(job_key.as_bytes(), b"hash_b");
        tracing::info!(
            "Job received: prev_block={} target={}...",
            hex::encode(&job.prev_block[..4]),
            hex::encode(&job.target[..4])
        );

        loop {
            match mining_loop.run_iteration(
                &job_key,
                iteration,
                cli.m,
                cli.n,
                cli.k,
                cli.rank,
                hash_a.as_bytes(),
                hash_b.as_bytes(),
                &job.target,
            ) {
                Ok(Some((tile_row, tile_col))) => {
                    tracing::info!(
                        "Block found at iteration {} tile ({}, {})",
                        iteration,
                        tile_row,
                        tile_col
                    );
                    let seed = vulkan_miner::rng::compute_seed(&job_key, iteration);
                    let (a_bytes, bt_bytes) = vulkan_miner::proof::derive_matrix_bytes(seed, cli.m, cli.n, cli.k);
                    let rows_pattern: Vec<u32> = (0..cli.tile_m).collect();
                    let cols_pattern: Vec<u32> = (0..cli.tile_n).collect();
                    let proof = vulkan_miner::proof::build_merkle_proof(
                        &a_bytes,
                        &bt_bytes,
                        cli.m as usize,
                        cli.n as usize,
                        cli.k as usize,
                        tile_row,
                        tile_col,
                        cli.tile_m,
                        cli.tile_n,
                        &rows_pattern,
                        &cols_pattern,
                        job_key.as_bytes(),
                        cli.rank as usize,
                    )?;
                    let proof_bytes = bincode::serialize(&proof)?;
                    gateway.submit_plain_proof(&proof_bytes, &job).await?;
                    tracing::info!("Block submitted");
                    if cli.once {
                        return Ok(());
                    }
                    break;
                }
                Ok(None) => {
                    iteration += 1;
                    if cli.once {
                        tracing::info!("One iteration completed (--once)");
                        return Ok(());
                    }
                    if iteration % 1000 == 0 {
                        tracing::info!("{} iterations completed", iteration);
                    }
                }
                Err(e) => {
                    tracing::error!("Iteration {} failed: {}; retrying after backoff", iteration, e);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                }
            }
        }
    }
}


