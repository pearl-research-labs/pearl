use anyhow::Result;
use clap::Parser;

use vulkan_miner::buffers::MiningBuffers;
use vulkan_miner::context::VulkanContext;
use vulkan_miner::mining::MiningLoop;
use vulkan_miner::pipelines::KernelPipelines;

#[derive(Parser)]
#[command(name = "vulkan-miner", about = "Vulkan 1.3 standalone PoW miner for pearl-gemm")]
struct Cli {
    /// Gateway Unix socket path
    #[arg(long, default_value = "/tmp/pearl-gateway.sock")]
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
    let buffers = MiningBuffers::new(&ctx, cli.m, cli.n, cli.k, cli.rank)?;
    tracing::info!("Buffers allocated ({:.2} MB)", buffers.allocation_size as f64 / 1_048_576.0);

    // 3. Create pipelines (load SPIR-V, create descriptor sets, create pipelines)
    let sizes = MiningBuffers::buffer_sizes(cli.m, cli.n, cli.k, cli.rank);
    let buffer_list = [
        buffers.a, buffers.b, buffers.eal, buffers.ear,
        buffers.ebl, buffers.ebr, buffers.jackpot, buffers.hash_a,
        buffers.hash_b, buffers.target, buffers.result,
    ];
    let size_list: Vec<u64> = sizes.iter().map(|(s, _)| *s).collect();
    let pipelines = KernelPipelines::new(&ctx, &buffer_list, &size_list)?;
    tracing::info!("Compute pipelines created");

    // 4. Mining loop
    let mut mining_loop = MiningLoop {
        context: ctx,
        pipelines,
        buffers,
        tile_m: cli.tile_m,
        tile_n: cli.tile_n,
        tile_k: cli.tile_k,
    };

    // Connect to gateway
    let mut gateway = vulkan_miner::gateway::client::GatewayClient::connect(&cli.gateway).await?;
    tracing::info!("Connected to gateway at {}", cli.gateway);

    // Main mining loop
    let mut iteration = 0u64;
    loop {
        let job = gateway.get_job().await?;
        tracing::info!(
            "Job received: m={} n={} k={} target={}...",
            job.m,
            job.n,
            job.k,
            hex::encode(&job.target[..4])
        );

        let job_key = blake3::hash(&job.header_prev_block); // simplified; actual job_key includes header+config

        loop {
            match mining_loop.run_iteration(
                &job_key,
                iteration,
                job.m,
                job.n,
                job.k,
                job.rank,
                &[0u8; 32], // hash_a — placeholder
                &[0u8; 32], // hash_b — placeholder
                &job.target,
            ) {
                Ok(Some((tile_row, tile_col))) => {
                    tracing::info!(
                        "Block found at iteration {} tile ({}, {})",
                        iteration,
                        tile_row,
                        tile_col
                    );
                    // Build and submit proof
                    let proof = build_dummy_proof(job.m, job.n, job.k, job.rank);
                    let proof_bytes = bincode::serialize(&proof)?;
                    gateway.submit_block(&proof_bytes).await?;
                    tracing::info!("Block submitted");
                    break; // Get new job
                }
                Ok(None) => {
                    iteration += 1;
                    if cli.once && iteration >= 1 {
                        tracing::info!("One iteration completed (--once)");
                        return Ok(());
                    }
                    if iteration % 1000 == 0 {
                        tracing::info!("{} iterations completed", iteration);
                    }
                }
                Err(e) => {
                    tracing::error!("Iteration failed: {}", e);
                    break; // Get new job
                }
            }
        }
    }
}

/// Temporary dummy proof for testing. Replace with real proof construction.
fn build_dummy_proof(m: u32, n: u32, k: u32, noise_rank: u32) -> zk_pow::ffi::plain_proof::PlainProof {
    use zk_pow::ffi::plain_proof::MatrixMerkleProof;
    use zk_pow::ffi::plain_proof::PlainProof;

    let empty_proof = pearl_blake3::MerkleProof {
        leaf_data: vec![],
        leaf_indices: vec![],
        total_leaves: 0,
        root: [0u8; 32],
        siblings: vec![],
    };

    PlainProof {
        m: m as usize,
        n: n as usize,
        k: k as usize,
        noise_rank: noise_rank as usize,
        a: MatrixMerkleProof {
            proof: empty_proof.clone(),
            row_indices: vec![],
        },
        bt: MatrixMerkleProof {
            proof: empty_proof,
            row_indices: vec![],
        },
    }
}
