#[cfg(unix)]
use tikv_jemallocator::Jemalloc;

#[cfg(unix)]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use log::info;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::time::Instant;
use zk_pow::api::proof::MoEConfig;
use zk_pow::api::{
    proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, PrivateProofParams, PublicProofParams},
    prove, verify,
};
use zk_pow::circuit::circuit_utils::CircuitCache;
use zk_pow::circuit::embedded_cache;
use zk_pow::circuit::pearl_circuit::{PearlRecursion, RecursionCircuit};
use zk_pow::ffi::mine::mine_moe;
use zk_pow::ffi::plain_proof::PlainProof;

fn test_block_header(nbits: u32) -> IncompleteBlockHeader {
    IncompleteBlockHeader {
        version: 0,
        prev_block: [1; 32],
        merkle_root: [2; 32],
        timestamp: 0x66666666,
        nbits,
    }
}

fn default_mining_config(common_dim: u32, rank: u16) -> MiningConfiguration {
    MiningConfiguration {
        common_dim,
        rank,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 64, 65, 72, 73]).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 64, 65, 72, 73]).unwrap(),
        moe: None,
    }
}

/// Larger h=w=16 config used by `bench()` to exercise `h+w = 32`.
fn bench_mining_config(common_dim: u32, rank: u16) -> MiningConfiguration {
    MiningConfiguration {
        common_dim,
        rank,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&(0..64).collect::<Vec<_>>()).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&(0..2).collect::<Vec<_>>()).unwrap(),
        moe: None,
    }
}

// for benchmarking
fn setup(
    common_dim: usize,
    nbits: Option<u32>,
    mining_configuration: MiningConfiguration,
) -> (PublicProofParams, PrivateProofParams) {
    let mut rng = StdRng::seed_from_u64(0x0);
    let tile_h = mining_configuration.rows_pattern.size() as usize;
    let tile_w = mining_configuration.cols_pattern.size() as usize;

    let mut rand_matrix = |rows: usize| -> Vec<Vec<i8>> {
        (0..rows)
            .map(|_| (0..common_dim).map(|_| (rng.next_u32() % 128) as i8 - 64).collect())
            .collect()
    };
    let s_a = rand_matrix(tile_h);
    let s_b = rand_matrix(tile_w);

    let private_params = PrivateProofParams {
        s_a,
        s_b,
        s_routing: vec![],
        external_msgs: vec![],
        external_cvs: vec![],
    };

    let block_header = test_block_header(nbits.unwrap_or(0x1D2FFFFF));

    let mut public_params = PublicProofParams::new_dummy(
        block_header,
        mining_configuration,
        6144, // m: rows of A
        4096, // n: columns of B
        128,
        256,
    );
    let private_params = public_params.fill_dummy_merkle_proof(private_params).unwrap();

    (public_params, private_params)
}

fn test_invalid_with_cache() {
    info!("\n========== Testing Invalid Proofs with Cache ==========");

    let mut cache = CircuitCache::default();

    // Test invalid commitment hash
    info!("\n=== Testing Invalid Commitment Hash ===");
    let (mut public_params, private_params) = setup(8192, None, default_mining_config(8192, 128));
    let proof = prove::prove_block(&mut public_params, private_params, &mut cache).unwrap();
    public_params.hash_a[0] ^= 1;
    let res = verify::verify_block(&public_params, &proof, &mut cache);
    assert!(res.is_err(), "Verification wrongly accepted a corrupted proof");
    info!(
        "Invalid commitment hash test passed. Verification error: {}",
        res.err().unwrap()
    );

    info!("\n=== Testing hard difficulty ===");
    let (mut public_params, private_params) = setup(8192, Some(0x170FFFFF), default_mining_config(8192, 128));
    let proof = prove::prove_block(&mut public_params, private_params, &mut cache).unwrap();
    let res = verify::verify_block(&public_params, &proof, &mut cache);
    assert!(res.is_err(), "Verification wrongly accepted a corrupted proof");
    info!("Hard difficulty test passed. Verification error: {}", res.err().unwrap());

    info!("\n=== All Invalid Tests Passed ===");
}

fn bench_double_prove_profile() {
    info!("\n========== Benchmark: Prove with Cache ==========");

    // Create a shared cache
    let mut cache = CircuitCache::default();

    // Cache warming using warmup_prove (NOT profiled)
    info!("\n=== Cache Warming (Building Circuits) - NOT PROFILED ===");
    let start = Instant::now();
    prove::warmup_prove(default_mining_config(32768, 128), &mut cache).unwrap();
    let warmup_time = start.elapsed();
    info!("Cache warmup time: {:?}", warmup_time);
    info!("Note: Warmup includes circuit building, FFT tables, and first-run JIT overhead");

    // Profiled prove - circuits are now cached
    info!("\n=== Profiled Prove (Using Cached Circuits) ===");
    let (mut public_params, private_params) = setup(32768, None, default_mining_config(32768, 128));
    let start = Instant::now();
    let _proof = prove::prove_block(&mut public_params, private_params, &mut cache).unwrap();
    let prove_time = start.elapsed();
    info!("Prove time: {:?}", prove_time);
}

fn bench() {
    info!("\n========== Benchmark: Prove Time vs Problem Size ==========");

    let mut cache = CircuitCache::default();

    for common_dim in [4096 * 7 / 8, 8192 * 7 / 8, 16384 * 7 / 8, 32768 * 7 / 8, 65536 * 7 / 8] {
        let config = bench_mining_config(common_dim as u32, 128);
        let h_plus_w = (config.rows_pattern.size() + config.cols_pattern.size()) as usize;
        let witness_kb = h_plus_w * common_dim / 1024;
        info!(
            "\n=== Testing common_dim = {} with tile_size = {} | raw witness = {} KB ===",
            common_dim, 64, witness_kb,
        );

        // First prove (warmup for this size) - disable logging
        let prev_level = log::max_level();
        log::set_max_level(log::LevelFilter::Off);
        prove::warmup_prove(config, &mut cache).unwrap();
        log::set_max_level(prev_level);

        // Second prove (timed)
        let (mut public_params2, private_params2) = setup(common_dim, Some(0x1F0FFFFF), config);
        let start = Instant::now();
        let proof2 = prove::prove_block(&mut public_params2, private_params2, &mut cache).unwrap();
        let prove_time = start.elapsed();
        info!("  Prove time: {:?}", prove_time);

        // Verify the second proof (timed)
        let start = Instant::now();
        verify::verify_block(&public_params2, &proof2, &mut cache).unwrap();
        let verify_time = start.elapsed();
        info!("  Verify time: {:?}", verify_time);
    }

    info!("\n=== Benchmark Complete ===");
}

fn bench_fill_cache() {
    info!("\n========== Benchmark: Load Embedded Cache ==========");
    let start = Instant::now();
    let cache = CircuitCache::from_bytes(embedded_cache::CACHE_DATA).unwrap_or_default();
    let load_time = start.elapsed();
    info!(
        "Cache load time: {:?} || # first circuits: {}, # second circuits: {}",
        load_time,
        cache.verifier_circuits_1.len(),
        cache.verifier_circuits_2.len()
    );

    // Benchmark 2: Fill verifier cache from scratch
    info!("\n=== Filling Verifier Cache from Scratch ===");
    let mut fresh_cache = CircuitCache::default();
    let start = Instant::now();
    PearlRecursion::fill_verifier_cache(&mut fresh_cache);
    let fill_time = start.elapsed();
    info!(
        "Cache fill time: {:?} || # first circuits: {}, # second circuits: {}",
        fill_time,
        fresh_cache.verifier_circuits_1.len(),
        fresh_cache.verifier_circuits_2.len()
    );
}

/// Test using FFI mine path with a specific rank
fn test_ffi_mine_prove_verify_with_rank(rank: u16, is_moe: bool) {
    info!("\n========== Testing FFI Mine -> Prove -> Verify (rank={}) ==========", rank);

    let block_header = test_block_header(0x1D2FFFFF);

    let m = 6144;
    let n = 4096;
    let k = (16 * rank).max(1024) as usize + 192;
    let mut mining_config = default_mining_config(k as u32, rank);

    if is_moe {
        mining_config.moe = Some(MoEConfig { e: 5, top_k: 2 });
    }

    info!("Mining with FFI: m={}, n={}, k={}, rank={}", m, n, k, mining_config.rank);

    // Step 1: Mine using FFI (same path as Python)
    let start = Instant::now();
    let pow_proof: PlainProof = mine_moe(m, n, k, block_header, mining_config, None, false).expect("Mining failed");

    info!("Mining took {:?}", start.elapsed());

    if let Some(moe) = &pow_proof.moe {
        info!(
            "MoE : m={}, n_e={}, k={}, e={}, expert_idx={}",
            pow_proof.m, pow_proof.n, pow_proof.k, moe.e, moe.expert_idx
        );
    } else {
        info!(
            "PlainProof: m={}, n={}, k={}, noise_rank={}",
            pow_proof.m, pow_proof.n, pow_proof.k, pow_proof.noise_rank
        );
    }
    info!("  A row indices: {:?}", pow_proof.a.row_indices);
    info!("  B col indices: {:?}", pow_proof.bt.row_indices);

    // Step 2: Parse plain proof to get private/public params
    let start = Instant::now();
    let (private_params, mut public_params) = pow_proof.parse_proof(block_header).expect("Failed to parse plain proof");
    info!("Parsing took {:?}", start.elapsed());

    // Step 3: Prove
    let mut cache = CircuitCache::default();
    let start = Instant::now();
    let proof = prove::prove_block(&mut public_params, private_params, &mut cache).expect("Proving failed");
    info!("Proving took {:?}", start.elapsed());

    // Step 4: Verify
    let start = Instant::now();
    let result = verify::verify_block(&public_params, &proof, &mut cache);
    info!("Verification took {:?}", start.elapsed());

    match result {
        Ok(_) => info!("✓ FFI mine -> prove -> verify (rank={}): SUCCESS", rank),
        Err(e) => {
            info!("✗ FFI mine -> prove -> verify (rank={}): FAILED - {}", rank, e);
            panic!("Verification failed for rank={}: {}", rank, e);
        }
    }
}

/// Test using FFI mine path (same as Python tests use)
fn test_correctness(is_moe: bool) {
    for rank in [32, 64, 128] {
        test_ffi_mine_prove_verify_with_rank(rank, is_moe);
    }
}

// Run with: RUST_LOG=debug RUSTFLAGS="-C target-cpu=native" cargo run --release --bin prove_verify -- correctness
fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        test_correctness(false);
        test_correctness(true);
        test_invalid_with_cache();
        bench_double_prove_profile();
        bench_fill_cache();
        return;
    }

    match args[1].as_str() {
        "correctness" => test_correctness(false),
        "correctness_moe" => test_correctness(true),
        "profile" => bench_double_prove_profile(),
        "bench" => bench(),
        "invalid" => test_invalid_with_cache(),
        "cache" => bench_fill_cache(),
        _ => {
            info!("Unknown test: {}", args[1]);
            info!("Available tests: correctness, profile, bench, invalid, cache, ffi");
        }
    }
}
