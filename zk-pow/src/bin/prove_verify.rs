#[cfg(unix)]
use tikv_jemallocator::Jemalloc;

#[cfg(unix)]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use log::info;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::time::Instant;
use zk_pow::api::{
    proof::{
        IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, PrivateProofParams, PublicProofParams, ZKProof,
    },
    prove, verify,
};
use zk_pow::circuit::circuit_utils::CircuitCache;
use zk_pow::circuit::embedded_cache;
use zk_pow::circuit::pearl_circuit::{PearlRecursion, RecursionCircuit};
use zk_pow::ffi::mine::mine;
use zk_pow::ffi::plain_proof::parse_plain_proof;

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
        reserved: MiningConfiguration::RESERVED_VALUE,
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
        reserved: MiningConfiguration::RESERVED_VALUE,
    }
}

fn public_alpha_mining_config(common_dim: u32, rank: u16) -> MiningConfiguration {
    MiningConfiguration {
        common_dim,
        rank,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&[0, 1]).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&(0..64).collect::<Vec<_>>()).unwrap(),
        reserved: MiningConfiguration::RESERVED_VALUE,
    }
}

// for benchmarking
fn setup(
    common_dim: usize,
    nbits: Option<u32>,
    mining_configuration: MiningConfiguration,
) -> (PublicProofParams, PrivateProofParams) {
    setup_with_shape(common_dim, nbits, mining_configuration, 6144, 4096, 128, 256)
}

fn setup_with_shape(
    common_dim: usize,
    nbits: Option<u32>,
    mining_configuration: MiningConfiguration,
    m: u32,
    n: u32,
    t_rows: u32,
    t_cols: u32,
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
        external_msgs: vec![],
        external_cvs: vec![],
    };

    let block_header = test_block_header(nbits.unwrap_or(0x1D2FFFFF));

    let mut public_params = PublicProofParams::new_dummy(block_header, mining_configuration, m, n, t_rows, t_cols);
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

fn bench_public_alpha_rank256_profile() {
    info!("\n========== Benchmark: Public Alpha rank-256 exact shape ==========");

    let mut cache = CircuitCache::default();
    let rank: u16 = 256;
    let common_dim = 4096usize;
    let m = 131072u32;
    let n = 131072u32;
    let config = public_alpha_mining_config(common_dim as u32, rank);
    let h = config.rows_pattern.size();
    let w = config.cols_pattern.size();
    info!(
        "shape=m:{} n:{} k:{} rank:{} h:{} w:{} nbits:{:#x} workers:{}",
        m,
        n,
        common_dim,
        rank,
        h,
        w,
        0x207FFFFFu32,
        prove::current_prove_thread_count(),
    );

    let (mut warm_public, warm_private) = setup_with_shape(common_dim, Some(0x207FFFFF), config, m, n, 0, 0);
    let start = Instant::now();
    let _warm_proof = prove::prove_block(&mut warm_public, warm_private, &mut cache).unwrap();
    info!("warmup_same_shape_prove={:?}", start.elapsed());

    let (mut public_params, private_params) = setup_with_shape(common_dim, Some(0x207FFFFF), config, m, n, 0, 0);
    let start = Instant::now();
    let proof = prove::prove_block(&mut public_params, private_params, &mut cache).unwrap();
    let prove_time = start.elapsed();
    info!("cached_public_rank256_prove={:?}", prove_time);

    let start = Instant::now();
    verify::verify_block(&public_params, &proof, &mut cache).unwrap();
    info!("verify={:?}", start.elapsed());
}

fn bench_split() {
    info!("\n========== Architecture C: miner STARK#0 / pool recursion split ==========");
    // Mine a REAL solution (so the block difficulty / jackpot check passes), then
    // split-prove it across the miner/pool boundary. Production-size config.
    let rank: u16 = 128;
    let block_header = test_block_header(0x1D2FFFFF);
    let (m, n, k) = (6144usize, 4096usize, 32768usize);
    let mining_config = default_mining_config(k as u32, rank);
    info!("Mining a real solution (m={} n={} k={} rank={})...", m, n, k, rank);
    let plain_proof = mine(m, n, k, block_header, mining_config, None, false).expect("mining failed");
    let (private_params, mut public_params) = parse_plain_proof(block_header, &plain_proof).expect("parse_plain_proof failed");

    let mut cache = CircuitCache::default();
    let (proof, miner, pool, ship) = prove::prove_block_split(&mut public_params, private_params, &mut cache).unwrap();
    let total = miner + pool;
    info!(
        "MINER (STARK#0 only, GPU-accel under PEARL_GPU_COMMIT): {:?}  ships {} bytes",
        miner, ship
    );
    info!("POOL  (Recursion#1+#2 only, no STARK#0):                {:?}", pool);
    info!(
        ">>> Pool sheds {:.0}% of the prove to the miner GPU (pool now does only {:?} of {:?})",
        100.0 * miner.as_secs_f64() / total.as_secs_f64(),
        pool,
        total
    );

    let start = Instant::now();
    verify::verify_block(&public_params, &proof, &mut cache).unwrap();
    info!(
        "Verify pool-assembled cert (from miner's shipped STARK proof): {:?} — SUCCESS",
        start.elapsed()
    );

    // === FULL WIRE-CERT round-trip: the pool serializes the cert to the EXACT bytes it
    // submits to pearld (public_data(164) | proof_data = pow_bits|rate_bits|zeta|plonky2_proof),
    // and the reconstructed-from-bytes cert must verify. Proves E.2 output is wire-valid. ===
    let (public_data, proof_data) = proof.serialize(&public_params);
    let wire_len = public_data.len() + proof_data.len();
    let (params_rt, proof_rt) = ZKProof::deserialize(public_params.block_header.clone(), &public_data, &proof_data).unwrap();
    // byte-exact: re-serializing the reconstructed cert reproduces the same wire bytes.
    let (pd2, prd2) = proof_rt.serialize(&params_rt);
    assert_eq!(pd2, public_data, "cert public_data wire round-trip not byte-exact");
    assert_eq!(prd2, proof_data, "cert proof_data wire round-trip not byte-exact");
    let start = Instant::now();
    verify::verify_block(&params_rt, &proof_rt, &mut cache).unwrap();
    info!(
        "Verify RECONSTRUCTED wire cert ({} bytes, public_data+proof_data, byte-exact round-trip): {:?} — SUCCESS",
        wire_len,
        start.elapsed()
    );
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
fn test_ffi_mine_prove_verify_with_rank(rank: u16) {
    info!("\n========== Testing FFI Mine -> Prove -> Verify (rank={}) ==========", rank);

    let block_header = test_block_header(0x1D2FFFFF);

    let m = 6144;
    let n = 4096;
    let k = (16 * rank).max(1024) as usize + 192;

    let mining_config = default_mining_config(k as u32, rank);

    info!("Mining with FFI: m={}, n={}, k={}, rank={}", m, n, k, mining_config.rank);

    // Step 1: Mine using FFI (same path as Python)
    let start = Instant::now();
    let plain_proof = mine(m, n, k, block_header, mining_config, None, false).expect("Mining failed");
    info!("Mining took {:?}", start.elapsed());

    info!(
        "PlainProof: m={}, n={}, k={}, noise_rank={}",
        plain_proof.m, plain_proof.n, plain_proof.k, plain_proof.noise_rank
    );
    info!("  A row indices: {:?}", plain_proof.a.row_indices);
    info!("  B col indices: {:?}", plain_proof.bt.row_indices);

    // Step 2: Parse plain proof to get private/public params
    let start = Instant::now();
    let (private_params, mut public_params) = parse_plain_proof(block_header, &plain_proof).expect("Failed to parse plain proof");
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
fn test_correctness() {
    for rank in [32, 64, 128] {
        test_ffi_mine_prove_verify_with_rank(rank);
    }
}

// Run with: RUST_LOG=debug RUSTFLAGS="-C target-cpu=native" cargo run --release --bin prove_verify -- correctness
fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        test_correctness();
        test_invalid_with_cache();
        bench_double_prove_profile();
        bench_fill_cache();
        return;
    }

    match args[1].as_str() {
        "correctness" => test_correctness(),
        "profile" => bench_double_prove_profile(),
        "split" => bench_split(),
        "bench" => bench(),
        "invalid" => test_invalid_with_cache(),
        "cache" => bench_fill_cache(),
        "public-rank256" => bench_public_alpha_rank256_profile(),
        _ => {
            info!("Unknown test: {}", args[1]);
            info!("Available tests: correctness, profile, split, bench, invalid, cache, public-rank256");
        }
    }
}
