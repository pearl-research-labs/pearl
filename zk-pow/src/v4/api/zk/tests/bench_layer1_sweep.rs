//! Layer-1 (stage-1 wrapper) proving anatomy: a per-phase breakdown of stage-1
//! proving, with repeat statistics.
//!
//! The stock `CircuitData::prove` discards the internal `TimingTree`, so the phase
//! split (witness generation, full witness, wire commitment/FFT, partial products,
//! quotient polys, FRI opening) is invisible in normal benches. This harness calls
//! `Fp8WrapperCircuits::prove_stage1_timed` — the same code path production takes,
//! with a live tree.
//!
//! Environment:
//! - `BENCH_WARMUP` (default 1): `0` skips the `warmup_prove` call (cold A/B).
//!
//! Run with:
//! `RAYON_NUM_THREADS=64 cargo test --release bench_layer1_sweep -- --ignored --nocapture`

use std::time::{Duration, Instant};

use plonky2::util::timing::TimingTree;

use super::super::*;
use super::bench_pipeline_tests::bench_job;

/// The geometry riding the ladder's top (degree-22 InputQuant, 21 Matmul).
const SWEEP_GEO: (usize, usize) = (128, 33280);

/// Repetitions for the stage-1 statistics; the phase split uses rep 0's tree.
const REPS: usize = 6;

#[test]
#[ignore]
fn bench_layer1_sweep() {
    let _ = env_logger::builder().format_timestamp(None).try_init();
    let device = Device::H100;
    let do_warmup = std::env::var("BENCH_WARMUP").map(|v| v != "0").unwrap_or(true);
    println!("SWEEP warmup={do_warmup}");

    // ---- Shared setup (LUT oracle + consensus wrapper pair). ----
    let mut setup_timing = TimingTree::default();
    let mut prover = Fp8Prover::setup_with_timing(device, &mut setup_timing).expect("fp8 setup");
    if do_warmup {
        let t = Instant::now();
        prover.warmup_prove(device).expect("warmup prove of the sample job");
        println!("SWEEP warmup_seconds={:.3}", t.elapsed().as_secs_f64());
    }
    let setup = prover.setups.get(&device).expect("setup was prebuilt");

    let (m, k) = SWEEP_GEO;
    let (header, plain) = bench_job(m, k);

    // ---- Layer 0, once: its batch proof feeds every stage-1 repetition. ----
    let mut timing = TimingTree::new("sweep", log::Level::Info);
    let (public, job, batch_proof) = prove_batch_statement(&header, &plain, setup, &mut timing).expect("layer-0 prove");
    let digest = public.digest(&header);
    println!("SWEEP k={k} degree_bits={:?}", job.system.degree_bits());

    // ---- One untimed warm-up stage-1 rep (page in the batch-proof-sized state). ----
    let mut warm_tree = TimingTree::new("stage1-warmrep", log::Level::Debug);
    drop(
        setup
            .circuits
            .prove_stage1_timed(&batch_proof, &job.system, digest, &mut warm_tree)
            .expect("stage-1 warm-up prove"),
    );
    warm_tree.filter(Duration::from_millis(100)).print();

    // ---- Stage-1 stats + phase split through the timed test-only path. ----
    let mut times = Vec::with_capacity(REPS);
    for rep in 0..REPS {
        let mut tree = TimingTree::new("stage1", log::Level::Debug);
        let t = Instant::now();
        let proof = setup
            .circuits
            .prove_stage1_timed(&batch_proof, &job.system, digest, &mut tree)
            .unwrap_or_else(|e| panic!("stage-1 prove failed on rep {rep}: {e:#}"));
        times.push(t.elapsed().as_secs_f64());
        if rep == 0 {
            println!("SWEEP stage-1 phase split (rep 0, {:.3}s):", times[0]);
            tree.filter(Duration::from_millis(1)).print();
        }
        drop(proof);
    }
    report("stage-1", &times);

    // ---- Sanity: the production wrap on the same batch proof must verify. ----
    let mut prod_timing = TimingTree::default();
    let wrapped = setup
        .circuits
        .prove(&job.system, &batch_proof, digest, &mut prod_timing)
        .expect("production wrap");
    let public_data = public.to_bytes();
    let proof_data = compact_proof_data(&job.system, &wrapped).expect("compact encode");
    let statement = decode_statement(&public_data).expect("decode statement");
    let verifier = Fp8Verifier::generate(&statement, &header, &mut TimingTree::default()).expect("verifier setup");
    verifier
        .verify_block(&header, &plain.ancestor_chain, &public_data, &proof_data)
        .expect("the wrapped proof must verify");
    println!("SWEEP verified=true");
}

fn report(label: &str, times: &[f64]) {
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = sorted.first().copied().unwrap_or(f64::NAN);
    let med = sorted[sorted.len() / 2];
    let max = sorted.last().copied().unwrap_or(f64::NAN);
    println!("SWEEP {label}: min={min:.3} median={med:.3} max={max:.3} (n={})", times.len());
}
