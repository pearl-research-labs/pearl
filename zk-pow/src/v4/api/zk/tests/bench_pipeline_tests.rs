//! Prover-side benchmark harness: full wrapped-pipeline proving on legal jobs of
//! growing geometry, reporting per-layer wall times with repeat statistics.
//!
//! Not a correctness test — the pipeline tests cover that — but a measuring stick:
//! `RAYON_NUM_THREADS=64 cargo test --release bench_pipeline -- --ignored --nocapture`
//! runs it.
//!
//! Layout:
//! - *setup* is one-time and job-independent (LUT oracle + wrapper circuits),
//!   measured separately from proving;
//! - *Layer 0* is the batch multi-STARK prove (`prove_batch_statement`);
//! - *Layer 1* is the stage-1 universal recursive wrapper;
//! - *Layer 2* is the stage-2 ZK wrap.
//!
//! The wrap stages are timed by the shared `TimingTree` (labels "prove the stage-1
//! wrapper" / "prove the stage-2 (ZK) wrapper"), printed for the last rep of each
//! geometry; layer 0 and the wrap total use plain `Instant`s.
//!
//! Each geometry prints its per-table committed degree bits (the driver's
//! `degree_bits()`), so runs are directly comparable across code versions.
//!
//! Environment:
//! - `BENCH_REPS` (default 6): measured repetitions per geometry, after one
//!   untimed warm-up prove (the box is shared; single runs are noise).
//! - `BENCH_WARMUP` (default 1): `0` skips the `warmup_prove` call so cold-start
//!   costs are visible; any other value warms up before the sweep.

use std::time::{Duration, Instant};

use plonky2::util::timing::TimingTree;

use super::super::*;
use crate::v2::ffi::plain_proof::MatrixMerkleProof;
use crate::v4::api::layout::DimType;
use crate::v4::api::plain_proof::PlainProofV4;
use crate::v4::api::primitives::Sides;
use crate::v4::api::public_params::JobParams;
use crate::v4::circuit::consistency::{fixture_headers, fixture_tree_keys};

/// The lottery tile: rows `[(4, Fold), (16, Blake)]` -> `h = 64`, cols `[(16, Fold)]`
/// -> `w = 16`: 16 Blake lanes, subtile 4*16 = 64, `h*w = 1024` — envelope-legal.
const ROWS_DIMS: &[(u32, DimType)] = &[(4, DimType::Fold), (16, DimType::Blake)];
const COLS_DIMS: &[(u32, DimType)] = &[(16, DimType::Fold)];

/// `(m = n, k)` pairs. InputQuant = `max(h,w)*k = 64k` hits 2^19..2^22; Matmul =
/// `h*w*k/32 = 32k` hits 2^18..2^21; XorFold = 2^10 and Scale = 2^7 sit on stops of
/// both the old dense ladder and the new sparse one, so committed heights are
/// identical across code versions for these jobs.
const GEOMETRIES: &[(usize, usize)] = &[(128, 8192), (128, 16384), (128, 24576), (128, 33280)];

/// One committed int8-plane byte: a splitmix64-style mix of `(seed, i, j)` — mutually
/// incoherent committed rows like honest workloads (the consolidated jackpot census
/// would reject coherent ramps).
fn bench_plane_byte(seed: usize, i: usize, j: usize) -> u8 {
    let mut x = ((seed as u64) << 48) ^ ((i as u64) << 24) ^ j as u64;
    x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    (x % 251) as u8
}

/// A keyed opening of the given rows of `rows_bytes` (the fixture helper, inlined here
/// so the bench is independent of test-fixture refactorings across branches).
fn keyed_plane_proof(rows_bytes: &[Vec<u8>], row_indices: &[usize], key: [u8; 32], hash_id: HashId) -> MatrixMerkleProof {
    use pearl_blake3::MerkleTree;

    let flat: Vec<u8> = rows_bytes.iter().flatten().copied().collect();
    let chunk_len = hash_id.chunk_len();
    let tree = MerkleTree::with_chunk_len(&hash_id.pad(&flat), key, chunk_len).unwrap();
    let leaves =
        MerkleTree::compute_leaf_indices_from_rows(row_indices, (rows_bytes.len(), rows_bytes[0].len()), chunk_len).unwrap();
    MatrixMerkleProof {
        proof: tree.get_multileaf_proof(&leaves),
        row_indices: row_indices.to_vec(),
    }
}

/// The bench job for one geometry: mutually incoherent planes under the cheap
/// `Blake3Chunk1024` operand trees, so the largest `k` stays inside the 2^22-byte
/// Blake3 work cap.
pub(super) fn bench_job(m: usize, k: usize) -> (IncompleteBlockHeader, PlainProofV4) {
    let plane = |seed: usize| -> Vec<Vec<u8>> {
        (0..m)
            .map(|i| (0..k).map(|j| bench_plane_byte(seed, i, j)).collect())
            .collect()
    };
    let n_blocks = k / BLOCK_SIZE;
    let rows_pattern = AxisPattern::new(ROWS_DIMS).unwrap();
    let cols_pattern = AxisPattern::new(COLS_DIMS).unwrap();
    let tile = |p: &AxisPattern| -> Vec<usize> { p.tile_offsets().iter().map(|&o| o as usize).collect() };
    let (a_rows, b_rows) = (tile(&rows_pattern), tile(&cols_pattern));
    let (header, ancestor_chain, ancestor_header) = fixture_headers();
    let (key_a, key_b) = fixture_tree_keys(&header, &ancestor_header);

    // Scales: bf16 near-identity codes, the fixture convention.
    let scale_tree = |seed: usize| -> Vec<Vec<u8>> {
        (0..m)
            .map(|i| {
                (0..n_blocks)
                    .flat_map(|b| (0x3f80u16 + ((seed + i + b) % 16) as u16).to_le_bytes())
                    .collect()
            })
            .collect()
    };
    let a_hash_id = HashId::Blake3Chunk1024;
    let b_hash_id = HashId::Blake3Chunk1024;
    let proof = PlainProofV4 {
        job: JobParams {
            ancestor_header,
            common: crate::v4::api::public_params::CommonParams {
                k: k as u32,
                r: 32,
                quant: crate::v4::api::public_params::Quant::Fp8E4M3Prequant,
                device: Device::H100,
            },
            operands: Sides {
                a: OperandParams {
                    num_rows: m as u32,
                    hash_id: a_hash_id,
                    pattern: rows_pattern,
                },
                b: OperandParams {
                    num_rows: m as u32,
                    hash_id: b_hash_id,
                    pattern: cols_pattern,
                },
            },
            moe: None,
        },
        ancestor_chain,
        values: Sides {
            a: keyed_plane_proof(&plane(0), &a_rows, key_a, a_hash_id),
            b: keyed_plane_proof(&plane(7), &b_rows, key_b, b_hash_id),
        },
        scales: Sides {
            a: keyed_plane_proof(&scale_tree(1), &a_rows, key_a, a_hash_id),
            b: keyed_plane_proof(&scale_tree(5), &b_rows, key_b, b_hash_id),
        },
        moe_witness: None,
    };
    (header, proof)
}

/// Linear-interpolated percentile of sorted values.
fn pct(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let k = (n - 1) as f64 * p;
    let f = k.floor() as usize;
    let c = (f + 1).min(n - 1);
    sorted[f] + (sorted[c] - sorted[f]) * (k - f as f64)
}

/// The shared-box report line: percentiles over reps.
fn report(label: &str, times: &[f64]) {
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "BENCH {label}: n={} p10={:.3} p25={:.3} p50={:.3} p75={:.3} p90={:.3} min={:.3} mean={:.3}",
        times.len(),
        pct(&sorted, 0.10),
        pct(&sorted, 0.25),
        pct(&sorted, 0.50),
        pct(&sorted, 0.75),
        pct(&sorted, 0.90),
        sorted[0],
        times.iter().sum::<f64>() / times.len() as f64,
    );
}

/// The geometries must derive: parse + `Fp8Job::derive` only, no proving. Keeps
/// benchmark runs honest (a rejected geometry would silently skip a size).
#[test]
fn bench_geometries_derive() {
    for (i, &(m, k)) in GEOMETRIES.iter().enumerate() {
        let expected_max = [19usize, 20, 21, 22][i];
        let (header, plain) = bench_job(m, k);
        let (_, params) = plain.parse_proof(&header).expect("bench job must parse");
        let job = Fp8Job::derive(&params, &header).expect("bench job must derive");
        let degrees = job.system.degree_bits();
        assert_eq!(
            degrees[..NUM_TABLES].iter().max().copied(),
            Some(expected_max),
            "k={k}: the profile must sweep the ladder's top stops: {degrees:?}"
        );
        println!("k={k} degree_bits={degrees:?}");
    }
}

#[test]
#[ignore]
fn bench_pipeline() {
    let _ = env_logger::builder().format_timestamp(None).try_init();
    let device = Device::H100;
    let reps: usize = std::env::var("BENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    let do_warmup = std::env::var("BENCH_WARMUP").map(|v| v != "0").unwrap_or(true);
    // Comma-separated geometry indices into `GEOMETRIES` (default: all).
    let geoms: Vec<usize> = std::env::var("BENCH_GEOMS")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| (0..GEOMETRIES.len()).collect());
    println!("BENCH reps={reps} warmup={do_warmup} geoms={geoms:?}");

    // ---- one-time, job-independent prover setup (LUT oracle + wrapper circuits) ----
    let mut setup_timing = TimingTree::new("setup", log::Level::Info);
    let t = Instant::now();
    let mut prover = Fp8Prover::setup_with_timing(device, &mut setup_timing).expect("fp8 setup");
    println!("BENCH setup_seconds={:.3}", t.elapsed().as_secs_f64());
    setup_timing.filter(Duration::from_millis(100)).print();

    // ---- optional one-time warmup (the cold/warm A/B lever) ----
    if do_warmup {
        let timing = TimingTree::new("warmup", log::Level::Info);
        let t = Instant::now();
        prover.warmup_prove(device).expect("warmup prove of the sample job");
        println!("BENCH warmup_seconds={:.3}", t.elapsed().as_secs_f64());
        timing.filter(Duration::from_millis(100)).print();
    }
    let setup = prover.setups.get(&device).expect("setup was prebuilt");

    // One universal verifier setup per device; built lazily from the first proof.
    let mut verifier: Option<Fp8Verifier> = None;

    for &gi in &geoms {
        let &(m, k) = &GEOMETRIES[gi];
        let (header, plain) = bench_job(m, k);

        // Profile (untimed): the job must parse and derive before it can prove.
        let profile = match plain
            .parse_proof(&header)
            .and_then(|(_, params)| Fp8Job::derive(&params, &header))
        {
            Ok(job) => job,
            Err(e) => {
                eprintln!("BENCH k={k} rejected: {e:#}");
                continue;
            }
        };
        println!("BENCH k={k} degree_bits={:?}", profile.system.degree_bits());

        // One untimed warm-up rep: page in the oracle trees, fault in the
        // allocator, fill any per-geometry caches. Its timing tree is the
        // phase split deliverable (the box is shared; phases of one rep only).
        let mut timing = TimingTree::new(&format!("bench-k{k}-warmrep"), log::Level::Info);
        let warm = prove_batch_statement(&header, &plain, setup, &mut timing)
            .unwrap_or_else(|e| panic!("BENCH k={k} layer-0 warm-up failed: {e:#}"));
        drop(warm);
        timing.filter(Duration::from_millis(100)).print();

        // ---- measured reps ----
        let mut layer0_times = Vec::with_capacity(reps);
        let mut wrap_times = Vec::with_capacity(reps);
        let mut last_timing = TimingTree::default();
        let mut last_batch: Option<(PublicParams, Fp8Job, BatchStarkProofWithPublicInputs<F, C, D>, _)> = None;
        for rep in 0..reps {
            let mut timing = TimingTree::new(&format!("bench-k{k}-rep{rep}"), log::Level::Info);
            let t = Instant::now();
            let (public, job, batch_proof) = match prove_batch_statement(&header, &plain, setup, &mut timing) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("BENCH k={k} layer-0 failed: {e:#}");
                    break;
                }
            };
            let layer0 = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let digest = public.digest(&header);
            let wrapped = match setup.circuits.prove(&job.system, &batch_proof, digest, &mut timing) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("BENCH k={k} wrap failed: {e:#}");
                    break;
                }
            };
            let wrap = t.elapsed().as_secs_f64();

            println!(
                "BENCH k={k} rep={rep} layer0_seconds={layer0:.3} wrap_seconds={wrap:.3} total_prove_seconds={:.3}",
                layer0 + wrap
            );
            layer0_times.push(layer0);
            wrap_times.push(wrap);
            last_timing = timing;
            last_batch = Some((public, job, batch_proof, wrapped));
        }

        if layer0_times.is_empty() {
            continue;
        }
        report(&format!("k={k} layer0"), &layer0_times);
        report(&format!("k={k} wrap"), &wrap_times);

        // Phase split of the final measured rep (unfiltered: every label counts).
        println!("BENCH k={k} final-rep phase split:");
        last_timing.print();

        // Correctness guard (untimed): the last wrapped proof must verify against a
        // verifier-side setup generated from the public data alone.
        let (public, job, _batch_proof, wrapped) = last_batch.expect("at least one rep");
        let public_data = public.to_bytes();
        let proof_data = match compact_proof_data(&job.system, &wrapped) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("BENCH k={k} compact encode failed: {e:#}");
                continue;
            }
        };
        let verifier = match &verifier {
            Some(v) => v.clone(),
            None => {
                let statement = decode_statement(&public_data).expect("public data must decode");
                let v = Fp8Verifier::generate(&statement, &header, &mut TimingTree::default()).expect("verifier-side setup");
                verifier = Some(v.clone());
                println!("BENCH verifier_setup_done");
                v
            }
        };
        verifier
            .verify_block(&header, &plain.ancestor_chain, &public_data, &proof_data)
            .expect("the wrapped bench proof must verify");
        println!("BENCH k={k} verified=true");
    }
}
