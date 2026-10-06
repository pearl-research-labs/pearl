//! Generates the verifier caches embedded by the `embedded_cache` feature
//! (`v2::circuit::embedded_cache`, `v1::embedded_cache`,
//! `v4::api::embedded_cache`), as invoked by CI and the Taskfile's
//! `build:zk-cache` task. Run WITHOUT that feature (the caches are being
//! produced, not consumed), from `zk-pow/`:
//!
//!   cargo run --release --no-default-features --bin build_cache
//!
//! With no arguments (the invocation the frozen `embedded_cache.rs` comments
//! reference) every cache is rebuilt at its canonical gitignored path:
//! `src/v4/api/fp8_cache.bin`, `src/v2/circuit/v2_cache.bin`,
//! `src/v1/v1_cache.bin`. Explicit paths override the defaults:
//!
//!   cargo run --release --no-default-features --bin build_cache \
//!       src/v4/api/fp8_cache.bin src/v2/circuit/v2_cache.bin src/v1/v1_cache.bin
//!
//! The trailing paths are optional: with fewer arguments only the leading
//! caches are written (fp8, then v2, then v1), and a path of `-` skips that
//! cache. The FP8 cache derivation bakes each device's freshly derived LUT cap
//! into its wrapper circuit; no separate LUT-cap artifact is written.

use anyhow::{Context, Result, ensure};

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() <= 4,
        "usage: build_cache [<fp8_cache.bin> [<v2_cache.bin> [<v1_cache.bin>]]] \
         (`-` skips a cache; no arguments rebuilds all three at their canonical paths)"
    );
    if args.is_empty() {
        args = vec![
            "src/v4/api/fp8_cache.bin".into(),
            "src/v2/circuit/v2_cache.bin".into(),
            "src/v1/v1_cache.bin".into(),
        ];
    }
    let mut args = args.into_iter();
    let skippable = |path: Option<String>| path.filter(|p| p != "-");
    let fp8_path = skippable(args.next());
    let v2_path = skippable(args.next());
    let v1_path = skippable(args.next());
    let fp16_path = skippable(args.next());

    if let Some(fp8_path) = fp8_path {
        let cache_bytes = build_fp8_cache()?;
        std::fs::write(&fp8_path, &cache_bytes).with_context(|| format!("writing {fp8_path}"))?;
        println!("wrote {fp8_path} ({} bytes)", cache_bytes.len());
    }

    if let Some(v2_path) = v2_path {
        use zk_pow::v2::circuit::pearl_circuit::{PearlRecursion, RecursionCircuit};
        let mut cache = zk_pow::v2::circuit::circuit_utils::CircuitCache::default();
        PearlRecursion::fill_verifier_cache(&mut cache);
        // fill_verifier_cache logs-and-continues on per-params failures; an
        // empty cache means nothing compiled, so fail loudly instead of
        // embedding a useless blob.
        ensure!(
            !cache.verifier_circuits_1.is_empty() && !cache.verifier_circuits_2.is_empty(),
            "V2 verifier cache came out empty"
        );
        let bytes = cache.to_bytes()?;
        std::fs::write(&v2_path, &bytes).with_context(|| format!("writing {v2_path}"))?;
        println!("wrote {v2_path} ({} bytes)", bytes.len());
    }

    if let Some(v1_path) = v1_path {
        use zk_pow::v1::circuit::pearl_circuit::{PearlRecursion, RecursionCircuit};
        let mut cache = zk_pow::v1::circuit::circuit_utils::CircuitCache::default();
        PearlRecursion::fill_verifier_cache(&mut cache);
        ensure!(
            !cache.verifier_circuits_1.is_empty() && !cache.verifier_circuits_2.is_empty(),
            "V1 verifier cache came out empty"
        );
        let bytes = cache.to_bytes()?;
        std::fs::write(&v1_path, &bytes).with_context(|| format!("writing {v1_path}"))?;
        println!("wrote {v1_path} ({} bytes)", bytes.len());
    }

    if let Some(fp16_path) = fp16_path {
        let bytes = build_fp16_cache()?;
        std::fs::write(&fp16_path, &bytes).with_context(|| format!("writing {fp16_path}"))?;
        println!("wrote {fp16_path} ({} bytes)", bytes.len());
    }

    Ok(())
}

/// Builds the sole FP8 setup artifact: one *universal* verifier per device. Each
/// setup derives its cap fresh from the LUT tables before baking it into the wrapper
/// circuit; the degree profile and geometry remain public inputs.
fn build_fp8_cache() -> Result<Vec<u8>> {
    use zk_pow::v4::api::primitives::IncompleteBlockHeader;
    use zk_pow::v4::api::public_params::Device;
    use zk_pow::v4::api::zk::{Fp8Verifier, Fp8VerifierCache, sample_dense_statement_for_device};

    let mut cache = Fp8VerifierCache::default();
    for device in Device::ALL {
        let params = sample_dense_statement_for_device(device)?;
        let mut timing = plonky2::util::timing::TimingTree::default();
        let verifier = Fp8Verifier::generate(&params, &IncompleteBlockHeader::zero(), &mut timing)?;
        cache.insert(device, verifier);
        println!("compiled the {device:?} fp8 verifier setup");
    }

    ensure!(
        cache.contains_all_devices(),
        "fp8 verifier cache must contain exactly the H100 and B200 setups"
    );
    cache.to_bytes()
}
/// Builds the FP16 wrapper verifier cache: one compiled stage-2 verifier circuit per
/// *reachable degree profile*. The FP16 wrapper is per-degree-profile (the universal
/// variant is deferred), but every table height snaps to
/// [`zk_pow::v5::circuit::driver::FP16_REACHABLE_DEGREE_BITS`], so the profile space
/// is small and fully enumerable — all consensus-legal geometries sharing a profile
/// share the one circuit, and the node verifier LOADS the setup for a proof's geometry
/// instead of rebuilding it (closing the verify-time rebuild / DoS).
///
/// We sweep a representative grid of consensus-legal geometries (the
/// [`Fp16Params`](zk_pow::v5::api::params::Fp16Params) envelope: `h >= 4`, `w >= 16`,
/// `h*w in [256, 2048]`, `k % 8 == 0`, `k*(h+w) <= 2^22`) across both operand hash ids,
/// dedup by the `Fp16System::degree_bits` profile, and compile one wrapper per distinct
/// profile. Each compile is expensive (minutes; the largest profiles are multi-GB), so
/// this is an offline one-time build.
fn build_fp16_cache() -> Result<Vec<u8>> {
    use std::collections::BTreeSet;
    use zk_pow::v5::api::params::Fp16Params;
    use zk_pow::v4::api::public_params::HashId;
    use zk_pow::v5::circuit::driver::Fp16System;
    use zk_pow::v5::circuit::verifier_cache::{Fp16Verifier, Fp16VerifierCache, Fp16VerifierKey};
    use zk_pow::v5::circuit::wrapper::{Fp16WrapperCircuits, InnerC, D, F};

    // Representative geometry grid. The reachable degree profiles are a small set
    // (every table snaps to the reachable ladder), so sweeping the envelope's corners
    // and a few interior points over both hash ids finds them all; dedup collapses
    // the rest. `k` spans small .. the per-shape maximum `k*(h+w) <= 2^22`.
    // `FP16_CACHE_SAMPLE=1` restricts the sweep to the single smallest legal tile +
    // one hash id (one profile) — a cheap way to generate a bootstrap blob so the
    // `embedded_cache` crate compiles and the load path can be exercised, WITHOUT the
    // full (heavy, multi-GB) envelope compile. Production omits it.
    let sample = std::env::var("FP16_CACHE_SAMPLE").is_ok();
    let hs: &[usize] = &[4, 8, 16, 32, 45, 64];
    let ws: &[usize] = &[16, 32, 45, 64, 128, 256, 512];
    let k_samples: &[usize] = &[8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536];
    let hash_ids: &[HashId] = &HashId::ALL;

    // A profile is wrapper-legal only if every table height lies on the consensus
    // ladder; off-ladder geometries cannot be wrapped (so the prover can't produce a
    // proof for them either), and are skipped. `FP16_CACHE_DRYRUN` prints the
    // reachable on-ladder profiles without compiling (cheap space-mapping).
    let on_ladder =
        |profile: &[usize]| profile.iter().all(|b| zk_pow::v5::circuit::driver::FP16_REACHABLE_DEGREE_BITS.contains(b));
    let dry_run = std::env::var("FP16_CACHE_DRYRUN").is_ok();

    let mut cache = Fp16VerifierCache::default();
    let mut seen: BTreeSet<Vec<usize>> = BTreeSet::new();
    let mut considered = 0usize;

    for &h in hs {
        for &w in ws {
            for &k in k_samples {
                // Consensus-legality gate (mirror Fp16Params::validate).
                let params = Fp16Params {
                    device: zk_pow::v5::api::params::Fp16Device::A100,
                    h,
                    w,
                    k,
                    r: zk_pow::v5::circuit::driver::FP16_QUANT_R,
                };
                if params.validate().is_err() {
                    continue;
                }
                for &a_hash in hash_ids {
                    for &b_hash in hash_ids {
                        considered += 1;
                        // Fp16System::new is cheap (AIR defs + closed-form snapped heights).
                        // validate() above gates the known realizability constraints; still
                        // skip (not abort) any residual geometry whose construction panics.
                        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let system = Fp16System::<F, D>::new(h, w, k, a_hash, b_hash);
                            let profile: Vec<usize> = system.degree_bits().to_vec();
                            (system, profile)
                        }));
                        let (system, profile) = match built {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        // Skip off-ladder (non-wrapper-legal) geometries; dedup profiles.
                        if !on_ladder(&profile) || !seen.insert(profile.clone()) {
                            continue;
                        }
                        if dry_run {
                            println!("profile {profile:?} (h={h} w={w} k={k} a_hash={a_hash:?} b_hash={b_hash:?})");
                            continue;
                        }
                        let mut timing = plonky2::util::timing::TimingTree::default();
                        let preprocessed = system.preprocessed_data::<InnerC>(&mut timing);
                        let circuits =
                            Fp16WrapperCircuits::build(&system, &preprocessed.cap(), &mut timing)?;
                        let key: Fp16VerifierKey = *system.degree_bits();
                        cache.insert(key, Fp16Verifier::new(circuits.verifier_data()));
                        println!(
                            "compiled fp16 wrapper for profile {profile:?} (h={h} w={w} k={k} \
                             a_hash={a_hash:?} b_hash={b_hash:?}); {} profiles so far",
                            cache.len()
                        );
                        if sample {
                            // Bootstrap mode: one on-ladder profile is enough to prove the
                            // generator + load path; the full envelope is the offline run.
                            println!("fp16 cache: sample mode — built 1 profile, stopping");
                            return cache.to_bytes();
                        }
                    }
                }
            }
        }
    }

    ensure!(!cache.is_empty(), "FP16 verifier cache came out empty");
    println!("fp16 cache: {} distinct profiles from {considered} legal geometries", cache.len());
    cache.to_bytes()
}

