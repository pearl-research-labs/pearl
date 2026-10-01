//! Generates the verifier caches embedded by the `embedded_cache` feature
//! (`v2::circuit::embedded_cache`, `v1::embedded_cache`,
//! `api::fp8::embedded_cache`), as invoked by CI and the Taskfile's
//! `build:zk-cache` task. Run WITHOUT that feature (the caches are being
//! produced, not consumed), from `zk-pow/`:
//!
//!   cargo run --release --no-default-features --bin build_cache
//!
//! With no arguments (the invocation the frozen `embedded_cache.rs` comments
//! reference) every cache is rebuilt at its canonical gitignored path:
//! `src/api/fp8/fp8_cache.bin`, `src/v2/circuit/v2_cache.bin`,
//! `src/v1/v1_cache.bin`. Explicit paths override the defaults:
//!
//!   cargo run --release --no-default-features --bin build_cache \
//!       src/api/fp8/fp8_cache.bin src/v2/circuit/v2_cache.bin src/v1/v1_cache.bin
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
        args.len() <= 3,
        "usage: build_cache [<fp8_cache.bin> [<v2_cache.bin> [<v1_cache.bin>]]] \
         (`-` skips a cache; no arguments rebuilds all three at their canonical paths)"
    );
    if args.is_empty() {
        args = vec![
            "src/api/fp8/fp8_cache.bin".into(),
            "src/v2/circuit/v2_cache.bin".into(),
            "src/v1/v1_cache.bin".into(),
        ];
    }
    let mut args = args.into_iter();
    let skippable = |path: Option<String>| path.filter(|p| p != "-");
    let fp8_path = skippable(args.next());
    let v2_path = skippable(args.next());
    let v1_path = skippable(args.next());

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

    Ok(())
}

/// Builds the sole FP8 setup artifact: one *universal* verifier per device. Each
/// setup derives its cap fresh from the LUT tables before baking it into the wrapper
/// circuit; the degree profile and geometry remain public inputs.
fn build_fp8_cache() -> Result<Vec<u8>> {
    use zk_pow::api::fp8::public_params::Device;
    use zk_pow::api::fp8::zk::{Fp8Verifier, Fp8VerifierCache, sample_dense_statement_for_device};

    let mut cache = Fp8VerifierCache::default();
    for device in [Device::H100, Device::B200] {
        let params = sample_dense_statement_for_device(device)?;
        let mut timing = plonky2::util::timing::TimingTree::default();
        let verifier = Fp8Verifier::generate(&params, &params.ancestor_header().incomplete, &mut timing)?;
        cache.insert(device, verifier);
        println!("compiled the {device:?} fp8 verifier setup");
    }

    ensure!(
        cache.contains_all_devices(),
        "fp8 verifier cache must contain exactly the H100 and B200 setups"
    );
    cache.to_bytes()
}
