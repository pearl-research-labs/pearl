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
//! cache. Regenerating the fp8 cache also rewrites the committed
//! LUT caps file (`lut_caps.bin`, next to the cache) — the cached circuits
//! bake the cap, so the two files move together.

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
        let caps_path = std::path::Path::new(&fp8_path).with_file_name("lut_caps.bin");
        let (caps_bytes, cache_bytes) = build_fp8_caches()?;
        std::fs::write(&caps_path, &caps_bytes).with_context(|| format!("writing {}", caps_path.display()))?;
        println!("wrote {} ({} bytes)", caps_path.display(), caps_bytes.len());
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

/// Builds the committed fp8 artifacts in one consistent pass: the LUT cap,
/// derived fresh from the tables ([`zk_pow::api::fp8::lut_caps`] — not from
/// the possibly stale embedded file), and the verifier cache — one *universal*
/// setup (the degree profile and geometry are public inputs, so a single
/// entry covers every envelope-legal job), compiled from the canonical statement
/// ([`zk_pow::api::fp8::zk::sample_dense_statement`]) against that same cap.
fn build_fp8_caches() -> Result<(Vec<u8>, Vec<u8>)> {
    use zk_pow::api::fp8::lut_caps::{cap_from_file_bytes, derive_lut_caps_bytes};
    use zk_pow::api::fp8::zk::{Fp8Verifier, Fp8VerifierCache, sample_dense_statement};

    let caps_bytes = derive_lut_caps_bytes()?;
    println!("derived the LUT cap");

    let mut cache = Fp8VerifierCache::default();
    let params = sample_dense_statement()?;
    let lut_cap = cap_from_file_bytes(&caps_bytes);
    let mut timing = plonky2::util::timing::TimingTree::default();
    let verifier = Fp8Verifier::generate_with_lut_cap(&params, params.ancestor_header(), lut_cap, &mut timing)?;
    cache.insert(&params, verifier);
    println!("compiled the fp8 verifier setup");

    ensure!(!cache.is_empty(), "fp8 verifier cache came out empty");
    Ok((caps_bytes, cache.to_bytes()?))
}
