//! Embedded FP16 ZK wrapper verifier cache
//! ([`crate::v5::circuit::verifier_cache::Fp16VerifierCache`] wire bytes).
//!
//! Generated offline by: `cargo run --release --no-default-features --bin build_cache`
//! (the `fp16_cache.bin` trailing path). The blob is a gitignored build artifact; when
//! the `embedded_cache` feature is off (e.g. while `build_cache` is *producing* it) the
//! cache is empty, and a loader treats an empty blob as an empty cache.

/// The embedded cache binary data (only when the `embedded_cache` feature is enabled).
#[cfg(feature = "embedded_cache")]
pub const CACHE_DATA: &[u8] = include_bytes!("fp16_cache.bin");

/// Empty cache when the `embedded_cache` feature is disabled.
#[cfg(not(feature = "embedded_cache"))]
pub const CACHE_DATA: &[u8] = &[];
