//! Embedded FP8 ZK verifier cache ([`crate::api::fp8::zk::Fp8VerifierCache`] wire bytes).
//!
//! Generate the gitignored `fp8_cache.bin` with `task build:zk-cache` or:
//! `cargo run --release --no-default-features --bin build_cache src/api/fp8/fp8_cache.bin`.
//! This also updates `lut_caps.bin`; cached circuits must use the matching cap.
//! Verification only looks up cached setups.

#[cfg(feature = "embedded_cache")]
pub const CACHE_DATA: &[u8] = include_bytes!("fp8_cache.bin");

#[cfg(not(feature = "embedded_cache"))]
pub const CACHE_DATA: &[u8] = &[];
