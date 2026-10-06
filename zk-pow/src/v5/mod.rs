//! V5 proofs: the FP16 accumulation-hardness scheme (certificate version 5, `PlainFp16`).
//!
//! "v5" and "fp16" name the same thing. Type names keep the `Fp16` prefix
//! (`Fp16System`, `Fp16VerifierCache`, ...); module paths use `v5`.
//!
//! - [`api`]: the plaintext verifier / miner witness and the header-bound ZK
//!   consensus certificate ([`api::zk_cert`]).
//! - [`circuit`]: the batched multi-STARK FP16 proving system and its recursive wrapper.
//!
//! Reuses the FP8/V4 machinery it is layered on (`crate::v4::api::{layout,
//! transcript, primitives, proof_utils}`, `crate::v4::circuit::{luts, blake3_stark,
//! utils, ...}`). May import from older versions (`v1`, `v2`, `v4`), never from
//! [`crate::ffi`].

pub mod api;
pub mod circuit;

/// The block certificate version that carries v5 proofs.
pub const CERT_VERSION: u32 = 5;
