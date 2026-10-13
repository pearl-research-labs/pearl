//! V4 proofs: the FP8 scheme (certificate version 4, `PlainFp8`).
//!
//! "v4" and "fp8" name the same thing. Type names keep the `Fp8` prefix
//! (`Fp8Prover`, `Fp8VerifierCache`, ...); module paths use `v4`.
//!
//! - [`api`]: the plain FP8 verifier, the miner witness ([`api::plain_proof::PlainProofV4`]),
//!   the public statement, and the ZK prove/verify entry points ([`api::zk`]).
//! - [`circuit`]: the multi-STARK FP8 proving system and its recursive wrapper.
//!
//! May import from older versions (`v1`, `v2`), never from newer ones or from [`crate::ffi`].

pub mod api;
pub mod circuit;

/// The block certificate version that carries v4 proofs.
pub const CERT_VERSION: u32 = 4;
