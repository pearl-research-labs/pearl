//! FP16 accumulation-hardness proof-of-useful-work scheme (A100 / `sm_80`).
//!
//! An alternative Pearl PoUW instantiation whose unit of work is FP16 matrix
//! multiplication on NVIDIA A100 tensor cores. Hardness comes from the
//! nonlinearity of the device's per-product accumulation truncation rather than
//! from coarse quantization, so recovered products carry FP16-level accuracy.
//! See `docs/fp16_scheme/` for the specification.
//!
//! This module provides the full plaintext path: the FP16 `Dtype` ([`dtype`]),
//! the bit-exact A100 accumulation model ([`accumulate`]), the noisy
//! quantization ([`quantization`]), the unpredictable-accumulation-steps jackpot
//! policy ([`policy`]), the tile parameters ([`params`]), and the plaintext tile
//! verifier ([`verify`]). The commitment wire codec/FFI and the ZK circuit are
//! built on top of these (see the implementation staging notes).

pub mod accumulate;
pub mod commitment;
pub mod dtype;
pub mod embedded_cache;
pub mod noise;
pub mod params;
pub mod plain_proof;
pub mod policy;
pub mod quantization;
pub mod verify;
pub mod zk;
pub mod zk_cert;
