//! ZK FP16 (A100) circuit — Stage 3 of the FP16 proof-of-useful-work scheme.
//!
//! This parallels `crate::v4::circuit` but proves the FP16 / A100 scheme whose
//! plaintext ground truth lives in [`crate::v5::api`]. The batched-FRI [`driver::Fp16System`]
//! batches, under one `batch_prove`/`batch_verify` with a recursive wrapper ([`wrapper`]): the
//! device matmul AIR ([`matmul_a100_stark`]), the rho/breakpoint policy AIR ([`policy_stark`]),
//! the lottery mixer ([`xor_fold_stark`]), the forked BLAKE3 engine ([`blake3_fp16_stark`]) driving
//! the operand-commitment + jackpot program ([`blake3_commit`]) and the noise-line keyed-XOF program
//! ([`noise_blake3`]), the seed-derived noise normalization ([`noise_stark`]), the per-row scale
//! derivation ([`row_scale_stark`]) and fused noisy quantization ([`noisy_quant_stark`],
//! [`noisy_quant_fma_stark`]), plus the committed LUTs — all tied together by the cross-table lookup
//! set ([`ctl`]).
//!
//! The AIRs reuse the fp8 circuit's shared, scheme-neutral plumbing (`circuit::utils` evaluators,
//! `circuit::fp8::columns_view`, the committed-LUT machinery of `circuit::fp8::luts`); the BLAKE3
//! engine is **forked** ([`blake3_fp16_stark`]) so FP8's active consensus engine stays untouched.
//!
//! **Binding status.** The proof now binds, end-to-end and tested, the committed statement:
//! committed operand roots (`HASH_A`/`HASH_B`, bit-exact with `commit_operand`) → seed-derived noise
//! (`noise_seeds` over those roots) → noised operands (`Q(α·raw + β·E@F)`) → matmul (with cross-cell
//! row/column sharing) → policy gate → lottery tile → `HASH_JACKPOT` → statement digest → native
//! difficulty. Operand provenance, the output/ticket, and the anti-grind noise are all bound.
//!
//! **Header binding CLOSED.** The header-bound gateways — [`driver::Fp16System::verify_with_headers`]
//! for the batch proof and [`wrapper::verify_wrapped_proof_with_headers`] for the published wrapped
//! proof — derive the opening keys (`keyA`/`keyB`), the `p` encodings, the noise seeds, and the
//! lottery `jackpot_key` from the block header + public job params (bit-exact with the plaintext
//! certificate), pin the proof's corresponding public inputs to them, derive `statement_digest` from
//! the proven `HASH_JACKPOT`, and run the native difficulty check. So a verified proof is bound to
//! the specific block header exactly as the plaintext certificate is; nothing feeding the keys/seeds/
//! jackpot key is caller-trusted. The only inherent boundary (shared with the plaintext FFI) is that
//! the caller supplies the header + `nbits` and window-authenticates `job.ancestor_header` first.
//!
//! **Remaining work (deployment, not soundness).** Making this proof the *wired* consensus path is a
//! deployment integration step: a wrapped-proof FFI entry, the Go/wire routing, and the miner
//! switching from plaintext-certificate assembly to proof generation. Until that lands the wired
//! consensus path remains the plaintext certificate ([`crate::v5::api::verify`]).

pub mod blake3_commit;
pub mod blake3_fp16_stark;
pub mod ctl;
pub mod driver;
pub mod matmul_a100_stark;
pub mod noise_blake3;
pub mod noise_stark;
pub mod noisy_quant_fma_stark;
pub mod noisy_quant_stark;
pub mod policy_stark;
pub mod row_scale_stark;
pub mod verifier_cache;
pub mod wrapper;
pub mod xor_fold_stark;

#[cfg(test)]
mod consistency;
