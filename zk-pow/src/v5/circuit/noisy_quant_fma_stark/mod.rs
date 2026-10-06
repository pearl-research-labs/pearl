//! FP16 single-rounding FMA AIR — group G2 of the fused noisy quantization.
//!
//! Proves `noised = fma(af, X, t)` bit-exact vs the Rust `af.mul_add(fp16_to_f32(raw), t)` of
//! [`crate::v5::api::quantization::noisy_quantize`], with `af = bf16_to_f32(alpha)`,
//! `X = fp16_to_f32(raw)`, and `t = RNE_f32(bf*N)` (group G1's output). It is a signed,
//! arbitrarily-aligned add of the exact product `af*X` and `t` with a SINGLE f32
//! round-to-nearest-ties-to-even and possible cancellation — the core of the kernel. The sibling
//! [`crate::v5::circuit::noisy_quant_stark`] proves G1 (`t`) and G3 (the FP16 cast); this AIR's
//! [`ctl::ctl_fma_pairing_looking`] exposes the exact tuple that module's
//! `ctl_fma_hook_looking` does, so the batch binds `t` (G1 output) and `noised` (G3 input) to this
//! AIR's proven FMA in one channel.
//!
//! The windowed-alignment datapath mirrors [`crate::v5::circuit::matmul_a100_stark`] (24-bit
//! normalization, FP16POW2/WIDTH32 alignment, signed sum, far-gap sticky), with the single RNE
//! round proved by the exact quarter-ulp bracket / parity / IS_BOTTOM (normalized-range) / is-zero
//! technique — no quarter-ulp grinding slack. See [`stark`] for the constraint groups and the
//! documented soundness/completeness envelope.

pub mod columns;
pub mod ctl;
pub mod stark;

pub use columns::{FMA_COL_MAP, FmaColumnsView, NUM_FMA_COLUMNS, NUM_FMA_PUBLIC_INPUTS};
pub use stark::{FmaProgram, NoisyQuantFmaStark};
