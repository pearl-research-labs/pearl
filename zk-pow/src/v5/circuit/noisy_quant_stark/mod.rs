//! FP16 fused noisy-quantization AIR: proves, in-AIR, two of the three per-element f32 roundings of
//! the plaintext ground truth [`crate::v5::api::quantization::noisy_quantize`]:
//!
//! * **G1** — the pre-FMA f32 multiply `t = RNE_f32(bf * N_ij)`.
//! * **G3** — the clamp + FP16 cast `out = f32_to_fp16(clamp(noised, -MAX, MAX))` (the clamp is
//!   subsumed by `f32_to_fp16`'s own saturation), covering all four branches: saturate, normal,
//!   subnormal and zero FP16 outputs.
//!
//! The single-rounding f32 FMA that connects them (`noised = fma(af, X, t)` — a signed add with
//! cancellation) is group **G2**, proved by the sibling AIR
//! [`crate::v5::circuit::noisy_quant_fma_stark`]; its `ctl_fma_pairing_looking` exposes the same
//! tuple [`ctl::ctl_fma_hook_looking`] does, so the batch binds `t` (G1) and `noised` (G3) to G2's
//! proven FMA — `noised` is a proven value, not a free witness. See [`stark`] for the constraint
//! groups and the documented soundness envelope.
//!
//! Both roundings are proved with ties-to-even quarter-ulp *bracket* gadgets (the binade-bottom
//! `BOTTOM` correction closing every power-of-two rounding) over the shared `RANGE16`/`FP16POW2`
//! committed LUTs — no new table is added. NoisyQuantStark is a CTL party (`requires_ctls`), so the
//! batch driver is the only supported proving path; wiring it into the batch is a later increment.

pub mod columns;
pub mod ctl;
pub mod stark;

pub use columns::{NUM_NOISY_QUANT_COLUMNS, NUM_NOISY_QUANT_PUBLIC_INPUTS, NOISY_QUANT_COL_MAP, NoisyQuantColumnsView};
pub use stark::{NoisyQuantProgram, NoisyQuantStark};
