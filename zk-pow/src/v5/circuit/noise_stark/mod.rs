//! FP16 noise-line normalization AIR: proves, in-AIR, the per-line normalization arithmetic that
//! the plaintext ground truth [`crate::v5::api::noise`]'s `normalize_line` performs — the
//! signed byte decode, the sum of squares, the integer `isqrt`, the BF16 scale derivation
//! (`f32_to_bf16` of the integer norm, then `bf16_div` of the fixed `8192` numerator), and the
//! per-entry `bf16_mul` + FP16 cast.
//!
//! The `rank` raw XOF bytes enter as witness **input** columns; binding them to the
//! keyed-BLAKE3 XOF is a separate later stage (the [`ctl`]'s byte channel is the hook). All BF16
//! roundings are proved with ties-to-even quarter-ulp *bracket* gadgets over the shared
//! `RANGE16`/`PAIR128`/`POW2D` committed LUTs — no new table is added. See [`stark`] for the
//! constraint groups and the documented soundness envelope.
//!
//! The FP16 analogue of the BF16-scale half of [`crate::v4::circuit::scale_stark`], scoped to a
//! single noise line. Status: the AIR and its constraint/LUT tests are complete; wiring the byte
//! and LUT channels into the batch driver is a later increment (NoiseStark is a CTL party —
//! `requires_ctls` — so the batch driver is the only supported proving path).

pub mod columns;
pub mod ctl;
pub mod stark;

pub use columns::{NUM_NOISE_COLUMNS, NUM_NOISE_PUBLIC_INPUTS, NOISE_COL_MAP, NoiseColumnsView};
pub use stark::{NoiseProgram, NoiseStark};
