//! FP16 per-row scale-derivation AIR: proves, in-AIR, the per-row norm + scale arithmetic of the
//! plaintext ground truth [`crate::v5::api::quantization`]'s `row_norms` + `derive_row_scales`
//! for one operand row of `k` FP16 values:
//!
//! ```text
//! sumsq = Σ (fp16_to_f32(x))^2                       // sequential f32 RNE sum of squares
//! l2    = round_l2_to_grid(f32_to_bf16(sqrt(sumsq/k)))
//! linf  = f32_to_bf16(max|x|)
//! l2    = bf16_max(l2, floor);  linf = bf16_max(linf, floor)       // floor = f32_to_bf16(2^-32)
//! noised_bound = bf16_fma(dr, l2, linf)             // dr = f32_to_bf16(DELTA*sqrt(r))
//! alpha = bf16_div(MAX_FP16_bf16, noised_bound)      // MAX_FP16_bf16 = 0x4780 = 2^16
//! beta  = bf16_mul(bf16_mul(alpha, l2), dos)         // dos = f32_to_bf16(DELTA*sqrt(r)/N^2)
//! ```
//!
//! `r` is the fixed noise rank 32, so `dr` and `dos` are compile-time constants (asserted in
//! [`stark::RowScaleProgram::new`], mirroring [`crate::v5::circuit::noise_stark`]'s fixed
//! numerator). The output columns `l2`, `linf`, `alpha`, `beta` (bf16 codes) bind to
//! `noisy_quant_stark` (alpha/beta) and to the operand commitment (raw row codes) via the
//! parameterized CTL hooks in [`ctl`]; this module does NOT wire the batch (it is not registered in
//! [`crate::v5::circuit::ctl`]), exactly like the other FP16 sub-STARKs built so far.
//!
//! All BF16/f32 roundings are proved with ties-to-even quarter-ulp *bracket* gadgets over the
//! shared FP16-batch LUTs (`FP16DECODE`, `RANGE16`, `FP16POW2`, `WIDTH32`) — no new table and no
//! grinding slack. See [`stark`] for the constraint groups and the documented soundness envelope.

pub mod columns;
pub mod ctl;
pub mod stark;

pub use columns::{
    NUM_ROW_SCALE_COLUMNS, NUM_ROW_SCALE_KNOWN_COLUMNS, NUM_ROW_SCALE_PUBLIC_INPUTS, ROW_SCALE_COL_MAP,
    RowScaleColumnsView,
};
pub use stark::{RowScaleProgram, RowScaleStark};
