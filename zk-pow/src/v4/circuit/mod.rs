//! ZK FP8: the multi-STARK proving system for prequant FP8 proof-of-work.
//!
//! Six main tables connected by cross-table lookups, all lookups (range checks included)
//! targeting the precommitted consensus LUT oracle — seventeen (H100) or nineteen (B200)
//! more tables of the same batch, one AIR per logical LUT. Each table's `stark.rs` module
//! docs carry its full mathematical statement and integer soundness bounds:
//!
//! 1. Blake3Stark — commitment/lottery hashing: [`blake3_stark`].
//! 2. InputQuantStark — strip decode, canonical norms and policy scores, the binary32
//!    noising fold and the final FP8 operand codes: [`input_quant_stark`].
//! 3. ScaleStark — per-row norm/scale chain, liveness budgets, signal-RMS floors and
//!    B200's Gaussian row gate: [`scale_stark`].
//! 4. Device-specific Matmul — H100 WGMMA ([`matmul_h100`]) or B200 tcgen05
//!    ([`matmul_b200_stark`]): the committed device's FP8 accumulation, cell results, and
//!    unpredictable-summand certificates.
//! 5. XorFoldStark — lottery extractor folds and the skip-total gate: [`xor_fold_stark`].
//! 6. CoefficientStark — each block's binary32 `alpha*scale` coefficient:
//!    [`coefficient_stark`].
//! 7. The device's `LutStark`s: [`luts`], batch tables 6.. (17 on H100, 19 on B200, so a
//!    batch has 23 or 25 AIRs).
//!
//! [`ctl`] carries the table indices and assembles every channel (seven main channels, the
//! coefficient channel, and one per LUT). Each table keeps its halves in its own `ctl`
//! submodule. [`luts`] owns the LUT descriptors, column layouts, AIRs, witness
//! multiplicities, and setup-time precommitment; [`driver`] assembles the per-table class (a)
//! columns the batch verifier recomputes and is the batch prover/verifier: it fixes the
//! device-specific canonical table order and runs `starky`'s batched multi-STARK argument
//! over one FRI instance. [`wrapper`] is the two-stage recursive wrapper (the batch verifier
//! encoded in a plonky2 circuit, then a zero-knowledge wrap) producing the constant-size
//! published proof. `consistency` (test-only) builds the shared end-to-end fixture: one
//! verifier-parsed job, every trace, every channel balanced. [`unpredictability`] is the
//! canonical integer mirror of the census skip rule and consensus budget the AIRs enforce,
//! and defines B200's Gaussian kernels for the row security gate. `fp32` documents the
//! shared signed alignment, single RNE, and round-to-odd proxy. Policy scores depend on
//! clean decoded inputs, while Matmul consumes the device's noised output bytes, including
//! B200's conditional downscale rewrite.
//!
//! The scheme boundary is prequant FP8: int8 values, bf16 scales per eight-element block,
//! and E4M3 operands. The program bridge rejects any commitment other than the required
//! four-plane prequant shape. The committed device fixes the arithmetic and verifier key.

pub mod blake3_stark;
pub mod chip;
pub mod circuit_utils;
pub mod coefficient_stark;
pub(crate) mod columns_view;
pub mod config;
#[cfg(test)]
pub(crate) mod consistency;
pub mod ctl;
pub mod driver;
pub(crate) mod fp32;
pub mod input_quant_stark;
pub mod luts;
pub mod matmul_b200_stark;
pub mod matmul_h100;
pub mod pearl_program;
pub mod scale_stark;
pub mod unpredictability;
pub mod utils;
pub mod wrapper;
pub mod xor_fold_stark;

pub(crate) mod trace_padding;
