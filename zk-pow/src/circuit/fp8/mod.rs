//! ZK FP8: the multi-STARK proving system for prequant FP8 proof-of-work.
//!
//! Five main tables connected by cross-table lookups, all lookups (range checks included)
//! targeting the precommitted consensus LUT oracle — fifteen more tables of the same batch,
//! one AIR per logical LUT. Each table's `stark.rs` module docs carry its full mathematical
//! description:
//!
//! 1. Blake3Stark — commitment/lottery hashing: [`blake3_stark`].
//! 2. InputQuantStark — strip decode + fp8 quantization: [`input_quant_stark`].
//! 3. ScaleStark — per-row norm/scale chain: [`scale_stark`].
//! 4. Device-specific Matmul — H100 WGMMA ([`matmul_stark`]) or B200 tcgen05
//!    ([`matmul_b200_stark`]).
//! 5. XorFoldStark — lottery extractor folds: [`xor_fold_stark`].
//! 6. The device's fifteen `LutStark`s: [`luts`], batch tables 5..19.
//!
//! [`ctl`] carries the table indices and assembles every channel (six main channels +
//! one per LUT). Each table keeps its halves in its own `ctl` submodule. [`luts`] owns
//! the LUT descriptors, column layouts, AIRs, witness multiplicities, and setup-time
//! precommitment; [`driver`] assembles the per-table class (a) columns the batch verifier
//! recomputes and is the batch prover/verifier: it fixes the twenty-table
//! canonical order and runs `starky`'s batched multi-STARK argument
//! over one FRI instance. [`wrapper`] is the two-stage recursive wrapper (the batch verifier
//! encoded in a plonky2 circuit, then a zero-knowledge wrap) producing the constant-size
//! published proof. `consistency` (test-only) builds the shared end-to-end fixture: one
//! verifier-parsed job, twenty traces, every channel balanced. [`unpredictability`] is
//! check 4's canonical integer mirror (the skip rule and consensus budget the AIRs enforce).
//!
//! **Scheme boundary.** This system proves exactly one scheme:
//! prequant FP8 (`Quant::Fp8E4M3Prequant`) with int8 values, BF16 block scales, and FP8 E4M3
//! matmul. The bridge from the shared job compiler rejects any program that is
//! not the required four-plane prequant shape.

pub mod blake3_stark;
pub mod circuit_utils;
pub(crate) mod columns_view;
#[cfg(test)]
pub(crate) mod consistency;
pub mod ctl;
pub mod driver;
pub mod input_quant_stark;
pub mod luts;
pub mod matmul_b200_stark;
pub mod matmul_stark;
pub mod scale_stark;
pub mod unpredictability;
pub mod wrapper;
pub mod xor_fold_stark;
