//! ZK FP8: the multi-STARK proving system for prequant FP8 proof-of-work.
//!
//! Six main tables connected by cross-table lookups, all lookups (range checks included)
//! targeting the precommitted consensus LUT oracle — sixteen more tables of the same batch,
//! one AIR per logical LUT. Each table's `stark.rs` module docs carry its full mathematical
//! description:
//!
//! 1. Blake3Stark — commitment/lottery hashing: [`blake3_stark`].
//! 2. InputQuantStark — strip decode + fp8 quantization: [`input_quant_stark`].
//! 3. ScaleStark — per-row norm/scale chain: [`scale_stark`].
//! 4. MatmulB200Stark — B200 tcgen05 window-accumulation emulation: [`matmul_b200_stark`].
//! 5. XorFoldStark — lottery extractor folds: [`xor_fold_stark`].
//! 6. TamedStark — jackpot checks 3+4 policy censuses: [`tamed_stark`].
//! 7. The sixteen `LutStark`s: [`luts`], batch tables 6..22.
//!
//! [`ctl`] carries the table indices, the shared CTL/LUT descriptor types and the channel
//! assembly (eight main channels + one per LUT); the main tables
//! keep their halves and LUT inventories in their own `ctl` submodules. [`luts`] generates
//! the committed LUT tables, their per-AIR layout, their CTL channels and the setup-time
//! precommitment; [`known_values`] assembles the per-table class (a) columns the batch
//! verifier recomputes. [`driver`] is the batch prover/verifier: it sorts the twenty-two
//! tables by height, renumbers the CTLs, and runs `starky`'s batched multi-STARK argument
//! over one FRI instance. [`wrapper`] is the two-stage recursive wrapper (the batch verifier
//! encoded in a plonky2 circuit, then a zero-knowledge wrap) producing the constant-size
//! published proof. `consistency` (test-only) builds the shared end-to-end fixture: one
//! verifier-parsed job, twenty-two traces, every channel balanced. [`unpredictability`] is
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
pub mod known_values;
pub mod luts;
pub mod matmul_b200_stark;
pub mod scale_stark;
pub mod tamed_stark;
pub mod unpredictability;
pub mod wrapper;
pub mod xor_fold_stark;
