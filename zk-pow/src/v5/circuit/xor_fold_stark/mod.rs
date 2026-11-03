//! FP16 lottery mixer: for each assigned f32 result word, proves
//! `fold_out = rotl32(low32(fold_state_in * 0x9E3779B1 + cell_word), 13)`.
//! The hexadecimal multiplier and 13-bit rotation are fixed protocol mixing constants (shared
//! with the plaintext extractor [`crate::v4::api::utils::xor_fold_extract`]). Sixteen independent
//! lanes produce the sixteen words of Blake3's jackpot message block.
//!
//! The FP16 analogue of [`crate::v4::circuit::xor_fold_stark`], without the FP8 consolidated
//! skip-gate (FP16 scores its jackpot in [`crate::v5::circuit::policy_stark`]).
//!
//! Status: the AIR and its constraint/LUT tests are complete; wiring the cell-results and
//! lottery-words CTL channels into the batch driver is the next increment (it needs the Blake3
//! commitment/jackpot table as the lottery-words counterparty — see
//! `docs/fp16_scheme/zk_binding_design.md`).

pub mod columns;
pub mod ctl;
pub mod stark;

pub use columns::{NUM_XOR_FOLD_COLUMNS, NUM_XOR_FOLD_PUBLIC_INPUTS, XOR_FOLD_COL_MAP, XorFoldColumnsView};
pub use stark::{XorFoldProgram, XorFoldStark};
