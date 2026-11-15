//! For each assigned f32 result word, proves
//! `fold_out = rotl32(low32(fold_state_in * 0x9E3779B1 + cell_word), 13)`.
//! The hexadecimal multiplier and 13-bit rotation are fixed protocol mixing constants.
//! Sixteen independent lanes produce the sixteen words of Blake3's jackpot message block.

pub mod columns;
pub mod ctl;
pub mod stark;

pub use columns::{NUM_XOR_FOLD_COLUMNS, NUM_XOR_FOLD_PUBLIC_INPUTS, XOR_FOLD_COL_MAP, XorFoldColumnsView};
pub use stark::{XorFoldProgram, XorFoldStark};
