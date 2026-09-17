//! TamedStark — jackpot check 3 (tamed products): one row per tile cell, importing the
//! cell's replay-magnitude dominator (Matmul) and noise stds (Scale), proving a tamed
//! certificate on every cell not counted untamed, and gating the untamed count against the
//! `TAME_LIMIT` public input. See [`stark`] for the full mathematical description.

pub mod columns;
pub mod ctl;
pub mod stark;
