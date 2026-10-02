//! Matmul row logic used by the Blake3 membership program. The matmul STARK
//! chip lives in the frozen `v1`/`v2` clones.

mod logic;

pub use logic::MatmulLogic;
