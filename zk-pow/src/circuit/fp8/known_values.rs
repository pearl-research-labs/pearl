//! Verifier-known columns for the FP8 batch.
//!
//! Each main trace begins with columns derived from public geometry, schedules
//! and noise seeds. They are committed with the trace; verification recomputes
//! their openings at `zeta` and `g*zeta`. Here `zeta` is the Fiat–Shamir evaluation
//! challenge and `g` is that table's trace-domain generator; multiplication by
//! `g` selects the next-row opening.
//!
//! This equality binds schedule flags and public noise to the statement even
//! when the AIR has no separate equations for those columns. Static lookup-table
//! columns are bound by the setup commitment.
//!
//! [`fp8_known_columns`] assembles each program's `known_values` in canonical
//! table order. Its [`BatchKnownColumns`] digest is initially unset: proving
//! and verification bind the statement digest before Fiat-Shamir. That digest
//! hashes the public statement, not these column buffers.

use plonky2::field::polynomial::PolynomialValues;
use plonky2::hash::hash_types::{HashOut, RichField};
use primitive_types::U256;
use starky::batch_verifier::BatchKnownColumns;

use super::blake3_stark::columns::NUM_BLAKE3_KNOWN_COLUMNS;
use super::ctl::{NUM_ALL_TABLES, NUM_TABLES};
use super::input_quant_stark::columns::NUM_INPUT_QUANT_KNOWN_COLUMNS;
use super::matmul_b200_stark::columns::NUM_MATMUL_B200_KNOWN_COLUMNS;
use super::scale_stark::columns::NUM_SCALE_KNOWN_COLUMNS;
use super::tamed_stark::columns::NUM_TAMED_KNOWN_COLUMNS;
use super::xor_fold_stark::columns::NUM_XOR_FOLD_KNOWN_COLUMNS;
use crate::api::primitives::Hash256;

/// Leading known-column count in `super::ctl::Table` order.
/// LUTs have no known columns: their static values are preprocessed and their
/// multiplicities belong to the proof.
pub const KNOWN_COLUMNS_PER_TABLE: [usize; NUM_TABLES] = [
    NUM_BLAKE3_KNOWN_COLUMNS,
    NUM_INPUT_QUANT_KNOWN_COLUMNS,
    NUM_SCALE_KNOWN_COLUMNS,
    NUM_MATMUL_B200_KNOWN_COLUMNS,
    NUM_XOR_FOLD_KNOWN_COLUMNS,
    NUM_TAMED_KNOWN_COLUMNS,
];

/// Reduce a 32-byte hash into four Goldilocks elements (little-endian integer mod `p^4`).
pub(crate) fn hash256_to_hash_out<F: RichField>(hash: Hash256) -> HashOut<F> {
    let field_order = U256::from(F::ORDER);
    let mut remaining_hash = U256::from_little_endian(&hash);
    let mut elements = [F::ZERO; 4];
    for element in elements.iter_mut() {
        *element = F::from_canonical_u64((remaining_hash % field_order).as_u64());
        remaining_hash /= field_order;
    }
    HashOut { elements }
}

/// Pack main-table values into [`BatchKnownColumns`], appending empty entries
/// for LUTs. Proving and verification must set the statement digest before use.
/// Panics on wrong column counts, unequal heights or non-power-of-two heights.
pub fn fp8_known_columns<F: RichField>(values_per_table: [Vec<PolynomialValues<F>>; NUM_TABLES]) -> BatchKnownColumns<F> {
    for (t, values) in values_per_table.iter().enumerate() {
        assert_eq!(
            values.len(),
            KNOWN_COLUMNS_PER_TABLE[t],
            "table {t}: wrong known-column count (tables mis-ordered?)"
        );
        for col in values {
            assert_eq!(col.len(), values[0].len(), "table {t}: ragged known columns");
        }
        assert!(values[0].len().is_power_of_two(), "table {t}: height not a power of two");
    }
    let mut columns_per_table: Vec<Vec<usize>> = KNOWN_COLUMNS_PER_TABLE.iter().map(|&n| (0..n).collect()).collect();
    columns_per_table.resize(NUM_ALL_TABLES, Vec::new());
    let mut values_per_table: Vec<Vec<PolynomialValues<F>>> = values_per_table.into();
    values_per_table.resize(NUM_ALL_TABLES, Vec::new());
    BatchKnownColumns {
        digest: None,
        columns_per_table,
        values_per_table,
    }
}
