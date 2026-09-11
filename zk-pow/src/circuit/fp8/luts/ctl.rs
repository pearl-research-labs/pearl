//! CTL halves for committed LUTs and the key/value metadata used to check their consumers.

use plonky2::field::types::Field;
use starky::cross_table_lookup::{TableIdx, TableWithColumns};
use starky::lookup::{Column, Filter};

use super::LutTable;
use super::columns::lut_slot_layout;

/// One lookup of a STARK trace into a committed LUT: the in-trace key/value column algebra and
/// row filter. [`crate::circuit::fp8::ctl::lut_cross_table_lookups`] turns the inventories into real lookup
/// arguments — one [`starky::cross_table_lookup::CrossTableLookup`] per table, this instance on its looking side.
#[derive(Clone, Debug)]
pub struct LutLookup<F: Field> {
    pub table: LutTable,
    /// Key component expressions (one `Column` per key tuple component).
    pub keys: Vec<Column<F>>,
    /// Value-binding expressions, in the table's value-column order (empty for pure range
    /// checks, whose statement is the key's domain membership).
    pub values: Vec<Column<F>>,
    /// Row filter (degree <= 2), default = every row.
    pub filter: Filter<F>,
}

impl<F: Field> LutLookup<F> {
    /// An unfiltered 16-bit range check of a column expression.
    pub fn rc16(key: Column<F>) -> Self {
        Self {
            table: LutTable::Range16,
            keys: vec![key],
            values: vec![],
            filter: Filter::default(),
        }
    }

    /// A filtered 16-bit range check of a column expression.
    pub fn rc16_filtered(key: Column<F>, filter: Filter<F>) -> Self {
        Self {
            table: LutTable::Range16,
            keys: vec![key],
            values: vec![],
            filter,
        }
    }
}

/// The looked half of one LUT slot of the batch table at `table_idx`: the slot's
/// `(key + offset, values...)` tuple over the LUT AIR's own trace, filtered by the slot's
/// multiplicity column — the filter value *is* the row's multiplicity in the channel
/// (degree 1, within the max filter degree 2).
pub fn ctl_looked_lut_slot<F: Field>(table_idx: TableIdx, table: LutTable, slot: usize) -> TableWithColumns<F> {
    let layout = lut_slot_layout(table, slot);
    TableWithColumns::new(
        table_idx,
        layout.looked_columns(),
        Filter::from_column(Column::single(layout.multiplicity_column)),
    )
}
