//! Checks committed-LUT lookups and accumulates their per-proof multiplicities.

use std::collections::BTreeMap;

use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::{Field, PrimeField64};

use super::LutTable;
use super::columns::{lut_height, num_slots, slot_height};
use starky::lookup::Column;

use super::ctl::LutLookup;
use super::stark::generate;

/// The per-proof half of the committed oracle: for every table, one count column per slot —
/// `counts[device LUT position][slot][row]`. Filled by
/// [`LutChecker::check_trace`] (or [`Self::add`]) and turned into trace columns by
/// [`Self::table_columns`] / [`super::stark::lut_trace`]. Honest counts are far below the field order
/// (instances x trace height), so the field encoding is exact.
#[derive(Clone, Debug)]
pub struct LutMultiplicities {
    tables: Vec<LutTable>,
    counts: Vec<Vec<Vec<u64>>>,
}

impl LutMultiplicities {
    pub fn new(tables: &[LutTable]) -> Self {
        let counts = tables
            .iter()
            .map(|&t| vec![vec![0u64; lut_height(t)]; num_slots(t)])
            .collect();
        Self {
            tables: tables.to_vec(),
            counts,
        }
    }

    /// The device-family position of `table`.
    fn table_position(&self, table: LutTable) -> usize {
        self.tables
            .iter()
            .position(|&t| t == table)
            .unwrap_or_else(|| panic!("{table:?} is not committed for this device"))
    }

    /// Resolves one looked key tuple to its `(slot, row)`. `Err` when the tuple falls outside
    /// the table's committed domain — for an honest trace that is a bug in the trace or the
    /// descriptor (several tables use the key domain itself as a range proof). XOR8 selects
    /// its slot by the tag it binds as its last value.
    pub fn resolve(table: LutTable, keys: &[u64], values: &[u64]) -> Result<(usize, usize), String> {
        let fold = |width: u32| -> Result<(usize, usize), String> {
            let (slot, row) = ((keys[0] >> width) as usize, (keys[0] & ((1 << width) - 1)) as usize);
            if slot >= num_slots(table) {
                return Err(format!("{table:?} key {} addresses nonexistent slot {slot}", keys[0]));
            }
            Ok((slot, row))
        };
        let single_slot = |row: u64| -> Result<(usize, usize), String> {
            if (row as usize) < slot_height(table) {
                Ok((0, row as usize))
            } else {
                Err(format!("{table:?} key {row} out of domain [0, {})", slot_height(table)))
            }
        };
        let expected_keys = match table {
            LutTable::Bytes2 | LutTable::Pair128 | LutTable::Xor8 => 2,
            _ => 1,
        };
        if keys.len() != expected_keys {
            return Err(format!(
                "{table:?} lookup carries {} key components, wants {expected_keys}",
                keys.len()
            ));
        }
        match table {
            // The tuple components must be in range individually — a component overflow must
            // not alias into another row's valid tuple.
            LutTable::Bytes2 => {
                if keys[0] >= 256 || keys[1] >= 256 {
                    return Err(format!("BYTES2 tuple ({}, {}) has an oversized byte", keys[0], keys[1]));
                }
                Ok((0, (keys[0] + (keys[1] << 8)) as usize))
            }
            LutTable::Pair128 => {
                if keys[0] >= 128 || keys[1] >= 128 {
                    return Err(format!("PAIR128 tuple ({}, {}) has an oversized half", keys[0], keys[1]));
                }
                Ok((0, (keys[0] + (keys[1] << 7)) as usize))
            }
            LutTable::Xor8 => {
                if keys[0] >= 256 || keys[1] >= 256 {
                    return Err(format!("XOR8 tuple ({}, {}) has an oversized byte", keys[0], keys[1]));
                }
                let slot = match values {
                    [_, _, tag] if (*tag as usize) < num_slots(table) => *tag as usize,
                    _ => return Err(format!("XOR8 values {values:?} carry no valid slot tag")),
                };
                Ok((slot, (keys[0] + (keys[1] << 8)) as usize))
            }
            // WIDTH32's stored key is the shifted ramp [1, 32]: row = key - 1. Key 0 has no
            // row — this is the "no zero-width claim" soundness point.
            LutTable::Width32 => {
                if (1..=slot_height(table) as u64).contains(&keys[0]) {
                    Ok((0, keys[0] as usize - 1))
                } else {
                    Err(format!("WIDTH32 key {} out of domain [1, 32]", keys[0]))
                }
            }
            LutTable::RneRnd => fold(17),
            LutTable::ProdAlign15 | LutTable::B200Align | LutTable::WidthNorm => fold(16),
            LutTable::Pow2G => match keys[0] {
                0..=63 => Ok((0, keys[0] as usize)),
                128..=191 => Ok((1, keys[0] as usize - 128)),
                192..=217 => Ok((2, keys[0] as usize - 192)),
                257..=305 => Ok((3, keys[0] as usize - 257)),
                key => Err(format!("POW2G key {key} outside carry, gap, and normalization domains")),
            },
            _ => single_slot(keys[0]),
        }
    }

    /// Records `mult` lookups of `keys` (bound to `values`) into `table`.
    pub fn add(&mut self, table: LutTable, keys: &[u64], values: &[u64], mult: u64) -> Result<(), String> {
        let (slot, row) = Self::resolve(table, keys, values)?;
        let table_position = self.table_position(table);
        self.counts[table_position][slot][row] += mult;
        Ok(())
    }

    /// One table's multiplicity columns, in slot order — the AIR's trailing columns.
    pub fn table_columns<F: Field>(&self, table: LutTable) -> Vec<Vec<F>> {
        self.counts[self.table_position(table)]
            .iter()
            .map(|col| col.iter().map(|&c| F::from_canonical_u64(c)).collect())
            .collect()
    }

    /// Adds `src`'s counts into `self` (every table `src` covers must be committed
    /// here). Exact: u64 addition commutes.
    pub fn merge(&mut self, src: &Self) {
        for (src_pos, &table) in src.tables.iter().enumerate() {
            let dst_position = self.table_position(table);
            for (dst_slot, src_slot) in self.counts[dst_position].iter_mut().zip(&src.counts[src_pos]) {
                for (dst_row, src_row) in dst_slot.iter_mut().zip(src_slot.iter()) {
                    *dst_row += src_row;
                }
            }
        }
    }

    /// Total lookups recorded into `table` (all slots, all rows).
    pub fn table_total(&self, table: LutTable) -> u64 {
        self.counts[self.table_position(table)].iter().flatten().sum()
    }
}

/// Debug/test-side oracle checker: walks LUT instance inventories over honest traces, checks
/// every instance is *served* by the committed tables (key resolves in-domain, bound values
/// equal the generated outputs), and accumulates the per-slot multiplicities — the committed-LUT
/// analogue of `starky::cross_table_lookup::debug_utils::check_ctls`, with per-instance error
/// reporting the multiset check cannot give.
pub struct LutChecker<F: PrimeField64> {
    pub multiplicities: LutMultiplicities,
    _phantom: core::marker::PhantomData<F>,
}

impl<F: PrimeField64> LutChecker<F> {
    pub fn new(tables: &[LutTable]) -> Self {
        Self {
            multiplicities: LutMultiplicities::new(tables),
            _phantom: core::marker::PhantomData,
        }
    }

    /// The production walk: accumulates multiplicities only. Skips the serving
    /// validation (value evaluation and generated-output comparison) — it never
    /// changes the committed counts, and the debug/test-side
    /// [`Self::check_trace`] (plus the consistency tests) owns that check.
    /// Key resolution errors still fail the walk: an out-of-domain key has no
    /// counter to increment.
    pub fn count_trace(
        &mut self,
        lookups: &[LutLookup<F>],
        trace: &[PolynomialValues<F>],
        public_inputs: &[F],
        ctx: &str,
    ) -> Result<(), String> {
        self.walk(lookups, trace, public_inputs, ctx, false)
    }

    /// Checks every instance of `lookups` on every row of `trace` (column-major poly values),
    /// reading filter values as multiplicities (the logup numerator semantics; honest filters
    /// are 0/1). `ctx` tags error messages. `public_inputs` is the host table's public-input
    /// vector (Scale's T3 gate keys read the `DEAD_LIMIT` slots).
    ///
    /// Rows are processed in parallel chunks, each accumulating into its own multiplicity
    /// counters; the chunks merge exactly (u64 addition commutes), and the first error in row
    /// order — identical to the serial walk — is reported. Per-chunk state stays minimal:
    /// counters are allocated only for the tables this inventory touches, and the generated
    /// table outputs are materialized once and shared immutably across chunks.
    pub fn check_trace(
        &mut self,
        lookups: &[LutLookup<F>],
        trace: &[PolynomialValues<F>],
        public_inputs: &[F],
        ctx: &str,
    ) -> Result<(), String> {
        self.walk(lookups, trace, public_inputs, ctx, true)
    }

    fn walk(
        &mut self,
        lookups: &[LutLookup<F>],
        trace: &[PolynomialValues<F>],
        public_inputs: &[F],
        ctx: &str,
        validate: bool,
    ) -> Result<(), String> {
        use plonky2_maybe_rayon::*;

        let num_rows = trace[0].len();
        if num_rows == 0 {
            return Ok(());
        }
        // Chunk boundaries: fixed-size chunks of >= 2^15 rows, capped at 512
        // chunks — enough to over-subscribe any pool (with the walk tasks of
        // all five tables sharing it) while keeping per-chunk counter
        // allocation amortized. The big tables (2^21..2^22 rows) dominate;
        // smaller ones fall back to one chunk.
        let chunk_rows = (num_rows / 512).max(1 << 15).min(num_rows);
        let num_chunks = num_rows.div_ceil(chunk_rows);
        debug_assert!(chunk_rows * num_chunks >= num_rows);

        // The distinct tables this inventory touches — chunk counters are
        // allocated for these only, not for the full committed set.
        let inventory_tables: Vec<LutTable> = {
            let mut t: Vec<LutTable> = lookups.iter().map(|l| l.table).collect();
            t.sort_unstable();
            t.dedup();
            t
        };
        // Compiled per-instance walk plans: each instance's filter, keys and
        // (when walked) values as flat term lists, hoisted out of the row
        // loop so the per-row work is pure arithmetic on column slices — no
        // per-row Vec allocations, no re-dispatch through the generic
        // `eval_table` iterators.
        let plans: Vec<WalkPlan<F>> = lookups.iter().map(WalkPlan::compile).collect();
        // Generated outputs, materialized once and shared across chunks
        // (the generator is deterministic per (table, slot)) — only the
        // validating walk needs them.
        let outputs = if validate {
            let mut m = BTreeMap::new();
            for &table in &inventory_tables {
                for slot in 0..num_slots(table) {
                    m.insert((table, slot), generate::<F>(table, slot));
                }
            }
            Some(m)
        } else {
            None
        };

        let chunk_results: Vec<Result<LutMultiplicities, String>> = (0..num_chunks)
            .into_par_iter()
            .map(|chunk| {
                let row_lo = chunk * chunk_rows;
                let row_hi = (row_lo + chunk_rows).min(num_rows);
                // Counters for this chunk: only the inventory's tables.
                let mut counts = LutMultiplicities::new(&inventory_tables);
                for (li, plan) in plans.iter().enumerate() {
                    let table_position = counts.table_position(plan.table);
                    let chunk_counts = &mut counts.counts[table_position];
                    for row in row_lo..row_hi {
                        let err = |e: String| format!("{ctx} lookup {li} ({:?}) row {row}: {e}", plan.table);
                        let mult = plan.eval_filter(trace, row, public_inputs).to_canonical_u64();
                        if mult == 0 {
                            continue;
                        }
                        if mult > 1 << 20 {
                            return Err(err(format!("implausible filter/multiplicity value {mult}")));
                        }
                        // Xor8's slot lives in its bound values (the tag), so the
                        // counting walk still evaluates them; every other table
                        // resolves on keys alone.
                        let words = plan.eval_instance(trace, row, public_inputs, validate || plan.walk_values);
                        let keys = &words.keys[..words.num_keys];
                        let values = &words.values[..words.num_values];
                        let (slot, table_row) = LutMultiplicities::resolve(plan.table, keys, values).map_err(err)?;
                        if let Some(outputs) = &outputs {
                            let outputs = &outputs[&(plan.table, slot)];
                            // BYTES2/PAIR128 bind only their key tuple, already checked by resolution.
                            let expected_arity = match plan.table {
                                LutTable::Bytes2 | LutTable::Pair128 => 0,
                                _ => outputs.len(),
                            };
                            let bound = &words.values[..words.num_values];
                            if bound.len() != expected_arity {
                                return Err(err(format!("binds {} values, table returns {expected_arity}", bound.len())));
                            }
                            for (i, &got) in bound.iter().enumerate() {
                                let want = outputs[i][table_row].to_canonical_u64();
                                if got != want {
                                    return Err(err(format!(
                                        "value {i} = {got:?} differs from {want:?} (slot {slot}, table row {table_row})"
                                    )));
                                }
                            }
                        }
                        chunk_counts[slot][table_row] += mult;
                    }
                }
                Ok(counts)
            })
            .collect();

        // Merge in row order; surface the first error exactly as the serial walk would.
        let mut first_err = None;
        for res in &chunk_results {
            if let Err(e) = res {
                first_err.get_or_insert(e.clone());
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }
        // Reduction merge: sum all chunks' counters slot-vector-wise. Every
        // (table, slot) pair is disjoint across chunks, so the sum over
        // chunks parallelizes over slot rows — one task per (table, slot)
        // vector, each folding its chunks in row order.
        let merged: Vec<_> = chunk_results
            .into_iter()
            .map(|res| res.expect("errors surfaced above"))
            .collect();
        for &table in inventory_tables.iter() {
            let dst_position = self.multiplicities.table_position(table);
            // Gather this table's per-chunk slot vectors once.
            let per_chunk: Vec<&Vec<Vec<u64>>> = merged
                .iter()
                .map(|counts| {
                    let src_pos = counts.table_position(table);
                    &counts.counts[src_pos]
                })
                .collect();
            self.multiplicities.counts[dst_position]
                .par_iter_mut()
                .enumerate()
                .for_each(|(slot, dst_slot)| {
                    for src in &per_chunk {
                        for (dst_row, src_row) in dst_slot.iter_mut().zip(&src[slot]) {
                            *dst_row += src_row;
                        }
                    }
                });
        }
        Ok(())
    }
}

/// A `LutLookup` compiled into flat term lists for the walk: the filter as a
/// sum of column products and linear columns, each key/value expression as a
/// borrowed flat term list, and the per-row results kept in stack buffers
/// (every inventory binds at most 2 keys / 5 values).
struct WalkPlan<'a, F: Field> {
    table: LutTable,
    /// Filter products `(col1, col2)` and linear columns.
    filter_products: Vec<(CompiledColumn<'a, F>, CompiledColumn<'a, F>)>,
    filter_constants: Vec<CompiledColumn<'a, F>>,
    keys: Vec<CompiledColumn<'a, F>>,
    values: Vec<CompiledColumn<'a, F>>,
    /// Xor8's slot tag rides in the bound values; every other table resolves
    /// on keys alone.
    walk_values: bool,
}

/// One `Column`'s flat term lists, borrowed from the inventory:
/// `(linear, next_row_linear, public_input_scaled, public_input_offsets, constant)`.
struct CompiledColumn<'a, F: Field>(&'a [(usize, F)], &'a [(usize, F)], &'a [(usize, usize)], &'a [usize], F);

impl<'a, F: Field> Clone for CompiledColumn<'a, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, F: Field> Copy for CompiledColumn<'a, F> {}

impl<'a, F: Field> CompiledColumn<'a, F> {
    fn of(col: &'a Column<F>) -> Self {
        let (linear, next_row, scaled, offsets, constant) = col.parts();
        Self(linear, next_row, scaled, offsets, constant)
    }

    #[inline]
    fn eval(&self, trace: &[PolynomialValues<F>], row: usize, public_inputs: &[F]) -> F {
        let mut acc = self.4;
        for &(c, f) in self.0 {
            acc += trace[c].values[row] * f;
        }
        for &(c, f) in self.1 {
            acc += trace[c].values[(row + 1) % trace[c].values.len()] * f;
        }
        for &(c, s) in self.2 {
            acc += trace[c].values[row] * public_inputs[s];
        }
        for &s in self.3 {
            acc += public_inputs[s];
        }
        acc
    }
}

/// Per-row instance results, stack-allocated: keys and values as fixed arrays
/// plus their live lengths.
struct InstanceWords {
    keys: [u64; 2],
    num_keys: usize,
    /// The largest binding is ProdAlign15's five value columns; eight keeps
    /// headroom for future tables.
    values: [u64; 8],
    num_values: usize,
}

impl<'a, F: PrimeField64> WalkPlan<'a, F> {
    fn compile(lookup: &'a LutLookup<F>) -> Self {
        let (products, constants) = lookup.filter.parts();
        debug_assert!(lookup.keys.len() <= 2, "stack key buffer overflows");
        debug_assert!(lookup.values.len() <= 8, "stack value buffer overflows");
        Self {
            table: lookup.table,
            filter_products: products
                .iter()
                .map(|(c1, c2)| (CompiledColumn::of(c1), CompiledColumn::of(c2)))
                .collect(),
            filter_constants: constants.iter().map(CompiledColumn::of).collect(),
            keys: lookup.keys.iter().map(CompiledColumn::of).collect(),
            values: lookup.values.iter().map(CompiledColumn::of).collect(),
            walk_values: lookup.table == LutTable::Xor8,
        }
    }

    #[inline]
    fn eval_filter(&self, trace: &[PolynomialValues<F>], row: usize, public_inputs: &[F]) -> F {
        let mut acc = F::ZERO;
        for (c1, c2) in &self.filter_products {
            acc += c1.eval(trace, row, public_inputs) * c2.eval(trace, row, public_inputs);
        }
        for c in &self.filter_constants {
            acc += c.eval(trace, row, public_inputs);
        }
        acc
    }

    #[inline]
    fn eval_instance(&self, trace: &[PolynomialValues<F>], row: usize, public_inputs: &[F], walk_values: bool) -> InstanceWords {
        let mut words = InstanceWords {
            keys: [0; 2],
            num_keys: self.keys.len(),
            values: [0; 8],
            num_values: if walk_values { self.values.len() } else { 0 },
        };
        for (k, col) in self.keys.iter().enumerate() {
            words.keys[k] = col.eval(trace, row, public_inputs).to_canonical_u64();
        }
        if walk_values {
            for (v, col) in self.values.iter().enumerate() {
                words.values[v] = col.eval(trace, row, public_inputs).to_canonical_u64();
            }
        }
        words
    }
}
