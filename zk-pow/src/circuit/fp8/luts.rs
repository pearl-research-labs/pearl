//! The committed LUT oracle: one AIR per logical table.
//!
//! Each of the sixteen [`LutTable`]s ([`LUT_TABLES`]) is its own table of
//! the batch STARK — a [`LutStark`] at its natural committed height [`lut_height`] (the
//! array orders the batch by descending height, so nothing pays for another table's
//! padding). A LUT AIR has no constraints of its own; its columns split into two classes:
//!
//! - **Precommitted columns** (class (c)): the key column(s) — a plain ramp for tables whose
//!   key domain fills the height, the enumerated key tuple for BYTES2/PAIR128, or a saturated
//!   key `min(i, live - 1)` for the sub-height tables — followed by every stored value column
//!   in slot order ([`lut_precommitted_values`]). Generated once system-wide and committed at
//!   setup by [`lut_preprocessed_data`] into a `BatchStarkPreprocessedData` whose Merkle cap
//!   is a consensus constant. Proofs never recommit these columns — the batch prover copies
//!   the setup commitment and FRI opens it alongside the trace oracles, while constraints and
//!   lookups keep addressing them through the table's ordinary column indices.
//! - **Multiplicity columns** (per-proof): one trace column per slot, counting how often each
//!   row is looked up — [`LutMultiplicities`], job-dependent, committed with the trace like
//!   any online column.
//!
//! The lookup argument is one [`CrossTableLookup`] per table ([`lut_cross_table_lookups`]):
//! the looking side collects every instance of every AIR's inventory (`keys ++ values` over
//! the consumer's own trace), the looked side is the table's slots — a multi-slot looked side
//! for the folded tables — each slot's `(ramp + key offset, stored values...)` tuple filtered
//! by its multiplicity column, whose value *is* the row's multiplicity in the channel (the
//! logup numerator semantics).
//!
//! Values are produced by exhaustively evaluating the canonical Rust mirror each table binds —
//! reusing the starks' own mirror functions wherever one exists, so the tables and the traces
//! cannot drift apart. [`generate`] returns the stored value columns (`columns[j][row]`); a
//! row's key is its index plus the slot's key offset (`2^17 * slot` for RNERND, `2^16 * rel`
//! for B200ALIGN). `RANGE16` stores nothing: its ramp
//! key column *is* the table, and `BYTES2`/`PAIR128` store the enumerated key tuple itself.
//!
//! Padding (sub-height tables only) is always a *repeated valid entry*, never an
//! out-of-domain key: the saturated key column repeats the last live key and the value columns
//! are padded alike, so a malicious nonzero multiplicity on a padding row only re-proves a
//! fact the table already serves.
//!
//! RNERND's slot index is a *signed* subnormal fade cut biased by +7: slot `s` serves
//! `cut = s - 7 in [-7, 18]`, i.e. `KEY_SCALE = -133 - cut`. The negative cuts (slots 0-6)
//! distinguish normal from subnormal cancellation results with `V < 128`, which share a
//! significand but not a scale; slot 0 also serves every `KEY_SCALE >= -126` (all normal)
//! and slot 25 every `KEY_SCALE <= -151` (all zero), so 26 slots cover the full scale range
//! bit-exactly.

use core::marker::PhantomData;
use std::collections::BTreeMap;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::GenericConfig;
use plonky2::util::timing::TimingTree;
use starky::batch_prover::BatchStarkPreprocessedData;
use starky::config::StarkConfig;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use starky::cross_table_lookup::{CrossTableLookup, TableIdx, TableWithColumns};
use starky::evaluation_frame::StarkFrame;
use starky::lookup::{Column, Filter};
use starky::stark::Stark;

use super::ctl::{LUT_TABLES, LutLookup, LutTable, NUM_LUT_TABLES, lut_table_idx};
use super::input_quant_stark::stark::qcast;
use super::matmul_b200_stark::stark::B200Product;
use super::scale_stark::stark::{CODE_448, rnernd_reference};
use super::unpredictability::{log2_fixed, sig_nonzero, sig_width};
use crate::api::fp8::compute::bf16_div;

/// Number of slots of a folded table (1 = unfolded).
pub const fn num_slots(table: LutTable) -> usize {
    match table {
        LutTable::RneRnd => 26,    // one per slot = signed cut + 7, cut in [-7, 18]
        LutTable::B200Align => 72, // one per REL in [0, 71]
        _ => 1,
    }
}

/// Live rows per slot (the key-domain size; [`lut_height`] rounds it up to the AIR height).
pub const fn slot_height(table: LutTable) -> usize {
    match table {
        LutTable::Range16 | LutTable::Bytes2 | LutTable::Qcast | LutTable::Div448 | LutTable::Width16 | LutTable::Log16 => {
            1 << 16
        }
        LutTable::Pair128 => 1 << 14,
        LutTable::XfPow2 => 1 << 11,
        LutTable::Int8Dec => 256,
        LutTable::ExpInfo => 255, // keys [0, 254]; the inf/NaN field 255 has no row
        LutTable::Clamp22 => 601, // keys x + 400, x in [-400, 200]
        LutTable::Pow2D => 20,    // keys [0, 19]; the FMA's gap cap of 19 sets the domain
        LutTable::Pow2Gb => 64,
        LutTable::Width32 => 32, // keys [1, 32] — a shifted ramp, no key 0 (see `generate`)
        LutTable::RneRnd => 1 << 17,
        LutTable::B200Align => 1 << 16,
    }
}

/// XFPOW2's saturating exponent cap (jackpot check 3, TamedStark): the table serves
/// `2^A`, `A = min(max(D, 0), 80)`, `D = 2*(E_CELL - exp(A) - exp(B) + 3949)` — the doubled
/// frame gap with `tau_tame^2 = 2^16` folded in. The compared bound `Y = k * PP^2` is below
/// `2^80`, so the cap never changes the verdict of `2^A <= Y`.
pub const XFPOW2_CAP: u64 = 80;
/// XFPOW2's key zero point: `D = 2*(key - 1024)`. The 2^11 key domain covers every reachable
/// gap: `E_CELL in [121, 172]` on keying rows and sigma exponents in `[1781, 2287]` keep the
/// key inside `[521, 1584]`.
pub const XFPOW2_ZERO_POINT: u64 = 1024;
/// XFPOW2's `2^A` value as base-2^16 limbs (`A <= 80`: 6 limbs, exactly one nonzero).
pub const XFPOW2_LIMBS: usize = 6;

/// Generates one table (or one slot of a folded table): the stored value columns, in the order
/// the consumers' `LutLookup::values` bind them.
pub fn generate<F: Field>(table: LutTable, slot: usize) -> Vec<Vec<F>> {
    assert!(slot < num_slots(table), "slot {slot} out of range for {table:?}");
    let height = slot_height(table);
    let f = F::from_canonical_u64;
    let mut columns: Vec<Vec<F>> = match table {
        LutTable::Range16 => vec![],
        _ => {
            let arity = match table {
                LutTable::Bytes2 | LutTable::Pair128 | LutTable::Width32 | LutTable::Width16 => 2,
                LutTable::Int8Dec | LutTable::RneRnd => 4,
                LutTable::B200Align => 5,
                LutTable::XfPow2 => XFPOW2_LIMBS,
                _ => 1,
            };
            vec![Vec::with_capacity(height); arity]
        }
    };

    for key in 0..height as u64 {
        match table {
            LutTable::Range16 => {}
            // All byte pairs (paired 8-bit range check); a lone byte keys (d, 0).
            LutTable::Bytes2 => {
                columns[0].push(f(key & 0xFF));
                columns[1].push(f(key >> 8));
            }
            // All 7-bit pairs (mantissa fields, ScaleStark's sqrt/l2-linf pairs).
            LutTable::Pair128 => {
                columns[0].push(f(key & 0x7F));
                columns[1].push(f(key >> 7));
            }
            // fp8 quantization of a bf16 code: `f32_to_fp8_e4m3(bf16_to_f32(clamp_±448(x)))`,
            // via InputQuant's mirror. Non-finite keys (exponent field 255) cannot decode; they
            // are unreachable because each consumer range-checks `254 - out_exp`; their rows
            // hold the saturation code.
            LutTable::Qcast => {
                let code = key as u16;
                let out = if code & 0x7F80 == 0x7F80 {
                    (code >> 8) as u8 & 0x80 | 0x7E
                } else {
                    qcast(code)
                };
                columns[0].push(f(u64::from(out)));
            }
            // Exact bf16 decode fields of the int8 value: byte 0 -> +0, else the normalized
            // `(SIGN, EXP, MANTISSA)` with `EXP_IS_ZERO = 0` (every int8 fits 8 significand
            // bits, so the decode is exact; byte 0x80 = -128 -> (1, 134, 0, 0)).
            LutTable::Int8Dec => {
                let v = key as u8 as i8;
                let (sign, exp, man, eiz) = if v == 0 {
                    (0, 0, 0, 1)
                } else {
                    let a = (v as i64).unsigned_abs(); // in [1, 128]
                    let w = 64 - a.leading_zeros() as u64;
                    (u64::from(v < 0), 126 + w, (a << (8 - w)) - 128, 0)
                };
                columns[0].push(f(sign));
                columns[1].push(f(exp));
                columns[2].push(f(man));
                columns[3].push(f(eiz));
            }
            // The alpha definition `RNE_bf16(448 / noised_bound)` via the native `bf16_div`.
            // Error paths (zero/subnormal-tiny denominators, non-finite keys) hold the
            // exponent-255 sentinel the alpha validity RCs reject; unreachable on the live
            // domain (the 2^-32 norm floors keep the noised bound at or above 2^-32).
            LutTable::Div448 => {
                let code = key as u16;
                let sentinel = code & 0x8000 | 0x7F80;
                let out = if code & 0x7F80 == 0x7F80 {
                    sentinel
                } else {
                    bf16_div(CODE_448, code).unwrap_or(sentinel)
                };
                columns[0].push(f(u64::from(out)));
            }
            // Subnormal flag of an exponent field; the [0, 254] key domain is the finiteness
            // proof.
            LutTable::ExpInfo => columns[0].push(f(u64::from(key == 0))),
            // The RNERND slot for fade argument `x = key - 400`: `clamp(x, -7, 18) + 7`.
            LutTable::Clamp22 => columns[0].push(f((key as i64 - 400 + 7).clamp(0, 25) as u64)),
            LutTable::Pow2D => columns[0].push(f(1 << key)),
            // Matmul's carry alignment: the 26-cap floors a far accumulator to zero
            // (`4*GROUP_OUTPUT_SIGNIFICAND < 2^26`), exactly as the window drops it.
            LutTable::Pow2Gb => columns[0].push(f(1 << key.min(26))),
            // Matmul's truncate-toward-zero back-end: W = key + 1 (the stored key column
            // is the shifted ramp [1, 32] — no key 0), value (2^max(W-24,0), 2^max(24-W,0)).
            LutTable::Width32 => {
                let w = key + 1;
                columns[0].push(f(1 << w.saturating_sub(24)));
                columns[1].push(f(1 << 24u64.saturating_sub(w)));
            }
            // Shared bf16 RNE back-end (ScaleStark's mirror; one slot per signed cut). See the
            // module header for the InputQuant cancellation case that also needs the LSB scale.
            LutTable::RneRnd => {
                let (mant, wa, oiz, oez) = rnernd_reference(key, slot as u64);
                columns[0].push(f(mant));
                columns[1].push(f(wa));
                columns[2].push(f(u64::from(oiz)));
                columns[3].push(f(u64::from(oez)));
            }
            // The whole per-lane fp8 product, pre-truncated to the B200 window (Matmul's
            // mirror; slot = REL). `ALIGNED_LANE_TERMS = ±floor(P*2^19 / 2^REL)` toward zero with
            // the sign folded; slots REL >= 27 hold 0 (P < 2^8). `PRODUCT_BIASED_EXPONENT`
            // is the biased stored-exponent sum (sentinel 0 for zero products,
            // slot-independent). `OPERAND_CODES_A/B` echo the key's operand bytes, pinning the
            // looking side's code columns individually. `BINADE` is the product's check-3
            // binade `floor(log2 |prod|) + 139` (0 for zero products, same in every slot).
            LutTable::B200Align => {
                let p = B200Product::new((key & 0xFF) as u8, (key >> 8) as u8);
                let aligned = if p.is_zero {
                    F::ZERO
                } else {
                    let mag = f((p.sig << 19) >> slot.min(63));
                    if p.sign { -mag } else { mag }
                };
                columns[0].push(aligned);
                columns[1].push(f(p.biased_exponent));
                columns[2].push(f(key & 0xFF));
                columns[3].push(f(key >> 8));
                columns[4].push(f(p.biased_binade()));
            }
            // TamedStark's squared-comparison power: key `t` decodes `D = 2*(t - 1024)`, the
            // value is `2^min(max(D,0), 80)` as base-2^16 limbs (exactly one limb nonzero —
            // a power of two).
            LutTable::XfPow2 => {
                let d = 2 * (key as i64 - XFPOW2_ZERO_POINT as i64);
                let a = (d.max(0) as u64).min(XFPOW2_CAP);
                for u in 0..XFPOW2_LIMBS as u64 {
                    columns[u as usize].push(if a / 16 == u { f(1 << (a % 16)) } else { F::ZERO });
                }
            }
            // Jackpot check 4's significand-product width: the bit length of the 16-bit key
            // and its nonzero flag (key 0 holds the (0, 0) sentinel).
            LutTable::Width16 => {
                columns[0].push(f(sig_width(key)));
                columns[1].push(f(sig_nonzero(key)));
            }
            // Jackpot check 4's fixed-point log: `floor(64 * log2 key)` (0 at key 0).
            LutTable::Log16 => columns[0].push(f(log2_fixed(key))),
        }
    }
    columns
}

// ==================================================================================================
// Per-table AIR layout
// ==================================================================================================

/// Committed height of the table's AIR: the slot key domain rounded up to a power of two.
/// Full-height for most tables; EXPINFO (255 -> 256), CLAMP22 (601 -> 1024) and POW2D
/// (20 -> 32) pad by key saturation (module docs).
pub const fn lut_height(table: LutTable) -> usize {
    slot_height(table).next_power_of_two()
}

/// Stored key columns: the enumerated tuple for BYTES2/PAIR128, one ramp or saturated key
/// column otherwise.
const fn num_key_columns(table: LutTable) -> usize {
    match table {
        LutTable::Bytes2 | LutTable::Pair128 => 2,
        _ => 1,
    }
}

/// Stored value columns over all slots (RANGE16 and the key-tuple tables store none).
const fn num_value_columns(table: LutTable) -> usize {
    match table {
        LutTable::Range16 | LutTable::Bytes2 | LutTable::Pair128 => 0,
        LutTable::Qcast
        | LutTable::Div448
        | LutTable::ExpInfo
        | LutTable::Clamp22
        | LutTable::Pow2D
        | LutTable::Pow2Gb
        | LutTable::Log16 => 1,
        LutTable::Width32 | LutTable::Width16 => 2,
        LutTable::Int8Dec => 4,
        LutTable::XfPow2 => XFPOW2_LIMBS,
        LutTable::B200Align => 76, // 72 per-slot aligned + 4 shared
        LutTable::RneRnd => 104,   // 4 per slot
    }
}

/// Number of setup-time committed columns: the keys followed by the stored values.
pub const fn num_precommitted_columns(table: LutTable) -> usize {
    num_key_columns(table) + num_value_columns(table)
}

/// Total column count of the table's AIR: the precommitted block, then one per-proof
/// multiplicity column per slot.
pub const fn lut_num_columns(table: LutTable) -> usize {
    num_precommitted_columns(table) + num_slots(table)
}

/// One slot's position in its AIR: everything the CTL wiring (and the debug checker) needs to
/// serve lookups into `(table, slot)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LutSlotLayout {
    /// The additive constant of the slot's key formula (`2^17 * cut`, `2^16 * shift/slot`;
    /// 0 for unfolded tables). Looking sides fold it into their key expressions.
    pub key_offset: u64,
    /// The key column(s): the leading ramp/saturated key, or the stored key tuple.
    pub key_columns: Vec<usize>,
    /// The slot's stored value columns, in the order consumers' `LutLookup::values` bind them.
    pub value_columns: Vec<usize>,
    /// The slot's per-proof multiplicity column.
    pub multiplicity_column: usize,
}

impl LutSlotLayout {
    /// The slot's looked tuple `keys ++ values` as column expressions over the AIR's trace
    /// (the key column carries the slot's additive offset), ready for the CTL channel.
    pub fn looked_columns<F: Field>(&self) -> Vec<Column<F>> {
        let offset = F::from_canonical_u64(self.key_offset);
        let mut cols: Vec<Column<F>> = self
            .key_columns
            .iter()
            .map(|&c| Column::linear_combination_with_constant([(c, F::ONE)], offset))
            .collect();
        cols.extend(self.value_columns.iter().map(|&c| Column::single(c)));
        cols
    }
}

/// Resolves `(table, slot)` to its columns (a pure function of `num_key_columns` and the
/// slot-major value layout).
pub fn lut_slot_layout(table: LutTable, slot: usize) -> LutSlotLayout {
    assert!(slot < num_slots(table), "slot {slot} out of range for {table:?}");
    let nk = num_key_columns(table);
    let (key_offset, value_columns) = match table {
        LutTable::Range16 | LutTable::Bytes2 | LutTable::Pair128 => (0, vec![]),
        LutTable::Qcast
        | LutTable::Div448
        | LutTable::ExpInfo
        | LutTable::Clamp22
        | LutTable::Pow2D
        | LutTable::Pow2Gb
        | LutTable::Log16 => (0, vec![nk]),
        LutTable::Width32 | LutTable::Width16 => (0, vec![nk, nk + 1]),
        LutTable::Int8Dec => (0, (nk..nk + 4).collect()),
        LutTable::XfPow2 => (0, (nk..nk + XFPOW2_LIMBS).collect()),
        LutTable::B200Align => {
            let shared = nk + num_slots(table);
            (
                (slot as u64) << 16,
                // (ALIGNED_LANE_TERMS[slot], PRODUCT_BIASED_EXPONENT, OPERAND_CODES_A/B, BINADE).
                vec![nk + slot, shared, shared + 1, shared + 2, shared + 3],
            )
        }
        LutTable::RneRnd => ((slot as u64) << 17, (0..4).map(|v| nk + 4 * slot + v).collect()),
    };
    LutSlotLayout {
        key_offset,
        key_columns: (0..nk).collect(),
        value_columns,
        multiplicity_column: num_precommitted_columns(table) + slot,
    }
}

/// Generates the table's full precommitted column block (`columns[c][row]`, the leading
/// [`num_precommitted_columns`] of its trace): the consensus-frozen data behind the setup
/// commitment. Deterministic; padding repeats valid entries (module docs).
pub fn lut_precommitted_values<F: Field>(table: LutTable) -> Vec<Vec<F>> {
    let height = lut_height(table);
    let live = slot_height(table);
    let mut cols: Vec<Vec<F>> = Vec::with_capacity(num_precommitted_columns(table));
    match table {
        // The stored key tuple is the whole precommitted block (live == height: both are
        // powers of two, so there is no padding).
        LutTable::Bytes2 | LutTable::Pair128 => cols.extend(generate::<F>(table, 0)),
        // Sub-height (or exactly-full) small tables: a saturated key column `min(i, live - 1)`
        // and the value columns padded alike — padding repeats the last live row's fact.
        LutTable::Int8Dec | LutTable::ExpInfo | LutTable::Clamp22 | LutTable::Pow2D | LutTable::Pow2Gb => {
            cols.push((0..height).map(|i| F::from_canonical_usize(i.min(live - 1))).collect());
            for mut col in generate::<F>(table, 0) {
                let last = *col.last().unwrap();
                col.resize(height, last);
                cols.push(col);
            }
        }
        // WIDTH32's key domain is [1, 32] — a *shifted* ramp (no key 0 row exists, so no
        // multiplicity can serve a zero width claim); live == height, no padding.
        LutTable::Width32 => {
            cols.push((0..height).map(|i| F::from_canonical_usize(i + 1)).collect());
            cols.extend(generate::<F>(table, 0));
        }
        // Ramp-keyed tables fill their height exactly (live == height, a power of two).
        _ => {
            cols.push((0..height).map(F::from_canonical_usize).collect());
            match table {
                LutTable::Range16 => {} // the ramp is the whole table
                LutTable::Qcast | LutTable::Div448 | LutTable::XfPow2 | LutTable::Width16 | LutTable::Log16 => {
                    cols.extend(generate::<F>(table, 0))
                }
                // Folded: the per-slot aligned column in slot-major order, then the
                // slot-independent columns (product decode facts — computed from the key
                // alone; asserted in tests).
                LutTable::B200Align => {
                    let mut slot0 = generate::<F>(table, 0);
                    let shared = slot0.split_off(1);
                    cols.extend(slot0);
                    for slot in 1..num_slots(table) {
                        let mut columns = generate::<F>(table, slot);
                        columns.truncate(1);
                        cols.extend(columns);
                    }
                    cols.extend(shared);
                }
                // Folded: all four value columns per slot.
                LutTable::RneRnd => {
                    for slot in 0..num_slots(table) {
                        cols.extend(generate::<F>(table, slot));
                    }
                }
                _ => unreachable!("handled above"),
            }
        }
    }
    debug_assert_eq!(cols.len(), num_precommitted_columns(table));
    debug_assert!(cols.iter().all(|c| c.len() == height));
    cols
}

/// Assembles one LUT AIR's full constraint-view trace: the precommitted block followed by the
/// per-slot multiplicity columns (e.g. from [`LutMultiplicities::table_columns`]). Under the
/// batch commitment only the multiplicity tail is committed per proof — the leading block is
/// served by [`lut_preprocessed_data`]'s setup oracle; both halves address identically inside
/// constraints and CTLs.
pub fn lut_trace<F: Field>(table: LutTable, multiplicities: Vec<Vec<F>>) -> Vec<PolynomialValues<F>> {
    assert_eq!(multiplicities.len(), num_slots(table));
    for col in &multiplicities {
        assert_eq!(col.len(), lut_height(table));
    }
    let mut cols = lut_precommitted_values::<F>(table);
    cols.extend(multiplicities);
    cols.into_iter().map(PolynomialValues::new).collect()
}

// ==================================================================================================
// The per-table AIR and its CTL channel
// ==================================================================================================

/// One committed LUT's AIR: a pure lookup table of `WIDTH` columns — the precommitted block,
/// then the per-slot multiplicity columns. It has **no constraints of its own**: the
/// precommitted columns are bound by the consensus setup cap, and the multiplicity columns
/// only by the table's CTL channel balance ([`lut_cross_table_lookups`]).
///
/// `WIDTH` must equal [`lut_num_columns`]`(table)` (asserted by [`Self::new`]; the per-table
/// aliases below pin it) — the `Stark` trait wants the column count as a compile-time constant.
#[derive(Clone, Copy, Debug)]
pub struct LutStark<F, const D: usize, const WIDTH: usize> {
    pub table: LutTable,
    _phantom: PhantomData<F>,
}

impl<F, const D: usize, const WIDTH: usize> LutStark<F, D, WIDTH> {
    pub fn new(table: LutTable) -> Self {
        assert_eq!(WIDTH, lut_num_columns(table), "wrong AIR width for {table:?}");
        Self {
            table,
            _phantom: PhantomData,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize, const WIDTH: usize> Stark<F, D> for LutStark<F, D, WIDTH> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, WIDTH, 0>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;
    type EvaluationFrameTarget = StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, WIDTH, 0>;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        _vars: &Self::EvaluationFrame<FE, P, D2>,
        _yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
    }

    fn eval_ext_circuit(
        &self,
        _builder: &mut CircuitBuilder<F, D>,
        _vars: &Self::EvaluationFrameTarget,
        _yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
    }

    fn constraint_degree(&self) -> usize {
        3
    }

    fn requires_ctls(&self) -> bool {
        true
    }
}

/// The AIRs at their exact widths (the batch instantiates sixteen of these, in [`LUT_TABLES`]
/// order).
pub type RneRndStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::RneRnd) }>;
pub type Range16Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Range16) }>;
pub type Bytes2Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Bytes2) }>;
pub type QcastStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Qcast) }>;
pub type Div448Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Div448) }>;
pub type Pair128Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Pair128) }>;
pub type Clamp22Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Clamp22) }>;
pub type Int8DecStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Int8Dec) }>;
pub type ExpInfoStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::ExpInfo) }>;
pub type Pow2DStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Pow2D) }>;
pub type B200AlignStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::B200Align) }>;
pub type Pow2GbStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Pow2Gb) }>;
pub type Width32Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Width32) }>;
pub type XfPow2Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::XfPow2) }>;
pub type Width16Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Width16) }>;
pub type Log16Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Log16) }>;

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

/// The committed LUT channels: one [`CrossTableLookup`] per table of `tables`
/// ([`LUT_TABLES`] — position `i` is batch table [`lut_table_idx`]`(i)`).
/// `inventories` are the main tables' LUT instance inventories at their batch indices; each
/// channel's looking side collects every instance of its table across all of them (tuple
/// `keys ++ values` over the consumer's trace, the instance's filter), and its looked side is
/// the table's slots — a multi-slot looked side for the folded tables, with looked-side helper
/// columns handled by starky.
///
/// Every table must have at least one looking instance (a consumerless LUT signals a wiring
/// bug); width consistency of every half is asserted by `CrossTableLookup::new`.
pub fn lut_cross_table_lookups<F: Field>(
    tables: &[LutTable; NUM_LUT_TABLES],
    inventories: &[(TableIdx, Vec<LutLookup<F>>)],
) -> Vec<CrossTableLookup<F>> {
    tables
        .iter()
        .enumerate()
        .map(|(i, &table)| {
            let looking: Vec<TableWithColumns<F>> = inventories
                .iter()
                .flat_map(|(table_idx, lookups)| {
                    let table_idx = *table_idx;
                    lookups.iter().filter(|l| l.table == table).map(move |l| {
                        TableWithColumns::new(table_idx, l.keys.iter().chain(&l.values).cloned().collect(), l.filter.clone())
                    })
                })
                .collect();
            assert!(!looking.is_empty(), "{table:?} has no looking instances");
            let looked = (0..num_slots(table))
                .map(|slot| ctl_looked_lut_slot(lut_table_idx(i), table, slot))
                .collect();
            CrossTableLookup::new(looking, looked)
        })
        .collect()
}

// ==================================================================================================
// Per-proof multiplicities and the served-lookup checker
// ==================================================================================================

/// The per-proof half of the committed oracle: for every table, one count column per slot —
/// `counts[LUT_TABLES position][slot][row]`. Filled by
/// [`LutChecker::check_trace`] (or [`Self::add`]) and turned into trace columns by
/// [`Self::table_columns`] / [`lut_trace`]. Honest counts are far below the field order
/// (instances x trace height), so the field encoding is exact.
#[derive(Clone, Debug)]
pub struct LutMultiplicities {
    counts: Vec<Vec<Vec<u64>>>,
}

impl Default for LutMultiplicities {
    fn default() -> Self {
        Self::new()
    }
}

impl LutMultiplicities {
    pub fn new() -> Self {
        let counts = LUT_TABLES
            .iter()
            .map(|&t| vec![vec![0u64; lut_height(t)]; num_slots(t)])
            .collect();
        Self { counts }
    }

    /// The [`LUT_TABLES`] position of `table`.
    fn table_position(table: LutTable) -> usize {
        LUT_TABLES
            .iter()
            .position(|&t| t == table)
            .expect("every LutTable is committed")
    }

    /// Resolves one looked key tuple to its `(slot, row)`. `Err` when the tuple falls outside
    /// the table's committed domain — for an honest trace that is a bug in the trace or the
    /// descriptor (several tables use the key domain itself as a range proof).
    pub fn resolve(table: LutTable, keys: &[u64]) -> Result<(usize, usize), String> {
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
            LutTable::Bytes2 | LutTable::Pair128 => 2,
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
            LutTable::B200Align => fold(16),
            _ => single_slot(keys[0]),
        }
    }

    /// Records `mult` lookups of `keys` into `table`.
    pub fn add(&mut self, table: LutTable, keys: &[u64], mult: u64) -> Result<(), String> {
        let (slot, row) = Self::resolve(table, keys)?;
        self.counts[Self::table_position(table)][slot][row] += mult;
        Ok(())
    }

    /// One table's multiplicity columns, in slot order — the AIR's trailing columns.
    pub fn table_columns<F: Field>(&self, table: LutTable) -> Vec<Vec<F>> {
        self.counts[Self::table_position(table)]
            .iter()
            .map(|col| col.iter().map(|&c| F::from_canonical_u64(c)).collect())
            .collect()
    }

    /// Total lookups recorded into `table` (all slots, all rows).
    pub fn table_total(&self, table: LutTable) -> u64 {
        self.counts[Self::table_position(table)].iter().flatten().sum()
    }
}

/// Debug/test-side oracle checker: walks LUT instance inventories over honest traces, checks
/// every instance is *served* by the committed tables (key resolves in-domain, bound values
/// equal the stored columns), and accumulates the per-slot multiplicities — the committed-LUT
/// analogue of `starky::cross_table_lookup::debug_utils::check_ctls`, with per-instance error
/// reporting the multiset check cannot give.
pub struct LutChecker<F: PrimeField64> {
    pub multiplicities: LutMultiplicities,
    /// Generated stored columns, cached per (table, slot) across inventories.
    cache: BTreeMap<(LutTable, usize), Vec<Vec<F>>>,
}

impl<F: PrimeField64> Default for LutChecker<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField64> LutChecker<F> {
    pub fn new() -> Self {
        Self {
            multiplicities: LutMultiplicities::new(),
            cache: BTreeMap::new(),
        }
    }

    /// Checks every instance of `lookups` on every row of `trace` (column-major poly values),
    /// reading filter values as multiplicities (the logup numerator semantics; honest filters
    /// are 0/1). `ctx` tags error messages. `public_inputs` is the host table's public-input
    /// vector (Scale's T3 gate keys read the `DEAD_LIMIT` slots).
    pub fn check_trace(
        &mut self,
        lookups: &[LutLookup<F>],
        trace: &[PolynomialValues<F>],
        public_inputs: &[F],
        ctx: &str,
    ) -> Result<(), String> {
        let num_rows = trace[0].len();
        for (li, lookup) in lookups.iter().enumerate() {
            for row in 0..num_rows {
                let err = |e: String| format!("{ctx} lookup {li} ({:?}) row {row}: {e}", lookup.table);
                let mult = lookup.filter.eval_table(trace, row, public_inputs).to_canonical_u64();
                if mult == 0 {
                    continue;
                }
                if mult > 1 << 20 {
                    return Err(err(format!("implausible filter/multiplicity value {mult}")));
                }
                let keys: Vec<u64> = lookup
                    .keys
                    .iter()
                    .map(|c| c.eval_table(trace, row, public_inputs).to_canonical_u64())
                    .collect();
                let (slot, table_row) = LutMultiplicities::resolve(lookup.table, &keys).map_err(&err)?;
                let stored = self
                    .cache
                    .entry((lookup.table, slot))
                    .or_insert_with(|| generate::<F>(lookup.table, slot));
                // BYTES2/PAIR128 bind no values (their stored tuple is the key, equal by
                // resolution); everything else binds exactly the stored value columns.
                let expected_arity = match lookup.table {
                    LutTable::Bytes2 | LutTable::Pair128 => 0,
                    _ => stored.len(),
                };
                if lookup.values.len() != expected_arity {
                    return Err(err(format!(
                        "binds {} values, table stores {expected_arity}",
                        lookup.values.len()
                    )));
                }
                for (vi, vcol) in lookup.values.iter().enumerate() {
                    let got = vcol.eval_table(trace, row, public_inputs);
                    let want = stored[vi][table_row];
                    if got != want {
                        return Err(err(format!(
                            "value {vi} = {got:?} differs from the stored {want:?} (slot {slot}, table row {table_row})"
                        )));
                    }
                }
                self.multiplicities.add(lookup.table, &keys, mult).map_err(&err)?;
            }
        }
        Ok(())
    }
}

// ==================================================================================================
// Setup-time precommitment
// ==================================================================================================

/// The `BatchStarkPreprocessedData::new` inputs for a batch of `num_tables` tables in which
/// LUT `i` (of [`LUT_TABLES`]) is table `table_positions[i]`:
/// `(values_per_table, columns_per_table)` — each LUT's precommitted columns at its position
/// (empty elsewhere). Positions must be strictly increasing (LUT heights descend in
/// [`LUT_TABLES`] order and the batch orders tables by descending height).
pub fn lut_preprocessed_inputs<F: Field>(
    num_tables: usize,
    table_positions: [usize; NUM_LUT_TABLES],
) -> (Vec<Vec<PolynomialValues<F>>>, Vec<Vec<usize>>) {
    assert!(
        table_positions.windows(2).all(|w| w[0] < w[1]) && table_positions[NUM_LUT_TABLES - 1] < num_tables,
        "table positions must be strictly increasing and within the batch"
    );
    let mut values = vec![Vec::new(); num_tables];
    let mut columns = vec![Vec::new(); num_tables];
    for (i, &table) in LUT_TABLES.iter().enumerate() {
        values[table_positions[i]] = lut_precommitted_values::<F>(table)
            .into_iter()
            .map(PolynomialValues::new)
            .collect();
        columns[table_positions[i]] = (0..num_precommitted_columns(table)).collect();
    }
    (values, columns)
}

/// Precommits every LUT column except the per-proof multiplicities:
/// builds the setup-time `BatchStarkPreprocessedData` — LDEs, the batched
/// Merkle tree, and the cap that becomes a consensus constant. The prover copies this data
/// into every proof's preprocessed oracle; the verifier checks proofs against the cap;
/// neither regenerates or recommits the tables per job.
pub fn lut_preprocessed_data<F, C, const D: usize>(
    num_tables: usize,
    table_positions: [usize; NUM_LUT_TABLES],
    config: &StarkConfig,
    timing: &mut TimingTree,
) -> BatchStarkPreprocessedData<F, C, D>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    let (values, columns) = lut_preprocessed_inputs::<F>(num_tables, table_positions);
    BatchStarkPreprocessedData::new(values, columns, config, timing)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::PrimeField64;

    use super::super::ctl::NUM_TABLES;
    use super::super::input_quant_stark::stark::rnernd_scaled;
    use super::super::matmul_b200_stark::columns::{GROUP_WIDTH, MATMUL_B200_COL_MAP, NUM_MATMUL_B200_COLUMNS};
    use super::super::matmul_b200_stark::stark::{MatmulProgram, generate_b200_trace};
    use super::*;
    use crate::api::fp8::dtype::{bf16_to_f32, f32_to_bf16, fp8_e4m3_to_f32};

    type F = GoldilocksField;

    fn to_u64(x: F) -> u64 {
        x.to_canonical_u64()
    }

    #[test]
    fn shapes_match_the_documented_layout_and_the_consumer_arities() {
        for table in LUT_TABLES {
            let columns = generate::<F>(table, 0);
            // Stored-column count = the consumers' value arity (BYTES2/PAIR128 store their key
            // tuple; RANGE16 is its ramp).
            let arity = match table {
                LutTable::Range16 => 0,
                LutTable::Bytes2 | LutTable::Pair128 | LutTable::Width32 | LutTable::Width16 => 2,
                LutTable::Int8Dec | LutTable::RneRnd => 4,
                LutTable::B200Align => 5,
                LutTable::XfPow2 => XFPOW2_LIMBS,
                _ => 1,
            };
            assert_eq!(columns.len(), arity, "{table:?} arity");
            for col in &columns {
                assert_eq!(col.len(), slot_height(table), "{table:?} height");
            }
        }
        // Documented entry counts (logical rows = slots * height).
        assert_eq!(num_slots(LutTable::RneRnd) * slot_height(LutTable::RneRnd), 26 << 17);
        assert_eq!(num_slots(LutTable::B200Align) * slot_height(LutTable::B200Align), 72 << 16);
    }

    #[test]
    fn heights_descend_and_widths_match_the_documented_layout() {
        // Committed heights: the natural (power-of-two) height per table, descending in
        // LUT_TABLES order — nothing is padded to another table's height.
        let heights: Vec<usize> = LUT_TABLES.iter().map(|&t| lut_height(t)).collect();
        assert!(
            heights.windows(2).all(|w| w[0] >= w[1]),
            "LUT_TABLES must descend: {heights:?}"
        );
        assert_eq!(lut_height(LutTable::RneRnd), 1 << 17);
        assert_eq!(lut_height(LutTable::Range16), 1 << 16);
        assert_eq!(lut_height(LutTable::Pair128), 1 << 14);
        assert_eq!(lut_height(LutTable::Clamp22), 1 << 10); // 601 live rows
        assert_eq!(lut_height(LutTable::Int8Dec), 256);
        assert_eq!(lut_height(LutTable::ExpInfo), 256); // 255 live rows
        assert_eq!(lut_height(LutTable::Pow2Gb), 64);
        assert_eq!(lut_height(LutTable::Pow2D), 32); // 20 live rows
        assert_eq!(lut_height(LutTable::Width32), 32);
        assert_eq!(lut_height(LutTable::XfPow2), 1 << 11);
        assert_eq!(lut_height(LutTable::Width16), 1 << 16);
        assert_eq!(lut_height(LutTable::Log16), 1 << 16);

        // Widths: precommitted (keys + stored values) plus one multiplicity column per slot.
        let widths: BTreeMap<LutTable, (usize, usize)> = LUT_TABLES
            .iter()
            .map(|&t| (t, (num_precommitted_columns(t), lut_num_columns(t))))
            .collect();
        assert_eq!(widths[&LutTable::RneRnd], (105, 131));
        assert_eq!(widths[&LutTable::Range16], (1, 2));
        assert_eq!(widths[&LutTable::Bytes2], (2, 3));
        assert_eq!(widths[&LutTable::B200Align], (77, 149));
        assert_eq!(widths[&LutTable::Int8Dec], (5, 6));
        assert_eq!(widths[&LutTable::Pow2Gb], (2, 3));
        assert_eq!(widths[&LutTable::Width32], (3, 4));
        assert_eq!(widths[&LutTable::XfPow2], (7, 8));
        assert_eq!(widths[&LutTable::Width16], (3, 4));
        assert_eq!(widths[&LutTable::Log16], (2, 3));
        for t in [LutTable::Qcast, LutTable::Div448, LutTable::Pair128, LutTable::Clamp22] {
            assert_eq!(widths[&t].1, 3, "{t:?}");
        }
    }

    #[test]
    fn small_tables_match_their_closed_forms() {
        let bytes2 = generate::<F>(LutTable::Bytes2, 0);
        let pair128 = generate::<F>(LutTable::Pair128, 0);
        for i in 0..1 << 16 {
            assert_eq!(to_u64(bytes2[0][i]), (i as u64) & 0xFF);
            assert_eq!(to_u64(bytes2[1][i]), (i as u64) >> 8);
        }
        for i in 0..1 << 14 {
            assert_eq!(to_u64(pair128[0][i]), (i as u64) & 0x7F);
            assert_eq!(to_u64(pair128[1][i]), (i as u64) >> 7);
        }
        let expinfo = generate::<F>(LutTable::ExpInfo, 0);
        assert_eq!(to_u64(expinfo[0][0]), 1);
        assert!((1..255).all(|e| expinfo[0][e] == F::ZERO));
        let clamp22 = generate::<F>(LutTable::Clamp22, 0);
        assert_eq!(to_u64(clamp22[0][0]), 0); // x = -400, clamped (cut -7)
        assert_eq!(to_u64(clamp22[0][392]), 0); // x = -8, clamped (cut -7)
        assert_eq!(to_u64(clamp22[0][393]), 0); // x = -7, the slot floor
        assert_eq!(to_u64(clamp22[0][400]), 7); // x = 0 (cut 0)
        assert_eq!(to_u64(clamp22[0][410]), 17); // x = 10 (cut 10)
        assert_eq!(to_u64(clamp22[0][418]), 25); // x = 18, the slot ceiling
        assert_eq!(to_u64(clamp22[0][600]), 25); // x = 200, clamped (cut 18)
        let pow2d = generate::<F>(LutTable::Pow2D, 0);
        assert_eq!(to_u64(pow2d[0][19]), 1 << 19);
        let pow2gb = generate::<F>(LutTable::Pow2Gb, 0);
        assert_eq!(to_u64(pow2gb[0][25]), 1 << 25);
        assert!((26..64).all(|d| to_u64(pow2gb[0][d]) == 1 << 26), "POW2GB min-cap");
        // WIDTH16: bit length of the 16-bit key and its nonzero flag.
        let width16 = generate::<F>(LutTable::Width16, 0);
        assert_eq!((to_u64(width16[0][0]), to_u64(width16[1][0])), (0, 0), "zero sentinel");
        assert_eq!((to_u64(width16[0][1]), to_u64(width16[1][1])), (1, 1));
        assert_eq!((to_u64(width16[0][0x8000]), to_u64(width16[1][0x8000])), (16, 1));
        assert_eq!((to_u64(width16[0][0xFFFF]), to_u64(width16[1][0xFFFF])), (16, 1));
        assert_eq!((to_u64(width16[0][255 * 255]), to_u64(width16[1][255 * 255])), (16, 1));
        // LOG16: floor(64 * log2 key), 0 at the key-0 sentinel.
        let log16 = generate::<F>(LutTable::Log16, 0);
        assert_eq!(to_u64(log16[0][0]), 0, "zero sentinel");
        assert_eq!(to_u64(log16[0][1]), 0);
        assert_eq!(to_u64(log16[0][1 << 13]), 832);
        assert_eq!(to_u64(log16[0][0xFFFF]), 1023);
        for key in [3usize, 1000, 8192, 12345, 40000, 65535] {
            assert_eq!(to_u64(log16[0][key]), (64.0 * (key as f64).log2()).floor() as u64);
        }
        // WIDTH32: row r holds width W = r + 1; (2^max(W-24,0), 2^max(24-W,0)).
        let width32 = generate::<F>(LutTable::Width32, 0);
        assert_eq!((to_u64(width32[0][0]), to_u64(width32[1][0])), (1, 1 << 23), "W = 1");
        assert_eq!((to_u64(width32[0][23]), to_u64(width32[1][23])), (1, 1), "W = 24");
        assert_eq!((to_u64(width32[0][31]), to_u64(width32[1][31])), (1 << 8, 1), "W = 32");
        for r in 0..32usize {
            let w = r as u64 + 1;
            assert_eq!(to_u64(width32[0][r]), 1 << w.saturating_sub(24));
            assert_eq!(to_u64(width32[1][r]), 1 << 24u64.saturating_sub(w));
        }
    }

    #[test]
    fn int8dec_decodes_every_byte_exactly() {
        let cols = generate::<F>(LutTable::Int8Dec, 0);
        for byte in 0..256usize {
            let v = byte as u8 as i8;
            let (sign, exp, man, eiz) = (
                to_u64(cols[0][byte]),
                to_u64(cols[1][byte]),
                to_u64(cols[2][byte]),
                to_u64(cols[3][byte]),
            );
            if v == 0 {
                assert_eq!((sign, exp, man, eiz), (0, 0, 0, 1));
                continue;
            }
            assert_eq!(eiz, 0, "int8 decodes are normal");
            let code = ((sign << 15) | (exp << 7) | man) as u16;
            assert_eq!(bf16_to_f32(code), v as f32, "byte {byte}");
        }
        // Spot value: byte 0x80 = -128 -> (1, 134, 0, 0), i.e. -128 = -1.0 * 2^7 in bf16.
        assert_eq!(
            (
                to_u64(cols[0][0x80]),
                to_u64(cols[1][0x80]),
                to_u64(cols[2][0x80]),
                to_u64(cols[3][0x80])
            ),
            (1, 134, 0, 0)
        );
    }

    #[test]
    fn qcast_saturates_clamps_and_roundtrips() {
        let col = &generate::<F>(LutTable::Qcast, 0)[0];
        // Every representable fp8 value roundtrips: fp8 -> bf16 code -> QCAST -> the same code.
        for code in 0..=0xFFu8 {
            if code & 0x7F == 0x7F {
                continue; // NaN encodings
            }
            let key = f32_to_bf16(fp8_e4m3_to_f32(code)).unwrap();
            assert_eq!(to_u64(col[key as usize]), u64::from(code), "fp8 {code:#04x}");
        }
        // Saturation beyond ±448 and at the non-finite sentinel keys.
        assert_eq!(to_u64(col[f32_to_bf16(1000.0).unwrap() as usize]), 0x7E);
        assert_eq!(to_u64(col[f32_to_bf16(-1e30).unwrap() as usize]), 0xFE);
        assert_eq!(to_u64(col[0x7F80]), 0x7E, "+inf key holds the saturation code");
        assert_eq!(to_u64(col[0xFFC0]), 0xFE, "NaN keys hold the (unreachable) saturation code");
        // Signed zero and the subnormal region.
        assert_eq!(to_u64(col[0x0000]), 0x00);
        assert_eq!(to_u64(col[0x8000]), 0x80);
        let tiny = f32_to_bf16(2.0f32.powi(-9)).unwrap(); // fp8 subnormal 2^-9 = code 0x01
        assert_eq!(to_u64(col[tiny as usize]), 0x01);
    }

    #[test]
    fn div448_matches_the_native_division_with_sentinels_off_domain() {
        let col = &generate::<F>(LutTable::Div448, 0)[0];
        for key in 0..1 << 16 {
            let code = key as u16;
            let got = to_u64(col[key]) as u16;
            if code & 0x7F80 == 0x7F80 {
                assert_eq!(got, code & 0x8000 | 0x7F80, "non-finite key {key:#06x}");
                continue;
            }
            match bf16_div(CODE_448, code) {
                Ok(alpha) => assert_eq!(got, alpha, "key {key:#06x}"),
                Err(_) => assert_eq!(got & 0x7F80, 0x7F80, "error path must hold an exp-255 sentinel"),
            }
        }
        // Spot values: 448/1 = 448, 448/448 = 1, 448/-1 = -448, 448/±0 -> sentinel.
        assert_eq!(to_u64(col[0x3F80]), 0x43E0);
        assert_eq!(to_u64(col[0x43E0]), 0x3F80);
        assert_eq!(to_u64(col[0xBF80]), 0xC3E0);
        assert_eq!(to_u64(col[0x0000]) & 0x7F80, 0x7F80);
        // The live-domain floor 2^-32 (exponent field 95): quotient 448*2^32, finite and normal.
        assert_eq!(to_u64(col[0x2F80]), u64::from(bf16_div(CODE_448, 0x2F80).unwrap()));
    }

    /// Independent ground truth for the RNERND tests: RNE-encode the exact value `v * 2^scale`
    /// to nonnegative finite bf16 fields `(EXP, MANTISSA, IS_ZERO, EXP_IS_ZERO)` from first
    /// principles (grid selection, ties-to-even, carry, subnormal placement), sharing no code
    /// with the table's pos/cut machinery.
    fn bf16_encode_exact(v: u64, scale: i64) -> (u64, u64, bool, bool) {
        if v == 0 {
            return (0, 0, true, true);
        }
        let e = scale + i64::from(64 - v.leading_zeros()) - 1; // exponent of the leading bit
        // bf16 keeps eight significand bits down to the subnormal floor 2^-133.
        let grid = (e - 7).max(-133);
        let pos = grid - scale;
        let q = if pos <= 0 {
            v << (-pos) as u32
        } else if pos >= 64 {
            0 // v < 2^64 sits far below half of the grid unit
        } else {
            let q0 = v >> pos;
            let rem = v & ((1u64 << pos) - 1);
            let half = 1u64 << (pos - 1);
            q0 + u64::from(rem > half || (rem == half && q0 & 1 == 1))
        };
        if q == 0 {
            return (0, 0, true, true);
        }
        let wq = i64::from(64 - q.leading_zeros());
        let e_out = grid + wq - 1;
        if e_out >= -126 {
            // Normal (a value with leading bit at e >= -126 rounds to >= 2^-126, so grid = e-7
            // implies this branch; the shift below is lossless: wq = 9 only for the even carry
            // q = 256).
            let m = if wq <= 8 { q << (8 - wq) } else { q >> (wq - 8) };
            ((e_out + 127) as u64, m - 128, false, false)
        } else {
            // Subnormal: e_out < -126 forces grid = -133, so q already sits on bf16's grid.
            (0, q, false, true)
        }
    }

    #[test]
    fn rnernd_matches_both_stark_mirrors_and_the_exact_encode_exhaustively() {
        // ScaleStark's `(v, slot)` mirror is the table row function; the slot encodes the
        // signed fade cut, so the pair determines the result everywhere — including the
        // formerly gapped cancellation band (`v < 128` with `KEY_SCALE in [-133, -127]`).
        // Each slot's rows are checked against InputQuant's scale-carrying mirror and the
        // independent exact encoder at its implied scale; the clamped boundary slots are also
        // swept over further scales they serve.
        for slot in 0..num_slots(LutTable::RneRnd) {
            let cols = generate::<F>(LutTable::RneRnd, slot);
            // slot = clamp(-133 - lsb_scale, -7, 18) + 7, so the implied scale is -126 - slot.
            let implied = -126 - slot as i64;
            let scales: &[i64] = match slot {
                0 => &[-126, -125, -100, 40],    // every KEY_SCALE >= -126 clamps here
                25 => &[-151, -152, -180, -260], // every KEY_SCALE <= -151 clamps here
                _ => &[implied],
            };
            for v in 0..1u64 << 17 {
                let expected = rnernd_reference(v, slot as u64);
                let got = (
                    to_u64(cols[0][v as usize]),
                    to_u64(cols[1][v as usize]),
                    to_u64(cols[2][v as usize]) == 1,
                    to_u64(cols[3][v as usize]) == 1,
                );
                assert_eq!(got, expected, "RNERND({v}, {slot})");
                for &lsb_scale in scales {
                    let iq = rnernd_scaled(v, slot as u64, lsb_scale);
                    assert_eq!(
                        (iq.mantissa, iq.width_adjust, iq.is_zero, iq.exp_is_zero),
                        expected,
                        "InputQuant mirror disagrees at ({v}, {slot})"
                    );
                    assert_eq!(
                        (iq.exp, iq.mantissa, iq.is_zero, iq.exp_is_zero),
                        bf16_encode_exact(v, lsb_scale),
                        "exact encode disagrees at ({v}, slot {slot}, scale {lsb_scale})"
                    );
                }
            }
        }
    }

    /// The shared operand recipe of the alignment-table trace tests.
    fn alignment_test_codes(len: usize, salt: u64) -> Vec<u8> {
        const POOL: [u8; 8] = [0x38, 0x40, 0xB9, 0x3A, 0xC1, 0x3B, 0xBA, 0x42];
        (0..len)
            .map(|i| {
                if i % 8 == 0 {
                    0
                } else {
                    POOL[((i as u64).wrapping_mul(salt) ^ (i as u64 >> 3)) as usize % POOL.len()]
                }
            })
            .collect()
    }

    /// Summand scores for the alignment-table trace tests, in the live domain
    /// `[12_928, 54_207]` (the alignment assertions never read them).
    fn alignment_test_lambdas(len: usize, salt: u64) -> Vec<u64> {
        (0..len).map(|i| 12_928 + (i as u64).wrapping_mul(salt) % 41_280).collect()
    }

    #[test]
    fn b200align_matches_matmul_trace_generation() {
        // Every B200ALIGN instance of an honest Matmul trace must
        // hit its table row exactly — key = OPERAND_CODES_A + 2^8*OPERAND_CODES_B,
        // slot = GROUP_MAX_BIASED_EXPONENT - PRODUCT_BIASED_EXPONENT,
        // values = the five bound columns.
        let program = MatmulProgram { h: 2, w: 2, k: 128 };
        let a = alignment_test_codes(program.h * program.k, 0x9E3779B97F4A7C15);
        let b = alignment_test_codes(program.w * program.k, 0xC2B2AE3D27D4EB4F);
        let la = alignment_test_lambdas(program.h * program.k, 0xA24BAED4963EE407);
        let lb = alignment_test_lambdas(program.w * program.k, 0x9FB21C651E98DF25);
        let (rows, _) = generate_b200_trace::<F>(&program, &a, &b, &la, &lb);

        let mut slots: BTreeMap<u64, Vec<Vec<F>>> = BTreeMap::new();
        for row in &rows as &[[F; NUM_MATMUL_B200_COLUMNS]] {
            for i in 0..GROUP_WIDTH {
                let m = &MATMUL_B200_COL_MAP;
                let key = (to_u64(row[m.operand_codes_a[i]]) + (to_u64(row[m.operand_codes_b[i]]) << 8)) as usize;
                let rel = to_u64(row[m.group_max_biased_exponent]) - to_u64(row[m.product_biased_exponents[i]]);
                let cols = slots
                    .entry(rel)
                    .or_insert_with(|| generate::<F>(LutTable::B200Align, rel as usize));
                let expected = [
                    row[m.aligned_lane_terms[i]],
                    row[m.product_biased_exponents[i]],
                    row[m.operand_codes_a[i]],
                    row[m.operand_codes_b[i]],
                    row[m.lane_binades[i]],
                ];
                for (j, &e) in expected.iter().enumerate() {
                    assert_eq!(cols[j][key], e, "lane {i}, key {key:#06x}, rel {rel}, value {j}");
                }
            }
        }
        // Far slots floor every term to zero (REL >= 27 => ALIGNED_LANE_TERMS = 0: P*2^19 < 2^27).
        let far = generate::<F>(LutTable::B200Align, 27);
        assert!(far[0].iter().all(|&t| t == F::ZERO));
        let farthest = generate::<F>(LutTable::B200Align, 71);
        assert!(farthest[0].iter().all(|&t| t == F::ZERO));
    }

    #[test]
    fn binade_column_is_exact() {
        // For every operand-code pair, BINADE equals
        // `floor(log2 |fp8(a) * fp8(b)|) + 139` (0 for zero products), recomputed here from
        // the exact f64 decode. BINADE is slot-independent; slot 0 covers the whole key domain.
        let b200 = generate::<F>(LutTable::B200Align, 0);
        for key in 0..1usize << 16 {
            let (a, b) = ((key & 0xFF) as u8, (key >> 8) as u8);
            // NaN codes never occur in-protocol (QCAST saturates); the operand split
            // treats them as ordinary maximal codes.
            if a & 0x7F == 0x7F || b & 0x7F == 0x7F {
                continue;
            }
            let product = f64::from(fp8_e4m3_to_f32(a)) * f64::from(fp8_e4m3_to_f32(b));
            let expected = if product == 0.0 {
                0
            } else {
                // Exact in f64 and >= 2^-18 in magnitude — a normal, whose unbiased
                // exponent is the binade.
                let unbiased = ((product.abs().to_bits() >> 52) & 0x7FF) as i64 - 1023;
                (unbiased + 139) as u64
            };
            assert_eq!(to_u64(b200[4][key]), expected, "key {key:#06x}");
        }
    }

    #[test]
    fn layout_is_consistent_and_disjoint() {
        for table in LUT_TABLES {
            let precommitted = num_precommitted_columns(table);
            let mut seen_mults = std::collections::BTreeSet::new();
            for slot in 0..num_slots(table) {
                let layout = lut_slot_layout(table, slot);
                for &c in layout.key_columns.iter().chain(&layout.value_columns) {
                    assert!(
                        c < precommitted,
                        "{table:?} slot {slot}: column {c} outside the precommitted block"
                    );
                }
                assert!((precommitted..lut_num_columns(table)).contains(&layout.multiplicity_column));
                assert!(
                    seen_mults.insert(layout.multiplicity_column),
                    "{table:?} slot {slot}: multiplicity column shared"
                );
                assert_eq!(
                    layout.looked_columns::<F>().len(),
                    layout.key_columns.len() + layout.value_columns.len()
                );
            }
            assert_eq!(seen_mults.len(), num_slots(table), "{table:?} mult columns not dense");
        }
        // Absolute spot positions pin the layout against silent reordering.
        assert_eq!(lut_slot_layout(LutTable::Bytes2, 0).key_columns, vec![0, 1]);
        assert_eq!(lut_slot_layout(LutTable::Range16, 0).multiplicity_column, 1);
        assert_eq!(lut_slot_layout(LutTable::RneRnd, 25).value_columns, vec![101, 102, 103, 104]);
        assert_eq!(lut_slot_layout(LutTable::B200Align, 7).value_columns, vec![8, 73, 74, 75, 76]);
        assert_eq!(lut_slot_layout(LutTable::Width32, 0).value_columns, vec![1, 2]);
        assert_eq!(lut_slot_layout(LutTable::RneRnd, 5).key_offset, 5 << 17);
        assert_eq!(lut_slot_layout(LutTable::B200Align, 40).key_offset, 40 << 16);
    }

    #[test]
    fn precommitted_blocks_match_the_generators() {
        for table in LUT_TABLES {
            let block = lut_precommitted_values::<F>(table);
            let height = lut_height(table);
            let live = slot_height(table);
            assert_eq!(block.len(), num_precommitted_columns(table), "{table:?} block width");
            assert!(block.iter().all(|c| c.len() == height), "{table:?} block heights");

            // Key column(s): the enumerated tuple, the shifted ramp, or the (saturated) ramp.
            match table {
                LutTable::Bytes2 | LutTable::Pair128 => {
                    let stored = generate::<F>(table, 0);
                    assert_eq!(block[0], stored[0], "{table:?} key tuple low");
                    assert_eq!(block[1], stored[1], "{table:?} key tuple high");
                }
                LutTable::Width32 => assert!(
                    (0..height).all(|i| block[0][i] == F::from_canonical_usize(i + 1)),
                    "WIDTH32 shifted ramp key [1, 32]"
                ),
                _ => assert!(
                    (0..height).all(|i| block[0][i] == F::from_canonical_usize(i.min(live - 1))),
                    "{table:?} (saturated) ramp key"
                ),
            }
            // Every slot's stored values at their layout positions; sub-height tables pad by
            // repeating the last live row.
            for slot in 0..num_slots(table) {
                let layout = lut_slot_layout(table, slot);
                let stored = generate::<F>(table, slot);
                for (v, &col) in layout.value_columns.iter().enumerate() {
                    assert_eq!(block[col][..live], stored[v][..], "{table:?} slot {slot} value {v}");
                    let last = *stored[v].last().unwrap();
                    assert!(
                        block[col][live..].iter().all(|&x| x == last),
                        "{table:?} slot {slot} value {v} padding"
                    );
                }
            }
        }
        // The soundness point of the saturation: EXPINFO's key column never reaches the
        // inf/NaN field 255, so no multiplicity can prove "exponent 255 is finite".
        let expinfo = lut_precommitted_values::<F>(LutTable::ExpInfo);
        assert!(expinfo[0].iter().all(|&k| to_u64(k) <= 254));
    }

    #[test]
    fn b200align_shared_columns_are_slot_independent() {
        // `lut_precommitted_values` stores B200ALIGN's
        // PRODUCT_BIASED_EXPONENT/OPERAND_CODES_A/B/BINADE once (from slot 0) for all
        // 72 slots (value 0 — the aligned term — is the per-slot column); every slot's
        // generator must agree on them.
        let slot0 = generate::<F>(LutTable::B200Align, 0);
        for slot in [1, 13, 26, 27, 50, 71] {
            let other = generate::<F>(LutTable::B200Align, slot);
            for v in 1..5 {
                assert_eq!(slot0[v], other[v], "shared column {v} differs at slot {slot}");
            }
        }
    }

    #[test]
    fn stark_types_and_ctl_halves_are_consistent() {
        use starky::stark::Stark;

        // The aliases pin the widths the `Stark` trait needs at compile time.
        assert_eq!(<RneRndStark<F, 2> as Stark<F, 2>>::COLUMNS, 131);
        assert_eq!(<Range16Stark<F, 2> as Stark<F, 2>>::COLUMNS, 2);
        assert_eq!(<B200AlignStark<F, 2> as Stark<F, 2>>::COLUMNS, 149);
        assert_eq!(<XfPow2Stark<F, 2> as Stark<F, 2>>::COLUMNS, 8);
        assert_eq!(<Pow2GbStark<F, 2> as Stark<F, 2>>::COLUMNS, 3);
        assert_eq!(<Width32Stark<F, 2> as Stark<F, 2>>::COLUMNS, 4);
        assert_eq!(<Width16Stark<F, 2> as Stark<F, 2>>::COLUMNS, 4);
        assert_eq!(<Log16Stark<F, 2> as Stark<F, 2>>::COLUMNS, 3);
        let stark = QcastStark::<F, 2>::new(LutTable::Qcast);
        assert!(stark.requires_ctls() && stark.lookups().is_empty());

        // Every slot's looked half constructs (its tuple width and batch index are checked
        // against the looking sides by `CrossTableLookup::new` at assembly).
        for (i, &table) in LUT_TABLES.iter().enumerate() {
            for slot in 0..num_slots(table) {
                let _ = ctl_looked_lut_slot::<F>(lut_table_idx(i), table, slot);
            }
        }
    }

    #[test]
    #[should_panic(expected = "wrong AIR width")]
    fn stark_with_mismatched_width_is_rejected() {
        // RNERND needs 116 columns; a 3-column instantiation must panic.
        let _ = LutStark::<F, 2, 3>::new(LutTable::RneRnd);
    }

    #[test]
    fn lut_ctls_assemble_from_inventories() {
        use starky::lookup::Filter;

        // A minimal fake system: table 0 uses RC16 and QCAST, table 1 uses every
        // remaining LUT via one dummy instance each (so the per-table assembly
        // sees every channel non-empty).
        let dummy = |table: LutTable| -> LutLookup<F> {
            let keys = match table {
                LutTable::Bytes2 | LutTable::Pair128 => vec![Column::single(0), Column::single(1)],
                _ => vec![Column::single(0)],
            };
            let values = (0..match table {
                LutTable::Range16 | LutTable::Bytes2 | LutTable::Pair128 => 0,
                LutTable::Width32 | LutTable::Width16 => 2,
                LutTable::Int8Dec | LutTable::RneRnd => 4,
                LutTable::B200Align => 5,
                LutTable::XfPow2 => XFPOW2_LIMBS,
                _ => 1,
            })
                .map(|v| Column::single(2 + v))
                .collect();
            LutLookup {
                table,
                keys,
                values,
                filter: Filter::default(),
            }
        };
        let inventories = [
            (0, vec![LutLookup::rc16(Column::single(0)), dummy(LutTable::Qcast)]),
            (
                1,
                LUT_TABLES
                    .iter()
                    .filter(|&&t| !matches!(t, LutTable::Range16 | LutTable::Qcast))
                    .map(|&t| dummy(t))
                    .collect(),
            ),
        ];
        let ctls = lut_cross_table_lookups::<F>(&LUT_TABLES, &inventories);
        assert_eq!(ctls.len(), NUM_LUT_TABLES, "one channel per LUT");
    }

    #[test]
    fn multiplicities_resolve_add_and_land_in_table_columns() {
        let mut mults = LutMultiplicities::new();
        mults.add(LutTable::Range16, &[0xFFFF], 2).unwrap();
        mults.add(LutTable::RneRnd, &[(5 << 17) + 123], 1).unwrap();
        mults.add(LutTable::Pair128, &[127, 127], 3).unwrap();
        mults.add(LutTable::Pow2Gb, &[63], 4).unwrap();
        // Out-of-domain tuples must be rejected, not folded into some other row.
        assert!(mults.add(LutTable::Range16, &[1 << 16], 1).is_err());
        assert!(
            mults.add(LutTable::Bytes2, &[300, 0], 1).is_err(),
            "oversized tuple component"
        );
        assert!(mults.add(LutTable::RneRnd, &[26 << 17], 1).is_err(), "RNERND has no slot 26");
        assert!(mults.add(LutTable::ExpInfo, &[255], 1).is_err(), "inf/NaN field has no row");
        assert!(mults.add(LutTable::Pair128, &[128, 0], 1).is_err());
        assert!(mults.add(LutTable::Bytes2, &[7], 1).is_err(), "missing tuple component");

        assert_eq!(mults.table_total(LutTable::Range16), 2);
        assert_eq!(mults.table_total(LutTable::RneRnd), 1);
        assert_eq!(mults.table_total(LutTable::Pair128), 3);
        assert_eq!(mults.table_total(LutTable::Pow2Gb), 4);
        assert_eq!(mults.table_total(LutTable::Qcast), 0);

        // The counts land in the right (slot, row) cells of the right tables.
        let rnernd = mults.table_columns::<F>(LutTable::RneRnd);
        assert_eq!(to_u64(rnernd[5][123]), 1);
        assert_eq!(rnernd.iter().flatten().map(|&x| to_u64(x)).sum::<u64>(), 1);
        assert_eq!(to_u64(mults.table_columns::<F>(LutTable::Range16)[0][0xFFFF]), 2);
        assert_eq!(to_u64(mults.table_columns::<F>(LutTable::Pair128)[0][(127 << 7) + 127]), 3);

        // `lut_trace` appends the multiplicities after the precommitted block.
        let trace = lut_trace::<F>(LutTable::Pow2Gb, mults.table_columns(LutTable::Pow2Gb));
        assert_eq!(trace.len(), lut_num_columns(LutTable::Pow2Gb));
        assert!(trace.iter().all(|c| c.len() == 64));
        let pow2gb = lut_slot_layout(LutTable::Pow2Gb, 0);
        assert_eq!(to_u64(trace[pow2gb.multiplicity_column].values[63]), 4);
        assert_eq!(to_u64(trace[pow2gb.key_columns[0]].values[63]), 63);
        assert_eq!(to_u64(trace[pow2gb.value_columns[0]].values[63]), 1 << 26);
    }

    #[test]
    fn multiplicities_resolve_the_matmul_backend_tables() {
        let mut mults = LutMultiplicities::new();
        // B200ALIGN folds at 2^16 into 72 slots.
        mults.add(LutTable::B200Align, &[(71 << 16) + 0x1234], 2).unwrap();
        assert!(
            mults.add(LutTable::B200Align, &[72 << 16], 1).is_err(),
            "B200ALIGN has no slot 72 (negative rel-shifts must stay unservable)"
        );
        // POW2GB is a plain [0, 63] key domain.
        mults.add(LutTable::Pow2Gb, &[63], 1).unwrap();
        assert!(mults.add(LutTable::Pow2Gb, &[64], 1).is_err());
        // WIDTH32's shifted ramp: keys [1, 32] land on rows [0, 31]; keys 0 and 33 have no
        // row (a zero-width claim cannot be served).
        mults.add(LutTable::Width32, &[1], 1).unwrap();
        mults.add(LutTable::Width32, &[32], 5).unwrap();
        assert!(mults.add(LutTable::Width32, &[0], 1).is_err(), "no zero-width row");
        assert!(mults.add(LutTable::Width32, &[33], 1).is_err());

        assert_eq!(to_u64(mults.table_columns::<F>(LutTable::B200Align)[71][0x1234]), 2);
        assert_eq!(to_u64(mults.table_columns::<F>(LutTable::Width32)[0][0]), 1);
        assert_eq!(to_u64(mults.table_columns::<F>(LutTable::Width32)[0][31]), 5);
        assert_eq!(mults.table_total(LutTable::Width32), 6);
    }

    #[test]
    fn checker_serves_honest_instances_and_rejects_forged_values() {
        use starky::lookup::Filter;

        // A four-row trace looking up QCAST(0x3F80) = 0x38 (bf16 1.0 -> fp8 1.0) on rows where
        // the filter is on, POW2GB unfiltered, and BYTES2 as a pure tuple check.
        let f = F::from_canonical_u64;
        let trace: Vec<PolynomialValues<F>> = vec![
            PolynomialValues::new(vec![f(0x3F80); 4]),                          // 0: QCAST key
            PolynomialValues::new(vec![f(0x38); 4]),                            // 1: bound fp8 code
            PolynomialValues::new(vec![f(1), f(0), f(1), f(1)]),                // 2: filter
            PolynomialValues::new(vec![f(3), f(13), f(63), f(20)]),             // 3: POW2GB key
            PolynomialValues::new(vec![f(8), f(8192), f(1 << 26), f(1 << 20)]), // 4: bound power
            PolynomialValues::new(vec![f(200), f(0), f(255), f(17)]),           // 5: a byte
        ];
        let lookups = vec![
            LutLookup {
                table: LutTable::Qcast,
                keys: vec![Column::single(0)],
                values: vec![Column::single(1)],
                filter: Filter::from_column(Column::single(2)),
            },
            LutLookup {
                table: LutTable::Pow2Gb,
                keys: vec![Column::single(3)],
                values: vec![Column::single(4)],
                filter: Filter::default(),
            },
            LutLookup {
                table: LutTable::Bytes2,
                keys: vec![Column::single(5), Column::constant(F::ZERO)],
                values: vec![],
                filter: Filter::default(),
            },
        ];
        let mut checker = LutChecker::<F>::new();
        checker.check_trace(&lookups, &trace, &[], "test").unwrap();
        assert_eq!(checker.multiplicities.table_total(LutTable::Qcast), 3, "filter off on row 1");
        assert_eq!(checker.multiplicities.table_total(LutTable::Pow2Gb), 4);
        assert_eq!(checker.multiplicities.table_total(LutTable::Bytes2), 4);
        let qcast_mults = checker.multiplicities.table_columns::<F>(LutTable::Qcast);
        assert_eq!(to_u64(qcast_mults[0][0x3F80]), 3);

        // A forged binding (QCAST(1.0) claimed = 0x39) must be rejected...
        let forged = LutLookup {
            table: LutTable::Qcast,
            keys: vec![Column::single(0)],
            values: vec![Column::constant(f(0x39))],
            filter: Filter::default(),
        };
        let err = LutChecker::<F>::new()
            .check_trace(&[forged], &trace, &[], "test")
            .unwrap_err();
        assert!(err.contains("differs from the stored"), "{err}");
        // ...as must an in-range key looking up the wrong number of values...
        let short = LutLookup {
            table: LutTable::Int8Dec,
            keys: vec![Column::single(5)],
            values: vec![Column::constant(F::ZERO)],
            filter: Filter::default(),
        };
        let err = LutChecker::<F>::new().check_trace(&[short], &trace, &[], "test").unwrap_err();
        assert!(err.contains("binds 1 values, table stores 4"), "{err}");
        // ...and an out-of-domain key (POW2GB caps its key domain at 63).
        let oob = LutLookup {
            table: LutTable::Pow2Gb,
            keys: vec![Column::single(0)],
            values: vec![Column::single(4)],
            filter: Filter::default(),
        };
        let err = LutChecker::<F>::new().check_trace(&[oob], &trace, &[], "test").unwrap_err();
        assert!(err.contains("out of domain"), "{err}");
    }

    #[test]
    fn preprocessed_inputs_place_the_tables_at_their_batch_positions() {
        // The fp8 arrangement: the sixteen LUTs right after the six main tables.
        let positions: [usize; NUM_LUT_TABLES] = core::array::from_fn(|i| NUM_TABLES + i);
        let (values, columns) = lut_preprocessed_inputs::<F>(NUM_TABLES + NUM_LUT_TABLES, positions);
        assert_eq!((values.len(), columns.len()), (22, 22));
        for t in 0..NUM_TABLES {
            assert!(values[t].is_empty() && columns[t].is_empty(), "main tables commit nothing");
        }
        for (i, &table) in LUT_TABLES.iter().enumerate() {
            let (vals, cols) = (&values[NUM_TABLES + i], &columns[NUM_TABLES + i]);
            assert_eq!(vals.len(), num_precommitted_columns(table));
            assert_eq!(cols, &(0..num_precommitted_columns(table)).collect::<Vec<_>>());
            assert!(vals.iter().all(|v| v.len() == lut_height(table)));
        }
        // Flattened column heights descend (batch positions are height-sorted), so
        // the flat order already is the setup oracle's canonical polynomial order
        // (descending degree, ties by table index).
        let heights: Vec<usize> = values.iter().flatten().map(|v| v.len()).collect();
        assert!(heights.windows(2).all(|w| w[0] >= w[1]));
    }

    /// Builds the real setup commitment — LDEs and the batched Merkle tree over ~20M field
    /// elements. Run with `cargo test --release -p zk-pow -- --ignored lut_precommit`.
    #[test]
    #[ignore = "heavy: full LDE + Merkle commitment of every LUT table; run in release"]
    fn lut_precommitment_commits_all_tables_once() {
        use plonky2::plonk::config::PoseidonGoldilocksConfig;

        let config = StarkConfig::standard_fast_config();
        let positions: [usize; NUM_LUT_TABLES] = core::array::from_fn(|i| NUM_TABLES + i);
        let data = lut_preprocessed_data::<F, PoseidonGoldilocksConfig, 2>(
            NUM_TABLES + NUM_LUT_TABLES,
            positions,
            &config,
            &mut TimingTree::default(),
        );
        assert_eq!(data.columns_per_table.len(), NUM_TABLES + NUM_LUT_TABLES);
        for (i, &table) in LUT_TABLES.iter().enumerate() {
            assert_eq!(
                data.columns_per_table[NUM_TABLES + i],
                (0..num_precommitted_columns(table)).collect::<Vec<_>>()
            );
        }
        let verifier_view = data.verifier_data();
        assert_eq!(verifier_view.cap, data.cap(), "the consensus cap round-trips");
        assert_eq!(verifier_view.columns_per_table, data.columns_per_table);
    }
}
