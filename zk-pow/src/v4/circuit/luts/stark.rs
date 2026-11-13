//! Committed LUT contents, trace generation, setup commitment, and AIR.
//! See the [parent module](super) for the lookup and padding invariants.

use core::marker::PhantomData;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::GenericConfig;
use plonky2::util::timing::TimingTree;
use starky::batch_prover::BatchStarkPreprocessedData;
use starky::batch_stark::BatchStark;
use starky::config::StarkConfig;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use starky::evaluation_frame::StarkFrame;
use starky::stark::Stark;

use super::super::input_quant_stark::stark::qcast;
use super::super::matmul_b200_stark::stark::B200Product;
use super::super::matmul_h100::stark::Fp8Product;
use super::super::scale_stark::stark::{CODE_448, rnernd_reference};
use super::super::unpredictability::{sig_nonzero, sig_width};
use super::LutTable;
use super::columns::{
    fp32_tool_key, log16_key, lut_live_rows, lut_num_columns, num_precommitted_columns, num_slots, rnernd_stored_values,
};
use crate::v4::api::compute::bf16_div;
use crate::v4::circuit::trace_padding::trace_padded_height;

/// Generates one slot's semantic values before fixed-column factoring, in the order
/// the consumers' `LutLookup::values` bind them.
pub fn generate<F: Field>(table: LutTable, slot: usize) -> Vec<Vec<F>> {
    assert!(slot < num_slots(table), "slot {slot} out of range for {table:?}");
    let height = lut_live_rows(table);
    let f = F::from_canonical_u64;
    let mut columns: Vec<Vec<F>> = match table {
        LutTable::Range16 => vec![],
        _ => {
            let arity = match table {
                LutTable::Bytes2
                | LutTable::Pair128
                | LutTable::WidthNorm
                | LutTable::Width32
                | LutTable::Width16
                | LutTable::Pow2G
                | LutTable::Fp32Tools => 2,
                LutTable::Xor8 => 3,
                LutTable::DownscaleThreshold | LutTable::Int8Dec | LutTable::RneRnd => 4,
                LutTable::ProdAlign15 | LutTable::B200Align => 5,
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
            // `(x ^ y, h_slot, slot)` for `(x, y) = (key & 0xFF, key >> 8)`.
            LutTable::Xor8 => {
                let z = (key & 0xFF) ^ (key >> 8);
                let high = match slot {
                    0 => 0,
                    1 => z >> 4,
                    _ => z >> 7,
                };
                columns[0].push(f(z));
                columns[1].push(f(high));
                columns[2].push(f(slot as u64));
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
            LutTable::Pow2G => {
                let width = (key + 1).min(49);
                let (power, flag) = match slot {
                    0 => (1 << key.min(14), 0),
                    1 => (if key <= 25 { 1 << key } else { 1 }, 2 + u64::from(key > 25)),
                    2 => (1, 3),
                    3 => (1 << width.abs_diff(24), 4 + u64::from(width <= 24)),
                    _ => unreachable!(),
                };
                columns[0].push(f(power));
                columns[1].push(f(flag));
            }
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
            // Shared bf16 RNE back-end (ScaleStark's mirror; one slot per signed cut),
            // used by Scale's arithmetic and InputQuant's bf16 decode. The signed cut
            // retains the LSB-scale information needed to classify small significands.
            LutTable::RneRnd => {
                let (mant, wa, oiz, oez) = rnernd_reference(key, slot as u64);
                columns[0].push(f(mant));
                columns[1].push(f(wa));
                columns[2].push(f(u64::from(oiz)));
                columns[3].push(f(u64::from(oez)));
            }
            LutTable::ProdAlign15 => {
                let p = Fp8Product::new((key & 0xFF) as u8, (key >> 8) as u8);
                let aligned = if p.is_zero {
                    F::ZERO
                } else {
                    let mag = f((p.sig << 7) >> slot);
                    if p.sign { -mag } else { mag }
                };
                columns[0].push(aligned);
                columns[1].push(f(p.fp22_exp));
                columns[2].push(f(key & 0xFF));
                columns[3].push(f(key >> 8));
                columns[4].push(f(p.biased_binade()));
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
            LutTable::WidthNorm => {
                let abs = ((slot as u64) << 16) | key;
                let (width, sig14) = if abs == 0 {
                    (0, 0)
                } else {
                    let width = 64 - abs.leading_zeros() as u64;
                    let sig14 = if width >= 14 {
                        abs >> (width - 14)
                    } else {
                        abs << (14 - width)
                    };
                    (width, sig14)
                };
                columns[0].push(f(width));
                columns[1].push(f(sig14));
            }
            // Jackpot check 4's significand-product width: the bit length of the 16-bit key
            // and its nonzero flag (key 0 holds the (0, 0) sentinel).
            LutTable::Width16 => {
                columns[0].push(f(sig_width(key)));
                columns[1].push(f(sig_nonzero(key)));
            }
            // Region 0 is check 3's logarithm (0 at the phantom sentinel); region `1 + class` is
            // check 4's capped Gaussian class at the lambda difference.
            LutTable::Log16 => {
                let (region, index) = log16_key(key as usize);
                columns[0].push(f(super::super::unpredictability::noise_coverage_lookup_value(region, index)));
            }
            LutTable::Fp32Tools => {
                let (operation, argument) = fp32_tool_key(key as usize);
                let (left, right) = match operation {
                    0 if argument == 0 => (0, 1),
                    0 => (1u64 << (32 - argument), 0),
                    1 => (1u64 << (-argument).max(0), 1u64 << argument.clamp(0, 37)),
                    2 => (1u64 << argument.min(35), 1u64 << (35 - argument.min(35))),
                    _ => unreachable!(),
                };
                columns[0].push(f(left));
                columns[1].push(f(right));
            }
            LutTable::DownscaleThreshold => {
                let (threshold, coefficient, region, base) =
                    super::super::unpredictability::noise_coverage_row_parameters(32768 + key, 269 + slot as u64);
                for (column, value) in columns.iter_mut().zip([threshold, coefficient, region, base]) {
                    column.push(f(value));
                }
            }
            LutTable::DownscaleRewrite => {
                let code = key as u8;
                let threshold = (key >> 8) & 127;
                let inside = key >> 15 != 0 && u64::from(code & 127) < threshold;
                let output = if inside {
                    (code & 128) | (slot as u8 * 8) | (code & 7) | 1
                } else {
                    code
                };
                columns[0].push(f(u64::from(output)));
            }
        }
    }
    columns
}

/// Generates the table's full precommitted column block (`columns[c][row]`, the leading
/// [`num_precommitted_columns`] of its trace): the consensus-frozen data behind the setup
/// commitment. Deterministic; padding repeats valid entries (module docs).
pub fn lut_precommitted_values<F: Field>(table: LutTable) -> Vec<Vec<F>> {
    let table_height = lut_live_rows(table);
    let padded_height = trace_padded_height(table_height);
    let mut cols: Vec<Vec<F>> = Vec::with_capacity(num_precommitted_columns(table));
    match table {
        // Enumerate valid key tuples before padding.
        LutTable::Fp32Tools => {
            let keys: Vec<_> = (0..table_height).map(fp32_tool_key).collect();
            cols.push(keys.iter().map(|&(operation, _)| F::from_canonical_u64(operation)).collect());
            cols.push(
                keys.iter()
                    .map(|&(_, argument)| {
                        let magnitude = F::from_canonical_u64(argument.unsigned_abs());
                        if argument < 0 { -magnitude } else { magnitude }
                    })
                    .collect(),
            );
            cols.extend(generate::<F>(table, 0));
        }
        LutTable::DownscaleThreshold => {
            cols.push((0..table_height).map(F::from_canonical_usize).collect());
            let small = generate::<F>(table, 0);
            let large = generate::<F>(table, 1);
            cols.extend([
                small[0].clone(),
                large[0].clone(),
                large[1].clone(),
                small[2].clone(),
                small[3].clone(),
                large[3].clone(),
            ]);
        }
        // The stored key tuple `(region, index)` and the value column.
        LutTable::Log16 => {
            let keys: Vec<_> = (0..table_height).map(log16_key).collect();
            cols.push(keys.iter().map(|&(region, _)| F::from_canonical_u64(region)).collect());
            cols.push(keys.iter().map(|&(_, index)| F::from_canonical_u64(index)).collect());
            cols.extend(generate::<F>(table, 0));
        }
        LutTable::DownscaleRewrite => {
            cols.push((0..table_height).map(F::from_canonical_usize).collect());
            cols.push(generate::<F>(table, 0).remove(0));
            cols.push(
                (0..table_height)
                    .map(|key| F::from_bool(key >> 15 != 0 && (key & 127) < ((key >> 8) & 127)))
                    .collect(),
            );
        }
        LutTable::Bytes2 | LutTable::Pair128 => cols.extend(generate::<F>(table, 0)),
        // The byte pair `(x, y)`, then `x ^ y`, `(x ^ y) >> 4` and `(x ^ y) >> 7`.
        LutTable::Xor8 => {
            cols.extend(generate::<F>(LutTable::Bytes2, 0));
            for shift in [0, 4, 7] {
                cols.push(
                    (0..table_height)
                        .map(|key| F::from_canonical_usize(((key & 0xFF) ^ (key >> 8)) >> shift))
                        .collect(),
                );
            }
        }
        // WIDTH32 admits exactly keys [1, 32], with no key 0.
        LutTable::Width32 => {
            cols.push((0..table_height).map(|i| F::from_canonical_usize(i + 1)).collect());
            cols.extend(generate::<F>(table, 0));
        }
        // Ramp-keyed tables enumerate their live domain.
        _ => {
            cols.push((0..table_height).map(F::from_canonical_usize).collect());
            match table {
                LutTable::Range16 => {} // the ramp is the whole table
                LutTable::Qcast
                | LutTable::Div448
                | LutTable::Width16
                | LutTable::Int8Dec
                | LutTable::ExpInfo
                | LutTable::Clamp22
                | LutTable::Pow2D
                | LutTable::Pow2Gb => {
                    // The argument selects slot zero; it is not a padding value.
                    cols.extend(generate::<F>(table, 0));
                }
                // Eight signed truncations span every alignment shift; decode fields are shared.
                LutTable::ProdAlign15 | LutTable::B200Align => {
                    let shift = if table == LutTable::ProdAlign15 { 7 } else { 19 };
                    let mut base = generate::<F>(table, shift);
                    cols.push(base.remove(0));
                    for j in 1..8 {
                        cols.push(generate::<F>(table, shift + j).remove(0));
                    }
                    let mut shared = base.into_iter();
                    cols.push(shared.next().unwrap()); // Product exponent.
                    cols.push(shared.next().unwrap()); // Operand A; B is derived from the key.
                    shared.next();
                    cols.push(shared.next().unwrap()); // Product binade.
                }
                LutTable::WidthNorm => {
                    cols.extend(generate::<F>(table, 0));
                    for shift in 3..=6 {
                        cols.push((0..table_height).map(|x| F::from_canonical_usize(x >> shift)).collect());
                    }
                }
                LutTable::RneRnd => {
                    let stored = rnernd_stored_values();
                    for slot in 0..num_slots(table) {
                        for (value, column) in generate::<F>(table, slot).into_iter().enumerate() {
                            if stored.contains(&(slot, value)) {
                                cols.push(column);
                            }
                        }
                    }
                }
                LutTable::Pow2G => {
                    cols.push((0..table_height).map(|r| F::from_canonical_usize(r.min(25))).collect());
                    cols.push((0..table_height).map(|r| F::from_canonical_usize((r + 1).min(49))).collect());
                    cols.push(generate::<F>(table, 0).remove(0));
                    let mut gap = generate::<F>(table, 1);
                    for flag in &mut gap[1] {
                        *flag -= F::TWO;
                    }
                    cols.extend(gap);
                    let mut normalization = generate::<F>(table, 3);
                    for flag in &mut normalization[1] {
                        *flag -= F::from_canonical_u64(4);
                    }
                    cols.extend(normalization);
                }
                _ => unreachable!("handled above"),
            }
        }
    }
    // SECURITY: Setup commits these verifier-trusted lookup facts. Padding must repeat
    // an existing key/value tuple across all columns, never introduce a new fact.
    for column in &mut cols {
        let last = *column.last().expect("nonempty LUT key domain");
        column.resize(padded_height, last);
    }
    debug_assert_eq!(cols.len(), num_precommitted_columns(table));
    debug_assert!(cols.iter().all(|c| c.len() == padded_height));
    cols
}

/// Assembles one LUT AIR's full constraint-view trace: the precommitted block followed by the
/// per-slot multiplicity columns (e.g. from [`super::witness::LutMultiplicities::table_columns`]). Under the
/// batch commitment only the multiplicity tail is committed per proof — the leading block is
/// served by [`lut_preprocessed_data`]'s setup oracle; both halves address identically inside
/// constraints and CTLs.
pub fn lut_trace<F: Field>(table: LutTable, multiplicities: Vec<Vec<F>>) -> Vec<PolynomialValues<F>> {
    assert_eq!(multiplicities.len(), num_slots(table));
    for col in &multiplicities {
        assert_eq!(col.len(), trace_padded_height(lut_live_rows(table)));
    }
    let mut cols = lut_precommitted_values::<F>(table);
    cols.extend(multiplicities);
    cols.into_iter().map(PolynomialValues::new).collect()
}

/// One committed LUT's AIR: a pure lookup table of `WIDTH` columns — the precommitted block,
/// then the per-slot multiplicity columns. It has **no constraints of its own**: the
/// precommitted columns are bound by the consensus setup cap, and the multiplicity columns
/// only by the table's CTL channel balance ([`super::super::ctl::lut_cross_table_lookups`]).
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

/// The AIRs at their exact widths for the selected device.
pub type RneRndStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::RneRnd) }>;
pub type Range16Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Range16) }>;
pub type Bytes2Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Bytes2) }>;
pub type Xor8Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Xor8) }>;
pub type QcastStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Qcast) }>;
pub type Div448Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Div448) }>;
pub type Pair128Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Pair128) }>;
pub type Clamp22Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Clamp22) }>;
pub type Int8DecStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Int8Dec) }>;
pub type ExpInfoStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::ExpInfo) }>;
pub type Pow2DStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Pow2D) }>;
pub type Pow2GStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Pow2G) }>;
pub type ProdAlign15Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::ProdAlign15) }>;
pub type WidthNormStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::WidthNorm) }>;
pub type B200AlignStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::B200Align) }>;
pub type Pow2GbStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Pow2Gb) }>;
pub type Width32Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Width32) }>;
pub type Width16Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Width16) }>;
pub type Log16Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Log16) }>;

/// One LUT AIR at its table's exact width (the width is a compile-time constant per table,
/// so the dispatch is a static match).
pub(crate) fn boxed_lut_stark<F: RichField + Extendable<D>, const D: usize>(table: LutTable) -> Box<dyn BatchStark<F, D>> {
    match table {
        LutTable::RneRnd => Box::new(RneRndStark::<F, D>::new(table)),
        LutTable::Range16 => Box::new(Range16Stark::<F, D>::new(table)),
        LutTable::Bytes2 => Box::new(Bytes2Stark::<F, D>::new(table)),
        LutTable::Xor8 => Box::new(Xor8Stark::<F, D>::new(table)),
        LutTable::Qcast => Box::new(QcastStark::<F, D>::new(table)),
        LutTable::Div448 => Box::new(Div448Stark::<F, D>::new(table)),
        LutTable::Pair128 => Box::new(Pair128Stark::<F, D>::new(table)),
        LutTable::Clamp22 => Box::new(Clamp22Stark::<F, D>::new(table)),
        LutTable::Int8Dec => Box::new(Int8DecStark::<F, D>::new(table)),
        LutTable::ExpInfo => Box::new(ExpInfoStark::<F, D>::new(table)),
        LutTable::Pow2D => Box::new(Pow2DStark::<F, D>::new(table)),
        LutTable::Pow2G => Box::new(Pow2GStark::<F, D>::new(table)),
        LutTable::ProdAlign15 => Box::new(ProdAlign15Stark::<F, D>::new(table)),
        LutTable::WidthNorm => Box::new(WidthNormStark::<F, D>::new(table)),
        LutTable::B200Align => Box::new(B200AlignStark::<F, D>::new(table)),
        LutTable::Pow2Gb => Box::new(Pow2GbStark::<F, D>::new(table)),
        LutTable::Width32 => Box::new(Width32Stark::<F, D>::new(table)),
        LutTable::Width16 => Box::new(Width16Stark::<F, D>::new(table)),
        LutTable::Log16 => Box::new(Log16Stark::<F, D>::new(table)),
        LutTable::Fp32Tools => Box::new(LutStark::<F, D, { lut_num_columns(LutTable::Fp32Tools) }>::new(table)),
        LutTable::DownscaleThreshold => Box::new(LutStark::<F, D, { lut_num_columns(LutTable::DownscaleThreshold) }>::new(
            table,
        )),
        LutTable::DownscaleRewrite => Box::new(LutStark::<F, D, { lut_num_columns(LutTable::DownscaleRewrite) }>::new(table)),
    }
}

/// The `BatchStarkPreprocessedData::new` inputs for a batch of `num_tables` tables in which
/// Committed LUT `i` is table `table_positions[i]`:
/// `(values_per_table, columns_per_table)` — each LUT's precommitted columns at its position
/// (empty elsewhere). Positions must be strictly increasing in committed inventory order.
pub fn lut_preprocessed_inputs<F: Field>(
    num_tables: usize,
    table_positions: &[usize],
    tables: &[LutTable],
) -> (Vec<Vec<PolynomialValues<F>>>, Vec<Vec<usize>>) {
    assert_eq!(table_positions.len(), tables.len(), "LUT position count");
    assert!(
        table_positions.windows(2).all(|w| w[0] < w[1]) && *table_positions.last().expect("empty LUT inventory") < num_tables,
        "table positions must be strictly increasing and within the batch"
    );
    let mut values = vec![Vec::new(); num_tables];
    let mut columns = vec![Vec::new(); num_tables];
    for (i, &table) in tables.iter().enumerate() {
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
    table_positions: &[usize],
    tables: &[LutTable],
    config: &StarkConfig,
    timing: &mut TimingTree,
) -> BatchStarkPreprocessedData<F, C, D>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    let (values, columns) = lut_preprocessed_inputs::<F>(num_tables, table_positions, tables);
    BatchStarkPreprocessedData::new(values, columns, config, timing)
}
