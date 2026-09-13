use std::collections::BTreeMap;

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::PrimeField64;

use super::super::ctl::lut_table_idx;
use super::super::ctl::{LUT_TABLES, NUM_LUT_TABLES, NUM_TABLES};
use super::super::input_quant_stark::stark::rnernd_scaled;
use super::super::matmul_b200_stark::columns::{GROUP_WIDTH, MATMUL_B200_COL_MAP, NUM_MATMUL_B200_COLUMNS};
use super::super::matmul_b200_stark::stark::{MatmulProgram, generate_b200_trace};
use super::super::scale_stark::stark::{CODE_448, rnernd_reference};
use super::ctl::LutLookup;
use super::*;
use crate::api::fp8::compute::bf16_div;
use crate::api::fp8::dtype::{bf16_to_f32, f32_to_bf16, fp8_e4m3_to_f32};
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::util::timing::TimingTree;
use starky::config::StarkConfig;
use starky::lookup::Column;

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
