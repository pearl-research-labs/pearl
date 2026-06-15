// Blake3 AIR: Constraint Definitions for Blake3 Compression within the Pearl STARK
// ================================================================================
//
// Each Blake3 compression occupies 8 consecutive STARK rows: 7 round rows + 1 finalization row.
// Each row stores 4 intermediate Blake3 states (1056 columns total in BLAKE3_ROUND).
//
// State representation (Blake3State, see blake3_utils.rs):
//   A 16-word Blake3 state [v0..v15] is stored as:
//     row1: v[0..4]   — packed u32 values
//     row2: v[4..8]   — 4 × 32 individual bits (128 columns)
//     row3: v[8..12]  — packed u32 values
//     row4: v[12..16] — 4 × 32 individual bits (128 columns)
//   Total: 264 columns per state, 4 states per row = 1056 columns.
//
// Round constraints (verify_round, conditional on next_is_same_blake):
//   Each round has 4 quarter-rounds: 2 column + 2 diagonal. Each quarter-round consists
//   of 4 1/16-round steps (air_half_eigth_round), using different message words and
//   rotation constants (16,12 then 8,7). Each 1/16-round step verifies:
//     a' = (a + b + m) mod 2^32     (add3_unchecked: unconditional)
//     d' = (a' XOR d) >>> rot        (xor_32_shift_if: conditional, output bits boolean-checked)
//     c' = (c + d') mod 2^32         (add2_unchecked: unconditional)
//     b' = (c' XOR b) >>> rot         (xor_32_shift_if: conditional, output bits boolean-checked)
//   The add_unchecked constraints are unconditional, to stay within degree 3. The trace
//   generator fills finalization-row intermediate states to satisfy them (postprocess_round8_rows).
//
// Finalization (finalize_blake, conditional on next_is_new_blake):
//   On the 8th row (finalization), no round is computed. Instead:
//     cv_out[0..4] = state[0..4] XOR state[8..12]   (row1 XOR row3)
//     cv_out[4..8] = state[4..8] XOR state[12..16]   (row2 XOR row4)
//   states[1].row2/row4 are repurposed as bit decompositions of states[0].row1/row3
//   to enable the XOR computation, implying the output is range-checked u32.
//
// Init state verification (verify_init_state, conditional on is_new_blake):
//   At the start of each compression (round 1), the init state must equal:
//     row1 = cv[0..4], row2 = cv[4..8], row3 = BLAKE3_IV, row4 = tweak parameters.

use super::blake3_compress::BLAKE3_IV;
use crate::v1::circuit::utils::air_utils::RowView;
use crate::v1::circuit::utils::evaluator::Evaluator;

/// Verify a quarter round of the Blake3 permutation.
#[allow(clippy::too_many_arguments)]
fn half_g<V, S, E>(
    eval: &mut E,
    a: V,
    b: &[V],
    c: V,
    d: &[V],
    m: V,
    flag: bool,
    expected_a: V,
    expected_b: &[V],
    expected_c: V,
    expected_d: &[V],
    is_activated: V,
) where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    debug_assert_eq!(b.len(), 32);
    debug_assert_eq!(d.len(), 32);
    debug_assert_eq!(expected_b.len(), 32);
    debug_assert_eq!(expected_d.len(), 32);
    let (rot_1, rot_2) = if flag { (8, 7) } else { (16, 12) };
    let two = eval.u64(2);
    let b_packed = eval.polyval(b, two);
    add3_unchecked(eval, expected_a, a, b_packed, m);
    xor_32_shift_if(eval, expected_a, d, expected_d, is_activated, rot_1);
    let expected_d_packed = eval.polyval(expected_d, two);
    add2_unchecked(eval, expected_c, c, expected_d_packed);
    xor_32_shift_if(eval, expected_c, b, expected_b, is_activated, rot_2);
}

/// Verify a full round of the Blake3 permutation.
fn verify_round<'a, 's, V, S, E>(eval: &mut E, states: &[Blake3State<'s, V>; 5], msg: &[V], is_activated: V)
where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
    'a: 's,
{
    (0..4).for_each(|i| {
        half_g(
            eval,
            states[0].row1[i],
            states[0].row2[i],
            states[0].row3[i],
            states[0].row4[i],
            msg[2 * i],
            false,
            states[1].row1[i],
            states[1].row2[i],
            states[1].row3[i],
            states[1].row4[i],
            is_activated,
        );
    });
    (0..4).for_each(|i| {
        half_g(
            eval,
            states[1].row1[i],
            states[1].row2[i],
            states[1].row3[i],
            states[1].row4[i],
            msg[2 * i + 1],
            true,
            states[2].row1[i],
            states[2].row2[i],
            states[2].row3[i],
            states[2].row4[i],
            is_activated,
        );
    });
    (0..4).for_each(|i| {
        half_g(
            eval,
            states[2].row1[i],
            states[2].row2[(i + 1) % 4],
            states[2].row3[(i + 2) % 4],
            states[2].row4[(i + 3) % 4],
            msg[8 + 2 * i],
            false,
            states[3].row1[i],
            states[3].row2[(i + 1) % 4],
            states[3].row3[(i + 2) % 4],
            states[3].row4[(i + 3) % 4],
            is_activated,
        );
    });
    (0..4).for_each(|i| {
        half_g(
            eval,
            states[3].row1[i],
            states[3].row2[(i + 1) % 4],
            states[3].row3[(i + 2) % 4],
            states[3].row4[(i + 3) % 4],
            msg[8 + 2 * i + 1],
            true,
            states[4].row1[i],
            states[4].row2[(i + 1) % 4],
            states[4].row3[(i + 2) % 4],
            states[4].row4[(i + 3) % 4],
            is_activated,
        );
    });
}

// outputs states[0].row1 ^ states[0].row3,
// then states[0].row2 ^ states[0].row4,
// all constraints are conditional on is_activated.
fn finalize_blake<'a, 's, V, S, E>(eval: &mut E, states: &[Blake3State<'s, V>; 5], is_activated: V) -> [V; 8]
where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
    'a: 's,
{
    let c2 = eval.u64(2);
    // abusing states[1].row2 so that states[0].row1 equal packed(states[1].row2).
    for i in 0..4 {
        let row2_packed = eval.polyval(states[1].row2[i], c2);
        eval.constraint_eq_if(is_activated, states[0].row1[i], row2_packed);
    }
    // abusing states[1].row4 so that states[0].row3 equal packed(states[1].row4).
    for i in 0..4 {
        let row4_packed = eval.polyval(states[1].row4[i], c2);
        eval.constraint_eq_if(is_activated, states[0].row3[i], row4_packed);
    }
    // Note: may materialize these elements as trace values if needed.
    let row1_xor_row3: [V; 4] = core::array::from_fn(|i| xor_32(eval, states[1].row2[i], states[1].row4[i]));
    let row2_xor_row4: [V; 4] = core::array::from_fn(|i| xor_32(eval, states[0].row2[i], states[0].row4[i]));
    // states[1].row2,row4 are checked bits in verify_round (regardless of is_activated).
    // Hence output is uint32 range-checked.
    core::array::from_fn(|i| if i < 4 { row1_xor_row3[i] } else { row2_xor_row4[i - 4] })
}

/// Verify init state when is_new_blake is true.
pub(crate) fn verify_init_state<'s, V, S, E>(
    eval: &mut E,
    init_state: &Blake3State<'s, V>,
    is_new_blake: V,
    cv: &[V],
    blake3_tweak: V,
) where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    let c2 = eval.u64(2);

    // row1 = cv[0..4], row3 = IV (conditional on is_new_blake)
    let civ_array: [V; 4] = core::array::from_fn(|i| eval.u64(BLAKE3_IV[i] as u64));
    for i in 0..4 {
        eval.constraint_eq_if(is_new_blake, init_state.row1[i], cv[i]);
        eval.constraint_eq_if(is_new_blake, init_state.row3[i], civ_array[i]);
    }

    // row2 = cv[4..8] (conditional on is_new_blake, bits boolean-checked unconditionally)
    for i in 0..4 {
        let packed = eval.polyval(init_state.row2[i], c2);
        eval.constraint_eq_if(is_new_blake, packed, cv[i + 4]);
    }

    // row4 encodes blake3 tweak: counter_low(32) | counter_high(16) | flags(8) | block_len(7)
    // Active bits: boolean-checked unconditionally, packed value matches blake3_tweak.
    // Remaining bits: forced to zero when is_new_blake.
    let active_bits = [
        init_state.row4[0],
        &init_state.row4[1][0..16],
        &init_state.row4[3][0..8],
        &init_state.row4[2][0..7],
    ]
    .concat();
    let packed = eval.polyval(&active_bits, c2);
    eval.constraint_eq_if(is_new_blake, packed, blake3_tweak);

    let zero_bits = [&init_state.row4[1][16..], &init_state.row4[2][7..], &init_state.row4[3][8..]].concat();
    for bit in zero_bits {
        let zeroed = eval.mul(is_new_blake, bit);
        eval.constraint(zeroed); // if is_new_blake ==> bit = 0
    }
}

// Goal is to check correctness of current row, and set all restrictions relevant for next row.
// Each blake3 is span into 8 stark rows.
// Result is checked uint32 regardless of round numbers.
// NOTE: may experiment with having only a quarter of round per row. This will increase blake3 efficiency to 28 / 29 rather than 7/8.
// Returns (blake3_output, init_state) where init_state is needed for verify_init_state in pearl_air.
pub(crate) fn blake3_eval_transition_constraints<'a, V, S, E>(
    eval: &mut E,
    trace: &mut RowView<'a, V>,
    msg: &[V], // 16 elements, each pack 4 uint8 elements, le.
    next_trace: &[V],
    next_is_new_blake: V, // Is next_trace starting a new blake3 (not continuing this row's)?
) -> ([V; 8], Blake3State<'a, V>)
// Returns:
// - blake3_output: if 8'th row, output of blake3 packed as 8 uint32s. Range-checked.
// - init_state: for verify_init_state called from pearl_air.
where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    debug_assert_eq!(msg.len(), 16);
    let one = eval.u64(1);
    let mut next_trace = RowView::new(next_trace);

    let init_state = trace.consume_blake3_state();
    let next_init_state = next_trace.consume_blake3_state();

    let states = [
        init_state,
        trace.consume_blake3_state(),
        trace.consume_blake3_state(),
        trace.consume_blake3_state(),
        next_init_state,
    ];

    let next_is_same_blake = eval.sub(one, next_is_new_blake);
    verify_round(eval, &states, msg, next_is_same_blake);
    // We could activate finalize_blake only if is_last_round, which is subset of next_is_new_blake. Either way is fine.
    let blake3_output = finalize_blake(eval, &states, next_is_new_blake);

    (blake3_output, init_state)
}

#[derive(Copy, Clone)]
pub(crate) struct Blake3State<'a, P: Copy> {
    pub row1: [P; 4],
    pub row2: [&'a [P]; 4],
    pub row3: [P; 4],
    pub row4: [&'a [P]; 4],
}

impl<'a, P: Copy> RowView<'a, P> {
    pub(crate) fn consume_blake3_state(&mut self) -> Blake3State<'a, P> {
        let row1 = [(); 4].map(|_| self.consume_single());
        let row2 = [(); 4].map(|_| self.consume_few(32));
        let row3 = [(); 4].map(|_| self.consume_single());
        let row4 = [(); 4].map(|_| self.consume_few(32));
        Blake3State { row1, row2, row3, row4 }
    }
}

// Your responsibility to check if res is u32 if you care about it. Either a+b+c or a+b+c-2^32 or a+b+c-2^33
pub(crate) fn add3_unchecked<V, S, E>(eval: &mut E, res: V, a: V, b: V, c: V)
where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    let sm1 = eval.add(a, b);
    let sm2 = eval.add(sm1, c);
    let c2_32 = eval.u64(1u64 << 32);
    let diff = eval.sub(sm2, res);
    let diff_1 = eval.sub(diff, c2_32);
    let diff_2 = eval.sub(diff_1, c2_32);
    let poly1 = eval.mul(diff, diff_1);
    let poly2 = eval.mul(poly1, diff_2);
    eval.constraint(poly2);
}

// Your responsibility to check whether res is u32 if you care about it. Either a+b or a+b-2^32
pub(crate) fn add2_unchecked<V, S, E>(eval: &mut E, res: V, a: V, b: V)
where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    let sm = eval.add(a, b);
    let diff = eval.sub(sm, res);
    let c2_32 = eval.u64(1u64 << 32);
    let diff_1 = eval.sub(diff, c2_32);
    let c = eval.mul(diff, diff_1);
    eval.constraint(c);
}

// verify res = a ^ (b <<< shift) and that b is composed of bits.
pub(crate) fn xor_32_shift_if<V, S, E>(eval: &mut E, res: V, a: &[V], b: &[V], is_activated: V, shift: usize)
where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    debug_assert!(shift < 32);
    debug_assert_eq!(a.len(), 32);
    debug_assert_eq!(b.len(), 32);
    let two = eval.u64(2);

    for &c in b.iter() {
        eval.constraint_bool(c);
    }

    let xor_bits: [V; 32] = core::array::from_fn(|i| {
        let a_bit = a[i];
        let b_bit = b[(i + 32 - shift) % 32];
        eval.xor_bit(a_bit, b_bit)
    });
    let xor = eval.polyval(&xor_bits, two);
    eval.constraint_eq_if(is_activated, res, xor);
}

// a,b are assumed correct, res is inferred correctly
pub(crate) fn xor_32<V, S, E>(eval: &mut E, a: &[V], b: &[V]) -> V
where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    debug_assert_eq!(a.len(), 32);
    debug_assert_eq!(b.len(), 32);
    let two = eval.u64(2);
    let xor_bits: [V; 32] = core::array::from_fn(|i| eval.xor_bit(a[i], b[i]));
    eval.polyval(&xor_bits, two)
}
