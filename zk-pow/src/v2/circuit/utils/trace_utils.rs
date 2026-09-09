use plonky2::{field::types::Field, hash::hash_types::RichField};
use plonky2_field::types::Field64;

/// Convert a byte array to u32 words (little-endian).
pub(crate) fn bytes_to_words<const B: usize, const W: usize>(bytes: &[u8; B]) -> [u32; W] {
    debug_assert_eq!(B, W * 4);
    std::array::from_fn(|i| u32::from_le_bytes(bytes[i * 4..(i + 1) * 4].try_into().unwrap()))
}

/// Unpack unsigned value with power-of-2 base.
pub(crate) fn u64_unpack_le(mut packed: u64, bit_width: usize, num_chunks: usize) -> Vec<u64> {
    let mask = (1u64 << bit_width) - 1;
    let mut res = Vec::with_capacity(num_chunks);
    for _ in 0..num_chunks {
        res.push(packed & mask);
        packed >>= bit_width;
    }
    res
}

/// Unpack signed value with arbitrary base. Each limb is in [-base/2, base/2].
pub(crate) fn i64_unpack_base(mut packed: i64, base: i64, num_chunks: usize) -> Vec<i64> {
    let half_range = base / 2;
    let mut res = Vec::with_capacity(num_chunks);
    for _ in 0..num_chunks {
        let mut limb = packed.rem_euclid(base);
        if limb > half_range {
            limb -= base;
        }
        res.push(limb);
        packed = (packed - limb) / base;
    }
    res
}

/// Pack unsigned value with power-of-2 base.
pub(crate) fn u64_pack_le<T: Copy + Into<u64>>(limbs: &[T], bit_width: usize) -> u64 {
    let mut res = 0;
    for (i, limb) in limbs.iter().enumerate() {
        res |= (*limb).into() << (i * bit_width);
    }
    res
}

/// Result = limbs[0] + base*limbs[1] + base^2*limbs[2] + ...
pub(crate) fn i64_pack_base<T: Copy + Into<i64>>(limbs: &[T], base: i64) -> i64 {
    limbs.iter().rev().fold(0i64, |acc, limb| acc * base + (*limb).into())
}

pub(crate) fn index_to_pair(mut idx: usize) -> (usize, usize) {
    // output idx'th lexicographically smallest tuple (j,i) with 0 <= i <= j.
    for j in 0.. {
        if idx > j {
            idx -= j + 1;
            continue;
        }
        return (idx, j);
    }
    unreachable!()
}

/// Trait for types that can be converted to/from a field element.
/// u32: direct unsigned conversion
/// i32: signed field encoding (negative values map to field order - |value|)
pub(crate) trait FieldConvert<F: RichField + Field64>: Sized + Copy {
    fn to_field(self) -> F;
    fn from_field(f: F) -> Self;
}

impl<F: RichField> FieldConvert<F> for u32 {
    #[inline]
    fn to_field(self) -> F {
        F::from_canonical_u64(self as u64)
    }
    #[inline]
    fn from_field(f: F) -> Self {
        f.to_canonical_u64() as u32
    }
}

impl<F: RichField> FieldConvert<F> for i32 {
    #[inline]
    fn to_field(self) -> F {
        F::from_canonical_u64(i64_to_u64::<F>(self as i64))
    }
    #[inline]
    fn from_field(f: F) -> Self {
        field_to_i64(f) as i32
    }
}

impl<F: RichField> FieldConvert<F> for u8 {
    #[inline]
    fn to_field(self) -> F {
        F::from_canonical_u64(self as u64)
    }
    #[inline]
    fn from_field(f: F) -> Self {
        f.to_canonical_u64() as u8
    }
}

/// Write a slice of values to trace columns starting at the given column index.
/// Works for u8, u32 (unsigned) and i32 (signed field encoding).
pub(crate) fn write_to_trace<F: RichField, T: FieldConvert<F>>(trace: &mut [F], col: usize, values: &[T]) {
    for (i, &val) in values.iter().enumerate() {
        trace[col + i] = val.to_field();
    }
}

/// Read N values from trace columns starting at the given column index.
/// Works for u8, u32 (unsigned) and i32 (signed field decoding).
pub(crate) fn read_from_trace<F: RichField, T: FieldConvert<F>, const N: usize>(trace: &[F], col: usize) -> [T; N] {
    std::array::from_fn(|i| T::from_field(trace[col + i]))
}

/// Read 128 bits (4 × 32 bits) from trace starting at col, return as 4 u32 values (little-endian packed)
pub(crate) fn read_bits_as_u32s<F: RichField>(trace: &[F], col: usize) -> [u32; 4] {
    std::array::from_fn(|i| {
        let mut val = 0u32;
        for j in 0..32 {
            val |= (trace[col + i * 32 + j].to_canonical_u64() as u32) << j;
        }
        val
    })
}

/// Write a single u32 value as 32 bits to trace (little-endian)
pub(crate) fn write_u32_as_bits<F: RichField>(trace: &mut [F], col: usize, val: u32) {
    let mut v = val;
    for j in 0..32 {
        trace[col + j] = F::from_canonical_u64((v & 1) as u64);
        v >>= 1;
    }
}

/// Write 4 u32 values as 128 bits to trace (4 × 32 bits, little-endian each)
pub(crate) fn write_u32s_as_bits<F: RichField>(trace: &mut [F], col: usize, vals: &[u32; 4]) {
    for (i, &val) in vals.iter().enumerate() {
        write_u32_as_bits(trace, col + i * 32, val);
    }
}

pub(crate) struct RowBuilder<'a, F: Field> {
    pub(crate) row: &'a mut [F],
    pub(crate) offset: usize,
}

impl<'a, F: Field> RowBuilder<'a, F> {
    pub(crate) fn dump_noop(&mut self) -> F {
        self.offset += 1;
        self.row[self.offset - 1]
    }

    pub(crate) fn dump_field(&mut self, value: F) {
        self.row[self.offset] = value;
        self.offset += 1;
    }

    pub(crate) fn dump_u64<T: Into<u64>>(&mut self, value: T) {
        self.dump_field(F::from_canonical_u64(value.into()));
    }

    pub(crate) fn dump_i64<T: Into<i64>>(&mut self, value: T)
    where
        F: RichField,
    {
        self.dump_u64(i64_to_u64::<F>(value.into()));
    }

    pub(crate) fn assert_end(&self) {
        debug_assert_eq!(
            self.offset,
            self.row.len(),
            "RowBuilder not fully used; remains [offset={}, row_len={})",
            self.offset,
            self.row.len()
        );
    }
}

/// Homomorphic encoding of i64 to F. Returns F::to_canonical_u64 of the result.
pub(crate) fn i64_to_u64<F: RichField>(i: i64) -> u64 {
    debug_assert!(F::ORDER > i64::MAX as u64, "F::ORDER must be greater than i64::MAX");
    if i < 0 { F::ORDER - ((-i) as u64) } else { i as u64 }
}

// Assuming a field element encodes a legitimate (-order/2, order/2) integer, return it.
// inverse of F::from_canonical_u64(i64_to_u64).
pub(crate) fn field_to_i64<F: RichField>(f: F) -> i64 {
    let u = f.to_canonical_u64();
    if u <= F::ORDER / 2 {
        u as i64
    } else {
        -((F::ORDER - u) as i64)
    }
}

// Supports idx <= LEN * (LEN + 1) / 2 options.
pub(crate) fn deg2_muxer_bits<const LEN: usize>(idx: Option<usize>) -> [bool; LEN] {
    if let Some(idx) = idx {
        let (i, j) = index_to_pair(idx);
        let mut res = [false; LEN];
        res[i] = true;
        res[j] = true;
        res
    } else {
        [false; LEN]
    }
}
