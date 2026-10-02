//! Committed extractor layout: typed mixed-radix axis patterns.
//!
//! Mirrors the reference miner's `layout.py`. An
//! [`AxisPattern`] is an ordered list of dims `(length, DimType)` with
//! implicit stride = product of the preceding lengths, covering one period
//! `[0..total)`. `Fold` dims span one subtile (folded by a single extractor
//! lane), `Blake` dims enumerate the subtiles (lanes), `Null` dims are the
//! free placement digits of the tile base offset within one period.
//!
//! Placement is periodic: valid tile bases are `lattice_point + q * total`
//! for any `q >= 0`, with `lattice_point` on the Null-digit lattice of one
//! period. The quotient above the top explicit dim is an implicit free
//! digit, so a trailing Null dim is redundant (rejected by construction).
//! Matrix bounds (`tA + tile_max < m`) are checked when the statement is built.
//!
//! The lottery tile per axis is the direct sum of the Fold and Blake digit
//! offsets; illegal layouts (overlapping subtile/grid) are unrepresentable
//! by construction.

use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};

/// Number of subtiles the lottery tile folds into = number of `u32` entries
/// in the extracted jackpot message (64 bytes, exactly one BLAKE3 block).
pub const JACKPOT_ENTRIES: usize = 16;

/// Per-subtile element bound.
pub const MAX_SUBTILE_ELEMS: usize = 256;

/// Lower bound on the per-subtile element count (`fold_size` product across
/// both axes): each lane must fold at least this much committed data.
pub const MIN_SUBTILE_ELEMS: usize = 16;

/// Minimum rows tile size (`tile_size = fold_size * blake_size`): the number
/// of strips opened from the activations matrix.
pub const MIN_TILE_ROWS: usize = 4;

/// Minimum cols tile size: the number of strips opened from the weight matrix.
pub const MIN_TILE_COLS: usize = 16;

/// Upper bound on the whole lottery tile (`rows.tile_size * cols.tile_size`).
pub const MAX_TILE_ELEMS: usize = 2048;

pub const MAX_TILE_ROWS: usize = MAX_TILE_ELEMS / MIN_TILE_COLS;

/// Upper bound on `AxisPattern::total()` (one placement-lattice period):
/// keeps `offset mod total` arithmetic in u32, under the `2^24` matrix-dim
/// cap.
pub const MAX_PATTERN_TOTAL: u64 = 1 << 24;

/// Extractor envelope on a pair of axis patterns: 16 Blake lanes, legal
/// subtile, and `256 ≤ h·w ≤ 2048` with `h ≥ 4`, `w ≥ 16`.
pub fn check_lottery_layout(rows: &AxisPattern, cols: &AxisPattern) -> Result<()> {
    let h = rows.tile_size() as usize;
    let w = cols.tile_size() as usize;
    ensure!(h >= MIN_TILE_ROWS, "h must be >= {MIN_TILE_ROWS} || h={h}");
    ensure!(w >= MIN_TILE_COLS, "w must be >= {MIN_TILE_COLS} || w={w}");
    let tile_elems = h
        .checked_mul(w)
        .ok_or_else(|| anyhow::anyhow!("h*w overflows usize || h={h} w={w}"))?;
    ensure!(tile_elems <= MAX_TILE_ELEMS, "h*w must be <= {MAX_TILE_ELEMS} || h={h} w={w}");
    ensure!(
        tile_elems >= JACKPOT_ENTRIES * MIN_SUBTILE_ELEMS,
        "h*w must be >= {} || h={h} w={w}",
        JACKPOT_ENTRIES * MIN_SUBTILE_ELEMS
    );
    let n_lanes = rows.blake_size() as usize * cols.blake_size() as usize;
    ensure!(
        n_lanes == JACKPOT_ENTRIES,
        "Blake dims must select exactly {JACKPOT_ENTRIES} subtiles, got {n_lanes}"
    );
    let subtile_elems = rows.fold_size() as usize * cols.fold_size() as usize;
    ensure!(
        (MIN_SUBTILE_ELEMS..=MAX_SUBTILE_ELEMS).contains(&subtile_elems),
        "subtile has {subtile_elems} elements, must be {MIN_SUBTILE_ELEMS}..={MAX_SUBTILE_ELEMS}"
    );
    Ok(())
}

/// Largest dim length a single serialized dim byte can carry.
const MAX_DIM_LEN: u32 = 64;

/// The role of one mixed-radix dim in the committed layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(eq, eq_int))]
pub enum DimType {
    /// Free placement digit: the tile may be based at any value of this digit.
    Null = 0,
    /// Subtile digit: folded (summed) by a single extractor lane.
    Fold = 1,
    /// Grid digit: enumerates the extractor lanes (subtiles).
    Blake = 2,
    /// Wire padding for an unused trailing dim slot in the fixed-size
    /// encoding: always length 1, never in a canonical dims list.
    None = 3,
}

impl TryFrom<u8> for DimType {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(DimType::Null),
            1 => Ok(DimType::Fold),
            2 => Ok(DimType::Blake),
            3 => Ok(DimType::None),
            _ => bail!("invalid DimType {value}"),
        }
    }
}

/// A typed mixed-radix pattern for one axis (see module docstring).
///
/// Canonical form (enforced by [`AxisPattern::new`], which normalizes):
/// every dim length `>= 2` and no two adjacent dims of the same type.
/// `total <= 2^24` is enforced at construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "[u8; 6]", into = "[u8; 6]")]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "AxisPattern"))]
pub struct AxisPattern {
    dims: Vec<(u32, DimType)>,
}

impl AxisPattern {
    /// Number of dim slots in the fixed-size wire encoding (one byte each).
    pub const NUM_DIMS: usize = 6;

    /// The canonical padding byte for unused trailing dim slots: a length-1
    /// `None` dim.
    const PAD_BYTE: u8 = DimType::None as u8;

    /// Builds a pattern from an ordered dims list (low stride first),
    /// normalizing to canonical form: length-1 dims dropped, adjacent
    /// same-type dims merged. Rejects length-0 dims, `None` dims with
    /// length > 1 (wire padding only), `total > 2^24`, a trailing `Null` dim
    /// (redundant: the quotient above the top dim is a free digit), and
    /// patterns [`Self::encode`] cannot serialize.
    pub fn new(dims: &[(u32, DimType)]) -> Result<Self> {
        let mut canonical: Vec<(u32, DimType)> = Vec::with_capacity(dims.len());
        let mut total: u64 = 1;
        for &(length, dim_type) in dims {
            ensure!(length >= 1, "dim length must be >= 1");
            ensure!(
                dim_type != DimType::None || length == 1,
                "None dims are wire padding only and must have length 1"
            );
            if length == 1 {
                continue;
            }
            total *= length as u64;
            ensure!(total <= MAX_PATTERN_TOTAL, "pattern total {total} exceeds 2^24");
            // Adjacent same-type dims merge into one run.
            match canonical.last_mut() {
                Some((last_len, last_type)) if *last_type == dim_type => *last_len *= length,
                _ => canonical.push((length, dim_type)),
            }
        }
        ensure!(
            canonical.last().is_none_or(|(_, t)| *t != DimType::Null),
            "trailing Null dim is redundant; omit it"
        );

        // Serializability is a construction-time invariant: reject patterns
        // whose encoding cannot be produced, so `to_bytes` cannot fail.
        Self::encode(&canonical)?;

        Ok(Self { dims: canonical })
    }

    /// The committed byte form of a canonical dims list: exactly
    /// [`AxisPattern::NUM_DIMS`] bytes, one per dim, `(length - 1) << 2 |
    /// type`, unused trailing slots padded with [`AxisPattern::PAD_BYTE`].
    /// Dims longer than 64 are split greedily (largest divisor <= 64 of the
    /// remaining run first). Encodable iff every dim length has no prime
    /// factor > 64 (the greedy split then reaches 1) and the split yields at
    /// most [`AxisPattern::NUM_DIMS`] dim bytes.
    fn encode(canonical: &[(u32, DimType)]) -> Result<[u8; Self::NUM_DIMS]> {
        let mut dim_bytes = Vec::new();
        for &(length, dim_type) in canonical {
            let mut rest = length;
            while rest > 1 {
                let part = (2..=MAX_DIM_LEN.min(rest))
                    .rev()
                    .find(|&d| rest.is_multiple_of(d))
                    .ok_or_else(|| {
                        anyhow::anyhow!("dim length {length} has a prime factor > {MAX_DIM_LEN} and cannot be serialized")
                    })?;
                dim_bytes.push((part as u8 - 1) << 2 | dim_type as u8);
                rest /= part;
            }
        }
        ensure!(
            dim_bytes.len() <= Self::NUM_DIMS,
            "pattern needs more than {} dim bytes to serialize",
            Self::NUM_DIMS
        );
        let mut bytes = [Self::PAD_BYTE; Self::NUM_DIMS];
        bytes[..dim_bytes.len()].copy_from_slice(&dim_bytes);
        Ok(bytes)
    }

    /// The canonical dims list (low stride first).
    pub fn dims(&self) -> &[(u32, DimType)] {
        &self.dims
    }

    /// Product of all dim lengths: one period of the placement lattice.
    pub fn total(&self) -> u32 {
        self.dim_product(|_| true)
    }

    /// Product of the dim lengths selected by `include`.
    fn dim_product(&self, include: impl Fn(DimType) -> bool) -> u32 {
        self.dims
            .iter()
            .filter(|&&(_, t)| include(t))
            .map(|&(length, _)| length)
            .product()
    }

    /// Product of the Fold dim lengths (elements per subtile on this axis).
    pub fn fold_size(&self) -> u32 {
        self.dim_product(|t| t == DimType::Fold)
    }

    /// Product of the Blake dim lengths (subtiles on this axis).
    pub fn blake_size(&self) -> u32 {
        self.dim_product(|t| t == DimType::Blake)
    }

    /// Sorted offsets generated by the dims selected by `include`: the sums
    /// of `{digit * stride}` over those dims, with stride the running product
    /// of lengths (a generalized arithmetic progression).
    fn offsets_where(&self, include: impl Fn(DimType) -> bool) -> Vec<u32> {
        let mut offsets = Vec::with_capacity(self.dim_product(&include) as usize);
        offsets.push(0);
        let mut stride: u32 = 1;
        for &(length, t) in &self.dims {
            if include(t) {
                let base = offsets.len();
                for d in 1..length {
                    for i in 0..base {
                        offsets.push(offsets[i] + d * stride);
                    }
                }
            }
            stride *= length;
        }
        // Sorted by construction: every offset so far is below the current
        // stride (the product of ALL preceding lengths, included or not), so
        // each digit's block `{prev + d * stride}` starts above the last.
        debug_assert!(offsets.is_sorted());
        offsets
    }

    /// Sorted offsets generated by the Fold dims (within-subtile offsets).
    pub fn fold_offsets(&self) -> Vec<u32> {
        self.offsets_where(|t| t == DimType::Fold)
    }

    /// Sorted offsets generated by the Blake dims (subtile base offsets).
    pub fn blake_offsets(&self) -> Vec<u32> {
        self.offsets_where(|t| t == DimType::Blake)
    }

    /// Sorted offsets of the lottery tile: the Fold and Blake digit offsets.
    pub fn tile_offsets(&self) -> Vec<u32> {
        self.offsets_where(|t| t != DimType::Null)
    }

    /// Number of rows/cols the tile selects (`fold_size * blake_size`).
    pub fn tile_size(&self) -> u32 {
        self.fold_size() * self.blake_size()
    }

    /// Largest offset the tile selects: every Fold/Blake digit at its
    /// maximum, `sum (length - 1) * stride` over the non-Null dims.
    pub fn tile_max(&self) -> u32 {
        let mut max = 0;
        let mut stride: u32 = 1;
        for &(length, t) in &self.dims {
            if t != DimType::Null {
                max += (length - 1) * stride;
            }
            stride *= length;
        }
        max
    }

    /// Whether `offset` is a valid tile base: every Fold/Blake digit of
    /// `offset mod total` is zero (Null digits are free, including the
    /// implicit quotient above the top dim). The base is otherwise unbounded;
    /// matrix-fit is the caller's responsibility.
    ///
    /// No-wrap guarantee callers rely on: a valid base satisfies
    /// `offset + total <= 2^32`, and every tile offset is `< total`, so
    /// `offset + tile_offset` never wraps u32.
    pub fn offset_is_valid(&self, offset: u32) -> bool {
        if offset > u32::MAX - (self.total() - 1) {
            return false;
        }
        let mut rest = offset;
        for &(length, dim_type) in &self.dims {
            let digit = rest % length;
            rest /= length;
            if digit != 0 && dim_type != DimType::Null {
                return false;
            }
        }
        true
    }

    /// All valid tile base offsets within one period `[0..total)`, sorted: the
    /// Null-digit lattice. The full placement set is `lattice_point + q * total`.
    pub fn valid_offsets(&self) -> Vec<u32> {
        self.offsets_where(|t| t == DimType::Null)
    }

    /// Serializes to the committed byte form ([`Self::encode`] documents the
    /// format). Cannot fail: serializability is enforced at construction.
    pub fn to_bytes(&self) -> [u8; Self::NUM_DIMS] {
        Self::encode(&self.dims).expect("serializability enforced at construction")
    }

    /// Parses a pattern from its fixed-size committed byte form. Enforces
    /// canonicality by round-trip: the bytes must equal the re-serialization
    /// (which also rejects mis-placed or mis-valued padding).
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        ensure!(
            data.len() == Self::NUM_DIMS,
            "Expected {} bytes, got {}",
            Self::NUM_DIMS,
            data.len()
        );
        let mut dims = Vec::with_capacity(Self::NUM_DIMS);
        for &byte in data {
            let dim_type = DimType::try_from(byte & 3)?;
            let length = (byte >> 2) as u32 + 1;
            dims.push((length, dim_type));
        }
        let pattern = Self::new(&dims)?;
        ensure!(pattern.to_bytes() == data, "non-canonical pattern encoding: {data:02x?}");
        Ok(pattern)
    }
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl AxisPattern {
    #[classattr]
    #[pyo3(name = "NUM_DIMS")]
    fn py_num_dims() -> usize {
        Self::NUM_DIMS
    }

    #[new]
    fn py_new(dims: Vec<(u32, DimType)>) -> pyo3::PyResult<Self> {
        Self::new(&dims).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn get_dims(&self) -> Vec<(u32, DimType)> {
        self.dims.clone()
    }

    #[pyo3(name = "to_bytes")]
    fn py_to_bytes(&self) -> Vec<u8> {
        self.to_bytes().to_vec()
    }

    #[staticmethod]
    #[pyo3(name = "from_bytes")]
    fn py_from_bytes(data: Vec<u8>) -> pyo3::PyResult<Self> {
        Self::from_bytes(&data).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

impl From<AxisPattern> for [u8; AxisPattern::NUM_DIMS] {
    fn from(value: AxisPattern) -> Self {
        value.to_bytes()
    }
}

impl TryFrom<[u8; AxisPattern::NUM_DIMS]> for AxisPattern {
    type Error = anyhow::Error;

    fn try_from(value: [u8; AxisPattern::NUM_DIMS]) -> Result<Self> {
        Self::from_bytes(&value)
    }
}

/// Map each lane to its subtile's flat tile indices, in the pinned fold order:
/// lane `rank(b_r) * |B_c| + rank(b_c)` (row-major over the sorted blake
/// offsets) folds subtile `(A_r + b_r) x (A_c + b_c)` row-major over the
/// sorted fold offsets; tile rows/cols are indexed by the sorted tile
/// offsets. Flat indices address the row-major `rows.tile_size() x
/// cols.tile_size()` output tile.
pub fn lane_assignment(rows: &AxisPattern, cols: &AxisPattern) -> Vec<Vec<usize>> {
    // Per axis: for each blake offset (sorted), that subtile's positions in
    // the sorted tile offsets, in fold-offset order.
    fn subtile_positions(axis: &AxisPattern) -> Vec<Vec<usize>> {
        let tile = axis.tile_offsets();
        let fold = axis.fold_offsets();
        let pos = |offset: u32| tile.binary_search(&offset).expect("valid direct sum");
        axis.blake_offsets()
            .iter()
            .map(|&b| fold.iter().map(|&a| pos(a + b)).collect())
            .collect()
    }

    let row_subtiles = subtile_positions(rows);
    let col_subtiles = subtile_positions(cols);
    let n_cols = cols.tile_size() as usize;
    let mut lanes = Vec::with_capacity(row_subtiles.len() * col_subtiles.len());
    for rs in &row_subtiles {
        for cs in &col_subtiles {
            lanes.push(rs.iter().flat_map(|&r| cs.iter().map(move |&c| r * n_cols + c)).collect());
        }
    }
    lanes
}

#[cfg(test)]
mod tests {
    use super::*;
    use DimType::{Blake, Fold, Null};

    fn axis(dims: &[(u32, DimType)]) -> AxisPattern {
        AxisPattern::new(dims).unwrap()
    }

    /// A realistic mining layout: 4 consecutive blake lanes, and 32 cols
    /// folded into 4 blake lanes. Total tile size 4 x 128.
    fn default_rows() -> AxisPattern {
        axis(&[(4, Blake)])
    }
    fn default_cols() -> AxisPattern {
        axis(&[(2, Fold), (4, Blake), (16, Fold)])
    }

    #[test]
    fn default_pattern_tile_layout() {
        let rows = default_rows();
        assert_eq!(rows.tile_offsets(), &[0, 1, 2, 3]);
        assert_eq!((rows.tile_size(), rows.total(), rows.tile_max()), (4, 4, 3));
        assert_eq!(rows.fold_offsets(), &[0]);
        assert_eq!(rows.blake_offsets(), &[0, 1, 2, 3]);

        let cols = default_cols();
        assert_eq!(cols.tile_offsets(), (0..128).collect::<Vec<u32>>());
        assert_eq!((cols.tile_size(), cols.total(), cols.tile_max()), (128, 128, 127));
        assert_eq!(cols.blake_offsets(), &[0, 2, 4, 6]);
        assert_eq!(
            cols.fold_offsets(),
            (0..16).flat_map(|b| [8 * b, 8 * b + 1]).collect::<Vec<u32>>()
        );

        assert_eq!(rows.blake_size() as usize * cols.blake_size() as usize, JACKPOT_ENTRIES);
        assert_eq!(rows.fold_size() as usize * cols.fold_size() as usize, 32);
    }

    /// A Null gap opens the placement lattice; placement is periodic. The
    /// lattice agrees with `offset_is_valid`, and the no-wrap bound rejects
    /// bases that would let `base + tile_max` leave u32.
    #[test]
    fn gapped_tile_offset_lattice() {
        let a = axis(&[(2, Fold), (4, Null), (2, Blake)]);
        assert_eq!(a.tile_offsets(), &[0, 1, 8, 9]);
        assert_eq!(a.valid_offsets(), &[0, 2, 4, 6]);
        assert!(a.offset_is_valid(0) && a.offset_is_valid(2) && a.offset_is_valid(16) && a.offset_is_valid(18));
        assert!(!a.offset_is_valid(1) && !a.offset_is_valid(8));

        let b = axis(&[(5, Null), (2, Fold), (3, Null), (2, Blake)]);
        let lattice = b.valid_offsets();
        for off in 0..2 * b.total() {
            assert_eq!(b.offset_is_valid(off), lattice.contains(&(off % b.total())), "offset {off}");
        }

        let limit = u32::MAX - (a.total() - 1);
        assert!(a.offset_is_valid(limit));
        assert!(!a.offset_is_valid(limit + 2) && !a.offset_is_valid(u32::MAX));
    }

    /// Normalization: length-1 dims drop, adjacent same-type dims merge;
    /// length-0, totals > 2^24, and trailing Null dims are rejected.
    #[test]
    fn constructor_normalizes() {
        assert_eq!(
            axis(&[(4, Null), (2, Fold), (1, Blake), (3, Fold)]),
            axis(&[(4, Null), (6, Fold)])
        );
        assert!(AxisPattern::new(&[(0, Fold)]).is_err());
        assert!(AxisPattern::new(&[(1 << 12, Null), (1 << 12, Fold)]).is_ok());
        assert!(AxisPattern::new(&[(1 << 13, Null), (1 << 12, Fold)]).is_err());
        assert!(AxisPattern::new(&[(2, Fold), (4, Null)]).is_err(), "trailing Null");
    }

    /// Serialization: 6 bytes, 1 per dim, `None`-padded; greedy largest-divisor
    /// splits (128->64*2, 96->48*2, 2^20->64*64*64*4). Non-canonical encodings
    /// are rejected on parse.
    #[test]
    fn serialization_roundtrip_and_rejections() {
        let byte = |len: u32, t: DimType| (len as u8 - 1) << 2 | t as u8;
        let pad = AxisPattern::PAD_BYTE;
        let roundtrip = |dims: &[(u32, DimType)]| {
            let a = axis(dims);
            assert_eq!(AxisPattern::from_bytes(&a.to_bytes()).unwrap(), a);
            a
        };

        let a = roundtrip(&[(6, Null), (4, Fold), (4, Blake)]);
        assert_eq!(a.to_bytes(), [byte(6, Null), byte(4, Fold), byte(4, Blake), pad, pad, pad]);

        for (run, parts) in [(128u32, &[64, 2][..]), (96, &[48, 2]), (1 << 20, &[64, 64, 64, 4])] {
            let a = roundtrip(&[(run, Null), (2, Fold)]);
            let mut expected = [pad; 6];
            for (i, &p) in parts.iter().enumerate() {
                expected[i] = byte(p, Null);
            }
            expected[parts.len()] = byte(2, Fold);
            assert_eq!(a.to_bytes(), expected, "run {run}");
        }

        assert!(AxisPattern::new(&[(134, Null), (2, Fold)]).is_err(), "prime factor > 64");
        assert!(
            AxisPattern::new(&[(2, Null), (2, Fold), (2, Null), (2, Fold), (2, Null), (2, Fold), (2, Blake)]).is_err(),
            "> 6 dim bytes"
        );
        // Wrong length inputs are rejected by the size check.
        assert!(
            AxisPattern::from_bytes(&[byte(4, Fold), pad, pad, pad, pad]).is_err(),
            "truncated"
        );
        assert!(
            AxisPattern::from_bytes(&[byte(4, Fold), pad, pad, pad, pad, pad, pad]).is_err(),
            "trailing bytes"
        );

        let bad: &[[u8; 6]] = &[
            [0x00, pad, pad, pad, pad, pad],
            [byte(4, Null), pad, pad, pad, pad, pad],
            [byte(2, Null), byte(64, Null), byte(2, Fold), pad, pad, pad],
            [byte(32, Fold), byte(2, Fold), pad, pad, pad, pad],
            [byte(4, Fold), byte(2, DimType::None), pad, pad, pad, pad],
            [byte(4, Fold), pad, byte(4, Blake), pad, pad, pad],
            [byte(4, Fold), 0x00, pad, pad, pad, pad],
            [
                byte(64, Null),
                byte(64, Null),
                byte(64, Null),
                byte(64, Null),
                byte(16, Fold),
                pad,
            ],
        ];
        for enc in bad {
            assert!(AxisPattern::from_bytes(enc).is_err(), "accepted bad encoding {enc:02x?}");
        }
    }

    /// Lane assignment on the default layout: 16 lanes, each folding a 1 x 32
    /// subtile of the 4 x 128 tile; every tile element folded exactly once.
    #[test]
    fn lane_assignment_default_layout() {
        let lanes = lane_assignment(&default_rows(), &default_cols());
        assert_eq!(lanes.len(), JACKPOT_ENTRIES);
        assert!(lanes.iter().all(|l| l.len() == 32));
        // Lane 0: blake (row 0, col 0) -> row 0 x cols subtile 0.
        assert_eq!(lanes[0], (0..16).flat_map(|b| [8 * b, 8 * b + 1]).collect::<Vec<usize>>());
        // Lane 15: blake (row 3, col 3) -> row 3 x cols subtile 3 (+6).
        assert_eq!(
            lanes[15],
            (0..16)
                .flat_map(|b| [3 * 128 + 8 * b + 6, 3 * 128 + 8 * b + 7])
                .collect::<Vec<usize>>()
        );
        let mut all: Vec<usize> = lanes.into_iter().flatten().collect();
        all.sort_unstable();
        assert_eq!(all, (0..512).collect::<Vec<usize>>());
    }
}
