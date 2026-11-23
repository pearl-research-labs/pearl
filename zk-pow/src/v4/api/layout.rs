//! Consecutive lottery tiles and their fixed row-major extractor order.
//!
//! The job commits one size per axis (`h`, `w`); the statement chooses the
//! starting row/column offsets (`tA`, `tB`), which must be multiples of the
//! size. Matmul packs the selected rows and columns into an `h * w` output buffer.
//! Its sixteen consecutive, equal-sized chunks are folded independently into
//! the sixteen words of the 64-byte BLAKE3 jackpot message. Matrix coordinates
//! never enter these buffer-relative chunk indices.

use anyhow::{Result, ensure};

/// Number of `u32` words in the 64-byte jackpot message.
pub const JACKPOT_ENTRIES: usize = 16;

/// Per-subtile element bound.
pub const MAX_SUBTILE_ELEMS: usize = 256;

/// Each extractor word must fold at least this many output cells.
pub const MIN_SUBTILE_ELEMS: usize = 16;

/// Minimum number of selected activation rows.
pub const MIN_TILE_ROWS: usize = 4;

/// Minimum number of selected weight rows (output columns).
pub const MIN_TILE_COLS: usize = 16;

/// Maximum number of output cells in the whole lottery tile.
pub const MAX_TILE_ELEMS: usize = 2048;

pub const MAX_TILE_ROWS: usize = MAX_TILE_ELEMS / MIN_TILE_COLS;

/// Validates the committed tile shape before allocating strips or lane data.
pub fn check_lottery_layout(h: usize, w: usize) -> Result<()> {
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
    ensure!(
        tile_elems.is_multiple_of(JACKPOT_ENTRIES),
        "h*w must be a multiple of {JACKPOT_ENTRIES} || h={h} w={w}"
    );
    let subtile_elems = tile_elems / JACKPOT_ENTRIES;
    ensure!(
        (MIN_SUBTILE_ELEMS..=MAX_SUBTILE_ELEMS).contains(&subtile_elems),
        "subtile has {subtile_elems} elements, must be {MIN_SUBTILE_ELEMS}..={MAX_SUBTILE_ELEMS}"
    );
    Ok(())
}

/// The fixed extractor schedule, shared by native extraction and the ZK program:
/// `lanes[j]` lists lane `j`'s cell indices in fold order, the `j`-th of sixteen
/// consecutive, equal-sized chunks of the compact row-major output tile, starting at zero.
///
/// # Panics
/// Panics if `h * w` is not divisible by sixteen. Public statement validation
/// enforces the stronger lottery envelope before calling this helper.
pub fn lane_assignment(h: usize, w: usize) -> Vec<Vec<usize>> {
    assert!(
        (h * w).is_multiple_of(JACKPOT_ENTRIES),
        "h*w must be a multiple of {JACKPOT_ENTRIES}"
    );
    let subtile_elems = h * w / JACKPOT_ENTRIES;
    (0..JACKPOT_ENTRIES)
        .map(|lane| (lane * subtile_elems..(lane + 1) * subtile_elems).collect())
        .collect()
}

/// Validates exactly `tile_size` consecutive opened indices starting at a multiple
/// of `tile_size`, and returns their start. Rejects empty, reordered, duplicated,
/// gapped, unaligned, and overflowing selections.
pub fn parse_axis(indices: &[u32], tile_size: u32) -> Result<u32> {
    ensure!(!indices.is_empty(), "pattern parsing error: Empty indices");
    ensure!(
        indices.windows(2).all(|w| w[0] < w[1]),
        "pattern parsing error: Indices not strictly increasing"
    );

    let base = indices[0];
    ensure!(
        tile_size > 0 && base.is_multiple_of(tile_size),
        "pattern parsing error: offset {base} is not a valid tile base for the committed pattern"
    );
    ensure!(
        indices.len() == tile_size as usize,
        "pattern parsing error: opened indices do not match the tile cardinality"
    );
    ensure!(
        u64::from(indices[indices.len() - 1]) == u64::from(base) + u64::from(tile_size) - 1,
        "pattern parsing error: opened indices do not match the committed tile at base {base}"
    );

    Ok(base)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_assignment_default_layout() {
        let lanes = lane_assignment(4, 128);
        assert_eq!(lanes.len(), JACKPOT_ENTRIES);
        assert!(lanes.iter().all(|l| l.len() == 32));
        assert_eq!(lanes[0], (0..32).collect::<Vec<usize>>());
        assert_eq!(lanes[15], (480..512).collect::<Vec<usize>>());
        let all: Vec<usize> = lanes.into_iter().flatten().collect();
        assert_eq!(all, (0..512).collect::<Vec<usize>>());
    }

    #[test]
    fn lottery_layout_envelope() {
        for (h, w) in [(4, 64), (4, 128), (16, 20), (8, 34), (128, 16)] {
            check_lottery_layout(h, w).unwrap();
        }
        for (h, w) in [(3, 128), (32, 8), (4, 16), (5, 53), (128, 32), (usize::MAX, 16)] {
            assert!(check_lottery_layout(h, w).is_err(), "accepted {h}x{w}");
        }
    }

    #[test]
    fn parse_axis_requires_aligned_consecutive_tiles() {
        assert_eq!(parse_axis(&[0, 1, 2, 3], 4).unwrap(), 0);
        assert_eq!(parse_axis(&[8, 9, 10, 11], 4).unwrap(), 8);
        assert_eq!(
            parse_axis(&[u32::MAX - 3, u32::MAX - 2, u32::MAX - 1, u32::MAX], 4).unwrap(),
            u32::MAX - 3
        );
        for indices in [
            &[][..],
            &[0, 1, 2],
            &[0, 1, 2, 3, 4],
            &[1, 2, 3, 4],
            &[4, 5, 6, 8],
            &[3, 2, 1, 0],
            &[0, 1, 1, 2],
        ] {
            assert!(parse_axis(indices, 4).is_err(), "{indices:?}");
        }
        assert!(parse_axis(&[0], 0).is_err());
    }
}
