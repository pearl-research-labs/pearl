//! Public parameters and bounds for the FP16 (A100) scheme.
//!
//! A standalone parameter type, deliberately not an arm of the FP8
//! `Device`/`Quant` enums: the FP16 scheme is a parallel scheme, and wire-level
//! unification of the two under one consensus `Device` enum is a later
//! integration step (it would force the ZK circuit layer to handle A100 before
//! its STARK exists). The constants here fix the plaintext scheme.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

/// The committed device for this scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(eq, eq_int))]
pub enum Fp16Device {
    /// NVIDIA A100 (GA100, `sm_80`).
    A100 = 0,
}

impl Fp16Device {
    /// The committed wire tag (folded into the noise-seed public-param encoding).
    pub const fn wire_tag(self) -> u8 {
        self as u8
    }
}

impl Fp16Device {
    /// Per-row relative noise weight `delta`.
    pub const fn delta(self) -> f64 {
        match self {
            Fp16Device::A100 => 0.5,
        }
    }
    /// Internal accumulator precision (FP32 significand bits).
    pub const fn window_bits(self) -> u32 {
        match self {
            Fp16Device::A100 => 24,
        }
    }
    /// Products per hardware accumulation group.
    pub const fn group(self) -> usize {
        match self {
            Fp16Device::A100 => 8,
        }
    }
}

/// The fixed noise rank `r`.
pub const NOISE_RANK: usize = 32;

/// The recursive ZK wrapper's fold-ladder top height (`2^16`,
/// `circuit::fp16::driver::FP16_REACHABLE_DEGREE_BITS[0]`). The current wrapper is
/// one-group-per-row, so every batch table's live height must fit under this; the
/// fuller envelope awaits the deferred groups-per-row packing. Kept here (api) as a
/// closed-form mirror so a geometry the wrapper cannot prove is rejected before any
/// circuit work; keep in sync with the circuit ladder.
pub const WRAPPER_LADDER_TOP: usize = 1 << 16;

/// Public parameters of one FP16 lottery tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp16Params {
    pub device: Fp16Device,
    /// Rows of `A` in the tile (`|I_A|`).
    pub h: usize,
    /// Rows of `B` in the tile (`|I_B|`), i.e. output-tile columns.
    pub w: usize,
    /// Inner dimension.
    pub k: usize,
    /// Noise rank.
    pub r: usize,
}

impl Fp16Params {
    /// Validates the tile bounds (the `CheckPublic` analogue). Mirrors the
    /// whitepaper's admissible ranges, with the inner dimension a multiple of
    /// the device group size.
    pub fn validate(&self) -> Result<()> {
        ensure!(self.r == NOISE_RANK, "noise rank must be {NOISE_RANK}");
        ensure!(self.k > 0 && self.k % self.device.group() == 0, "k must be a positive multiple of the group size");
        ensure!(self.h >= 4, "|I_A| must be >= 4");
        ensure!(self.w >= 16, "|I_B| must be >= 16");
        let cells = self.h.checked_mul(self.w).expect("tile area overflow");
        ensure!((256..=2048).contains(&cells), "tile area |I_A|*|I_B| must be in [256, 2048]");
        ensure!(self.k.checked_mul(self.h + self.w).map(|x| x <= 1 << 22).unwrap_or(false), "k*(|I_A|+|I_B|) must be <= 2^22");

        // Wrapper-legal envelope: every batch table's live height must fit under the
        // one-group-per-row wrapper's 2^16 ladder top (see [`WRAPPER_LADDER_TOP`]), or
        // the tile cannot be ZK-proved/verified. These mirror the circuit table-height
        // formulas in `circuit::fp16::driver` (matmul/policy `h*w*k/g`; quant chain
        // `(h+w)*k`; per-side noise matmul `max(h,w)*k*(r/g)`; NoiseStark `(h+w+2k)*r`),
        // with `r == NOISE_RANK` and `g` the device group; the noise-BLAKE3 table
        // (`8*(h+w+2k+3)`) is looser than NoiseStark and thus implied.
        let (h, w, k, r, g) = (self.h, self.w, self.k, self.r, self.device.group());
        let top = WRAPPER_LADDER_TOP;
        let fits = |live: Option<usize>| live.map(|x| x <= top).unwrap_or(false);
        ensure!(
            fits(h.checked_mul(w).and_then(|x| x.checked_mul(k)).map(|x| x / g)),
            "tile exceeds the FP16 wrapper matmul height (|I_A|*|I_B|*k/{g} <= 2^16)"
        );
        ensure!(
            fits((h + w).checked_mul(k)),
            "tile exceeds the FP16 wrapper quant height ((|I_A|+|I_B|)*k <= 2^16)"
        );
        ensure!(
            fits(h.max(w).checked_mul(k).and_then(|x| x.checked_mul(r)).map(|x| x / g)),
            "tile exceeds the FP16 wrapper noise-matmul height (max(|I_A|,|I_B|)*k*r/{g} <= 2^16)"
        );
        ensure!(
            fits((h + w).checked_add(2 * k).and_then(|s| s.checked_mul(r))),
            "tile exceeds the FP16 wrapper NoiseStark height ((|I_A|+|I_B|+2k)*r <= 2^16)"
        );

        // The jackpot lottery extracts exactly 16 lanes as a `br x bc` Blake split of
        // the tile (`br | h`, `bc | w`, `br*bc = 16`). A tile admitting no such split
        // cannot produce a ticket, so require at least one factorization of 16 to
        // divide the two tile dims.
        const LANES: usize = 16;
        ensure!(
            (1..=LANES).filter(|br| LANES % br == 0).any(|br| h % br == 0 && w % (LANES / br) == 0),
            "tile admits no 16-lane Blake split (need br|{h}, bc|{w}, br*bc=16)"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a100_constants() {
        assert_eq!(Fp16Device::A100.delta(), 0.5);
        assert_eq!(Fp16Device::A100.window_bits(), 24);
        assert_eq!(Fp16Device::A100.group(), 8);
    }

    #[test]
    fn bounds() {
        let ok = Fp16Params { device: Fp16Device::A100, h: 4, w: 64, k: 256, r: 32 };
        ok.validate().unwrap();
        // k not a multiple of 8.
        assert!(Fp16Params { k: 100, ..ok }.validate().is_err());
        // tile too small.
        assert!(Fp16Params { h: 4, w: 16, ..ok }.validate().is_err());
        // wrong rank.
        assert!(Fp16Params { r: 16, ..ok }.validate().is_err());
    }

    #[test]
    fn wrapper_legal_envelope() {
        let ok = Fp16Params { device: Fp16Device::A100, h: 4, w: 64, k: 256, r: 32 };
        ok.validate().unwrap();
        // k=256 is the largest wrapper-legal k for w=64 (noise matmul max(h,w)*k*(r/g)
        // = 64*256*4 = 2^16, exactly the ladder top). k=512 overflows it.
        let err = Fp16Params { k: 512, ..ok }.validate().unwrap_err();
        assert!(format!("{err:#}").contains("wrapper"), "got: {err:#}");
        // A large-k tile that passed the old k*(h+w) <= 2^22 bound but blows the
        // one-group-per-row wrapper (h*w*k/8 = 2^17 > 2^16) is now rejected.
        let err = Fp16Params { k: 4096, ..ok }.validate().unwrap_err();
        assert!(format!("{err:#}").contains("wrapper"), "got: {err:#}");
        // A narrow tile reaches larger k (w=16): noise matmul 16*k*4 <= 2^16 -> k <= 1024,
        // but NoiseStark (h+w+2k)*r <= 2^16 -> h+w+2k <= 2048 binds k <= ~990.
        Fp16Params { device: Fp16Device::A100, h: 16, w: 16, k: 960, r: 32 }.validate().unwrap();
        assert!(Fp16Params { device: Fp16Device::A100, h: 16, w: 16, k: 1024, r: 32 }.validate().is_err());
    }
}
