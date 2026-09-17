use serde::{Deserialize, Serialize};

use pearl_blake3::BLAKE3_MSG_LEN;

/// Tweak parameters for BLAKE3 compression.
#[derive(Serialize, Deserialize, Clone, Debug, Copy)]
pub struct Blake3Tweak {
    // stored little-endian, in order of appearance.
    pub(crate) counter_low: u32,
    pub(crate) counter_high: u16, // bits 32..48 of the 64-bit counter
    pub(crate) block_len: u32,    // actual block length (1-64 bytes), or 0 for empty input
    pub(crate) flags: u32,        // u8 flags widened to u32 to match the compression state word size
}

impl Default for Blake3Tweak {
    fn default() -> Self {
        Self {
            counter_low: 0,
            counter_high: 0,
            block_len: BLAKE3_MSG_LEN as u32,
            flags: 0,
        }
    }
}

pub(crate) const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

pub(crate) const BLAKE3_MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// Apply BLAKE3_MSG_PERMUTATION in-place.
#[inline(always)]
pub(crate) fn blake3_permute_msg<T: Copy>(msg: &mut [T; 16]) {
    // Cycle 1: 0→2→3→10→12→9→11→5→0
    let t = msg[5];
    msg[5] = msg[0];
    msg[0] = msg[2];
    msg[2] = msg[3];
    msg[3] = msg[10];
    msg[10] = msg[12];
    msg[12] = msg[9];
    msg[9] = msg[11];
    msg[11] = t;

    // Cycle 2: 1→6→4→7→13→14→15→8→1
    let t = msg[8];
    msg[8] = msg[1];
    msg[1] = msg[6];
    msg[6] = msg[4];
    msg[4] = msg[7];
    msg[7] = msg[13];
    msg[13] = msg[14];
    msg[14] = msg[15];
    msg[15] = t;
}

#[inline(always)]
pub(crate) fn blake3_compress(msg: &[u8; 64], cv_in: [u8; 32], tweak: Blake3Tweak) -> [u8; 32] {
    let cv: [u32; 8] =
        std::array::from_fn(|i| u32::from_le_bytes([cv_in[i * 4], cv_in[i * 4 + 1], cv_in[i * 4 + 2], cv_in[i * 4 + 3]]));
    let m: [u32; 16] = std::array::from_fn(|i| u32::from_le_bytes([msg[i * 4], msg[i * 4 + 1], msg[i * 4 + 2], msg[i * 4 + 3]]));

    let state = blake3_internal::compress(
        &cv,
        &m,
        tweak.counter_low,
        tweak.counter_high as u32,
        tweak.block_len,
        tweak.flags,
    );

    let mut out = [0u8; 32];
    for i in 0..8 {
        out[i * 4..i * 4 + 4].copy_from_slice(&state[i].to_le_bytes());
    }
    out
}

mod blake3_internal {
    use super::BLAKE3_IV;

    #[inline(always)]
    pub fn compress(cv: &[u32; 8], m: &[u32; 16], counter_low: u32, counter_high: u32, block_len: u32, flags: u32) -> [u32; 16] {
        let mut s = [
            cv[0],
            cv[1],
            cv[2],
            cv[3],
            cv[4],
            cv[5],
            cv[6],
            cv[7],
            BLAKE3_IV[0],
            BLAKE3_IV[1],
            BLAKE3_IV[2],
            BLAKE3_IV[3],
            counter_low,
            counter_high,
            block_len,
            flags,
        ];

        macro_rules! g {
            ($s:expr, $a:expr, $b:expr, $c:expr, $d:expr, $mx:expr, $my:expr) => {
                $s[$a] = $s[$a].wrapping_add($s[$b]).wrapping_add($mx);
                $s[$d] = ($s[$d] ^ $s[$a]).rotate_right(16);
                $s[$c] = $s[$c].wrapping_add($s[$d]);
                $s[$b] = ($s[$b] ^ $s[$c]).rotate_right(12);
                $s[$a] = $s[$a].wrapping_add($s[$b]).wrapping_add($my);
                $s[$d] = ($s[$d] ^ $s[$a]).rotate_right(8);
                $s[$c] = $s[$c].wrapping_add($s[$d]);
                $s[$b] = ($s[$b] ^ $s[$c]).rotate_right(7);
            };
        }

        macro_rules! round {
            ($s:expr, $m:expr,
             $i0:literal, $i1:literal, $i2:literal, $i3:literal,
             $i4:literal, $i5:literal, $i6:literal, $i7:literal,
             $i8:literal, $i9:literal, $i10:literal, $i11:literal,
             $i12:literal, $i13:literal, $i14:literal, $i15:literal) => {
                g!($s, 0, 4, 8, 12, $m[$i0], $m[$i1]);
                g!($s, 1, 5, 9, 13, $m[$i2], $m[$i3]);
                g!($s, 2, 6, 10, 14, $m[$i4], $m[$i5]);
                g!($s, 3, 7, 11, 15, $m[$i6], $m[$i7]);
                g!($s, 0, 5, 10, 15, $m[$i8], $m[$i9]);
                g!($s, 1, 6, 11, 12, $m[$i10], $m[$i11]);
                g!($s, 2, 7, 8, 13, $m[$i12], $m[$i13]);
                g!($s, 3, 4, 9, 14, $m[$i14], $m[$i15]);
            };
        }

        // 7 rounds with precomputed message schedule
        round!(s, m, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        round!(s, m, 2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8);
        round!(s, m, 3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1);
        round!(s, m, 10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6);
        round!(s, m, 12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4);
        round!(s, m, 9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7);
        round!(s, m, 11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13);

        // Feed-forward XOR
        for i in 0..8 {
            s[i] ^= s[i + 8];
            s[i + 8] ^= cv[i];
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blake3_permute_msg() {
        let mut msg: [usize; 16] = std::array::from_fn(|i| i);
        blake3_permute_msg(&mut msg);

        assert_eq!(msg, BLAKE3_MSG_PERMUTATION);
    }
}
