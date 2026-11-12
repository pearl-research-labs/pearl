//! Consensus trace padding for the supported native FRI heights.

// Keeping every stop from 18 through 22 avoids padding large tables; sparse lower
// stops reduce the inner verifier's FRI boundaries. Traces, LUTs and known-column
// evaluation all round up through this shared helper.
pub const V4_REACHABLE_DEGREE_BITS: [usize; 9] = [22, 21, 20, 19, 18, 16, 13, 10, 7];

/// Smallest supported height covering the live rows; no extra degree increase.
pub(crate) fn checked_trace_padded_height(rows: u64) -> Option<u64> {
    V4_REACHABLE_DEGREE_BITS
        .iter()
        .rev()
        .map(|&bits| 1u64 << bits)
        .find(|&height| rows <= height)
}

pub(crate) fn trace_padded_height(rows: usize) -> usize {
    checked_trace_padded_height(rows as u64).expect("trace exceeds the supported degree ladder") as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_up_to_next_supported_degree() {
        assert_eq!(trace_padded_height(0), 1 << 7);
        for natural in 0..=22 {
            let rows = 1usize << natural;
            let expected = *V4_REACHABLE_DEGREE_BITS.iter().rev().find(|&&d| d >= natural).unwrap();
            assert_eq!(trace_padded_height(rows), 1 << expected);
            assert_eq!(trace_padded_height(rows.saturating_sub(1)), 1 << expected);
            if natural < 22 {
                let next = *V4_REACHABLE_DEGREE_BITS.iter().rev().find(|&&d| d > natural).unwrap();
                assert_eq!(trace_padded_height(rows + 1), 1 << next);
            }
        }
        assert_eq!(checked_trace_padded_height((1 << 22) + 1), None);
        assert_eq!(checked_trace_padded_height(u64::MAX), None);
    }
}
