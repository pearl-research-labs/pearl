//go:build zkpow

// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package zkpow

import (
	"encoding/binary"
	"math/big"
	"testing"

	"github.com/stretchr/testify/require"
)

// Dense PublicData prefix layout, mirrored from zk-pow/src/api/proof_utils.rs.
const (
	densePublicDataSize = 164
	jackpotOffset       = 116
	jackpotSize         = 32
)

// hashTileSize is h*w for the index patterns of the default mining configuration.
// A mismatch makes the bound expectations below fail rather than silently pass.
const hashTileSize = 4 * 8

// Byte offsets of the two mining configuration fields these tests vary, mirrored
// from MiningConfiguration::to_bytes in zk-pow/src/api/proof_utils.rs.
const (
	commonDimOffset = 0
	rankOffset      = 4
)

// rankPenaltyPublicData builds dense public data with a rank, common dimension, and
// jackpot. It starts from the default configuration so the index patterns keep the
// canonical encoding the Rust parser round-trips against, and patches the two fields
// under test.
func rankPenaltyPublicData(t *testing.T, rank uint16, commonDim uint32, jackpot []byte) []byte {
	t.Helper()
	config, err := defaultMiningConfig(0, 0)
	require.NoError(t, err)
	binary.LittleEndian.PutUint32(config[commonDimOffset:], commonDim)
	binary.LittleEndian.PutUint16(config[rankOffset:], rank)

	data := make([]byte, densePublicDataSize)
	copy(data, config[:])
	copy(data[jackpotOffset:jackpotOffset+jackpotSize], jackpot)
	return data
}

// compactToTarget converts a compact difficulty encoding to its target value.
// Mirrors blockchain/internal/workmath.CompactToBig, which is not importable
// from this package.
func compactToTarget(bits uint32) *big.Int {
	mantissa := int64(bits & 0x007fffff)
	exponent := uint(bits >> 24)
	if exponent <= 3 {
		return big.NewInt(mantissa >> (8 * (3 - exponent)))
	}
	return new(big.Int).Lsh(big.NewInt(mantissa), 8*(exponent-3))
}

// penalizedBound is the largest jackpot value the rank-penalty rule accepts:
// target * h * w * (k / rank) * MinNoiseRank.
func penalizedBound(bits uint32, rank uint16, commonDim uint32) *big.Int {
	reductions := int64(commonDim) / int64(rank)
	factor := big.NewInt(hashTileSize * reductions * MinNoiseRank)
	return new(big.Int).Mul(compactToTarget(bits), factor)
}

// jackpotFor encodes a value as a jackpot hash, which the rule reads as a
// little-endian 256-bit integer.
func jackpotFor(value *big.Int) []byte {
	bytes := value.Bytes()
	jackpot := make([]byte, jackpotSize)
	for i, b := range bytes {
		jackpot[len(bytes)-1-i] = b
	}
	return jackpot
}

// TestCheckRankPenaltyRankFloor rejects ranks below MinNoiseRank.
func TestCheckRankPenaltyRankFloor(t *testing.T) {
	bestPossibleJackpot := make([]byte, jackpotSize)

	for _, rank := range []uint16{MinNoiseRank / 4, MinNoiseRank / 2} {
		data := rankPenaltyPublicData(t, rank, 16*uint32(rank), bestPossibleJackpot)
		require.Errorf(t, CheckRankPenalty(DefaultNBits, data),
			"rank %d is below the minimum and must be rejected", rank)
	}

	data := rankPenaltyPublicData(t, MinNoiseRank, 16*MinNoiseRank, bestPossibleJackpot)
	require.NoError(t, CheckRankPenalty(DefaultNBits, data),
		"rank %d is the minimum and must be accepted", MinNoiseRank)
}

// TestCheckRankPenaltyBound independently computes the rank-penalized bounds,
// pinning the Rust rule to a Go implementation of the same formula. Testing each
// rank at its own bound also covers the property the fork exists for: a jackpot
// that wins at the base rank is rejected at a larger one, since an unpenalized
// comparison would accept bound+1 there.
func TestCheckRankPenaltyBound(t *testing.T) {
	// Ranks valid for this k: 16 * rank <= commonDim <= 4 * rank^2.
	const commonDim = uint32(8192)

	for _, rank := range []uint16{MinNoiseRank, 2 * MinNoiseRank, 4 * MinNoiseRank} {
		bound := penalizedBound(DefaultNBits, rank, commonDim)

		atBound := rankPenaltyPublicData(t, rank, commonDim, jackpotFor(bound))
		require.NoErrorf(t, CheckRankPenalty(DefaultNBits, atBound),
			"rank %d: a jackpot exactly at the bound must be accepted", rank)

		aboveBound := rankPenaltyPublicData(t, rank, commonDim,
			jackpotFor(new(big.Int).Add(bound, big.NewInt(1))))
		require.Errorf(t, CheckRankPenalty(DefaultNBits, aboveBound),
			"rank %d: a jackpot above the bound must be rejected", rank)
	}
}

// TestCheckRankPenaltyRejectsMalformedPublicData covers the guards on input the
// rule cannot parse: the empty slice the Go wrapper checks before indexing, a
// length the Rust parser rejects, and the zero-filled dense blob that block
// templates carry (rank 0).
func TestCheckRankPenaltyRejectsMalformedPublicData(t *testing.T) {
	for name, data := range map[string][]byte{
		"empty":     {},
		"truncated": make([]byte, densePublicDataSize-1),
		"all zero":  make([]byte, densePublicDataSize),
	} {
		t.Run(name, func(t *testing.T) {
			require.Error(t, CheckRankPenalty(DefaultNBits, data))
		})
	}
}
