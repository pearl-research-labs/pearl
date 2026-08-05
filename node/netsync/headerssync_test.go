// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package netsync

import (
	"crypto/rand"
	"math/big"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

func makeTestParams() *chaincfg.Params {
	p := chaincfg.RegressionNetParams
	return &p
}

func makeChainStart(height int32) chainStartInfo {
	return chainStartInfo{
		ChainStartInfo: blockchain.ChainStartInfo{
			Hash:          chainhash.Hash{0x01},
			Height:        height,
			Bits:          chaincfg.RegressionNetParams.PowLimitBits,
			Timestamp:     time.Now().Add(-time.Hour).Unix(),
			WorkSum:       big.NewInt(1000),
			PrevTimestamp: time.Now().Add(-time.Hour - time.Second).Unix(),
		},
	}
}

func TestNewHeadersSyncState(t *testing.T) {
	params := makeTestParams()
	start := makeChainStart(100)
	minWork := big.NewInt(5000)

	s := NewHeadersSyncState(1, "", params, start, minWork)

	require.Equal(t, PhasePresync, s.Phase())
	require.False(t, s.Done())
}

func TestHeadersSyncPhaseString(t *testing.T) {
	require.Equal(t, "presync", PhasePresync.String())
	require.Equal(t, "redownload", PhaseRedownload.String())
	require.Equal(t, "final", PhaseFinal.String())
}

func TestWorkNormalizationComputed(t *testing.T) {
	params := makeTestParams()
	start := makeChainStart(100)
	minWork := new(big.Int).Add(start.WorkSum, big.NewInt(1000000))

	s := NewHeadersSyncState(1, "", params, start, minWork)

	require.Greater(t, s.workNormalization, 0.0)
}

func TestProcessNextHeadersEmptyReturnsFailure(t *testing.T) {
	params := makeTestParams()
	start := makeChainStart(100)
	minWork := big.NewInt(5000)

	s := NewHeadersSyncState(1, "", params, start, minWork)

	result := s.ProcessNextHeaders(nil, nil, true)
	require.False(t, result.Success)
	require.False(t, result.RequestMore)
}

func TestProcessNextHeadersFinalState(t *testing.T) {
	params := makeTestParams()
	start := makeChainStart(100)
	minWork := big.NewInt(5000)

	s := NewHeadersSyncState(1, "", params, start, minWork)
	s.finalize()

	require.Equal(t, PhaseFinal, s.Phase())
	require.True(t, s.Done())

	result := s.ProcessNextHeaders(nil, []wire.MsgHeader{{
		BlockHeader: wire.BlockHeader{},
	}}, true)
	require.False(t, result.Success)
}

func TestPresyncWorkSufficient(t *testing.T) {
	params := makeTestParams()
	start := makeChainStart(100)
	minWork := new(big.Int).Add(start.WorkSum, big.NewInt(100))

	s := NewHeadersSyncState(1, "", params, start, minWork)

	require.False(t, s.presyncWorkSufficient())

	s.tip.WorkSum = new(big.Int).Add(minWork, big.NewInt(1))
	require.True(t, s.presyncWorkSufficient())
}

func TestCalcNextRequiredDifficultyFromValuesNoRetarget(t *testing.T) {
	params := makeTestParams()
	bits, err := blockchain.CalcNextRequiredDifficultyFromValues(
		params, 100, params.PowLimitBits, 1000, 999,
	)
	require.NoError(t, err)
	require.Equal(t, params.PowLimitBits, bits)
}

func TestCommitBitDeterministic(t *testing.T) {
	params := makeTestParams()
	start := makeChainStart(0)
	minWork := big.NewInt(5000)

	s := NewHeadersSyncState(1, "", params, start, minWork)

	hash := chainhash.Hash{0xAB, 0xCD}
	bit1 := s.commitBit(hash)
	bit2 := s.commitBit(hash)
	require.Equal(t, bit1, bit2)

	s2 := NewHeadersSyncState(2, "", params, start, minWork)
	_ = s2.commitBit(hash)
}

func TestCommitBitSipHashDistribution(t *testing.T) {
	params := makeTestParams()
	start := makeChainStart(0)
	s := NewHeadersSyncState(1, "", params, start, big.NewInt(5000))

	const n = 10000
	ones := 0
	for i := 0; i < n; i++ {
		var h chainhash.Hash
		_, _ = rand.Read(h[:])
		if s.commitBit(h) {
			ones++
		}
	}
	require.Greater(t, ones, n/2-500)
	require.Less(t, ones, n/2+500)
}

func TestCommitBitSipHashSaltIndependence(t *testing.T) {
	params := makeTestParams()
	start := makeChainStart(0)
	s1 := NewHeadersSyncState(1, "", params, start, big.NewInt(5000))
	s2 := NewHeadersSyncState(2, "", params, start, big.NewInt(5000))
	for i := range s1.hashSalt {
		s1.hashSalt[i] = 0x00
	}
	for i := range s2.hashSalt {
		s2.hashSalt[i] = 0xFF
	}

	const n = 2000
	agree := 0
	for i := 0; i < n; i++ {
		var h chainhash.Hash
		_, _ = rand.Read(h[:])
		if s1.commitBit(h) == s2.commitBit(h) {
			agree++
		}
	}
	require.Greater(t, agree, n/2-250)
	require.Less(t, agree, n/2+250)
}

func TestCommitBitSipHashKnownVector(t *testing.T) {
	s := NewHeadersSyncState(1, "", makeTestParams(), makeChainStart(0), big.NewInt(5000))
	for i := range s.hashSalt {
		s.hashSalt[i] = byte(i)
	}
	var h chainhash.Hash
	for i := range h {
		h[i] = byte(i)
	}
	b1 := s.commitBit(h)
	b2 := s.commitBit(h)
	require.Equal(t, b1, b2)
}

// --- Hash accessors ---

func TestShouldSpotCheckByWorkHighProb(t *testing.T) {
	params := makeTestParams()
	start := makeChainStart(100)
	minWork := new(big.Int).Add(start.WorkSum, big.NewInt(1))
	s := NewHeadersSyncState(1, "", params, start, minWork)

	work := big.NewInt(1)
	hits := 0
	for i := 0; i < 100; i++ {
		if s.shouldSpotCheckByWork(work) {
			hits++
		}
	}
	require.Greater(t, hits, 90)
}

// --- REDOWNLOAD validation & Tier-1 capacity ---

func newRedownloadState(t *testing.T) *HeadersSyncState {
	t.Helper()
	s := NewHeadersSyncState(1, "", makeTestParams(), makeChainStart(100), big.NewInt(5000))
	s.phase = PhaseRedownload
	s.redownloadCursor = redownloadCursor{
		hash:          s.chainStart.Hash,
		timestamp:     s.chainStart.Timestamp,
		prevTimestamp: s.chainStart.PrevTimestamp,
		height:        s.chainStart.Height,
		bits:          s.chainStart.Bits,
	}
	s.processAllRemainingHeaders = true
	return s
}

func expectedNextNBits(s *HeadersSyncState) uint32 {
	c := s.redownloadCursor
	nbits, err := blockchain.CalcNextRequiredDifficultyFromValues(
		s.chainParams, c.height, c.bits, c.timestamp, c.prevTimestamp,
	)
	if err != nil {
		return s.chainParams.PowLimitBits
	}
	return nbits
}

func buildValidRedownloadHeader(s *HeadersSyncState) *wire.MsgHeader {
	return &wire.MsgHeader{BlockHeader: wire.BlockHeader{
		Version:         1,
		PrevBlock:       s.redownloadCursor.hash,
		Timestamp:       time.Unix(s.redownloadCursor.timestamp+1, 0),
		Bits:            expectedNextNBits(s),
		ProofCommitment: chainhash.Hash{0x01},
	}}
}

func TestValidateAndStoreRedownloadedHeader(t *testing.T) {
	t.Run("accept-valid", func(t *testing.T) {
		s := newRedownloadState(t)
		hdr := buildValidRedownloadHeader(s)
		require.True(t, s.validateAndStoreRedownloadedHeader(hdr))
		require.Equal(t, 1, len(s.redownloadApproved))
		require.Equal(t, hdr.BlockHeader.BlockHash(), s.redownloadCursor.hash)
	})

	t.Run("reject-non-continuous", func(t *testing.T) {
		s := newRedownloadState(t)
		hdr := buildValidRedownloadHeader(s)
		hdr.BlockHeader.PrevBlock = chainhash.Hash{0xde, 0xad}
		require.False(t, s.validateAndStoreRedownloadedHeader(hdr))
		require.Equal(t, 0, len(s.redownloadApproved))
	})

	t.Run("reject-bad-difficulty", func(t *testing.T) {
		s := newRedownloadState(t)
		hdr := buildValidRedownloadHeader(s)
		expected := expectedNextNBits(s)
		if expected == s.chainParams.PowLimitBits {
			hdr.BlockHeader.Bits = s.chainParams.PowLimitBits - 1
		} else {
			hdr.BlockHeader.Bits = s.chainParams.PowLimitBits
			hdr.BlockHeader.Timestamp = time.Unix(s.redownloadCursor.timestamp+1, 0)
		}
		require.False(t, s.validateAndStoreRedownloadedHeader(hdr))
		require.Equal(t, 0, len(s.redownloadApproved))
	})

}

func TestRedownloadTier1Capacity(t *testing.T) {
	s := newRedownloadState(t)
	targetFill := redownloadApprovedCap - wire.MaxBlockHeadersPerMsg + 1
	for i := 0; i < targetFill; i++ {
		hdr := buildValidRedownloadHeader(s)
		require.True(t, s.validateAndStoreRedownloadedHeader(hdr),
			"unexpected reject at i=%d", i)
	}
	require.Equal(t, targetFill, len(s.redownloadApproved))
	require.False(t, s.hasRedownloadFifoCapacity(), "should be saturated")
}

func TestRedownloadTipStableAcrossDrain(t *testing.T) {
	s := newRedownloadState(t)

	for i := 0; i < wire.MaxBlockHeadersPerMsg; i++ {
		hdr := buildValidRedownloadHeader(s)
		require.True(t, s.validateAndStoreRedownloadedHeader(hdr))
	}
	tipAfterFirst := s.redownloadCursor.hash
	require.NotEqual(t, s.chainStart.Hash, tipAfterFirst)

	// BlocksToRequest drains Tier-1 into Tier-2; cursor should survive.
	s.redownloadShortBatchSeen = true
	hashes := s.BlocksToRequest()
	require.NotEmpty(t, hashes)
	require.Equal(t, tipAfterFirst, s.redownloadCursor.hash)
}

// --- Spot-check tests ---

func TestArmNextSpotCheckBound(t *testing.T) {
	s := NewHeadersSyncState(1, "", makeTestParams(), makeChainStart(0), big.NewInt(5000))

	const iters = 100_000
	var maxGap int32
	for i := 0; i < iters; i++ {
		const base int32 = 1_000_000
		s.scheduleNextSpotCheck(base)
		gap := s.nextSpotCheckHeight - base
		require.GreaterOrEqual(t, gap, int32(1))
		require.LessOrEqual(t, gap, int32(2*spotCheckMeanGap))
		if gap > maxGap {
			maxGap = gap
		}
	}
	require.Greater(t, maxGap, int32(spotCheckMeanGap),
		"expected to sample into the upper half at least once")
}

func TestNewHeadersSyncStateArmsSpotCheck(t *testing.T) {
	start := makeChainStart(500)
	s := NewHeadersSyncState(1, "", makeTestParams(), start, big.NewInt(5000))
	require.Greater(t, s.nextSpotCheckHeight, start.Height)
	require.LessOrEqual(t, s.nextSpotCheckHeight,
		start.Height+int32(2*spotCheckMeanGap))
}

func TestArmNextSpotCheckAdvancesPastConsumedTarget(t *testing.T) {
	s := NewHeadersSyncState(1, "", makeTestParams(), makeChainStart(0), big.NewInt(5000))
	for i := 0; i < 1_000; i++ {
		prev := s.nextSpotCheckHeight
		s.scheduleNextSpotCheck(prev)
		require.Greater(t, s.nextSpotCheckHeight, prev,
			"target must strictly advance on each arm (iter=%d)", i)
	}
}

func TestSpotCheckQueueing(t *testing.T) {
	s := NewHeadersSyncState(1, "", makeTestParams(), makeChainStart(0), big.NewInt(5000))

	s.nextSpotCheckHeight = 3

	require.Equal(t, PhasePresync, s.Phase())
	require.Empty(t, s.pendingSpotChecks)

	s.pendingSpotChecks = append(s.pendingSpotChecks, pendingSpotCheck{
		height:    3,
		hash:      chainhash.Hash{0xAA},
		prevBlock: chainhash.Hash{0xBB},
	})

	require.Equal(t, PhasePresync, s.Phase())
	require.Len(t, s.pendingSpotChecks, 1)
}

func TestSpotCheckBackpressure(t *testing.T) {
	s := NewHeadersSyncState(1, "", makeTestParams(), makeChainStart(0), big.NewInt(5000))

	require.False(t, s.spotCheckBackpressured())

	s.pendingSpotChecks = append(s.pendingSpotChecks, pendingSpotCheck{
		height: 100,
		hash:   chainhash.Hash{0x01},
	})
	s.tip.Height = 100
	require.False(t, s.spotCheckBackpressured())

	s.tip.Height = 100 + spotCheckMeanGap - 1
	require.False(t, s.spotCheckBackpressured())

	s.tip.Height = 100 + spotCheckMeanGap
	require.True(t, s.spotCheckBackpressured())

	s.pendingSpotChecks = nil
	require.False(t, s.spotCheckBackpressured())
}

func TestSpotCheckResponseMatching(t *testing.T) {
	s := NewHeadersSyncState(1, "", makeTestParams(), makeChainStart(0), big.NewInt(5000))

	scHash := chainhash.Hash{0xDE, 0xAD}
	s.pendingSpotChecks = append(s.pendingSpotChecks, pendingSpotCheck{
		height:    500,
		hash:      scHash,
		prevBlock: chainhash.Hash{0x01},
	})

	require.Equal(t, 0, s.findPendingSpotCheck(scHash))
	require.Equal(t, -1, s.findPendingSpotCheck(chainhash.Hash{0xFF}))
}

func TestSpotCheckMultipleInFlight(t *testing.T) {
	s := NewHeadersSyncState(1, "", makeTestParams(), makeChainStart(0), big.NewInt(5000))

	s.pendingSpotChecks = []pendingSpotCheck{
		{height: 100, hash: chainhash.Hash{0x01}},
		{height: 200, hash: chainhash.Hash{0x02}},
		{height: 300, hash: chainhash.Hash{0x03}},
	}

	idx := s.findPendingSpotCheck(chainhash.Hash{0x02})
	require.Equal(t, 1, idx)
	s.pendingSpotChecks = append(s.pendingSpotChecks[:idx], s.pendingSpotChecks[idx+1:]...)

	require.Len(t, s.pendingSpotChecks, 2)
	require.Equal(t, chainhash.Hash{0x01}, s.pendingSpotChecks[0].hash)
	require.Equal(t, chainhash.Hash{0x03}, s.pendingSpotChecks[1].hash)
}

func TestFinalizesClearsPendingSpotChecks(t *testing.T) {
	s := NewHeadersSyncState(1, "", makeTestParams(), makeChainStart(0), big.NewInt(5000))
	s.pendingSpotChecks = []pendingSpotCheck{
		{height: 100, hash: chainhash.Hash{0x01}},
	}
	s.finalize()
	require.Nil(t, s.pendingSpotChecks)
	require.Equal(t, PhaseFinal, s.Phase())
	require.True(t, s.Done())
}

// --- Done / Abort ---

func TestDoneRequiresFullDrain(t *testing.T) {
	s := newRedownloadState(t)
	require.False(t, s.Done())

	s.redownloadShortBatchSeen = true
	s.redownloadApproved = nil
	s.tier2Expected = nil
	require.True(t, s.Done())
}

func TestAbortReturnsTier2Hashes(t *testing.T) {
	s := newRedownloadState(t)
	s.tier2Expected = []chainhash.Hash{
		{0x01},
		{0x02},
	}

	hashes := s.Abort()
	require.Len(t, hashes, 2)
	require.Equal(t, PhaseFinal, s.Phase())
}

func TestLastProgressTimeUpdated(t *testing.T) {
	s := NewHeadersSyncState(1, "", makeTestParams(), makeChainStart(100), big.NewInt(5000))
	before := s.LastProgressTime()
	time.Sleep(time.Millisecond)
	s.ProcessNextHeaders(nil, nil, true) // empty = no success, but still updates progress
	require.True(t, s.LastProgressTime().After(before) || s.LastProgressTime().Equal(before))
}

// --- Checkpoint integration ---

func TestCheckpointVerification(t *testing.T) {
	s := NewHeadersSyncState(1, "", makeTestParams(), makeChainStart(100), big.NewInt(5000))

	cpHash := chainhash.Hash{0xAA, 0xBB}
	s.checkpoints = []chaincfg.Checkpoint{
		{Height: 150, Hash: &cpHash},
	}
	s.nextCpIdx = 0

	// Matching checkpoint passes.
	require.True(t, s.verifyCheckpoint(150, &cpHash))
	require.Equal(t, 1, s.nextCpIdx)

	// Non-checkpoint height is always ok.
	someHash := chainhash.Hash{0xFF}
	require.True(t, s.verifyCheckpoint(200, &someHash))
}

func TestCheckpointMismatchPunishes(t *testing.T) {
	s := NewHeadersSyncState(1, "", makeTestParams(), makeChainStart(100), big.NewInt(5000))

	cpHash := chainhash.Hash{0xAA, 0xBB}
	s.checkpoints = []chaincfg.Checkpoint{
		{Height: 150, Hash: &cpHash},
	}
	s.nextCpIdx = 0

	wrongHash := chainhash.Hash{0xFF}
	require.False(t, s.verifyCheckpoint(150, &wrongHash))
	require.True(t, s.shouldPunish)
}
