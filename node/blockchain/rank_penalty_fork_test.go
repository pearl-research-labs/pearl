// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"encoding/hex"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// These tests cover when the rank-penalty rule is applied. The rule itself (the
// rank floor and the rank-scaled jackpot bound) lives in Rust and is tested in
// zk-pow/src/api/sanity_checks.rs and node/zkpow/rank_penalty_test.go, since it
// needs the proof verification backend.

const rankPenaltyTestHeight = int32(10)

func rankPenaltyParams(net wire.PearlNet) *chaincfg.Params {
	return &chaincfg.Params{
		Net:                   net,
		MoEForkHeight:         1,
		RankPenaltyForkHeight: rankPenaltyTestHeight,
	}
}

// TestCheckCertificateRulesRankPenalty covers when the rank-penalty rule is
// applied: at and after the activation height, and only where proof of work is
// verified.
func TestCheckCertificateRulesRankPenalty(t *testing.T) {
	header := &wire.BlockHeader{Bits: 0x1e01ffff}
	// A dense V2 certificate whose public data is all zeros: valid in length but
	// not a rank the rule accepts, so reaching the rule means rejection.
	denseCert := &wire.CertificateV2{PublicDataLen: wire.PublicDataSizeDenseV2}
	regTest := rankPenaltyParams(wire.RegTest)

	t.Run("BeforeActivation", func(t *testing.T) {
		require.NoError(t, CheckCertificateRules(header, denseCert,
			rankPenaltyTestHeight-1, regTest, BFNone))
	})

	t.Run("NoPoWCheck", func(t *testing.T) {
		require.NoError(t, CheckCertificateRules(header, denseCert,
			rankPenaltyTestHeight, regTest, BFNoPoWCheck))
	})

	t.Run("SimNet", func(t *testing.T) {
		// SimNet is skipped by the flag its parameters imply, not by the rule
		// inspecting the network.
		simNet := rankPenaltyParams(wire.SimNet)
		require.NoError(t, CheckCertificateRules(header, denseCert,
			rankPenaltyTestHeight, simNet, NetBehaviorFlags(simNet)))
	})

	t.Run("BlockTemplate", func(t *testing.T) {
		// mining.NewBlockTemplate attaches a placeholder certificate with no
		// public data. Templates are checked with BFNoPoWCheck; without it
		// getblocktemplate would fail from the activation height on.
		placeholder := &wire.CertificateV2{}
		require.NoError(t, CheckCertificateRules(header, placeholder,
			rankPenaltyTestHeight, regTest, BFNoPoWCheck))
		requireRuleError(t, CheckCertificateRules(header, placeholder,
			rankPenaltyTestHeight, regTest, BFNone), ErrRankPenalty)
	})

	t.Run("V1Certificate", func(t *testing.T) {
		// V1 certificates have no noise rank, and the version rule rejects them
		// at this height before the rank-penalty rule is reached.
		requireRuleError(t, CheckCertificateRules(header, &wire.CertificateV1{},
			rankPenaltyTestHeight, regTest, BFNone), ErrDisallowedCertVersion)
	})

	t.Run("AtActivation", func(t *testing.T) {
		// In non-zkpow builds the zkpow.CheckRankPenalty stub returns a generic
		// error that still maps to ErrRankPenalty, so this gates activation, not
		// the rule itself; the rule is exercised under the zkpow build tag in
		// node/zkpow/rank_penalty_test.go.
		requireRuleError(t, CheckCertificateRules(header, denseCert,
			rankPenaltyTestHeight, regTest, BFNone), ErrRankPenalty)
	})

	t.Run("MissingCertificate", func(t *testing.T) {
		requireRuleError(t, CheckCertificateRules(header, nil,
			rankPenaltyTestHeight, regTest, BFNone), ErrCertificateMissing)
	})
}

// TestRankPenaltyForkAcceptsTemplatesAndBlocks builds a SimNet chain across the
// activation height. SimNet does not verify proofs and SolveBlock emits
// placeholder certificates there, so enabling the fork must not start rejecting
// blocks or templates. Guards against the template-generation breakage the
// dense-only fork once caused.
func TestRankPenaltyForkAcceptsTemplatesAndBlocks(t *testing.T) {
	params := moeSimNetParams()
	params.RankPenaltyForkHeight = moeForkTestHeight + 2
	chain, teardown, err := chainSetup("rank_penalty_template", &params)
	require.NoError(t, err)
	defer teardown()

	tip := btcutil.NewBlock(chain.chainParams.GenesisBlock)
	tip.SetHeight(0)

	for h := int32(1); h <= params.RankPenaltyForkHeight+1; h++ {
		block, _, err := addBlock(chain, tip, nil)
		require.NoErrorf(t, err,
			"block at height %d must be accepted after the rank-penalty fork", h)
		tip = block
	}

	template, _, err := newBlock(chain, tip, nil)
	require.NoError(t, err)
	require.NoError(t, chain.CheckConnectBlockTemplate(template),
		"post-fork block template must be accepted")
}

// TestTestNetRankPenaltyReschedule embeds the certificate and header bits of
// canonical testnet block 36761 (hash
// 3ea775de59c8c812c4290fa2e4cca47a6962572d1514c738702b7bdeb516ba9e), which a
// pre-fork build mined with a rank-64 certificate. Under the rescheduled
// TestNetParams.RankPenaltyForkHeight the certificate rules must accept it at
// the two heights the chain contains, while the same certificate remains a
// rank-penalty violation from 36763 on. Guards against rescheduling the fork
// onto the violating blocks again.
func TestTestNetRankPenaltyReschedule(t *testing.T) {
	publicData, err := hex.DecodeString("0020000040000000070103010000000103" +
		"0101030000000000000000000000000000000000000000000000000000000000000000" +
		"c269a4df54b936913d91d20eb138b69b4d08e7f9624400ea5064083743db6d3665" +
		"3b23b547f2b6e172398da78efa9796b0e4ecccdc8c3b3395708ed4bfc606262d" +
		"6c1902f88bed3a15e92776b55bc3fe565fa0449b278f188064dd000000000000" +
		"08000000280000935e000004070000")
	require.NoError(t, err)
	require.Len(t, publicData, wire.PublicDataSizeDenseV2)

	header := &wire.BlockHeader{Bits: 0x1948d970}
	cert := &wire.CertificateV2{PublicDataLen: wire.PublicDataSizeDenseV2}
	copy(cert.PublicData[:], publicData)

	// The block must satisfy the certificate rules at the two heights the
	// canonical chain holds: the dense-only and version rules still apply
	// there, only the rank check is inactive.
	require.NoError(t, CheckCertificateRules(header, cert, 36761,
		&chaincfg.TestNetParams, BFNone))
	require.NoError(t, CheckCertificateRules(header, cert, 36762,
		&chaincfg.TestNetParams, BFNone))

	// From the rescheduled activation height the same certificate is a
	// rank-penalty violation (in non-zkpow builds the CheckRankPenalty stub's
	// generic error still maps to ErrRankPenalty).
	requireRuleError(t, CheckCertificateRules(header, cert, 36763,
		&chaincfg.TestNetParams, BFNone), ErrRankPenalty)
}

// TestNewRejectsRankPenaltyBeforeMoEFork pins the params invariant the rule
// depends on: only V2 certificates carry a noise rank.
func TestNewRejectsRankPenaltyBeforeMoEFork(t *testing.T) {
	params := moeSimNetParams()
	params.RankPenaltyForkHeight = params.MoEForkHeight - 1
	_, _, err := chainSetup("rank_penalty_ordering", &params)
	require.Error(t, err)
	require.Contains(t, err.Error(), "RankPenaltyForkHeight")
}
