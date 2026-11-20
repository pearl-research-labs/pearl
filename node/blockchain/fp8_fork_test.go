// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

const fp8ForkTestHeight = int32(4)

func fp8SimNetParams() chaincfg.Params {
	params := chaincfg.SimNetParams
	params.ReduceMinDifficulty = false
	params.MoEForkHeight = 1
	params.SaltedSeedForkHeight = 1
	params.Fp8ForkHeight = fp8ForkTestHeight
	return params
}

func v4Cert(proofData, publicData []byte) *wire.CertificateV4 {
	return &wire.CertificateV4{
		PublicData: publicData,
		ProofData:  proofData,
	}
}

func TestFp8ForkActivationAcceptsRequiredVersion(t *testing.T) {
	params := fp8SimNetParams()
	chain, teardown, err := chainSetup("fp8_fork_accept", &params)
	require.NoError(t, err)
	defer teardown()

	tip := btcutil.NewBlock(chain.chainParams.GenesisBlock)
	tip.SetHeight(0)

	for h := int32(1); h <= fp8ForkTestHeight+2; h++ {
		block, _, err := addBlock(chain, tip, nil)
		require.NoErrorf(t, err, "block at height %d should be accepted", h)

		want := wire.CertificateVersionV3
		if h >= fp8ForkTestHeight {
			want = wire.CertificateVersionV4
		}
		require.Equalf(t, want, block.MsgBlock().BlockCertificate().Version(),
			"unexpected certificate version at height %d", h)

		tip = block
	}

	require.Equal(t, fp8ForkTestHeight+2, chain.BestSnapshot().Height)
}

func TestFp8ForkRejectsWrongVersion(t *testing.T) {
	params := fp8SimNetParams()
	chain, teardown, err := chainSetup("fp8_fork_reject", &params)
	require.NoError(t, err)
	defer teardown()

	tip := btcutil.NewBlock(chain.chainParams.GenesisBlock)
	tip.SetHeight(0)

	for h := int32(1); h <= fp8ForkTestHeight-2; h++ {
		block, _, err := addBlock(chain, tip, nil)
		require.NoError(t, err)
		tip = block
	}

	bad := newBlockForcedCert(t, chain, tip, v4Cert([]byte{0x00}, nil))
	_, _, err = chain.ProcessBlock(bad, BFNone)
	requireRuleError(t, err, ErrDisallowedCertVersion)

	block, _, err := addBlock(chain, tip, nil)
	require.NoError(t, err)
	tip = block

	bad = newBlockForcedCert(t, chain, tip, v3Cert([]byte{0x00}, 0))
	_, _, err = chain.ProcessBlock(bad, BFNone)
	requireRuleError(t, err, ErrDisallowedCertVersion)

	require.Equal(t, fp8ForkTestHeight-1, chain.BestSnapshot().Height)
}

func TestFp8ForkBlockTemplateVersion(t *testing.T) {
	params := fp8SimNetParams()
	chain, teardown, err := chainSetup("fp8_fork_template", &params)
	require.NoError(t, err)
	defer teardown()

	tip := btcutil.NewBlock(chain.chainParams.GenesisBlock)
	tip.SetHeight(0)

	for h := int32(1); h < fp8ForkTestHeight; h++ {
		block, _, err := addBlock(chain, tip, nil)
		require.NoError(t, err)
		tip = block
	}

	v4Template := newBlockForcedCert(t, chain, tip,
		&wire.CertificateV4{Hash: *tip.Hash()})
	require.Equal(t, wire.CertificateVersionV4,
		v4Template.MsgBlock().BlockCertificate().Version())
	require.NoError(t, chain.CheckConnectBlockTemplate(v4Template),
		"V4 template must be accepted at the fork height")

	v3Template := newBlockForcedCert(t, chain, tip,
		&wire.CertificateV3{CertificateV2: wire.CertificateV2{Hash: *tip.Hash()}})
	requireRuleError(t, chain.CheckConnectBlockTemplate(v3Template),
		ErrDisallowedCertVersion)
}

func TestSolveBlockRequiresFp8Fork(t *testing.T) {
	simNet := fp8SimNetParams()
	regTest := chaincfg.RegressionNetParams
	regTest.Fp8ForkHeight = fp8ForkTestHeight

	tests := []struct {
		name    string
		params  *chaincfg.Params
		height  int32
		wantErr string
	}{
		{"simnet before the fork", &simNet, fp8ForkTestHeight - 1, "requires certificate version 3"},
		{"simnet at the fork", &simNet, fp8ForkTestHeight, ""},
		{"regtest before the fork", &regTest, fp8ForkTestHeight - 1, "requires certificate version 3"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			header := &wire.BlockHeader{}
			cert, err := SolveBlock(header, &wire.BlockHeader{}, tt.params, tt.height)
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, wire.CertificateVersionV4, cert.Version())
			require.Equal(t, cert.ProofCommitment(), header.ProofCommitment)
			require.Equal(t, header.BlockHash(), cert.BlockHash())
		})
	}
}

func TestCheckCertificateRulesV4SkipsRankPenalty(t *testing.T) {
	params := &chaincfg.Params{
		MoEForkHeight:         1,
		SaltedSeedForkHeight:  1,
		Fp8ForkHeight:         1,
		RankPenaltyForkHeight: 1,
	}
	cert := v4Cert([]byte{0x00}, []byte{0x01})
	require.NoError(t, CheckCertificateRules(&wire.BlockHeader{}, cert, 1, params, BFNone))
}

// TestCheckBlockSanityRejectsV4WithoutFp8Fork asserts that the context-free block checks reject a V4 certificate on a
// network with no FP8 fork height, and leave the version to the height rule when the fork is scheduled.
func TestCheckBlockSanityRejectsV4WithoutFp8Fork(t *testing.T) {
	params := fp8SimNetParams()
	chain, teardown, err := chainSetup("fp8_fork_unscheduled", &params)
	require.NoError(t, err)
	defer teardown()

	tip := btcutil.NewBlock(chain.chainParams.GenesisBlock)
	tip.SetHeight(0)
	block := newBlockForcedCert(t, chain, tip, v4Cert([]byte{0x00}, nil))

	require.NoError(t, CheckBlockSanity(block, &params, chain.timeSource))

	unscheduled := params
	unscheduled.Fp8ForkHeight = 0
	err = CheckBlockSanity(block, &unscheduled, chain.timeSource)
	requireRuleError(t, err, ErrDisallowedCertVersion)
	require.ErrorContains(t, err, "FP8 fork is not scheduled")
}

func TestNewRejectsFp8BeforeSaltedSeedFork(t *testing.T) {
	params := fp8SimNetParams()
	params.SaltedSeedForkHeight = 5
	params.Fp8ForkHeight = 4
	_, _, err := chainSetup("fp8_fork_ordering", &params)
	require.Error(t, err)
	require.Contains(t, err.Error(), "Fp8ForkHeight")
}
