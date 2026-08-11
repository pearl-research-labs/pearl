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

// Tests for the salted noise-seed (V3) certificate version cutover on SimNet,
// mirroring the MoE fork tests in moe_fork_test.go.

const saltedForkTestHeight = int32(4)

func saltedSimNetParams() chaincfg.Params {
	params := chaincfg.SimNetParams
	params.ReduceMinDifficulty = false
	params.MoEForkHeight = 1
	params.SaltedSeedForkHeight = saltedForkTestHeight
	return params
}

func v3Cert(proofData []byte, publicDataLen uint32) *wire.CertificateV3 {
	cert := &wire.CertificateV3{}
	cert.ProofData = proofData
	cert.PublicDataLen = publicDataLen
	return cert
}

// TestSaltedSeedForkActivationAcceptsRequiredVersion builds a chain across the
// activation boundary and asserts each block is accepted with the version
// required at its height (V2 before the fork, V3 at and after it). Blocks are
// produced by SolveBlock, so mining-side version selection is covered too.
func TestSaltedSeedForkActivationAcceptsRequiredVersion(t *testing.T) {
	params := saltedSimNetParams()
	chain, teardown, err := chainSetup("salted_fork_accept", &params)
	require.NoError(t, err)
	defer teardown()

	tip := btcutil.NewBlock(chain.chainParams.GenesisBlock)
	tip.SetHeight(0)

	for h := int32(1); h <= saltedForkTestHeight+2; h++ {
		block, _, err := addBlock(chain, tip, nil)
		require.NoErrorf(t, err, "block at height %d should be accepted", h)

		want := wire.CertificateVersionV2
		if h >= saltedForkTestHeight {
			want = wire.CertificateVersionV3
		}
		require.Equalf(t, want, block.MsgBlock().BlockCertificate().Version(),
			"unexpected certificate version at height %d", h)

		tip = block
	}

	require.Equal(t, saltedForkTestHeight+2, chain.BestSnapshot().Height)
}

// TestSaltedSeedForkRejectsWrongVersion asserts the strict cutover: a V3
// certificate is rejected below the activation height, and a V2 certificate is
// rejected at it.
func TestSaltedSeedForkRejectsWrongVersion(t *testing.T) {
	params := saltedSimNetParams()
	chain, teardown, err := chainSetup("salted_fork_reject", &params)
	require.NoError(t, err)
	defer teardown()

	tip := btcutil.NewBlock(chain.chainParams.GenesisBlock)
	tip.SetHeight(0)

	// Extend until the next block is pre-fork (height saltedForkTestHeight-1).
	for h := int32(1); h <= saltedForkTestHeight-2; h++ {
		block, _, err := addBlock(chain, tip, nil)
		require.NoError(t, err)
		tip = block
	}

	bad := newBlockForcedCert(t, chain, tip, v3Cert([]byte{0x00}, 0))
	_, _, err = chain.ProcessBlock(bad, BFNone)
	requireRuleError(t, err, ErrDisallowedCertVersion)

	// Advance to the activation height with a valid block, then force V2.
	block, _, err := addBlock(chain, tip, nil)
	require.NoError(t, err)
	tip = block

	bad = newBlockForcedCert(t, chain, tip,
		&wire.CertificateV2{ProofData: []byte{0x00}})
	_, _, err = chain.ProcessBlock(bad, BFNone)
	requireRuleError(t, err, ErrDisallowedCertVersion)

	// The tip must be unchanged by the rejected blocks.
	require.Equal(t, saltedForkTestHeight-1, chain.BestSnapshot().Height)
}

// TestCheckCertificateRulesDenseOnlyAppliesToV3 asserts the dense-only rule
// reads the V2 payload embedded in a V3 certificate.
func TestCheckCertificateRulesDenseOnlyAppliesToV3(t *testing.T) {
	params := &chaincfg.Params{
		MoEForkHeight:        1,
		SaltedSeedForkHeight: 2,
		DenseOnlyForkHeight:  2,
	}

	dense := v3Cert(nil, wire.PublicDataSizeDenseV2)
	moe := v3Cert(nil, wire.PublicDataSizeDenseV2+1)
	placeholder := v3Cert(nil, 0)

	require.NoError(t, checkCertVersion(dense, 2, params))
	require.NoError(t, checkCertVersion(placeholder, 2, params))
	requireRuleError(t, checkCertVersion(moe, 2, params),
		ErrDisallowedCertVersion)
}
