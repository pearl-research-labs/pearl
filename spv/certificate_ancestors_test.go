//go:build zkpow

// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package neutrino

import (
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

func TestHeaderSanityCertificateAncestors(t *testing.T) {
	params := chaincfg.RegressionNetParams
	bm := &blockManager{cfg: &blockManagerCfg{
		ChainParams: params,
		TimeSource:  blockchain.NewMedianTime(),
	}}

	parent := *params.GenesisBlock.BlockHeader()
	parent.PrevBlock[0] = 1
	header := parent
	header.PrevBlock = parent.BlockHash()
	header.Timestamp = parent.Timestamp.Add(
		time.Duration(blockchain.MinTimestampDeltaSeconds) * time.Second,
	)
	forged := parent
	forged.MerkleRoot[0] ^= 1
	prefix := forged.IncompleteHeaderBytes()
	cert := &wire.CertificateV4{
		PublicData:      prefix[:],
		ProofData:       []byte{1},
		AncestorHeaders: []wire.BlockHeader{forged},
	}
	header.ProofCommitment = cert.ProofCommitment()
	cert.Hash = header.BlockHash()

	for name, reorg := range map[string]bool{"extension": false, "reorg": true} {
		t.Run(name, func(t *testing.T) {
			// Ancestor authentication needs no header store or list, and
			// rejects the disconnected witness before native proof verification.
			err := bm.checkHeaderSanity(&header, &parent, cert, reorg, 1)
			var ruleErr blockchain.RuleError
			require.ErrorAs(t, err, &ruleErr)
			require.Equal(t, blockchain.ErrHighHash, ruleErr.ErrorCode)
			require.Contains(t, ruleErr.Description, "ancestor")
		})
	}
}
