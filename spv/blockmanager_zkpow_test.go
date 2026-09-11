//go:build zkpow

// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package neutrino

import (
	"encoding/binary"
	"os"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

func TestCheckHeaderSanityRejectsDisconnectedV4Ancestor(t *testing.T) {
	// Fixture framing is header(76), public length(4), public data, proof.
	raw, err := os.ReadFile("../node/zkpow/testdata/fp8_zk_proof_b200.bin")
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(raw), 80)
	publicLen := int(binary.LittleEndian.Uint32(raw[76:80]))
	require.LessOrEqual(t, 80+publicLen, len(raw))

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
	cert := &wire.CertificateV4{
		PublicData:      raw[80 : 80+publicLen],
		ProofData:       []byte{1},
		AncestorHeaders: []wire.BlockHeader{forged},
	}
	header.ProofCommitment = cert.ProofCommitment()
	cert.Hash = header.BlockHash()

	for name, reorg := range map[string]bool{"extension": false, "reorg": true} {
		t.Run(name, func(t *testing.T) {
			// Native ancestry validation needs no header store or list and
			// rejects the disconnected witness before proof verification.
			err := bm.checkHeaderSanity(&header, &parent, cert, reorg, 1)
			var ruleErr blockchain.RuleError
			require.ErrorAs(t, err, &ruleErr)
			require.Equal(t, blockchain.ErrHighHash, ruleErr.ErrorCode)
			require.Contains(t, ruleErr.Description, "v4 ancestor header at depth 1 does not connect")
		})
	}
}
