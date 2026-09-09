// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain/internal/testhelper"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// flowHeaders builds a header chain of the given length on top of genesis
// with increasing timestamps, all at PowLimitBits.
func flowHeaders(chain *BlockChain, n int) []wire.BlockHeader {
	genesis := chain.chainParams.GenesisBlock.MsgHeader.BlockHeader
	headers := make([]wire.BlockHeader, 0, n)
	prevHash := genesis.BlockHash()
	prevTs := genesis.Timestamp
	for i := 0; i < n; i++ {
		header := wire.BlockHeader{
			Version:   1,
			PrevBlock: prevHash,
			Timestamp: prevTs.Add(time.Duration(MinTimestampDeltaSeconds) * time.Second),
			Bits:      chain.chainParams.PowLimitBits,
		}
		prevHash = header.BlockHash()
		prevTs = header.Timestamp
		headers = append(headers, header)
	}
	return headers
}

// flowBlock builds a minimal block from the given header, merkle-tying a
// BIP0034-compliant coinbase for the given height.
func flowBlock(t *testing.T, chain *BlockChain, header wire.BlockHeader,
	height int32, cert wire.BlockCertificate) *btcutil.Block {

	t.Helper()
	coinbase := testhelper.CreateCoinbaseTx(
		height, CalcBlockSubsidy(height, chain.chainParams))
	header.MerkleRoot = calcMerkleRoot([]*wire.MsgTx{coinbase})
	return btcutil.NewBlock(&wire.MsgBlock{
		MsgHeader: wire.MsgHeader{
			BlockHeader:    header,
			MsgCertificate: wire.MsgCertificate{Certificate: cert},
		},
		Transactions: []*wire.MsgTx{coinbase},
	})
}

// Check contextual acceptance directly: full ProcessBlock first requires a
// valid native proof, while SimNet bypasses the ancestor rule entirely.
func TestCheckBlockContextAncestorWindow(t *testing.T) {
	params := chaincfg.RegressionNetParams
	params.ReduceMinDifficulty = false
	chain, teardown, err := chainSetup("v4_ctx_window", &params)
	require.NoError(t, err)
	defer teardown()

	// h3's window is (h2, h1): parent h2, grandparent h1.
	headers := flowHeaders(chain, 3)
	genesisNode := chain.bestChain.Tip()
	h1Node := newBlockNode(&headers[0], genesisNode, statusDataStored, 0)
	h2Node := newBlockNode(&headers[1], h1Node, statusDataStored, 0)

	rogue := wire.BlockHeader{Version: 9, Timestamp: time.Unix(1, 0), Bits: 0x207fffff}
	for _, flags := range []BehaviorFlags{BFNone, BFFastAdd, BFNoPoWCheck} {
		block := flowBlock(t, chain, headers[2], 3, contextCert(t, &headers[0]))
		require.NoError(t, chain.checkBlockContext(block, h2Node, flags))

		block = flowBlock(t, chain, headers[2], 3, contextCert(t, &rogue))
		err := chain.checkBlockContext(block, h2Node, flags)
		if flags&BFNoPoWCheck != 0 {
			require.NoError(t, err)
		} else {
			requireRuleError(t, err, ErrHighHash)
		}
	}
}
