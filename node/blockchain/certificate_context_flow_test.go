// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"bytes"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain/internal/testhelper"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// These tests exercise the v4 ancestor-context rule through the block
// validation flow. Full ProcessBlock acceptance with the rule active would
// require a real fp8 proof (checkProofOfWork verifies it before the
// contextual check runs), so the flow tests drive checkBlockContext — the
// exact function the rule is wired into — with hand-built blocks, plus the
// SimNet full flow, where SolveBlock's dummy V4 certificates must stay
// valid because NetBehaviorFlags exempts SimNet.

// v4FlowParams returns parameters whose NetBehaviorFlags do not exempt
// the network (RegTest), with the fp8 fork active from genesis, so the
// ancestor-context rule fires in the flow tests. The genesis block is
// SimNet's; the chain never cross-checks it against Net.
func v4FlowParams() chaincfg.Params {
	params := chaincfg.SimNetParams
	params.Net = wire.RegTest
	params.ReduceMinDifficulty = false
	params.Fp8ForkHeight = 1
	return params
}

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

// v4CertFor returns a V4 certificate whose public data commits the given
// header as its proof-carried ancestor σ_Δ.
func v4CertFor(ancestor *wire.BlockHeader) *wire.CertificateV4 {
	var serialized bytes.Buffer
	if err := ancestor.Serialize(&serialized); err != nil {
		panic(err)
	}
	incompleteHeaderSize := wire.MaxBlockHeaderPayload - chainhash.HashSize
	public := append(
		serialized.Bytes()[:incompleteHeaderSize:incompleteHeaderSize],
		bytes.Repeat([]byte{0xAB}, 32)...,
	)
	return &wire.CertificateV4{PublicData: public}
}

// flowBlock builds a minimal block from the given header, merkle-tying a
// BIP0034-compliant coinbase for the given height. The certificate is
// built by mkCert from the finalized header, so a depth-0 σ_Δ (the
// proposed header itself) commits the header as it will be validated.
func flowBlock(t *testing.T, chain *BlockChain, header wire.BlockHeader,
	height int32,
	mkCert func(finalized *wire.BlockHeader) wire.BlockCertificate) *btcutil.Block {

	t.Helper()
	coinbase := testhelper.CreateCoinbaseTx(
		height, CalcBlockSubsidy(height, chain.chainParams))
	header.MerkleRoot = calcMerkleRoot([]*wire.MsgTx{coinbase})
	cert := mkCert(&header)
	return btcutil.NewBlock(&wire.MsgBlock{
		MsgHeader: wire.MsgHeader{
			BlockHeader:    header,
			MsgCertificate: wire.MsgCertificate{Certificate: cert},
		},
		Transactions: []*wire.MsgTx{coinbase},
	})
}

// ancestorCert builds a V4 certificate committing the given fixed ancestor
// as σ_Δ.
func ancestorCert(ancestor *wire.BlockHeader) func(*wire.BlockHeader) wire.BlockCertificate {
	return func(*wire.BlockHeader) wire.BlockCertificate {
		return v4CertFor(ancestor)
	}
}

// TestCheckBlockContextAncestorWindow pins the wiring: checkBlockContext
// reconstructs the parent/grandparent from blockNode links (the path
// maybeAcceptBlock takes) and invokes the rule. The depth/shallow/short
// matrices live in TestCheckCertificateContext_*.
func TestCheckBlockContextAncestorWindow(t *testing.T) {
	params := v4FlowParams()
	chain, teardown, err := chainSetup("v4_ctx_window", &params)
	require.NoError(t, err)
	defer teardown()

	// h3's window is (h2, h1): parent h2, grandparent h1.
	headers := flowHeaders(chain, 3)
	genesisNode := chain.bestChain.Tip()
	h1Node := newBlockNode(&headers[0], genesisNode, statusDataStored, 0)
	h2Node := newBlockNode(&headers[1], h1Node, statusDataStored, 0)

	rogue := wire.BlockHeader{Version: 9, Timestamp: time.Unix(1, 0), Bits: 0x207fffff}
	for _, flags := range []BehaviorFlags{BFNone, BFFastAdd} {
		block := flowBlock(t, chain, headers[2], 3, ancestorCert(&headers[0]))
		require.NoError(t, chain.checkBlockContext(block, h2Node, flags))

		block = flowBlock(t, chain, headers[2], 3, ancestorCert(&rogue))
		requireRuleError(t, chain.checkBlockContext(block, h2Node, flags),
			ErrHighHash)
	}
}
