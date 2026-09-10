// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package neutrino

import (
	"bytes"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/spv/headerfs"
	"github.com/pearl-research-labs/pearl/spv/headerlist"
	"github.com/stretchr/testify/require"
)

// ctxTestParams returns SimNet parameters re-badged as RegTest so
// NetBehaviorFlags does not exempt the network. The genesis block stays
// SimNet's, which the store was seeded with.
func ctxTestParams() chaincfg.Params {
	params := chaincfg.SimNetParams
	params.Net = wire.RegTest
	params.ReduceMinDifficulty = false
	params.Fp8ForkHeight = 1
	return params
}

// ctxHeaders builds n headers past genesis and writes them into the store
// so fallback lookups succeed.
func ctxHeaders(t *testing.T, store headerfs.BlockHeaderStore,
	n int) []wire.BlockHeader {

	t.Helper()
	genesis := chaincfg.SimNetParams.GenesisBlock.MsgHeader.BlockHeader
	headers := make([]wire.BlockHeader, 0, n)
	prevHash := genesis.BlockHash()
	prevTs := genesis.Timestamp
	for i := 0; i < n; i++ {
		header := wire.BlockHeader{
			Version:   int32(i + 1),
			PrevBlock: prevHash,
			Timestamp: prevTs.Add(
				time.Duration(blockchain.MinTimestampDeltaSeconds) * time.Second),
			Bits: chaincfg.SimNetParams.PowLimitBits,
		}
		headers = append(headers, header)
		prevHash = header.BlockHash()
		prevTs = header.Timestamp
	}
	writes := make([]headerfs.BlockHeader, 0, len(headers))
	for i := range headers {
		writes = append(writes, headerfs.BlockHeader{
			BlockHeader: &headers[i], Height: uint32(i + 1),
		})
	}
	require.NoError(t, store.WriteHeaders(writes...))
	return headers
}

func spvContextCert(t *testing.T, ancestor *wire.BlockHeader) *wire.CertificateV4 {
	t.Helper()
	var serialized bytes.Buffer
	require.NoError(t, ancestor.Serialize(&serialized))
	size := wire.MaxBlockHeaderPayload - chainhash.HashSize
	return &wire.CertificateV4{PublicData: serialized.Bytes()[:size]}
}

func TestGrandparentHeaderResolution(t *testing.T) {
	bm, hdrStore, _, err := setupBlockManager(t)
	require.NoError(t, err)
	bm.cfg.ChainParams = ctxTestParams()

	headers := ctxHeaders(t, hdrStore, 3)

	// In-memory predecessor: the list holds (h0, h1); h2's grandparent
	// resolves to the node before its parent (h0).
	bm.headerList.ResetHeaderState(headerlist.Node{
		Header: headers[0], Height: 1,
	})
	bm.headerList.PushBack(headerlist.Node{
		Header: headers[1], Height: 2,
	})
	parentNode := bm.headerList.Back()
	require.NotNil(t, parentNode.Prev())

	parent := headers[1]
	gp, err := bm.grandparentHeader(parentNode.Prev(), &parent)
	require.NoError(t, err)
	require.NotNil(t, gp)
	require.Equal(t, headers[0].BlockHash(), gp.BlockHash())

	// Startup/batch boundary: the list holds only the parent, so the
	// grandparent must come from the store via the parent's PrevBlock.
	bm.headerList.ResetHeaderState(headerlist.Node{
		Header: headers[1], Height: 2,
	})
	require.Nil(t, bm.headerList.Back().Prev())

	gp, err = bm.grandparentHeader(nil, &parent)
	require.NoError(t, err)
	require.NotNil(t, gp, "the store fallback must resolve it")
	require.Equal(t, headers[0].BlockHash(), gp.BlockHash())

	unknown := &wire.BlockHeader{
		Version:   99,
		PrevBlock: chainhash.Hash{0xEE},
		Timestamp: time.Unix(1, 0),
		Bits:      chaincfg.SimNetParams.PowLimitBits,
	}
	gp, err = bm.grandparentHeader(nil, unknown)
	require.Error(t, err)
	require.Nil(t, gp)

	gp, err = bm.grandparentHeader(nil, nil)
	require.NoError(t, err)
	require.Nil(t, gp)
}

func TestReorgCertificateContext(t *testing.T) {
	bm, hdrStore, _, err := setupBlockManager(t)
	require.NoError(t, err)
	bm.cfg.ChainParams = ctxTestParams()

	headers := ctxHeaders(t, hdrStore, 3)
	fork := &headers[1]
	grandparent, err := bm.grandparentHeader(nil, fork)
	require.NoError(t, err)

	ctx := blockchain.CertificateHeaderContext{
		Parent:      fork,
		Grandparent: grandparent,
	}
	first := wire.BlockHeader{
		Version:   4,
		PrevBlock: fork.BlockHash(),
		Timestamp: fork.Timestamp.Add(time.Second),
		Bits:      fork.Bits,
	}
	require.NoError(t, blockchain.CheckCertificateContext(
		&first, ctx, spvContextCert(t, grandparent), blockchain.BFNone,
	))
	ctx.Advance(&first)

	second := wire.BlockHeader{
		Version:   5,
		PrevBlock: first.BlockHash(),
		Timestamp: first.Timestamp.Add(time.Second),
		Bits:      first.Bits,
	}
	require.NoError(t, blockchain.CheckCertificateContext(
		&second, ctx, spvContextCert(t, fork), blockchain.BFNone,
	))

	before := ctx
	err = blockchain.CheckCertificateContext(
		&second, ctx, spvContextCert(t, grandparent), blockchain.BFNone,
	)
	require.Error(t, err)
	require.Equal(t, before, ctx)
}
