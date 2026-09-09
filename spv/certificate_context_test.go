// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package neutrino

import (
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

// ctxHeaders builds n headers past genesis and writes them into the store
// so fallback lookups succeed.
func ctxHeaders(t *testing.T, store headerfs.BlockHeaderStore,
	n int) []wire.BlockHeader {

	t.Helper()
	parent := chaincfg.SimNetParams.GenesisBlock.MsgHeader.BlockHeader
	headers := make([]wire.BlockHeader, n)
	writes := make([]headerfs.BlockHeader, n)
	for i := range headers {
		headers[i] = wire.BlockHeader{
			Version:   int32(i + 1),
			PrevBlock: parent.BlockHash(),
			Timestamp: parent.Timestamp.Add(
				time.Duration(blockchain.MinTimestampDeltaSeconds) * time.Second),
			Bits: chaincfg.SimNetParams.PowLimitBits,
		}
		writes[i] = headerfs.BlockHeader{BlockHeader: &headers[i], Height: uint32(i + 1)}
		parent = headers[i]
	}
	require.NoError(t, store.WriteHeaders(writes...))
	return headers
}

func spvContextCert(ancestor *wire.BlockHeader) *wire.CertificateV4 {
	prefix := ancestor.IncompleteHeaderBytes()
	return &wire.CertificateV4{PublicData: prefix[:]}
}

func TestCertificateHeaderContext(t *testing.T) {
	bm, hdrStore, _, err := setupBlockManager(t)
	require.NoError(t, err)

	ctx, err := bm.certificateHeaderContext(bm.headerList.Back())
	require.NoError(t, err)
	require.Equal(t, bm.headerList.Back().Header, *ctx.Parent)
	require.Nil(t, ctx.Grandparent, "genesis has no parent")

	headers := ctxHeaders(t, hdrStore, 3)
	bm.headerList.ResetHeaderState(headerlist.Node{Header: headers[0], Height: 1})
	bm.headerList.PushBack(headerlist.Node{Header: headers[1], Height: 2})
	ctx, err = bm.certificateHeaderContext(bm.headerList.Back())
	require.NoError(t, err)
	require.Equal(t, headers[1], *ctx.Parent)
	require.Equal(t, headers[0], *ctx.Grandparent)

	// With only the parent retained, resolve the same context from storage.
	bm.headerList.ResetHeaderState(headerlist.Node{Header: headers[1], Height: 2})
	stored, err := bm.certificateHeaderContext(bm.headerList.Back())
	require.NoError(t, err)
	require.Equal(t, headers[1], *stored.Parent)
	require.Equal(t, headers[0], *stored.Grandparent)

	unknown := &headerlist.Node{Header: wire.BlockHeader{PrevBlock: chainhash.Hash{0xEE}}}
	ctx, err = bm.certificateHeaderContext(unknown)
	require.Error(t, err)
	require.Empty(t, ctx)
}

func TestReorgCertificateContext(t *testing.T) {
	bm, hdrStore, _, err := setupBlockManager(t)
	require.NoError(t, err)
	bm.cfg.ChainParams.Net = wire.RegTest
	bm.cfg.ChainParams.PoWNoRetargeting = true

	headers := ctxHeaders(t, hdrStore, 3)
	fork := headers[1]
	bm.reorgList.ResetHeaderState(headerlist.Node{Header: fork, Height: 2})
	ctx, err := bm.certificateHeaderContext(bm.reorgList.Back())
	require.NoError(t, err)
	first := wire.BlockHeader{
		Version:   4,
		PrevBlock: fork.BlockHash(),
		Timestamp: fork.Timestamp.Add(time.Second),
		Bits:      fork.Bits,
	}
	require.NoError(t, blockchain.CheckCertificateContext(
		&first, ctx, spvContextCert(&headers[0]), blockchain.BFNone,
	))

	// The next context follows the accepted reorg header, not the stored
	// main-chain header at the same height.
	bm.reorgList.PushBack(headerlist.Node{Header: first, Height: 3})
	ctx, err = bm.certificateHeaderContext(bm.reorgList.Back())
	require.NoError(t, err)
	require.Equal(t, first, *ctx.Parent)
	require.Equal(t, fork, *ctx.Grandparent)
	second := wire.BlockHeader{
		Version:   5,
		PrevBlock: first.BlockHash(),
		Timestamp: first.Timestamp.Add(time.Second),
		Bits:      first.Bits,
	}
	require.NoError(t, blockchain.CheckCertificateContext(
		&second, ctx, spvContextCert(&fork), blockchain.BFNone,
	))

	// Assert the specific context error through SPV validation before the
	// placeholder certificate reaches native proof verification.
	err = bm.checkHeaderSanity(&second, ctx, spvContextCert(&headers[0]), true, 3)
	var ruleErr blockchain.RuleError
	require.ErrorAs(t, err, &ruleErr)
	require.Equal(t, blockchain.ErrHighHash, ruleErr.ErrorCode)
	require.Contains(t, ruleErr.Description, "ancestor header is not")
}
