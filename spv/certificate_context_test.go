// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package neutrino

import (
	"errors"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/spv/blockntfns"
	"github.com/pearl-research-labs/pearl/spv/headerfs"
	"github.com/pearl-research-labs/pearl/spv/headerlist"
	"github.com/stretchr/testify/require"
)

// Handler tests use SimNet placeholder certificates to exercise ancestor
// resolution and chain transitions. The active rule is tested separately below.
func headerContextMessages(parent wire.BlockHeader, count int) *wire.MsgHeaders {
	msg := &wire.MsgHeaders{}
	delta := time.Duration(blockchain.MinTimestampDeltaSeconds) * time.Second
	for range count {
		header := wire.BlockHeader{
			Version:   parent.Version + 1,
			PrevBlock: parent.BlockHash(),
			Timestamp: parent.Timestamp.Add(delta),
			Bits:      chaincfg.SimNetParams.PowLimitBits,
		}
		msg.Headers = append(msg.Headers, wire.MsgHeader{
			BlockHeader: header,
			MsgCertificate: wire.MsgCertificate{
				Certificate: &wire.CertificateV4{ProofData: []byte{0}},
			},
		})
		parent = header
	}
	return msg
}

func setupHeaderContextManager(t *testing.T) (*blockManager, *ServerPeer) {
	t.Helper()
	bm, _, _, err := setupBlockManager(t)
	require.NoError(t, err)
	bm.cfg.ChainParams.PoWNoRetargeting = true
	p, err := peer.NewOutboundPeer(&peer.Config{}, "127.0.0.1:12345")
	require.NoError(t, err)
	t.Cleanup(p.Disconnect)
	sp := &ServerPeer{Peer: p}
	bm.syncPeer = sp
	return bm, sp
}

func requireContextPeerNotDisconnected(t *testing.T, sp *ServerPeer) {
	t.Helper()
	select {
	case <-sp.Done():
		t.Fatal("header processing disconnected the peer")
	default:
	}
}

type contextHeaderStore struct {
	headerfs.BlockHeaderStore
	failHash chainhash.Hash
	fetchErr error
	fetches  []chainhash.Hash
}

func (s *contextHeaderStore) FetchHeader(hash *chainhash.Hash) (*wire.BlockHeader, uint32, error) {
	s.fetches = append(s.fetches, *hash)
	if s.fetchErr != nil && *hash == s.failHash {
		return nil, 0, s.fetchErr
	}
	return s.BlockHeaderStore.FetchHeader(hash)
}

func TestCertificateHeaderContext(t *testing.T) {
	bm, sp := setupHeaderContextManager(t)
	genesis := bm.headerList.Back().Header
	ctx, err := bm.certificateHeaderContext(bm.headerList.Back())
	require.NoError(t, err)
	require.Equal(t, genesis, *ctx.Parent)
	require.Nil(t, ctx.Grandparent)

	msg := headerContextMessages(genesis, 3)
	bm.handleHeadersMsg(&headersMsg{headers: msg, peer: sp})
	requireContextPeerNotDisconnected(t, sp)
	parent := bm.headerList.Back()
	ctx, err = bm.certificateHeaderContext(parent)
	require.NoError(t, err)
	require.Equal(t, msg.Headers[2].BlockHeader, *ctx.Parent)
	require.Equal(t, msg.Headers[1].BlockHeader, *ctx.Grandparent)

	// With only the parent retained, the same context comes from the store.
	bm.headerList.ResetHeaderState(headerlist.Node{Header: parent.Header, Height: parent.Height})
	stored, err := bm.certificateHeaderContext(bm.headerList.Back())
	require.NoError(t, err)
	require.Equal(t, ctx, stored)

	missing := &headerlist.Node{Header: wire.BlockHeader{PrevBlock: chainhash.Hash{0xEE}}}
	_, err = bm.certificateHeaderContext(missing)
	require.Error(t, err)
}

func TestHandleHeadersContextAcrossBatches(t *testing.T) {
	bm, sp := setupHeaderContextManager(t)
	first := headerContextMessages(bm.headerList.Back().Header, 2)
	bm.handleHeadersMsg(&headersMsg{headers: first, peer: sp})
	requireContextPeerNotDisconnected(t, sp)
	require.EqualValues(t, 2, bm.headerTip)

	// Simulate the one-header memory state after restarting. Only the first
	// header in the next batch should fetch its grandparent from the store.
	parent := first.Headers[1].BlockHeader
	bm.headerList.ResetHeaderState(headerlist.Node{Header: parent, Height: 2})
	store := &contextHeaderStore{BlockHeaderStore: bm.cfg.BlockHeaders}
	bm.cfg.BlockHeaders = store
	second := headerContextMessages(parent, 2)
	bm.handleHeadersMsg(&headersMsg{headers: second, peer: sp})
	requireContextPeerNotDisconnected(t, sp)
	require.Equal(t, []chainhash.Hash{first.Headers[0].BlockHeader.BlockHash()}, store.fetches)
	last := second.Headers[1].BlockHeader
	tip, height, err := store.ChainTip()
	require.NoError(t, err)
	require.EqualValues(t, 4, height)
	require.Equal(t, last, *tip)
	require.Equal(t, last, bm.headerList.Back().Header)
	require.Equal(t, last.BlockHash(), bm.headerTipHash)
	require.Equal(t, second.Headers[0].BlockHeader, bm.headerList.Back().Prev().Header)
}

func TestHandleHeadersReorgContext(t *testing.T) {
	bm, sp := setupHeaderContextManager(t)
	main := headerContextMessages(bm.headerList.Back().Header, 3)
	bm.handleHeadersMsg(&headersMsg{headers: main, peer: sp})
	requireContextPeerNotDisconnected(t, sp)

	// Replace height three with a longer branch rooted at height two.
	fork := main.Headers[1].BlockHeader
	reorg := headerContextMessages(fork, 1)
	reorg.Headers[0].BlockHeader.Version++
	next := headerContextMessages(reorg.Headers[0].BlockHeader, 1)
	reorg.Headers = append(reorg.Headers, next.Headers...)
	bm.blockNtfnChan = make(chan blockntfns.BlockNtfn, 1)
	bm.handleHeadersMsg(&headersMsg{headers: reorg, peer: sp})
	requireContextPeerNotDisconnected(t, sp)

	tip, height, err := bm.cfg.BlockHeaders.ChainTip()
	require.NoError(t, err)
	require.EqualValues(t, 4, height)
	require.Equal(t, reorg.Headers[1].BlockHeader, *tip)
	require.Equal(t, *tip, bm.headerList.Back().Header)
	require.Equal(t, *tip, bm.reorgList.Back().Header)
	require.Equal(t, reorg.Headers[0].BlockHeader, bm.reorgList.Back().Prev().Header)
	require.Equal(t, fork, bm.reorgList.Back().Prev().Prev().Header)
	oldHash := main.Headers[2].BlockHeader.BlockHash()
	_, _, err = bm.cfg.BlockHeaders.FetchHeader(&oldHash)
	require.Error(t, err)
	select {
	case ntfn := <-bm.Notifications():
		require.IsType(t, &blockntfns.Disconnected{}, ntfn)
		require.Equal(t, main.Headers[2].BlockHeader, ntfn.Header())
	default:
		t.Fatal("missing notification for the replaced header")
	}
}

func TestHandleHeadersContextReadFailure(t *testing.T) {
	for _, reorg := range []bool{false, true} {
		name := "extension"
		if reorg {
			name = "reorg"
		}
		t.Run(name, func(t *testing.T) {
			bm, sp := setupHeaderContextManager(t)
			main := headerContextMessages(bm.headerList.Back().Header, 3)
			bm.handleHeadersMsg(&headersMsg{headers: main, peer: sp})
			requireContextPeerNotDisconnected(t, sp)
			oldTip := main.Headers[2].BlockHeader
			parent := oldTip
			grandparent := main.Headers[1].BlockHeader
			if reorg {
				parent = main.Headers[1].BlockHeader
				grandparent = main.Headers[0].BlockHeader
			}
			bm.headerList.ResetHeaderState(headerlist.Node{Header: oldTip, Height: 3})
			store := &contextHeaderStore{
				BlockHeaderStore: bm.cfg.BlockHeaders,
				failHash:         grandparent.BlockHash(),
				fetchErr:         errors.New("header read failed"),
			}
			bm.cfg.BlockHeaders = store
			msg := headerContextMessages(parent, 1)
			msg.Headers[0].BlockHeader.Version++
			bm.handleHeadersMsg(&headersMsg{headers: msg, peer: sp})
			requireContextPeerNotDisconnected(t, sp)
			require.Contains(t, store.fetches, store.failHash)
			tip, height, err := store.ChainTip()
			require.NoError(t, err)
			require.EqualValues(t, 3, height)
			require.Equal(t, oldTip, *tip)
			require.Equal(t, oldTip, bm.headerList.Back().Header)
			require.Equal(t, oldTip.BlockHash(), bm.headerTipHash)
			if reorg {
				require.Equal(t, parent, bm.reorgList.Back().Header)
				require.Nil(t, bm.reorgList.Back().Prev())
			}
		})
	}
}

func TestHeaderSanityRejectsCertificateAncestor(t *testing.T) {
	bm, sp := setupHeaderContextManager(t)
	main := headerContextMessages(bm.headerList.Back().Header, 2)
	bm.handleHeadersMsg(&headersMsg{headers: main, peer: sp})
	requireContextPeerNotDisconnected(t, sp)
	ctx, err := bm.certificateHeaderContext(bm.headerList.Back())
	require.NoError(t, err)
	proposed := headerContextMessages(*ctx.Parent, 1).Headers[0].BlockHeader
	rogue := proposed
	rogue.Version++
	ancestor := rogue.IncompleteHeaderBytes()
	cert := &wire.CertificateV4{PublicData: ancestor[:]}

	// Unlike the handler transition tests, enable the rule. The specific
	// error proves rejection happens before the missing/invalid proof does.
	bm.cfg.ChainParams.Net = wire.RegTest
	err = bm.checkHeaderSanity(&proposed, ctx, cert, false, 2)
	var ruleErr blockchain.RuleError
	require.ErrorAs(t, err, &ruleErr)
	require.Equal(t, blockchain.ErrHighHash, ruleErr.ErrorCode)
	require.Contains(t, ruleErr.Description, "ancestor header is not")
}
