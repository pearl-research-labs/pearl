// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package netsync

import (
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/database"
	_ "github.com/pearl-research-labs/pearl/node/database/ffldb"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

type headerStub map[chainhash.Hash]wire.BlockHeader

func (s headerStub) HeaderByHash(hash *chainhash.Hash) (wire.BlockHeader, error) {
	h, ok := s[*hash]
	if !ok {
		return wire.BlockHeader{}, errors.New("block is not known")
	}
	return h, nil
}

func seedChain() (headerStub, []wire.BlockHeader) {
	h0 := *chaincfg.SimNetParams.GenesisBlock.BlockHeader()
	h1 := wire.BlockHeader{
		Version: 2, PrevBlock: h0.BlockHash(), Timestamp: time.Unix(200, 0),
	}
	h2 := wire.BlockHeader{
		Version: 3, PrevBlock: h1.BlockHash(), Timestamp: time.Unix(300, 0),
	}
	s := headerStub{
		h0.BlockHash(): h0,
		h1.BlockHash(): h1,
		h2.BlockHash(): h2,
	}
	return s, []wire.BlockHeader{h0, h1, h2}
}

func TestSeedHeaderCtx(t *testing.T) {
	t.Run("seed and advance", func(t *testing.T) {
		src, hs := seedChain()
		sm := &SyncManager{chainParams: &chaincfg.SimNetParams}

		h2 := hs[2].BlockHash()
		require.NoError(t, sm.seedHeaderCtx(&h2, src.HeaderByHash))
		require.NotNil(t, sm.syncHeaderCtx.Parent)
		require.Equal(t, hs[2], *sm.syncHeaderCtx.Parent)
		require.NotNil(t, sm.syncHeaderCtx.Grandparent)
		require.Equal(t, hs[1], *sm.syncHeaderCtx.Grandparent)

		accepted := wire.BlockHeader{
			Version: 4, PrevBlock: h2, Timestamp: time.Unix(400, 0),
		}
		sm.syncHeaderCtx.Advance(&accepted)
		require.Equal(t, accepted, *sm.syncHeaderCtx.Parent)
		require.Equal(t, hs[2], *sm.syncHeaderCtx.Grandparent)
	})

	t.Run("unknown hash leaves window unseeded", func(t *testing.T) {
		src, _ := seedChain()
		sm := &SyncManager{chainParams: &chaincfg.SimNetParams}

		unknown := chainhash.Hash{0xAB}
		require.Error(t, sm.seedHeaderCtx(&unknown, src.HeaderByHash))
		require.Nil(t, sm.syncHeaderCtx.Parent)
	})

	t.Run("genesis-depth parent has no grandparent", func(t *testing.T) {
		src, hs := seedChain()
		sm := &SyncManager{chainParams: &chaincfg.SimNetParams}

		h0 := hs[0].BlockHash()
		require.NoError(t, sm.seedHeaderCtx(&h0, src.HeaderByHash))
		require.Equal(t, hs[0], *sm.syncHeaderCtx.Parent)
		require.Nil(t, sm.syncHeaderCtx.Grandparent)
	})
}

func TestHandleHeadersContextAcrossBatchesAndReset(t *testing.T) {
	previousLog := log
	DisableLog()
	t.Cleanup(func() { log = previousLog })
	params := chaincfg.SimNetParams
	checkpointHash := chainhash.Hash{0xAB}
	params.Checkpoints = []chaincfg.Checkpoint{{Height: 10, Hash: &checkpointHash}}
	db, err := database.Create("ffldb", filepath.Join(t.TempDir(), "blocks"), params.Net)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, db.Close()) })
	chain, err := blockchain.New(&blockchain.Config{
		DB:          db,
		ChainParams: &params,
		TimeSource:  blockchain.NewMedianTime(),
		Checkpoints: params.Checkpoints,
		SigCache:    txscript.NewSigCache(0),
		HashCache:   txscript.NewHashCache(0),
	})
	require.NoError(t, err)
	sm, err := New(&Config{Chain: chain, ChainParams: &params})
	require.NoError(t, err)
	require.NotNil(t, sm.nextCheckpoint)
	p, err := peer.NewOutboundPeer(&peer.Config{}, "127.0.0.1:12345")
	require.NoError(t, err)
	t.Cleanup(p.Disconnect)
	sm.syncPeer = p
	sm.peerStates[p] = &peerSyncState{}
	sm.headersFirstMode = true

	genesis := *params.GenesisBlock.BlockHeader()
	parent := genesis
	var headers []wire.MsgHeader
	delta := time.Duration(blockchain.MinTimestampDeltaSeconds) * time.Second
	for range 4 {
		header := wire.BlockHeader{
			Version:   parent.Version + 1,
			PrevBlock: parent.BlockHash(),
			Timestamp: parent.Timestamp.Add(delta),
			Bits:      params.PowLimitBits,
		}
		headers = append(headers, wire.MsgHeader{
			BlockHeader: header,
			MsgCertificate: wire.MsgCertificate{
				Certificate: &wire.CertificateV4{ProofData: []byte{0}},
			},
		})
		parent = header
	}

	// These SimNet headers exercise the sync window while proof checks are
	// disabled. Header-only acceptance must not require storing full blocks.
	sm.handleHeadersMsg(&headersMsg{peer: p, headers: &wire.MsgHeaders{Headers: headers[:2]}})
	require.Equal(t, 3, sm.headerList.Len())
	require.Equal(t, headers[1].BlockHeader, *sm.syncHeaderCtx.Parent)
	require.Equal(t, headers[0].BlockHeader, *sm.syncHeaderCtx.Grandparent)
	for _, msgHeader := range headers[:2] {
		hash := msgHeader.BlockHeader.BlockHash()
		_, err := chain.HeaderByHash(&hash)
		require.Error(t, err, "the next batch must use its cached ancestors")
	}

	sm.handleHeadersMsg(&headersMsg{peer: p, headers: &wire.MsgHeaders{Headers: headers[2:]}})
	require.Equal(t, 5, sm.headerList.Len())
	require.Equal(t, headers[3].BlockHeader, *sm.syncHeaderCtx.Parent)
	require.Equal(t, headers[2].BlockHeader, *sm.syncHeaderCtx.Grandparent)
	require.Zero(t, chain.BestSnapshot().Height)

	// Restarting from the stored chain discards the downloaded window. The
	// first newly accepted header must use genesis, not the old batch tip.
	sm.resetHeaderState(params.GenesisHash, 0)
	require.Nil(t, sm.syncHeaderCtx.Parent)
	require.Nil(t, sm.syncHeaderCtx.Grandparent)
	sm.headersFirstMode = true
	sm.handleHeadersMsg(&headersMsg{peer: p, headers: &wire.MsgHeaders{Headers: headers[:1]}})
	require.Equal(t, 2, sm.headerList.Len())
	require.Equal(t, headers[0].BlockHeader, *sm.syncHeaderCtx.Parent)
	require.Equal(t, genesis, *sm.syncHeaderCtx.Grandparent)
	select {
	case <-p.Done():
		t.Fatal("header processing disconnected the peer")
	default:
	}
}
