// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package netsync

import (
	"errors"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
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
