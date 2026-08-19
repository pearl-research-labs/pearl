// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package netsync

import (
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/database"
	_ "github.com/pearl-research-labs/pearl/node/database/ffldb"
	"github.com/pearl-research-labs/pearl/node/mempool"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

type stubPeerNotifier struct{}

func (stubPeerNotifier) AnnounceNewTransactions([]*mempool.TxDesc) {}
func (stubPeerNotifier) UpdatePeerHeights(*chainhash.Hash, int32, *peer.Peer) {}
func (stubPeerNotifier) RelayInventory(*wire.InvVect, interface{}) {}
func (stubPeerNotifier) TransactionConfirmed(*btcutil.Tx)          {}

func newTestSyncManager(t *testing.T) *SyncManager {
	t.Helper()
	DisableLog()

	params := chaincfg.SimNetParams
	db, err := database.Create("ffldb", t.TempDir(), params.Net)
	require.NoError(t, err)
	t.Cleanup(func() { db.Close() })

	chain, err := blockchain.New(&blockchain.Config{
		DB:          db,
		ChainParams: &params,
		TimeSource:  blockchain.NewMedianTime(),
	})
	require.NoError(t, err)

	sm, err := New(&Config{
		PeerNotifier:       stubPeerNotifier{},
		Chain:              chain,
		ChainParams:        &params,
		DisableCheckpoints: true,
		MaxPeers:           8,
	})
	require.NoError(t, err)
	sm.Start()
	t.Cleanup(func() {
		done := make(chan error, 1)
		go func() { done <- sm.Stop() }()
		select {
		case err := <-done:
			require.NoError(t, err)
		case <-time.After(5 * time.Second):
			require.Fail(t, "sync manager Stop timed out")
		}
	})
	return sm
}

func requireHandlerLive(t *testing.T, sm *SyncManager) {
	t.Helper()
	done := make(chan struct{})
	go func() {
		_ = sm.IsCurrent()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		require.FailNow(t, "sync manager block handler did not respond")
	}
}

func TestProcessBlock_RejectedKeepsHandlerAlive(t *testing.T) {
	sm := newTestSyncManager(t)
	genesis := btcutil.NewBlock(chaincfg.SimNetParams.GenesisBlock)

	_, err := sm.ProcessBlock(genesis, blockchain.BFNone)
	require.Error(t, err)

	requireHandlerLive(t, sm)

	_, err = sm.ProcessBlock(genesis, blockchain.BFNone)
	require.Error(t, err)
	requireHandlerLive(t, sm)
}
