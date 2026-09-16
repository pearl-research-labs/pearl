// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package netsync

import (
	"path/filepath"
	"testing"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/database"
	_ "github.com/pearl-research-labs/pearl/node/database/ffldb"
	"github.com/pearl-research-labs/pearl/node/mempool"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type noopPeerNotifier struct{}

func (noopPeerNotifier) AnnounceNewTransactions([]*mempool.TxDesc)            {}
func (noopPeerNotifier) UpdatePeerHeights(*chainhash.Hash, int32, *peer.Peer) {}
func (noopPeerNotifier) RelayInventory(*wire.InvVect, interface{})            {}
func (noopPeerNotifier) TransactionConfirmed(*btcutil.Tx)                     {}

// newTestSyncManager returns a sync manager over a fresh genesis-only chain. The params are copied so tests cannot
// mutate the package-level instances.
func newTestSyncManager(t *testing.T, params *chaincfg.Params) *SyncManager {
	t.Helper()

	paramsCopy := *params
	db, err := database.Create("ffldb", filepath.Join(t.TempDir(), "ffldb"), paramsCopy.Net)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, db.Close()) })

	chain, err := blockchain.New(&blockchain.Config{
		DB:          db,
		ChainParams: &paramsCopy,
		TimeSource:  blockchain.NewMedianTime(),
		SigCache:    txscript.NewSigCache(0),
		HashCache:   txscript.NewHashCache(0),
	})
	require.NoError(t, err)

	sm, err := New(&Config{
		PeerNotifier: noopPeerNotifier{},
		Chain:        chain,
		ChainParams:  &paramsCopy,
	})
	require.NoError(t, err)

	return sm
}

// TestIsSyncCandidate pins the services a peer must advertise to be a sync candidate. Regtest gets no localhost
// exemption: an integration harness that wants to feed blocks has to advertise that it serves them.
func TestIsSyncCandidate(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		flags     wire.ServiceFlag
		lastBlock int32
		want      bool
	}{
		{name: "just node network", flags: wire.SFNodeNetwork, want: true},
		{name: "just limited network", flags: wire.SFNodeNetworkLimited, want: true},
		{
			name:      "limited network with block ahead",
			flags:     wire.SFNodeNetworkLimited,
			lastBlock: wire.NodeNetworkLimitedBlockThreshold + 1,
		},
		{
			name:  "node network and limited node network",
			flags: wire.SFNodeNetwork | wire.SFNodeNetworkLimited,
			want:  true,
		},
		{name: "no flags"},
		{name: "different flag", flags: wire.SFNodeBloom},
	}

	for _, params := range []*chaincfg.Params{&chaincfg.RegressionNetParams, &chaincfg.SimNetParams} {
		t.Run(params.Name, func(t *testing.T) {
			t.Parallel()

			sm := newTestSyncManager(t, params)
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					p := peer.NewInboundPeer(&peer.Config{ChainParams: sm.chainParams, Services: tt.flags})
					p.UpdateLastBlockHeight(tt.lastBlock)

					assert.Equal(t, tt.want, sm.isSyncCandidate(p))
				})
			}
		})
	}
}
