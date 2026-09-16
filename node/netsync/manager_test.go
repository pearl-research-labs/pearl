// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package netsync

import (
	"net"
	"os"
	"path/filepath"
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
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The package logger is nil until the node wires it up; handlers log unconditionally.
func TestMain(m *testing.M) {
	DisableLog()
	os.Exit(m.Run())
}

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

// remoteInbox collects the block requests the remote end of a test connection receives.
type remoteInbox struct {
	getHeaders chan *wire.MsgGetHeaders
	getData    chan *wire.MsgGetData
}

// connectedTestPeer hands back a peer with a live loopback connection; QueueMessage drops messages on peers
// without one, so nothing less lets a test see what was sent.
func connectedTestPeer(t *testing.T, params *chaincfg.Params, inbound bool) (*peer.Peer, *remoteInbox) {
	t.Helper()

	inbox := &remoteInbox{
		getHeaders: make(chan *wire.MsgGetHeaders, 8),
		getData:    make(chan *wire.MsgGetData, 8),
	}
	verack := make(chan struct{}, 2)
	newCfg := func(listeners peer.MessageListeners) *peer.Config {
		listeners.OnVerAck = func(*peer.Peer, *wire.MsgVerAck) { verack <- struct{}{} }
		return &peer.Config{
			Listeners:        listeners,
			UserAgentName:    "netsync-test",
			UserAgentVersion: "1.0",
			ChainParams:      params,
			ProtocolVersion:  wire.ProtocolVersion,
			Services:         wire.SFNodeNetwork | wire.SFNodeWitness,
			TrickleInterval:  10 * time.Second,
			AllowSelfConns:   true,
		}
	}
	localCfg := newCfg(peer.MessageListeners{})
	remoteCfg := newCfg(peer.MessageListeners{
		OnGetHeaders: func(_ *peer.Peer, msg *wire.MsgGetHeaders) { inbox.getHeaders <- msg },
		OnGetData:    func(_ *peer.Peer, msg *wire.MsgGetData) { inbox.getData <- msg },
	})

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	t.Cleanup(func() { _ = listener.Close() })

	var local, remote, accepting, dialing *peer.Peer
	if inbound {
		local = peer.NewInboundPeer(localCfg)
		remote, err = peer.NewOutboundPeer(remoteCfg, listener.Addr().String())
		require.NoError(t, err)
		accepting, dialing = local, remote
	} else {
		remote = peer.NewInboundPeer(remoteCfg)
		local, err = peer.NewOutboundPeer(localCfg, listener.Addr().String())
		require.NoError(t, err)
		accepting, dialing = remote, local
	}

	accepted := make(chan error, 1)
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			accepted <- err
			return
		}
		accepting.AssociateConnection(conn)
		accepted <- nil
	}()
	conn, err := net.Dial("tcp", listener.Addr().String())
	require.NoError(t, err)
	dialing.AssociateConnection(conn)
	require.NoError(t, <-accepted)

	t.Cleanup(func() {
		local.Disconnect()
		remote.Disconnect()
		local.WaitForDisconnect()
		remote.WaitForDisconnect()
	})

	for i := 0; i < 2; i++ {
		select {
		case <-verack:
		case <-time.After(5 * time.Second):
			require.FailNow(t, "handshake did not complete")
		}
	}

	return local, inbox
}

func expectNone[T any](t *testing.T, ch <-chan T, what string) {
	t.Helper()

	select {
	case msg := <-ch:
		require.Failf(t, "unexpected message", "%s: %v", what, msg)
	case <-time.After(100 * time.Millisecond):
	}
}

// TestInvGateWithoutSyncPeer pins block-inv handling while not current and no sync peer is set: low-quality peers
// are probed, high-quality peers are served directly, and once a sync peer exists other peers are ignored.
//
// Not parallel: blockchain.New mutates deployment starters shared by every copy of the same params.
func TestInvGateWithoutSyncPeer(t *testing.T) {
	tests := []struct {
		name           string
		inbound        bool
		witnessInv     bool
		withSyncPeer   bool
		wantGetHeaders bool
		wantGetData    bool
	}{
		{name: "low-quality peer is probed", inbound: true, wantGetHeaders: true},
		{name: "low-quality peer is probed for witness invs", inbound: true, witnessInv: true, wantGetHeaders: true},
		{name: "high-quality peer is served directly", wantGetData: true},
		{name: "non-sync peer is dropped once a sync peer exists", inbound: true, withSyncPeer: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sm := newTestSyncManager(t, &chaincfg.RegressionNetParams)
			require.False(t, sm.current(), "a genesis-only chain must not be current")

			p, inbox := connectedTestPeer(t, sm.chainParams, tt.inbound)
			sm.handleNewPeerMsg(p)
			require.Nil(t, sm.syncPeer, "a peer at our height must not be picked as sync peer")

			if tt.withSyncPeer {
				syncPeer, _ := connectedTestPeer(t, sm.chainParams, false)
				sm.handleNewPeerMsg(syncPeer)
				sm.syncPeer = syncPeer
			}

			announced := chainhash.Hash{0x01}
			invType := wire.InvTypeBlock
			if tt.witnessInv {
				invType = wire.InvTypeWitnessBlock
			}
			inv := wire.NewMsgInv()
			require.NoError(t, inv.AddInvVect(wire.NewInvVect(invType, &announced)))
			sm.handleInvMsg(&invMsg{inv: inv, peer: p})

			if tt.wantGetHeaders {
				select {
				case msg := <-inbox.getHeaders:
					assert.Equal(t, announced, msg.HashStop, "probe must stop at the announced block")
					assert.False(t, msg.IncludeCertificates, "probe must be cert-less")
				case <-time.After(5 * time.Second):
					require.FailNow(t, "expected a getheaders probe")
				}
			} else {
				expectNone(t, inbox.getHeaders, "getheaders")
			}

			_, requested := sm.requestedBlocks[announced]
			assert.Equal(t, tt.wantGetData, requested, "requestedBlocks must mirror the getdata decision")
			if tt.wantGetData {
				select {
				case msg := <-inbox.getData:
					require.Len(t, msg.InvList, 1)
					assert.Equal(t, announced, msg.InvList[0].Hash)
				case <-time.After(5 * time.Second):
					require.FailNow(t, "expected a getdata for the announced block")
				}
			} else {
				expectNone(t, inbox.getData, "getdata")
			}
		})
	}
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
