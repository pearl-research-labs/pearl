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

// TestPickStaleOutboundPeer pins which peers the no-candidate rotation
// may evict (#301): outbound peers only, strictly below our height,
// lowest advertised height first. A peer at our height is never
// selected — it may simply be waiting for the same next block we are —
// and an inbound peer is never selected because disconnecting one
// frees no outbound slot for the connection manager to refill.
func TestPickStaleOutboundPeer(t *testing.T) {
	sm := newTestSyncManager(t, &chaincfg.RegressionNetParams)

	mkPeer := func(inbound bool, addr string, height int32) *peer.Peer {
		t.Helper()
		cfg := &peer.Config{ChainParams: sm.chainParams}
		var p *peer.Peer
		if inbound {
			p = peer.NewInboundPeer(cfg)
		} else {
			var err error
			p, err = peer.NewOutboundPeer(cfg, addr)
			require.NoError(t, err)
		}
		p.UpdateLastBlockHeight(height)
		sm.peerStates[p] = &peerSyncState{syncCandidate: true}
		return p
	}

	tipHash := &chainhash.Hash{0x0a}
	otherHash := &chainhash.Hash{0x0b}

	low := mkPeer(false, "10.0.0.1:44108", 2)
	low.UpdateLastAnnouncedBlock(otherHash)
	mkPeer(false, "10.0.0.2:44108", 5)
	mkPeer(true, "10.0.0.3:44108", 1)

	// A peer advertising a low height whose last announced block is
	// our tip provably holds the tip — LastBlock is only a lower
	// bound — so it must never be rotated, even though it advertises
	// the lowest height of all.
	quiet := mkPeer(false, "10.0.0.4:44108", 1)
	quiet.UpdateLastAnnouncedBlock(tipHash)

	assert.Equal(t, low, sm.pickStaleOutboundPeer(10, tipHash),
		"the lowest strictly-behind outbound peer must be picked")
	assert.Nil(t, sm.pickStaleOutboundPeer(2, tipHash),
		"a peer at our height must never be rotated")
	assert.Nil(t, sm.pickStaleOutboundPeer(0, tipHash),
		"no peer can be below genesis height")
	assert.Equal(t, quiet, sm.pickStaleOutboundPeer(10, otherHash),
		"the announced-tip exemption must follow the actual tip hash")
}

// TestHandleStallSampleNoSyncPeer pins the stall-sample accounting for
// the #301 wedge: with no sync peer, selection is retried on every
// sample even though startSync keeps finding no viable candidate, the
// no-candidate streak keeps counting, a peer at our own height is never
// disconnected no matter how long the streak runs, and the streak
// resets once a sync peer exists again.
func TestHandleStallSampleNoSyncPeer(t *testing.T) {
	sm := newTestSyncManager(t, &chaincfg.RegressionNetParams)

	p, _ := connectedTestPeer(t, sm.chainParams, false)
	sm.handleNewPeerMsg(p)
	require.Nil(t, sm.syncPeer, "a peer at our height must not be picked as sync peer")

	samples := noSyncPeerRotateSamples + 5
	for i := 0; i < samples; i++ {
		sm.handleStallSample()
	}
	assert.Equal(t, samples, sm.noSyncPeerSamples,
		"every candidate-less sample must extend the streak")
	assert.True(t, p.Connected(),
		"a peer at our height must not be rotated out")

	// Once a sync peer exists the streak resets; the peer is at our
	// height, so the stall path must not disconnect it either.
	sm.syncPeer = p
	sm.handleStallSample()
	assert.Equal(t, 0, sm.noSyncPeerSamples,
		"having a sync peer must reset the streak")
	assert.True(t, p.Connected())
}

// TestHandleNoSyncPeerAtTip pins that the no-candidate rotation never
// runs while the chain believes it is current: at a fresh tip, having
// no sync peer is the normal steady state and the outbound peers'
// advertised heights lag the tip by construction, so there is nothing
// to rotate for. The streak is reset, not merely paused, so the grace
// period restarts when the node actually becomes stranded. The
// genesis-only test chain is never current, so the current case is
// posed by passing the flag explicitly — the same way
// TestPickStaleOutboundPeer poses the tip height.
func TestHandleNoSyncPeerAtTip(t *testing.T) {
	sm := newTestSyncManager(t, &chaincfg.RegressionNetParams)
	require.False(t, sm.chain.IsCurrent())

	p, _ := connectedTestPeer(t, sm.chainParams, false)
	sm.handleNewPeerMsg(p)
	require.Nil(t, sm.syncPeer)

	sm.noSyncPeerSamples = noSyncPeerRotateSamples + 5
	sm.handleNoSyncPeer(true)
	assert.Equal(t, 0, sm.noSyncPeerSamples,
		"a current chain must reset the no-candidate streak")
	assert.True(t, p.Connected(),
		"no peer may be rotated while the chain is current")

	// Not current: the streak counts again from zero.
	sm.handleNoSyncPeer(false)
	assert.Equal(t, 1, sm.noSyncPeerSamples)
}

// TestAnyPeerAnnouncedUnknownBlock pins the rotation stand-down
// signal: an eligible sync candidate that announced a block we do
// not have is evidence of reachable work, so rotation must not
// churn the peer set while selection retries are still working
// through the candidates.
func TestAnyPeerAnnouncedUnknownBlock(t *testing.T) {
	sm := newTestSyncManager(t, &chaincfg.RegressionNetParams)
	assert.False(t, sm.anyPeerAnnouncedUnknownBlock(),
		"no peers connected")

	p, _ := connectedTestPeer(t, sm.chainParams, false)
	sm.handleNewPeerMsg(p)
	assert.False(t, sm.anyPeerAnnouncedUnknownBlock(),
		"peer has announced nothing")

	tip := sm.chain.BestSnapshot().Hash
	p.UpdateLastAnnouncedBlock(&tip)
	assert.False(t, sm.anyPeerAnnouncedUnknownBlock(),
		"the announced block is our own tip")

	unknown := chainhash.Hash{0x07}
	p.UpdateLastAnnouncedBlock(&unknown)
	assert.True(t, sm.anyPeerAnnouncedUnknownBlock(),
		"the announced block is one we do not have")
}

// TestAnyPeerAnnouncedUnknownBlockEligibility pins which
// announcements may stand rotation down. LastAnnouncedBlock is
// written from an unauthenticated inv before any quality gate and
// is only cleared when a matching block is accepted, so an
// announcement may only count from a peer startSync could actually
// promote on a coming sample: outbound, a sync candidate,
// high-quality, and not in post-stall cooldown. Anything else —
// above all a fake inv from an inbound peer, which is never
// promoted while outbound candidates exist — must not freeze
// rotation for the whole stranded window.
func TestAnyPeerAnnouncedUnknownBlockEligibility(t *testing.T) {
	sm := newTestSyncManager(t, &chaincfg.RegressionNetParams)
	unknown := chainhash.Hash{0x07}

	mkPeer := func(inbound bool, addr string, state *peerSyncState) *peer.Peer {
		t.Helper()
		cfg := &peer.Config{ChainParams: sm.chainParams}
		var p *peer.Peer
		if inbound {
			p = peer.NewInboundPeer(cfg)
		} else {
			var err error
			p, err = peer.NewOutboundPeer(cfg, addr)
			require.NoError(t, err)
		}
		p.UpdateLastAnnouncedBlock(&unknown)
		sm.peerStates[p] = state
		return p
	}

	// Inbound announcer: never counts, even as a high-quality
	// candidate — pickSyncCandidate will not promote it while
	// outbound candidates exist.
	mkPeer(true, "", &peerSyncState{syncCandidate: true})
	assert.False(t, sm.anyPeerAnnouncedUnknownBlock(),
		"an inbound announcement must not stand rotation down")

	// Outbound but not a sync candidate.
	mkPeer(false, "10.0.0.1:44108", &peerSyncState{syncCandidate: false})
	assert.False(t, sm.anyPeerAnnouncedUnknownBlock(),
		"a non-candidate announcement must not stand rotation down")

	// Outbound candidate that has struck out to low quality: its
	// announcements are gated through the getheaders probe, not
	// acted on as sync evidence.
	mkPeer(false, "10.0.0.2:44108", &peerSyncState{
		syncCandidate: true, nonTipStrikes: lowQualityStrikeLimit,
	})
	assert.False(t, sm.anyPeerAnnouncedUnknownBlock(),
		"a low-quality announcement must not stand rotation down")

	// Eligible announcer in post-stall cooldown: startSync skips
	// it, so its stale announcement must not stand rotation down
	// either — this is what bounds the stand-down after a promoted
	// announcer stalls without delivering its block.
	cooling := mkPeer(false, "10.0.0.3:44108",
		&peerSyncState{syncCandidate: true})
	sm.recentlyFailedSync[cooling.Addr()] = time.Now()
	assert.False(t, sm.anyPeerAnnouncedUnknownBlock(),
		"a cooling-down announcement must not stand rotation down")

	// The same announcer with the cooldown expired counts again.
	sm.recentlyFailedSync[cooling.Addr()] =
		time.Now().Add(-syncPeerCooldown - time.Second)
	assert.True(t, sm.anyPeerAnnouncedUnknownBlock(),
		"an eligible announcement with an expired cooldown counts")
}

// TestAtTipSyncPromotionBlocked pins the at-tip promotion bar:
// while the chain is current, no candidate may be promoted in a way
// that flips current() false — neither on its unauthenticated
// version height alone, nor on an unverified announced hash paired
// with an advertised height above ours. The honest next-block
// announcer (announced a block we lack, advertised height at or
// below ours, because its LastBlock only advances on blocks we
// accepted) is still promoted, and nothing is barred once the
// chain is not current.
func TestAtTipSyncPromotionBlocked(t *testing.T) {
	unknown := &chainhash.Hash{0x07}

	assert.False(t, atTipSyncPromotionBlocked(false, nil, 100, 10),
		"not current: a height claim still promotes")
	assert.False(t, atTipSyncPromotionBlocked(false, unknown, 100, 10),
		"not current: an announced block still promotes")

	assert.True(t, atTipSyncPromotionBlocked(true, nil, 100, 10),
		"current + height claim only: blocked")
	assert.True(t, atTipSyncPromotionBlocked(true, nil, 10, 10),
		"current + no announcement: blocked at any height")

	assert.True(t, atTipSyncPromotionBlocked(true, unknown, 11, 10),
		"current + announced block + advertised height above ours: "+
			"blocked, the promotion would flip current() false")
	assert.False(t, atTipSyncPromotionBlocked(true, unknown, 10, 10),
		"current + announced block + advertised height at ours: "+
			"promoted, current() stays true")
	assert.False(t, atTipSyncPromotionBlocked(true, unknown, 9, 10),
		"current + announced block + advertised height below ours: "+
			"promoted, current() stays true")
}

// TestStartSyncPromotesHeightClaimWhenNotCurrent pins that the at-tip
// promotion guard does not weaken syncing on a version-height claim
// while the chain is not current (IBD / stranded): a lone outbound
// candidate advertising a height above ours is promoted on that claim
// alone, exactly as before the guard.
func TestStartSyncPromotesHeightClaimWhenNotCurrent(t *testing.T) {
	sm := newTestSyncManager(t, &chaincfg.RegressionNetParams)
	require.False(t, sm.chain.IsCurrent())

	p, _ := connectedTestPeer(t, sm.chainParams, false)
	p.UpdateLastBlockHeight(5)
	sm.handleNewPeerMsg(p)
	require.Equal(t, p, sm.syncPeer,
		"a height claim must still promote while the chain is not current")
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
