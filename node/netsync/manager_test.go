// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package netsync

import (
	"bytes"
	"fmt"
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
	"github.com/pearl-research-labs/pearl/node/integration/rpctest"
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
//
// Announcing our tip does NOT exempt a peer. LastAnnouncedBlock is
// written in handleInvMsg from an unauthenticated inv, before any
// possession check, and the node never requests a hash it already
// has — so in the not-current state where rotation runs, an
// announcement of the tip is unverifiable: the tip hash is public
// (any peer can learn it, e.g. via getblocks) and a peer that is
// still behind can repeat it without holding the block. The height
// update that would prove possession only runs while the chain is
// current (handleInvMsg), and rotation never runs then. A peer that
// announced our tip while we were current already had its LastBlock
// raised to the tip height by that path, so the height rule above
// protects every verifiable holder; the unverifiable claim must not
// protect a liar instead.
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

	// A peer advertising the lowest height whose last announced
	// block is our tip: the announcement is an unauthenticated inv
	// of a public hash, not proof of possession, so it must not
	// exempt the peer — it is picked first, ahead of the
	// higher-advertising peer that announced some other hash.
	quiet := mkPeer(false, "10.0.0.4:44108", 1)
	quiet.UpdateLastAnnouncedBlock(tipHash)

	assert.Equal(t, quiet, sm.pickStaleOutboundPeer(10),
		"announcing the tip must not exempt a strictly-behind peer: "+
			"the announcement is unverified and the hash is public")
	delete(sm.peerStates, quiet)
	assert.Equal(t, low, sm.pickStaleOutboundPeer(10),
		"the lowest remaining strictly-behind outbound peer must be picked")
	assert.Nil(t, sm.pickStaleOutboundPeer(2),
		"a peer at our height must never be rotated")
	assert.Nil(t, sm.pickStaleOutboundPeer(0),
		"no peer can be below genesis height")
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
// while the chain is current, no candidate may be promoted at all.
// The pre-promotion checks cannot make a promotion safe: they only
// inspect the candidate's advertised height at selection time, but
// once the peer is the sync peer an orphan it serves raises its
// LastBlock to the block's unauthenticated coinbase height
// (handleBlockMsg), which flips current() false and silences every
// other peer's announcements until the stall timer fires. Nothing
// is barred once the chain is not current, and while current a
// promotion is not needed to fetch an announced block: with no
// sync peer, handleInvMsg fetches announcements directly.
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
	assert.True(t, atTipSyncPromotionBlocked(true, unknown, 10, 10),
		"current + announced block + advertised height at ours: "+
			"blocked, the promoted peer can raise its height on "+
			"an orphan's coinbase claim after promotion")
	assert.True(t, atTipSyncPromotionBlocked(true, unknown, 9, 10),
		"current + announced block + advertised height below ours: "+
			"blocked, the promoted peer can raise its height on "+
			"an orphan's coinbase claim after promotion")
}

// TestAnyPeerAnnouncedUnknownBlockSpent pins the stand-down bound
// when there are more announcers than fit inside one cooldown
// window. The previous bound relied on every announcer sitting in
// syncPeerCooldown at once; with the default eight outbound peers
// that can never happen — each promotion burns a full
// maxStallDuration before its announcer enters the 10-minute
// cooldown, so early cooldowns expire while later announcers are
// still being tried, and some eligible announcer is always outside
// cooldown, standing rotation down forever. An announcer that has
// already stalled once over the very hash it announced has had its
// chance: that announcement must never stand rotation down again,
// however long ago the stall was. A new announcement from the same
// peer counts again.
func TestAnyPeerAnnouncedUnknownBlockSpent(t *testing.T) {
	sm := newTestSyncManager(t, &chaincfg.RegressionNetParams)
	unknown := chainhash.Hash{0x07}

	mkPeer := func(addr string) *peer.Peer {
		t.Helper()
		p, err := peer.NewOutboundPeer(
			&peer.Config{ChainParams: sm.chainParams}, addr)
		require.NoError(t, err)
		h := unknown
		p.UpdateLastAnnouncedBlock(&h)
		sm.peerStates[p] = &peerSyncState{syncCandidate: true}
		return p
	}

	// More announcers than a 10-minute cooldown divided by a
	// 3-minute stall cycle can hold at once.
	announcers := make([]*peer.Peer, 0, 5)
	for i := 0; i < 5; i++ {
		announcers = append(announcers,
			mkPeer(fmt.Sprintf("10.0.1.%d:44108", i+1)))
	}
	require.True(t, sm.anyPeerAnnouncedUnknownBlock(),
		"eligible announcers stand rotation down before any has stalled")

	// Each announcer in turn is promoted, stalls over its
	// announcement, and cools down; by the time the later ones
	// have stalled, the earlier cooldowns have expired, so no
	// announcer is in cooldown when the last one finishes.
	for _, p := range announcers {
		sm.coolDownSyncCandidate(p)
		sm.recentlyFailedSync[p.Addr()] =
			time.Now().Add(-syncPeerCooldown - time.Second)
	}
	assert.False(t, sm.anyPeerAnnouncedUnknownBlock(),
		"every announcer has stalled over its announcement: the "+
			"stand-down must lift even with nobody in cooldown")

	// A fresh announcement from a spent announcer counts again.
	fresh := chainhash.Hash{0x09}
	announcers[0].UpdateLastAnnouncedBlock(&fresh)
	assert.True(t, sm.anyPeerAnnouncedUnknownBlock(),
		"a new announcement from a spent announcer counts again")
}

// TestStartSyncSkipsSpentAnnouncement pins that an announcement a
// peer has already stalled over is nothing new to startSync. The
// stand-down bound (announcementSpent) only lifts rotation's freeze
// if the spent announcer cannot simply be promoted again on the same
// hash once its cooldown expires: each such promotion also zeroes
// noSyncPeerSamples in handleStallSample, restarting the rotation
// grace, so a cycle of three or more spent announcers — each
// promoted in turn, stalling for maxStallDuration, then cooling
// down — keeps the grace from ever elapsing and rotation never
// disconnects anyone. A spent candidate is skipped exactly like a
// candidate whose announced block we already have. A fresh
// announcement from the same peer promotes again.
func TestStartSyncSkipsSpentAnnouncement(t *testing.T) {
	sm := newTestSyncManager(t, &chaincfg.RegressionNetParams)
	require.False(t, sm.chain.IsCurrent())

	p, _ := connectedTestPeer(t, sm.chainParams, false)
	spent := chainhash.Hash{0x07}
	p.UpdateLastAnnouncedBlock(&spent)
	if sm.announcementSpent == nil {
		sm.announcementSpent = make(map[string]chainhash.Hash)
	}
	sm.announcementSpent[p.Addr()] = spent

	sm.handleNewPeerMsg(p)
	assert.Nil(t, sm.syncPeer,
		"a candidate whose only announcement is spent must not be promoted")

	// A new announcement from the same peer is new evidence and
	// promotes as usual.
	fresh := chainhash.Hash{0x09}
	p.UpdateLastAnnouncedBlock(&fresh)
	sm.startSync()
	assert.Equal(t, p, sm.syncPeer,
		"a fresh announcement from a spent announcer must promote")
}

// TestHandleStallSampleSpentKeepsStreak pins the rotation-grace half
// of the spent-promotion fix: a stall sample whose only candidate
// carries a spent announcement must extend the no-candidate streak,
// not reset it via a promotion that can never deliver.
func TestHandleStallSampleSpentKeepsStreak(t *testing.T) {
	sm := newTestSyncManager(t, &chaincfg.RegressionNetParams)

	p, _ := connectedTestPeer(t, sm.chainParams, false)
	spent := chainhash.Hash{0x07}
	p.UpdateLastAnnouncedBlock(&spent)
	if sm.announcementSpent == nil {
		sm.announcementSpent = make(map[string]chainhash.Hash)
	}
	sm.announcementSpent[p.Addr()] = spent
	sm.handleNewPeerMsg(p)
	require.Nil(t, sm.syncPeer)

	sm.noSyncPeerSamples = 3
	sm.handleStallSample()
	assert.Nil(t, sm.syncPeer,
		"the spent announcer must not be promoted on a stall sample")
	assert.Equal(t, 4, sm.noSyncPeerSamples,
		"a candidate-less sample must extend the streak, not reset it")
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

// TestLimitAdd pins that limitAdd evicts only to make room for a key it doesn't already hold.
func TestLimitAdd(t *testing.T) {
	t.Parallel()

	const limit = 16
	full := func() map[chainhash.Hash]struct{} {
		m := make(map[chainhash.Hash]struct{}, limit)
		for i := byte(0); i < limit; i++ {
			m[chainhash.Hash{i}] = struct{}{}
		}
		return m
	}

	t.Run("present keys at the limit", func(t *testing.T) {
		t.Parallel()

		m := full()
		for i := byte(0); i < limit; i++ {
			limitAdd(m, chainhash.Hash{i}, limit)
		}
		assert.Equal(t, full(), m)
	})

	t.Run("new key at the limit", func(t *testing.T) {
		t.Parallel()

		m := full()
		added := chainhash.Hash{limit}
		limitAdd(m, added, limit)
		assert.Len(t, m, limit)
		assert.Contains(t, m, added)
	})
}

type testNode struct {
	t  *testing.T
	sm *SyncManager
}

type testPeer struct {
	node  *testNode
	peer  *peer.Peer
	inbox *remoteInbox
}

// newTestNode gives the genesis block a current timestamp: on a chain still syncing, handleStallSample would start a
// sync that resets the global request map and hide whether a timeout released a request.
func newTestNode(t *testing.T) *testNode {
	t.Helper()

	params := chaincfg.SimNetParams
	genesis := params.GenesisBlock.Copy()
	genesis.BlockHeader().Timestamp = time.Now().UTC().Truncate(time.Second)
	genesisHash := genesis.BlockHash()
	params.GenesisBlock = genesis
	params.GenesisHash = &genesisHash

	sm := newTestSyncManager(t, &params)
	require.True(t, sm.current())
	return &testNode{t: t, sm: sm}
}

func newCatchingUpTestNode(t *testing.T) *testNode {
	t.Helper()

	sm := newTestSyncManager(t, &chaincfg.SimNetParams)
	require.False(t, sm.current())
	return &testNode{t: t, sm: sm}
}

// outboundPeer is the default because only high-quality peers are asked for announced blocks directly.
func (n *testNode) outboundPeer() *testPeer {
	n.t.Helper()
	return n.connect(false)
}

func (n *testNode) connect(inbound bool) *testPeer {
	n.t.Helper()

	p, inbox := connectedTestPeer(n.t, n.sm.chainParams, inbound)
	n.sm.handleNewPeerMsg(p)
	return &testPeer{node: n, peer: p, inbox: inbox}
}

// block pays the coinbase to a key derived from seed so siblings built on the same parent get distinct hashes.
func (n *testNode) block(parent *btcutil.Block, seed byte) *btcutil.Block {
	n.t.Helper()

	addr, err := btcutil.NewAddressTaproot(bytes.Repeat([]byte{seed}, 32), n.sm.chainParams)
	require.NoError(n.t, err)
	block, err := rpctest.CreateBlock(parent, nil, rpctest.BlockVersion, time.Time{}, addr, nil, n.sm.chainParams)
	require.NoError(n.t, err)
	return block
}

func (n *testNode) stallCheck() {
	n.sm.handleStallSample()
}

func (n *testNode) tip() chainhash.Hash {
	return n.sm.chain.BestSnapshot().Hash
}

func (n *testNode) fetching(b *btcutil.Block) bool {
	_, requested := n.sm.requestedBlocks[*b.Hash()]
	return requested
}

func (p *testPeer) announce(blocks ...*btcutil.Block) {
	p.node.t.Helper()

	inv := wire.NewMsgInv()
	for _, b := range blocks {
		require.NoError(p.node.t, inv.AddInvVect(wire.NewInvVect(wire.InvTypeBlock, b.Hash())))
	}
	p.node.sm.handleInvMsg(&invMsg{inv: inv, peer: p.peer})
}

func (p *testPeer) expectAskedFor(blocks ...*btcutil.Block) {
	p.node.t.Helper()

	select {
	case msg := <-p.inbox.getData:
		require.Len(p.node.t, msg.InvList, len(blocks))
		for i, iv := range msg.InvList {
			require.Equal(p.node.t, *blocks[i].Hash(), iv.Hash)
		}
	case <-time.After(5 * time.Second):
		require.FailNow(p.node.t, "expected a getdata for the blocks")
	}
}

func (p *testPeer) expectNotAsked(why string) {
	p.node.t.Helper()
	expectNone(p.node.t, p.inbox.getData, why)
}

func (p *testPeer) deliver(b *btcutil.Block) {
	p.node.t.Helper()
	require.NoError(p.node.t, p.node.sm.handleBlockMsg(&blockMsg{block: b, peer: p.peer}))
}

func (p *testPeer) deliverDuplicate(b *btcutil.Block) {
	p.node.t.Helper()

	err := p.node.sm.handleBlockMsg(&blockMsg{block: b, peer: p.peer})
	var ruleErr blockchain.RuleError
	require.ErrorAs(p.node.t, err, &ruleErr)
	require.Equal(p.node.t, blockchain.ErrDuplicateBlock, ruleErr.ErrorCode)
}

func (p *testPeer) notFound(invs ...*wire.InvVect) {
	p.node.t.Helper()

	reply := wire.NewMsgNotFound()
	for _, iv := range invs {
		require.NoError(p.node.t, reply.AddInvVect(iv))
	}
	p.node.sm.handleNotFoundMsg(&notFoundMsg{notFound: reply, peer: p.peer})
}

func (p *testPeer) sendHeaders(blocks ...*btcutil.Block) {
	p.node.t.Helper()

	msg := wire.NewMsgHeaders()
	for _, b := range blocks {
		require.NoError(p.node.t, msg.AddBlockHeader(*b.MsgBlock().BlockHeader(), nil))
	}
	p.node.sm.handleHeadersMsg(&headersMsg{headers: msg, peer: p.peer})
}

func (p *testPeer) owes(b *btcutil.Block) bool {
	_, owed := p.node.sm.peerStates[p.peer].requestedBlocks[*b.Hash()]
	return owed
}

func (p *testPeer) idleFor(d time.Duration) {
	p.node.sm.peerStates[p.peer].blockProgress = time.Now().Add(-d)
}

func (p *testPeer) missDeadline() {
	p.idleFor(blockDeliveryTimeout + time.Second)
}

func (p *testPeer) connected() bool {
	return p.peer.Connected()
}

func (p *testPeer) leave() {
	p.node.sm.handleDonePeerMsg(p.peer)
}

// TestOnlyExactBlockNotFoundSettlesRequest pins that a notfound settles a block request only when it names that
// block. Any other reply must leave the peer on the delivery deadline, or it could hold the request forever.
func TestOnlyExactBlockNotFoundSettlesRequest(t *testing.T) {
	tests := []struct {
		name    string
		reply   func(requested, other *btcutil.Block) []*wire.InvVect
		settles bool
	}{
		{
			name:  "empty",
			reply: func(_, _ *btcutil.Block) []*wire.InvVect { return nil },
		},
		{
			name: "another block",
			reply: func(_, other *btcutil.Block) []*wire.InvVect {
				return []*wire.InvVect{wire.NewInvVect(wire.InvTypeWitnessBlock, other.Hash())}
			},
		},
		{
			name: "transaction with the block's hash",
			reply: func(requested, _ *btcutil.Block) []*wire.InvVect {
				return []*wire.InvVect{wire.NewInvVect(wire.InvTypeWitnessTx, requested.Hash())}
			},
		},
		{
			name: "the requested block",
			reply: func(requested, _ *btcutil.Block) []*wire.InvVect {
				return []*wire.InvVect{wire.NewInvVect(wire.InvTypeWitnessBlock, requested.Hash())}
			},
			settles: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := newTestNode(t)
			p := node.outboundPeer()
			requested := node.block(nil, 0)
			other := node.block(nil, 1)
			p.announce(requested)
			p.expectAskedFor(requested)

			p.notFound(tt.reply(requested, other)...)
			assert.Equal(t, !tt.settles, node.fetching(requested))

			node.stallCheck()
			require.True(t, p.connected(), "the delivery deadline has not passed")

			p.missDeadline()
			node.stallCheck()
			assert.Equal(t, tt.settles, p.connected(), "only a peer that still owes the block is dropped")

			p.leave()
			assert.False(t, node.fetching(requested), "other announcers must be able to fetch the block")
		})
	}
}

// TestOnlyConnectingBlockResetsDeliveryDeadline pins that orphans do not count as delivery progress; a peer could
// otherwise hold a request forever by feeding them.
func TestOnlyConnectingBlockResetsDeliveryDeadline(t *testing.T) {
	tests := []struct {
		name          string
		deliverOrphan bool
	}{
		{name: "orphan", deliverOrphan: true},
		{name: "connecting block"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := newTestNode(t)
			p := node.outboundPeer()
			parent := node.block(nil, 0)
			child := node.block(parent, 0)
			p.announce(parent, child)
			p.expectAskedFor(parent, child)
			p.missDeadline()

			delivered, owed := parent, child
			if tt.deliverOrphan {
				delivered, owed = child, parent
			}
			p.deliver(delivered)
			require.True(t, p.owes(owed), "the peer still owes the other block")

			node.stallCheck()
			assert.Equal(t, !tt.deliverOrphan, p.connected(), "only a connecting block restarts the deadline")
		})
	}
}

// TestOnlyConnectingDuplicateResetsDeliveryDeadline pins that a peer is not dropped for losing the race to deliver a
// block that another peer was also asked for, unless the block is an orphan: colluding peers could otherwise hold a
// request forever by delivering the same orphans.
func TestOnlyConnectingDuplicateResetsDeliveryDeadline(t *testing.T) {
	tests := []struct {
		name   string
		orphan bool
	}{
		{name: "connecting block"},
		{name: "orphan", orphan: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := newTestNode(t)
			slow := node.outboundPeer()
			fast := node.outboundPeer()
			var missingParent *btcutil.Block
			if tt.orphan {
				missingParent = node.block(nil, 0)
			}
			raced := node.block(missingParent, 0)
			child := node.block(raced, 0)
			other := node.block(nil, 1)
			slow.announce(raced, other)
			slow.expectAskedFor(raced, other)

			fast.announce(child)
			fast.expectAskedFor(child)
			fast.deliver(child)
			fast.expectAskedFor(raced)
			fast.deliver(raced)

			slow.missDeadline()
			slow.deliverDuplicate(raced)
			require.True(t, slow.owes(other), "the slow peer still owes the other block")

			node.stallCheck()
			assert.Equal(t, !tt.orphan, slow.connected(), "only a duplicate of a connecting block restarts the deadline")
		})
	}
}

// TestPeerOwingNothingIsNotDropped pins that a peer stops owing a block once it delivers it, even when another peer
// delivered it first, so it can go idle afterwards without missing a deadline.
func TestPeerOwingNothingIsNotDropped(t *testing.T) {
	tests := []struct {
		name    string
		deliver func(node *testNode, p *testPeer, b *btcutil.Block)
	}{
		{name: "delivered", deliver: func(_ *testNode, p *testPeer, b *btcutil.Block) { p.deliver(b) }},
		{name: "lost the delivery race", deliver: func(node *testNode, slow *testPeer, raced *btcutil.Block) {
			fast := node.outboundPeer()
			child := node.block(raced, 0)
			fast.announce(child)
			fast.expectAskedFor(child)
			fast.deliver(child)
			fast.expectAskedFor(raced)
			fast.deliver(raced)
			slow.deliverDuplicate(raced)
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := newTestNode(t)
			p := node.outboundPeer()
			b := node.block(nil, 0)
			p.announce(b)
			p.expectAskedFor(b)
			tt.deliver(node, p, b)

			p.missDeadline()
			node.stallCheck()
			assert.True(t, p.connected(), "a peer owing nothing has no deadline")
		})
	}
}

// TestBlockDeadlineStartsAtFirstOutstandingRequest pins that idle time is not held against a new request, and that
// further requests cannot push back the deadline of one already outstanding.
func TestBlockDeadlineStartsAtFirstOutstandingRequest(t *testing.T) {
	node := newTestNode(t)
	p := node.outboundPeer()
	first := node.block(nil, 0)
	second := node.block(nil, 1)

	p.idleFor(time.Hour)
	p.announce(first)
	p.expectAskedFor(first)
	node.stallCheck()
	require.True(t, p.connected(), "idle time must not count against a new request")

	p.missDeadline()
	p.announce(second)
	p.expectAskedFor(second)
	node.stallCheck()
	assert.False(t, p.connected(), "a later request must not extend the deadline")
}

// TestWithholdingPeerIsNotPickedAsSyncPeer pins that an outbound peer dropped for withholding blocks is kept out of
// sync-peer selection, as the stall watchdog does for a stalled sync peer.
func TestWithholdingPeerIsNotPickedAsSyncPeer(t *testing.T) {
	node := newTestNode(t)
	withholder := node.outboundPeer()
	withheld := node.block(nil, 0)
	withholder.announce(withheld)
	withholder.expectAskedFor(withheld)

	withholder.missDeadline()
	node.stallCheck()
	require.False(t, withholder.connected())
	assert.Nil(t, node.sm.pickSyncCandidate(), "a peer dropped for withholding must not be a sync candidate")
}

// TestOrphanSenderIsAskedForParentHeldElsewhere pins that a peer withholding a block cannot keep the chain stuck by
// refreshing its own deadline with side-chain blocks: the sender of an orphan built on that block is asked for it.
func TestOrphanSenderIsAskedForParentHeldElsewhere(t *testing.T) {
	node := newTestNode(t)
	withholder := node.outboundPeer()
	honest := node.outboundPeer()
	base := node.block(nil, 0)
	side := node.block(nil, 1)
	withheld := node.block(base, 0)
	child := node.block(withheld, 0)

	honest.announce(base)
	honest.expectAskedFor(base)
	honest.deliver(base)
	withholder.announce(withheld)
	withholder.expectAskedFor(withheld)

	withholder.missDeadline()
	withholder.announce(side)
	withholder.expectAskedFor(side)
	withholder.deliver(side)
	require.Equal(t, *base.Hash(), node.tip(), "the side block must not become the tip")
	node.stallCheck()
	require.True(t, withholder.connected(), "a side-chain block restarts the withholder's deadline")

	honest.announce(child)
	honest.expectAskedFor(child)
	honest.deliver(child)
	honest.expectAskedFor(withheld)

	honest.deliver(withheld)
	assert.Equal(t, *child.Hash(), node.tip())
	assert.True(t, honest.connected(), "delivering the requested parent is not misbehavior")
	require.True(t, withholder.connected(), "the chain recovers without waiting for the withholder to be dropped")

	withholder.missDeadline()
	node.stallCheck()
	assert.False(t, withholder.connected(), "the withholder still owes the block it withheld")
}

// TestOrphanParentIsRequestedOncePerSender pins that asking orphan senders for a held parent does not fan out to every
// announcer: only a peer that delivered an orphan is asked, and never twice.
func TestOrphanParentIsRequestedOncePerSender(t *testing.T) {
	node := newTestNode(t)
	withholder := node.outboundPeer()
	first := node.outboundPeer()
	second := node.outboundPeer()
	third := node.outboundPeer()
	withheld := node.block(nil, 0)
	child := node.block(withheld, 0)
	grandchild := node.block(child, 0)
	greatGrandchild := node.block(grandchild, 0)
	withholder.announce(withheld)
	withholder.expectAskedFor(withheld)

	first.announce(child)
	first.expectAskedFor(child)
	second.announce(child)
	third.announce(child)
	first.deliver(child)
	first.expectAskedFor(withheld)

	second.announce(child)
	third.announce(withheld, child)
	second.expectNotAsked("getdata to a peer that only announced the orphan")
	third.expectNotAsked("getdata to a peer that only listed the parent")

	first.announce(grandchild)
	first.expectAskedFor(grandchild)
	first.deliver(grandchild)
	first.expectNotAsked("a second getdata for the parent to the same sender")

	second.announce(greatGrandchild)
	second.expectAskedFor(greatGrandchild)
	second.deliver(greatGrandchild)
	second.expectAskedFor(withheld)
	third.expectNotAsked("getdata to a peer that never delivered an orphan")
}

// TestMissingAncestorIsRequestedOnlyAtTip pins that an orphan's sender is asked for the missing ancestor only once the
// chain is current: while catching up, walking back from an orphan far ahead one block at a time thrashes the orphan
// pool. A sync peer reporting a greater height must not suppress it at the tip, since every orphan raises its
// sender's height.
func TestMissingAncestorIsRequestedOnlyAtTip(t *testing.T) {
	tests := []struct {
		name          string
		newNode       func(*testing.T) *testNode
		syncPeerAhead bool
		wantAsked     bool
	}{
		{name: "catching up", newNode: newCatchingUpTestNode},
		{name: "at tip", newNode: newTestNode, wantAsked: true},
		{name: "at tip behind a sync peer", newNode: newTestNode, syncPeerAhead: true, wantAsked: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := tt.newNode(t)
			sender := node.outboundPeer()
			if tt.syncPeerAhead {
				sender.peer.UpdateLastBlockHeight(10)
				node.sm.syncPeer = sender.peer
				require.False(t, node.sm.current())
			}
			parent := node.block(nil, 0)
			child := node.block(parent, 0)

			sender.announce(child)
			sender.expectAskedFor(child)
			sender.deliver(child)
			if tt.wantAsked {
				sender.expectAskedFor(parent)
			} else {
				sender.expectNotAsked("getdata for the missing ancestor while catching up")
			}
		})
	}
}

// TestReleasedBlockIsRequestedOnlyFromOutboundAnnouncers pins who is asked for a block once the peer holding its
// request leaves or answers notfound for it: an outbound peer that announced it, while the chain is current. Honest
// peers announce a block once, so nothing else would request it before the next block. Inbound peers can be opened at
// will, a peer asked for a block it lacks answers notfound and is penalized for it, the peer that answered notfound
// would only repeat itself, and while catching up the sync peer drives requests.
func TestReleasedBlockIsRequestedOnlyFromOutboundAnnouncers(t *testing.T) {
	releases := []struct {
		name    string
		release func(node *testNode, holder *testPeer, b *btcutil.Block)
	}{
		{name: "dropped", release: func(node *testNode, holder *testPeer, _ *btcutil.Block) {
			holder.missDeadline()
			node.stallCheck()
			require.False(node.t, holder.connected())
			holder.leave()
		}},
		{name: "notfound", release: func(_ *testNode, holder *testPeer, b *btcutil.Block) {
			holder.notFound(wire.NewInvVect(wire.InvTypeWitnessBlock, b.Hash()))
		}},
	}
	candidates := []struct {
		name       string
		catchingUp bool
		inbound    bool
		announced  bool
		wantAsked  bool
	}{
		{name: "outbound announcer", announced: true, wantAsked: true},
		{name: "inbound announcer", inbound: true, announced: true},
		{name: "peer that announced another block"},
		{name: "outbound announcer while catching up", catchingUp: true, announced: true},
	}

	for _, rr := range releases {
		t.Run(rr.name, func(t *testing.T) {
			for _, tt := range candidates {
				t.Run(tt.name, func(t *testing.T) {
					newNode := newTestNode
					if tt.catchingUp {
						newNode = newCatchingUpTestNode
					}
					node := newNode(t)
					holder := node.outboundPeer()
					candidate := node.connect(tt.inbound)
					b := node.block(nil, 0)
					if !tt.announced {
						other := node.block(nil, 1)
						candidate.announce(other)
						candidate.expectAskedFor(other)
						candidate.deliver(other)
					}
					holder.announce(b)
					holder.expectAskedFor(b)
					if tt.announced {
						candidate.announce(b)
						candidate.expectNotAsked("getdata while another peer holds the block")
					}

					rr.release(node, holder, b)
					holder.expectNotAsked("getdata to the peer that released the block")
					if !tt.wantAsked {
						candidate.expectNotAsked("getdata for the released block")
						return
					}
					candidate.expectAskedFor(b)
					candidate.deliver(b)
					assert.Equal(t, *b.Hash(), node.tip())
				})
			}
		})
	}
}

// TestOwedBlockIsNotRequestedAgain pins that a peer is not asked again for a block it still owes, even after another
// peer that owed it left and took it out of the global request map: the peer would send the block twice, and the
// second copy would count as unrequested.
func TestOwedBlockIsNotRequestedAgain(t *testing.T) {
	tests := []struct {
		name    string
		trigger func(announcer *testPeer, b *btcutil.Block)
	}{
		{name: "release", trigger: func(*testPeer, *btcutil.Block) {}},
		{name: "announcement", trigger: func(p *testPeer, b *btcutil.Block) { p.announce(b) }},
		{name: "headers", trigger: func(p *testPeer, b *btcutil.Block) { p.sendHeaders(b) }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := newTestNode(t)
			announcer := node.outboundPeer()
			orphanSender := node.outboundPeer()
			parent := node.block(nil, 0)
			child := node.block(parent, 0)

			announcer.announce(parent)
			announcer.expectAskedFor(parent)
			orphanSender.announce(child)
			orphanSender.expectAskedFor(child)
			orphanSender.deliver(child)
			orphanSender.expectAskedFor(parent)

			orphanSender.leave()
			tt.trigger(announcer, parent)
			announcer.expectNotAsked("getdata for a block the peer already owes")

			announcer.deliver(parent)
			assert.Equal(t, *child.Hash(), node.tip())
		})
	}
}
