//go:build rpctest
// +build rpctest

// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

// Regression test for the per-peer trust gate in netsync.handleInvMsg
// (node/netsync/manager.go). A low-quality peer's block inv must be
// answered with `getheaders` first; once the peer has supplied a
// tip-extending block (peerQualityCounter resets to 0), subsequent
// block invs must be answered with `getdata` directly.

package integration

import (
	"net"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/integration/rpctest"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

const (
	// pqMessageBudget bounds the wait for a "must arrive" wire message
	// from the victim. Generous enough to absorb single-message round
	// trips on a loaded CI machine.
	pqMessageBudget = 5 * time.Second

	// pqDrainBudget is how long we wait while asserting that a wire
	// message must NOT arrive. Short enough to keep the test snappy.
	pqDrainBudget = 1 * time.Second
)

// scriptedPeer is a fake p2p peer used to observe wire-level replies
// from a victim pearld instance. Its OnGetHeaders / OnGetData callbacks
// push received messages onto buffered channels so the test driver can
// assert what the victim sent.
type scriptedPeer struct {
	*peer.Peer
	getHeadersCh chan *wire.MsgGetHeaders
	getDataCh    chan *wire.MsgGetData
}

// drain empties any pending messages from the observation channels.
func (sp *scriptedPeer) drain() {
	for {
		select {
		case <-sp.getHeadersCh:
		case <-sp.getDataCh:
		default:
			return
		}
	}
}

// expectCertlessGetHeaders waits for a getheaders from the victim and
// asserts it requested cert-less headers.
func (sp *scriptedPeer) expectCertlessGetHeaders(t *testing.T) {
	t.Helper()
	select {
	case msg := <-sp.getHeadersCh:
		require.False(t, msg.IncludeCertificates,
			"low-quality probe must ask for cert-less headers")
	case <-time.After(pqMessageBudget):
		t.Fatal("scripted: timed out waiting for getheaders")
	}
}

// expectGetDataFor blocks for up to pqMessageBudget for a getdata
// message from the victim that includes hash. The loop tolerates
// interleaved relay/trickle getdatas that don't carry our target.
func (sp *scriptedPeer) expectGetDataFor(t *testing.T, hash *chainhash.Hash) {
	t.Helper()
	deadline := time.After(pqMessageBudget)
	for {
		select {
		case msg := <-sp.getDataCh:
			for _, iv := range msg.InvList {
				if iv.Hash.IsEqual(hash) {
					return
				}
			}
		case <-deadline:
			t.Fatalf("scripted: timed out waiting for getdata "+
				"for %s", hash)
			return
		}
	}
}

// assertQuiet waits pqDrainBudget and fails if any getheaders or
// getdata message arrives in that window.
func (sp *scriptedPeer) assertQuiet(t *testing.T) {
	t.Helper()
	select {
	case msg := <-sp.getHeadersCh:
		t.Fatalf("scripted: unexpected getheaders: %+v", msg)
	case msg := <-sp.getDataCh:
		t.Fatalf("scripted: unexpected getdata: %+v", msg)
	case <-time.After(pqDrainBudget):
	}
}

// newScriptedPeer dials the victim, completes the v2 handshake, and
// returns a peer that records every getheaders / getdata it receives.
// The peer is inbound from the victim's POV so pickSyncCandidate
// excludes it; it advertises LastBlock=0 so the victim never demotes
// itself out of `current()` on its account.
func newScriptedPeer(t *testing.T, nodeAddr string) *scriptedPeer {
	t.Helper()

	conn, err := net.DialTimeout("tcp", nodeAddr, 5*time.Second)
	require.NoError(t, err, "scripted: dial victim")

	sp := &scriptedPeer{
		getHeadersCh: make(chan *wire.MsgGetHeaders, 16),
		getDataCh:    make(chan *wire.MsgGetData, 16),
	}

	verackCh := make(chan struct{})
	cfg := &peer.Config{
		Listeners: peer.MessageListeners{
			OnVerAck: func(_ *peer.Peer, _ *wire.MsgVerAck) {
				close(verackCh)
			},
			OnGetHeaders: func(_ *peer.Peer, msg *wire.MsgGetHeaders) {
				select {
				case sp.getHeadersCh <- msg:
				default:
				}
			},
			OnGetData: func(_ *peer.Peer, msg *wire.MsgGetData) {
				select {
				case sp.getDataCh <- msg:
				default:
				}
			},
		},
		// LastBlock=0 in our outgoing version handshake so the victim
		// never demotes itself out of current() on our account.
		NewestBlock: func() (*chainhash.Hash, int32, error) {
			return &chainhash.Hash{}, 0, nil
		},
		UserAgentName:       "scripted-peer",
		UserAgentVersion:    "1.0.0",
		Services:            wire.SFNodeNetwork | wire.SFNodeWitness | wire.SFNodeP2PV2,
		ChainParams:         &chaincfg.SimNetParams,
		DisableStallHandler: true,
	}

	p, err := peer.NewOutboundPeer(cfg, nodeAddr)
	if err != nil {
		conn.Close()
		t.Fatalf("scripted: NewOutboundPeer: %v", err)
	}
	p.AssociateConnection(conn)

	select {
	case <-verackCh:
		sp.Peer = p
		return sp
	case <-time.After(15 * time.Second):
		p.Disconnect()
		p.WaitForDisconnect()
		t.Fatal("scripted: timed out waiting for verack")
		return nil
	}
}

// setupPeerQuality starts a fresh simnet victim with a one-block tip
// (so chain.IsCurrent reports true) and connects a scripted peer to
// it. Both are registered for teardown via t.Cleanup.
func setupPeerQuality(t *testing.T) (*rpctest.Harness, *scriptedPeer) {
	t.Helper()

	victim, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	require.NoError(t, victim.SetUp(true, 0))
	t.Cleanup(func() { require.NoError(t, victim.TearDown()) })

	_, err = victim.Client.Generate(1)
	require.NoError(t, err, "setupPeerQuality: Generate(1)")
	require.Eventually(t, func() bool {
		_, h, err := victim.Client.GetBestBlock()
		return err == nil && h >= 1
	}, 30*time.Second, 100*time.Millisecond,
		"setupPeerQuality: tip didn't advance to >=1")

	sp := newScriptedPeer(t, victim.P2PAddress())
	t.Cleanup(func() {
		sp.Disconnect()
		sp.WaitForDisconnect()
	})
	sp.drain()
	return victim, sp
}

// blockOnVictimTip constructs a simnet child block building on the
// victim's current tip. The zero time.Time tells CreateBlock to use
// prevBlockTime + 1s, satisfying the strictly-increasing timestamp
// rule regardless of test pacing.
func blockOnVictimTip(t *testing.T, victim *rpctest.Harness) *btcutil.Block {
	t.Helper()

	prevHash, prevHeight, err := victim.Client.GetBestBlock()
	require.NoError(t, err)
	prevMsg, err := victim.Client.GetBlock(prevHash)
	require.NoError(t, err)
	prevBlock := btcutil.NewBlock(prevMsg)
	prevBlock.SetHeight(prevHeight)

	addr, err := victim.NewAddress()
	require.NoError(t, err)

	blk, err := rpctest.CreateBlock(prevBlock, nil, rpctest.BlockVersion,
		time.Time{}, addr, nil, &chaincfg.SimNetParams)
	require.NoError(t, err)
	return blk
}

// TestPeerQualityInvGating exercises netsync.handleInvMsg's per-peer
// gate: a low-quality peer's block inv triggers a cert-less getheaders
// first; once the peer extends our tip, subsequent invs go straight to
// getdata.
func TestPeerQualityInvGating(t *testing.T) {
	t.Run("low_quality_peer_inv_triggers_getheaders", func(t *testing.T) {
		_, sp := setupPeerQuality(t)

		bogus := chainhash.Hash{0xab, 0xcd, 0xef}
		inv := wire.NewMsgInv()
		require.NoError(t, inv.AddInvVect(
			wire.NewInvVect(wire.InvTypeBlock, &bogus)))
		sp.QueueMessage(inv, nil)

		sp.expectCertlessGetHeaders(t)
		sp.assertQuiet(t)
	})

	t.Run("high_quality_peer_inv_triggers_getdata", func(t *testing.T) {
		victim, sp := setupPeerQuality(t)

		blockA := blockOnVictimTip(t, victim)
		blockAHash := blockA.MsgBlock().BlockHeader().BlockHash()

		// Phase A: deliver blockA via inv -> getheaders -> getdata to
		// flip the peer quality counter to 0 (high-quality).
		invA := wire.NewMsgInv()
		require.NoError(t, invA.AddInvVect(
			wire.NewInvVect(wire.InvTypeBlock, &blockAHash)))
		sp.QueueMessage(invA, nil)

		sp.expectCertlessGetHeaders(t)

		hdrs := wire.NewMsgHeaders()
		require.NoError(t, hdrs.AddBlockHeader(
			*blockA.MsgBlock().BlockHeader(), nil))
		sp.QueueMessage(hdrs, nil)

		sp.expectGetDataFor(t, &blockAHash)
		sp.QueueMessage(blockA.MsgBlock(), nil)

		// Synchronize on the tip advance so handleBlockMsg has run
		// and reset peerQualityCounter to 0 before Phase B.
		require.Eventually(t, func() bool {
			_, h, err := victim.Client.GetBestBlock()
			return err == nil && h >= 2
		}, 30*time.Second, 100*time.Millisecond,
			"victim failed to accept blockA from scripted peer")

		// Drain any relay/trickle messages queued by blockA's accept
		// before asserting Phase B's quietness.
		sp.drain()

		blockB := blockOnVictimTip(t, victim)
		blockBHash := blockB.MsgBlock().BlockHeader().BlockHash()

		// Phase B: as a high-quality peer, the blockB inv must bypass
		// the getheaders gate and yield a direct getdata, with no
		// follow-up probe.
		invB := wire.NewMsgInv()
		require.NoError(t, invB.AddInvVect(
			wire.NewInvVect(wire.InvTypeBlock, &blockBHash)))
		sp.QueueMessage(invB, nil)

		sp.expectGetDataFor(t, &blockBHash)
		sp.assertQuiet(t)
	})
}
