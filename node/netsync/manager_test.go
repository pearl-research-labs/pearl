// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package netsync

import (
	"bytes"
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
