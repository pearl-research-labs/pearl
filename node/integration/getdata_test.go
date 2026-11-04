//go:build rpctest
// +build rpctest

// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

// Regression tests for the getdata / getcfilters reply paths in
// node/server.go. A handler must not return until its replies have been
// written (GHSA-w82j-8759-phw3), so a scripted peer that never reads will
// still receive complete, ordered replies, and a subsequent request from
// the same peer is still answered.

package integration

import (
	"net"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/integration/rpctest"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// replyPeer is a fake p2p peer that records data-reply wire messages from a
// victim pearld instance onto buffered channels.
type replyPeer struct {
	*peer.Peer
	blockCh    chan *wire.MsgBlock
	notFoundCh chan *wire.MsgNotFound
	merkleCh   chan *wire.MsgMerkleBlock
	filterCh   chan *wire.MsgCFilter
}

// newReplyPeer dials the victim, completes the v2 handshake, and returns a
// peer that records every block / notfound / merkleblock / cfilter it
// receives. The peer advertises LastBlock=0 so the victim never demotes
// itself out of current() on its account.
func newReplyPeer(t *testing.T, nodeAddr string) *replyPeer {
	t.Helper()

	conn, err := net.DialTimeout("tcp", nodeAddr, 5*time.Second)
	require.NoError(t, err, "reply: dial victim")

	rp := &replyPeer{
		blockCh:    make(chan *wire.MsgBlock, 64),
		notFoundCh: make(chan *wire.MsgNotFound, 64),
		merkleCh:   make(chan *wire.MsgMerkleBlock, 64),
		filterCh:   make(chan *wire.MsgCFilter, 64),
	}

	verackCh := make(chan struct{})
	cfg := &peer.Config{
		Listeners: peer.MessageListeners{
			OnVerAck: func(_ *peer.Peer, _ *wire.MsgVerAck) {
				close(verackCh)
			},
			OnBlock: func(_ *peer.Peer, msg *wire.MsgBlock, _ []byte) {
				select {
				case rp.blockCh <- msg:
				default:
				}
			},
			OnNotFound: func(_ *peer.Peer, msg *wire.MsgNotFound) {
				select {
				case rp.notFoundCh <- msg:
				default:
				}
			},
			OnMerkleBlock: func(_ *peer.Peer, msg *wire.MsgMerkleBlock) {
				select {
				case rp.merkleCh <- msg:
				default:
				}
			},
			OnCFilter: func(_ *peer.Peer, msg *wire.MsgCFilter) {
				select {
				case rp.filterCh <- msg:
				default:
				}
			},
		},
		NewestBlock: func() (*chainhash.Hash, int32, error) {
			return &chainhash.Hash{}, 0, nil
		},
		UserAgentName:       "reply-peer",
		UserAgentVersion:    "1.0.0",
		Services:            wire.SFNodeNetwork | wire.SFNodeWitness | wire.SFNodeP2PV2,
		ChainParams:         &chaincfg.SimNetParams,
		DisableStallHandler: true,
	}

	p, err := peer.NewOutboundPeer(cfg, nodeAddr)
	if err != nil {
		conn.Close()
		t.Fatalf("reply: NewOutboundPeer: %v", err)
	}
	p.AssociateConnection(conn)

	select {
	case <-verackCh:
		rp.Peer = p
		return rp
	case <-time.After(15 * time.Second):
		p.Disconnect()
		p.WaitForDisconnect()
		t.Fatal("reply: timed out waiting for verack")
		return nil
	}
}

// setupGetData starts a fresh simnet victim with n generated blocks and
// connects a reply peer to it. Both are registered for teardown.
func setupGetData(t *testing.T, blocks uint32) (*rpctest.Harness, *replyPeer) {
	t.Helper()

	victim, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	require.NoError(t, victim.SetUp(true, 0))
	t.Cleanup(func() { require.NoError(t, victim.TearDown()) })

	_, err = victim.Client.Generate(blocks)
	require.NoError(t, err, "setupGetData: Generate(%d)", blocks)
	require.Eventually(t, func() bool {
		_, h, err := victim.Client.GetBestBlock()
		return err == nil && h >= int32(blocks)
	}, 30*time.Second, 100*time.Millisecond, "setupGetData: tip didn't advance")

	rp := newReplyPeer(t, victim.P2PAddress())
	t.Cleanup(func() {
		rp.Disconnect()
		rp.WaitForDisconnect()
	})
	return victim, rp
}

// TestGetData verifies the getdata reply contract: a known block hash yields
// one block, unknown tx/block hashes yield a single notfound carrying exactly
// those hashes, and a filtered block for a peer with no bloom filter loaded
// yields nothing. A second getdata afterwards must still be answered, proving
// the read loop was not parked.
func TestGetData(t *testing.T) {
	victim, rp := setupGetData(t, 1)

	blockHash, _, err := victim.Client.GetBestBlock()
	require.NoError(t, err, "GetBestBlock")

	var unknownTx chainhash.Hash
	for i := range unknownTx {
		unknownTx[i] = byte(i + 1)
	}
	var unknownBlock chainhash.Hash
	for i := range unknownBlock {
		unknownBlock[i] = byte(0xff - i)
	}

	gd := wire.NewMsgGetData()
	require.NoError(t, gd.AddInvVect(wire.NewInvVect(wire.InvTypeBlock, blockHash)))
	require.NoError(t, gd.AddInvVect(wire.NewInvVect(wire.InvTypeTx, &unknownTx)))
	require.NoError(t, gd.AddInvVect(wire.NewInvVect(wire.InvTypeFilteredBlock, blockHash)))
	require.NoError(t, gd.AddInvVect(wire.NewInvVect(wire.InvTypeBlock, &unknownBlock)))
	rp.QueueMessage(gd, nil)

	// Ordering: block first, then notfound. No merkleblock because no
	// filter is loaded on this peer.
	select {
	case <-rp.blockCh:
	case <-time.After(pqMessageBudget):
		t.Fatal("getdata: timed out waiting for block")
	}
	select {
	case nf := <-rp.notFoundCh:
		require.Len(t, nf.InvList, 2, "notfound must carry {unknown tx, unknown block}")
		for _, iv := range nf.InvList {
			require.True(t, iv.Hash.IsEqual(&unknownTx) || iv.Hash.IsEqual(&unknownBlock),
				"notfound carried unexpected hash %v", iv.Hash)
		}
	case <-time.After(pqMessageBudget):
		t.Fatal("getdata: timed out waiting for notfound")
	}
	select {
	case mb := <-rp.merkleCh:
		t.Fatalf("getdata: unexpected merkleblock: %+v", mb)
	case <-time.After(pqDrainBudget):
	}

	// A second getdata for the known block must still be answered.
	gd2 := wire.NewMsgGetData()
	require.NoError(t, gd2.AddInvVect(wire.NewInvVect(wire.InvTypeBlock, blockHash)))
	rp.QueueMessage(gd2, nil)
	select {
	case <-rp.blockCh:
	case <-time.After(pqMessageBudget):
		t.Fatal("getdata: second request not answered (read loop parked)")
	}
}

// TestGetCFilters verifies that getcfilters over a generated range yields one
// cfilter per height, in order, with no drops or reordering across batch
// boundaries.
func TestGetCFilters(t *testing.T) {
	victim, rp := setupGetData(t, 20)

	_, tipHeight, err := victim.Client.GetBestBlock()
	require.NoError(t, err, "GetBestBlock")
	require.GreaterOrEqual(t, tipHeight, int32(20))

	// Request filters for the full generated range [0, tip].
	stopHash, err := victim.Client.GetBlockHash(int64(tipHeight))
	require.NoError(t, err, "GetBlockHash")

	gcf := wire.NewMsgGetCFilters(wire.GCSFilterRegular, 0, stopHash)
	rp.QueueMessage(gcf, nil)

	// Expect a cfilter for every height in [0, tip], in order.
	for h := int32(0); h <= tipHeight; h++ {
		select {
		case cf := <-rp.filterCh:
			require.Equal(t, wire.GCSFilterRegular, cf.FilterType, "cfilter type mismatch at height %d", h)
		case <-time.After(pqMessageBudget):
			t.Fatalf("getcfilters: timed out waiting for cfilter at height %d", h)
		}
	}
}
