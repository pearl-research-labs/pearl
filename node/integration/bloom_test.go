//go:build rpctest
// +build rpctest

// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package integration

import (
	"net"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// bloomFilterMsgs returns one of each BIP0037 filter message.
func bloomFilterMsgs() []wire.Message {
	return []wire.Message{
		wire.NewMsgFilterLoad([]byte{0x01}, 1, 0, wire.BloomUpdateNone),
		wire.NewMsgFilterAdd([]byte{0x01}),
		wire.NewMsgFilterClear(),
	}
}

// requireDisconnected fails the test unless p disconnects within
// rawPeerTimeout.
func requireDisconnected(t *testing.T, p *peer.Peer) {
	t.Helper()

	done := make(chan struct{})
	go func() {
		p.WaitForDisconnect()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(rawPeerTimeout):
		t.Fatal("node did not disconnect the peer")
	}
}

// requireBanned dials the node and fails the test unless it drops the
// connection, which it does for a banned address once the handshake ends.
func requireBanned(t *testing.T, nodeAddr string) {
	t.Helper()

	conn, err := net.DialTimeout("tcp", nodeAddr, rawPeerTimeout)
	require.NoError(t, err)
	p, err := peer.NewOutboundPeer(&peer.Config{
		ChainParams:         &chaincfg.SimNetParams,
		UserAgentName:       "raw-peer",
		UserAgentVersion:    "1.0.0",
		DisableStallHandler: true,
	}, nodeAddr)
	if err != nil {
		conn.Close()
		require.NoError(t, err)
	}
	p.AssociateConnection(conn)
	t.Cleanup(p.Disconnect)

	requireDisconnected(t, p)
}

// TestBloomFilteringRetired verifies that the node does not advertise bloom
// filtering and answers filtered-block requests with notfound.
func TestBloomFilteringRetired(t *testing.T) {
	h := startHarness(t, nil)

	notFound := make(chan *wire.MsgNotFound, 1)
	merkleBlocks := make(chan *wire.MsgMerkleBlock, 1)
	p := connectRawPeer(t, h.P2PAddress(), peer.Config{
		Listeners: peer.MessageListeners{
			OnNotFound:    func(_ *peer.Peer, msg *wire.MsgNotFound) { notFound <- msg },
			OnMerkleBlock: func(_ *peer.Peer, msg *wire.MsgMerkleBlock) { merkleBlocks <- msg },
		},
	})

	services := p.Services()
	require.False(t, services.HasFlag(wire.SFNodeBloom), "advertised %v", services)
	require.True(t, services.HasFlag(wire.SFNodeNetwork|wire.SFNodeWitness|wire.SFNodeCF|wire.SFNodeP2PV2),
		"advertised %v", services)

	info, err := h.Client.GetNetworkInfo()
	require.NoError(t, err)
	require.NotContains(t, info.LocalServicesNames, "BLOOM")

	bestHash, _, err := h.Client.GetBestBlock()
	require.NoError(t, err)
	for _, invType := range []wire.InvType{wire.InvTypeFilteredBlock, wire.InvTypeFilteredWitnessBlock} {
		requestInv(t, p, invType, bestHash)
		got := receive(t, notFound)
		require.Len(t, got.InvList, 1)
		require.Equal(t, invType, got.InvList[0].Type)
	}
	select {
	case msg := <-merkleBlocks:
		t.Fatalf("unexpected merkleblock for %v", msg.Header.BlockHash())
	default:
	}
}

// TestNoPeerBloomFiltersOptionAccepted verifies that a node configured with the
// deprecated nopeerbloomfilters option still starts.
func TestNoPeerBloomFiltersOptionAccepted(t *testing.T) {
	h := startHarness(t, []string{"--nopeerbloomfilters"})

	_, _, err := h.Client.GetBestBlock()
	require.NoError(t, err)
}

// TestBloomFilterMessagesBanPeer verifies that a peer sending any BIP0037
// filter message is disconnected and its address banned.
func TestBloomFilterMessagesBanPeer(t *testing.T) {
	for _, msg := range bloomFilterMsgs() {
		t.Run(msg.Command(), func(t *testing.T) {
			h := startHarness(t, nil)

			p := connectRawPeer(t, h.P2PAddress(), peer.Config{})
			p.QueueMessage(msg, nil)
			requireDisconnected(t, p)

			requireBanned(t, h.P2PAddress())
		})
	}
}

// TestBloomFilterMessagesWithoutBanning verifies that with banning disabled a
// peer sending a BIP0037 filter message is disconnected but may reconnect.
func TestBloomFilterMessagesWithoutBanning(t *testing.T) {
	h := startHarness(t, []string{"--nobanning"})

	for _, msg := range bloomFilterMsgs() {
		p := connectRawPeer(t, h.P2PAddress(), peer.Config{})
		p.QueueMessage(msg, nil)
		requireDisconnected(t, p)
	}

	requireRegistered(t, h, connectRawPeer(t, h.P2PAddress(), peer.Config{}))
}
