//go:build rpctest
// +build rpctest

// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package integration

import (
	"bytes"
	"net"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/integration/rpctest"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// rawPeerTimeout bounds every wait on a raw peer: the handshake and each
// expected reply.
const rawPeerTimeout = 15 * time.Second

// connectRawPeer dials nodeAddr with cfg and blocks until the version
// handshake completes. The peer is disconnected on test cleanup.
func connectRawPeer(t *testing.T, nodeAddr string, cfg peer.Config) *peer.Peer {
	t.Helper()

	verack := make(chan struct{}, 1)
	cfg.Listeners.OnVerAck = func(*peer.Peer, *wire.MsgVerAck) {
		select {
		case verack <- struct{}{}:
		default:
		}
	}
	cfg.ChainParams = &chaincfg.SimNetParams
	cfg.UserAgentName = "raw-peer"
	cfg.UserAgentVersion = "1.0.0"
	cfg.DisableStallHandler = true

	conn, err := net.DialTimeout("tcp", nodeAddr, rawPeerTimeout)
	require.NoError(t, err)
	p, err := peer.NewOutboundPeer(&cfg, nodeAddr)
	if err != nil {
		conn.Close()
		require.NoError(t, err)
	}
	p.AssociateConnection(conn)
	t.Cleanup(func() {
		p.Disconnect()
		p.WaitForDisconnect()
	})

	select {
	case <-verack:
	case <-time.After(rawPeerTimeout):
		t.Fatal("timed out waiting for verack")
	}
	return p
}

// startHarness starts a simnet node with the given extra arguments and tears
// it down on test cleanup.
func startHarness(t *testing.T, extraArgs []string) *rpctest.Harness {
	t.Helper()

	h, err := rpctest.New(&chaincfg.SimNetParams, nil, extraArgs, "")
	require.NoError(t, err)
	require.NoError(t, h.SetUp(false, 0))
	t.Cleanup(func() { require.NoError(t, h.TearDown()) })
	return h
}

// requireRegistered fails the test unless the node lists p as a connected
// peer.
func requireRegistered(t *testing.T, h *rpctest.Harness, p *peer.Peer) {
	t.Helper()

	local := p.LocalAddr().String()
	require.Eventually(t, func() bool {
		peers, err := h.Client.GetPeerInfo()
		if err != nil {
			return false
		}
		for _, info := range peers {
			if info.Addr == local {
				return true
			}
		}
		return false
	}, rawPeerTimeout, 50*time.Millisecond, "node did not register the peer")
	require.True(t, p.Connected())
}

// receive returns the next value on ch or fails the test after rawPeerTimeout.
func receive[T any](t *testing.T, ch <-chan T) T {
	t.Helper()

	select {
	case v := <-ch:
		return v
	case <-time.After(rawPeerTimeout):
		t.Fatal("timed out waiting for reply")
	}
	var zero T
	return zero
}

// requestInv sends a getdata for a single inventory vector.
func requestInv(t *testing.T, p *peer.Peer, invType wire.InvType, hash *chainhash.Hash) {
	t.Helper()

	getData := wire.NewMsgGetData()
	require.NoError(t, getData.AddInvVect(wire.NewInvVect(invType, hash)))
	p.QueueMessage(getData, nil)
}

// TestPeersWithoutWitnessServiceConnect verifies that the node accepts and
// registers peers configured like the DNS seeders and the wallet's pruned-block
// dispatcher, neither of which advertises SFNodeWitness.
func TestPeersWithoutWitnessServiceConnect(t *testing.T) {
	h := startHarness(t, nil)

	tests := []struct {
		name string
		cfg  peer.Config
	}{
		{name: "dns seeder", cfg: peer.Config{Services: wire.SFNodeP2PV2}},
		{name: "pruned block dispatcher", cfg: peer.Config{DisableRelayTx: true}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := connectRawPeer(t, h.P2PAddress(), tt.cfg)
			require.True(t, p.Services().HasFlag(wire.SFNodeWitness),
				"node must keep advertising witness support")
			requireRegistered(t, h, p)
		})
	}
}

// TestGetDataServesWitnessForBaseInvTypes verifies that getdata for MSG_TX and
// MSG_BLOCK returns the witness serialization, byte-identical to the
// MSG_WITNESS_TX and MSG_WITNESS_BLOCK replies, to a peer that advertises no
// services.
func TestGetDataServesWitnessForBaseInvTypes(t *testing.T) {
	h, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	require.NoError(t, h.SetUp(true, 1))
	t.Cleanup(func() { require.NoError(t, h.TearDown()) })

	addr, err := h.NewAddress()
	require.NoError(t, err)
	pkScript, err := txscript.PayToAddrScript(addr)
	require.NoError(t, err)
	tx, err := h.CreateTransaction([]*wire.TxOut{wire.NewTxOut(1e6, pkScript)}, 100, true)
	require.NoError(t, err)
	require.True(t, tx.HasWitness(), "harness wallet must sign with witness data")
	_, err = h.Client.SendRawTransaction(tx, true)
	require.NoError(t, err)

	txs := make(chan *wire.MsgTx, 1)
	blocks := make(chan *wire.MsgBlock, 1)
	p := connectRawPeer(t, h.P2PAddress(), peer.Config{
		Listeners: peer.MessageListeners{
			OnTx:    func(_ *peer.Peer, msg *wire.MsgTx) { txs <- msg },
			OnBlock: func(_ *peer.Peer, msg *wire.MsgBlock, _ []byte) { blocks <- msg },
		},
		DisableRelayTx: true,
	})

	txHash := tx.TxHash()
	for _, invType := range []wire.InvType{wire.InvTypeTx, wire.InvTypeWitnessTx} {
		requestInv(t, p, invType, &txHash)
		got := receive(t, txs)
		require.True(t, got.HasWitness(), "%v reply lost its witness", invType)
		require.Equal(t, tx.WitnessHash(), got.WitnessHash(), invType)
	}

	blockHashes, err := h.Client.Generate(1)
	require.NoError(t, err)
	want, err := h.Client.GetBlock(blockHashes[0])
	require.NoError(t, err)
	minedTxHashes, err := want.TxHashes()
	require.NoError(t, err)
	require.Contains(t, minedTxHashes, txHash, "block must carry the witness tx")
	var wantBytes bytes.Buffer
	require.NoError(t, want.Serialize(&wantBytes))

	for _, invType := range []wire.InvType{wire.InvTypeBlock, wire.InvTypeWitnessBlock} {
		requestInv(t, p, invType, blockHashes[0])
		got := receive(t, blocks)
		var gotBytes bytes.Buffer
		require.NoError(t, got.Serialize(&gotBytes))
		require.Equal(t, wantBytes.Bytes(), gotBytes.Bytes(), invType)
	}
}
