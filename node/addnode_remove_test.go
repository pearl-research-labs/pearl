// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/connmgr"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/stretchr/testify/require"
)

// addNodeTestRig wires a real connection manager to a local TCP listener
// that accepts and holds every dial, counting accepts so a test can tell a
// redial (a second accept) from a connection that stays down.
type addNodeTestRig struct {
	cmgr    *connmgr.ConnManager
	connReq *connmgr.ConnReq
	addr    string
	accepts atomic.Int32

	mu    sync.Mutex
	conns []net.Conn
}

func newAddNodeTestRig(t *testing.T) *addNodeTestRig {
	t.Helper()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	t.Cleanup(func() { _ = listener.Close() })

	rig := &addNodeTestRig{addr: listener.Addr().String()}
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			rig.accepts.Add(1)
			rig.mu.Lock()
			rig.conns = append(rig.conns, conn)
			rig.mu.Unlock()
		}
	}()
	t.Cleanup(func() {
		rig.mu.Lock()
		defer rig.mu.Unlock()
		for _, conn := range rig.conns {
			_ = conn.Close()
		}
	})

	cmgr, err := connmgr.New(&connmgr.Config{
		Dial: func(addr net.Addr) (net.Conn, error) {
			var dialer net.Dialer
			return dialer.Dial("tcp", addr.String())
		},
		RetryDuration:   25 * time.Millisecond,
		TargetOutbound:  0,
		OnConnection:    func(*connmgr.ConnReq, net.Conn) {},
		OnDisconnection: func(*connmgr.ConnReq) {},
	})
	require.NoError(t, err)
	rig.cmgr = cmgr
	cmgr.Start()
	t.Cleanup(cmgr.Stop)

	rig.connReq = &connmgr.ConnReq{Addr: listener.Addr(), Permanent: true}
	cmgr.Connect(rig.connReq)
	require.Eventually(t, func() bool {
		return rig.connReq.State() == connmgr.ConnEstablished &&
			rig.accepts.Load() == 1
	}, 5*time.Second, 5*time.Millisecond, "permanent connection never established")

	return rig
}

// serverPeer returns a persistent serverPeer backed by rig's established
// connection request, registered in a fresh peerState the way the peer
// handler would hold it.
func (rig *addNodeTestRig) serverPeer(t *testing.T) (*server, *serverPeer, *peerState) {
	t.Helper()

	s := &server{connManager: rig.cmgr}
	sp := newServerPeer(s, true)
	p, err := peer.NewOutboundPeer(
		&peer.Config{ChainParams: &chaincfg.SimNetParams}, rig.addr,
	)
	require.NoError(t, err)
	sp.Peer = p
	sp.connReq = rig.connReq

	state := &peerState{
		inboundPeers:    make(map[int32]*serverPeer),
		outboundPeers:   make(map[int32]*serverPeer),
		persistentPeers: make(map[int32]*serverPeer),
		banned:          make(map[string]time.Time),
		outboundGroups:  make(map[string]int),
	}
	state.persistentPeers[sp.ID()] = sp

	return s, sp, state
}

// TestAddNodeRemoveStopsReconnect is the regression test for issue #364:
// a peer added with `addnode <addr> add` and then removed with
// `addnode <addr> remove` must stay removed. Before the fix the removal
// only disconnected the peer; handleDonePeerMsg then handed the still
// permanent connection request to connManager.Disconnect, which redialed
// it after the retry delay and the peer reappeared in getaddednodeinfo.
func TestAddNodeRemoveStopsReconnect(t *testing.T) {
	rig := newAddNodeTestRig(t)
	s, sp, state := rig.serverPeer(t)

	reply := make(chan error, 1)
	s.handleQuery(state, removeNodeMsg{
		cmp:   func(other *serverPeer) bool { return other.Addr() == rig.addr },
		reply: reply,
	})
	require.NoError(t, <-reply)
	require.NotContains(t, state.persistentPeers, sp.ID())

	// The peer reporting done is what used to trigger the redial.
	s.handleDonePeerMsg(state, sp)

	// With a 25ms retry duration a redial would land almost at once;
	// give it ample time, then the connection must still be down and
	// the request fully removed.
	time.Sleep(500 * time.Millisecond)
	require.Equal(t, int32(1), rig.accepts.Load(),
		"removed persistent peer was redialed")
	require.Equal(t, connmgr.ConnDisconnected, rig.connReq.State(),
		"removed persistent peer's connection request was not dropped")
}

// TestPersistentPeerReconnectsWhenNotRemoved is the control case: a
// permanent peer that drops for any other reason must keep the existing
// retry behavior and be redialed by the connection manager.
func TestPersistentPeerReconnectsWhenNotRemoved(t *testing.T) {
	rig := newAddNodeTestRig(t)
	s, sp, state := rig.serverPeer(t)

	s.handleDonePeerMsg(state, sp)

	require.Eventually(t, func() bool {
		return rig.accepts.Load() == 2 &&
			rig.connReq.State() == connmgr.ConnEstablished
	}, 5*time.Second, 5*time.Millisecond,
		"persistent peer that was not removed was not redialed")
}
