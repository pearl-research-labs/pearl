// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"net"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/internal/inbound"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMain(m *testing.M) {
	// logWriter forwards every subsystem log line to logRotator; admission warnings emitted by these tests would
	// otherwise hit a nil rotator.
	logDir, err := os.MkdirTemp("", "pearld-server-test")
	if err != nil {
		panic(err)
	}
	initLogRotator(filepath.Join(logDir, "pearld.log"), 10240)

	code := m.Run()
	_ = logRotator.Close()
	_ = os.RemoveAll(logDir)
	os.Exit(code)
}

// newTestServerPeer creates a serverPeer that exercises the lifecycle logic without starting the full server.
func newTestServerPeer(t *testing.T) (*server, *serverPeer) {
	t.Helper()

	s := &server{peerLifecycle: make(chan peerLifecycleEvent, 10)}
	sp := newServerPeer(s, false)
	sp.Peer = peer.NewInboundPeer(&peer.Config{ChainParams: &chaincfg.SimNetParams})

	return s, sp
}

func recvLifecycleEvent(t *testing.T, ch <-chan peerLifecycleEvent) peerLifecycleEvent {
	t.Helper()

	select {
	case ev := <-ch:
		return ev
	case <-time.After(5 * time.Second):
		require.Fail(t, "timed out waiting for peerLifecycleEvent")
		return peerLifecycleEvent{}
	}
}

func requireClosed(t *testing.T, ch <-chan struct{}, msg string) {
	t.Helper()

	select {
	case <-ch:
	default:
		require.Fail(t, msg)
	}
}

func TestOnVerAckDoubleCall(t *testing.T) {
	t.Parallel()

	_, sp := newTestServerPeer(t)
	var releases atomic.Uint32
	sp.releaseInboundHandshake = func() { releases.Add(1) }

	sp.OnVerAck(nil, nil)
	requireClosed(t, sp.verAckCh, "verAckCh must close on first OnVerAck")

	require.NotPanics(t, func() { sp.OnVerAck(nil, nil) })
	requireClosed(t, sp.verAckCh, "verAckCh must stay closed")
	assert.Equal(t, uint32(1), releases.Load(), "the source-prefix slot must be released exactly once")
}

func TestHandshakeReleaseOnDisconnect(t *testing.T) {
	t.Parallel()

	s, sp := newTestServerPeer(t)
	var releases atomic.Uint32
	sp.releaseInboundHandshake = func() { releases.Add(1) }

	sp.Disconnect()
	go s.peerLifecycleHandler(sp)

	event := recvLifecycleEvent(t, s.peerLifecycle)
	assert.Equal(t, peerDone, event.action)
	assert.Equal(t, uint32(1), releases.Load())
}

// TestInboundPeerReservation pins the composition of the three budget functions: listener capacity derives from
// the peer mode while connmgr keeps its automatic target.
func TestInboundPeerReservation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		maxPeers       int
		permanentPeers int
		automatic      bool
		wantTarget     int
		wantReserved   int
		wantInbound    uint32
	}{
		{name: "connect only", maxPeers: 8, permanentPeers: 1, wantReserved: 1, wantInbound: 7},
		{name: "connect only capped", maxPeers: 8, permanentPeers: 10, wantReserved: 8, wantInbound: 0},
		{name: "simnet without peers", maxPeers: 8, wantReserved: 0, wantInbound: 8},
		{name: "simnet with peers", maxPeers: 8, permanentPeers: 3, wantReserved: 3, wantInbound: 5},
		{
			name: "automatic without add peers", maxPeers: 125, automatic: true,
			wantTarget: 8, wantReserved: 8, wantInbound: 117,
		},
		{
			name: "add peers below target", maxPeers: 125, permanentPeers: 3, automatic: true,
			wantTarget: 8, wantReserved: 11, wantInbound: 114,
		},
		{
			name: "add peers above target", maxPeers: 10, permanentPeers: 9, automatic: true,
			wantTarget: 1, wantReserved: 10, wantInbound: 0,
		},
		{
			name: "add peers at max peers", maxPeers: 10, permanentPeers: 10, automatic: true,
			wantReserved: 10, wantInbound: 0,
		},
		{
			name: "add peers above max peers", maxPeers: 10, permanentPeers: 12, automatic: true,
			wantReserved: 10, wantInbound: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			targetOutbound := targetOutboundPeers(tt.maxPeers, tt.permanentPeers, tt.automatic)
			assert.Equal(t, tt.wantTarget, targetOutbound)

			reserved := reservedOutboundPeers(tt.maxPeers, targetOutbound, tt.permanentPeers, tt.automatic)
			assert.Equal(t, tt.wantReserved, reserved)
			assert.Equal(t, tt.wantInbound, maxInboundPeers(tt.maxPeers, reserved))
		})
	}
}

// TestInboundPeerAdmissionSourceLimits verifies that loopback and whitelisted peers keep the ordinary
// pending-handshake and v2 source limits; whitelisting only affects banning.
func TestInboundPeerAdmissionSourceLimits(t *testing.T) {
	_, whitelist, err := net.ParseCIDR("192.0.2.0/24")
	require.NoError(t, err)

	originalCfg := cfg
	t.Cleanup(func() { cfg = originalCfg })

	tests := []struct {
		name            string
		addr            net.Addr
		whitelists      []*net.IPNet
		wantWhitelisted bool
	}{
		{
			name: "loopback",
			addr: &net.TCPAddr{IP: net.ParseIP("127.0.0.2"), Port: 8333},
		},
		{
			name:            "whitelisted",
			addr:            &net.TCPAddr{IP: net.ParseIP("192.0.2.1"), Port: 8333},
			whitelists:      []*net.IPNet{whitelist},
			wantWhitelisted: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg = &config{whitelists: tt.whitelists}
			s := &server{inboundAdmission: inbound.New()}

			var releases []func()
			for i := 0; i < 20; i++ {
				whitelisted, release, _, err := s.acquireInboundPeerAdmission(tt.addr)
				if err != nil {
					break
				}

				assert.Equal(t, tt.wantWhitelisted, whitelisted)
				releases = append(releases, release)
			}
			require.Less(t, len(releases), 20, "the source pending limit must reject a peer")
			for _, release := range releases {
				release()
			}

			v2Rejections := 0
			for i := 0; i < 20; i++ {
				whitelisted, releaseSource, v2Admission, err := s.acquireInboundPeerAdmission(tt.addr)
				require.NoError(t, err)
				assert.Equal(t, tt.wantWhitelisted, whitelisted)
				releaseSource()

				releaseV2, err := v2Admission.Acquire()
				if err != nil {
					v2Rejections++
					break
				}
				releaseV2()

				releaseV2, err = v2Admission.Acquire()
				require.NoError(t, err, "the second CPU phase must not consume another rate token")
				releaseV2()
			}
			assert.Equal(t, 1, v2Rejections, "the v2 source rate must reject a peer")
		})
	}
}

// TestPeerLifecycleOrdering verifies that verack before disconnect yields peerAdd followed by peerDone, never the
// reverse.
func TestPeerLifecycleOrdering(t *testing.T) {
	t.Parallel()

	s, sp := newTestServerPeer(t)
	close(sp.verAckCh)

	go s.peerLifecycleHandler(sp)

	first := recvLifecycleEvent(t, s.peerLifecycle)
	require.Equal(t, peerAdd, first.action)
	assert.Same(t, sp, first.sp)

	sp.Disconnect()

	second := recvLifecycleEvent(t, s.peerLifecycle)
	require.Equal(t, peerDone, second.action)
	assert.Same(t, sp, second.sp)
}

// TestPeerLifecycleSimultaneousReady covers both channels being ready before the handler runs. select is
// nondeterministic, so peerAdd may be skipped; peerDone must always arrive, and after peerAdd if that was emitted.
func TestPeerLifecycleSimultaneousReady(t *testing.T) {
	t.Parallel()

	const iterations = 100
	addEmitted := 0

	for i := 0; i < iterations; i++ {
		s, sp := newTestServerPeer(t)

		close(sp.verAckCh)
		sp.Disconnect()

		go s.peerLifecycleHandler(sp)

		first := recvLifecycleEvent(t, s.peerLifecycle)
		if first.action == peerAdd {
			addEmitted++
			second := recvLifecycleEvent(t, s.peerLifecycle)
			assert.Equal(t, peerDone, second.action, "iteration %d: peerAdd must be followed by peerDone", i)
			continue
		}

		assert.Equal(t, peerDone, first.action, "iteration %d: sole event must be peerDone", i)
	}

	t.Logf("peerAdd emitted in %d/%d iterations", addEmitted, iterations)
}
