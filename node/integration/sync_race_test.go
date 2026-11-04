//go:build rpctest
// +build rpctest

package integration

import (
	"errors"
	"fmt"
	"math/rand"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/integration/rpctest"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/rpcclient"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

const (
	syncRaceIterations = 1000

	// Every fake peer originates from 127.0.0.1, so one wave cannot exceed the node's per-source pending-handshake
	// budget, and the per-source v2 handshake rate (2/s, burst 4) bounds how fast a wave can be admitted. Rejected
	// attempts are retried until the wave deadline.
	syncRaceConcurrency          = 8
	syncRaceHandshakeConcurrency = 4
	syncRaceHandshakeDeadline    = 30 * time.Second
	syncRaceRetryDelay           = 250 * time.Millisecond
	syncRaceRegistrationWait     = 15 * time.Second
	syncRaceRunDuration          = 90 * time.Second
	syncRaceProofBlocks          = 5
	syncRaceProofWait            = 8 * time.Second
)

var errRejectedBeforeVerack = errors.New("disconnected before verack")

func fakePeerConfig(listeners peer.MessageListeners) *peer.Config {
	return &peer.Config{
		Listeners:        listeners,
		UserAgentName:    "fake-peer",
		UserAgentVersion: "1.0.0",
		Services:         wire.SFNodeNetwork | wire.SFNodeWitness | wire.SFNodeP2PV2,
		ChainParams:      &chaincfg.SimNetParams,
	}
}

// registerFakePeer completes the v2 transport and version/verack handshake with the node and then proves, with a
// ping/pong round trip, that the node processed our verack. Admission rejections surface as
// errRejectedBeforeVerack so the caller can retry.
func registerFakePeer(nodeAddr string) (*peer.Peer, error) {
	conn, err := net.DialTimeout("tcp", nodeAddr, 5*time.Second)
	if err != nil {
		return nil, err
	}

	verackCh := make(chan struct{})
	pongCh := make(chan struct{})
	pingNonce := uint64(rand.Int63())
	var pongOnce sync.Once
	p, err := peer.NewOutboundPeer(fakePeerConfig(peer.MessageListeners{
		OnVerAck: func(*peer.Peer, *wire.MsgVerAck) {
			close(verackCh)
		},
		OnPong: func(_ *peer.Peer, msg *wire.MsgPong) {
			if msg.Nonce == pingNonce {
				pongOnce.Do(func() { close(pongCh) })
			}
		},
	}), nodeAddr)
	if err != nil {
		_ = conn.Close()
		return nil, err
	}

	p.AssociateConnection(conn)

	select {
	case <-verackCh:
	case <-p.Done():
		return nil, errRejectedBeforeVerack
	case <-time.After(15 * time.Second):
		p.Disconnect()
		p.WaitForDisconnect()
		return nil, errors.New("timeout waiting for verack")
	}

	p.QueueMessage(wire.NewMsgPing(pingNonce), nil)
	select {
	case <-pongCh:
		return p, nil
	case <-p.Done():
		return nil, errors.New("disconnected before pong")
	case <-time.After(15 * time.Second):
		p.Disconnect()
		p.WaitForDisconnect()
		return nil, errors.New("timeout waiting for pong")
	}
}

// fakePeerConn registers one peer with the node, retrying admission rejections, then holds the connection open
// until the whole wave is ready to disconnect together. Only registered peers are counted toward the wave.
func fakePeerConn(
	nodeAddr string, handshakeSlots chan struct{}, ready chan<- struct{}, disconnect <-chan struct{},
) error {

	select {
	case handshakeSlots <- struct{}{}:
	case <-disconnect:
		return nil
	}
	slotHeld := true
	defer func() {
		if slotHeld {
			<-handshakeSlots
		}
	}()

	deadline := time.Now().Add(syncRaceHandshakeDeadline)
	var p *peer.Peer
	for {
		select {
		case <-disconnect:
			return nil
		default:
		}

		var err error
		p, err = registerFakePeer(nodeAddr)
		if err == nil {
			break
		}
		if !errors.Is(err, errRejectedBeforeVerack) || time.Now().After(deadline) {
			return err
		}

		select {
		case <-time.After(syncRaceRetryDelay):
		case <-disconnect:
			return nil
		}
	}

	<-handshakeSlots
	slotHeld = false

	select {
	case ready <- struct{}{}:
	case <-disconnect:
	}
	<-disconnect

	p.Disconnect()
	p.WaitForDisconnect()
	return nil
}

// waitForConnectionCount polls the node until its registered peer count matches want; registration and removal
// both run through the peerHandler goroutine, so the count is the observable barrier for NewPeer/DonePeer.
func waitForConnectionCount(client *rpcclient.Client, want int64) error {
	deadline := time.Now().Add(syncRaceRegistrationWait)
	var got int64 = -1
	for time.Now().Before(deadline) {
		count, err := client.GetConnectionCount()
		if err != nil {
			return err
		}
		got = count
		if got == want {
			return nil
		}

		time.Sleep(10 * time.Millisecond)
	}

	return fmt.Errorf("timed out waiting for %d registered peers, last count %d", want, got)
}

// runFakePeerBatch registers one full wave of peers, disconnects them together, and waits for every worker and
// for server-side peer removal.
func runFakePeerBatch(nodeAddr string, client *rpcclient.Client) error {
	handshakeSlots := make(chan struct{}, syncRaceHandshakeConcurrency)
	ready := make(chan struct{}, syncRaceConcurrency)
	disconnect := make(chan struct{})
	errCh := make(chan error, syncRaceConcurrency)

	var disconnectOnce sync.Once
	disconnectAll := func() {
		disconnectOnce.Do(func() { close(disconnect) })
	}
	defer disconnectAll()

	for i := 0; i < syncRaceConcurrency; i++ {
		go func() {
			errCh <- fakePeerConn(nodeAddr, handshakeSlots, ready, disconnect)
		}()
	}

	var firstErr error
	results := 0
	for readyPeers := 0; readyPeers < syncRaceConcurrency && firstErr == nil; {
		select {
		case <-ready:
			readyPeers++

		case err := <-errCh:
			results++
			if err == nil {
				err = errors.New("exited before the disconnect barrier")
			}
			firstErr = fmt.Errorf("fake peer: %w", err)
		}
	}

	if firstErr == nil {
		err := waitForConnectionCount(client, syncRaceConcurrency)
		if err != nil {
			firstErr = fmt.Errorf("peer registration barrier: %w", err)
		}
	}

	disconnectAll()
	for results < syncRaceConcurrency {
		err := <-errCh
		results++
		if firstErr == nil && err != nil {
			firstErr = err
		}
	}
	if firstErr != nil {
		return firstErr
	}

	if err := waitForConnectionCount(client, 0); err != nil {
		return fmt.Errorf("peer removal barrier: %w", err)
	}

	return nil
}

// TestSyncManagerRaceCorruption stresses a single simnet node with waves of inbound peers that complete the
// handshake and disconnect together. It then proves the node is not stuck with a dead sync peer: a fresh node
// generates blocks and the stressed node must sync them.
func TestSyncManagerRaceCorruption(t *testing.T) {
	stressedHarness, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	require.NoError(t, stressedHarness.SetUp(true, 0))
	t.Cleanup(func() {
		require.NoError(t, stressedHarness.TearDown())
	})

	nodeAddr := stressedHarness.P2PAddress()
	deadline := time.Now().Add(syncRaceRunDuration)
	done := 0
	for time.Now().Before(deadline) && done < syncRaceIterations {
		require.NoError(t, runFakePeerBatch(nodeAddr, stressedHarness.Client))
		done += syncRaceConcurrency
	}

	newHarness, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	require.NoError(t, newHarness.SetUp(true, 0))
	defer func() { _ = newHarness.TearDown() }()

	require.NoError(t, rpctest.ConnectNode(stressedHarness, newHarness), "stressed node must connect to the new node")

	_, heightBefore, err := stressedHarness.Client.GetBestBlock()
	require.NoError(t, err)

	_, err = newHarness.Client.Generate(syncRaceProofBlocks)
	require.NoError(t, err)

	time.Sleep(syncRaceProofWait)

	_, heightAfter, err := stressedHarness.Client.GetBestBlock()
	require.NoError(t, err)

	expected := heightBefore + int32(syncRaceProofBlocks)
	require.GreaterOrEqualf(t, heightAfter, expected,
		"sync manager corruption after %d fake peer cycles: node stuck with dead sync peer (height %d -> %d)",
		done, heightBefore, heightAfter)

	t.Logf("completed %d fake peer cycles; node synced (height %d -> %d)", done, heightBefore, heightAfter)
}

// dialPreVerackPeer exchanges version messages with the node and disconnects before the peer package writes
// verack. It returns errRejectedBeforeVerack when admission closed the connection before the exchange completed.
func dialPreVerackPeer(nodeAddr string) error {
	conn, err := net.DialTimeout("tcp", nodeAddr, 5*time.Second)
	if err != nil {
		return err
	}

	var versionSeen atomic.Bool
	p, err := peer.NewOutboundPeer(fakePeerConfig(peer.MessageListeners{
		OnVersion: func(p *peer.Peer, _ *wire.MsgVersion) *wire.MsgReject {
			// Runs on the negotiation goroutine ahead of the verack write, so closing here leaves the node with
			// a version exchange and no verack.
			versionSeen.Store(true)
			p.Disconnect()
			return nil
		},
	}), nodeAddr)
	if err != nil {
		_ = conn.Close()
		return err
	}

	p.AssociateConnection(conn)
	defer func() {
		p.Disconnect()
		p.WaitForDisconnect()
	}()

	select {
	case <-p.Done():
		if !versionSeen.Load() {
			return errRejectedBeforeVerack
		}
		return nil

	case <-time.After(5 * time.Second):
		return errors.New("timeout waiting for version exchange")
	}
}

// TestPreVerackDisconnect verifies that peers disconnecting after the version exchange but before verack
// (peerDone without peerAdd) leave the sync manager healthy.
func TestPreVerackDisconnect(t *testing.T) {
	harness, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	require.NoError(t, harness.SetUp(true, 0))
	t.Cleanup(func() { _ = harness.TearDown() })

	nodeAddr := harness.P2PAddress()

	const (
		preVerackAttempts     = 50
		preVerackRetryTimeout = 10 * time.Second
	)

	retries := 0
	for i := 0; i < preVerackAttempts; i++ {
		deadline := time.Now().Add(preVerackRetryTimeout)
		for {
			err := dialPreVerackPeer(nodeAddr)
			if err == nil {
				break
			}
			require.Truef(t, time.Now().Before(deadline), "pre-verack attempt %d did not complete: %v", i+1, err)

			retries++
			time.Sleep(syncRaceRetryDelay)
		}
	}
	t.Logf("completed %d pre-verack disconnects with %d retries", preVerackAttempts, retries)

	// Allow the node time to process all the disconnects.
	time.Sleep(2 * time.Second)

	helper, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	require.NoError(t, helper.SetUp(true, 0))
	defer func() { _ = helper.TearDown() }()

	require.NoError(t, rpctest.ConnectNode(harness, helper))

	_, heightBefore, err := harness.Client.GetBestBlock()
	require.NoError(t, err)

	_, err = helper.Client.Generate(3)
	require.NoError(t, err)

	time.Sleep(5 * time.Second)

	_, heightAfter, err := harness.Client.GetBestBlock()
	require.NoError(t, err)

	require.GreaterOrEqual(t, heightAfter, heightBefore+3, "node failed to sync after pre-verack disconnects")

	t.Logf("node healthy after %d pre-verack disconnects (height %d -> %d)", preVerackAttempts, heightBefore,
		heightAfter)
}
