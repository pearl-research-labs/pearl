//go:build rpctest
// +build rpctest

package integration

import (
	"fmt"
	"net"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/integration/rpctest"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

const (
	syncRaceIterations  = 1000
	syncRaceConcurrency = 300
	syncRaceRunDuration = 90 * time.Second
	syncRaceProofBlocks = 5
	syncRaceProofWait   = 8 * time.Second
)

// fakePeerConn connects to the node at nodeAddr, performs the full BIP324 v2
// handshake + version/verack so the node registers a peer (NewPeer) and then
// disconnects so the node runs DonePeer. This simulates attacker traffic: many
// connections that complete handshake then drop, stressing the sync manager's
// ordering of NewPeer/DonePeer.
func fakePeerConn(nodeAddr string) error {
	conn, err := net.DialTimeout("tcp", nodeAddr, 5*time.Second)
	if err != nil {
		return err
	}

	verackCh := make(chan struct{})
	p, err := peer.NewOutboundPeer(&peer.Config{
		Listeners: peer.MessageListeners{
			OnVerAck: func(p *peer.Peer, msg *wire.MsgVerAck) {
				close(verackCh)
			},
		},
		UserAgentName:    "fake-peer",
		UserAgentVersion: "1.0.0",
		Services:         wire.SFNodeNetwork | wire.SFNodeWitness | wire.SFNodeP2PV2,
		ChainParams:      &chaincfg.SimNetParams,
	}, nodeAddr)
	if err != nil {
		conn.Close()
		return err
	}

	p.AssociateConnection(conn)

	select {
	case <-verackCh:
		p.Disconnect()
		p.WaitForDisconnect()
		return nil
	case <-time.After(15 * time.Second):
		p.Disconnect()
		p.WaitForDisconnect()
		return fmt.Errorf("timeout")
	}
}

// TestSyncManagerRaceCorruption stresses a single simnet node with many inbound
// connections that complete the version/verack handshake then disconnect. It then
// proves corruption without a dedicated RPC: connect a fresh node that generates
// blocks; if the stressed node does not sync, it was stuck with a dead sync peer
// (getpeerinfo returns 0 peers when all disconnected; in the corrupted state the
// sync manager still has a dead peer as sync peer, so it ignores the new live one).
func TestSyncManagerRaceCorruption(t *testing.T) {
	stressedHarness, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	require.NoError(t, stressedHarness.SetUp(true, 0))
	t.Cleanup(func() {
		require.NoError(t, stressedHarness.TearDown())
	})

	nodeAddr := stressedHarness.P2PAddress()
	deadline := time.Now().Add(syncRaceRunDuration)
	iter := 0
	var done int
	doneCh := make(chan struct{}, syncRaceConcurrency*2)

	for time.Now().Before(deadline) && iter < syncRaceIterations {
		for i := 0; i < syncRaceConcurrency; i++ {
			go func() {
				_ = fakePeerConn(nodeAddr)
				doneCh <- struct{}{}
			}()
		}
		for i := 0; i < syncRaceConcurrency; i++ {
			<-doneCh
			done++
		}
		iter += syncRaceConcurrency
	}

	// Prove corruption: connect a live node and generate blocks.
	// If the stressed node was corrupted, it will not sync from the new one.
	newHarness, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	require.NoError(t, newHarness.SetUp(true, 0))
	defer func() { _ = newHarness.TearDown() }()

	_, err = newHarness.Client.Generate(syncRaceProofBlocks)
	require.NoError(t, err)

	_, heightBefore, err := stressedHarness.Client.GetBestBlock()
	require.NoError(t, err)

	require.NoError(t, rpctest.ConnectNode(stressedHarness, newHarness),
		"stressed node must connect to the new node")

	time.Sleep(syncRaceProofWait)

	_, heightAfter, err := stressedHarness.Client.GetBestBlock()
	require.NoError(t, err)

	expected := heightBefore + int32(syncRaceProofBlocks)
	require.GreaterOrEqualf(t, heightAfter, expected,
		"sync manager corruption after %d fake "+
			"peer cycles: node stuck with dead "+
			"sync peer (height %d -> %d)",
		done, heightBefore, heightAfter)

	t.Logf("completed %d fake peer cycles; "+
		"node synced (height %d -> %d)",
		done, heightBefore, heightAfter)
}

// TestPreVerackDisconnect verifies that a peer disconnecting
// before completing the version/verack handshake does not corrupt
// the sync manager state. The peer initiates the v2 handshake but
// disconnects before verack completes. The node must remain healthy.
func TestPreVerackDisconnect(t *testing.T) {
	harness, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	require.NoError(t, harness.SetUp(true, 0))
	t.Cleanup(func() { _ = harness.TearDown() })

	nodeAddr := harness.P2PAddress()

	for i := 0; i < 50; i++ {
		conn, err := net.DialTimeout("tcp", nodeAddr, 5*time.Second)
		if err != nil {
			continue
		}

		p, err := peer.NewOutboundPeer(&peer.Config{
			UserAgentName:    "fake-peer",
			UserAgentVersion: "1.0.0",
			Services:         wire.SFNodeNetwork | wire.SFNodeWitness | wire.SFNodeP2PV2,
			ChainParams:      &chaincfg.SimNetParams,
		}, nodeAddr)
		if err == nil {
			p.AssociateConnection(conn)
			time.Sleep(10 * time.Millisecond)
			p.Disconnect()
			p.WaitForDisconnect()
		} else {
			conn.Close()
		}
	}

	// Allow the node time to process all the disconnects.
	time.Sleep(2 * time.Second)

	// Verify the node is still healthy: connect a real peer, generate
	// blocks, and confirm the harness syncs them.
	helper, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	require.NoError(t, helper.SetUp(true, 0))
	defer func() { _ = helper.TearDown() }()

	_, err = helper.Client.Generate(3)
	require.NoError(t, err)

	_, heightBefore, err := harness.Client.GetBestBlock()
	require.NoError(t, err)

	require.NoError(t, rpctest.ConnectNode(harness, helper))

	time.Sleep(5 * time.Second)

	_, heightAfter, err := harness.Client.GetBestBlock()
	require.NoError(t, err)

	require.GreaterOrEqual(t, heightAfter, heightBefore+3,
		"node failed to sync after pre-verack disconnects")

	t.Logf("node healthy after 50 pre-verack disconnects "+
		"(height %d -> %d)", heightBefore, heightAfter)
}
