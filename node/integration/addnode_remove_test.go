//go:build rpctest
// +build rpctest

package integration

import (
	"errors"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/integration/rpctest"
	"github.com/pearl-research-labs/pearl/node/rpcclient"
	"github.com/stretchr/testify/require"
)

// addNodeRemovedQuietPeriod is how long a removed peer must stay disconnected. connmgr redials a persistent peer
// after retryCount*5s, and a connection shorter than 30s does not reset retryCount, so after the reconnect in
// TestAddNodeRemoveStopsReconnect a node that still redialed would do so at 10s.
const addNodeRemovedQuietPeriod = 15 * time.Second

// TestAddNodeRemoveStopsReconnect checks that a node redials a peer added with addnode add after the peer drops
// the connection, and stops redialing it once it is removed with addnode remove.
func TestAddNodeRemoveStopsReconnect(t *testing.T) {
	a, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	require.NoError(t, a.SetUp(false, 0))
	t.Cleanup(func() { require.NoError(t, a.TearDown()) })

	b, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	require.NoError(t, b.SetUp(false, 0))
	t.Cleanup(func() { require.NoError(t, b.TearDown()) })

	// ConnectNode uses addnode add, so b is a persistent peer of a.
	require.NoError(t, rpctest.ConnectNode(a, b))
	bAddr := b.P2PAddress()
	connectedToB := func(want bool) func() bool {
		return func() bool {
			peers, err := a.Client.GetPeerInfo()
			if err != nil {
				return false
			}
			for _, p := range peers {
				if p.Addr == bAddr {
					return want
				}
			}
			return !want
		}
	}

	// b drops the connection, so a must redial it.
	peers, err := a.Client.GetPeerInfo()
	require.NoError(t, err)
	var aAddr string
	for _, p := range peers {
		if p.Addr == bAddr {
			aAddr = p.AddrLocal
		}
	}
	require.NotEmpty(t, aAddr, "a is not connected to b")
	require.NoError(t, b.Client.Node(btcjson.NDisconnect, aAddr, nil))
	require.Eventually(t, connectedToB(false), 20*time.Second, 10*time.Millisecond,
		"b did not drop a")
	require.Eventually(t, connectedToB(true), 20*time.Second, 10*time.Millisecond,
		"a did not redial its persistent peer b")

	// addnode remove disconnects b, and a must not redial it.
	require.NoError(t, a.Client.AddNode(bAddr, rpcclient.ANRemove))
	require.Eventually(t, connectedToB(false), 20*time.Second, 10*time.Millisecond,
		"addnode remove did not disconnect b")
	require.Never(t, connectedToB(true), addNodeRemovedQuietPeriod, 10*time.Millisecond,
		"a redialed b after addnode remove")

	_, err = a.Client.GetAddedNodeInfoNoDNS(bAddr)
	var rpcErr *btcjson.RPCError
	require.True(t, errors.As(err, &rpcErr), "getaddednodeinfo: %v", err)
	require.Equal(t, btcjson.ErrRPCClientNodeNotAdded, rpcErr.Code)
}
