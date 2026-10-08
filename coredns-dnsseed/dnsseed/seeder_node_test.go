//go:build rpctest
// +build rpctest

package dnsseed

import (
	"context"
	"net/netip"
	"testing"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/integration/rpctest"
	"github.com/stretchr/testify/require"
)

// TestSeederAcceptsLiveNode verifies that the seeder completes the handshake
// with a live pearld, which requires the node to satisfy the serving policy.
func TestSeederAcceptsLiveNode(t *testing.T) {
	h, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	require.NoError(t, h.SetUp(false, 0))
	t.Cleanup(func() { require.NoError(t, h.TearDown()) })

	s := newTestSeeder(t, "simnet")
	t.Cleanup(s.disconnectAllPeers)

	addr := netip.MustParseAddrPort(h.P2PAddress())
	_, err = s.connect(context.Background(), addr)
	require.NoError(t, err)
	require.NotNil(t, s.livePeer(addr), "node must be a live, handshaken peer")
}
