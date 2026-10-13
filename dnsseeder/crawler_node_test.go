//go:build rpctest
// +build rpctest

package main

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/integration/rpctest"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// TestCrawlLiveNode verifies that the crawler completes the handshake with a
// live pearld and records the services the node advertises.
func TestCrawlLiveNode(t *testing.T) {
	h, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	require.NoError(t, h.SetUp(false, 0))
	t.Cleanup(func() { require.NoError(t, h.TearDown()) })

	s := &dnsseeder{
		chainParams: &chaincfg.SimNetParams,
		pver:        wire.ProtocolVersion,
		theList:     make(map[string]*node),
		maxSize:     1250,
	}
	r := &result{node: h.P2PAddress()}

	_, crawlErr := crawlIP(s, r)
	require.Nil(t, crawlErr)
	require.True(t, r.services.HasFlag(wire.SFNodeNetwork|wire.SFNodeWitness),
		"recorded services: %v", r.services)
}
