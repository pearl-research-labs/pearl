// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"crypto/sha256"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/metrics"
	"github.com/stretchr/testify/require"
)

// scrapeSeries returns the value of one series from a live scrape, or 0 if it is absent. The metrics
// package keeps its collectors unexported, so a scrape is how this package observes them. Counters
// accumulate across every test in the binary, so callers compare deltas, never absolute values.
func scrapeSeries(t *testing.T, series string) float64 {
	t.Helper()

	srv, err := metrics.NewServer([]string{"127.0.0.1:0"})
	require.NoError(t, err)
	srv.Start()
	defer srv.Stop()

	resp, err := http.Get("http://" + srv.ListenAddrs()[0] + "/metrics")
	require.NoError(t, err)
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)

	for _, line := range strings.Split(string(body), "\n") {
		if value, ok := strings.CutPrefix(line, series+" "); ok {
			v, err := strconv.ParseFloat(value, 64)
			require.NoError(t, err)
			return v
		}
	}
	return 0
}

// WebSocket-only commands never reach standardCmdResult, so runCommand must record them itself, under
// their own name rather than unknown.
func TestRunCommandRecordsWebsocketOnlyMethods(t *testing.T) {
	const series = `pearld_rpc_requests_total{method="session",result="success"}`
	before := scrapeSeries(t, series)

	client := &wsClient{addr: "test-client"}
	client.runCommand(&parsedRPCCmd{jsonrpc: btcjson.RpcVersion1, id: float64(1), method: "session"})

	require.GreaterOrEqual(t, scrapeSeries(t, series)-before, 1.0)
}

// A wrong in-band authenticate is as much an auth failure as bad HTTP Basic credentials.
func TestWrongWebsocketCredentialsCounted(t *testing.T) {
	const series = "pearld_rpc_auth_failures_total"
	before := scrapeSeries(t, series)

	c := &wsClient{server: &rpcServer{adminCredHash: sha256.Sum256([]byte("admin:adminpass"))}}
	var req btcjson.Request
	require.NoError(t, json.Unmarshal(
		[]byte(`{"jsonrpc":"1.0","method":"authenticate","params":["admin","wrong"],"id":1}`), &req))
	require.True(t, c.authorizeRequest(&req).disconnect)

	require.GreaterOrEqual(t, scrapeSeries(t, series)-before, 1.0)
}

// disconnectPeer takes the peer out of its list, so handleDonePeerMsg cannot count the disconnect later.
func TestDisconnectPeerCountsDisconnect(t *testing.T) {
	const series = `pearld_p2p_peer_disconnects_total{direction="inbound"}`
	before := scrapeSeries(t, series)

	_, sp := newTestServerPeer(t)
	peers := map[int32]*serverPeer{sp.ID(): sp}
	require.True(t, disconnectPeer(peers, func(*serverPeer) bool { return true }, nil))

	require.Empty(t, peers)
	require.GreaterOrEqual(t, scrapeSeries(t, series)-before, 1.0)
}

// Scrapes read peer counts from the published copies, so they must track peerState, counting persistent
// peers as outbound.
func TestPublishPeerCounts(t *testing.T) {
	t.Parallel()

	s := &server{}
	s.publishPeerCounts(&peerState{
		inboundPeers:    map[int32]*serverPeer{1: nil, 2: nil, 3: nil},
		outboundPeers:   map[int32]*serverPeer{4: nil},
		persistentPeers: map[int32]*serverPeer{5: nil},
	})

	inbound, outbound := s.MetricsSource().PeerCount()
	require.Equal(t, int64(3), inbound)
	require.Equal(t, int64(2), outbound)
}
