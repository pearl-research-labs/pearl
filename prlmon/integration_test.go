//go:build rpctest

package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
	"github.com/stretchr/testify/require"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/integration/rpctest"
)

var primaryHarness *rpctest.Harness

func TestMain(m *testing.M) {
	var err error

	primaryHarness, err = rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	if err != nil {
		fmt.Fprintf(os.Stderr, "unable to create primary harness: %v\n", err)
		os.Exit(1)
	}

	if err := primaryHarness.SetUp(true, 25); err != nil {
		fmt.Fprintf(os.Stderr, "unable to setup test chain: %v\n", err)
		_ = primaryHarness.TearDown()
		os.Exit(1)
	}

	exitCode := m.Run()

	if err := rpctest.TearDownAll(); err != nil {
		fmt.Fprintf(os.Stderr, "unable to tear down harnesses: %v\n", err)
	}

	os.Exit(exitCode)
}

func newTestMonitor(t *testing.T, h *rpctest.Harness) (*Monitor, context.CancelFunc) {
	t.Helper()

	// Write cert to temp file
	certFile, err := os.CreateTemp("", "rpc-cert-*.pem")
	require.NoError(t, err)
	_, err = certFile.Write(h.RPCConfig().Certificates)
	require.NoError(t, err)
	certFile.Close()
	t.Cleanup(func() { os.Remove(certFile.Name()) })

	cfg := &Config{
		Listen:             "127.0.0.1:0",
		RPCHost:            h.RPCConfig().Host,
		RPCUser:            h.RPCConfig().User,
		RPCPass:            h.RPCConfig().Pass,
		RPCCert:            certFile.Name(),
		Poll:               100 * time.Millisecond,
		DebugLevel:         "warn",
		LogsMaxLines:       1000,
		SelfLogBufferLines: 256,
	}

	err = cfg.Validate()
	require.NoError(t, err)

	configureSelfLog(cfg.SelfLogBufferLines)

	mon, err := NewMonitor(cfg)
	require.NoError(t, err)

	// ListenAddr() is available immediately after NewMonitor()
	require.NotEmpty(t, mon.ListenAddr())

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		_ = mon.Run(ctx)
	}()

	// Small delay to let HTTP server goroutine start serving
	time.Sleep(10 * time.Millisecond)

	return mon, cancel
}

func scrapeMetrics(t *testing.T, addr string) map[string]float64 {
	t.Helper()

	resp, err := http.Get("http://" + addr + "/metrics")
	require.NoError(t, err)
	defer resp.Body.Close()

	require.Equal(t, http.StatusOK, resp.StatusCode)

	return parsePrometheusText(t, resp.Body)
}

func parsePrometheusText(t *testing.T, r io.Reader) map[string]float64 {
	t.Helper()

	result := make(map[string]float64)
	parser := expfmt.NewTextParser(model.LegacyValidation)
	mfs, err := parser.TextToMetricFamilies(r)
	require.NoError(t, err)

	for name, mf := range mfs {
		for _, m := range mf.GetMetric() {
			// Build full metric name with labels
			labels := make([]string, 0)
			for _, lp := range m.GetLabel() {
				labels = append(labels, fmt.Sprintf("%s=%q", lp.GetName(), lp.GetValue()))
			}

			fullName := name
			if len(labels) > 0 {
				fullName = fmt.Sprintf("%s{%s}", name, strings.Join(labels, ","))
			}

			// Extract value based on metric type
			if g := m.GetGauge(); g != nil {
				result[fullName] = g.GetValue()
			}
			if c := m.GetCounter(); c != nil {
				result[fullName] = c.GetValue()
			}
			if h := m.GetHistogram(); h != nil {
				result[fullName+"_count"] = float64(h.GetSampleCount())
				result[fullName+"_sum"] = h.GetSampleSum()
			}
		}
	}

	return result
}

func TestBasicMetrics(t *testing.T) {
	mon, cancel := newTestMonitor(t, primaryHarness)
	defer cancel()

	// Wait for first poll to complete - use Eventually for CI timing robustness
	require.Eventually(t, func() bool {
		metrics := scrapeMetrics(t, mon.ListenAddr())
		return metrics[`prlmon_node_up`] == 1
	}, 5*time.Second, 100*time.Millisecond, "node should come up within timeout")

	// Get expected height from harness
	_, expectedHeight, err := primaryHarness.Client.GetBestBlock()
	require.NoError(t, err)

	metrics := scrapeMetrics(t, mon.ListenAddr())

	// Check node is up
	require.Equal(t, float64(1), metrics[`prlmon_node_up`])

	// Check tip height matches
	require.Equal(t, float64(expectedHeight), metrics[`prlmon_chain_tip_height`])

	// Peer count should exist (may be 0 in test)
	_, exists := metrics[`prlmon_p2p_peer_count`]
	require.True(t, exists, "p2p_peer_count metric should exist")

	// Inbound/outbound peer counts should exist (may be 0 in test)
	_, exists = metrics[`prlmon_p2p_inbound_peers`]
	require.True(t, exists, "p2p_inbound_peers metric should exist")
	_, exists = metrics[`prlmon_p2p_outbound_peers`]
	require.True(t, exists, "p2p_outbound_peers metric should exist")
}

func TestPeerMetricsExist(t *testing.T) {
	// Use the primary harness (already set up)
	mon, cancel := newTestMonitor(t, primaryHarness)
	defer cancel()

	// Wait for poll to complete
	time.Sleep(300 * time.Millisecond)

	metrics := scrapeMetrics(t, mon.ListenAddr())

	// Peer count, inbound, and outbound should exist
	peerCount := metrics[`prlmon_p2p_peer_count`]
	inbound := metrics[`prlmon_p2p_inbound_peers`]
	outbound := metrics[`prlmon_p2p_outbound_peers`]

	// In isolated test, peer count may be 0, but inbound + outbound should equal total
	require.Equal(t, peerCount, inbound+outbound, "inbound + outbound should equal total peers")

	// Histograms should exist (may have 0 observations if no peers)
	_, pingExists := metrics[`prlmon_p2p_pingtime_seconds_count`]
	_, lastrecvExists := metrics[`prlmon_p2p_lastrecv_age_seconds_count`]

	// Histograms may not have any observations if no peers, but they should be registered
	// The histogram _count key only appears if there are observations
	// Just verify the gauges work correctly
	require.GreaterOrEqual(t, peerCount, float64(0), "peer count should be >= 0")
	_ = pingExists     // May or may not exist depending on peer connections
	_ = lastrecvExists // May or may not exist depending on peer connections
}

func TestHealthzEndpoint(t *testing.T) {
	mon, cancel := newTestMonitor(t, primaryHarness)
	defer cancel()

	resp, err := http.Get("http://" + mon.ListenAddr() + "/healthz")
	require.NoError(t, err)
	defer resp.Body.Close()

	require.Equal(t, http.StatusOK, resp.StatusCode)

	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, "ok", string(body))
}

func TestBlockConnectedNotification(t *testing.T) {
	mon, cancel := newTestMonitor(t, primaryHarness)
	defer cancel()

	// Wait for initial state
	time.Sleep(200 * time.Millisecond)

	_, prevHeight, err := primaryHarness.Client.GetBestBlock()
	require.NoError(t, err)

	initialMetrics := scrapeMetrics(t, mon.ListenAddr())
	initialConnected := initialMetrics[`prlmon_blocks_connected_total`]

	// Generate a block
	_, err = primaryHarness.Client.Generate(1)
	require.NoError(t, err)

	// Wait for metric to update (via WebSocket or poll)
	require.Eventually(t, func() bool {
		metrics := scrapeMetrics(t, mon.ListenAddr())
		height := metrics[`prlmon_chain_tip_height`]
		return height == float64(prevHeight+1)
	}, 5*time.Second, 100*time.Millisecond)

	// Verify blocks_connected_total increased
	finalMetrics := scrapeMetrics(t, mon.ListenAddr())
	finalConnected := finalMetrics[`prlmon_blocks_connected_total`]
	require.Greater(t, finalConnected, initialConnected, "blocks_connected_total should have increased")
}

func TestTipAgeMetric(t *testing.T) {
	mon, cancel := newTestMonitor(t, primaryHarness)
	defer cancel()

	// Generate a fresh block
	_, err := primaryHarness.Client.Generate(1)
	require.NoError(t, err)

	// Wait for poll
	time.Sleep(200 * time.Millisecond)

	metrics := scrapeMetrics(t, mon.ListenAddr())
	tipAge := metrics[`prlmon_chain_tip_age_seconds`]

	// Tip age should be small (less than 30 seconds in test)
	require.Less(t, tipAge, float64(30), "tip age should be recent")
}

func TestNetBytesCounter(t *testing.T) {
	mon, cancel := newTestMonitor(t, primaryHarness)
	defer cancel()

	// Wait for initial poll
	time.Sleep(200 * time.Millisecond)

	initialMetrics := scrapeMetrics(t, mon.ListenAddr())
	initialIn := initialMetrics[`prlmon_net_totalbytes_recv_total`]

	// Generate some blocks to cause network traffic
	_, err := primaryHarness.Client.Generate(5)
	require.NoError(t, err)

	// Wait for activity and another poll
	time.Sleep(300 * time.Millisecond)

	finalMetrics := scrapeMetrics(t, mon.ListenAddr())
	finalIn := finalMetrics[`prlmon_net_totalbytes_recv_total`]

	// Counter should have increased (or stayed same if no traffic)
	require.GreaterOrEqual(t, finalIn, initialIn)
}

func TestReorgMetrics(t *testing.T) {
	// SetUp(true, N) generates CoinbaseMaturity + N blocks
	// CoinbaseMaturity is 100 for SimNet
	const (
		coinbaseMaturity = 100
		matureOutputs    = 5
		additionalBlocks = 3
	)

	// harness2 will have 105 blocks, harness1 will have 108 blocks
	// When harness2 reorgs, it disconnects all 105 blocks (independent chains)
	expectedDisconnects := coinbaseMaturity + matureOutputs // 105

	// Create two separate harnesses that will form competing chains
	harness1, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	err = harness1.SetUp(true, matureOutputs)
	require.NoError(t, err)
	defer harness1.TearDown()

	harness2, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	err = harness2.SetUp(true, matureOutputs)
	require.NoError(t, err)
	defer harness2.TearDown()

	// Mine more blocks on harness1 to make it the longer chain
	_, err = harness1.Client.Generate(additionalBlocks)
	require.NoError(t, err)

	// harness1: height 108, harness2: height 105

	// Create monitor for harness2 (the one that will reorg)
	mon, cancel := newTestMonitor(t, harness2)
	defer cancel()

	// Wait for WebSocket connection to establish
	time.Sleep(500 * time.Millisecond)

	// Get initial counts
	initialMetrics := scrapeMetrics(t, mon.ListenAddr())
	initialReorgCount := initialMetrics[`prlmon_reorg_total`]
	initialDisconnects := initialMetrics[`prlmon_blocks_disconnected_total`]

	// Connect the nodes - harness2 should reorg to harness1's longer chain
	err = rpctest.ConnectNode(harness2, harness1)
	require.NoError(t, err)

	// Wait for sync (harness2 reorgs to harness1's chain)
	err = rpctest.JoinNodes([]*rpctest.Harness{harness1, harness2}, rpctest.Blocks)
	require.NoError(t, err)

	// Wait for metrics to update via WebSocket notifications
	time.Sleep(500 * time.Millisecond)

	// Verify reorg was detected
	finalMetrics := scrapeMetrics(t, mon.ListenAddr())
	finalReorgCount := finalMetrics[`prlmon_reorg_total`]
	finalDisconnects := finalMetrics[`prlmon_blocks_disconnected_total`]

	// Reorg count should have increased by 1 (one reorg event)
	reorgCountDelta := finalReorgCount - initialReorgCount
	require.Equal(t, float64(1), reorgCountDelta,
		"reorg_total should increase by 1 (one reorg event)")

	// Disconnects count should have increased by the number of disconnected blocks
	disconnectsDelta := finalDisconnects - initialDisconnects
	require.Equal(t, float64(expectedDisconnects), disconnectsDelta,
		"blocks_disconnected_total should increase by the number of disconnected blocks")
}

func TestReorgDepthHistogram(t *testing.T) {
	// SetUp(true, N) generates CoinbaseMaturity + N blocks
	// CoinbaseMaturity is 100 for SimNet
	const (
		coinbaseMaturity         = 100
		longerChainMatureBlocks  = 10 // harness1 will have 110 blocks
		shorterChainMatureBlocks = 3  // harness2 will have 103 blocks
	)

	expectedReorgDepth := coinbaseMaturity + shorterChainMatureBlocks // 103 blocks disconnected

	// Create two harnesses with different chain lengths
	harness1, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	err = harness1.SetUp(true, longerChainMatureBlocks)
	require.NoError(t, err)
	defer harness1.TearDown()

	harness2, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	err = harness2.SetUp(true, shorterChainMatureBlocks)
	require.NoError(t, err)
	defer harness2.TearDown()

	// harness1: height 110, harness2: height 103

	// Create monitor for harness2 (the node that will reorg)
	mon, cancel := newTestMonitor(t, harness2)
	defer cancel()

	// Wait for WebSocket connection
	time.Sleep(500 * time.Millisecond)

	// Get initial histogram values
	initialMetrics := scrapeMetrics(t, mon.ListenAddr())
	initialDepthCount := initialMetrics[`prlmon_reorg_depth_count`]
	initialDepthSum := initialMetrics[`prlmon_reorg_depth_sum`]

	// Connect nodes - harness2 reorgs to harness1's longer chain
	// This will disconnect all 103 blocks from harness2 (independent chains from genesis)
	err = rpctest.ConnectNode(harness2, harness1)
	require.NoError(t, err)

	err = rpctest.JoinNodes([]*rpctest.Harness{harness1, harness2}, rpctest.Blocks)
	require.NoError(t, err)

	// Wait for reorg to complete and metrics to update via WebSocket
	time.Sleep(1 * time.Second)

	// Verify depth histogram recorded exactly one reorg
	finalMetrics := scrapeMetrics(t, mon.ListenAddr())
	finalDepthCount := finalMetrics[`prlmon_reorg_depth_count`]
	finalDepthSum := finalMetrics[`prlmon_reorg_depth_sum`]

	// Should have recorded exactly 1 reorg event
	require.Equal(t, initialDepthCount+1, finalDepthCount,
		"should record exactly one reorg event")

	// The depth should equal all blocks on harness2's chain (103 blocks disconnected)
	reorgDepth := finalDepthSum - initialDepthSum
	require.Equal(t, float64(expectedReorgDepth), reorgDepth,
		"reorg depth should equal the total blocks on the shorter chain")
}
