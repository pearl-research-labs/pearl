package metrics

import (
	"errors"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMethodSanitization(t *testing.T) {
	InitKnownRPCMethods([]string{"getblock", "getpeerinfo", "getmempoolinfo"})

	assert.Equal(t, "getblock", SanitizeRPCMethod("getblock"))
	assert.Equal(t, "getpeerinfo", SanitizeRPCMethod("getpeerinfo"))
	assert.Equal(t, "getmempoolinfo", SanitizeRPCMethod("getmempoolinfo"))
	assert.Equal(t, "unknown", SanitizeRPCMethod("malicious_unregistered_method"))
	assert.Equal(t, "unknown", SanitizeRPCMethod("dropdatabase"))
}

func TestRecordingHelpers(t *testing.T) {
	InitKnownRPCMethods([]string{"getblock", "getpeerinfo"})

	SetInfo("v0.1.0", "simnet", "70016")
	RecordBlockConnected()
	RecordBlockDisconnected()
	RecordBlockAccepted()
	RecordPeerConnect(true)
	RecordPeerDisconnect(false)
	RecordPeerBanned()
	RecordPeerRejected("max_peers")
	RecordWireMessage(true, "inv", 128)
	RecordWireMessage(false, "block", 1024)
	RecordWireMessage(true, "", 64)
	RecordRPCRequest("getblock", 5*time.Millisecond, nil)
	RecordRPCRequest("unknownMethod", 10*time.Millisecond, errors.New("not found"))
	RecordRPCAuthFailure()
	AddWSClient()

	assert.Greater(t, testutil.ToFloat64(chainBlocksConnectedTotal), float64(0))
	assert.Greater(t, testutil.ToFloat64(chainBlocksDisconnectedTotal), float64(0))
	assert.Greater(t, testutil.ToFloat64(chainBlocksAcceptedTotal), float64(0))
	assert.Greater(t, testutil.ToFloat64(p2pPeersBannedTotal), float64(0))
	assert.Greater(t, testutil.ToFloat64(rpcAuthFailuresTotal), float64(0))
	assert.Equal(t, float64(1), testutil.ToFloat64(rpcWebsocketClients))

	assert.Equal(t, float64(128),
		testutil.ToFloat64(p2pWireBytesTotal.WithLabelValues("inbound", "inv")))
	assert.Equal(t, float64(64),
		testutil.ToFloat64(p2pWireBytesTotal.WithLabelValues("inbound", "unknown")),
		"a message with no command must fall back to the unknown label")

	RemoveWSClient()
	assert.Equal(t, float64(0), testutil.ToFloat64(rpcWebsocketClients))
}

// A rejected method never reaches a handler, so it must count as a request
// without contributing a zero-second sample to the latency histogram.
func TestMethodNotFoundRecordsNoLatency(t *testing.T) {
	InitKnownRPCMethods([]string{"getblock"})

	before := testutil.CollectAndCount(rpcRequestDurationSecs)
	countBefore := testutil.ToFloat64(rpcRequestsTotal.WithLabelValues("unknown", "error"))

	RecordRPCMethodNotFound("nosuchmethod")

	assert.Equal(t, countBefore+1,
		testutil.ToFloat64(rpcRequestsTotal.WithLabelValues("unknown", "error")))
	assert.Equal(t, before, testutil.CollectAndCount(rpcRequestDurationSecs),
		"rejected methods must not add a latency series")
}

func TestScrapeCollector(t *testing.T) {
	now := time.Now().Truncate(time.Second)

	src := Source{
		ChainTipHeight:     func() int32 { return 100 },
		ChainTipTimestamp:  func() time.Time { return now },
		ChainTotalTxs:      func() int64 { return 500 },
		ChainIsCurrent:     func() bool { return true },
		PeerCount:          func() (int64, int64) { return 8, 4 },
		NetTotals:          func() (uint64, uint64) { return 1000, 2000 },
		MempoolTxCount:     func() int { return 25 },
		MempoolBytes:       func() uint64 { return 50000 },
		MempoolMaxBytes:    func() uint64 { return 300000000 },
		MempoolLastUpdated: func() time.Time { return now },
	}

	collector := NewScrapeCollector(src)
	assert.Equal(t, 12, testutil.CollectAndCount(collector))
}

func TestServerAndPprofIsolation(t *testing.T) {
	server, err := NewServer([]string{"127.0.0.1:0"})
	require.NoError(t, err)
	defer server.Stop()

	server.Start()

	addrs := server.ListenAddrs()
	require.Len(t, addrs, 1)

	metricsURL := "http://" + addrs[0] + "/metrics"
	resp, err := http.Get(metricsURL)
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusOK, resp.StatusCode)
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	assert.Contains(t, string(body), "pearld_info")

	// Pprof isolation regression test: ensure /debug/pprof/ returns 404
	// on the metrics server HTTP listener.
	pprofURL := "http://" + addrs[0] + "/debug/pprof/"
	pprofResp, err := http.Get(pprofURL)
	require.NoError(t, err)
	defer pprofResp.Body.Close()

	assert.Equal(t, http.StatusNotFound, pprofResp.StatusCode,
		"Metrics endpoint MUST NOT expose DefaultServeMux or /debug/pprof/")
}
