package metrics

import (
	"errors"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A native-only timing gives text-format readers, and any Prometheus not ingesting native histograms,
// no buckets to compute quantiles from.
func TestTimingsExposeBothForms(t *testing.T) {
	RecordRPCRequest("getblock", time.Millisecond, nil)

	families, err := registry.Gather()
	require.NoError(t, err)

	timings := 0
	for _, mf := range families {
		if mf.GetType() != dto.MetricType_HISTOGRAM || !strings.HasSuffix(mf.GetName(), "_seconds") {
			continue
		}
		timings++
		for _, m := range mf.GetMetric() {
			h := m.GetHistogram()
			assert.Len(t, h.GetBucket(), len(timingBuckets), "%s classic buckets", mf.GetName())
			assert.NotNil(t, h.Schema, "%s native schema", mf.GetName())
		}
	}
	assert.GreaterOrEqual(t, timings, 5)
}

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

// Rebroadcasts are routine, so a duplicate must not show up as a rejection at all.
func TestRecordTxRejectedSkipsDuplicates(t *testing.T) {
	series := testutil.CollectAndCount(mempoolTxRejectedTotal)
	RecordTxRejected(wire.RejectDuplicate)
	assert.Equal(t, series, testutil.CollectAndCount(mempoolTxRejectedTotal), "a duplicate must not create a series")

	dust := testutil.ToFloat64(mempoolTxRejectedTotal.WithLabelValues(wire.RejectDust.String()))
	RecordTxRejected(wire.RejectDust)
	assert.Equal(t, dust+1, testutil.ToFloat64(mempoolTxRejectedTotal.WithLabelValues(wire.RejectDust.String())))
}

// blockingCollector holds every gather in Collect until release is closed, like a collector stuck
// behind a node lock.
type blockingCollector struct {
	desc    *prometheus.Desc
	entered atomic.Int32
	release chan struct{}
}

func (c *blockingCollector) Describe(ch chan<- *prometheus.Desc) { ch <- c.desc }

func (c *blockingCollector) Collect(chan<- prometheus.Metric) {
	c.entered.Add(1)
	<-c.release
}

// Stalled gathers must not pile up on an unauthenticated endpoint: once every in-flight slot is held,
// further scrapes are turned away instead of queuing.
func TestServerRejectsScrapesBeyondInFlightLimit(t *testing.T) {
	blocker := &blockingCollector{
		desc:    prometheus.NewDesc("pearld_test_blocking", "Blocks every gather", nil, nil),
		release: make(chan struct{}),
	}
	registry.MustRegister(blocker)
	t.Cleanup(func() { registry.Unregister(blocker) })

	server, err := NewServer([]string{"127.0.0.1:0"})
	require.NoError(t, err)
	server.Start()
	t.Cleanup(server.Stop)
	url := "http://" + server.ListenAddrs()[0] + "/metrics"

	var stalled sync.WaitGroup
	for range maxScrapesInFlight {
		stalled.Add(1)
		go func() {
			defer stalled.Done()
			if resp, err := http.Get(url); err == nil {
				resp.Body.Close()
			}
		}()
	}
	require.Eventually(t, func() bool { return blocker.entered.Load() == maxScrapesInFlight },
		5*time.Second, 10*time.Millisecond)

	// The handler timeout would also answer 503, but only after scrapeTimeout; the cap answers at once.
	impatient := &http.Client{Timeout: scrapeTimeout / 5}
	resp, err := impatient.Get(url)
	require.NoError(t, err, "a scrape over the cap must be turned away immediately")
	resp.Body.Close()
	assert.Equal(t, http.StatusServiceUnavailable, resp.StatusCode)

	close(blocker.release)
	stalled.Wait()
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

// A server stopped before Start must release its ports.
func TestStopWithoutStartReleasesListeners(t *testing.T) {
	server, err := NewServer([]string{"127.0.0.1:0"})
	require.NoError(t, err)
	addr := server.ListenAddrs()[0]

	server.Stop()

	l, err := net.Listen("tcp", addr)
	require.NoError(t, err, "port %s is still bound after Stop", addr)
	require.NoError(t, l.Close())
}
