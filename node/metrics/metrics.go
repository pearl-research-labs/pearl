package metrics

import (
	"sync"
	"time"

	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
)

const Namespace = "pearld"

var (
	// registry is the private Prometheus registry for pearld.
	registry *prometheus.Registry

	// Chain counters
	chainBlocksConnectedTotal    prometheus.Counter
	chainBlocksDisconnectedTotal prometheus.Counter
	chainBlocksAcceptedTotal     prometheus.Counter
	chainBlocksRejectedTotal     *prometheus.CounterVec
	chainReorgDepthBlocks        prometheus.Histogram

	// Timings
	chainBlockProcessSeconds prometheus.Histogram
	chainBlockSanitySeconds  prometheus.Histogram
	mempoolTxProcessSeconds  prometheus.Histogram
	utxoCacheFlushSeconds    prometheus.Histogram

	// Mempool
	mempoolTxRejectedTotal *prometheus.CounterVec

	// UTXO cache
	utxoCacheEntries  prometheus.Gauge
	utxoCacheBytes    prometheus.Gauge
	utxoCacheMaxBytes prometheus.Gauge

	// P2P counters
	p2pPeerConnectsTotal    *prometheus.CounterVec
	p2pPeerDisconnectsTotal *prometheus.CounterVec
	p2pPeersBannedTotal     prometheus.Counter
	p2pPeersRejectedTotal   *prometheus.CounterVec
	p2pWireBytesTotal       *prometheus.CounterVec
	p2pWireMessagesTotal    *prometheus.CounterVec

	// RPC metrics
	rpcRequestsTotal       *prometheus.CounterVec
	rpcRequestDurationSecs *prometheus.HistogramVec
	rpcAuthFailuresTotal   prometheus.Counter
	rpcWebsocketClients    prometheus.Gauge

	// Info metric
	pearldInfo *prometheus.GaugeVec

	// Known RPC methods set for sanitization
	knownRPCMethodsMu sync.RWMutex
	knownRPCMethods   map[string]struct{}
)

func init() {
	registry = prometheus.NewRegistry()

	// Register Go and Process collectors
	registry.MustRegister(collectors.NewGoCollector())
	registry.MustRegister(collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))

	// Chain events
	chainBlocksConnectedTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: Namespace,
		Name:      "chain_blocks_connected_total",
		Help:      "Total number of blocks connected to the main chain",
	})
	chainBlocksDisconnectedTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: Namespace,
		Name:      "chain_blocks_disconnected_total",
		Help:      "Total number of blocks disconnected from the main chain",
	})
	chainBlocksAcceptedTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: Namespace,
		Name:      "chain_blocks_accepted_total",
		Help:      "Total number of blocks accepted into the blockchain",
	})
	chainBlocksRejectedTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: Namespace,
		Name:      "chain_blocks_rejected_total",
		Help:      "Total number of blocks rejected, by rule violated; duplicates are not counted",
	}, []string{"reason"})
	chainReorgDepthBlocks = prometheus.NewHistogram(prometheus.HistogramOpts{
		Namespace: Namespace,
		Name:      "chain_reorg_depth_blocks",
		Help:      "Blocks disconnected by each chain reorganization",
		Buckets:   []float64{1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048},
	})

	// Timings
	chainBlockProcessSeconds = newTiming("chain_block_process_duration_seconds",
		"Time to process a block that was accepted, including sanity, certificate and orphan handling")
	chainBlockSanitySeconds = newTiming("chain_block_sanity_duration_seconds",
		"Time for the context-free checks of a block that passed them, mostly certificate verification")
	mempoolTxProcessSeconds = newTiming("mempool_tx_process_duration_seconds",
		"Time to process a transaction that was accepted, including any orphans it promoted")
	utxoCacheFlushSeconds = newTiming("utxo_cache_flush_duration_seconds",
		"Time to write the UTXO cache into a flush transaction, excluding the database commit")

	// Mempool
	mempoolTxRejectedTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: Namespace,
		Name:      "mempool_tx_rejected_total",
		Help:      "Total number of transactions rejected from peers or sendrawtransaction, by reject code; duplicates are not counted",
	}, []string{"reason"})

	// UTXO cache
	utxoCacheEntries = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: Namespace,
		Name:      "utxo_cache_entries",
		Help:      "Entries in the UTXO cache, as of the last connected block or cache flush",
	})
	utxoCacheBytes = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: Namespace,
		Name:      "utxo_cache_bytes",
		Help:      "Memory used by the UTXO cache in bytes, as of the last connected block or cache flush",
	})
	utxoCacheMaxBytes = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: Namespace,
		Name:      "utxo_cache_max_bytes",
		Help:      "Configured UTXO cache memory limit in bytes",
	})

	// P2P metrics
	p2pPeerConnectsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: Namespace,
		Name:      "p2p_peer_connects_total",
		Help:      "Total number of peer connections established",
	}, []string{"direction"})

	p2pPeerDisconnectsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: Namespace,
		Name:      "p2p_peer_disconnects_total",
		Help:      "Total number of peer disconnections",
	}, []string{"direction"})

	p2pPeersBannedTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: Namespace,
		Name:      "p2p_peers_banned_total",
		Help:      "Total number of peers banned due to misbehavior",
	})

	p2pPeersRejectedTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: Namespace,
		Name:      "p2p_peers_rejected_total",
		Help:      "Total number of peer connection attempts rejected",
	}, []string{"reason"})

	p2pWireBytesTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: Namespace,
		Name:      "p2p_wire_bytes_total",
		Help:      "Total bytes transferred over the P2P wire",
	}, []string{"direction", "command"})

	p2pWireMessagesTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: Namespace,
		Name:      "p2p_wire_messages_total",
		Help:      "Total P2P wire messages sent or received",
	}, []string{"direction", "command"})

	// RPC metrics
	rpcRequestsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: Namespace,
		Name:      "rpc_requests_total",
		Help:      "Total JSON-RPC requests processed",
	}, []string{"method", "result"})

	rpcRequestDurationSecs = prometheus.NewHistogramVec(timingOpts("rpc_request_duration_seconds",
		"JSON-RPC request execution duration in seconds"), []string{"method"})

	rpcAuthFailuresTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: Namespace,
		Name:      "rpc_auth_failures_total",
		Help:      "Total RPC authentication failures",
	})

	rpcWebsocketClients = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: Namespace,
		Name:      "rpc_websocket_clients",
		Help:      "Current number of connected WebSocket clients",
	})

	// Info metric
	pearldInfo = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: Namespace,
		Name:      "info",
		Help:      "Pearld node version and network information",
	}, []string{"version", "network", "protocol_version"})

	// Register metrics with private registry
	registry.MustRegister(
		chainBlocksConnectedTotal,
		chainBlocksDisconnectedTotal,
		chainBlocksAcceptedTotal,
		chainBlocksRejectedTotal,
		chainReorgDepthBlocks,
		chainBlockProcessSeconds,
		chainBlockSanitySeconds,
		mempoolTxProcessSeconds,
		utxoCacheFlushSeconds,
		mempoolTxRejectedTotal,
		utxoCacheEntries,
		utxoCacheBytes,
		utxoCacheMaxBytes,
		p2pPeerConnectsTotal,
		p2pPeerDisconnectsTotal,
		p2pPeersBannedTotal,
		p2pPeersRejectedTotal,
		p2pWireBytesTotal,
		p2pWireMessagesTotal,
		rpcRequestsTotal,
		rpcRequestDurationSecs,
		rpcAuthFailuresTotal,
		rpcWebsocketClients,
		pearldInfo,
	)
}

// timingOpts configures a native histogram with buckets √2 apart, capped at 34 (wider ranges halve
// the resolution). PromQL's histogram_avg, histogram_stddev and histogram_quantile apply.
func timingOpts(name, help string) prometheus.HistogramOpts {
	return prometheus.HistogramOpts{
		Namespace:                      Namespace,
		Name:                           name,
		Help:                           help,
		NativeHistogramBucketFactor:    1.5,
		NativeHistogramMaxBucketNumber: 34,
	}
}

func newTiming(name, help string) prometheus.Histogram {
	return prometheus.NewHistogram(timingOpts(name, help))
}

// InitKnownRPCMethods initializes the set of valid RPC methods for label sanitization.
func InitKnownRPCMethods(methods []string) {
	knownRPCMethodsMu.Lock()
	defer knownRPCMethodsMu.Unlock()

	knownRPCMethods = make(map[string]struct{}, len(methods))
	for _, m := range methods {
		knownRPCMethods[m] = struct{}{}
	}
}

// SanitizeRPCMethod returns method if it is in the known set, or "unknown" otherwise.
func SanitizeRPCMethod(method string) string {
	knownRPCMethodsMu.RLock()
	defer knownRPCMethodsMu.RUnlock()

	if _, ok := knownRPCMethods[method]; ok {
		return method
	}
	return "unknown"
}

// SetInfo sets the pearld_info metric labels.
func SetInfo(version, network, protocolVersion string) {
	pearldInfo.WithLabelValues(version, network, protocolVersion).Set(1)
}

// Chain event recording helpers
func RecordBlockConnected() {
	chainBlocksConnectedTotal.Inc()
}

func RecordBlockDisconnected() {
	chainBlocksDisconnectedTotal.Inc()
}

func RecordBlockAccepted() {
	chainBlocksAcceptedTotal.Inc()
}

// RecordBlockRejected counts a rejected block. reason must come from a fixed set, such as a
// blockchain.ErrorCode name, so peers cannot grow the series count.
func RecordBlockRejected(reason string) {
	chainBlocksRejectedTotal.WithLabelValues(reason).Inc()
}

// RecordBlockProcessed times a block that was accepted (not an orphan).
func RecordBlockProcessed(d time.Duration) {
	chainBlockProcessSeconds.Observe(d.Seconds())
}

// RecordBlockSanityChecked times the context-free checks of a block that passed them.
func RecordBlockSanityChecked(d time.Duration) {
	chainBlockSanitySeconds.Observe(d.Seconds())
}

// RecordReorg records how many blocks a reorganization disconnected.
func RecordReorg(depth int) {
	chainReorgDepthBlocks.Observe(float64(depth))
}

// RecordUtxoCacheFlushed times a UTXO cache flush, excluding the database commit.
func RecordUtxoCacheFlushed(d time.Duration) {
	utxoCacheFlushSeconds.Observe(d.Seconds())
}

// RecordTxProcessed times a transaction that was accepted into the pool (not an orphan).
func RecordTxProcessed(d time.Duration) {
	mempoolTxProcessSeconds.Observe(d.Seconds())
}

// RecordUtxoCache sets the UTXO cache occupancy.
func RecordUtxoCache(entries int, memoryBytes, maxMemoryBytes uint64) {
	utxoCacheEntries.Set(float64(entries))
	utxoCacheBytes.Set(float64(memoryBytes))
	utxoCacheMaxBytes.Set(float64(maxMemoryBytes))
}

// Recorder adapts the Record functions to blockchain.Metrics, which blockchain uses to avoid importing this package.
type Recorder struct{}

func (Recorder) Reorganized(depth int)              { RecordReorg(depth) }
func (Recorder) BlockSanityChecked(d time.Duration) { RecordBlockSanityChecked(d) }
func (Recorder) UtxoCacheFlushed(d time.Duration)   { RecordUtxoCacheFlushed(d) }
func (Recorder) UtxoCacheChanged(entries int, memoryBytes, maxMemoryBytes uint64) {
	RecordUtxoCache(entries, memoryBytes, maxMemoryBytes)
}

// RecordTxRejected counts a rejected transaction by reject code. Duplicates are skipped, as for
// blocks: wallets rebroadcast routinely, and every resend of a known transaction is rejected as one.
func RecordTxRejected(code wire.RejectCode) {
	if code == wire.RejectDuplicate {
		return
	}
	mempoolTxRejectedTotal.WithLabelValues(code.String()).Inc()
}

// P2P recording helpers
func RecordPeerConnect(inbound bool) {
	p2pPeerConnectsTotal.WithLabelValues(directionLabel(inbound)).Inc()
}

func RecordPeerDisconnect(inbound bool) {
	p2pPeerDisconnectsTotal.WithLabelValues(directionLabel(inbound)).Inc()
}

func RecordPeerBanned() {
	p2pPeersBannedTotal.Inc()
}

func RecordPeerRejected(reason string) {
	p2pPeersRejectedTotal.WithLabelValues(reason).Inc()
}

// RecordWireMessage records a single P2P message.  Callers should only install
// the wire hooks when the metrics server is configured, since this runs on every
// frame.
func RecordWireMessage(inbound bool, command string, bytes int) {
	if command == "" {
		command = "unknown"
	}
	dir := directionLabel(inbound)
	p2pWireMessagesTotal.WithLabelValues(dir, command).Inc()
	p2pWireBytesTotal.WithLabelValues(dir, command).Add(float64(bytes))
}

func directionLabel(inbound bool) string {
	if inbound {
		return "inbound"
	}
	return "outbound"
}

// RPC recording helpers
func RecordRPCRequest(method string, duration time.Duration, err error) {
	method = SanitizeRPCMethod(method)
	result := "success"
	if err != nil {
		result = "error"
	}
	rpcRequestsTotal.WithLabelValues(method, result).Inc()
	rpcRequestDurationSecs.WithLabelValues(method).Observe(duration.Seconds())
}

// RecordRPCMethodNotFound counts a request rejected before dispatch.  No
// latency is observed because no handler ran.
func RecordRPCMethodNotFound(method string) {
	rpcRequestsTotal.WithLabelValues(SanitizeRPCMethod(method), "error").Inc()
}

func RecordRPCAuthFailure() {
	rpcAuthFailuresTotal.Inc()
}

func AddWSClient() {
	rpcWebsocketClients.Inc()
}

func RemoveWSClient() {
	rpcWebsocketClients.Dec()
}
