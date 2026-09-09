package metrics

import (
	"sync"
	"time"

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

	rpcRequestDurationSecs = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: Namespace,
		Name:      "rpc_request_duration_seconds",
		Help:      "JSON-RPC request execution duration in seconds",
		Buckets:   []float64{0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10},
	}, []string{"method"})

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

	if knownRPCMethods == nil {
		return method
	}
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
