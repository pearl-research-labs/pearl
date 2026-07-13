package main

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

const namespace = "prlmon"

var (
	// Liveness / RPC
	nodeUp = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: namespace,
		Name:      "node_up",
		Help:      "Whether the node is reachable (1 = up, 0 = down)",
	})

	// Chain / sync
	chainTipHeight = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: namespace,
		Name:      "chain_tip_height",
		Help:      "Current tip height of the node",
	})

	chainTipAgeSeconds = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: namespace,
		Name:      "chain_tip_age_seconds",
		Help:      "Age of the current tip in seconds",
	})

	chainDifficulty = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: namespace,
		Name:      "chain_difficulty",
		Help:      "Current proof-of-work difficulty as a multiple of minimum difficulty",
	})

	// Block events
	blocksConnectedTotal = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: namespace,
		Name:      "blocks_connected_total",
		Help:      "Total number of block connected events observed via WebSocket",
	})

	blocksDisconnectedTotal = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: namespace,
		Name:      "blocks_disconnected_total",
		Help:      "Total number of block disconnected events observed via WebSocket",
	})

	// Reorg tracking
	reorgTotal = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: namespace,
		Name:      "reorg_total",
		Help:      "Total number of reorg events (bursts of disconnections)",
	})

	reorgDepth = promauto.NewHistogram(prometheus.HistogramOpts{
		Namespace: namespace,
		Name:      "reorg_depth",
		Help:      "Depth of reorganizations (consecutive disconnects before reconnect)",
		Buckets:   []float64{1, 2, 3, 4, 6, 10},
	})

	// P2P connectivity
	p2pPeerCount = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: namespace,
		Name:      "p2p_peer_count",
		Help:      "Number of connected peers",
	})

	p2pInboundPeers = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: namespace,
		Name:      "p2p_inbound_peers",
		Help:      "Number of inbound peer connections",
	})

	p2pOutboundPeers = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: namespace,
		Name:      "p2p_outbound_peers",
		Help:      "Number of outbound peer connections",
	})

	p2pPingTimeSeconds = promauto.NewHistogram(prometheus.HistogramOpts{
		Namespace: namespace,
		Name:      "p2p_pingtime_seconds",
		Help:      "Ping time to peers in seconds",
		Buckets:   []float64{0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5},
	})

	p2pLastRecvAgeSeconds = promauto.NewHistogram(prometheus.HistogramOpts{
		Namespace: namespace,
		Name:      "p2p_lastrecv_age_seconds",
		Help:      "Age since last message received from each peer",
		Buckets:   []float64{1, 5, 15, 30, 60, 120, 300, 600},
	})

	// Network totals
	netTotalBytesSentTotal = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: namespace,
		Name:      "net_totalbytes_sent_total",
		Help:      "Total bytes sent by the node",
	})

	netTotalBytesRecvTotal = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: namespace,
		Name:      "net_totalbytes_recv_total",
		Help:      "Total bytes received by the node",
	})

	// Mempool
	mempoolTxCount = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: namespace,
		Name:      "mempool_tx_count",
		Help:      "Number of transactions in the mempool",
	})

	mempoolBytes = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: namespace,
		Name:      "mempool_bytes",
		Help:      "Total size of the mempool in bytes",
	})
)
