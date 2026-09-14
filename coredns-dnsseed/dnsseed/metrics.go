package dnsseed

import (
	"github.com/coredns/coredns/plugin"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// requestCount counts queries handled by the plugin (any qtype, once
	// the query matched a served zone).
	requestCount = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: plugin.Namespace,
		Subsystem: pluginName,
		Name:      "request_count_total",
		Help:      "Counter of requests handled by the dnsseed plugin.",
	}, []string{"server"})

	// addressCount tracks the number of verified, servable peer addresses
	// in the book. It is updated after each crawl.
	addressCount = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: plugin.Namespace,
		Subsystem: pluginName,
		Name:      "addresses",
		Help:      "Number of verified peer addresses available to serve.",
	})
)
