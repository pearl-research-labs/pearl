package metrics

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// Source provides accessor functions for scrape-time metrics collection.
type Source struct {
	ChainTipHeight     func() int32
	ChainTipTimestamp  func() time.Time
	ChainTotalTxs      func() int64
	ChainIsCurrent     func() bool
	PeerCount          func() (inbound, outbound int64)
	NetTotals          func() (totalRecv, totalSent uint64)
	MempoolTxCount     func() int
	MempoolBytes       func() uint64
	MempoolMaxBytes    func() uint64
	MempoolLastUpdated func() time.Time
}

// ScrapeCollector implements prometheus.Collector to collect live state at scrape time.
type ScrapeCollector struct {
	source Source

	chainTipHeightDesc              *prometheus.Desc
	chainTipTimestampDesc           *prometheus.Desc
	chainTotalTxsDesc               *prometheus.Desc
	chainIsCurrentDesc              *prometheus.Desc
	p2pPeersDesc                    *prometheus.Desc
	p2pNetBytesRecvDesc             *prometheus.Desc
	p2pNetBytesSentDesc             *prometheus.Desc
	mempoolTransactionsDesc         *prometheus.Desc
	mempoolBytesDesc                *prometheus.Desc
	mempoolMaxBytesDesc             *prometheus.Desc
	mempoolLastUpdatedTimestampDesc *prometheus.Desc
}

func NewScrapeCollector(src Source) *ScrapeCollector {
	return &ScrapeCollector{
		source: src,
		chainTipHeightDesc: prometheus.NewDesc(
			prometheus.BuildFQName(Namespace, "", "chain_tip_height"),
			"Current chain tip height",
			nil, nil,
		),
		chainTipTimestampDesc: prometheus.NewDesc(
			prometheus.BuildFQName(Namespace, "", "chain_tip_timestamp_seconds"),
			"Unix timestamp of current chain tip block",
			nil, nil,
		),
		chainTotalTxsDesc: prometheus.NewDesc(
			prometheus.BuildFQName(Namespace, "", "chain_total_transactions"),
			"Total transactions in the main chain",
			nil, nil,
		),
		chainIsCurrentDesc: prometheus.NewDesc(
			prometheus.BuildFQName(Namespace, "", "chain_is_current"),
			"Whether the chain is synced / current (1 = true, 0 = false)",
			nil, nil,
		),
		p2pPeersDesc: prometheus.NewDesc(
			prometheus.BuildFQName(Namespace, "", "p2p_peers"),
			"Current number of connected peers",
			[]string{"direction"}, nil,
		),
		p2pNetBytesRecvDesc: prometheus.NewDesc(
			prometheus.BuildFQName(Namespace, "", "net_totalbytes_recv_total"),
			"Total bytes received across network interfaces",
			nil, nil,
		),
		p2pNetBytesSentDesc: prometheus.NewDesc(
			prometheus.BuildFQName(Namespace, "", "net_totalbytes_sent_total"),
			"Total bytes sent across network interfaces",
			nil, nil,
		),
		mempoolTransactionsDesc: prometheus.NewDesc(
			prometheus.BuildFQName(Namespace, "", "mempool_transactions"),
			"Current number of transactions in the mempool",
			nil, nil,
		),
		mempoolBytesDesc: prometheus.NewDesc(
			prometheus.BuildFQName(Namespace, "", "mempool_bytes"),
			"Current size of the mempool in bytes",
			nil, nil,
		),
		mempoolMaxBytesDesc: prometheus.NewDesc(
			prometheus.BuildFQName(Namespace, "", "mempool_max_bytes"),
			"Configured maximum size of the mempool in bytes",
			nil, nil,
		),
		mempoolLastUpdatedTimestampDesc: prometheus.NewDesc(
			prometheus.BuildFQName(Namespace, "", "mempool_last_updated_timestamp_seconds"),
			"Unix timestamp when mempool was last updated",
			nil, nil,
		),
	}
}

func (c *ScrapeCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.chainTipHeightDesc
	ch <- c.chainTipTimestampDesc
	ch <- c.chainTotalTxsDesc
	ch <- c.chainIsCurrentDesc
	ch <- c.p2pPeersDesc
	ch <- c.p2pNetBytesRecvDesc
	ch <- c.p2pNetBytesSentDesc
	ch <- c.mempoolTransactionsDesc
	ch <- c.mempoolBytesDesc
	ch <- c.mempoolMaxBytesDesc
	ch <- c.mempoolLastUpdatedTimestampDesc
}

func (c *ScrapeCollector) Collect(ch chan<- prometheus.Metric) {
	s := c.source

	if s.ChainTipHeight != nil {
		ch <- prometheus.MustNewConstMetric(c.chainTipHeightDesc, prometheus.GaugeValue, float64(s.ChainTipHeight()))
	}
	if s.ChainTipTimestamp != nil {
		ts := s.ChainTipTimestamp()
		if !ts.IsZero() {
			ch <- prometheus.MustNewConstMetric(c.chainTipTimestampDesc, prometheus.GaugeValue, float64(ts.Unix()))
		}
	}
	if s.ChainTotalTxs != nil {
		ch <- prometheus.MustNewConstMetric(c.chainTotalTxsDesc, prometheus.GaugeValue, float64(s.ChainTotalTxs()))
	}
	if s.ChainIsCurrent != nil {
		v := 0.0
		if s.ChainIsCurrent() {
			v = 1.0
		}
		ch <- prometheus.MustNewConstMetric(c.chainIsCurrentDesc, prometheus.GaugeValue, v)
	}
	if s.PeerCount != nil {
		inbound, outbound := s.PeerCount()
		ch <- prometheus.MustNewConstMetric(c.p2pPeersDesc, prometheus.GaugeValue, float64(inbound), "inbound")
		ch <- prometheus.MustNewConstMetric(c.p2pPeersDesc, prometheus.GaugeValue, float64(outbound), "outbound")
	}
	if s.NetTotals != nil {
		recv, sent := s.NetTotals()
		ch <- prometheus.MustNewConstMetric(c.p2pNetBytesRecvDesc, prometheus.CounterValue, float64(recv))
		ch <- prometheus.MustNewConstMetric(c.p2pNetBytesSentDesc, prometheus.CounterValue, float64(sent))
	}
	if s.MempoolTxCount != nil {
		ch <- prometheus.MustNewConstMetric(c.mempoolTransactionsDesc, prometheus.GaugeValue, float64(s.MempoolTxCount()))
	}
	if s.MempoolBytes != nil {
		ch <- prometheus.MustNewConstMetric(c.mempoolBytesDesc, prometheus.GaugeValue, float64(s.MempoolBytes()))
	}
	if s.MempoolMaxBytes != nil {
		ch <- prometheus.MustNewConstMetric(c.mempoolMaxBytesDesc, prometheus.GaugeValue, float64(s.MempoolMaxBytes()))
	}
	if s.MempoolLastUpdated != nil {
		lu := s.MempoolLastUpdated()
		if !lu.IsZero() {
			ch <- prometheus.MustNewConstMetric(c.mempoolLastUpdatedTimestampDesc, prometheus.GaugeValue, float64(lu.Unix()))
		}
	}
}
