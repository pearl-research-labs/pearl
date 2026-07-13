package main

import (
	"context"
	"time"
)

func (m *Monitor) runPollingLoop(ctx context.Context) {
	ticker := time.NewTicker(m.cfg.Poll)
	defer ticker.Stop()

	// Initial poll
	m.pollNode()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.pollNode()
		}
	}
}

func (m *Monitor) pollNode() {
	client, err := m.newHTTPRPCClient()
	if err != nil {
		log.Warnf("Failed to create RPC client for %s: %v", m.cfg.RPCHost, err)
		m.setNodeDown()
		return
	}
	defer client.Shutdown()

	// Get best block
	hash, height, err := client.GetBestBlock()
	if err != nil {
		log.Warnf("GetBestBlock failed for %s: %v", m.cfg.RPCHost, err)
		m.setNodeDown()
		return
	}

	// Get block header for timestamp
	header, err := client.GetBlockHeader(hash)
	if err != nil {
		log.Warnf("GetBlockHeader failed for %s: %v", m.cfg.RPCHost, err)
		m.setNodeDown()
		return
	}

	// Get detailed peer info
	peerInfo, err := client.GetPeerInfo()
	if err != nil {
		log.Warnf("GetPeerInfo failed for %s: %v", m.cfg.RPCHost, err)
		m.setNodeDown()
		return
	}

	// Get network totals
	netTotals, err := client.GetNetTotals()
	if err != nil {
		log.Warnf("GetNetTotals failed for %s: %v", m.cfg.RPCHost, err)
		m.setNodeDown()
		return
	}

	// Get current difficulty
	difficulty, err := client.GetDifficulty()
	if err != nil {
		log.Warnf("GetDifficulty failed for %s: %v", m.cfg.RPCHost, err)
		m.setNodeDown()
		return
	}

	// Get mempool info
	mempoolVerbose, err := client.GetRawMempoolVerbose()
	if err != nil {
		log.Warnf("GetRawMempoolVerbose failed for %s: %v", m.cfg.RPCHost, err)
		m.setNodeDown()
		return
	}

	// Count inbound vs outbound peers and collect metrics
	var inbound, outbound int64
	now := time.Now()
	for _, peer := range peerInfo {
		if peer.Inbound {
			inbound++
		} else {
			outbound++
		}
		// Pearl's getpeerinfo "pingtime" field is in microseconds (pearl
		// node/rpcserver.go casts LastPingMicros to float64 directly), so
		// divide by 1e6 to land in the seconds-bucketed histogram.
		if peer.PingTime > 0 {
			p2pPingTimeSeconds.Observe(peer.PingTime / 1e6)
		}
		// Record lastrecv age histogram
		if peer.LastRecv > 0 {
			lastRecvTime := time.Unix(peer.LastRecv, 0)
			lastRecvAge := now.Sub(lastRecvTime).Seconds()
			if lastRecvAge >= 0 {
				p2pLastRecvAgeSeconds.Observe(lastRecvAge)
			}
		}
	}
	peerCount := int64(len(peerInfo))

	// Update state
	m.mu.Lock()
	m.state.up = true
	m.state.height = height
	m.state.hash = *hash
	m.state.tipTime = header.Timestamp
	m.state.peerCount = peerCount
	m.mu.Unlock()

	// Update metrics
	nodeUp.Set(1)
	chainTipHeight.Set(float64(height))
	chainTipAgeSeconds.Set(time.Since(header.Timestamp).Seconds())
	chainDifficulty.Set(difficulty)
	p2pPeerCount.Set(float64(peerCount))
	p2pInboundPeers.Set(float64(inbound))
	p2pOutboundPeers.Set(float64(outbound))

	// Update network byte counters
	m.updateNetBytes(netTotals.TotalBytesRecv, netTotals.TotalBytesSent)

	// Update mempool metrics
	var totalMempoolBytes int64
	for _, entry := range mempoolVerbose {
		totalMempoolBytes += int64(entry.Size)
	}
	mempoolTxCount.Set(float64(len(mempoolVerbose)))
	mempoolBytes.Set(float64(totalMempoolBytes))
}

func (m *Monitor) setNodeDown() {
	m.mu.Lock()
	m.state.up = false
	m.mu.Unlock()

	nodeUp.Set(0)
}

func (m *Monitor) updateNetBytes(recv, sent uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	prev := m.prevNetBytes

	if prev.recv > 0 || prev.sent > 0 {
		// Handle counter reset (node restart)
		recvDelta := recv
		sentDelta := sent
		if recv >= prev.recv {
			recvDelta = recv - prev.recv
		}
		if sent >= prev.sent {
			sentDelta = sent - prev.sent
		}

		netTotalBytesRecvTotal.Add(float64(recvDelta))
		netTotalBytesSentTotal.Add(float64(sentDelta))
	}

	m.prevNetBytes = netBytes{recv: recv, sent: sent}
}
