package main

import (
	"context"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/rpcclient"
)

const wsReconnectDelay = 3 * time.Second

func (m *Monitor) runWebSocketWorker(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if err := m.connectAndSubscribe(ctx); err != nil {
			log.Warnf("WebSocket error for %s: %v", m.cfg.RPCHost, err)
		}

		// Wait before reconnecting
		select {
		case <-ctx.Done():
			return
		case <-time.After(wsReconnectDelay):
		}
	}
}

func (m *Monitor) connectAndSubscribe(ctx context.Context) error {
	handlers := &rpcclient.NotificationHandlers{
		OnBlockConnected: func(hash *chainhash.Hash, height int32, t time.Time) {
			m.handleBlockConnected(hash, height, t)
		},
		OnBlockDisconnected: func(hash *chainhash.Hash, height int32, t time.Time) {
			m.handleBlockDisconnected(hash, height, t)
		},
	}

	client, err := m.newRPCClient(handlers)
	if err != nil {
		return err
	}

	if err := client.NotifyBlocks(); err != nil {
		client.Shutdown()
		return err
	}

	log.Infof("WebSocket connected to %s", m.cfg.RPCHost)

	// Wait for disconnect or context cancellation
	done := make(chan struct{})
	go func() {
		client.WaitForShutdown()
		close(done)
	}()

	select {
	case <-ctx.Done():
		client.Shutdown()
		<-done
		return ctx.Err()
	case <-done:
		log.Warnf("WebSocket disconnected from %s", m.cfg.RPCHost)
		return nil
	}
}

func (m *Monitor) handleBlockConnected(hash *chainhash.Hash, height int32, t time.Time) {
	log.Debugf("Block connected on %s: height=%d hash=%s", m.cfg.RPCHost, height, hash)

	// Update node state
	m.mu.Lock()
	m.state.height = height
	m.state.hash = *hash
	m.state.tipTime = t
	m.mu.Unlock()

	// Update metrics
	chainTipHeight.Set(float64(height))
	chainTipAgeSeconds.Set(time.Since(t).Seconds())
	blocksConnectedTotal.Inc()

	// Clear reorg burst (block connected after potential disconnects)
	m.reorg.OnConnect()
}

func (m *Monitor) handleBlockDisconnected(hash *chainhash.Hash, height int32, t time.Time) {
	log.Debugf("Block disconnected on %s: height=%d hash=%s", m.cfg.RPCHost, height, hash)

	// Increment disconnect counter
	blocksDisconnectedTotal.Inc()

	// Track reorg
	m.reorg.OnDisconnect()
}
