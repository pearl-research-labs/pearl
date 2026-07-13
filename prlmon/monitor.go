package main

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/rpcclient"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Monitor orchestrates polling and WebSocket connection to the configured node.
type Monitor struct {
	cfg *Config

	// HTTP server (listener bound at construction, serving starts in Run)
	httpListener net.Listener
	httpServer   *http.Server

	// Node state
	mu           sync.RWMutex
	state        *nodeState
	prevNetBytes netBytes

	// Reorg tracker
	reorg *ReorgTracker

	// Process start time, exposed via /node as prlmon.uptimeSec.
	startedAt time.Time
}

type nodeState struct {
	up        bool
	height    int32
	hash      chainhash.Hash
	tipTime   time.Time
	peerCount int64
}

type netBytes struct {
	recv uint64
	sent uint64
}

// NewMonitor creates a new Monitor instance.
// The HTTP listener is bound immediately, so ListenAddr() is available before Run().
func NewMonitor(cfg *Config) (*Monitor, error) {
	// Bind HTTP listener early so address is known before Run()
	listener, err := net.Listen("tcp", cfg.Listen)
	if err != nil {
		return nil, fmt.Errorf("failed to bind HTTP listener: %w", err)
	}

	m := &Monitor{
		cfg:          cfg,
		httpListener: listener,
		state:        &nodeState{},
		reorg:        NewReorgTracker(),
		startedAt:    time.Now(),
	}

	return m, nil
}

// ListenAddr returns the actual listen address (useful when port 0 is used).
// Safe to call immediately after NewMonitor() - no need to wait for Run().
func (m *Monitor) ListenAddr() string {
	return m.httpListener.Addr().String()
}

// Run starts the monitor and blocks until context is cancelled.
func (m *Monitor) Run(ctx context.Context) error {
	// Start HTTP server (listener already bound in NewMonitor)
	m.startHTTPServer()

	var wg sync.WaitGroup

	// Start WebSocket worker
	wg.Add(1)
	go func() {
		defer wg.Done()
		m.runWebSocketWorker(ctx)
	}()

	// Start polling loop
	wg.Add(1)
	go func() {
		defer wg.Done()
		m.runPollingLoop(ctx)
	}()

	// Wait for shutdown
	<-ctx.Done()

	// Graceful shutdown
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := m.httpServer.Shutdown(shutdownCtx); err != nil {
		log.Warnf("HTTP server shutdown error: %v", err)
	}

	wg.Wait()
	return ctx.Err()
}

func (m *Monitor) startHTTPServer() {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	// Sidecar's own resources sit at the root alongside /metrics and /healthz.
	mux.HandleFunc("/logs", m.handleLogsSelf)

	// Resources for the monitored pearld node.
	mux.HandleFunc("/node", m.handleStatus)
	mux.HandleFunc("/node/peers", m.handlePeers)
	mux.HandleFunc("/node/mempool", m.handleMempool)
	mux.HandleFunc("/node/chaintips", m.handleChainTips)
	mux.HandleFunc("/node/logs", m.handleLogsFile)
	mux.HandleFunc("/node/logs/files", m.handleLogFilesList)
	mux.HandleFunc(filesPrefix, m.handleLogFileDownload)

	m.httpServer = &http.Server{Handler: mux}

	go func() {
		if err := m.httpServer.Serve(m.httpListener); err != nil && err != http.ErrServerClosed {
			log.Errorf("HTTP server error: %v", err)
			os.Exit(1)
		}
	}()

	log.Infof("HTTP server listening on %s", m.ListenAddr())
}

func (m *Monitor) getRPCCerts() ([]byte, error) {
	if m.cfg.NoTLS {
		return nil, nil
	}
	return os.ReadFile(m.cfg.RPCCert)
}

func (m *Monitor) newRPCClient(handlers *rpcclient.NotificationHandlers) (*rpcclient.Client, error) {
	certs, err := m.getRPCCerts()
	if err != nil {
		return nil, fmt.Errorf("failed to read RPC cert: %w", err)
	}

	connCfg := &rpcclient.ConnConfig{
		Host:                 m.cfg.RPCHost,
		Endpoint:             "ws",
		User:                 m.cfg.RPCUser,
		Pass:                 m.cfg.RPCPass,
		Certificates:         certs,
		DisableTLS:           m.cfg.NoTLS,
		DisableAutoReconnect: true,
	}

	return rpcclient.New(connCfg, handlers)
}

func (m *Monitor) newHTTPRPCClient() (*rpcclient.Client, error) {
	certs, err := m.getRPCCerts()
	if err != nil {
		return nil, fmt.Errorf("failed to read RPC cert: %w", err)
	}

	connCfg := &rpcclient.ConnConfig{
		Host:         m.cfg.RPCHost,
		User:         m.cfg.RPCUser,
		Pass:         m.cfg.RPCPass,
		Certificates: certs,
		DisableTLS:   m.cfg.NoTLS,
		HTTPPostMode: true,
	}

	return rpcclient.New(connCfg, nil)
}
