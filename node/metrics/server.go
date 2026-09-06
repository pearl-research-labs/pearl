package metrics

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Server manages the standalone HTTP metrics server.  It does no logging of its
// own; callers report bound addresses via ListenAddrs and serve failures via
// Errors.
type Server struct {
	listeners    []net.Listener
	httpServers  []*http.Server
	errs         chan error
	collector    *ScrapeCollector
	collectorReg prometheus.Collector
	mu           sync.Mutex
	started      bool
}

// NewServer binds TCP listeners for the provided addresses and prepares the metrics HTTP server.
func NewServer(listenAddrs []string) (*Server, error) {
	if len(listenAddrs) == 0 {
		return nil, fmt.Errorf("no metrics listen addresses specified")
	}

	var listeners []net.Listener
	for _, addr := range listenAddrs {
		l, err := net.Listen("tcp", addr)
		if err != nil {
			// Close any already bound listeners on failure
			for _, bound := range listeners {
				bound.Close()
			}
			return nil, fmt.Errorf("failed to bind metrics listener on %s: %w", addr, err)
		}
		listeners = append(listeners, l)
	}

	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(registry, promhttp.HandlerOpts{}))
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	var httpServers []*http.Server
	for range listeners {
		httpServers = append(httpServers, &http.Server{
			Handler:           mux,
			ReadHeaderTimeout: 5 * time.Second,
		})
	}

	return &Server{
		listeners:   listeners,
		httpServers: httpServers,
		errs:        make(chan error, len(listeners)),
	}, nil
}

// Errors receives an error for each listener that stops unexpectedly.  The
// channel is closed once every listener has stopped, so a caller can simply
// range over it.  It is buffered to one slot per listener and each listener
// sends at most once, so sends never block.
func (s *Server) Errors() <-chan error {
	return s.errs
}

// SetSource attaches or replaces the scrape-time metric source.
func (s *Server) SetSource(src Source) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.collectorReg != nil {
		registry.Unregister(s.collectorReg)
	}

	s.collector = NewScrapeCollector(src)
	s.collectorReg = s.collector
	registry.MustRegister(s.collectorReg)
}

// ListenAddrs returns the actual bound addresses for the server listeners.
func (s *Server) ListenAddrs() []string {
	var addrs []string
	for _, l := range s.listeners {
		addrs = append(addrs, l.Addr().String())
	}
	return addrs
}

// Start launches background HTTP serving for all bound listeners.
func (s *Server) Start() {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return
	}
	s.started = true
	s.mu.Unlock()

	var wg sync.WaitGroup
	for i, l := range s.listeners {
		srv := s.httpServers[i]
		wg.Add(1)
		go func(listener net.Listener, server *http.Server) {
			defer wg.Done()
			if err := server.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
				s.errs <- fmt.Errorf("metrics listener %s: %w", listener.Addr(), err)
			}
		}(l, srv)
	}

	go func() {
		wg.Wait()
		close(s.errs)
	}()
}

// Stop gracefully shuts down the metrics HTTP servers.
func (s *Server) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	for _, srv := range s.httpServers {
		_ = srv.Shutdown(ctx)
	}
}
