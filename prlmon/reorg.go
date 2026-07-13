package main

import (
	"sync"
	"time"
)

const reorgBurstTimeout = 60 * time.Second

// ReorgTracker tracks reorganization depth based on block disconnect bursts.
type ReorgTracker struct {
	mu          sync.Mutex
	burstDepth  int
	burstStart  time.Time
	total       int       // mirrors reorgTotal counter for /node
	lastReorgAt time.Time // wall-clock time of the most recent finalised reorg
}

// NewReorgTracker creates a new reorg tracker.
func NewReorgTracker() *ReorgTracker {
	return &ReorgTracker{}
}

// OnDisconnect records a block disconnection.
func (r *ReorgTracker) OnDisconnect() {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Start burst if not already started
	if r.burstDepth == 0 {
		r.burstStart = time.Now()
	}

	r.burstDepth++
	// Note: individual disconnects are counted by blocksDisconnectedTotal in websocket.go
}

// OnConnect records a block connection, which ends any active burst.
func (r *ReorgTracker) OnConnect() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.burstDepth > 0 {
		// A reorg event is complete (burst of disconnects followed by connect)
		reorgTotal.Inc()
		reorgDepth.Observe(float64(r.burstDepth))
		r.total++
		r.lastReorgAt = time.Now()
		r.burstDepth = 0
		r.burstStart = time.Time{}
	}
}

// FlushStale flushes any burst that has been open too long without a connect.
// This handles extended partitions where the node never reconnects.
func (r *ReorgTracker) FlushStale() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.burstDepth > 0 && !r.burstStart.IsZero() && time.Since(r.burstStart) > reorgBurstTimeout {
		reorgTotal.Inc()
		reorgDepth.Observe(float64(r.burstDepth))
		r.total++
		r.lastReorgAt = time.Now()
		r.burstDepth = 0
		r.burstStart = time.Time{}
	}
}

// Snapshot returns the current state for diagnostic endpoints.
func (r *ReorgTracker) Snapshot() (total, burstDepth int, lastReorgAt time.Time) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.total, r.burstDepth, r.lastReorgAt
}
