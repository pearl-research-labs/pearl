package main

// Sidecar's own logs (in-memory ring buffer) and the GET /logs handler.
//
// `selfLog` (set up by configureSelfLog in log.go) is installed as a tee
// target alongside stdout on prlmon's btclog backend, so every line we
// emit lands in the buffer and can be served back out via /logs. There is
// no follow mode — the live source for that is the container's stdout.

import (
	"bytes"
	"fmt"
	"net/http"
	"sync"
)

// logBuffer is a fixed-capacity ring buffer of log lines.
//
// It implements io.Writer so it can be installed as a tee target alongside
// stdout in the global btclog backend. Each Write call may contain one or
// more '\n'-terminated lines; partial trailing fragments are stitched onto
// the next Write so a line is never split across stored entries.
type logBuffer struct {
	mu      sync.Mutex
	cap     int
	entries []string
	next    int  // next index to overwrite
	full    bool // true once we've wrapped around
	pending []byte
}

func newLogBuffer(capacity int) *logBuffer {
	if capacity < 1 {
		capacity = 1
	}
	return &logBuffer{
		cap:     capacity,
		entries: make([]string, capacity),
	}
}

// Write implements io.Writer. It is safe for concurrent use.
func (b *logBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	n := len(p)
	if len(b.pending) > 0 {
		p = append(b.pending, p...)
		b.pending = nil
	}

	for {
		i := bytes.IndexByte(p, '\n')
		if i < 0 {
			// remainder is partial; carry forward.
			if len(p) > 0 {
				b.pending = append(b.pending[:0], p...)
			}
			break
		}
		line := string(p[:i])
		b.entries[b.next] = line
		b.next++
		if b.next == b.cap {
			b.next = 0
			b.full = true
		}
		p = p[i+1:]
	}
	return n, nil
}

// snapshot returns the buffered lines in chronological order (oldest first).
func (b *logBuffer) snapshot() []string {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.full {
		out := make([]string, b.next)
		copy(out, b.entries[:b.next])
		return out
	}
	out := make([]string, b.cap)
	copy(out, b.entries[b.next:])
	copy(out[b.cap-b.next:], b.entries[:b.next])
	return out
}

// handleLogsSelf serves prlmon's own recent log lines from selfLog.
//
// Supports the head/tail subset of /node/logs. No follow — this buffer is
// updated by prlmon's own logger and live tailing is what the container's
// stdout is for. With no params the full buffer is returned.
func (m *Monitor) handleLogsSelf(w http.ResponseWriter, r *http.Request) {
	if selfLog == nil {
		http.Error(w, "self log buffer not configured", http.StatusServiceUnavailable)
		return
	}

	q, err := parseLogQuery(r, m.cfg.LogsMaxLines)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if q.Follow {
		http.Error(w, "follow=true is not supported on /logs", http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")

	raw := selfLog.snapshot()

	switch {
	case q.Head > 0:
		if q.Head < len(raw) {
			raw = raw[:q.Head]
		}
	case q.Tail > 0:
		if q.Tail < len(raw) {
			raw = raw[len(raw)-q.Tail:]
		}
	}
	for _, l := range raw {
		if _, err := fmt.Fprintln(w, l); err != nil {
			return
		}
	}
}
