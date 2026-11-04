// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package peer

import (
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRecoverFromPanic(t *testing.T) {
	t.Parallel()

	p := &Peer{quit: make(chan struct{})}

	done := make(chan struct{})
	go func() {
		defer close(done)
		defer p.recoverFromPanic()

		panic("test: crafted message decode")
	}()

	<-done

	assert.Equal(t, int32(1), atomic.LoadInt32(&p.disconnect))
	select {
	case <-p.quit:
	default:
		assert.Fail(t, "quit channel must be closed by Disconnect")
	}
}

// TestStartReportsNegotiationPanic pins that a panic during negotiation fails start() immediately rather than
// after negotiateTimeout. A nil V2Transport panics on the first handshake call.
func TestStartReportsNegotiationPanic(t *testing.T) {
	t.Parallel()

	p := &Peer{quit: make(chan struct{}), inbound: true}

	errCh := make(chan error, 1)
	go func() { errCh <- p.start() }()

	select {
	case err := <-errCh:
		require.ErrorContains(t, err, "panic during protocol negotiation")
	case <-time.After(5 * time.Second):
		require.FailNow(t, "start() did not return promptly after a negotiation panic")
	}

	assert.Equal(t, int32(1), atomic.LoadInt32(&p.disconnect))
	select {
	case <-p.quit:
	default:
		assert.Fail(t, "quit channel must be closed by Disconnect")
	}
}
