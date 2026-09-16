// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package peer

import (
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
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

func TestQueueHandlerRecoversFromPanic(t *testing.T) {
	t.Parallel()

	p := &Peer{
		quit:      make(chan struct{}),
		queueQuit: make(chan struct{}),
		// Zero duration: NewTicker panics, so recoverFromPanic must run.
		cfg: Config{TrickleInterval: 0},
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		p.queueHandler()
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("queueHandler did not return after panic")
	}

	assert.Equal(t, int32(1), atomic.LoadInt32(&p.disconnect))
	select {
	case <-p.quit:
	default:
		assert.Fail(t, "quit channel must be closed by Disconnect")
	}
	select {
	case <-p.queueQuit:
	default:
		assert.Fail(t, "queueQuit must close even when queueHandler panics")
	}
}
