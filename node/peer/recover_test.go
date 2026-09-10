// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package peer

import (
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRecoverFromPanic(t *testing.T) {
	t.Parallel()

	p := &Peer{
		quit: make(chan struct{}),
	}

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
