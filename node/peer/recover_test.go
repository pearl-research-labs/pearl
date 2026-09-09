package peer

import (
	"sync/atomic"
	"testing"
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

	if atomic.LoadInt32(&p.disconnect) == 0 {
		t.Fatal("expected disconnect flag to be set " +
			"after panic recovery")
	}
}
