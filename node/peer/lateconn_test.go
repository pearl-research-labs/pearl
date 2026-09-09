// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package peer

import (
	"net"
	"testing"
	"time"
)

func TestAssociateConnectionAfterDisconnect(t *testing.T) {
	t.Parallel()

	p := NewInboundPeer(&Config{AllowSelfConns: true})
	p.Disconnect()

	local, remote := net.Pipe()
	defer remote.Close()

	p.AssociateConnection(local)

	_ = remote.SetReadDeadline(time.Now().Add(time.Second))
	var buf [1]byte
	if _, err := remote.Read(buf[:]); err == nil {
		t.Fatal("expected late connection to be closed")
	}
}
