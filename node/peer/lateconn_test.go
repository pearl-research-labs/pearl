// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package peer

import (
	"io"
	"net"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAssociateConnectionAfterDisconnect(t *testing.T) {
	t.Parallel()

	p := NewInboundPeer(&Config{AllowSelfConns: true})
	p.Disconnect()

	local, remote := net.Pipe()
	defer remote.Close()

	// Bound the read so a leaked late connection fails instead of hanging.
	require.NoError(t, remote.SetReadDeadline(time.Now().Add(time.Second)))

	p.AssociateConnection(local)

	var buf [1]byte
	_, err := remote.Read(buf[:])
	assert.ErrorIs(t, err, io.EOF)
	assert.False(t, p.Connected())
}
